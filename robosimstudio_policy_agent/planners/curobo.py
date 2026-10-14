# SPDX-License-Identifier: Apache-2.0
"""cuRobo as a *motion* backend: collision-aware plans, not per-step IK.

**Why this is not an IK backend, which is the first thing to get right.** The
runtime's ``ik=`` slot wants ``solve_ik`` and ``plan_linear``, and it would be
natural to hand it cuRobo. That would be a mistake with a measurable cost.
``plan_linear`` is dead code on the default path -- the runtime calls it from
one place, ``go_to_pose_cartesian``, and
:func:`~robosimstudio_policy_agent.connector.make_connector` overrides that to
:class:`~robosimstudio_policy_agent.servo.CartesianServo` whenever a servo exists,
which is the default. Every Cartesian motion is therefore
``CartesianServo.move`` calling ``solve_ik`` **once per control step, up to 400
per move**. A GPU motion planner on that loop is seconds per motion.

So this owns *plans*, called once per node, and
:class:`~robosimstudio_policy_agent.ik.IKBackend` pairs it (as its ``planner``) with a
cheap solver to fill the runtime's single slot.

**The primitive is cuRobo's own.** ``ToolPoseCriteria.linear_motion`` is what
``MotionPlanner.plan_grasp`` uses for its approach-to-grasp segment, and it is
exactly the shape the reference stack's ``plan_directed_linear(allowed_axes=
["Z"], orientation_mode="LOCK")`` describes: zero non-terminal cost on the
travel axis, full weight on the other two and on all three rotations. Measured
here on a 20 cm Franka move against the unconstrained planner:

    criteria                  perpendicular bow    rotation wander
    default                        28.40 mm            3.273 deg
    linear_motion(axis="z")         3.53 mm            0.065 deg

That eight-fold tightening is the point. ``tool_sort``'s own
``waypoint_move.py`` records what the unconstrained path costs: *"TrajOpt's
shortest smooth path routinely dips LOW while translating -- recorded trial
videos show the held object dragged through scene clutter, shoving the
container ~12 cm off its perceived pose."* A rectangular lift-translate-descend
is not an inefficiency to optimise away; it is a constraint an author chose, and
this is how the constraint is stated to a planner.

**The trap in that primitive, measured both ways.** With
``project_distance_to_goal=True`` the axis is the *goal frame's*, not the world's.
Pointed at a travel direction that is not that axis it constrains the wrong two
axes and returns a path **worse than not constraining at all** -- 43.00 mm on the
same move. The plan succeeds; it just bows. :meth:`MotionBackendCuRobo.plan_linear`
therefore derives the axis from the actual travel direction rather than taking it
on faith, and says so when it cannot.

**Two hazards this module is careful about.**

*Streams.* cuRobo runs 30+ Warp modules, in a process already running Newton on
Warp. :mod:`robosimstudio.core.streams` documents that Warp's current stream is a
process-global and that losing it permanently breaks every CUDA graph capture in
the process. Every entry point here holds ``reading()``.

*Frames.* :class:`~robosimstudio_policy_agent.ik.IKBackend` accepts
and ignores ``tcp_offset`` because ``ArmIK`` already builds its objective with
``link_offset=ee_offset``. cuRobo has no such property and must apply a tool-to-link
transform -- but it applies the one it **measured**, not the one the runtime
passes. ``ik_backend.py``'s own table records what getting that backwards costs:
0.00 mm on the Panda, whose offset is the zero vector and which is why the bug
could hide, but 134.70 mm on yam_i2rt and 254.73 mm on droid.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from robosimstudio.arms import arm_name
from robosimstudio.core.bus import Bus
from robosimstudio.core.math import (
    mat_to_quat,
    quat_to_mat,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from robosimstudio.core.streams import reading

logger = logging.getLogger(__name__)

__all__ = ["MotionBackendCuRobo", "MotionUnavailable"]

CUROBO_INSTALL_HINT = (
    "cuRobo is not importable. Install the extra: 'uv sync --extra curobo'. "
    "It is pure Python on v0.8 -- no CUDA toolkit and no nvcc -- but it does "
    "need a CUDA device at run time."
)

MIN_TRAVEL_M = 1.0e-4
"""Below this a 'linear' move has no direction to constrain, so it is planned free.

A 0.1 mm move has a travel direction dominated by float noise, and constraining
the two axes perpendicular to *noise* is how a trivial nudge becomes an
unsolvable plan.
"""
# CuRobo defaults to 5 mm, but every returned endpoint is independently gated
# by simulator FK at 3 mm below. Asking the optimizer to stop outside the
# acceptance envelope creates valid-looking plans that this backend must then
# discard. Keep the solve target strictly inside the parity gate so numerical
# differences between the two FK implementations still have headroom.
PLANNER_POSITION_TOLERANCE_M = 0.002
PLANNER_ORIENTATION_TOLERANCE_RAD = 0.03

TOOL_DRIFT_MM = 0.5
"""How far the measured tool-to-TCP transform may move between configurations.

It is claimed to be a *fixed* offset, so movement means the claim is false --
and an average of a moving offset is wrong everywhere instead of wrong once.
Half a millimetre is float-and-solver noise; anything above it is two frames
that are not rigidly related.
"""

ENDPOINT_PARITY_MM = 3.0
ENDPOINT_PARITY_DEG = 3.0
"""Maximum disagreement between a planned endpoint and simulator FK.

This is checked before a trajectory is exposed. A planner can report success
under a URDF whose joint axes or ordering differ from the executable model;
streaming that trajectory is how an apparently valid plan moves the wrong way.
"""

AXIS_ALIGNMENT = 0.98
"""How well travel must line up with a tool axis before that axis is freed.

``cos(11.5 deg)``. The constraint frees one axis and holds the other two; if
travel is not substantially along the freed one, the two being held are the ones
the move needs, and the measured result is a path *worse* than unconstrained
(43.00 mm against 28.40 mm). Below this the move is planned free and the caller
is told, which is the honest outcome -- a silently bowed path is the failure
this whole module exists to avoid.
"""

ROBOT_SELF_MARGIN_M = 0.03
"""Added to every robot sphere when removing the arm's own pixels from an RGB-D
frame before TSDF integration: the sphere model is a conservative-ish fit, and
a depth edge on the arm's silhouette carries the arm's depth a pixel or two
past it."""
LINEAR_TUBE_M = 0.006
"""How far the tool point may leave the straight line before a "linear" plan is
not one. Measured 2026-09-10 over 403 executed ``plan_linear`` legs of the
sharps task: cuRobo's criterion is a *weight* on the non-terminal pose axes, not
a constraint, and the median leg bowed 12 mm, the 90th percentile 103 mm; a
54 mm insertion stroke went down, back up 30 mm and down again (trial t22, the
syringe came out of the hole sideways and caught the rim) and one wandered
560 mm (t11). Raising the weight did not help (a 611 mm leg bowed 125 mm at
every weight tried). The 0-2 deg well-aligned legs sat at 4 mm median, which
this tolerance admits; anything beyond it is re-planned as an IK-tracked line."""
LINEAR_BACKTRACK_M = 0.003
"""Progress along the line may not reverse by more than this."""
LINEAR_STEP_M = 0.005
"""Spacing of the IK-tracked line's waypoints: one per control step when
streamed, 12 cm/s at 25 Hz, smoothstepped so the leg starts and ends at rest."""
LINEAR_JOINT_JUMP_RAD = 0.3
"""Largest joint change between consecutive line waypoints; more is the solver
switching branches mid-line, which the arm would fly as a swing."""
LINEAR_MIN_LINE_M = 0.01
"""Below this travel a "line" is a re-posture (the yaw-in-place fallback asks
for one with the hand already at the goal point), which the planner plans."""
LINEAR_TURN_MAX_DEG = 5.0
"""A line holds the goal orientation at every waypoint, so the arm must already
hold it, to within this; a leg that has to turn the hand as well is not a line."""

IK_BATCH_MAX_CANDIDATES = 32
"""Ceiling on one solve_ik_batch call, and the size every cached batch planner
is built for (padded with the first pose when fewer are asked for). One size
because a cuRobo batch planner is expensive to (re)build -- capturing its CUDA
graph at a new batch shape costs about as much as building a fresh planner --
so reusing one fixed-size planner across calls of different real candidate
counts beats rebuilding one per call. 32 comfortably covers a heading search's
worst case (YAWS_DEG + YAWS_FINE_DEG, 18) with headroom."""

ROBOT_SCENE_FILTER_MARGIN_M = 0.015
"""Extra radius used to remove robot returns from reconstructed RGB-D meshes.

Depth edges and camera calibration are not exact.  Fifteen millimetres removes
the thin halo around a link without globally deleting a nearby support surface.
The filter operates on triangles, so table geometry outside the actual robot
envelope remains available to the planner.
"""


def _filter_mesh_near_spheres(
    vertices: np.ndarray,
    faces: np.ndarray,
    spheres: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove mesh triangles overlapping live robot collision spheres.

    A vertex/centroid test misses a large tabletop triangle whose interior
    crosses a link. Exact closest points are therefore evaluated for triangles
    whose AABB overlaps each sphere. Returned vertices are compacted and faces
    remapped for CuRobo.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    spheres = np.asarray(spheres, dtype=np.float64).reshape(-1, 4)
    spheres = spheres[spheres[:, 3] > 0.0] if len(spheres) else spheres
    if not len(vertices) or not len(faces) or not len(spheres):
        return vertices, faces, 0

    import trimesh

    triangles = vertices[faces]
    tri_min = triangles.min(axis=1)
    tri_max = triangles.max(axis=1)
    remove = np.zeros(len(faces), dtype=bool)
    extra = max(float(margin), 0.0)
    for sphere in spheres:
        center = sphere[:3]
        radius = float(sphere[3]) + extra
        candidates = np.where(
            np.all(center >= tri_min - radius, axis=1) & np.all(center <= tri_max + radius, axis=1) & ~remove
        )[0]
        if not len(candidates):
            continue
        candidate_triangles = triangles[candidates]
        query = np.repeat(center[None], len(candidates), axis=0)
        closest = trimesh.triangles.closest_point(candidate_triangles, query)
        distance_sq = np.sum((closest - query) ** 2, axis=1)
        remove[candidates[distance_sq <= radius * radius]] = True
    kept_faces = faces[~remove]
    removed = int(np.count_nonzero(remove))
    if not len(kept_faces):
        return vertices[:0], faces[:0], removed
    used = np.unique(kept_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[kept_faces], removed


class MotionUnavailable(NotImplementedError):
    """cuRobo could not be brought up for this embodiment.

    ``NotImplementedError`` rather than a bare ``RuntimeError``, so that "this
    backend cannot serve this robot" is **one** contract across the solver
    slots instead of two spellings of it: the PyRoKi IK backend already says the
    same thing that way, and the connector had to know both names to fall back.
    A registry cannot special-case each backend's private exception, so the
    contract has to be nameable without importing the backend that raises it.
    """


def _se3_to_arrays(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """A runtime ``Se3Pose`` dict to ``(position [m], quaternion xyzw)``.

    The runtime speaks **wxyz**; everything in ``robosimstudio.core.math``
    speaks xyzw. Converting once, here, is what keeps the rest of this module
    from having to remember which it is holding.
    """
    position = pose["position"]
    rotation = pose["rotation"]
    return (
        np.array([float(position["x"]), float(position["y"]), float(position["z"])]),
        quat_wxyz_to_xyzw([float(rotation[k]) for k in ("w", "x", "y", "z")]),
    )


def _homogeneous(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """A 4x4 from a position and an xyzw quaternion."""
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_mat(quat_xyzw)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


def _as_int(value: Any) -> int | None:
    """A cuRobo scalar as an ``int``, whatever container it arrived in.

    ``interpolated_last_tstep`` is a ``(batch, 1)`` CUDA tensor and ``dt`` is a
    ``(1, 1)`` one, so neither survives ``int()`` directly and both are absent on
    an older result. Returning ``None`` rather than a guess is what lets the
    caller fall back to the whole buffer -- wrong, and wrong in the direction
    that only wastes steps.
    """
    if value is None:
        return None
    try:
        return int(np.asarray(getattr(value, "cpu", lambda: value)()).reshape(-1)[0])
    except Exception:  # pragma: no cover - shape or type this build does not use
        return None


def _as_float(value: Any) -> float | None:
    """The same, for a rate."""
    out = None
    if value is None:
        return None
    try:
        out = float(np.asarray(getattr(value, "cpu", lambda: value)()).reshape(-1)[0])
    except Exception:  # pragma: no cover
        return None
    return out


def _trajectory(rows: np.ndarray) -> dict:
    """Joint rows to the runtime's ``Trajectory``, in plain floats.

    Plain floats rather than numpy, per ``sim_tools.py``'s return-shape rule: a
    tool that hands back an array serialises fine in process and fails the
    moment the same graph is pointed at an RPC tool bundle.
    """
    return {"waypoints": [{"positions": [float(v) for v in row]} for row in np.asarray(rows)]}


def _capture_thread_local() -> None:
    """Make ``torch.cuda.graph`` capture in ``thread_local`` mode by default.

    cuRobo captures its solvers into CUDA graphs with torch's default
    ``capture_error_mode="global"``, under which *any* thread's ``cudaMalloc``
    or synchronise fails and invalidates the capture. The graph policy plans
    on its own thread while the main thread keeps stepping the simulator, and
    a live RTX renderer allocates a small warp array every frame it draws
    (``viewer_rtx._queue_picking_line_transform``): the render died with
    ``Failed to allocate 28 bytes`` and the planner's warmup with
    ``cudaErrorStreamCaptureInvalidated`` -- reproduced 100% with two
    threads, never on one, never under the tiled renderer, which allocates
    nothing per frame. ``thread_local`` is the mode CUDA documents for
    exactly this: only the capturing thread's own unsafe calls count.
    cuRobo passes ``pool``/``stream`` and never the mode, so a default change
    is the whole patch; explicit callers are untouched.
    """
    import torch

    graph_cls = torch.cuda.graph
    if getattr(graph_cls, "_rss_thread_local_capture", False):
        return
    original = graph_cls.__init__

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="thread_local"):
        original(self, cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode)

    __init__.__wrapped__ = original  # type: ignore[attr-defined]
    graph_cls.__init__ = __init__  # type: ignore[method-assign]
    graph_cls._rss_thread_local_capture = True  # type: ignore[attr-defined]


class MotionBackendCuRobo:
    """Collision-aware plans for one embodiment, over cuRobo.

    Args:
        env: The toolkit-layer environment, for the model and the arm indices.
        bus: Marshals anything that must run on the owner thread.

            Planning itself does **not** need the bus, which is the interesting
            part. cuRobo needs seed joints and obstacle poses, and both are on
            ``bus.snapshot`` -- published every step -- so a plan runs on the
            worker thread and never stalls the simulation. That is strictly
            better than :class:`~robosimstudio_policy_agent.ik.IKBackend`,
            whose every solve is a ``bus.call`` the owner thread has to service
            and whose iteration count had to be measured against the 30 s
            watchdog for exactly that reason.
        config_path: The embodiment's cuRobo robot YAML. Empty means this robot
            has no cuRobo model, which is an honest state and raises
            :class:`MotionUnavailable` rather than guessing one.
        urdf_path: Overrides the URDF the YAML names, for a repo-vendored one.
        lock_joints: Joints held fixed during planning, as ``{name: value}``.
            Fingers, normally. Pass the value the *scene* holds them at, not a
            nominal open: a parity check against a mis-set lock reads tens of
            millimetres of disagreement that is not there.
        max_attempts: Retries inside one plan call. Each is a full solve, so
            this multiplies wall clock on the hard problems that need it.
    """

    honours_roll = True
    """cuRobo solves full SE(3).

    Read by :class:`~robosimstudio_policy_agent.servo.CartesianServo` to decide
    whether to test arrival on a full quaternion. **The backend deliberately
    does not forward this one** -- see :class:`~robosimstudio_policy_agent.ik.IKBackend`,
    which answers it from its own solver instead, because the servo's arrival
    test must describe whatever is solving its per-step goals and that is not
    this object.
    """

    def __init__(
        self,
        env: Any,
        bus: Bus,
        *,
        config_path: str = "",
        urdf_path: str = "",
        lock_joints: dict[str, float] | None = None,
        max_attempts: int = 10,
        time_scale: float = 1.0,
    ) -> None:
        if not config_path:
            # Eagerly, not at the first plan. An embodiment with no cuRobo
            # model is a routine state -- most of them have none yet -- and the
            # connector answers it by installing no planner at all. Discovering
            # it mid-episode instead would surface as a node error on whichever
            # move happened to come first, which reads as an unreachable pose
            # rather than a missing file.
            raise MotionUnavailable(
                "this embodiment declares no cuRobo config, so there is no model to plan "
                "against. Set EmbodimentSpec.curobo_config, or let the connector fall back "
                "to the Newton backend."
            )
        self.env = env
        self.bus = bus
        self.config_path = config_path
        self.urdf_path = urdf_path
        self.lock_joints = dict(lock_joints or {})
        self.max_attempts = int(max_attempts)
        self.time_scale = max(1.0, float(time_scale))
        self._planner: Any = None
        self._attached_planners: dict[tuple, Any] = {}
        self._batch_planners: dict[tuple, Any] = {}
        self._frame: str = ""
        self._tcp_cache: dict[int, np.ndarray] = {}
        self._base_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        _capture_thread_local()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _import():
        """Lazy, so a machine with no cuRobo still imports the toolkit.

        Nothing under ``robosimstudio/`` may name cuRobo at module scope.
        """
        try:
            from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.types import GoalToolPose, JointState, Pose
        except ImportError as exc:  # pragma: no cover - depends on the [curobo] extra
            raise MotionUnavailable(f"{CUROBO_INSTALL_HINT} ({exc})") from exc
        return MotionPlanner, MotionPlannerCfg, ToolPoseCriteria, GoalToolPose, JointState, Pose

    def _robot_config(self) -> Any:
        """The robot description to hand ``MotionPlannerCfg.create``.

        A dict rather than a path whenever this stack has something to say
        about the robot, because **the planner config takes no ``urdf_path``
        keyword** -- unlike ``KinematicsCfg.from_robot_yaml_file``, which does.
        The URDF and the locked joints live *inside* the description, under
        ``robot_cfg.kinematics``, so overriding them means editing the mapping
        rather than passing an argument. Passing ``urdf_path=`` raises
        ``TypeError`` at planner construction, which is a slow way to find out.

        **A bare name is cuRobo's own, and only its own constructor knows it.**
        ``EmbodimentSpec.curobo_config`` is ``"franka.yml"`` -- a name in
        cuRobo's shipped robot-config directory, which ``MotionPlannerCfg.create``
        resolves and :func:`curobo.config_io.load_yaml` does not. Whenever this
        method has an override to install it must open the file itself, so the
        two paths disagreed and only the overriding one was ever taken: every
        embodiment here locks its fingers, so ``--motion curobo`` raised
        ``FileNotFoundError: 'franka.yml'`` at the first *plan* -- long after
        construction, as a node error on whichever transport came first. Only a
        name with no separator is resolved this way; a path the caller wrote
        stays the caller's.
        """
        config = self.config_path
        if not (self.urdf_path or self.lock_joints):
            return config
        if isinstance(config, str):
            from curobo.config_io import load_yaml

            resolved_config = self._resolve_config_path(config)
            config = load_yaml(resolved_config)
            # Project-local configs commonly keep URDF meshes beside the YAML.
            # cuRobo resolves its built-in names itself, but a loaded mapping
            # has lost that context, so make relative asset roots explicit.
            from pathlib import Path

            config_dir = Path(resolved_config).resolve().parent
            loaded_kinematics = config.get("robot_cfg", {}).get("kinematics", {})
            asset_root = str(loaded_kinematics.get("asset_root_path", "") or "")
            if asset_root and not Path(asset_root).is_absolute():
                loaded_kinematics["asset_root_path"] = str((config_dir / asset_root).resolve())
            loaded_urdf = str(loaded_kinematics.get("urdf_path", "") or "")
            if loaded_urdf and not Path(loaded_urdf).is_absolute():
                loaded_kinematics["urdf_path"] = str((config_dir / loaded_urdf).resolve())
        config = copy.deepcopy(config)
        kinematics = config.setdefault("robot_cfg", {}).setdefault("kinematics", {})
        if self.urdf_path:
            kinematics["urdf_path"] = self.urdf_path
        if self.lock_joints:
            kinematics["lock_joints"] = dict(self.lock_joints)
        return config

    @staticmethod
    def _resolve_config_path(name: str) -> str:
        """A bare cuRobo robot-config name against cuRobo's own directory.

        Args:
            name: What the embodiment declared. A name with no path separator is
                looked up where cuRobo keeps its shipped robot descriptions;
                anything else is returned unchanged, because a caller who wrote a
                path meant that path.

        Returns:
            A path to open. An unresolvable bare name is returned as written, so
            the failure is still ``FileNotFoundError`` naming what was asked for
            rather than a directory nobody mentioned.
        """
        from pathlib import Path

        if Path(name).parent != Path("."):
            return name
        try:
            from curobo.config_io import join_path
            from curobo.content import get_robot_configs_path
        except ImportError:  # pragma: no cover - depends on the [curobo] extra
            return name
        resolved = join_path(get_robot_configs_path(), name)
        return resolved if Path(resolved).is_file() else name

    def _inline_collision_spheres(self, kinematics: dict) -> dict:
        """``kinematics["collision_spheres"]`` as a mapping, stored back in place.

        cuRobo accepts a mapping, ``None``, or a path to a sphere YAML; adding
        an attached object's spheres to the tool link needs the mapping. A
        relative path is tried beside the robot config first, then where
        cuRobo itself looks -- its shipped robot-config directory.
        """
        spheres = kinematics.get("collision_spheres")
        if isinstance(spheres, str):
            from pathlib import Path

            from curobo.config_io import join_path, load_yaml
            from curobo.content import get_robot_configs_path

            candidates = [] if not Path(spheres).is_absolute() else [spheres]
            if not candidates:
                if isinstance(self.config_path, str):
                    config_dir = Path(self._resolve_config_path(self.config_path)).resolve().parent
                    candidates.append(str(config_dir / spheres))
                candidates.append(join_path(get_robot_configs_path(), spheres))
            path = next((c for c in candidates if Path(c).is_file()), candidates[-1])
            loaded = load_yaml(path) or {}
            spheres = loaded.get("collision_spheres", loaded)
        spheres = dict(spheres or {})
        kinematics["collision_spheres"] = spheres
        return spheres

    @property
    def planner(self) -> Any:
        """The planner, built once and warmed.

        Warmup is paid here rather than inside episode one. cuRobo compiles its
        kernels through NVRTC on first use and captures CUDA graphs when
        ``use_cuda_graph`` is on; a graph's step budget should not be spending
        that.
        """
        if self._planner is None:
            MotionPlanner, MotionPlannerCfg, _, _, _, _ = self._import()
            # cuRobo 0.8 imports ``wp.torch.device_from_torch``. Warp 1.16
            # kept the functions but moved the namespace to ``warp._src``.
            # Install the compatibility alias only when the public namespace
            # is absent; older supported Warp releases remain untouched.
            import warp as wp

            if not hasattr(wp, "torch"):
                from warp._src import torch as warp_torch

                wp.torch = warp_torch
            _capture_thread_local()
            with reading():
                # Allocate a mesh cache up front. The graph may supply a new
                # RGB-D reconstruction for each episode; without a scene
                # checker ``update_world`` cannot make an already-created
                # planner collision aware.
                planner = MotionPlanner(
                    MotionPlannerCfg.create(
                        robot=self._robot_config(),
                        # The voxel entry pre-allocates the ESDF layer build_world_tsdf fills;
                        # its dims/voxel_size must cover that grid exactly, and cuRobo
                        # wants a dict here (an int is a mesh-style cache and crashes
                        # planner construction with 'int' has no attribute 'get').
                        collision_cache={
                            "mesh": 32,
                            # 128^3 at 10 mm: the mapper caps the grid at 128 voxels per axis,
                            # so this is the whole grid it can produce at this resolution --
                            # 1.28 m cubed, centred on the bench, which covers the task volume.
                            "voxel": {"layers": 1, "dims": [1.28, 1.28, 1.28], "voxel_size": 0.01},
                        },
                        position_tolerance=PLANNER_POSITION_TOLERANCE_M,
                        orientation_tolerance=PLANNER_ORIENTATION_TOLERANCE_RAD,
                    )
                )
                planner.warmup()
            # Published only once warm: a warmup that raises (a CUDA-graph
            # capture invalidated, an NVRTC failure) must not leave a
            # half-built planner behind with no tool frame, which every later
            # call would then trip over as ``Link  not found``.
            self._planner = planner
            self._frame = planner.tool_frames[0]
            logger.info(
                "MotionBackendCuRobo: planner up on %s, tool frames %s",
                self.config_path,
                self._planner.tool_frames,
            )
        return self._planner

    def _planner_for(self, attached_object: dict | None, arm_id: int) -> Any:
        """Return a planner whose ``link_6`` carries the supplied spheres.

        CuRobo 0.8's installed ``attachment_manager`` is incomplete. Robot
        collision spheres are nevertheless part of the supported robot
        configuration, so an attachment-specific planner is equivalent for
        collision checking and avoids modifying the third-party package.

        Sphere centers arrive in the runtime TCP frame. They are transformed
        into the configured CuRobo tool-link frame before being appended.
        """
        spheres = list((attached_object or {}).get("spheres") or [])
        if not spheres:
            return self.planner
        tool_tcp = self._tool_from_tcp(arm_id)
        converted = []
        for sphere in spheres:
            center = sphere.get("center") or {}
            tcp_center = np.asarray(
                [center.get("x", 0.0), center.get("y", 0.0), center.get("z", 0.0), 1.0],
                dtype=np.float64,
            )
            tool_center = (tool_tcp @ tcp_center)[:3]
            radius = float(sphere.get("radius", 0.0))
            if radius <= 0.0:
                continue
            converted.append((tool_center, radius))
        if not converted:
            return self.planner
        # The two YAM arms share one arm-only URDF but have independently
        # measured tool-to-TCP transforms.  Never let a geometrically identical
        # attachment on one arm reuse the other arm's planner.
        key = self._attachment_cache_key(arm_id, converted)
        cached = self._attached_planners.get(key)
        if cached is not None:
            return cached

        MotionPlanner, MotionPlannerCfg, _, _, _, _ = self._import()
        config = self._robot_config()
        if isinstance(config, str):
            from curobo.config_io import load_yaml

            config = load_yaml(self._resolve_config_path(config))
        config = copy.deepcopy(config)
        kinematics = config.setdefault("robot_cfg", {}).setdefault("kinematics", {})
        tool_link = self._frame or self.planner.tool_frames[0]
        collision_spheres = self._inline_collision_spheres(kinematics)
        link_spheres = list(collision_spheres.get(tool_link) or [])
        link_spheres.extend(
            {
                "center": [float(v) for v in center],
                "radius": float(radius),
            }
            for center, radius in converted
        )
        collision_spheres[tool_link] = link_spheres

        import warp as wp

        if not hasattr(wp, "torch"):
            from warp._src import torch as warp_torch

            wp.torch = warp_torch
        with reading():
            planner = MotionPlanner(
                MotionPlannerCfg.create(
                    robot=config,
                    # The voxel entry pre-allocates the ESDF layer build_world_tsdf fills;
                    # its dims/voxel_size must cover that grid exactly, and cuRobo
                    # wants a dict here (an int is a mesh-style cache and crashes
                    # planner construction with 'int' has no attribute 'get').
                    collision_cache={
                        "mesh": 32,
                        # 128^3 at 10 mm: the mapper caps the grid at 128 voxels per axis,
                        # so this is the whole grid it can produce at this resolution --
                        # 1.28 m cubed, centred on the bench, which covers the task volume.
                        "voxel": {"layers": 1, "dims": [1.28, 1.28, 1.28], "voxel_size": 0.01},
                    },
                    position_tolerance=PLANNER_POSITION_TOLERANCE_M,
                    orientation_tolerance=PLANNER_ORIENTATION_TOLERANCE_RAD,
                )
            )
            planner.warmup()
        self._attached_planners[key] = planner
        logger.info(
            "MotionBackendCuRobo: attached %d collision spheres to %s",
            len(converted),
            tool_link,
        )
        return planner

    @staticmethod
    def _attachment_cache_key(arm_id: int, converted: list[tuple[np.ndarray, float]]) -> tuple:
        """Stable planner-cache identity for one arm's TCP-frame attachment."""
        return (
            int(arm_id),
            tuple(
                (round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4), round(r, 4)) for c, r in converted
            ),
        )

    def _batch_planner_for(self, attached_object: dict | None, arm_id: int) -> Any:
        """A planner built for :data:`IK_BATCH_MAX_CANDIDATES` independent
        problems at once, with the same attachment spheres :meth:`_planner_for`
        would give a single-goal call -- cached separately, because it is not
        interchangeable with one.

        Two things make this its own planner rather than a parameter on the
        existing one:

        * cuRobo sizes its goal buffers from the *first* solve after
          construction and, with CUDA graphs on, cannot resize afterwards
          (measured: a batch=1 solve first, batch=32 second, raises
          ``output with shape [1, 7] doesn't match the broadcast shape
          [32, 7]``; calling ``reset_shape()`` to force it raises ``CUDA
          graph reset is not available``). A planner that ever answers a
          single-goal call can therefore never also answer a batched one.
        * :meth:`MotionPlanner.warmup` itself solves at batch=1 internally
          when ``max_goalset <= 1`` (its ``else`` branch) -- so calling the
          library's own ``warmup()`` on a batch-sized planner is what
          triggers the failure above. This planner warms itself instead, by
          solving at its own batch size before the first real call.
        """
        spheres = list((attached_object or {}).get("spheres") or [])
        converted = []
        if spheres:
            tool_tcp = self._tool_from_tcp(arm_id)
            for sphere in spheres:
                center = sphere.get("center") or {}
                tcp_center = np.asarray(
                    [center.get("x", 0.0), center.get("y", 0.0), center.get("z", 0.0), 1.0],
                    dtype=np.float64,
                )
                tool_center = (tool_tcp @ tcp_center)[:3]
                radius = float(sphere.get("radius", 0.0))
                if radius > 0.0:
                    converted.append((tool_center, radius))
        # A coarser key than the single-goal path's: the batch planner backs
        # a *search* whose winning candidate is re-validated (and its actual
        # trajectory re-planned) at execution time by plan_to_pose/plan_linear
        # against the exact attachment -- so this cache does not need to
        # distinguish two grasps of the same object that measured a
        # millimetre apart. It needs to, on the two-syringe task: the second
        # syringe's perceived sphere geometry never matches the first's
        # exactly, so the single-goal path's 4-decimal key missed on pass 2
        # and rebuilt (re-JITed) a whole planner for it -- measured live,
        # both passes of a two-syringe episode paying ~20s each instead of
        # the second reusing the first's warm one. Sphere count and a 1 cm
        # rounded reach are enough to tell "roughly this size of thing" apart
        # without over-matching.
        reach = round(max((float(np.linalg.norm(c)) + r for c, r in converted), default=0.0), 2)
        key = ("batch", int(arm_id), len(converted), reach)
        cached = self._batch_planners.get(key)
        if cached is not None:
            return cached

        MotionPlanner, MotionPlannerCfg, _, GoalToolPose, JointState, Pose = self._import()
        config = self._robot_config()
        if isinstance(config, str):
            from curobo.config_io import load_yaml

            config = load_yaml(self._resolve_config_path(config))
        config = copy.deepcopy(config)
        if converted:
            kinematics = config.setdefault("robot_cfg", {}).setdefault("kinematics", {})
            tool_link = self._frame or self.planner.tool_frames[0]
            collision_spheres = self._inline_collision_spheres(kinematics)
            link_spheres = list(collision_spheres.get(tool_link) or [])
            link_spheres.extend(
                {"center": [float(v) for v in center], "radius": float(radius)} for center, radius in converted
            )
            collision_spheres[tool_link] = link_spheres

        import warp as wp

        if not hasattr(wp, "torch"):
            from warp._src import torch as warp_torch

            wp.torch = warp_torch
        with reading():
            planner = MotionPlanner(
                MotionPlannerCfg.create(
                    robot=config,
                    collision_cache={
                        "mesh": 32,
                        "voxel": {"layers": 1, "dims": [1.28, 1.28, 1.28], "voxel_size": 0.01},
                    },
                    position_tolerance=PLANNER_POSITION_TOLERANCE_M,
                    orientation_tolerance=PLANNER_ORIENTATION_TOLERANCE_RAD,
                    max_batch_size=IK_BATCH_MAX_CANDIDATES,
                )
            )
            frame = self._frame or planner.tool_frames[0]
            solver = planner.ik_solver
            import torch

            warm_position = torch.zeros((IK_BATCH_MAX_CANDIDATES, 3), dtype=torch.float32, device="cuda")
            warm_quat = torch.zeros((IK_BATCH_MAX_CANDIDATES, 4), dtype=torch.float32, device="cuda")
            warm_quat[:, 0] = 1.0
            warm_goal = GoalToolPose.from_poses(
                {frame: Pose(position=warm_position, quaternion=warm_quat)},
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )
            warm_seed = torch.zeros(
                (IK_BATCH_MAX_CANDIDATES, len(self.env.handles.arms[arm_name(self.env, arm_id)].arm_coords)),
                dtype=torch.float32,
                device="cuda",
            )
            warm_seeds = (
                warm_seed[:, None, :]
                .expand(IK_BATCH_MAX_CANDIDATES, int(solver.config.num_seeds), warm_seed.shape[-1])
                .contiguous()
            )
            for _ in range(3):
                solver.solve_pose(warm_goal, current_state=JointState.from_position(warm_seed), seed_config=warm_seeds)
            torch.cuda.synchronize()
        self._batch_planners[key] = planner
        return planner

    def solve_ik_batch(
        self,
        target_world_poses: list[dict],
        *,
        arm_id: int = 0,
        seed_joints: Any = None,
        attached_object: dict | None = None,
    ) -> list[dict | None]:
        """Independent IK feasibility for many candidate poses at once, from
        one shared seed -- the wrist-turn heading search's own primitive
        (its serial ``motion.plan_to_pose(hold_position=True)`` per yaw,
        keeping only ``_end_joints(turned)``), batched on the GPU.

        IK only: no trajopt, no executable trajectory. That is what the
        search's serial calls were already discarding down to -- it read
        just the last waypoint of each planned trajectory as the seed for
        its next stage, so a bare final joint configuration answers the
        search's own question exactly. The winning candidate's actual
        executable motion is planned again at execution time
        (``execute_reorientation.py`` calls ``motion.plan_to_pose``/
        ``motion.plan_linear`` itself, from the live posture) -- this
        method's output is never streamed to the robot.

        Measured against :data:`IK_BATCH_MAX_CANDIDATES` serial single-goal
        solves through a freshly built single-goal planner (both warm, same
        GPU): 13.4 ms for one batched call of 12 vs 42.7 ms serial -- 3.2x,
        and the gap only widens with more candidates since the serial side
        is linear and the batched side is close to flat up to the planner's
        configured ceiling.

        Returns one entry per input pose, ``None`` for a pose IK could not
        reach, else ``{"joint_config": [...]}`` -- the arm's own joint order,
        ready to seed a following ``motion.plan_linear``/``plan_to_pose``
        call the way the serial search's ``_end_joints(turned)`` did.
        """
        import torch

        _, _, _, GoalToolPose, JointState, Pose = self._import()
        n = len(target_world_poses)
        if n == 0:
            return []
        if n > IK_BATCH_MAX_CANDIDATES:
            raise ValueError(f"solve_ik_batch: {n} candidates exceeds the {IK_BATCH_MAX_CANDIDATES} ceiling")
        planner = self._batch_planner_for(attached_object, arm_id)
        frame = self._frame or planner.tool_frames[0]
        solver = planner.ik_solver
        seed_values = self._seed(arm_id, seed_joints)
        dof = len(seed_values)
        tool_inv = np.linalg.inv(self._tool_from_tcp(arm_id))

        # Pad up to the planner's fixed batch size by repeating the last real
        # candidate -- a wasted solve, not a wrong one, and the padding rows
        # are dropped again before returning.
        padded = list(target_world_poses) + [target_world_poses[-1]] * (IK_BATCH_MAX_CANDIDATES - n)
        positions, quats = [], []
        for pose in padded:
            position, quat = _se3_to_arrays(pose)
            world_tool = _homogeneous(position, quat) @ tool_inv
            base_position, base_quat = self._to_base(world_tool[:3, 3], mat_to_quat(world_tool[:3, :3]), arm_id)
            positions.append(base_position)
            quats.append(quat_xyzw_to_wxyz(base_quat))
        pos_t = torch.as_tensor(np.stack(positions), dtype=torch.float32, device="cuda")
        quat_t = torch.as_tensor(np.stack(quats), dtype=torch.float32, device="cuda")
        goal = GoalToolPose.from_poses(
            {frame: Pose(position=pos_t, quaternion=quat_t)},
            ordered_tool_frames=planner.tool_frames,
            num_goalset=1,
        )
        seed_batch = torch.as_tensor(
            np.tile(seed_values[None], (IK_BATCH_MAX_CANDIDATES, 1)), dtype=torch.float32, device="cuda"
        )
        seeds = seed_batch[:, None, :].expand(IK_BATCH_MAX_CANDIDATES, int(solver.config.num_seeds), dof).contiguous()

        # plan_pose (what every single-goal call in this file goes through)
        # retries a failed IK solve up to self.max_attempts times -- a bare
        # single solve_pose call measurably fails candidates the serial path
        # would have found, since a candidate seed set can miss on one draw
        # and land on the next (pretested against three recorded episodes:
        # a bare single attempt turned three previously-successful runs into
        # "no heading" failures; matching plan_pose's own retry count fixed
        # all three). Retried as a *batch* each time, not per candidate --
        # cheap regardless, since only the still-unsolved rows matter to a
        # caller and none are dropped by retrying the ones already solved.
        success = np.zeros(IK_BATCH_MAX_CANDIDATES, dtype=bool)
        solutions = np.zeros((IK_BATCH_MAX_CANDIDATES, dof), dtype=np.float64)
        with reading():
            for _attempt in range(max(1, self.max_attempts)):
                remaining = ~success
                if not remaining.any():
                    break
                try:
                    result = solver.solve_pose(
                        goal, current_state=JointState.from_position(seed_batch), seed_config=seeds
                    )
                except Exception:
                    logger.exception("MotionBackendCuRobo: solve_ik_batch raised")
                    break
                attempt_success = result.success.reshape(-1).detach().cpu().numpy()
                active = planner.kinematics.get_active_js(result.js_solution)
                attempt_solutions = np.asarray(active.position.detach().cpu().numpy(), dtype=np.float64).reshape(
                    IK_BATCH_MAX_CANDIDATES, -1
                )[:, :dof]
                newly_solved = remaining & attempt_success
                solutions[newly_solved] = attempt_solutions[newly_solved]
                success |= attempt_success

        out: list[dict | None] = []
        for i in range(n):
            if not bool(success[i]):
                out.append(None)
                continue
            joints = solutions[i]
            position, quat = target_world_poses[i].get("position"), target_world_poses[i].get("rotation")
            target_pos, target_quat = _se3_to_arrays({"position": position, "rotation": quat})
            if not self._endpoint_matches(joints, target_pos, target_quat, arm_id):
                out.append(None)
                continue
            out.append({"joint_config": [float(v) for v in joints]})
        return out

    @property
    def supports_world_aware_plan(self) -> bool:
        """True: a plan here routes around the world it was handed, which is the whole reason to ask for it."""
        return True

    @property
    def trajectory_needs_joint_reverse(self) -> bool:
        """False: cuRobo returns this arm's joints in its own declared order,
        which the link map aligns with the model's."""
        return False

    # ------------------------------------------------------------------ frames
    def _base(self, arm_id: int) -> tuple[np.ndarray, np.ndarray]:
        """This arm's base transform in world, as ``(R, t)``.

        Measured off the live model rather than read from a config, and as a
        full SE(3) rather than a translation. The runtime's own cuRobo backend
        subtracts an ``arm_bases`` xyz and copies the rotation through verbatim,
        which is wrong for any yawed pedestal -- and the UR5's own
        ``base_offset`` is a half turn about z.
        """
        cached = self._base_cache.get(arm_id)
        if cached is not None:
            return cached
        name = arm_name(self.env, arm_id)
        body = getattr(self.env.handles.arms[name], "base_body", -1)
        if body < 0:
            # A scene-resident robot declares no base body -- tool_sort's Panda
            # is welded into the robosuite arena and reports -1 -- so fall back
            # to the body the *planner* calls its base and find it by label
            # suffix, which is how ik_pyroki anchors its own URDF. The suffix
            # match is doing real work: cuRobo says ``panda_link0`` and the
            # scene says ``.../robot0_base/robot0_link0``, so only the tail
            # segment is common to both vocabularies.
            # Prefer the fixed base link whenever it is unique.  The
            # scene-resident Panda has no declared base_body, but its
            # robot0_link0 is retained and is the stable anchor; deriving its
            # base through the wrist makes the cached base depend on posture.
            try:
                body = self._body_like(self.planner.kinematics.base_link)
            except MotionUnavailable:
                body = -1
            if body >= 0:
                xform = np.asarray(self.env.state.body_q.numpy()[body], dtype=np.float64)
                result = (quat_to_mat(xform[3:7]), xform[:3])
                self._base_cache[arm_id] = result
                return result

            # A bimanual import can legitimately collapse both fixed mount
            # bodies. In that case a name search for the generic URDF
            # ``base_link`` is either empty or ambiguous. Anchor the base from
            # a frame both models retain instead: the selected arm's wrist.
            #
            #   world_T_base = world_T_wrist(sim) * inv(base_T_wrist(curobo))
            #
            # Cache the result. Re-estimating it at every configuration would
            # hide a wrong URDF by letting the base move to absorb FK error.
            wrist = int(self.env.handles.arms[name].ee_body)
            world_wrist_raw = np.asarray(self.env.state.body_q.numpy()[wrist], dtype=np.float64)
            world_wrist = _homogeneous(world_wrist_raw[:3], world_wrist_raw[3:7])
            base_wrist = self._planner_tool_in_base(self._seed(arm_id, None))
            world_base = world_wrist @ np.linalg.inv(base_wrist)
            result = (world_base[:3, :3], world_base[:3, 3])
            self._base_cache[arm_id] = result
            logger.info(
                "MotionBackendCuRobo: derived %s base from retained wrist %s at %s",
                name,
                self._frame,
                np.array2string(world_base[:3, 3], precision=5),
            )
            return result
        xform = np.asarray(self.env.state.body_q.numpy()[body], dtype=np.float64)
        result = (quat_to_mat(xform[3:7]), xform[:3])
        self._base_cache[arm_id] = result
        return result

    def _planner_tool_in_base(self, arm_joints: np.ndarray) -> np.ndarray:
        """CuRobo's configured tool frame in its base frame at ``arm_joints``."""
        import torch  # only on the planning path

        _, _, _, _, JointState, _ = self._import()
        planner = self.planner
        with reading():
            state = planner.compute_kinematics(
                JointState.from_position(torch.as_tensor(np.asarray(arm_joints, dtype=np.float32)[None], device="cuda"))
            )
        pose = state.tool_poses[self._frame]
        position = pose.position.detach().cpu().numpy().astype(np.float64).reshape(3)
        quaternion = quat_wxyz_to_xyzw(pose.quaternion.detach().cpu().numpy().astype(np.float64).reshape(4))
        return _homogeneous(position, quaternion)

    def _body_like(self, link: str) -> int:
        """The one Newton body whose label ends the way *link* does.

        Ambiguity is an error rather than a first match: on a bimanual robot
        ``link0`` matches both arms, and silently taking one of them would plan
        the whole trajectory in the wrong shoulder's frame -- an error of the
        distance between the two mounts, which is metres.
        """
        tail = link.rsplit("_", 1)[-1] if "_" in link else link
        labels = [str(v) for v in self.env.model.body_label]
        matches = [i for i, label in enumerate(labels) if label.endswith(tail)]
        if len(matches) != 1:
            raise MotionUnavailable(
                f"cannot anchor the planner's base link {link!r}: {len(matches)} bodies end "
                f"in {tail!r}. Give the arm a base_body, or map this embodiment's link names "
                f"explicitly -- guessing one of {len(matches)} would plan in the wrong frame."
            )
        return matches[0]

    def _to_base(self, position: np.ndarray, quat_xyzw: np.ndarray, arm_id: int) -> tuple[np.ndarray, np.ndarray]:
        """A world pose into the arm's base frame, which is what cuRobo plans in."""
        rotation, translation = self._base(arm_id)
        return rotation.T @ (position - translation), mat_to_quat(rotation.T @ quat_to_mat(quat_xyzw))

    def _tool_from_tcp(self, arm_id: int) -> np.ndarray:
        """``T_tcp→tool`` for this arm, **measured**, as a 4x4.

        The runtime hands TCP-frame poses. cuRobo's ``tool_frames`` are URDF
        links -- ``panda_hand``, not the tool point -- so a TCP pose handed
        straight over asks the planner to put the *wrist* where the *fingertips*
        should be. Measured on ``tool_sort/level1_6`` before this existed: a
        120 mm descend arrived **96.5 mm** short, which is the Panda's
        103 mm hand-to-TCP offset almost exactly.

        Measured rather than read from config, per the rule
        ``ik_backend.py``'s offset table exists to enforce. And measured at two
        configurations rather than one, because a constant offset is what this
        is claiming: if it moves with the arm, the two frames are not related by
        a fixed transform and applying an average would be worse than applying
        nothing.
        """
        cached = self._tcp_cache.get(arm_id)
        if cached is not None:
            return cached

        name = arm_name(self.env, arm_id)
        ik = self.env.arm(name)
        coords = list(self.env.handles.arms[name].arm_coords)
        joints = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)

        transforms = []
        for nudge in (0.0, 0.15):
            probe = joints.copy()
            for coord in coords:
                probe[coord] += nudge
            tcp_position, tcp_quat = self._fk_tcp(ik, probe)
            tool_position, tool_quat = self._fk_tool(probe[coords], arm_id)
            world_tcp = _homogeneous(tcp_position, tcp_quat)
            world_tool = _homogeneous(tool_position, tool_quat)
            transforms.append(np.linalg.inv(world_tool) @ world_tcp)

        drift = float(np.linalg.norm(transforms[0][:3, 3] - transforms[1][:3, 3])) * 1000.0
        if drift > TOOL_DRIFT_MM:
            raise MotionUnavailable(
                f"arm {name!r}: the tool-to-TCP transform moved {drift:.2f} mm between two "
                f"configurations, so the two frames are not a fixed offset apart. Applying an "
                f"average would put every plan wrong by a varying amount."
            )
        self._tcp_cache[arm_id] = transforms[0]
        return transforms[0]

    def _fk_tcp(self, ik: Any, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where the *runtime* says the tool point is, at this configuration."""
        # ``ee_body`` is only the wrist/link frame.  On Panda that frame happens
        # to be the connector's TCP, which let the old body-only implementation
        # look correct.  YAM's TCP is the grasp site 134.7 mm along link_6 and
        # also carries a fixed rotation.  ArmIK owns both transforms and its
        # FK is the single source of truth used by the Cartesian controller.
        #
        # Omitting this offset corrupted two independent safety boundaries:
        # ``_tool_from_tcp`` calibrated link_6->link_6 (identity), so CuRobo was
        # asked to place the wrist at every TCP goal; and ``_endpoint_matches``
        # validated that same wrong wrist pose, allowing the bad trajectory to
        # execute.  Delegate instead of duplicating the offset composition.
        position, quaternion = ik.fk_ee(np.asarray(joints, dtype=np.float32))
        return (
            np.asarray(position, dtype=np.float64).reshape(3),
            np.asarray(quaternion, dtype=np.float64).reshape(4),
        )

    def _fk_tool(self, arm_joints: np.ndarray, arm_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Where *cuRobo* says its tool frame is, in world, at the same joints."""
        base_tool = self._planner_tool_in_base(arm_joints)
        rotation, translation = self._base(arm_id)
        world_tool = _homogeneous(translation, mat_to_quat(rotation)) @ base_tool
        return world_tool[:3, 3], mat_to_quat(world_tool[:3, :3])

    # ------------------------------------------------------------------ the API
    def plan_linear(
        self,
        start_world_pose: dict,
        end_world_pose: dict,
        *,
        arm_id: int = 0,
        tcp_offset: Any = None,
        seed_joints: Any = None,
        num_waypoints: int = 0,
        axis: str = "",
        non_terminal_scale: float = 1.0,
        world_config: dict | None = None,
        attached_object: dict | None = None,
        allow_start_contact: bool = False,
        allow_goal_contact: bool = False,
        contact_margin: float = 0.005,
        tolerance: float | None = None,
    ) -> dict | None:
        """A straight line to *end*, held straight by the planner rather than by
        interpolation.

        ``tolerance`` is the position error the caller will accept at the
        endpoint [m], the same keyword the solved line takes; ``None`` keeps
        the planner's own parity gate (``ENDPOINT_PARITY_MM``).

        This is the primitive the reference stack calls ``plan_directed_linear``.
        Unlike :meth:`IKBackend.plan_linear` it is collision aware: the
        path is a *plan* that happens to be constrained, not a Cartesian
        interpolation solved per waypoint, so an obstacle between the endpoints
        is routed around within the constraint rather than driven through.

        Args:
            axis: Which tool axis travel is along, ``"x"``, ``"y"`` or ``"z"``.
                Empty derives it from the actual travel direction, which is the
                safe default -- naming an axis that travel is not along
                constrains the two axes the move needs and produces a path
                measurably worse than not constraining at all.
            non_terminal_scale: Weight on the held axes. 1.0 is cuRobo's own.

        Returns:
            A ``Trajectory``, or ``None`` if no plan was found -- never an
            exception, because the connector's tools decide what an unreachable
            pose means.
        """
        del tcp_offset  # the calibrated transform governs; see the module docstring
        _, _, ToolPoseCriteria, _, _, _ = self._import()

        # Only the start's *position* matters: the travel direction is what
        # picks the constrained axis, and the orientation held along the way is
        # the goal's, not the start's.
        start_position, _ = _se3_to_arrays(start_world_pose)
        end_position, end_quat = _se3_to_arrays(end_world_pose)
        travel = end_position - start_position
        distance = float(np.linalg.norm(travel))

        criteria = None
        if distance >= MIN_TRAVEL_M:
            chosen = axis or self._axis_for(travel, end_quat)
            if chosen:
                criteria = ToolPoseCriteria.linear_motion(
                    axis=chosen,
                    non_terminal_scale=float(non_terminal_scale),
                    project_distance_to_goal=True,
                )
            else:
                logger.info(
                    "plan_linear: travel is not along a tool axis (%.1f mm), planning free "
                    "rather than constraining the axes the move needs",
                    distance * 1000.0,
                )

        plan = None
        if criteria is not None:
            plan = self._plan(
                end_position,
                end_quat,
                arm_id,
                seed_joints,
                criteria,
                num_waypoints,
                world_config=world_config,
                attached_object=attached_object,
                allow_start_contact=allow_start_contact,
                allow_goal_contact=allow_goal_contact,
                contact_margin=contact_margin,
            )
            if plan is not None:
                straight, lateral_m, backtrack_m = self._straightness(plan, arm_id)
                if not straight:
                    logger.info(
                        "plan_linear: the planner's constrained path bows %.1f mm off the line "
                        "(backtracks %.1f mm); tracking the line by IK instead",
                        lateral_m * 1000.0,
                        backtrack_m * 1000.0,
                    )
                    plan = None
        if plan is None and criteria is None:
            # No tool axis to free (travel is not along one, or there is
            # almost none) -- the planner's own plan, as before, decides pure
            # rotations and re-postures; only a real translation is a line.
            if distance < LINEAR_MIN_LINE_M or not self._orientation_held(seed_joints, end_quat, arm_id):
                plan = self._plan(
                    end_position,
                    end_quat,
                    arm_id,
                    seed_joints,
                    None,
                    num_waypoints,
                    world_config=world_config,
                    attached_object=attached_object,
                    allow_start_contact=allow_start_contact,
                    allow_goal_contact=allow_goal_contact,
                    contact_margin=contact_margin,
                )
                if plan is not None:
                    return plan
        if plan is None and distance >= LINEAR_MIN_LINE_M and self._orientation_held(seed_joints, end_quat, arm_id):
            plan = self._ik_line(
                end_position,
                end_quat,
                arm_id,
                seed_joints,
                world_config=world_config,
                attached_object=attached_object,
                allow_start_contact=allow_start_contact,
                allow_goal_contact=allow_goal_contact,
                contact_margin=contact_margin,
                # The caller's endpoint gate, which the solved line checks its
                # last waypoint against. It used to be read as a free name here
                # and raised NameError on every leg that reached the IK line --
                # measured on sharps_disposal/tray_pair_both_t00's carry.
                tolerance=tolerance,
            )
        return plan

    def _orientation_held(self, seed_joints: Any, end_quat: np.ndarray, arm_id: int) -> bool:
        """Whether the arm already holds the goal orientation, within
        :data:`LINEAR_TURN_MAX_DEG` -- the precondition for a line that holds
        the goal orientation at every waypoint. A leg that also has to turn the
        hand (the yaw-in-place fallback, a re-posture) is the planner's."""
        name = arm_name(self.env, arm_id)
        handle = self.env.handles.arms[name]
        full = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64).copy()
        full[list(handle.arm_coords)] = self._seed(arm_id, seed_joints)[: len(handle.arm_coords)]
        _, quat = self._fk_tcp(self.env.arm(name), full)
        dot = abs(float(np.dot(quat / np.linalg.norm(quat), end_quat / np.linalg.norm(end_quat))))
        return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))) <= LINEAR_TURN_MAX_DEG

    def _straightness(self, plan: dict, arm_id: int) -> tuple[bool, float, float]:
        """Whether a plan's tool point stays on the chord between its ends.

        Runtime FK of every row, against the chord from the first row's tool
        point to the last's: the largest distance off the chord, and the
        largest reversal of progress along it. The chord is the plan's own,
        not the caller's nominal start, because the arm flies from where it
        is.
        """
        name = arm_name(self.env, arm_id)
        handle = self.env.handles.arms[name]
        arm = self.env.arm(name)
        full = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64).copy()
        points = []
        for waypoint in plan.get("waypoints", []):
            full[list(handle.arm_coords)] = np.asarray(waypoint["positions"], dtype=np.float64)[
                : len(handle.arm_coords)
            ]
            points.append(self._fk_tcp(arm, full)[0])
        if len(points) < 2:
            return True, 0.0, 0.0
        points = np.asarray(points)
        chord = points[-1] - points[0]
        length = float(np.linalg.norm(chord))
        if length < MIN_TRAVEL_M:
            return True, 0.0, 0.0
        direction = chord / length
        along = (points - points[0]) @ direction
        lateral = float(np.linalg.norm((points - points[0]) - np.outer(along, direction), axis=1).max())
        backtrack = float(np.max(np.maximum.accumulate(along) - along))
        return (lateral <= LINEAR_TUBE_M and backtrack <= LINEAR_BACKTRACK_M), lateral, backtrack

    def _ik_line(
        self,
        end_position: np.ndarray,
        end_quat: np.ndarray,
        arm_id: int,
        seed_joints: Any,
        *,
        world_config: dict | None,
        attached_object: dict | None,
        allow_start_contact: bool,
        allow_goal_contact: bool,
        contact_margin: float,
        tolerance: float | None = None,
    ) -> dict | None:
        """A straight line held straight by construction: the chord from the
        arm's tool point to ``end``, sampled every :data:`LINEAR_STEP_M` and
        solved waypoint by waypoint with the planner's own collision-aware IK,
        each solve seeded from the last so the chain stays in one joint-space
        branch. The goal orientation is held at every waypoint.

        Collision aware in the same world and with the same attachment the
        planner uses -- the IK solver shares its scene checker -- so this is
        not the Newton backend's blind interpolation; a waypoint the world
        forbids refuses the whole line, and a branch switch does too.
        """
        import torch  # only on the planning path

        _, _, _, GoalToolPose, JointState, Pose = self._import()
        planner = self._planner_for(attached_object, arm_id)
        seed_values = self._seed(arm_id, seed_joints)
        contact_spheres = self._endpoint_contact_spheres(
            attached_object,
            seed_values,
            end_position,
            end_quat,
            arm_id,
            allow_start_contact=allow_start_contact,
            allow_goal_contact=allow_goal_contact,
            margin=contact_margin,
        )
        try:
            self._update_world(
                planner,
                world_config,
                arm_id,
                allowed_contact_spheres=contact_spheres,
                exclude_contacting_meshes=bool(allow_start_contact != allow_goal_contact),
            )
        except Exception:
            logger.exception("MotionBackendCuRobo: could not load collision world")
            return None
        name = arm_name(self.env, arm_id)
        handle = self.env.handles.arms[name]
        arm = self.env.arm(name)
        full = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64).copy()
        full[list(handle.arm_coords)] = seed_values[: len(handle.arm_coords)]
        start_position, _ = self._fk_tcp(arm, full)
        travel = end_position - start_position
        distance = float(np.linalg.norm(travel))
        if distance < MIN_TRAVEL_M:
            return _trajectory(np.asarray([seed_values]))
        steps = max(2, int(np.ceil(distance / LINEAR_STEP_M)))
        tool_from_tcp_inv = np.linalg.inv(self._tool_from_tcp(arm_id))
        solver = planner.ik_solver
        rows = [np.asarray(seed_values, dtype=np.float64)]
        seed = rows[0]
        dof = len(seed)
        for k in range(1, steps + 1):
            fraction = k / steps
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            position = start_position + travel * eased
            world_tool = _homogeneous(position, end_quat) @ tool_from_tcp_inv
            base_position, base_quat = self._to_base(world_tool[:3, 3], mat_to_quat(world_tool[:3, :3]), arm_id)
            goal = Pose(
                position=torch.as_tensor(base_position[None], dtype=torch.float32, device="cuda"),
                quaternion=torch.as_tensor(quat_xyzw_to_wxyz(base_quat)[None], dtype=torch.float32, device="cuda"),
            )
            goal_tool = GoalToolPose.from_poses(
                {self._frame: goal}, ordered_tool_frames=planner.tool_frames, num_goalset=1
            )
            seed_tensor = torch.as_tensor(seed[None], dtype=torch.float32, device="cuda")
            # Every seed the solver runs starts at the previous waypoint's
            # solution -- the point is to stay in that branch, not to explore.
            seeds = seed_tensor[:, None, :].expand(1, int(solver.config.num_seeds), dof).contiguous()
            with reading():
                try:
                    result = solver.solve_pose(
                        goal_tool, current_state=JointState.from_position(seed_tensor), seed_config=seeds
                    )
                except Exception:
                    logger.exception("MotionBackendCuRobo: IK line solve raised")
                    return None
            success = result is not None and result.success is not None and bool(result.success.reshape(-1)[0])
            if not success:
                logger.warning(
                    "plan_linear: IK line refused at %.0f%% of a %.0f mm leg (waypoint unreachable or in collision)",
                    100.0 * fraction,
                    distance * 1000.0,
                )
                return None
            active = planner.kinematics.get_active_js(result.js_solution)
            solution = np.asarray(active.position.detach().cpu().numpy(), dtype=np.float64).reshape(-1, dof)[0]
            jump = float(np.abs(solution - seed).max())
            if jump > LINEAR_JOINT_JUMP_RAD:
                logger.warning(
                    "plan_linear: IK line switched joint branch at %.0f%% (%.2f rad between neighbours); refusing",
                    100.0 * fraction,
                    jump,
                )
                return None
            rows.append(solution)
            seed = solution
        rows = np.asarray(rows)
        if not self._endpoint_matches(rows[-1], end_position, end_quat, arm_id, tolerance):
            return None
        return _trajectory(rows)

    def _axis_for(self, travel: np.ndarray, goal_quat_xyzw: np.ndarray) -> str:
        """The goal-frame axis travel runs along, or ``""`` if none does.

        The whole reason :attr:`AXIS_ALIGNMENT` exists. ``linear_motion`` frees
        one axis of the *goal* frame and holds the other five degrees of
        freedom, so picking the axis by anything other than where the move
        actually goes inverts the constraint.
        """
        direction = travel / np.linalg.norm(travel)
        frame = quat_to_mat(goal_quat_xyzw)
        alignments = {name: abs(float(direction @ frame[:, i])) for i, name in enumerate("xyz")}
        best = max(alignments, key=alignments.__getitem__)
        return best if alignments[best] >= AXIS_ALIGNMENT else ""

    def plan_to_pose(
        self,
        target_world_pose: dict,
        *,
        arm_id: int = 0,
        seed_joints: Any = None,
        num_waypoints: int = 0,
        world_config: dict | None = None,
        hold_position: bool = False,
        attached_object: dict | None = None,
        allow_start_contact: bool = False,
        allow_goal_contact: bool = False,
        contact_margin: float = 0.005,
    ) -> dict | None:
        """A free-space collision-aware plan. Deliberately *not* the default.

        Offered, and never used to back an existing verb. ``waypoint_move.py``
        records the measured cost of letting a planner optimise a shaped path:
        the shortest smooth route dips low while translating and drags the held
        object through the scene. A caller reaching for this is asking for the
        shortest safe path, which is a different request from "go in a straight
        line", and should have to say so.
        """
        position, quat = _se3_to_arrays(target_world_pose)
        _, _, ToolPoseCriteria, _, _, _ = self._import()
        criteria = None
        if hold_position:
            criteria = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            )
        return self._plan(
            position,
            quat,
            arm_id,
            seed_joints,
            criteria,
            num_waypoints,
            world_config=world_config,
            attached_object=attached_object,
            allow_start_contact=allow_start_contact,
            allow_goal_contact=allow_goal_contact,
            contact_margin=contact_margin,
        )

    def solve_ik(
        self,
        target_world_pose: dict,
        *,
        arm_id: int = 0,
        seed_joints: Any = None,
        tcp_offset: Any = None,
    ) -> list[float] | None:
        """The last configuration of a plan to *target*.

        Present for interface parity and **not** what the servo should call: it
        is a full motion plan, tens of milliseconds, where the servo wants
        hundreds of microseconds per control step. The composite routes
        ``solve_ik`` to a cheap solver for that reason.
        """
        del tcp_offset
        plan = self.plan_to_pose(target_world_pose, arm_id=arm_id, seed_joints=seed_joints)
        if plan is None or not plan["waypoints"]:
            return None
        return list(plan["waypoints"][-1]["positions"])

    # ------------------------------------------------------------------ solving
    def _seed(self, arm_id: int, seed_joints: Any) -> np.ndarray:
        """This arm's joints to plan from -- the caller's, or the live ones."""
        if seed_joints is not None:
            return np.asarray(seed_joints, dtype=np.float64).reshape(-1)
        name = arm_name(self.env, arm_id)
        coords = self.env.handles.arms[name].arm_coords
        joints = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)
        return joints[list(coords)]

    def _plan(
        self,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        arm_id: int,
        seed_joints: Any,
        criteria: Any,
        num_waypoints: int,
        world_config: dict | None = None,
        attached_object: dict | None = None,
        allow_start_contact: bool = False,
        allow_goal_contact: bool = False,
        contact_margin: float = 0.005,
    ) -> dict | None:
        import torch  # only on the planning path

        _, _, ToolPoseCriteria, GoalToolPose, JointState, Pose = self._import()
        planner = self._planner_for(attached_object, arm_id)
        frames = planner.tool_frames

        # The start is the current proprioceptive joint state; the goal is the
        # requested TCP pose.  Endpoint contact permissions carve only the
        # observed held-object volume at those two poses.  The remainder of
        # every mesh, and therefore the complete trajectory interior, stays
        # collision checked.
        seed_values = self._seed(arm_id, seed_joints)
        contact_spheres = self._endpoint_contact_spheres(
            attached_object,
            seed_values,
            position,
            quat_xyzw,
            arm_id,
            allow_start_contact=allow_start_contact,
            allow_goal_contact=allow_goal_contact,
            margin=contact_margin,
        )

        # Perception reports meshes in the shared world frame. cuRobo's scene
        # checker lives in this arm's base frame, which differs for the two
        # YAM mounts, so transform every vertex before loading the scene.
        try:
            self._update_world(
                planner,
                world_config,
                arm_id,
                allowed_contact_spheres=contact_spheres,
                exclude_contacting_meshes=bool(allow_start_contact != allow_goal_contact),
            )
        except Exception:
            # Never downgrade a requested collision-aware move to free space.
            logger.exception("MotionBackendCuRobo: could not load collision world")
            return None

        # TCP-frame in, tool-frame out. Without this the planner is asked to put
        # the wrist where the fingertips belong -- 96.5 mm of a 120 mm descend
        # on the Panda. See _tool_from_tcp.
        world_tool = _homogeneous(position, quat_xyzw) @ np.linalg.inv(self._tool_from_tcp(arm_id))
        base_position, base_quat = self._to_base(world_tool[:3, 3], mat_to_quat(world_tool[:3, :3]), arm_id)
        goal = Pose(
            position=torch.as_tensor(base_position[None], dtype=torch.float32, device="cuda"),
            quaternion=torch.as_tensor(quat_xyzw_to_wxyz(base_quat)[None], dtype=torch.float32, device="cuda"),
        )
        goal_tool = GoalToolPose.from_poses({self._frame: goal}, ordered_tool_frames=frames, num_goalset=1)
        # (batch, dof). A bare (dof,) raises inside the seed IK solver.
        seed = torch.as_tensor(seed_values[None], dtype=torch.float32, device="cuda")
        state = JointState.from_position(seed)

        with reading():
            if criteria is not None:
                planner.update_tool_pose_criteria(dict.fromkeys(frames, criteria))
            try:
                result = planner.plan_pose(goal_tool, state, max_attempts=self.max_attempts)
            except Exception:
                logger.exception("MotionBackendCuRobo: plan_pose raised")
                return None
            finally:
                if criteria is not None:
                    # Restore, always. The criteria are planner *state*, so a
                    # leaked constraint silently narrows every later plan --
                    # which is why cuRobo's own plan_grasp restores them too.
                    planner.update_tool_pose_criteria(dict.fromkeys(frames, ToolPoseCriteria()))

            if result is None or not bool(result.success.any()):
                free_space_reachable = None
                if result is None and world_config:
                    # Diagnostic only: never return or execute this unchecked
                    # trajectory. It distinguishes world-collision rejection
                    # from an unreachable goal/attachment model.
                    planner.clear_scene_cache()
                    free_result = planner.plan_pose(goal_tool, state, max_attempts=1)
                    free_space_reachable = bool(
                        free_result is not None and free_result.success is not None and free_result.success.any()
                    )
                logger.warning(
                    "MotionBackendCuRobo: plan rejected (status=%s, feasible=%s, "
                    "position_error=%s, rotation_error=%s, debug=%s, "
                    "free_space_reachable=%s)",
                    getattr(result, "status", None),
                    getattr(result, "feasible", None),
                    getattr(result, "position_error", None),
                    getattr(result, "rotation_error", None),
                    getattr(result, "debug_info", None),
                    free_space_reachable,
                )
                return None
            rows = self._rows(
                planner,
                result,
                num_waypoints,
                self._stream_hz() * self.time_scale,
            )
        if len(rows) and not self._endpoint_matches(rows[-1], position, quat_xyzw, arm_id):
            return None
        return _trajectory(rows)

    def build_world_tsdf(
        self,
        cameras: list[dict],
        exclude_masks: list | None = None,
        voxel_size: float = 0.01,
        clear_cylinders: list[dict] | None = None,
        extent: tuple[float, float, float] = (1.28, 1.28, 1.28),
        center: tuple[float, float, float] = (0.1, 0.0, 0.9),
        depth_max: float = 3.0,
        arm_id: int = -1,
    ) -> dict:
        """A collision world as an ESDF voxel grid, integrated from RGB-D.

        cuRobo 0.8's own block-sparse TSDF mapper
        (:class:`curobo.perception.Mapper`) replaces the alpha-shape mesh
        reconstruction path: depth in, ``compute_esdf`` out, all on device.
        Measured against ``geometry.build_world_config`` on this suite's
        overhead camera, that CPU path costs 42-44 s per episode; the
        integration here is GPU kernels end to end.

        The grid is built directly in the **robot base frame** -- the frame
        every obstacle handed to :meth:`_update_world` must end up in -- by
        transforming each camera's pose before integration rather than
        transforming the grid after.

        Pixels under any ``exclude_masks`` entry (the grasp target, the robot's
        own silhouette) are zeroed in the depth image, which is the TSDF
        equivalent of the mesh path's sphere-margin carving: an invalid pixel
        integrates nothing. ``clear_cylinders`` (``{x, y, radius, z_lo, z_hi}``
        in world coordinates) are cleared on the finished ESDF -- the aperture
        corridor, which RGB-D bridging would otherwise seal exactly as the
        alpha shape did.

        The finished :class:`VoxelGrid` is **kept on the backend** and picked
        up by :meth:`_update_world` when a ``world_config`` says
        ``{"use_tsdf": true}``; the return value is summary metadata, so the
        tool stays JSON-traceable without round-tripping a multi-megabyte
        field through the bus.
        """
        import torch
        from curobo.perception import Mapper, MapperCfg
        from curobo.types import CameraObservation, Pose

        rotation, translation = self._base(arm_id)
        device = "cuda:0"
        center_base = (np.asarray(center, dtype=np.float64) - translation) @ rotation
        mapper = Mapper(
            MapperCfg(
                voxel_size=float(voxel_size),
                esdf_voxel_size=float(voxel_size),
                extent_meters_xyz=tuple(float(v) for v in extent),
                grid_center=torch.tensor(center_base, dtype=torch.float32, device=device),
                depth_maximum_distance=float(depth_max),
                num_cameras=1,
                device=device,
            )
        )
        masks = [np.asarray(m) > 0 for m in (exclude_masks or []) if m is not None]
        robot_spheres = np.asarray(self._robot_spheres_world(), dtype=np.float64).reshape(-1, 4)
        integrated = 0
        for camera in cameras:
            depth = np.asarray(camera["depth"], dtype=np.float32).copy()
            for mask in masks:
                if mask.shape == depth.shape:
                    depth[mask] = 0.0
            # The voxel-project integrator reshapes the RGB unconditionally --
            # colour is part of its kernel contract, not an option.
            rgb = np.ascontiguousarray(np.asarray(camera["rgb"], dtype=np.uint8)[..., :3])
            pose = camera["pose"]
            cam_position = np.array([pose["position"][k] for k in ("x", "y", "z")], dtype=np.float64)
            q = pose["rotation"]
            cam_rotation = quat_to_mat(np.array([q["x"], q["y"], q["z"], q["w"]]))
            # The robot's own body, by proprioception rather than by a
            # segmenter: any pixel whose depth point lies inside the arm's
            # sphere model integrates nothing. A text-prompted mask covers
            # the links it was asked for and not the rest -- on this suite's
            # overhead camera "robot arm" + "robot gripper" left the shoulder
            # and upper arm unmasked (~3000 pixels at 1.0-1.3 m), and those
            # became a mid-air obstacle block over the bench that every plan
            # then bowed around and the straight-line legs refused through.
            if len(robot_spheres):
                removed = self._pixels_on_robot(depth, camera["intrinsics"], cam_rotation, cam_position, robot_spheres)
                depth[removed] = 0.0
            base_position = (cam_position - translation) @ rotation
            base_rotation = rotation.T @ cam_rotation
            quat_xyzw = mat_to_quat(base_rotation)
            observation = CameraObservation(
                rgb_image=torch.as_tensor(rgb, device=device)[None],
                depth_image=torch.as_tensor(depth, device=device)[None],
                intrinsics=torch.as_tensor(np.asarray(camera["intrinsics"], dtype=np.float32), device=device)[None],
                pose=Pose(
                    position=torch.tensor(base_position, dtype=torch.float32, device=device)[None],
                    quaternion=torch.tensor(
                        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                        dtype=torch.float32,
                        device=device,
                    )[None],
                ),
                depth_to_meter=1.0,
            )
            mapper.integrate(observation)
            integrated += 1

        grid = mapper.compute_esdf()
        cleared = 0
        if clear_cylinders and grid.feature_tensor is not None:
            # The feature tensor is (nx, ny, nz), indexed so that
            # world = grid_center + (i - (n-1)/2) * voxel per axis, and holds
            # distance-to-surface: 0 at occupied voxels, growing into free
            # space -- both measured empirically against a synthetic floor,
            # since the cfg docstring's (nz, ny, nx) claim is not what the
            # tensor does. Clearing is therefore writing the tensor's own
            # far-free value into the corridor.
            feature = grid.feature_tensor
            free_value = float(feature.max().item())
            axes = []
            for axis in range(3):
                n = feature.shape[axis]
                axes.append(
                    float(center_base[axis])
                    + (torch.arange(n, dtype=torch.float32, device=feature.device) - (n - 1) / 2.0) * float(voxel_size)
                )
            for cylinder in clear_cylinders:
                cyl_center = (np.array([cylinder["x"], cylinder["y"], 0.0], dtype=np.float64) - translation) @ rotation
                z_lo = float(cylinder.get("z_lo", -10.0)) - float(translation[2])
                z_hi = float(cylinder.get("z_hi", 10.0)) - float(translation[2])
                dx = (axes[0] - float(cyl_center[0]))[:, None, None]
                dy = (axes[1] - float(cyl_center[1]))[None, :, None]
                dz = axes[2][None, None, :]
                inside = (dx * dx + dy * dy <= float(cylinder["radius"]) ** 2) & (dz >= z_lo) & (dz <= z_hi)
                cleared += int(inside.sum().item())
                feature[inside] = free_value
        self._tsdf_grid = grid
        # The frame the grid lives in. Carving must use the same one, and an
        # arm mounted elsewhere cannot plan against it unchanged.
        self._tsdf_base = (rotation, translation)
        # For the per-plan endpoint carving in _update_world: the corridor-
        # cleared state is the baseline every plan restores before carving its
        # own allowed-contact spheres, and these are the world coordinates of
        # each tensor axis, in the base frame.
        self._tsdf_pristine = grid.feature_tensor.clone() if grid.feature_tensor is not None else None
        self._tsdf_axes = None
        if grid.feature_tensor is not None:
            self._tsdf_axes = [
                float(center_base[axis])
                + (
                    torch.arange(
                        grid.feature_tensor.shape[axis], dtype=torch.float32, device=grid.feature_tensor.device
                    )
                    - (grid.feature_tensor.shape[axis] - 1) / 2.0
                )
                * float(voxel_size)
                for axis in range(3)
            ]
        stats = mapper.get_stats()
        return {
            "cameras_integrated": integrated,
            "voxel_size": float(voxel_size),
            "cleared_voxels": cleared,
            "mapper_stats": {k: v for k, v in stats.items() if np.isscalar(v)},
        }

    def _carve_tsdf_contact_spheres(self, grid: Any, spheres: np.ndarray | None) -> None:
        """Restore the corridor-cleared ESDF and carve this plan's endpoint spheres.

        The voxel twin of ``_filter_mesh_near_spheres`` over
        ``allowed_contact_spheres``: a plan whose start or goal deliberately
        touches the scene -- the held syringe over the aperture, the fingers at
        the tray -- carves the held object's spheres out of the field for this
        plan only, or cuRobo refuses the waypoint the graph explicitly asked to
        reach. Restore-then-carve, so one plan's permits never leak into the
        next: the baseline is the clone ``build_world_tsdf`` took after its own
        corridor clearing.
        """

        feature = grid.feature_tensor
        pristine = getattr(self, "_tsdf_pristine", None)
        axes = getattr(self, "_tsdf_axes", None)
        if feature is None or pristine is None or axes is None:
            return
        feature.copy_(pristine)
        if spheres is None or not len(spheres):
            return
        free_value = float(feature.max().item())
        rotation, translation = getattr(self, "_tsdf_base", None) or self._base(-1)
        carved = 0
        for sphere in np.asarray(spheres, dtype=np.float64).reshape(-1, 4):
            center = (sphere[:3] - translation) @ rotation
            # An ESDF fattens surfaces by about a voxel; the carve allows for it.
            radius = float(sphere[3]) + 0.015
            dx = (axes[0] - float(center[0]))[:, None, None]
            dy = (axes[1] - float(center[1]))[None, :, None]
            dz = (axes[2] - float(center[2]))[None, None, :]
            inside = dx * dx + dy * dy + dz * dz <= radius * radius
            carved += int(inside.sum().item())
            feature[inside] = free_value
        if carved:
            logger.info("MotionBackendCuRobo: carved %d ESDF voxels for endpoint contact", carved)

    def _update_world(
        self,
        planner: Any,
        world_config: dict | None,
        arm_id: int,
        allowed_contact_spheres: np.ndarray | None = None,
        exclude_contacting_meshes: bool = False,
    ) -> list[Any]:
        """Load a planner-agnostic RGB-D mesh world into cuRobo v0.8.

        ``geometry.build_world_config`` deliberately returns JSON-compatible
        dictionaries. Keeping conversion here makes ``motion.*`` usable both
        in-process and over RPC, and avoids leaking cuRobo classes into GAP.
        """
        meshes = list((world_config or {}).get("meshes") or [])
        planner.clear_scene_cache()

        from curobo._src.geom.types import Mesh, SceneCfg

        if (world_config or {}).get("use_tsdf"):
            # The grid the last build_world_tsdf call left on this backend --
            # already in the base frame, already corridor-cleared. Meshes may
            # ride alongside it, so this falls through to the mesh loop below
            # and the SceneCfg at the end carries both.
            grid = getattr(self, "_tsdf_grid", None)
            if grid is None:
                raise ValueError(
                    "world_config asks for use_tsdf but no TSDF grid has been built; call motion.build_world_tsdf first"
                )
            grid_base = getattr(self, "_tsdf_base", None)
            arm_base = self._base(arm_id)
            if grid_base is not None and not (
                np.allclose(grid_base[0], arm_base[0], atol=1e-6) and np.allclose(grid_base[1], arm_base[1], atol=1e-6)
            ):
                # The grid is in the base frame of the arm it was built for; on a
                # robot whose arms are mounted apart, another arm would read every
                # obstacle shifted by the mount offset.
                raise ValueError(
                    f"the TSDF grid was built in another arm's base frame than arm {arm_id}'s; "
                    f"call motion.build_world_tsdf with arm_id={arm_id} first"
                )
            self._carve_tsdf_contact_spheres(grid, allowed_contact_spheres)
            if not meshes:
                planner.update_world(SceneCfg(voxel=[grid]))
                logger.info("MotionBackendCuRobo: loaded TSDF voxel world")
                return []
        elif not meshes:
            return []

        rotation, translation = self._base(arm_id)
        robot_spheres = self._robot_spheres_world()
        converted = []
        total_removed = 0
        for index, item in enumerate(meshes):
            raw_vertices = item.get("vertices")
            raw_faces = item.get("faces")
            vertices = np.asarray([] if raw_vertices is None else raw_vertices, dtype=np.float64).reshape(-1, 3)
            faces = np.asarray([] if raw_faces is None else raw_faces, dtype=np.int32).reshape(-1, 3)
            if not len(vertices) or not len(faces):
                continue
            pose = item.get("pose") or {}
            pose_position, pose_quat = _se3_to_arrays(
                {
                    "position": pose.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                    "rotation": pose.get("rotation", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
                }
            )
            world_vertices = vertices @ quat_to_mat(pose_quat).T + pose_position
            world_vertices, faces, removed = _filter_mesh_near_spheres(
                world_vertices,
                faces,
                robot_spheres,
                ROBOT_SCENE_FILTER_MARGIN_M,
            )
            total_removed += removed
            if allowed_contact_spheres is not None and len(allowed_contact_spheres):
                world_vertices, faces, contact_removed = _filter_mesh_near_spheres(
                    world_vertices,
                    faces,
                    allowed_contact_spheres,
                    0.0,
                )
                total_removed += contact_removed
                if exclude_contacting_meshes and contact_removed:
                    logger.info(
                        "MotionBackendCuRobo: endpoint contact permits held object to disengage from mesh %s",
                        item.get("name") or f"mesh_{index}",
                    )
                    continue
            if not len(world_vertices) or not len(faces):
                logger.debug(
                    "MotionBackendCuRobo: removed scene mesh %s as robot return",
                    item.get("name") or f"mesh_{index}",
                )
                continue
            base_vertices = (world_vertices - translation) @ rotation
            converted.append(
                Mesh(
                    name=str(item.get("name") or f"mesh_{index}"),
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    vertices=base_vertices.astype(np.float32).tolist(),
                    faces=faces.tolist(),
                    device_cfg=planner.config.device_cfg,
                )
            )
        grid = getattr(self, "_tsdf_grid", None) if (world_config or {}).get("use_tsdf") else None
        if converted or grid is not None:
            # The cache was cleared above, so a TSDF world whose every mesh was
            # filtered out still has to be loaded, or the plan runs in free space.
            planner.update_world(SceneCfg(mesh=converted or None, voxel=[grid] if grid is not None else None))
        logger.info(
            "MotionBackendCuRobo: loaded %d/%d scene meshes after removing %d "
            "robot-overlapping triangles using %d live spheres",
            len(converted),
            len(meshes),
            total_removed,
            len(robot_spheres),
        )
        return converted

    def _endpoint_contact_spheres(
        self,
        attached_object: dict | None,
        seed_joints: np.ndarray,
        goal_position: np.ndarray,
        goal_quat_xyzw: np.ndarray,
        arm_id: int,
        *,
        allow_start_contact: bool,
        allow_goal_contact: bool,
        margin: float,
    ) -> np.ndarray:
        """Held-object spheres at explicitly allowed trajectory endpoints.

        Attachment spheres are expressed in the TCP frame.  The start TCP is
        obtained from live robot proprioception and FK; the goal TCP is the
        requested planner target.  Carving these bounded volumes permits
        support contact at t=0 and fixture contact at t=1 without disabling
        collision checking through the swept path.
        """
        spheres = list((attached_object or {}).get("spheres") or [])
        if not spheres or not (allow_start_contact or allow_goal_contact):
            return np.empty((0, 4), dtype=np.float64)

        transforms = []
        if allow_start_contact:
            name = arm_name(self.env, arm_id)
            handle = self.env.handles.arms[name]
            full = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64).copy()
            full[list(handle.arm_coords)] = np.asarray(seed_joints, dtype=np.float64)
            start_position, start_quat = self._fk_tcp(self.env.arm(name), full)
            transforms.append(_homogeneous(start_position, start_quat))
        if allow_goal_contact:
            transforms.append(_homogeneous(goal_position, goal_quat_xyzw))

        extra = max(float(margin), 0.0)
        out = []
        for transform in transforms:
            for sphere in spheres:
                center = sphere.get("center") or {}
                local = np.array(
                    [
                        float(center.get("x", 0.0)),
                        float(center.get("y", 0.0)),
                        float(center.get("z", 0.0)),
                        1.0,
                    ]
                )
                world = (transform @ local)[:3]
                radius = float(sphere.get("radius", 0.0)) + extra
                if radius > 0.0:
                    out.append([float(world[0]), float(world[1]), float(world[2]), radius])
        return np.asarray(out, dtype=np.float64).reshape(-1, 4)

    def _robot_spheres_world(self) -> np.ndarray:
        """Current collision spheres for every arm, in the shared world frame.

        RGB-D sees the robot, but the planner must not treat those returns as
        external obstacles.  Using the same CuRobo sphere model for exclusion
        and collision checking keeps this embodiment-independent and avoids a
        second, hand-maintained YAM geometry approximation.
        """
        import torch  # only on the planning path

        _, _, _, _, JointState, _ = self._import()
        model = self.planner
        spheres_world: list[np.ndarray] = []
        for other_arm_id in range(len(self.env.handles.arms)):
            joints = torch.as_tensor(
                self._seed(other_arm_id, None)[None],
                dtype=torch.float32,
                device="cuda",
            )
            with reading():
                state = model.compute_kinematics(JointState.from_position(joints))
            spheres = state.get_link_spheres().detach().cpu().numpy().astype(np.float64).reshape(-1, 4)
            spheres = spheres[spheres[:, 3] > 0.0]
            if not len(spheres):
                continue
            base_rotation, base_translation = self._base(other_arm_id)
            spheres[:, :3] = spheres[:, :3] @ base_rotation.T + base_translation
            spheres_world.append(spheres)
        return np.concatenate(spheres_world, axis=0) if spheres_world else np.empty((0, 4), dtype=np.float64)

    @staticmethod
    def _pixels_on_robot(
        depth: np.ndarray,
        intrinsics: Any,
        cam_rotation: np.ndarray,
        cam_position: np.ndarray,
        spheres: np.ndarray,
        margin: float = ROBOT_SELF_MARGIN_M,
    ) -> np.ndarray:
        """Pixels whose back-projected depth point lies inside the robot's sphere model.

        Same pinhole convention the mapper integrates with (z-depth along the
        camera's +z, x right, y down), so a pixel removed here is the pixel
        the mapper would otherwise have placed on the arm.
        """
        K = np.asarray(intrinsics, dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        height, width = depth.shape
        vs, us = np.mgrid[0:height, 0:width]
        z = depth.astype(np.float64)
        points = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1).reshape(-1, 3)
        world = points @ cam_rotation.T + cam_position
        inside = np.zeros(len(world), dtype=bool)
        for row in spheres:
            reach = float(row[3]) + margin
            near = np.all(np.abs(world - row[:3]) <= reach, axis=1)  # cheap box before the exact test
            if near.any():
                idx = np.flatnonzero(near)
                d2 = ((world[idx] - row[:3]) ** 2).sum(axis=1)
                inside[idx[d2 <= reach * reach]] = True
        return inside.reshape(height, width) & (depth > 0.0)

    def robot_collision_spheres(self, arm_id: int = -1) -> list[dict]:
        """Expose live planner geometry for synchronized RGB-D self-removal.

        This is robot proprioception, not scene ground truth. ``arm_id=-1``
        returns every arm so a multi-camera reconstruction can remove all
        visible robot bodies before meshing.
        """
        spheres = self._robot_spheres_world()
        return [
            {
                "center": {"x": float(row[0]), "y": float(row[1]), "z": float(row[2])},
                "radius": float(row[3]),
            }
            for row in spheres
        ]

    def _endpoint_matches(
        self,
        arm_joints: np.ndarray,
        target_position: np.ndarray,
        target_quat_xyzw: np.ndarray,
        arm_id: int,
        tolerance: float | None = None,
    ) -> bool:
        """Reject a CuRobo path whose final joints disagree with simulator FK.

        ``tolerance`` widens (or narrows) the position gate to what the caller
        asked for [m]; the rotation gate stays the planner's own.
        """
        name = arm_name(self.env, arm_id)
        handle = self.env.handles.arms[name]
        full = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64).copy()
        full[list(handle.arm_coords)] = np.asarray(arm_joints, dtype=np.float64)
        sim_position, sim_quat = self._fk_tcp(self.env.arm(name), full)
        position_mm = float(np.linalg.norm(sim_position - target_position)) * 1000.0
        dot = float(
            abs(np.dot(sim_quat / np.linalg.norm(sim_quat), target_quat_xyzw / np.linalg.norm(target_quat_xyzw)))
        )
        rotation_deg = float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))
        limit_mm = ENDPOINT_PARITY_MM if tolerance is None else float(tolerance) * 1000.0
        if position_mm > limit_mm or rotation_deg > ENDPOINT_PARITY_DEG:
            logger.error(
                "MotionBackendCuRobo: rejecting %s arm plan: simulator FK endpoint "
                "misses requested TCP by %.1f mm / %.1f deg (limits %.1f mm / %.1f deg)",
                name,
                position_mm,
                rotation_deg,
                limit_mm,
                ENDPOINT_PARITY_DEG,
            )
            return False
        return True

    def _stream_hz(self) -> float:
        """The rate the connector will stream this plan at [Hz].

        The cell's own ``control_hz``, the way every suite planner in ``tasks/``
        reads it -- a densification rate is the cell's and never a module
        constant, and a plan resampled
        here is a densification. Zero when there is no cell to ask, which leaves
        the plan at cuRobo's own rate rather than at a guessed one.
        """
        physics = getattr(getattr(self.env, "task", None), "physics", None)
        return float(getattr(physics, "control_hz", 0.0) or 0.0)

    @staticmethod
    def _rows(planner: Any, result: Any, num_waypoints: int, stream_hz: float = 0.0) -> np.ndarray:
        """The solution as joint rows: the plan's own length, at the stream rate.

        cuRobo interpolates into a **fixed-size buffer** and repeats the goal to
        fill it, so the array's length says nothing about the plan. Measured on a
        20 cm Franka move: ``interpolated_trajectory`` is 5000 rows,
        ``interpolated_last_tstep`` is **41**, and only 36 of those 4999 steps
        move at all. Handing the buffer to the connector -- which streams one
        waypoint per control step, each a ``bus.call`` -- is 5000 control steps
        for a one-second motion. ``tool_sort``'s ceiling is 2400, so the episode
        truncates inside the first transport with the arm parked on its goal.

        So the buffer is cut at ``interpolated_last_tstep`` first, and *then*
        resampled -- because after the cut the count is the plan's duration at
        cuRobo's own ``dt``, and streaming that at the cell's rate is the
        stream-rate defect, in a fourth place. At 41
        waypoints of 0.025 s streamed at 60 Hz the move runs 1.46x fast.

        Args:
            num_waypoints: An explicit count from the caller, which wins.
            stream_hz: The rate the waypoints will be streamed at. With neither
                this nor ``num_waypoints``, the plan is returned at cuRobo's own
                interpolation rate -- honest, and slow or fast by whatever the
                two rates differ by.
        """
        trajectory = result.interpolated_trajectory
        joints = planner.kinematics.get_active_js(trajectory)
        rows = np.asarray(joints.position.detach().cpu().numpy(), dtype=np.float64)
        rows = rows.reshape(-1, rows.shape[-1])

        last = _as_int(getattr(result, "interpolated_last_tstep", None))
        if last is not None and 0 < last < len(rows):
            rows = rows[: last + 1]  # inclusive: the last step is the arrival

        wanted = int(num_waypoints)
        if wanted <= 0 and stream_hz > 0.0:
            dt = _as_float(getattr(trajectory, "dt", None))
            if dt and dt > 0.0:
                wanted = max(2, int(round(len(rows) * dt * float(stream_hz))))
        if wanted > 0 and wanted != len(rows) and len(rows) > 1:
            # Interpolated rather than index-resampled, because this now runs in
            # both directions. Thinning by nearest index is fine; *lengthening*
            # by it repeats waypoints, and a repeated waypoint is a stationary
            # control step -- which slows the move, correctly, while making the
            # path a stair rather than a line. Along the path parameter the two
            # agree exactly wherever a sample lands on a row.
            source = np.linspace(0.0, 1.0, len(rows))
            target = np.linspace(0.0, 1.0, wanted)
            rows = np.stack([np.interp(target, source, rows[:, j]) for j in range(rows.shape[1])], axis=1)
        return rows
