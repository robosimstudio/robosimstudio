# SPDX-License-Identifier: Apache-2.0
"""The observation dict the pinned runtime reads, from a published snapshot.

The runtime's ``Connector`` expects ``robot_joint_pos_<i>`` (this arm's joint
coordinates with the gripper's *open* fraction appended) and
``robot_cartesian_pos_<i>`` (tool-centre position plus a **wxyz** quaternion),
plus one block per camera and the episode's instruction.

Extracted from the single-world facade so both paths format the same dict from
the same code. The two conversions this contains are the ones that hurt when
they drift apart: the runtime is scalar-first where Newton is xyzw, and the
runtime's gripper 1.0 is fully *open* where this stack's command is 1.0 closed.
A batched run that got either backwards would produce a robot that opens to
grasp, in five hundred worlds at once, and read as a planning failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from robosimstudio.arms import arm_order
from robosimstudio.core.math import quat_xyzw_to_wxyz
from robosimstudio.solvers.ik.tool_frame import tcp_pose, tool_offsets

__all__ = ["RENDER", "camera_blocks", "format_runtime_obs", "world_observation", "world_observations"]


class _HostRows:
    """A host array wearing the ``.numpy()`` a measurement expects of a device one."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Any) -> None:
        self._rows = np.asarray(rows)

    def numpy(self) -> np.ndarray:
        return self._rows


class _CoordsState:
    """Just enough of a ``newton.State`` for :meth:`Gripper.closed_fraction`: one
    world's joint coordinates, indexed like the template's."""

    __slots__ = ("joint_q",)

    def __init__(self, coords: Any) -> None:
        self.joint_q = _HostRows(coords)


RENDER = "render"
"""The rendezvous group a fresh picture is asked for through.

Named here rather than in the driver because this is the module that knows what
a frame is for. A node calling ``get_observation`` is asking to *look now*, and
the single-world facade answers by rendering on the spot; on a fleet the same
call joins this group, so every world that is looking is photographed in one
pass of the rig rather than in one pass each.
"""


def world_observations(fleet, snapshot) -> list[dict[str, Any]]:
    """Every world's raw observation, as :func:`world_observation` builds each.

    The grippers' commands are read once for the fleet and handed in: the bank
    answers with a copy of all W of them, so reading it per world made one
    publish O(W^2) -- 8k copies of a 16 KB array per tick at 4096 worlds.
    """
    commands = {name: bank.commands() for name, bank in fleet.grippers.items()}
    return [world_observation(fleet, world, snapshot, commands=commands) for world in range(fleet.num_worlds)]


def world_observation(fleet, world: int, snapshot, *, commands: dict[str, Any] | None = None) -> dict[str, Any]:
    """One world's raw observation, in the keys the formatter expects.

    Built by the fleet's own reader rather than by the runtime's, so what counts
    as an observation stays a property of the scene and not of the graph.
    *commands* is each gripper bank's ``commands()``, read once by a caller
    observing every world (:func:`world_observations`); ``None`` reads the bank.
    """
    embodiment = fleet.scene.embodiment
    obs: dict[str, Any] = {"instruction": ""}
    episode = fleet.episodes[world] if world < len(fleet.episodes) else None
    if episode is not None:
        obs["instruction"] = episode.instruction
    if embodiment is None:
        return obs

    coords = snapshot.coords(world)
    body_q = snapshot.body(world)
    # This world's rows, wearing the two attributes the embodiment's own readers
    # ask a state for. The single-world observation is ``embodiment.observe(state)``;
    # the same arithmetic runs here on a world's slice, so the TOOL CENTRE and
    # the MEASURED closure come out, not the flange and the command. Both were
    # wrong before: ``ee_pose`` was the end-effector body (165 mm from the tool
    # centre on a Panda with the 2F-85, so every height a graph built on it was
    # off by a hand), and ``gripper`` was the bank's command, which reads 1.0
    # closed whether the fingers stalled on a gear or shut on air -- the one
    # signal a grasp check has, erased.
    proprio: dict[str, Any] = {}
    # Role order -- work arm first -- which is the order a runtime ``arm_id``
    # counts and the order the fleet's *commands* already use. The single-world
    # facade looks arms up by name in this order; the formatter below recovers
    # names from this dict's order, so declaration order here would hand
    # ``arm_id`` 0 the idle hand on any robot whose work arm is declared second.
    try:
        names = list(arm_order(fleet))
    except Exception:  # a scene with no pairing: declaration order is all there is
        names = list(embodiment.arms)
    names += [name for name in embodiment.arms if name not in names]
    for name in names:
        index = embodiment.arms[name]
        proprio[f"{name}/joint_q"] = np.asarray(coords[index.arm_coords], dtype=np.float64)
        # The tool-centre point in the grasp frame, which is what the robot's
        # own observation publishes under this key (``iks[name].ee_pose``) and
        # what every commanded pose is stated in. The raw ``ee_body`` row used
        # to go here, and on the i2rt YAM that is 134.7 mm and a half-turn away
        # from the hand the graph is steering -- see :mod:`robosimstudio.solvers.ik.tool_frame`.
        position, quat = tcp_pose(body_q, index.ee_body, *tool_offsets(fleet, name))
        proprio[f"{name}/ee_pose"] = np.concatenate([position, quat]).astype(np.float64)
        bank = fleet.grippers.get(name)
        if bank is None:
            command = 0.0
        else:
            command = float((commands[name] if commands is not None else bank.commands())[world])
        # The **measured** closure, as the single-world observation publishes
        # it, and the command beside it. The command alone used to stand in
        # for both, and the difference between them is the only signal a hand
        # has that it is holding something: a close on a sponge stalls short of
        # its command, a close on air reaches it. Reported as the command, the
        # two are the same number, and a graph testing "did the jaws stop
        # short?" concluded empty in every world, on every grasp.
        gripper = getattr(embodiment, "grippers", {}).get(name)
        measured = float(gripper.closed_fraction(_CoordsState(coords))) if gripper is not None else command
        proprio[f"{name}/gripper"] = measured
        proprio[f"{name}/gripper_command"] = command
    obs["proprio"] = proprio
    return obs


class _Array:
    __slots__ = ("_values",)

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)

    def numpy(self) -> np.ndarray:
        return self._values


class _WorldStateView:
    """One world's ``joint_q`` and ``body_q`` behind a state's two attribute names."""

    __slots__ = ("body_q", "joint_q")

    def __init__(self, coords: np.ndarray, body_q: np.ndarray) -> None:
        self.joint_q = _Array(coords)
        self.body_q = _Array(body_q)


def format_runtime_obs(
    snapshot,
    *,
    arm_count: int,
    frames: dict[str, Any] | None = None,
    cameras: Sequence[Any] = (),
) -> dict[str, Any]:
    """The runtime's observation dict for one world.

    Args:
        snapshot: That world's published snapshot, whose ``obs`` carries the
            scene's own observation under ``proprio`` and whose ``body_q`` is
            this world's rows, indexed like the template's.
        arm_count: How many arms the runtime should see, in ``arm_id`` order.
        frames: Camera frames to use instead of the snapshot's, for a caller
            that has just asked the rig for a fresh look. ``None`` takes what
            was published, which is empty unless somebody rendered.
        cameras: The rig's specs, so a frame can be handed over with the
            calibration that makes it usable. **A picture without them is not
            an observation**: a perception node locates something in pixels and
            then needs a camera matrix and a camera pose to say where that is in
            the world, so a block carrying only ``rgb`` and ``depth`` fails at
            exactly the point where the expensive models have already run. Left
            empty this formats the picture alone, which is what a caller with no
            rig means.
    """
    obs = snapshot.obs or {}
    proprio = obs.get("proprio") or {}
    names = [name.split("/")[0] for name in proprio if name.endswith("/joint_q")]
    out: dict[str, Any] = {}
    for arm_id in range(arm_count):
        name = names[arm_id] if arm_id < len(names) else ""
        joints = np.asarray(proprio.get(f"{name}/joint_q", np.zeros(0)), dtype=np.float64)
        grip_closed = float(proprio.get(f"{name}/gripper", 0.0))
        # The runtime's fraction is 1.0 open; the gripper's command is 1.0 closed.
        out[f"robot_joint_pos_{arm_id}"] = np.append(joints, 1.0 - grip_closed)
        pose = np.asarray(proprio.get(f"{name}/ee_pose", np.zeros(7)), dtype=np.float64)
        out[f"robot_cartesian_pos_{arm_id}"] = np.concatenate([pose[:3], quat_xyzw_to_wxyz(pose[3:7])])

    # This world's own parent body, because the snapshot's rows are this
    # world's rebased to template indices -- so an eye-in-hand camera answers
    # with this world's wrist and not world zero's. The camera specs carry
    # template parent bodies for exactly this reason.
    out.update(
        camera_blocks(
            cameras,
            (snapshot.frames if frames is None else frames) or {},
            getattr(snapshot, "body_q", None),
        )
    )
    out["instruction"] = obs.get("instruction", "")
    return out


def camera_blocks(cameras, frames: dict[str, dict[str, Any]], body_q: np.ndarray | None) -> dict[str, Any]:
    """One world's camera blocks, keyed by camera name, as the runtime reads them.

    The same block the single-world facade builds (``GapEnvFacade._observation``):
    ``images.rgb`` and, when rendered, ``images.depth`` *and* ``depth_data`` --
    the runtime's own camera builder reads the first, the wire convention the
    bridge speaks is the second, and setting only one is silent -- plus the
    spec's intrinsics and its pose in world (position, then a wxyz rotation),
    posed off this world's body rows for a camera that rides one.
    """
    from robosimstudio.core.math import quat_xyzw_to_wxyz, xform_from_matrix

    out: dict[str, Any] = {}
    for camera_name, frame in (frames or {}).items():
        spec = next((c for c in cameras if c.name == camera_name), None)
        block: dict[str, Any] = {"images": {"rgb": frame.get("rgb")}}
        if "depth" in frame:
            block["images"]["depth"] = frame["depth"]
            block["depth_data"] = frame["depth"]
        if spec is not None:
            block["intrinsics"] = spec.intrinsics
            xform = xform_from_matrix(spec.pose_on_body(body_q))
            block["pose"] = np.concatenate([xform[:3], quat_xyzw_to_wxyz(xform[3:7])])
        out[camera_name] = block
    return out
