# SPDX-License-Identifier: Apache-2.0
"""The IK backend the graph runtime drives: the arm's solver behind the GaP contract.

The **only** place the runtime's vocabulary -- ``Se3Pose`` dicts, ``arm_id``,
``Trajectory`` waypoints, ``honours_roll``, the bus -- meets a solver. Below it
sits :class:`~robosimstudio.solvers.ik.solver.IKSolver`, numpy in and numpy
out; above it the ``robot.*``/``motion.*`` tools. Three things happen here and
nowhere else:

* **Frames.** A ``tcp_offset`` handed by the runtime is accepted and ignored,
  because the solver already owns it -- :class:`ArmKinematics` carries the
  tool centre as ``ee_offset``, so the position it is handed *is* the tool
  centre. Applying the runtime's copy on top would be the same offset twice,
  which was measured at 0.00 mm on the Panda (offset zero) and 134.70 mm on the
  i2rt YAM.
* **The roll contract.** ``roll="auto"`` reads the robot's own declaration
  (:attr:`EmbodimentSpec.honour_roll`, off everywhere but the UR5e); ``"free"``
  and ``"held"`` say so outright. Every shipped graph on every other embodiment
  was authored and measured against a free wrist: the one cell where a held
  roll was measured went 2/2 to 0/5, and a servo testing arrival on a roll
  nobody asked for burns its whole budget and reads as a slow move rather than
  a converged one. Full-pose authority is per call (``hold_rotation``) or per
  robot, never a side effect of the solver family.
* **The planner.** A collision-aware planner (cuRobo) sits beside the solver
  when asked for; ``plan_linear`` prefers it and falls back to the solved line
  unless the request carried world geometry or a contact allowance the fallback
  cannot honour; ``plan_to_pose``, ``build_world_tsdf``, ``solve_ik_batch`` and
  ``robot_collision_spheres`` are the planner's alone. ``honours_roll`` is the
  *solver's*, never the planner's: a servo that tested arrival on the planner's
  full-pose promise while the solver aligned an axis waited on a 4.3-degree
  roll error against a 3-degree gate, forever.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np

from robosimstudio.arms import arm_name
from robosimstudio.core.math import quat_to_mat
from robosimstudio.solvers.ik.arm import ArmKinematics
from robosimstudio.solvers.ik.solver import IKSolver, LineRefusal, approach_axes, plan_line

from .types import se3, se3_arrays

__all__ = [
    "ROLLS",
    "ROLL_OFFSET_LIMIT",
    "IKBackend",
    "build_solver",
    "declares_roll",
    "make_solver",
]

logger = logging.getLogger(__name__)

SOLVER = "pyroki"
"""The solver: PyRoKi over the arm's URDF (:mod:`robosimstudio.solvers.ik.pyroki`)."""

FALLBACK = "newton"
"""What serves an arm PyRoKi cannot -- no URDF, or one that drifts from the model:
damped least squares over the model's own kinematics (:mod:`robosimstudio.solvers.ik.dls`).
Not a choice anyone makes; the URDF decides."""

_SOLVERS: dict[str, str] = {
    SOLVER: "robosimstudio.solvers.ik.pyroki.IKSolverPyroki",
    FALLBACK: "robosimstudio.solvers.ik.dls.IKSolverDLS",
}

ROLLS: tuple[str, ...] = ("auto", "free", "held")
"""What ``roll`` may say."""

ROLL_OFFSET_LIMIT = 0.005
"""How far off its last joint's axis a tool centre may sit before the damped
least-squares solver's roll correction -- one joint, turned after the solve --
would translate it rather than turn it [m]. 5 mm: the i2rt YAM's 44 mm offset
sweeps 62 mm at a quarter turn. Only that family closes the roll that way; a
full-pose solver is not bound by it."""


def declares_roll(env: Any, *, family: str = "") -> bool:
    """Whether this scene's arm can be asked to hold a wrist roll under ``roll="auto"``.

    The *robot* says whether its graphs want the roll solved
    (:attr:`EmbodimentSpec.honour_roll`, read rather than inferred). For the
    damped-least-squares family the *hand* then has to allow it: that solver
    closes the roll on the last joint, so a tool centre off that joint's axis
    is translated by the correction, and a robot arriving in an off-axis hand
    keeps the free wrist and says so.
    """
    spec = getattr(getattr(getattr(env, "scene", None), "embodiment", None), "spec", None)
    if not bool(getattr(spec, "honour_roll", False)):
        return False
    if family != "newton":
        return True
    for name, handle in env.handles.arms.items():
        offset = float(np.linalg.norm(np.asarray(handle.ee_offset, dtype=np.float64)[:2]))
        if offset > ROLL_OFFSET_LIMIT:
            print(
                f"[gap] {spec.name} declares honour_roll but arm {name!r}'s tool centre sits "
                f"{offset * 1000:.1f} mm off its last joint's axis (limit {ROLL_OFFSET_LIMIT * 1000:.0f} mm); "
                "leaving the wrist free"
            )
            return False
    return True


def _arm_frame(env: Any, handle: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """One world's joint vector and the arm's base pose, for either kind of env.

    A serial environment holds ``.state``; a fleet keeps its ``W`` worlds on
    ``pipeline.state_0`` and carries no ``.state`` at all. Reading only the
    attribute left every fleet arm without a ``base_xform``, which is the one
    thing PyRoKi refuses to solve without
    (:mod:`robosimstudio.solvers.ik.pyroki`), so every wave fell back to damped
    least squares with the reason printed -- silently, as far as a score is
    concerned, on suites whose numbers depend on which family served the arm.
    Every world of a wave carries the same robot at the same mount, so world
    0's rows are the wave's, and the joint vector is sliced to one world
    because :meth:`~robosimstudio.solvers.ik.arm.ArmKinematics.full` indexes it
    by the arm's own coordinates.

    Args:
        env: The environment or fleet, as :func:`make_solver` takes it.
        handle: The arm's handle, for its ``base_body``.

    Returns:
        ``(template, base_xform)``, either of which is ``None`` when the
        environment cannot answer it.
    """
    state = getattr(env, "state", None)
    if state is None:
        state = getattr(getattr(env, "pipeline", None), "state_0", None)
    if state is None:
        return None, None
    joint_q = np.asarray(state.joint_q.numpy(), dtype=np.float64)
    coords = int(getattr(getattr(env, "worlds", None), "coords_per_world", 0) or 0)
    template = joint_q[:coords] if 0 < coords < joint_q.size else joint_q
    body = int(getattr(handle, "base_body", -1))
    base_xform = np.asarray(state.body_q.numpy()[body], dtype=np.float64) if body >= 0 else None
    return template, base_xform


def make_solver(env: Any, arm: str, family: str = SOLVER, *, fk: Any = None) -> IKSolver:
    """The solver over one of the scene's arms, read off the scene.

    Args:
        env: The environment (or a fleet: anything with ``scene``, ``handles`` and ``arm``).
        arm: The arm's name.
        family: :data:`SOLVER`, or :data:`FALLBACK` when :func:`build_solver` is
            falling back. Internal: nothing above this module names one.
        fk: A batched forward kinematics for the arm, when the caller has one (a wave's).

    Raises:
        ValueError: The solver cannot serve this arm -- no URDF, or one that
            does not describe it -- with the reason; the caller decides whether
            to fall back.
    """
    from robosimstudio_bench.taskbuild.artifacts import resolve_dotted

    path = _SOLVERS.get(family) or (family if "." in family else "")
    if not path:
        raise ValueError(f"unknown IK solver {family!r}")
    handle = env.handles.arms[arm]
    spec = getattr(getattr(getattr(env, "scene", None), "embodiment", None), "spec", None)
    template, base_xform = _arm_frame(env, handle)
    kinematics = ArmKinematics.of(
        env.arm(arm) if hasattr(env, "arm") else env.scene.embodiment.iks[arm],
        template=template,
        urdf=str(getattr(spec, "urdf", "") or ""),
        fk=fk,
        home_q=np.asarray(handle.home_q, dtype=np.float64) if getattr(handle, "home_q", None) else None,
        base_xform=base_xform,
    )
    return resolve_dotted(path)(kinematics)


def build_solver(
    env: Any, name: str, family: str = SOLVER, *, fallback: bool = True, fk: Any = None
) -> tuple[IKSolver, str]:
    """The solver over one arm, and the name of what actually serves it.

    PyRoKi when it can serve the arm; otherwise, with ``fallback``, the
    damped-least-squares solver with the reason printed. The one rule behind
    the single-world backend and a wave's shared solver: an embodiment that
    ships no URDF cannot be served by PyRoKi, and it has to plan the same joint
    ease under ``--num-worlds`` as it does alone.
    Before the wave's solver followed it, every ``motion.plan_joint`` a fleet
    was asked for on that robot raised where the serial connector answered --
    and ``tool_hanging/pliers_level1``, whose profile validates its symmetry
    candidates with that verb, went 0/25 on "no feasible orientation".

    Args:
        env: The environment (or a fleet), as :func:`make_solver` takes it.
        name: The arm's name.
        family: Internal; :data:`SOLVER` unless a test says otherwise.
        fallback: Whether an arm PyRoKi cannot serve falls back to
            :data:`FALLBACK`, rather than raising.
        fk: A batched forward kinematics for the arm, when the caller has one.

    Returns:
        ``(solver, family)`` -- the family in hand, which is what the caller
        adopts its roll contract from.
    """
    try:
        return make_solver(env, name, family, fk=fk), family
    except (ValueError, ImportError) as exc:
        if not fallback or family == FALLBACK:
            raise
        print(f"[gap] {family} cannot serve arm {name!r} ({exc}); solving it with {FALLBACK} instead")
        return make_solver(env, name, FALLBACK, fk=fk), FALLBACK


_CONTACT_KWARGS = frozenset({"allow_start_contact", "allow_goal_contact", "contact_margin"})


class IKBackend:
    """Solve, plan lines and eases for the graph runtime, over the arm's solver.

    Args:
        env: The toolkit-layer environment.
        bus: What marshals a solve onto the simulation's thread.
        roll: The roll contract: ``"auto"`` (the robot's declaration), ``"free"``, ``"held"``.
        planner: A collision-aware planner to prefer for ``plan_linear`` and to
            answer ``plan_to_pose`` and friends, or ``None``.
        tolerance: Position error above which a solve or a waypoint is out of reach [m].
        waypoints: How many solves a straight line is cut into.
        iterations: The solver's budget per solve, in its own units.
        fallback: Whether an arm PyRoKi cannot serve (no URDF, or one that
            drifts) falls back to the damped-least-squares solver with a printed
            reason, rather than raising.
        solver: A built :class:`IKSolver` for arm 0, for a test that brings its
            own; ``None`` builds from the scene.
    """

    rot_tolerance = np.radians(5.0)
    """Orientation error above which a roll-honouring solve is a failure [rad].

    Looser than the servo's own 3-degree arrival test is tight, or a solve
    good enough to arrive on gets thrown away as unreachable; measured across
    every heading of ``wiring_sort``'s four grasp points the residual is 0.93
    to 1.60 degrees."""

    def __init__(
        self,
        env: Any,
        bus: Any,
        *,
        roll: str = "auto",
        planner: Any = None,
        tolerance: float = 0.03,
        waypoints: int = 40,
        iterations: int = 80,
        fallback: bool = True,
        solver: IKSolver | None = None,
    ) -> None:
        self.env = env
        self.bus = bus
        self.planner = planner
        self.prefer_planned = True
        self.tolerance = float(tolerance)
        self.waypoints = int(waypoints)
        self.iterations = int(iterations)
        self.fallback = bool(fallback)
        self._solvers: dict[str, IKSolver] = {}
        if roll not in ROLLS:
            raise ValueError(f"roll must be one of {ROLLS}, not {roll!r}")
        self.family = SOLVER
        """What the backend solves with; :data:`FALLBACK` once an arm that PyRoKi cannot serve resolved."""
        if solver is not None:
            self.family = type(solver).__name__
            self._solvers[self._arm_name(0)] = solver
        self.roll = roll
        self.honours_roll = roll == "held" if roll != "auto" else declares_roll(env, family=self.family)
        """Whether solves and arrival tests hold the whole orientation. The
        servo reads this; a planner never sets it."""
        self.last_solve_error: tuple[float, float | None] | None = None
        self.last_position_error: float | None = None
        """How far the most recent plan_linear's last waypoint landed from the
        pose it was asked for [m], or None when unmeasured -- ``planned: true``
        alone is not the answer a caller needs, because acceptance is a gate at
        :attr:`tolerance`, 30 mm, wider than a hammer's handle."""
        self.last_refusal = ""
        """Why the most recent plan_linear returned None, for the caller to report."""
        self._own_refusal = ""

    # ---------------------------------------------------------------- solvers
    def _arm_name(self, arm_id: int) -> str:
        return arm_name(self.env, arm_id)

    def _arm(self, arm_id: int):
        return self.env.arm(self._arm_name(arm_id))

    def solver_for(self, arm_id: int = 0) -> IKSolver:
        """This arm's solver, built on first use from the scene."""
        name = self._arm_name(arm_id)
        if name not in self._solvers:
            self._solvers[name] = self._build_solver(name)
        return self._solvers[name]

    def _build_solver(self, name: str) -> IKSolver:
        solver, family = build_solver(self.env, name, self.family, fallback=self.fallback)
        self._adopt(family)
        return solver

    def _adopt(self, family: str) -> None:
        """Take on the roll contract of the family that actually serves the arm.

        Nothing changes when it is the family asked for. When the build fell
        back, the contract is re-read for the family in hand -- the
        damped-least-squares solver closes the roll on the last joint and an
        off-axis hand keeps the wrist free -- exactly as ``__init__`` read it
        for the family that could not serve.
        """
        if family != self.family:
            self.honours_roll = self.roll == "held" if self.roll != "auto" else declares_roll(self.env, family=family)

    def _seed(self, arm_id: int, seed_joints: Any | None) -> tuple[np.ndarray, np.ndarray]:
        """``(arm row [1, n], template)``: this arm's seed, and the whole model's pose around it."""
        ik = self._arm(arm_id)
        template = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)
        row = template[list(ik.arm_coords)].copy()
        if seed_joints is not None:
            values = np.asarray(seed_joints, dtype=np.float64).reshape(-1)
            row[: len(values)] = values[: len(row)]
        return row.reshape(1, -1), template

    def _contract(self, hold_rotation: bool = False) -> str:
        return "held" if (self.honours_roll or hold_rotation) else "free"

    def _bound(self, solver: IKSolver, template: np.ndarray) -> IKSolver:
        """The solver as one world's asks see it: a row in, a row out.

        A solver whose ``solve`` wants the whole model's pose (the DLS family)
        is bound to this template; the fleet's backend binds the wave's shared
        solver to its own world's row instead. Every solve and every chained
        line goes through here, so the two cannot drift apart.
        """
        if "template" in inspect.signature(solver.solve).parameters:
            return _Templated(solver, template)
        return solver

    def _solve_rows(
        self,
        solver: IKSolver,
        seed: np.ndarray,
        position: np.ndarray,
        quat: np.ndarray,
        roll: str,
        template: np.ndarray,
    ):
        return self._bound(solver, template).solve(
            seed, position.reshape(1, 3), quat.reshape(1, 4), roll=roll, iterations=self.iterations
        )

    # -------------------------------------------------------------------- flags
    @property
    def trajectory_needs_joint_reverse(self) -> bool:
        source = self.planner if (self.planner is not None and self.prefer_planned) else None
        return bool(getattr(source, "trajectory_needs_joint_reverse", False))

    @property
    def supports_world_aware_plan(self) -> bool:
        """Whether a plan here routes around obstacles or merely interpolates."""
        return self.planner is not None and self.prefer_planned

    @property
    def _planner(self) -> Any:
        return self.planner if (self.planner is not None and self.prefer_planned) else None

    # -------------------------------------------------------------------- API
    def solve_ik(
        self, target_world_pose: dict, *, arm_id: int = 0, seed_joints: Any | None = None, tcp_offset: Any | None = None
    ) -> list[float] | None:
        """Solve one TCP pose. Returns this arm's joints, or ``None`` if short."""
        del tcp_offset  # the solver owns the tool centre; see the module docstring
        return self.bus.call(self._solve, target_world_pose, arm_id, seed_joints)

    def _solve(self, pose: dict, arm_id: int, seed_joints: Any | None) -> list[float] | None:
        solver = self.solver_for(arm_id)
        seed, template = self._seed(arm_id, seed_joints)
        position, quat = se3_arrays(pose)
        roll = self._contract()
        q, err = self._solve_rows(solver, seed, position, quat, roll, template)
        distance, rotation = float(err[0, 0]), float(err[0, 1])
        self.last_solve_error = (distance, rotation if roll == "held" else None)
        if distance > self.tolerance or (roll == "held" and rotation > self.rot_tolerance):
            return None
        return [float(v) for v in q[0]]

    def current_pose(self, arm_id: int = 0) -> dict:
        """Where this arm's tool centre is now, as an ``Se3Pose``, from the same kinematics that solve."""

        def read() -> dict:
            position, quat = self._arm(arm_id).ee_pose(self.env.state)
            return se3(position, quat)

        return self.bus.call(read)

    def plan_joint(
        self,
        target_world_pose: dict,
        *,
        arm_id: int = 0,
        tcp_offset: Any | None = None,
        seed_joints: Any | None = None,
        num_waypoints: int = 0,
        hold_rotation: bool = False,
    ) -> dict:
        """One pose, reached as a **posture**: solve once, ease there in joint space.

        It goes where the arm can go, it reports rather than refuses (a via
        that solves 161 mm short is still a direction to swing through), and it
        eases (``3t^2 - 2t^3``, both ends). Nothing here is ever ``None``.
        """
        del tcp_offset
        return self.bus.call(self._plan_joint, target_world_pose, arm_id, seed_joints, num_waypoints, hold_rotation)

    def _plan_joint(self, pose, arm_id, seed_joints, num_waypoints, hold_rotation) -> dict:
        solver = self.solver_for(arm_id)
        seed, template = self._seed(arm_id, seed_joints)
        position, quat = se3_arrays(pose)
        roll = self._contract(hold_rotation)
        q, err = self._solve_rows(solver, seed, position, quat, roll, template)
        steps = max(1, int(num_waypoints) or self.waypoints)
        start, end = seed[0], q[0]
        waypoints = []
        for i in range(1, steps + 1):
            t = i / steps
            s = t * t * (3.0 - 2.0 * t)
            waypoints.append({"positions": [float(v) for v in (start + s * (end - start))]})
        return {
            "waypoints": waypoints,
            "position_error_m": float(err[0, 0]),
            "rotation_error_rad": float(err[0, 1]) if roll == "held" else 0.0,
            "joint_config": [float(v) for v in end],
        }

    def plan_linear(self, *args: Any, **kwargs: Any) -> dict | None:
        """A straight line: the planner's when there is one, else a solved line.

        Keywords: ``arm_id``, ``tcp_offset`` (ignored), ``seed_joints``,
        ``num_waypoints``, ``hold_rotation``, ``tolerance`` for the solved
        line; ``axis``, ``world_config``, ``attached_object`` and the contact
        allowances for the planner. Each member gets the keywords its own
        signature declares. A planner that finds no path falls back to the
        solved line unless the request carried world geometry or a contact
        allowance, which the line cannot honour -- a trajectory reported under
        a constraint nobody applied is worse than a refusal.
        """
        self._own_refusal = ""
        # Only the solved line measures its arrival; a planner's answer leaves
        # it unmeasured rather than reporting an earlier line's number as its own.
        self.last_position_error = None
        planner = self._planner
        if planner is None:
            return self._own_plan_linear(*args, **self._for(self._own_plan_linear, kwargs)[0])
        planner_kwargs, _ = self._for(planner.plan_linear, kwargs)
        planned = planner.plan_linear(*args, **planner_kwargs)
        if planned is not None:
            return planned
        if kwargs.get("world_config") is not None or kwargs.get("attached_object") is not None:
            logger.warning(
                "plan_linear: collision-aware planner found no path; refusing to discard the requested world/attachment"
            )
            self._own_refusal = (
                "the collision-aware planner found no path and the request carried "
                "world/attachment geometry the interpolated fallback cannot honour"
            )
            return None
        own_kwargs, dropped = self._for(self._own_plan_linear, kwargs)
        asked_contact = sorted(k for k in dropped if k in _CONTACT_KWARGS)
        if asked_contact:
            self._own_refusal = "the planner found no path and the interpolated fallback cannot honour " + ", ".join(
                asked_contact
            )
            return None
        logger.info("plan_linear: planner found nothing; falling back to the interpolated path")
        return self._own_plan_linear(*args, **own_kwargs)

    @staticmethod
    def _for(fn: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Whichever of ``kwargs`` this callable's signature declares, and the names it dropped."""
        params = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs), []
        kept = {k: v for k, v in kwargs.items() if k in params}
        return kept, [k for k in kwargs if k not in params]

    def _own_plan_linear(
        self,
        start_world_pose: dict,
        end_world_pose: dict,
        *,
        arm_id: int = 0,
        tcp_offset: Any | None = None,
        seed_joints: Any | None = None,
        num_waypoints: int = 0,
        hold_rotation: bool = False,
        tolerance: float | None = None,
    ) -> dict | None:
        del tcp_offset
        return self.bus.call(
            self._plan_line,
            start_world_pose,
            end_world_pose,
            arm_id,
            seed_joints,
            num_waypoints,
            hold_rotation,
            tolerance,
        )

    def _plan_line(
        self, start_pose, end_pose, arm_id, seed_joints, num_waypoints, hold_rotation, tolerance
    ) -> dict | None:
        """The solved straight line, refusing with the reason the caller needs."""
        self.last_refusal = ""
        self.last_position_error = None
        solver = self.solver_for(arm_id)
        seed, template = self._seed(arm_id, seed_joints)
        start, _ = se3_arrays(start_pose)
        end, quat = se3_arrays(end_pose)
        roll = self._contract(hold_rotation)
        tol = float(self.tolerance if tolerance is None else tolerance)
        steps = int(num_waypoints) or self.waypoints
        bound = _LastError(self._bound(solver, template))
        path, refusals = plan_line(
            bound,
            seed,
            start.reshape(1, 3),
            end.reshape(1, 3),
            quat.reshape(1, 4),
            roll=roll,
            waypoints=steps,
            tolerance=tol,
            rot_tolerance=self.rot_tolerance,
            iterations=self.iterations,
        )
        refusal = refusals[0]
        if refusal is not None:
            self.last_refusal = self._why_no_line(solver, seed, template, end, quat, roll, refusal, tol)
            return None
        # The arrival error is the one number that says whether "planned" means "will get there":
        # the last waypoint's own, not a fresh solve's, which would land closer than the path does.
        self.last_position_error = float(bound.err[0, 0])
        return {"waypoints": [{"positions": [float(v) for v in row]} for row in path[0]]}

    def _why_no_line(self, solver, seed, template, end, quat, roll, refusal: LineRefusal, tol: float) -> str:
        """Whether the *target* is out of reach, or only the straight line to it.

        The two want opposite fixes: an unreachable pose has to move, while a
        reachable one at the end of an impossible line wants ``motion.plan_joint``.
        One extra solve of the endpoint separates them.
        """
        if roll == "held":
            return "orientation='lock' holds the end pose's whole rotation along the leg; " + refusal.text(
                tolerance=tol, rot_tolerance=self.rot_tolerance
            )
        try:
            _, err = self._solve_rows(solver, seed, end, quat, roll, template)
            error = float(err[0, 0])
        except Exception:
            return ""
        if error > tol:
            return (
                f"the end pose itself is out of reach for this arm -- solving it alone still misses by "
                f"{error * 1000:.0f} mm against a {tol * 1000:.0f} mm tolerance. No approach makes "
                "a pose the arm cannot hold reachable; move the target."
            )
        return (
            f"the end pose is reachable on its own (within {error * 1000:.0f} mm), but no straight line from "
            "here holds every intermediate point. A Cartesian line pins the whole path; motion.plan_joint "
            "reaches the same pose as a posture and does not."
        )

    # ------------------------------------------------------------- the planner's
    def plan_to_pose(self, *args: Any, **kwargs: Any) -> Any:
        """The planner's alone; ``None`` without one, and no fallback: nothing interpolates around obstacles."""
        planner = self._planner
        return None if planner is None else planner.plan_to_pose(*args, **kwargs)

    def robot_collision_spheres(self, *args: Any, **kwargs: Any) -> Any:
        planner = self._planner
        return [] if planner is None else planner.robot_collision_spheres(*args, **kwargs)

    @property
    def build_world_tsdf(self) -> Any:
        """The planner's world builder, when there is one -- the tool registers only then."""
        planner = self._planner
        if planner is None or not hasattr(planner, "build_world_tsdf"):
            raise AttributeError("build_world_tsdf: no collision-aware planner is installed")
        return planner.build_world_tsdf

    @property
    def solve_ik_batch(self) -> Any:
        planner = self._planner
        if planner is None or not hasattr(planner, "solve_ik_batch"):
            raise AttributeError("solve_ik_batch: no collision-aware planner is installed")
        return planner.solve_ik_batch

    @property
    def last_refusal_reason(self) -> str:
        return self.last_refusal

    def __getattribute__(self, attr: str) -> Any:
        if attr == "last_refusal":
            own = (
                object.__getattribute__(self, "_own_refusal")
                if "_own_refusal" in object.__getattribute__(self, "__dict__")
                else ""
            )
            if own:
                return own
            mine = object.__getattribute__(self, "__dict__").get("last_refusal", "")
            if mine:
                return mine
            planner = object.__getattribute__(self, "__dict__").get("planner")
            return str(getattr(planner, "last_refusal", "") or "") if planner is not None else ""
        return object.__getattribute__(self, attr)

    def __getattr__(self, attr: str) -> Any:
        """A planner's own attribute, so a graph that reached one through the composite still can."""
        planner = self.__dict__.get("planner")
        if planner is not None and self.__dict__.get("prefer_planned", True):
            return getattr(planner, attr)
        raise AttributeError(attr)


class _Templated:
    """A solver whose ``solve`` needs the whole model's pose, bound to one."""

    def __init__(self, solver: IKSolver, template: np.ndarray) -> None:
        self.solver = solver
        self.template = template
        self.kinematics = solver.kinematics

    def solve(self, *args: Any, **kwargs: Any):
        return self.solver.solve(*args, template=self.template, **kwargs)


class _LastError:
    """A solver that keeps its last solve's error: a line's last is its arrival."""

    def __init__(self, solver: IKSolver) -> None:
        self.solver = solver
        self.kinematics = solver.kinematics
        self.err: np.ndarray | None = None

    def solve(self, *args: Any, **kwargs: Any):
        q, self.err = self.solver.solve(*args, **kwargs)
        return q, self.err


def approach_of(pose: dict) -> np.ndarray:
    """A pose's approach axis in the world: its local +z, the whole of what a roll-free solve is asked for."""
    _, quat = se3_arrays(pose)
    return approach_axes(quat.reshape(1, 4))[0]


def _rotation_matrix(pose: dict) -> np.ndarray:
    return quat_to_mat(se3_arrays(pose)[1])
