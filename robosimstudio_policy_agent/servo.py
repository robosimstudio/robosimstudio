# SPDX-License-Identifier: Apache-2.0
"""The reference benchmark's motion, for graphs: a clipped Cartesian feedback servo.

robosuite's ``OSC_POSE`` never commands a far target. Each control step it
measures the end effector, bounds the remaining error by ``output_max``
(0.05 m / 0.5 rad), and applies impedance torque toward that bounded delta -- so
the arm has a hard Cartesian speed cap and, because the delta shrinks with the
error, an exponential taper into the target. That is where the reference's
precision comes from, and it is exactly what the runtime's ``go_to_pose`` does
not do on a stiff position servo: one IK solve, then a joint-space move, which is
a step input taken at whatever speed the gains allow.

This reproduces that loop position-controlled: measure the TCP, advance a
setpoint toward the goal under the same law the ``osc_pose`` action space uses,
solve that nearby setpoint (seeded with the last commanded joints, so the
velocity-cost solve stays on the arm's branch), command the joints, step once.
Feedback is the point: a blocked or disturbed arm re-aims from where it actually
is, which no pre-planned waypoint chain does.

**The law is not restated here.** It is
:class:`~robosimstudio.solvers.OscSetpoint`, shared with the controller.

**Arrival is tested on what the solver was asked for.** A backend that leaves
wrist roll free (:class:`~robosimstudio_policy_agent.ik.IKBackend` under ``--roll free``,
the default on nearly every embodiment) does not converge a full quaternion, so testing one guarantees every
waypoint runs its whole budget -- the move never *fails*, it just always takes the
maximum, which is indistinguishable from a slow simulator. So the orientation
test follows the backend's own ``honours_roll``: the approach axis for a roll-free
solver, the whole rotation for one that solves it.

**And so is the goal the law is driven toward**, which is the same principle
applied one step earlier and is not optional either. :meth:`OscSetpoint.advance`
bounds how far the setpoint may lead the arm, and that lead clamp is a bound on a
*full quaternion angle*. Hand it a goal whose roll the solver is going to discard
and the clamp spends its budget on that roll: it slerps from the measurement only
``rot_lead / total`` of the way, and the approach axis -- the one component the
solver does control -- is corrected by that same fraction. The residual is then a
fixed point rather than a transient, at ``solver_slop * total / rot_lead``.

Measured on ``tool_sort/level1_6``, where ``sim.get_object_obb``'s ``grasp_quat``
asks for a cross-the-handle roll about 84 degrees off the one a free wrist holds:
``total`` 1.46 rad against a ``rot_lead`` of 0.5, so the solver's own 1.5 degrees
of slop stood at **4.3 degrees** of approach error and never improved. Against a
3 degree arrival tolerance that is a move which cannot arrive, and each one burned
its whole 400-iteration budget -- about 1030 control steps apiece, two of them per
episode, against a 2400-step ceiling. The graph did not fail; it ran out of time.

So :meth:`CartesianServo._solver_goal` restates the goal in the terms the backend
solves in: for a roll-free one, the measured orientation turned by the shortest
arc that puts its approach axis on the goal's. The commanded roll is dropped here
rather than silently downweighted, which is what a roll-free
:meth:`~robosimstudio_policy_agent.ik.IKBackend.solve_ik` does with it half a step later anyway.

**Restated once per move, from the first measurement -- not once per turn.** The
restated goal carries the roll the wrist *has*; restating it every turn made the
roll the wrist has *now* the roll the next solve is asked to keep, so a full-pose
solve that rolled the wrist to gain position (which the LM solve does whenever
the position is blocked: the rotation objective weighs 0.6 against the position)
was ratified on the next turn and rolled again. Measured on
``tool_hanging/wrenches_level1``'s insert legs, where the ring is pushed onto the
hook and the position stops short: 25-29 degrees of roll on the approach leg and
47-51 degrees cumulative on the push, in every serial episode that lost the
wrench at the release (10 of 13 with a correct registration), against 0.2-4
degrees under the old roll-free DLS solve, which never rolled the wrist at all.
One goal orientation per move -- the goal's approach axis with the roll the arm
had when the move began -- turns that ratchet into a bounded trade the law
pulls back every tick. Along an unblocked arc the two laws agree: the lead
slerps along the geodesic between the start and goal axes, on which the
transported roll is the same whether carried once or tick by tick.

**The per-turn solve is the fleet's per-step solve, at one world.** Each turn's
setpoint -- the law's rate-capped position *and* its slerped quaternion -- is
handed to a :class:`~robosimstudio.solvers.ik.batched.BatchedArmIK` built with
``num_worlds=1`` from the same ingredients
:class:`~robosimstudio_bench.envs.fleet.FleetEnv` builds its rig from
(:func:`~robosimstudio.scene.ik_model.plan_ik_model` and
:func:`~robosimstudio.scene.ik_model.build_ik_rig`): the IK model, the arm's
coordinates, the tool-centre body and offsets, eight Levenberg-Marquardt
iterations at rotation weight 0.6, warm-started from the arm's live joints.
It used to be the gap backend's own per-turn solve, roll-free on nearly every
embodiment, and that solve is what the fleet does *not* do. Measured on tool
hanging level 1, episode 8, same code otherwise: the serial pre-grasp
``go_to_pose`` timed out at 400 steps with the tool 102 mm short and the wrist
161 degrees off in roll -- the roll-free solve had settled into a flipped wrist
branch and stayed there, its approach axis 8.5 degrees off and nothing in the
residual it minimised to pull it out -- while the fleet's full-pose solve of
the same law and setpoint arrived in 71 steps (3.2 mm, 0.7 degrees). Over
thirty episodes: serial 26/30 with four such stalls, fleet 30/30. One solver
class and one set of numbers is what makes the two runners comparable.

The gap backend keeps every other job -- planner seeds, the ``solve_ik`` tools,
``plan_linear`` -- and one job here: the **unreachable verdict**. The fleet's
rig drives from live state and reports no residual, so a goal beyond the arm
has to be asked of a solver that refuses; the fleet asks its template solver
once a stalled move is a stall window old and still outside the 30 mm reach
gate (``fleet/stepper.py``, ``_beyond_reach``), and this loop asks
:meth:`~robosimstudio_policy_agent.ik.IKBackend.solve_ik` for the *goal* pose at the
same moment. A refusal raises the same ``unreachable`` text the graph's pose
verb routes on. A robot whose scene yields no rig -- no arm segments, or a
model Newton's LM solver cannot host (:class:`~robosimstudio.scene.ik_model.IKPlan`
tier ``host``) -- keeps the old per-turn backend solve, and says so.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

# Aliased: ``arm_name`` is the local this method already binds, and shadowing an
# import with a local is how a rename silently stops resolving arms at all.
from robosimstudio.arms import arm_name as resolve_arm
from robosimstudio.core.bus import BusTimeout
from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio.core.streams import reading
from robosimstudio.solvers import OscSetpoint
from robosimstudio.solvers.ik.solver import POS_TOLERANCE, ROT_TOLERANCE_DEG

__all__ = [
    "POS_TOLERANCE",
    "ROT_TOLERANCE_DEG",
    "UNREACHABLE_GATE_M",
    "ArmLM",
    "CartesianServo",
    "build_arm_lm",
    "rotation_error",
    "solver_goal",
]

MAX_STEPS = 1000
"""Control steps before giving up: 16.7 s at 60 Hz.

Was 400 when a step was one loop turn, and a turn spanned 2-3 simulator steps on the
machines the graphs were written on (2.3 idle, 3.0 at three processes per GPU); the
servo now counts every simulator step, so the number states the duration the graphs
always relied on. The fleet's ``ServoToPose.max_steps`` default says the same."""

POLL_S = 0.0005
"""How long the loop sleeps when the simulator has not stepped since its last turn."""
STALL_ESCAPE_S = 2.0
"""How long a turn waits for a new simulator step before counting anyway.

A bus that never publishes -- a synchronous harness driving the env in this same
thread -- is not a stalled one, and a loop that waited for it forever would hang
every such caller."""
WALL_BUDGET_PER_STEP_S = 60.0 / 240.0
"""Wall-clock a single servo step may take before the move is abandoned [s].

The budget below is in *simulator* steps, and a turn that never sees one costs
:data:`STALL_ESCAPE_S`. Without a wall bound, a 400-step move on a bus that has
gone quiet blocks for 400 x 2 s -- thirteen minutes in which nothing is wrong
enough to report. Sized off the slowest control rate the suite runs (240 Hz) so
a healthy move never approaches it, and floored below."""
WALL_BUDGET_FLOOR_S = 120.0
"""Never abandon a move for wall clock before this, however small its budget."""
UNREACHABLE_GATE_M = 0.03
"""How far a stalled move must still be from its goal before the goal is judged for reach [m].

The gap backend's own reach gate (``IKBackend.tolerance``), which is read off
the backend when it carries one; this is the value for a backend that does not.
The fleet's stepper gates its verdict on the same number."""


def rotation_error(measured_quat: Any, goal_quat: Any, honours_roll: bool) -> float:
    """Orientation error [rad], measured the way the backend controls it.

    Roll-honouring: the angle between the two quaternions. Roll-free: the angle
    between the approach axes (tool +z), which is the whole of what a roll-free
    solver is asked to align. Shared by :class:`CartesianServo` and the fleet's
    stepper, so the serial and batched runners measure arrival alike.
    """
    measured = np.asarray(measured_quat, dtype=np.float64)
    goal = np.asarray(goal_quat, dtype=np.float64)
    if honours_roll:
        return 2.0 * float(np.arccos(np.clip(abs(float(np.dot(measured, goal))), -1.0, 1.0)))
    measured_axis = quat_to_mat(measured)[:, 2]
    goal_axis = quat_to_mat(goal)[:, 2]
    return float(np.arccos(np.clip(float(np.dot(measured_axis, goal_axis)), -1.0, 1.0)))


def solver_goal(measured_quat: Any, goal_quat: Any, honours_roll: bool) -> np.ndarray:
    """The goal orientation, restated in what the backend actually solves.

    The goal unchanged when roll is honoured; otherwise the *measured*
    orientation turned by the shortest arc that lands its approach axis (local
    +z) on the goal's -- same axis as the goal, same roll as the arm. Asked once
    per motion, by :meth:`CartesianServo._solver_goal` and by the fleet's
    stepper alike, so the two runners close the same orientation.
    """
    goal = np.asarray(goal_quat, dtype=np.float64)
    if honours_roll:
        return goal
    measured = np.asarray(measured_quat, dtype=np.float64)
    measured_axis = quat_to_mat(measured)[:, 2]
    goal_axis = quat_to_mat(goal)[:, 2]
    cross = np.cross(measured_axis, goal_axis)
    sin = float(np.linalg.norm(cross))
    cos = float(np.dot(measured_axis, goal_axis))
    if sin < 1.0e-9:
        if cos > 0.0:  # already aligned: the measured pose *is* the goal
            return measured
        # Exactly opposed. Every arc is shortest, so pick one that exists:
        # any axis perpendicular to the one being reversed will do.
        perpendicular = np.cross(measured_axis, (1.0, 0.0, 0.0))
        if float(np.linalg.norm(perpendicular)) < 1.0e-6:
            perpendicular = np.cross(measured_axis, (0.0, 1.0, 0.0))
        turn = quat_from_axis_angle(perpendicular, np.pi)
    else:
        turn = quat_from_axis_angle(cross / sin, float(np.arctan2(sin, cos)))
    return np.asarray(quat_multiply(turn, measured), dtype=np.float64)


class ArmLM:
    """One arm's fleet solver at one world, and the scratch a serial turn solves through.

    Args:
        rig: A :class:`~robosimstudio.solvers.ik.batched.RobotIKRig` holding
            exactly this arm at ``num_worlds=1``.
        name: The arm's name in the rig.
        scene_coords: The arm's ``joint_q`` indices in the scene's model -- what
            the solved coordinates are read back at, and what the fleet's
            scatter writes them to.

    The two scratch arrays are the whole reason this is a class. The rig
    captures its solve into a CUDA graph keyed on the ``(joint_q,
    joint_target_q)`` pointer pair, and the environment's live ``joint_q``
    alternates between two state buffers; solving straight out of it would
    recapture every other turn. So the live joints are copied into
    :attr:`seed` -- the fleet's own warm start, gathered from live state -- and
    the solution lands in :attr:`out`, and the pair never changes.
    """

    def __init__(self, rig: Any, name: str, scene_coords: Any) -> None:
        import warp as wp

        self.rig = rig
        self.name = str(name)
        self.arm = rig.arms[self.name]
        self.scene_coords = np.asarray([int(c) for c in scene_coords], dtype=np.int64)
        device = rig.model.device
        self.seed = wp.zeros(int(rig.coords_per_world), dtype=wp.float32, device=device)
        self.out = wp.zeros(int(rig.coords_per_world), dtype=wp.float32, device=device)
        # The one world is always driven; the scatter is gated on it.
        self.arm.enable_world(0, True)

    def solve(self, joint_q: Any, position: Any, quat_xyzw: Any) -> np.ndarray:
        """Solve one setpoint from the live joints. Returns this arm's coordinates.

        Owner thread. ``joint_q`` is the environment's live ``joint_q`` (a warp
        array over the whole model); the setpoint is the law's full pose, both
        halves, exactly as the fleet's ``_advance_lead`` hands it to
        :meth:`BatchedArmIK.set_target`.
        """
        import warp as wp

        self.arm.set_target(0, np.asarray(position, dtype=np.float64), np.asarray(quat_xyzw, dtype=np.float64))
        self.arm.flush()
        wp.copy(self.seed, joint_q)
        self.rig.solve_into(self.seed, self.out)
        with reading():
            solved = self.out.numpy()
        return np.asarray(solved[self.scene_coords], dtype=np.float64)


def build_arm_lm(env: Any, arm_name: str) -> ArmLM | None:
    """One arm's fleet rig at one world, from the scene the environment holds -- or ``None``, with the reason printed.

    The ingredients are :class:`~robosimstudio_bench.envs.fleet.FleetEnv`'s,
    through the same two functions it calls: :func:`plan_ik_model` chooses
    the model the LM solver can host (the scene's own, a reduced rebuild
    without the task's objects, or none), and :func:`build_ik_rig` assembles
    the arm's :class:`~robosimstudio.solvers.ik.batched.BatchedArmIK` -- tool
    body, offsets, home pose, iterations, rotation weight -- over it. Nothing
    about the solve is restated here.

    Returns ``None`` for a scene with no rig ingredients (no arm segments, no
    request or episode to plan from) and for a plan with no batched tier; the
    servo then keeps its old per-turn backend solve. Owner thread: the plan
    reads the model and a reduced tier builds one.
    """
    scene = getattr(env, "scene", None)
    embodiment = getattr(scene, "embodiment", None)
    arms = getattr(embodiment, "arms", None)
    request = getattr(scene, "request", None)
    episode = getattr(env, "episode", None)
    if not arms or arm_name not in arms or request is None or episode is None:
        # No arm segments (or nothing to plan a model from): a scene stand-in,
        # or a robot the scene indexes no arm for. The old path serves it.
        return None
    from robosimstudio.scene.ik_model import build_ik_rig, plan_ik_model

    try:
        plan = plan_ik_model(request, episode, scene)
        if not plan.batched:
            print(f"[servo] arm {arm_name!r}: no batched LM rig ({plan.reason}); solving each turn with the IK backend")
            return None
        rig = build_ik_rig(scene, plan, num_worlds=1, arms=(arm_name,))
    except ValueError as exc:
        # The two builds disagree about the robot (body order, coordinate
        # counts). A rig aimed at the wrong link is worse than none.
        print(f"[servo] arm {arm_name!r}: no batched LM rig ({exc}); solving each turn with the IK backend")
        return None
    if rig is None or arm_name not in rig.arms:
        print(
            f"[servo] arm {arm_name!r}: no batched LM rig (plan covers {sorted(plan.arm_coords)}); solving each turn with the IK backend"
        )
        return None
    lm = ArmLM(rig, arm_name, arms[arm_name].arm_coords)
    print(
        f"[servo] arm {arm_name!r}: per-turn solve is the fleet's BatchedArmIK at one world "
        f"(ik tier {plan.tier}, {lm.arm.iterations} LM iterations)"
    )
    return lm


class CartesianServo:
    """Drive one arm to a world TCP pose the way ``OSC_POSE`` would.

    Args:
        env: The toolkit-layer environment.
        facade: The gap facade; every simulation touch goes through its bus.
        ik: The IK backend. Its ``honours_roll`` decides how arrival is tested;
            see the module docstring.
    """

    @property
    def honours_roll(self) -> bool:
        """The backend's roll contract, read live rather than cached at construction.

        The backend builds its solver lazily, and a PyRoKi build that falls back
        to the DLS family can drop the roll declaration then (``IKBackend._adopt``):
        a flag cached here would test arrival on a roll the solver never closes.
        A value assigned directly (the contract tests do) overrides the backend.
        """
        override = self.__dict__.get("_honours_roll")
        if override is not None:
            return override
        return bool(getattr(getattr(self, "ik", None), "honours_roll", False))

    @honours_roll.setter
    def honours_roll(self, value: bool) -> None:
        self._honours_roll = bool(value)

    def __init__(self, env, facade, ik) -> None:
        self.env = env
        self.facade = facade
        self.ik = ik
        self.law = OscSetpoint(control_dt=1.0 / float(env.task.physics.control_hz))
        self.base_speed_scale = self.law.speed_scale_for(getattr(env.task, "servo_speed_mps", None))
        """The task's declared servo speed as a factor on every move: ``min(1, servo_speed_mps / cruise)``.

        ``(:servo-speed-mps X)`` in the task's staging, 1.0 where it states
        none. A graph's ``speed_scale`` multiplies this -- it is relative to the
        speed the graph was written for, not to the law's nominal cruise -- and
        the fleet's lead law (``fleet/stepper.py``, ``_lead``) applies the same
        factor, so both runners drive the same law at the same speed. This is
        where the speed the paper's graphs were authored against lives now: a
        turn of this loop used to span two or three simulator steps with the
        law advanced once per turn, which ran the tool at 0.10-0.13 m/s on the
        authoring machines; the loop now counts every step, and gear_mesh's and
        cable_route's fine legs failed at the nominal 0.31 m/s until their
        suites stated the speed they had always been run at. Nothing here reads
        the host: a factor derived from the machine would move the measurement
        with the machine, which is the defect this replaces.
        """
        self._lm: dict[str, ArmLM | None] = {}
        """Per arm, its one-world fleet solver, or ``None`` for an arm the old path serves. Built on first use."""
        self._lm_scene: int | None = None
        """``id`` of the scene the cache was built over: a rebuilt scene (a reset that changes the model) drops it."""
        self._scaled: dict[float, OscSetpoint] = {}
        """One law per distinct ``speed_scale``, kept rather than rebuilt per move.

        Not for the allocation. A law built inside :meth:`move` and dropped at
        its end is an object no caller and no test can reach, and
        ``servo.law.advance`` is exactly what a test patches to see what goal
        the law was handed. A cached law is a real object; the dict is bounded
        by how many distinct scales a graph asks for (three, on tool hanging)."""

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _wait_for_step(bus: Any, seen: int, *, until: float) -> None:
        """Block until the bus publishes a step past ``seen``, or ``until`` (monotonic) passes.

        The bus's own wait, because in a wave it parks the world and the
        barrier steps everyone; a stub bus without one, or one that stops
        answering, just hands the turn back to the loop's own escape.
        """
        wait = getattr(bus, "wait_for_step", None)
        remaining = until - time.monotonic()
        if wait is None or remaining <= 0:
            time.sleep(POLL_S)
            return
        try:
            wait(after=seen, timeout=remaining)
        except BusTimeout:
            return

    def _measure(self, arm_name: str):
        """TCP pose off the published snapshot -- no bus round trip.

        Every ``bus.call`` from this worker thread costs one rollout iteration to
        get serviced, so a measure-call plus an apply-call made each servo step
        tripled the env steps per iteration and cut the effective speed to a
        third of the cap. The snapshot is published after every step and carries
        ``body_q``, which is all a TCP read needs; it is one step stale, which for
        a servo whose whole job is feedback at bounded speed is the difference
        between k and k+1 of a 2 cm step.

        The rotation is the **grasp** frame, not the wrist: ``ee_offset_rot``
        composed on, matching what :meth:`ArmIK.ee_pose` returns and what every
        commanded pose is stated in.
        """
        from robosimstudio.core.math import transform_point

        handle = self.env.handles.arms[arm_name]
        body_q = self.facade.bus.snapshot.body_q
        if body_q is None:  # before the first publish: read once, marshalled
            env = self.env
            body_q = self.facade.bus.call(lambda: np.asarray(env.state.body_q.numpy(), dtype=np.float64))
        xform = np.asarray(body_q[handle.ee_body], dtype=np.float64)
        pos = transform_point(xform, handle.ee_offset)
        quat = quat_multiply(xform[3:7], handle.ee_offset_rot)
        return pos, np.asarray(quat, dtype=np.float64)

    def _rotation_error(self, measured_quat: np.ndarray, goal_quat: np.ndarray) -> float:
        """Orientation error [rad], measured the way the backend controls it: :func:`rotation_error`."""
        return rotation_error(measured_quat, goal_quat, self.honours_roll)

    def _solver_goal(self, measured_quat: np.ndarray, goal_quat: np.ndarray) -> np.ndarray:
        """The goal orientation, restated in what the backend actually solves.

        For a solver that honours roll this is the goal unchanged. For a
        roll-free one it is the *measured* orientation turned by the shortest arc
        that lands its approach axis on the goal's -- same axis as the goal,
        same roll as the arm. :meth:`move` asks once, from its first
        measurement, and drives every tick of :meth:`OscSetpoint.advance` toward
        that one orientation: restating it from each turn's measurement let a
        full-pose solve ratchet the roll on a blocked leg (module docstring).

        Dropping the roll is not a loss of authority the servo had: a
        roll-free solve keeps only the pose's local +z. What it
        removes is the roll's *weight in the lead clamp*, which is where a
        discarded component was still costing the axis its correction rate --
        see the module docstring for the measurement.
        """
        return solver_goal(measured_quat, goal_quat, self.honours_roll)

    def _lm_for(self, arm_name: str) -> ArmLM | None:
        """This arm's one-world fleet solver, built once per scene, or ``None`` for the old path.

        ``getattr`` throughout: the contract and cadence tests build a servo
        through ``__new__`` with only the law bound, and their scene stand-ins
        have no arm segments -- which is exactly the case the old path is kept
        for. The build is marshalled onto the owner thread because it reads the
        model and may build a reduced one.
        """
        # Off unless asked for (ROBOSIMSTUDIO_SERVO_LM=1). Measured on tool_hanging
        # level 1, thirty episodes: with the one-world LM solve driving the
        # serial turn, 19 of 49 in-hand registrations read the wrong end of the
        # wrench where the per-turn backend solve (0 of 26) and the fleet's own
        # stepper (0 of 30) never did, and the cell scored 59 % against 82 %;
        # the cause was not found in the day. The user ruled the serial servo
        # back to its per-turn backend solve, keeping the exact step count and
        # the once-per-move orientation restatement. The LM path stays for the
        # study that will close that gap.
        if os.environ.get("ROBOSIMSTUDIO_SERVO_LM", "0") != "1":
            return None
        scene = getattr(getattr(self, "env", None), "scene", None)
        if getattr(self, "_lm_scene", None) != id(scene) or not hasattr(self, "_lm"):
            self._lm = {}
            self._lm_scene = id(scene)
        if arm_name not in self._lm:
            bus = getattr(getattr(self, "facade", None), "bus", None)
            call = getattr(bus, "call", None)
            env = getattr(self, "env", None)
            self._lm[arm_name] = call(build_arm_lm, env, arm_name) if call is not None else build_arm_lm(env, arm_name)
        return self._lm[arm_name]

    def _lm_turn(self, lm: ArmLM, position: np.ndarray, quat_xyzw: np.ndarray, arm_id: int) -> np.ndarray:
        """One turn's solve and command, on the owner thread.

        Solve the law's setpoint from the arm's live joints, then command the
        solution through :meth:`~robosimstudio_policy_agent.facade.GapEnvFacade.apply_joint_action`
        -- the same call the old path made, one bus round trip earlier. The
        old turn was two calls (solve, then apply), each serviced between
        rollout steps; one call is one servicing, so a turn spans fewer
        simulator steps, and the per-step law (``k`` ticks per turn) keeps the
        arm's speed and the budget the same in steps either way.
        """
        solution = lm.solve(self.env.state.joint_q, position, quat_xyzw)
        if not np.isfinite(solution).all():
            # A non-finite command would poison the simulator quietly; a
            # raised one names the turn it happened on.
            raise RuntimeError(f"servo: LM solve returned non-finite joints {solution.tolist()}")
        self.facade.apply_joint_action(np.asarray(solution, dtype=np.float64), arm_id)
        return solution

    # ------------------------------------------------------------------- move
    def move(
        self,
        pose: dict,
        *,
        arm_id: int = 0,
        max_steps: int = 0,
        speed_scale: float = 1.0,
        pos_tolerance: float | None = None,
        stop_on_stall: bool = False,
        stall_steps: int = 12,
        stall_epsilon: float = 0.0002,
    ) -> dict[str, float | str]:
        """Servo to a world TCP pose. Blocks until arrived or out of budget.

        ``pos_tolerance`` is this move's own arrival ball [m], else
        :data:`POS_TOLERANCE`. A pick that has to put a bore over a peg with
        0.3 mm of clearance cannot live with 4 mm: the pads centre the gear
        along the closing axis only, and whatever the hand missed by along the
        pad faces is carried to the peg and lands the gear on its tip
        (gearmesh-easy-single-05 under rtx: eight tries around the peg, none
        within the cone's 2.3 mm). A fine leg asks for 1.5 mm and pays for it
        in steps.

        ``speed_scale`` in (0, 1] lowers the law's output ceiling, and with it
        both the OSC-derived cruise rate and the maximum command lead, by the
        same factor. It is a genuine Cartesian speed limit -- trajectory time
        scaling cannot reach this feedback-servo path -- and tool hanging's
        graph carries 0.40 to 0.60 on the scissors and screwdriver carries,
        where a full-speed reorientation swings the tool out of the jaws. It is
        relative to the task's declared servo speed where one is stated
        (:attr:`base_speed_scale`): the effective factor on the ceiling is the
        product, so a graph's 0.5 means half the speed it was written for.

        The two are orthogonal: one bounds how far the setpoint may lead per
        turn, the other decides when a turn counts as arrived. A slow move
        simply needs more turns to close the same ball.

        Raises:
            RuntimeError: When the goal itself is unreachable, which the caller's
                subgraph should route as a failure rather than have the arm park
                at whatever was closest.
        """
        # Role order, not declaration order -- the same arm_id the caller's
        # go_to_pose used. See robosimstudio.arms.
        arm_name = resolve_arm(self.env, arm_id)
        p = pose["position"]
        q = pose["rotation"]
        goal_p = np.array([float(p["x"]), float(p["y"]), float(p["z"])])
        goal_q = np.array([float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])])
        goal_q = goal_q / np.linalg.norm(goal_q)
        speed_scale = float(speed_scale)
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}")
        # The graph's scale is relative to the task's declared speed, so the
        # factor on the law's ceiling is the product. ``getattr``: the contract
        # tests build a servo through ``__new__`` without a task behind it.
        scale = speed_scale * float(getattr(self, "base_speed_scale", 1.0))
        # Zero or negative is "use the default", not a budget that ends the
        # move before its first turn.
        budget = int(max_steps) if int(max_steps) > 0 else MAX_STEPS
        rot_tol = np.radians(ROT_TOLERANCE_DEG)
        pos_tol = POS_TOLERANCE if pos_tolerance is None else float(pos_tolerance)
        # The fleet's solver at one world, or None for the old per-turn backend
        # solve -- see the module docstring.
        lm = self._lm_for(arm_name)
        reach_gate = float(getattr(self.ik, "tolerance", UNREACHABLE_GATE_M) or UNREACHABLE_GATE_M)
        stall_window = max(1, int(stall_steps))

        # Seed from what was last *commanded* -- the facade holds it, it is free
        # to read, and it is the branch the arm is on. Falls back to one
        # marshalled read before anything has been commanded.
        held = self.facade._held.get(int(arm_id))
        if held is not None:
            seed = [float(v) for v in held]
        else:
            env = self.env
            arm_coords = env.arm(arm_name).arm_coords
            seed = [
                float(v)
                for v in self.facade.bus.call(
                    lambda: np.asarray(env.state.joint_q.numpy(), dtype=np.float64)[arm_coords]
                )
            ]

        # The setpoint starts where the arm is, then leads it. Re-seeding per
        # move rather than per episode is what keeps a move that begins after a
        # disturbance from inheriting a command aimed at the old pose.
        measured = self._measure(arm_name)
        law = self.law
        if scale < 1.0:
            law = self._scaled.get(scale)
            if law is None:
                law = self._scaled[scale] = OscSetpoint(
                    control_dt=self.law.control_dt,
                    gains=self.law.gains,
                    output_max=self.law.output_max * scale,
                )
            # Seed the unscaled law too. Nothing in this method reads it again,
            # but the studio panel and OscController both read ``self.law``'s
            # setpoint, and a scaled move would otherwise leave it showing a
            # pose from two moves ago.
            self.law.reset(*measured)
        law.reset(*measured)
        # ONE goal orientation for the whole move: the goal's approach axis with
        # the roll the arm has now (see the module docstring, "restated once per
        # move"). Restating it from every turn's measurement ratcheted the roll
        # on a blocked leg, one solve at a time.
        solver_goal = self._solver_goal(measured[1], goal_q)

        misses = 0
        best_dist = float("inf")
        stalled = 0
        judged_at = 0
        """The stall count at which the goal was last judged for reach, so a
        move that stays stalled is judged once per stall window, not per turn."""
        # ONE ITERATION PER SIMULATOR STEP, counted off the published snapshot.
        # The loop used to count its own turns: a bus call is serviced between
        # steps, and on a box running sixteen of these the worker thread came
        # back from its call, read the same snapshot it had read before, and saw
        # a hand that "had not moved" -- fifteen such turns were a "contact"
        # 17-20 mm above the target (gearmesh under rtx at load average 970:
        # every carry stopped short at every offset, easy 12/30 against 14/14 on
        # the same code unloaded), and a hundred and twenty of them were a
        # "timeout" the arm never got to use. A turn on which the simulator has
        # not stepped is not a step of the move.
        #
        # AND A TURN THAT SPANS k STEPS IS k STEPS OF THE MOVE. The rule above
        # fixed the turn that saw no new step and left the turn that saw
        # several counting one. On a loaded host the simulator runs on between
        # the worker's turns, so every turn stood for however many steps had
        # passed meanwhile: measured at three processes per GPU, k was 3 -- a
        # 400-turn leg ran exactly 1200 simulator steps, its budget silently
        # tripled, and the setpoint law, advanced once per turn at the control
        # period, travelled at a third of its speed. The budget and the arm's
        # speed were both functions of host load, which is why a serial cell's
        # numbers moved with what else the box was running.
        #
        # So k = seen - last_seen is what a turn counts, what the stall window
        # grows by, and how many control periods the law is advanced from this
        # turn's one measurement (the last setpoint is the one sent) -- the
        # same rule the batched runner applies per readback at its cadence
        # (fleet/stepper.py, ``_advance_lead`` and the servo branch of
        # ``_finished``). The first turn, a bus with no step counter, and a
        # turn released by STALL_ESCAPE_S count 1, which is what they always
        # did. The budget, the stall window and the returned ``steps`` are
        # simulator steps whatever the host is doing.
        step = 0
        dist = -1.0  # no measurement yet; a row cannot carry an unbound name or an infinity
        last_seen = None
        waited_since = time.monotonic()
        deadline = waited_since + max(WALL_BUDGET_FLOOR_S, budget * WALL_BUDGET_PER_STEP_S)
        while step < budget:
            # ``getattr`` rather than ``self.facade``: the contract tests build
            # a CartesianServo through ``__new__`` with only ``law`` and
            # ``honours_roll`` set, and an attribute access here would make this
            # loop unreachable to any arithmetic-only test.
            bus = getattr(getattr(self, "facade", None), "bus", None)
            seen = getattr(getattr(bus, "snapshot", None), "step", None)
            if seen is not None and seen == last_seen and time.monotonic() - waited_since < STALL_ESCAPE_S:
                # The budget is in simulator steps and this turn did not get
                # one. Wait for the next -- through the bus, whose wait parks
                # the world in a wave so the barrier can close, rather than by
                # polling the clock -- but not past the wall deadline: a bus
                # that has gone quiet for good would otherwise hold the caller
                # for budget x STALL_ESCAPE_S with nothing wrong enough to report.
                if time.monotonic() > deadline:
                    # ``best_dist`` is still infinite when the bus went quiet
                    # before the first measurement, and an infinity is not a
                    # number a result row can carry.
                    remaining = best_dist if np.isfinite(best_dist) else -1.0
                    return {"outcome": "timeout", "remaining_distance": remaining, "steps": step}
                self._wait_for_step(bus, seen, until=min(deadline, waited_since + STALL_ESCAPE_S))
                continue
            # How many simulator steps this turn stands for. Only a counter that
            # is known on both sides and moved can say more than one: the escape
            # above hands back a turn with ``seen == last_seen``, and a counter
            # that went backwards (a bus reset under the move) is not a negative
            # number of steps.
            k = 1 if seen is None or last_seen is None else max(1, int(seen) - int(last_seen))
            last_seen = seen
            waited_since = time.monotonic()
            step += k
            pos, quat = self._measure(arm_name)
            dist = float(np.linalg.norm(goal_p - pos))
            ang = self._rotation_error(quat, goal_q)
            if dist < pos_tol and ang < rot_tol:
                return {"outcome": "reached", "remaining_distance": dist, "steps": step}

            # Contact is observable without simulator ground truth: while a
            # Cartesian command continues toward a reachable target, the TCP
            # stops making progress.  This is the same signal a real position
            # controlled arm can expose even when it has no force sensor.
            if dist < best_dist - float(stall_epsilon):
                best_dist = dist
                stalled = 0
                judged_at = 0
            else:
                stalled += k
                if lm is not None and stalled - judged_at >= stall_window and dist > reach_gate:
                    # Stalled for a stall window and still well short: is the
                    # goal beyond the arm? The fleet's rig drives from live
                    # state and reports no residual, and so does this one, so
                    # the verdict is asked of the gap backend -- for the GOAL
                    # pose, seeded from the arm's last command, as the fleet's
                    # ``_beyond_reach`` asks its template solver -- once per
                    # stall window. A goal it solves is not unreachable: the
                    # arm is blocked or slow, and the budget decides. A goal it
                    # refuses raises the text the graph's pose verb routes on.
                    judged_at = stalled
                    if self.ik.solve_ik(pose, arm_id=arm_id, seed_joints=seed) is None:
                        raise RuntimeError(
                            f"servo: pose unreachable after {stalled} steps without progress "
                            f"(remaining {dist * 1000:.0f} mm / {np.degrees(ang):.0f} deg)"
                        )
                if stop_on_stall and step >= int(stall_steps) and stalled >= int(stall_steps):
                    return {"outcome": "contact", "remaining_distance": dist, "steps": step}

            # k control periods of the law from this one measurement, the ticks
            # taken one at a time because the clamp is nonlinear -- scaling
            # ``dt`` by k would also let a blocked arm creep. The last setpoint
            # is the target, exactly as the fleet's ``_advance_lead`` sends it.
            for _ in range(k):
                target_p, target_q = law.advance(pos, quat, goal_p, solver_goal)
            if lm is not None:
                # The fleet's per-step solve: the law's full setpoint, position
                # and slerped quaternion, solved from the live joints by the
                # one-world BatchedArmIK and commanded in the same owner-thread
                # call. The solution is kept only as the verdict's seed.
                seed = [float(v) for v in self.facade.bus.call(self._lm_turn, lm, target_p, target_q, arm_id)]
                continue
            # No rig for this arm (no arm segments, or a model Newton's LM
            # solver cannot host): the per-turn backend solve, as it always was,
            # with its own verdict -- three consecutive refusals.
            step_pose = {
                "position": {"x": float(target_p[0]), "y": float(target_p[1]), "z": float(target_p[2])},
                "rotation": {
                    "x": float(target_q[0]),
                    "y": float(target_q[1]),
                    "z": float(target_q[2]),
                    "w": float(target_q[3]),
                },
            }
            solution = self.ik.solve_ik(step_pose, arm_id=arm_id, seed_joints=seed)
            if solution is None:
                # An intermediate along the path can sit in a fold two reachable
                # endpoints do not. Aim at the goal itself before giving up --
                # bigger step, but a well-defined one.
                solution = self.ik.solve_ik(pose, arm_id=arm_id, seed_joints=seed)
            if solution is None:
                misses += 1
                if misses >= 3:
                    raise RuntimeError(
                        f"servo: pose unreachable after {misses} consecutive solve failures "
                        f"(remaining {dist * 1000:.0f} mm / {np.degrees(ang):.0f} deg)"
                    )
                continue
            misses = 0
            seed = [float(v) for v in solution]
            self.facade.bus.call(self.facade.apply_joint_action, np.asarray(solution, dtype=np.float64), arm_id)
        # Out of budget is the reference's own behaviour too: go_to holds for a
        # fixed count and moves on. The caller's verify decides what it meant.
        # ``step``, not ``budget``: the last turn may have carried the count past
        # the budget by up to k - 1, and the row reports what the move cost.
        return {"outcome": "timeout", "remaining_distance": dist, "steps": step}
