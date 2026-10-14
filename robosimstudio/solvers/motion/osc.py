# SPDX-License-Identifier: Apache-2.0
"""Operational-space control of one arm: the reference benchmark's own action space.

The VAB-industrial scripts (``mujoco/run_hammer_pick_place.py``,
``run_tool_sort.py``, ``vab_bddl/run_bddl_tool_sort.py``) all drive a Panda
through robosuite's ``OSC_POSE``, and every one of them does it through the same
two functions: :func:`osc_action`, which turns a waypoint into a *normalized*
Cartesian delta, and a ``go_to`` loop that re-issues that delta for a fixed
number of control steps. So the behaviour those scripts produce is not really a
property of their waypoints -- it is a property of this controller, and porting
the waypoints without it ports the geometry and not the motion.

Three things define the behaviour, and all three are here:

* **The delta is measured, not integrated.** Every control step the error is
  taken from where the tool centre *actually is*. This is the whole difference
  from :class:`~robosimstudio.solvers.ActionEEDelta`, which integrates an
  internal setpoint on purpose -- that is the right shape for a VLA trained
  against an open-loop command stream, and the wrong one for a scripted
  waypoint, where a blocked arm has to keep pushing rather than let the setpoint
  run away from it.
* **The delta is clipped, per axis, per step.** ``output_max`` is 0.05 m and
  0.5 rad in the reference's config, and because the error is re-measured the
  clip is what makes the approach taper: far away the command saturates, near
  the target it shrinks in proportion. The reference's precision comes from that
  taper, not from its gains.
* **The gains set the speed, and the speed is not the clip over the timestep.**
  ``kp=150`` with ``damping_ratio=1`` is a critically damped response, and a
  *saturated* error settles where ``kp*e`` meets ``kd*v``: ``(kp/kd) *
  output_max``, i.e. ``sqrt(kp)/2 * 0.05`` -- about **0.31 m/s**, not the
  1.0 m/s that "0.05 m per 20 Hz step" suggests. The clip bounds the error the
  law sees; the gains turn that into a velocity. A step budget sized on the
  wrong one of those is wrong by a factor of three.

**How the command reaches the arm.** :class:`OscController` computes a task-space
setpoint and hands it to an :class:`~robosimstudio.solvers.ArmIK`, which
solves it onto ``joint_target_q``. It does *not* write joint torques. That is a
deliberate choice and not an approximation of convenience: every arm this package
drives is position-actuated (``<position>`` actuators, with servo gains set on the
model), because ``joint_position`` -- the mode a motion planner like cuRobo emits
into -- reads ``joint_target_q`` and nothing else. A torque realization would need those
actuators switched *off* for the arm, since ``control.joint_f`` lands in
MuJoCo's ``qfrc_applied`` and adds to actuator force rather than replacing it,
so the two modes could not share one model and a scene would have to declare
which one it was for. Landing the law on the position servo keeps one model
serving both, which is what lets the same task variant be evaluated under a
planner and under a policy.

**What "realized on a position servo" has to mean, and what it cannot.** The
tempting reading is to integrate the impedance as if it were the arm --
``v += (kp*e - kd*v)*dt`` -- and command ``measured + v*dt``. That does not work,
and it fails quietly rather than loudly: the servo *is already* the plant, so
anchoring the setpoint one step ahead of the measurement caps the arm at whatever
that servo does with a 6 mm standing error. Measured on a tool-sorting scene, that
was **0.04 m/s** against the reference's 0.31 -- the eight-leg pick ran its whole
budget and never reached the grasp, with every leg ending centimetres short and
nothing in the logs to say why.

So what is reproduced is the impedance's two *observable* consequences, and the
setpoint is integrated in its own right so that it may lead the arm:

* a **rate limit** on the setpoint, at the cruise speed the gains imply, which is
  what makes the arm cross a workcell at the reference's speed rather than at the
  servo's;
* a **lead clamp** on how far the setpoint may get ahead of the measurement, at
  ``output_max`` -- which is the standing error the reference's own law tolerates
  at cruise (``kd*v/kp``, and ``kd*0.306/150 = 0.05`` exactly). It is what bounds
  the force the arm can push with, and what stops a blocked arm from winding its
  command up across the room.

The second-order transient is *not* reproduced: about 0.25 s of settling at the
start of a move, against waypoints the reference holds for three seconds. What is
genuinely given up is contact compliance -- a stiff servo inside the lead clamp
pushes harder than ``kp`` times a penetration would -- which costs nothing on
top-down picks approached from above and would matter for insertion or wiping.
The seam for that case is :meth:`OscController.advance`, which returns a setpoint
and knows nothing about how it is realized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robosimstudio.core.math import (
    quat_from_axis_angle,
    quat_inverse,
    quat_multiply,
    quat_slerp,
    quat_to_axis_angle,
    transform_point,
)
from robosimstudio.solvers.ik.arm_ik import ArmIK

__all__ = [
    "OSC_OUTPUT_MAX",
    "OscController",
    "OscGains",
    "OscSetpoint",
    "OscWaypoints",
    "Waypoint",
    "osc_action",
]

OSC_OUTPUT_MAX = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
"""``output_max`` from robosuite's ``OSC_POSE`` default config [m, m, m, rad, rad, rad].

The per-step, per-axis ceiling on the Cartesian delta a unit action asks for.
Transcribed rather than derived: it is the number the reference's own
``load_controller_config(default_controller="OSC_POSE")`` returns, and the
scripts read ``pos_scale`` and ``rot_scale`` straight off it.
"""


@dataclass(frozen=True)
class OscGains:
    """Task-space impedance gains, in the reference's parameterization.

    robosuite states a stiffness and a damping *ratio* rather than two gains, so
    that is what this states too -- ``kd`` is derived, and cannot drift out of
    step with ``kp`` the way two independently specified numbers do.

    Attributes:
        kp: Position stiffness [1/s²]. The reference's default is 150.
        damping_ratio: 1.0 is critically damped, which is the reference's
            default and the only value it ships a run at.
        kp_rot: Orientation stiffness [1/s²]. 0 uses :attr:`kp`, which is what
            ``OSC_POSE`` does -- its ``kp`` is a scalar applied to all six axes.
    """

    kp: float = 150.0
    damping_ratio: float = 1.0
    kp_rot: float = 0.0

    def __post_init__(self) -> None:
        if self.kp <= 0.0:
            raise ValueError(f"kp must be positive, got {self.kp}")
        if self.damping_ratio <= 0.0:
            raise ValueError(f"damping_ratio must be positive, got {self.damping_ratio}")
        if self.kp_rot < 0.0:
            raise ValueError(f"kp_rot must be non-negative (0 means 'use kp'), got {self.kp_rot}")

    @property
    def kd(self) -> float:
        """Position damping [1/s]: ``2 * sqrt(kp) * damping_ratio``."""
        return 2.0 * float(np.sqrt(self.kp)) * self.damping_ratio

    @property
    def stiffness_rot(self) -> float:
        """Orientation stiffness actually used [1/s²]."""
        return self.kp_rot or self.kp

    @property
    def kd_rot(self) -> float:
        """Orientation damping [1/s]."""
        return 2.0 * float(np.sqrt(self.stiffness_rot)) * self.damping_ratio

    def cruise_speed(self, output_max: float = OSC_OUTPUT_MAX[0]) -> float:
        """Steady speed [m/s] under a saturated position error.

        The velocity at which ``kp * e`` and ``kd * v`` balance, with ``e`` at
        the per-step clip. This -- not ``output_max / control_dt`` -- is how fast
        the reference's arm crosses a workcell, and it is the number to size a
        step budget against.
        """
        return self.kp / self.kd * float(output_max)


class OscSetpoint:
    """The realized half of the law: a rate limit and a lead clamp, over poses.

    Split out from :class:`OscController` because two very different callers need
    the same arithmetic. The controller reads a Warp ``State`` and drives an
    action space; the graph runtime's Cartesian servo works from a published
    snapshot on a worker thread, through a pluggable IK backend, and may not touch
    Warp at all. Sharing the law rather than the plumbing is what keeps the two
    from drifting.

    Both limits are derived from the gains, not tuned. Nothing here reads state,
    so it is testable in arithmetic.

    Args:
        control_dt: One control period [s].
        gains: Task-space impedance gains.
        output_max: Per-axis delta ceiling, as :data:`OSC_OUTPUT_MAX`.
    """

    def __init__(
        self,
        *,
        control_dt: float,
        gains: OscGains | None = None,
        output_max: Sequence[float] = OSC_OUTPUT_MAX,
    ) -> None:
        if control_dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")
        self.control_dt = float(control_dt)
        self.gains = gains or OscGains()
        self.output_max = np.asarray(output_max, dtype=np.float64)
        if self.output_max.shape != (6,):
            raise ValueError(f"output_max needs 6 entries, got {self.output_max.shape[0]}")

        # ``max`` over the axes because a speed cap is a bound on a norm, so an
        # anisotropic ``output_max`` still yields one isotropic limit.
        pos_clip = float(np.max(self.output_max[:3]))
        rot_clip = float(np.max(self.output_max[3:]))
        self.pos_speed = self.gains.kp / self.gains.kd * pos_clip
        """Cruise speed [m/s] -- what the gains do with a saturated error."""
        self.rot_speed = self.gains.stiffness_rot / self.gains.kd_rot * rot_clip
        """Cruise turn rate [rad/s]."""
        self.pos_lead = pos_clip
        """How far ahead of the arm the command may sit [m]."""
        self.rot_lead = rot_clip
        """The same for orientation [rad]."""

        self._setpoint: tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    def reset(self, position: Sequence[float], quat_xyzw: Sequence[float]) -> None:
        """Seed the setpoint at a pose, normally where the arm currently is."""
        self._setpoint = (
            np.asarray(position, dtype=np.float64).copy(),
            np.asarray(quat_xyzw, dtype=np.float64).copy(),
        )

    @property
    def setpoint(self) -> tuple[np.ndarray, np.ndarray]:
        """The last commanded pose."""
        pos, quat = self._setpoint
        return pos.copy(), quat.copy()

    def speed_scale_for(self, speed_mps: float | None) -> float:
        """The factor on ``output_max`` at which this law cruises at ``speed_mps``, capped at 1.

        ``None`` is the law's own cruise (:attr:`pos_speed`) and scales nothing.
        Scaling the ceiling lowers the cruise rate and the command lead by the
        same factor, which is what a Cartesian speed limit is; the cap is there
        because a speed *above* the cruise is not something the law can give by
        raising a ceiling the gains were sized against. This is the arithmetic
        behind ``TaskSpec.servo_speed_mps`` -- a task states the speed its
        graphs were written for, and both the serial servo and the batched one
        turn it into the same factor here rather than each in its own words.

        Raises:
            ValueError: A speed that is not positive.
        """
        if speed_mps is None:
            return 1.0
        speed = float(speed_mps)
        if not speed > 0.0:
            raise ValueError(f"servo speed must be positive [m/s], got {speed_mps!r}")
        return min(1.0, speed / self.pos_speed)

    def advance(
        self,
        measured_pos: Sequence[float],
        measured_quat: Sequence[float],
        goal_pos: Sequence[float],
        goal_quat: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step the setpoint towards the goal and clamp its lead over the arm.

        Two bounds, in this order, and the order matters: clamping before
        stepping would leave room to step again, and a blocked arm would creep.
        """
        set_pos, set_quat = self._setpoint
        dt = self.control_dt

        set_pos = _step_toward(set_pos, goal_pos, self.pos_speed * dt)
        set_quat = _turn_toward(set_quat, goal_quat, self.rot_speed * dt)

        set_pos = _step_toward(measured_pos, set_pos, self.pos_lead)
        set_quat = _turn_toward(measured_quat, set_quat, self.rot_lead)

        self._setpoint = (set_pos, set_quat)
        return set_pos, set_quat


class OscController:
    """One arm under operational-space control, realized on its position servo.

    Args:
        ik: The arm's IK. Supplies the tool-centre frame the error is measured
            in (``ee_body`` plus ``ee_offset`` / ``ee_offset_rot``) and the
            solver the setpoint is realized through.
        control_dt: One control period [s]. The impedance is integrated at this
            rate, so it has to be the rate :meth:`apply` is actually called at
            -- passing the physics timestep instead makes the arm crawl.
        gains: Task-space impedance gains.
        output_max: Per-axis ceiling on the commanded delta, as
            :data:`OSC_OUTPUT_MAX`. Six entries: three translations [m], three
            rotations [rad].
        output_min: The lower ceiling. ``None`` mirrors ``output_max``, which is
            what ``OSC_POSE``'s config does.
        input_max: Upper end of the action range being scaled *from*. Actions
            outside ``[input_min, input_max]`` are clipped to it first, exactly
            as robosuite's ``scale_action`` does.
        input_min: Lower end of that range.
    """

    def __init__(
        self,
        ik: ArmIK,
        *,
        control_dt: float,
        gains: OscGains | None = None,
        output_max: Sequence[float] = OSC_OUTPUT_MAX,
        output_min: Sequence[float] | None = None,
        input_max: float = 1.0,
        input_min: float = -1.0,
    ) -> None:
        if control_dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")
        if input_max <= input_min:
            raise ValueError(f"input_max ({input_max}) must exceed input_min ({input_min})")

        self.ik = ik
        self.control_dt = float(control_dt)
        self.gains = gains or OscGains()
        self.output_max = np.asarray(output_max, dtype=np.float64)
        if self.output_max.shape != (6,):
            raise ValueError(f"output_max needs 6 entries, got {self.output_max.shape[0]}")
        self.output_min = -self.output_max if output_min is None else np.asarray(output_min, dtype=np.float64)
        if self.output_min.shape != (6,):
            raise ValueError(f"output_min needs 6 entries, got {self.output_min.shape[0]}")
        self.input_max = float(input_max)
        self.input_min = float(input_min)

        # The tool body's centre of mass in its own frame, read once. It is a
        # model constant, and reading it off device per step would cost a
        # synchronize for a number that never changes.
        self._body_com = np.asarray(ik.model.body_com.numpy()[ik.ee_body], dtype=np.float64)

        # The impedance's two observable consequences -- see :class:`OscSetpoint`,
        # which the graph runtime's servo shares.
        self.law = OscSetpoint(control_dt=self.control_dt, gains=self.gains, output_max=self.output_max)
        self.law.reset(*ik.home_pose)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, state) -> None:
        """Re-seed the setpoint from where the arm actually is.

        Called on every environment reset. Without it the first action of an
        episode inherits the last setpoint of the previous one, which on a scene
        that re-poses rather than rebuilds is a command aimed at the old layout --
        and since the setpoint is what leads the arm, the first step of the new
        episode would lunge at it.
        """
        self.law.reset(*self.measure(state)[:2])

    @property
    def setpoint(self) -> tuple[np.ndarray, np.ndarray]:
        """The last commanded tool-centre pose, for readouts and recording."""
        return self.law.setpoint

    @property
    def pos_speed(self) -> float:
        """Cruise speed [m/s] the gains imply."""
        return self.law.pos_speed

    @property
    def pos_lead(self) -> float:
        """How far ahead of the arm the command may sit [m]."""
        return self.law.pos_lead

    @property
    def rot_lead(self) -> float:
        """The same for orientation [rad]."""
        return self.law.rot_lead

    # -------------------------------------------------------------- measurement
    def measure(self, state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The tool centre's world pose and twist: ``(pos, quat, linear, angular)``.

        The twist is the *body's* spatial velocity moved to the tool-centre
        point, not the body's own: ``State.body_qd`` carries the linear velocity
        of the centre of mass, and the tool centre sits some way off it (0.1 m
        along the wrist for a YAM). Skipping the ``omega x r`` term would report
        a wrist that is purely rotating as stationary, and the damping term would
        have nothing to damp.
        """
        body_q, pos, quat = self._pose(state)
        twist = np.asarray(state.body_qd.numpy()[self.ik.ee_body], dtype=np.float64)
        linear, angular = twist[:3], twist[3:]
        com = transform_point(body_q, self._body_com)
        return pos, quat, linear + np.cross(angular, pos - com), angular

    def _pose(self, state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(body_q, pos, quat)``: :meth:`measure` without the twist.

        One host copy of ``body_q`` and none of ``body_qd``, for the per-step
        callers that discard the twist anyway.
        """
        body_q = np.asarray(state.body_q.numpy()[self.ik.ee_body], dtype=np.float64)
        return body_q, transform_point(body_q, self.ik.ee_offset), quat_multiply(body_q[3:7], self.ik.ee_offset_rot)

    # ------------------------------------------------------------------ the law
    def scale_action(self, action: Sequence[float]) -> np.ndarray:
        """Map a normalized action onto the output range, as robosuite does.

        The affine map from ``[input_min, input_max]`` onto
        ``[output_min, output_max]``, with the input clipped to its range first.
        This is ``Controller.scale_action`` verbatim: a unit action is exactly
        ``output_max``, and a policy trained against the reference's action
        space needs the same 0.05 m per unit here as it saw there.
        """
        flat = np.asarray(action, dtype=np.float64).reshape(-1)
        if flat.shape[0] != 6:
            raise ValueError(f"an OSC action is 6 numbers (3 translation, 3 rotation), got {flat.shape[0]}")
        span = (self.output_max - self.output_min) / (self.input_max - self.input_min)
        offset = self.output_min - span * self.input_min
        return span * np.clip(flat, self.input_min, self.input_max) + offset

    def goal(self, action: Sequence[float], state) -> tuple[np.ndarray, np.ndarray]:
        """Where a scaled action asks the tool centre to be: measured plus delta.

        The rotation delta is applied on the left, i.e. about world axes -- the
        convention ``set_goal_orientation`` uses and the one the reference's
        waypoint quaternions are composed in.
        """
        _, pos, quat = self._pose(state)
        return self._goal_from(pos, quat, action)

    def _goal_from(self, pos: np.ndarray, quat: np.ndarray, action: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`goal` from a pose already measured."""
        delta = self.scale_action(action)
        return pos + delta[:3], quat_multiply(quat_from_axis_angle(delta[3:], float(np.linalg.norm(delta[3:]))), quat)

    def advance(self, action: Sequence[float], state) -> tuple[np.ndarray, np.ndarray]:
        """Advance the setpoint one control period and return it.

        Two bounds, in this order, and the order matters:

        1. **Rate limit.** The setpoint travels towards the goal at no more than
           the cruise speed the gains imply. Integrated from the *setpoint*, not
           from the measurement -- that is what lets it lead the arm, and without
           the lead a position servo never reaches the commanded speed at all.
        2. **Lead clamp.** Having moved, the setpoint is pulled back to within
           ``output_max`` of where the arm actually is. This is the whole of the
           measured feedback in the realized command: a blocked arm holds its
           setpoint one clamp away instead of winding it across the room, and the
           force the arm pushes with is bounded by what the servo makes of a
           5 cm error.

        Applying the clamp *after* the step rather than before is what keeps a
        blocked arm from creeping: clamping first would leave room to step again.
        """
        _, pos, quat = self._pose(state)
        goal_pos, goal_quat = self._goal_from(pos, quat, action)
        return self.law.advance(pos, quat, goal_pos, goal_quat)

    def apply(self, action: Sequence[float], control, state) -> None:
        """Advance the law and solve the setpoint onto this arm's joint targets."""
        pos, quat = self.advance(action, state)
        self.ik.set_target(pos, quat)
        self.ik.solve_into(state.joint_q, control.joint_target_q)


def _step_toward(origin: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    """``target``, or the point ``limit`` metres from ``origin`` towards it."""
    offset = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    distance = float(np.linalg.norm(offset))
    if distance <= limit or distance < 1.0e-12:
        return np.asarray(target, dtype=np.float64).copy()
    return np.asarray(origin, dtype=np.float64) + offset * (limit / distance)


def _turn_toward(origin: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    """``target``, or the orientation ``limit`` radians from ``origin`` towards it."""
    angle = 2.0 * float(np.arccos(np.clip(abs(float(np.dot(origin, target))), -1.0, 1.0)))
    if angle <= limit or angle < 1.0e-9:
        return np.asarray(target, dtype=np.float64).copy()
    return quat_slerp(origin, target, limit / angle)


def osc_action(
    measured_pos: Sequence[float],
    measured_quat: Sequence[float],
    target_pos: Sequence[float],
    target_quat: Sequence[float],
    *,
    output_max: Sequence[float] = OSC_OUTPUT_MAX,
) -> np.ndarray:
    """The normalized 6-vector that aims the tool centre at a waypoint.

    A transcription of the reference scripts' own ``osc_action``: the remaining
    error, divided by the per-step ceiling, clipped into the unit cube. Saturated
    while the waypoint is far away and proportional once it is within one step's
    reach, which is the taper the whole approach is built on.

    Returns:
        ``[dx, dy, dz, drx, dry, drz]`` in ``[-1, 1]``. The caller appends its
        own gripper command to make a full arm action.
    """
    ceiling = np.asarray(output_max, dtype=np.float64)
    position_error = np.asarray(target_pos, dtype=np.float64) - np.asarray(measured_pos, dtype=np.float64)
    rotation_error = quat_to_axis_angle(quat_multiply(target_quat, quat_inverse(measured_quat)))
    return np.concatenate(
        [
            np.clip(position_error / ceiling[:3], -1.0, 1.0),
            np.clip(rotation_error / ceiling[3:], -1.0, 1.0),
        ]
    )


@dataclass(frozen=True)
class Waypoint:
    """One leg of a scripted move: a pose to aim at, and how long to hold it.

    Attributes:
        label: What the leg is for, for logs. The reference prints these and
            they are how a failed run is read.
        position: Target tool-centre position [m].
        quat: Target tool-centre orientation, xyzw.
        grip: Gripper command over the hold, 0 open .. 1 closed.
        steps: Control steps to hold it. The reference holds 45-80 at 20 Hz;
            a hold is not a convergence test, it is a fixed budget, and a leg
            that has not arrived by the end of it simply moves on.
    """

    label: str
    position: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    grip: float
    steps: int

    @property
    def target(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.position, dtype=np.float64), np.asarray(self.quat, dtype=np.float64)


class OscWaypoints:
    """A waypoint list, stepped one OSC action at a time.

    The reference's ``go_to`` loop, as an object a routine can hold instead of a
    ``for`` loop it cannot yield out of: an environment is stepped by whoever
    owns the rollout, so a scripted policy has to be able to hand back one
    action and be called again.

    Args:
        waypoints: The legs, in order.
        output_max: The per-step ceiling :func:`osc_action` normalizes against.
            Must match the controller's, or the actions are scaled twice.
    """

    def __init__(self, waypoints: Sequence[Waypoint] = (), *, output_max: Sequence[float] = OSC_OUTPUT_MAX) -> None:
        self.waypoints = list(waypoints)
        self.output_max = np.asarray(output_max, dtype=np.float64)
        self.index = 0
        self.frame = 0

    def load(self, waypoints: Sequence[Waypoint]) -> None:
        """Replace the list and start it from the beginning."""
        self.waypoints = list(waypoints)
        self.index = 0
        self.frame = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def current(self) -> Waypoint | None:
        return None if self.done else self.waypoints[self.index]

    @property
    def finishing(self) -> bool:
        """Whether the step about to be taken is the last of the current leg."""
        leg = self.current
        return leg is not None and self.frame + 1 >= leg.steps

    def act(self, measured_pos: Sequence[float], measured_quat: Sequence[float]) -> np.ndarray:
        """One control step's arm action: ``[dx..drz, grip]``.

        Advances the leg counter as a side effect, so this is called exactly
        once per environment step. When the list is exhausted it holds the last
        pose with a zero delta rather than raising -- an episode outlives its
        script, and the steps after the last waypoint are what let a released
        object settle.
        """
        leg = self.current
        if leg is None:
            grip = self.waypoints[-1].grip if self.waypoints else 0.0
            return np.append(np.zeros(6), grip)

        target_pos, target_quat = leg.target
        delta = osc_action(measured_pos, measured_quat, target_pos, target_quat, output_max=self.output_max)
        self.frame += 1
        if self.frame >= leg.steps:
            self.frame = 0
            self.index += 1
        return np.append(delta, leg.grip)
