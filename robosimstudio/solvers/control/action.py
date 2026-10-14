# SPDX-License-Identifier: Apache-2.0
"""Action interfaces: what a policy hands an environment, over explicit DOFs.

Four shapes, all over one or more arms, each arm contributing its joints plus
one 0 (open) .. 1 (closed) gripper command:

* :class:`ActionJointPosition` -- ``[q_0 .. q_{n-1}, grip]`` per arm. Absolute
  joint targets. The canonical space for replay, scripted routines and the
  graph-policy connector, because it is the one that round-trips exactly, and the space a
  motion planner emits into.
* :class:`ActionEEDelta` -- ``[dx, dy, dz, drx, dry, drz, grip]`` per arm. A
  delta in world axes, integrated onto an *internal* setpoint and solved through
  :class:`~robosimstudio.solvers.ArmIK`.
* :class:`ActionOsc` -- the same seven numbers, read the reference benchmark's
  way: normalized, clipped per step, and applied to the tool centre's *measured*
  pose through a task-space impedance. See :mod:`robosimstudio.solvers.motion.osc`
  for why that is a different action space rather than a variation on the last
  one.
* :class:`ActionEEPoseAbs` -- ``[x, y, z, qx, qy, qz, qw, grip]`` per arm.
  Absolute Cartesian, for waypoint-shaped policies.

All four know only about DOF indices, an :class:`ArmIK` and a
:class:`~robosimstudio.solvers.Gripper`, never about a task. Which one a
scene uses is a spec field the toolkit layer sets.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_inverse, quat_multiply, quat_to_mat
from robosimstudio.registry import register_alias
from robosimstudio.solvers.ik.arm_ik import ArmIK
from robosimstudio.solvers.motion.gripper import Gripper
from robosimstudio.solvers.motion.osc import OSC_OUTPUT_MAX, OscController, OscGains, osc_action

__all__ = [
    "CONTROL_MODES",
    "EE_DELTA_CONVENTIONS",
    "ActionEEDelta",
    "ActionEEPoseAbs",
    "ActionJointPosition",
    "ActionOsc",
    "ActionSpace",
    "UnsupportedMode",
    "control_mode_class",
    "make_action_space",
    "make_action_spaces",
    "register_control_mode",
    "zero_action",
]

CONTROL_MODES: dict[str, str] = {
    "joint_position": "robosimstudio.solvers.control.action.ActionJointPosition",
    "ee_delta": "robosimstudio.solvers.control.action.ActionEEDelta",
    "osc_pose": "robosimstudio.solvers.control.action.ActionOsc",
    "ee_pose_abs": "robosimstudio.solvers.control.action.ActionEEPoseAbs",
}
"""Mode name -> dotted path, in the order an environment builds them.

Open on the same terms as every other implementation slot here: an alias may be
added with :func:`register_control_mode`, and **a dotted path needs no
registration at all**, so a control mode written outside this tree is nameable
by whoever wrote it. Anything answering the four methods of
:class:`~robosimstudio.core.env.ActionSpace` is drivable.

A mode that cannot serve an embodiment raises :class:`UnsupportedMode` from its
constructor and :func:`make_action_spaces` leaves it out; any other error
propagates and fails the build.
"""


def register_control_mode(mode: str, cls: str | type) -> str:
    """Register a control mode under a name. Identical re-registration is accepted.

    A mode that cannot serve an embodiment raises :class:`UnsupportedMode` and is
    skipped by :func:`make_action_spaces`; any other error propagates.

    Raises:
        ValueError: If ``mode`` is taken by a different class.
    """
    return register_alias(CONTROL_MODES, "control mode", mode, cls)


def control_mode_class(mode: str) -> type:
    """The class a mode names: a registered alias, or a dotted path.

    A dotted path resolves with no registration, which is the extensibility
    contract -- the same one
    :func:`~robosimstudio.specs.physics.get_backend` states for solvers.
    """
    path = CONTROL_MODES.get(mode) or (mode if "." in mode else "")
    if not path:
        raise ValueError(
            f"unknown control mode {mode!r}: neither a registered alias ({list(CONTROL_MODES)}) nor a dotted path"
        )
    module_name, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), name)


"""Every mode :func:`make_action_space` builds, in a stable order.

An environment builds *all* of them rather than the one a spec names, because
they are thin wrappers over the same :class:`ArmIK` and
:class:`~robosimstudio.solvers.Gripper` objects and a script legitimately
wants more than one within an episode: a planner's joint trajectory to get across
the workcell, then normalized Cartesian deltas for the part that needs feedback.
See :meth:`~robosimstudio.core.env.EnvBase.use_control`.
"""


class UnsupportedMode(ValueError):
    """This embodiment or environment cannot do this control mode.

    The one failure :func:`make_action_spaces` skips a mode for. Distinct from a
    plain ``ValueError`` so a misconfigured option -- a misspelled convention, a
    backwards ``gripper_range`` -- still raises with its own message instead of
    surfacing later as a mode that is merely missing.
    """


class ActionSpace:
    """Base: a flat float vector applied onto a control's target arrays.

    Args:
        arms: One ``(ArmIK, Gripper | None)`` pair per arm, in the order their
            slices appear in the action vector.
        by_dof: Whether joint targets index by ``joint_qd`` rather than
            ``joint_q``. See :meth:`ArmIK.set_joint_targets`.
    """

    per_arm: int = 0
    """Width of one arm's slice of the action vector."""

    def __init__(self, arms: Sequence[tuple[ArmIK, Gripper | None]], *, by_dof: bool = False) -> None:
        if not arms:
            raise ValueError("an action space needs at least one arm")
        self.arms = list(arms)
        self.by_dof = bool(by_dof)

    @property
    def action_dim(self) -> int:
        return len(self.arms) * self.per_arm

    def _split(self, action: Sequence[float]) -> list[np.ndarray]:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.shape[0] != self.action_dim:
            raise ValueError(f"action has {flat.shape[0]} entries, expected {self.action_dim}")
        return [flat[i * self.per_arm : (i + 1) * self.per_arm] for i in range(len(self.arms))]

    def reset(self, state) -> None:
        """Re-seed any internal setpoint from the scene's current pose."""

    def apply(self, action: Sequence[float], control, state) -> None:
        raise NotImplementedError

    def hold_action(self, state) -> np.ndarray:
        """An action that leaves the scene where it is, in this space's layout.

        Each space answers for itself rather than a helper switching on type,
        because "hold" means something different in each and the differences are
        load bearing -- see :class:`ActionOsc.hold_action` for the one that is not
        a zero vector. The default is a zero vector, which is right for any space
        whose entries are deltas.
        """
        del state
        return np.zeros(self.action_dim, dtype=np.float32)


class ActionJointPosition(ActionSpace):
    """``[q_0 .. q_{n-1}, grip]`` per arm: absolute joint position targets."""

    def __init__(self, arms: Sequence[tuple[ArmIK, Gripper | None]], *, by_dof: bool = False) -> None:
        super().__init__(arms, by_dof=by_dof)
        widths = {len(ik.arm_coords) for ik, _ in self.arms}
        if len(widths) != 1:
            raise UnsupportedMode(f"joint_position needs arms of equal width, got {sorted(widths)}")
        self.per_arm = widths.pop() + 1

    def apply(self, action: Sequence[float], control, state) -> None:
        del state
        for (ik, gripper), values in zip(self.arms, self._split(action), strict=True):
            ik.set_joint_targets(values[:-1], control.joint_target_q, by_dof=self.by_dof)
            if gripper is not None:
                gripper.command = float(values[-1])
                gripper.apply(control.joint_target_q)

    def hold_action(self, state) -> np.ndarray:
        """The live configuration, which is where the arm already is."""
        joint_q = state.joint_q.numpy()
        parts = []
        for ik, gripper in self.arms:
            parts.append(np.array([joint_q[c] for c in ik.arm_coords], dtype=np.float32))
            parts.append(np.array([gripper.command if gripper is not None else 0.0], dtype=np.float32))
        return np.concatenate(parts)

    def target_action(self, control) -> np.ndarray:
        """The action equivalent to the targets already in ``control``.

        The inverse of :meth:`apply`, and the reason it exists is recording a
        front end that writes ``joint_target_q`` directly -- a teleop session,
        where the panels drive IK into the target vector and no action was ever
        formed. Reading the *target* rather than
        :meth:`hold_action`'s measured ``joint_q`` is what makes the recording
        replayable: a position servo runs behind its setpoint, so replaying
        measurements as commands reproduces a trajectory that lags the one that
        was recorded, once per step, cumulatively.

        Args:
            control: The control whose ``joint_target_q`` to read back.

        Returns:
            One arm's joints then its gripper command, per arm -- this space's
            own layout, so it round-trips through :meth:`apply`.
        """
        target_q = control.joint_target_q.numpy()
        parts = []
        for ik, gripper in self.arms:
            # Whichever indexing set_joint_targets wrote with, read the same one.
            coords = ik.arm_dofs if self.by_dof else ik.arm_coords
            parts.append(np.asarray(target_q[coords], dtype=np.float32))
            parts.append(np.array([gripper.command if gripper is not None else 0.0], dtype=np.float32))
        return np.concatenate(parts)


class ActionEEPoseAbs(ActionSpace):
    """``[x, y, z, qx, qy, qz, qw, grip]`` per arm: absolute Cartesian targets."""

    per_arm = 8

    def apply(self, action: Sequence[float], control, state) -> None:
        for (ik, gripper), values in zip(self.arms, self._split(action), strict=True):
            ik.set_target(values[:3], values[3:7])
            ik.solve_into(state.joint_q, control.joint_target_q)
            if gripper is not None:
                gripper.command = float(values[7])
                gripper.apply(control.joint_target_q)

    def hold_action(self, state) -> np.ndarray:
        """The pose the hand is at, which is an absolute command to stay there."""
        parts = []
        for ik, gripper in self.arms:
            pos, quat = ik.ee_pose(state)
            parts.append(np.concatenate([pos, quat, [gripper.command if gripper is not None else 0.0]]))
        return np.concatenate(parts).astype(np.float32)


class _RangedGripper:
    """The gripper number of a Cartesian space: an interval mapped onto 0..1.

    Shared by :class:`ActionEEDelta` and :class:`ActionOsc` so the open/closed
    convention lives in one place; two copies could drift apart without either
    raising.
    """

    gripper_range: tuple[float, float] = (0.0, 1.0)

    def _set_gripper_range(self, gripper_range: tuple[float, float]) -> None:
        low, high = (float(v) for v in gripper_range)
        if high <= low:
            raise ValueError(f"gripper_range must be increasing, got {gripper_range}")
        self.gripper_range = (low, high)

    def grip_command(self, value: float, gripper: Gripper | None = None) -> float:
        """Map a gripper command out of :attr:`gripper_range` onto 0 open .. 1 closed.

        **Which end of the range closes is the hand's to say**, not this space's.
        A benchmark's gripper controller maps a normalized action linearly onto
        the joint's own ``[low, high]``, so a hand whose fingers *open* with
        increasing travel is driven opposite to one whose fingers rotate shut --
        and the two SimplerEnv robots are exactly that pair. Getting it wrong is
        silent in every direction that matters: a scripted routine asking for
        "closed" still closes, and only a trained checkpoint asking for -1
        discovers that the hand opens.
        """
        low, high = self.gripper_range
        normalized = float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))
        if gripper is not None and not gripper.action_closed_at_high:
            return 1.0 - normalized
        return normalized

    def grip_action(self, command: float, gripper: Gripper | None = None) -> float:
        """The inverse of :meth:`grip_command`: 0..1 back into the action's range."""
        low, high = self.gripper_range
        fraction = float(np.clip(command, 0.0, 1.0))
        if gripper is not None and not gripper.action_closed_at_high:
            fraction = 1.0 - fraction
        return float(low + fraction * (high - low))


EE_DELTA_CONVENTIONS = ("align", "align2", "tool")
"""How a delta pose composes onto the standing setpoint.

Named after the reference implementation each is transcribed from
(ManiSkill2's ``PDEEPoseController.compute_target_pose``), because these are not
three shades of the same thing -- a policy trained against one and stepped
through another moves differently on every step that rotates:

``"align"``
    Rotate in the *frame's* axes, translate in the frame's axes. The default,
    and what ``ee_align`` means: ``p' = p + R_f dpos``, ``q' = (q_f dq q_f^-1) q``.
``"align2"``
    The same rotation, but taken about the tool centre's **measured** position
    rather than about the setpoint's. Identical while the arm is caught up and
    different exactly when it is not, which is why the reference has both.
``"tool"``
    Both in the tool's own frame -- ``ee`` upstream, ``prev_pose * delta``:
    ``p' = p + R dpos``, ``q' = q dq``, the translation along the orientation
    *before* this step's rotation. Frame-independent, so
    :attr:`ActionEEDelta.frame_bodies` does not enter.
"""


class ActionEEDelta(_RangedGripper, ActionSpace):
    """``[dx, dy, dz, drx, dry, drz, grip]`` per arm: a delta pose, integrated.

    The setpoint is *internal*, integrated from deltas rather than read back off
    the arm each step, which is what the VLA policies this serves were trained
    against: reading back would let a blocked arm drag the setpoint with it and
    the policy would never see its command take effect.

    **Which axes the six numbers are in is a property of the policy, not of the
    robot**, so it is a constructor argument rather than a convention baked in
    here. World axes with :data:`EE_DELTA_CONVENTIONS`' ``"align"`` is what this
    class has always done and stays the default. A checkpoint trained against a
    SAPIEN/ManiSkill benchmark emits deltas in the robot's **base** frame, and
    both of those robots stand yawed 180 degrees from world -- so handing such a
    policy world-frame deltas drives the arm backwards along x and y, at full
    speed, on the very first step. That failure looks like a policy that cannot
    reach rather than like a frame error, which is why the axes are stated.

    Args:
        arms: As :class:`ActionSpace`.
        pos_scale: Metres per unit of translation delta.
        rot_scale: Radians per unit of rotation delta.
        frame_bodies: Per arm, the body whose orientation the deltas are
            expressed in, or -1 for world axes. A base-mounted arm names its
            base link, which is welded and therefore constant -- but read from
            ``body_q`` all the same, so a scene that moves the mount needs no
            second statement of where it went.
        convention: One of :data:`EE_DELTA_CONVENTIONS`.
        gripper_range: The interval the gripper command arrives in, mapped onto
            the :class:`~robosimstudio.solvers.Gripper`'s own 0 (open) to 1
            (closed). ``(0.0, 1.0)`` is this package's convention;
            ``(-1.0, 1.0)`` is what a ManiSkill-trained checkpoint emits.

            Deliberately not "which end closes": both of the SimplerEnv robots
            take +1 as *one* end of a normalized joint range and which end that
            closes the hand differs between them, so the sign lives in the
            hand's own ``finger_open_q`` / ``finger_closed_q`` where it is a
            measurement, and this stays a plain interval map.
        use_target: Compose each delta onto the **standing setpoint** rather
            than onto where the hand measurably is.

            The reference benchmarks ship both and it is not a detail. With a
            target, a blocked arm still sees its commands accumulate, which is
            what an open-loop action stream expects; without one, the delta is
            applied to the arm's actual pose each step, so a command the arm
            failed to execute is simply lost. Two robots in one benchmark can
            differ here -- SimplerEnv drives its WidowX
            ``arm_pd_ee_target_delta_pose_align2`` and its Google robot
            ``arm_pd_ee_delta_pose_align``, and a policy trained against the
            second, stepped through the first, integrates error it was never
            trained to see.
        gripper_delta: The gripper number is a **change** to the standing
            command rather than an absolute position.

            Also a per-robot fact in the same benchmark: the Bridge tasks use
            ``gripper_pd_joint_pos`` (absolute) and the fractal ones
            ``gripper_pd_joint_target_delta_pos``. RT-1's own wrapper emits a
            *relative* gripper action for the Google robot and an absolute one
            for the WidowX, so reading one as the other turns a policy that
            nudges its jaw closed into one that slams it to a limit every step.
        by_dof: As :class:`ActionSpace`.
    """

    per_arm = 7

    def __init__(
        self,
        arms: Sequence[tuple[ArmIK, Gripper | None]],
        *,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        frame_bodies: Sequence[int] = (),
        convention: str = "align",
        gripper_range: tuple[float, float] = (0.0, 1.0),
        use_target: bool = True,
        gripper_delta: bool = False,
        by_dof: bool = False,
    ) -> None:
        super().__init__(arms, by_dof=by_dof)
        if convention not in EE_DELTA_CONVENTIONS:
            raise ValueError(f"unknown ee_delta convention {convention!r}; have {list(EE_DELTA_CONVENTIONS)}")
        self._set_gripper_range(gripper_range)
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.convention = convention
        self.use_target = bool(use_target)
        self.gripper_delta = bool(gripper_delta)
        self.frame_bodies = list(frame_bodies) if frame_bodies else [-1] * len(self.arms)
        if len(self.frame_bodies) != len(self.arms):
            raise ValueError(f"frame_bodies has {len(self.frame_bodies)} entries for {len(self.arms)} arms")
        self._setpoints: list[tuple[np.ndarray, np.ndarray]] = [ik.home_pose for ik, _ in self.arms]

    def reset(self, state) -> None:
        """Re-seed each setpoint from where the hand actually is."""
        self._setpoints = [ik.ee_pose(state) for ik, _ in self.arms]

    @property
    def setpoints(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The current per-arm Cartesian setpoints, for readouts and recording."""
        return [(p.copy(), q.copy()) for p, q in self._setpoints]

    def _frame_quat(self, arm: int, state) -> np.ndarray:
        """The orientation the deltas are expressed in, as an xyzw quaternion."""
        body = self.frame_bodies[arm]
        if body < 0 or state.body_q is None:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return np.asarray(state.body_q.numpy()[body][3:7], dtype=np.float64)

    def apply(self, action: Sequence[float], control, state) -> None:
        for i, ((ik, gripper), values) in enumerate(zip(self.arms, self._split(action), strict=True)):
            # The base the delta composes onto. With ``use_target`` that is the
            # standing setpoint, so commands accumulate whether or not the arm
            # kept up; without it, it is where the hand measurably is, so a
            # command the arm failed to execute is not carried forward.
            pos, quat = self._setpoints[i] if self.use_target else ik.ee_pose(state)
            dpos = np.asarray(values[:3], dtype=np.float64) * self.pos_scale
            rot_vec = np.asarray(values[3:6], dtype=np.float64) * self.rot_scale
            angle = float(np.linalg.norm(rot_vec))

            if self.convention == "tool":
                # Both halves in the tool's own frame, so the mount never enters:
                # ``prev_pose * delta_pose`` upstream, which translates along the
                # *pre-step* orientation and then rotates. Translating along the
                # rotated frame instead swings every combined move off its axis.
                pos = pos + quat_to_mat(quat) @ dpos
                if angle > 1.0e-9:
                    quat = quat_multiply(quat, quat_from_axis_angle(rot_vec / angle, angle))
                    quat = quat / np.linalg.norm(quat)
            else:
                frame = self._frame_quat(i, state)
                # The delta rotation, taken out of the frame's axes into world:
                # q_f dq q_f^-1, which is what makes "rotate about the base's z"
                # mean the same thing wherever the base is pointing.
                delta = None
                if angle > 1.0e-9:
                    delta = quat_multiply(
                        quat_multiply(frame, quat_from_axis_angle(rot_vec / angle, angle)), quat_inverse(frame)
                    )
                if self.convention == "align2" and delta is not None:
                    # About the tool centre's measured position rather than the
                    # setpoint's: identical while the arm has caught up, and the
                    # whole difference between the two conventions when it has not.
                    measured, _ = ik.ee_pose(state)
                    pos = measured + quat_to_mat(delta) @ (pos - measured)
                pos = pos + quat_to_mat(frame) @ dpos
                if delta is not None:
                    quat = quat_multiply(delta, quat)
                    quat = quat / np.linalg.norm(quat)

            self._setpoints[i] = (pos, quat)
            ik.set_target(pos, quat)
            ik.solve_into(state.joint_q, control.joint_target_q)
            if gripper is not None:
                gripper.command = self._grip(values[6], gripper)
                gripper.apply(control.joint_target_q)

    def _grip(self, value: float, gripper: Gripper) -> float:
        """This step's gripper command, absolute or accumulated.

        The delta form reproduces ``pd_joint_target_delta_pos``: the normalized
        number scales onto **half** the jaw's coordinate range and is added to
        the standing target, then clipped -- so a full-scale command takes two
        steps to cross the stroke, which is what makes a policy's small nudges
        readable as nudges rather than as slams to a limit.
        """
        if not self.gripper_delta:
            return self.grip_command(value, gripper)
        low, high = self.gripper_range
        span = high - low
        step = (2.0 * (float(np.clip(value, low, high)) - low) / span - 1.0) if span else 0.0
        if not gripper.action_closed_at_high:
            step = -step
        return float(np.clip(gripper.command + 0.5 * step, 0.0, 1.0))

    def hold_action(self, state) -> np.ndarray:
        """A zero delta, which leaves the internal setpoint standing."""
        del state
        out = np.zeros(self.action_dim, dtype=np.float32)
        for i, (_, gripper) in enumerate(self.arms):
            if self.gripper_delta:
                # Zero *change*, which for a delta gripper is the midpoint of
                # the range rather than either end of it.
                low, high = self.gripper_range
                out[i * self.per_arm + 6] = 0.5 * (low + high)
            else:
                out[i * self.per_arm + 6] = self.grip_action(gripper.command if gripper is not None else 0.0, gripper)
        return out


class ActionOsc(_RangedGripper, ActionSpace):
    """``[dx, dy, dz, drx, dry, drz, grip]`` per arm, robosuite ``OSC_POSE``'s space.

    The same width as :class:`ActionEEDelta` and a different contract, which is
    the point of having both. Here the six deltas are *normalized* -- a unit
    action means ``output_max``, 0.05 m or 0.5 rad -- and they are applied to
    where the tool centre measurably is, through the impedance of
    :class:`~robosimstudio.solvers.OscController`. There the deltas are in
    metres and accumulate on a setpoint the arm never gets a vote on.

    Which to use is a question about the policy, not the robot. A VLA trained
    against robosuite, LIBERO or a ManiSkill ``pd_ee_delta_pose`` task emits into
    this one; one trained against an open-loop command stream emits into the
    other. Handing either policy the wrong one costs a factor of ``output_max``
    in every step it takes.

    Args:
        arms: As :class:`ActionSpace`.
        control_hz: The rate :meth:`apply` is called at [Hz]. The impedance is
            integrated over ``1 / control_hz``, so this must be the environment's
            control rate and not its physics rate.
        gains: Task-space impedance gains, defaulting to the reference's
            ``kp=150``, critically damped.
        output_max: Per-axis ceiling on one step's delta, as
            :data:`~robosimstudio.solvers.motion.osc.OSC_OUTPUT_MAX`.
        gripper_range: The interval the gripper command arrives in, mapped onto
            the :class:`~robosimstudio.solvers.Gripper`'s own 0..1. Left at
            ``(0.0, 1.0)`` -- this package's convention, so a run can swap
            control modes without its gripper commands changing meaning. Set it
            to ``(-1.0, 1.0)`` to take robosuite's convention unaltered, which is
            what a checkpoint trained there emits: -1 open, +1 closed.
        by_dof: As :class:`ActionSpace`.
    """

    per_arm = 7

    def __init__(
        self,
        arms: Sequence[tuple[ArmIK, Gripper | None]],
        *,
        control_hz: float,
        gains: OscGains | None = None,
        output_max: Sequence[float] = OSC_OUTPUT_MAX,
        gripper_range: tuple[float, float] = (0.0, 1.0),
        by_dof: bool = False,
    ) -> None:
        super().__init__(arms, by_dof=by_dof)
        if control_hz <= 0.0:
            raise UnsupportedMode(f"osc_pose needs the environment's control rate, got control_hz={control_hz}")
        self._set_gripper_range(gripper_range)
        self.controllers = [
            OscController(ik, control_dt=1.0 / float(control_hz), gains=gains, output_max=output_max)
            for ik, _ in self.arms
        ]

    def reset(self, state) -> None:
        for controller in self.controllers:
            controller.reset(state)

    @property
    def setpoints(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each arm's last commanded tool-centre pose, for readouts and recording."""
        return [controller.setpoint for controller in self.controllers]

    def apply(self, action: Sequence[float], control, state) -> None:
        for controller, ((_, gripper), values) in zip(
            self.controllers, zip(self.arms, self._split(action), strict=True), strict=True
        ):
            controller.apply(values[:6], control, state)
            if gripper is not None:
                gripper.command = self.grip_command(values[6], gripper)
                gripper.apply(control.joint_target_q)

    def hold_action(self, state) -> np.ndarray:
        """Re-aim at the pose last *commanded* -- not a zero delta.

        A zero delta reads as "put the goal where the tool already is", which asks
        a position servo for no correcting effort at all: a loaded arm sags, the
        next step anchors the goal to the sag, and the hand walks away over a
        settle. Aiming at the standing setpoint is what the reference's own
        ``go_to`` does on every step of a hold, and it holds.
        """
        parts = []
        for controller, (_, gripper) in zip(self.controllers, self.arms, strict=True):
            pos, quat, _, _ = controller.measure(state)
            target_pos, target_quat = controller.setpoint
            delta = osc_action(pos, quat, target_pos, target_quat, output_max=controller.output_max)
            parts.append(np.append(delta, self.grip_action(gripper.command if gripper is not None else 0.0, gripper)))
        return np.concatenate(parts).astype(np.float32)


def make_action_space(
    mode: str,
    arms: Sequence[tuple[ArmIK, Gripper | None]],
    *,
    by_dof: bool = False,
    control_hz: float = 0.0,
    **kwargs,
) -> ActionSpace:
    """Build the action space named by ``mode``.

    Args:
        mode: ``"joint_position"``, ``"ee_delta"``, ``"osc_pose"`` or
            ``"ee_pose_abs"``.
        arms: One ``(ArmIK, Gripper | None)`` pair per arm.
        by_dof: Whether joint targets index by ``joint_qd``.
        control_hz: The environment's control rate, passed on only to the modes
            that integrate over it. Named here rather than left to ``**kwargs``
            so a caller can hand it over unconditionally: an environment always
            knows its rate, and which modes care is this factory's business.
        **kwargs: Forwarded to the chosen class.
    """
    cls = control_mode_class(mode)
    # ``control_hz`` goes only to a class that declares it: handing the rate to
    # a class that does not take one is a ``TypeError``, which would cost every
    # environment the joint mode. Asked of the signature, so a mode registered
    # from outside states its own needs the same way.
    wants_rate = "control_hz" in inspect.signature(cls).parameters
    return cls(arms, by_dof=by_dof, **({"control_hz": control_hz} if wants_rate else {}), **kwargs)


def make_action_spaces(
    arms: Sequence[tuple[ArmIK, Gripper | None]],
    *,
    modes: Sequence[str] = CONTROL_MODES,
    by_dof: bool = False,
    control_hz: float = 0.0,
    options: dict[str, dict[str, Any]] | None = None,
    **kwargs,
) -> dict[str, ActionSpace]:
    """Build every mode an environment should be able to switch between.

    A mode that cannot be built for this embodiment is left out rather than
    raising: :class:`ActionJointPosition` needs arms of equal width, and a scene
    pairing a 7-DOF arm with a 6-DOF one legitimately supports the Cartesian modes
    and not the joint one. Asking for such a mode later fails then, naming what is
    available -- which is a better error than a scene that would not compile.

    Only :class:`UnsupportedMode` is read that way. Any other error -- a bad
    option, an argument the class does not take -- raises here with its own
    message, rather than costing the mode and reappearing later as "this scene
    has no control mode".

    Args:
        arms: One ``(ArmIK, Gripper | None)`` pair per arm.
        modes: Which modes to try. Defaults to :data:`CONTROL_MODES`.
        by_dof: Whether joint targets index by ``joint_qd``.
        control_hz: The environment's control rate.
        options: Per-mode extra arguments, ``{mode: kwargs}``. **Per mode rather
            than shared**, because a shared argument reaches every constructor:
            handing ``convention="align2"`` to all of them makes
            ``ActionJointPosition`` raise ``TypeError``.
        **kwargs: Forwarded to every mode whose constructor accepts them, asked
            of its signature. One that no mode in ``modes`` accepts raises
            ``TypeError`` rather than vanishing.
    """
    spaces: dict[str, ActionSpace] = {}
    per_mode = dict(options or {})
    taken: set[str] = set()
    for mode in modes:
        shared = _accepted(control_mode_class(mode), kwargs)
        taken.update(shared)
        try:
            spaces[mode] = make_action_space(
                mode, arms, by_dof=by_dof, control_hz=control_hz, **shared, **per_mode.get(mode, {})
            )
        except UnsupportedMode:
            continue
    unused = sorted(set(kwargs) - taken)
    if unused:
        raise TypeError(f"no control mode in {list(modes)} accepts {unused}")
    return spaces


def _accepted(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The part of ``kwargs`` that ``cls``'s constructor takes by name, or all of it given ``**kwargs``."""
    params = inspect.signature(cls).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def zero_action(space: ActionSpace, state) -> np.ndarray:
    """An action that holds the scene where it is.

    Delegates to :meth:`ActionSpace.hold_action`. Kept as a function because that
    is what callers import, and because "hold" is a property of the space rather
    than of whoever is asking.
    """
    return space.hold_action(state)
