# SPDX-License-Identifier: Apache-2.0
"""A hand as a single 0 (open) .. 1 (closed) command, whatever its linkage.

Two shapes of gripper, one command:

* **Mirrored prismatic fingers** -- the Panda hand, the YAM gripper. Two joints
  travelling ``+w`` and ``-w`` metres, and the command interpolates both.
* **A geared linkage** -- a Robotiq 2F-85. *One* actuated revolute joint whose
  0.785 rad of travel closes the whole hand, with five passive joints geared to
  it by solver-side mimic constraints. Its open position is 0 and its closed
  position is the limit, so the mirrored convention does not describe it at all:
  driving it to ``+w`` and ``-w`` would command the second joint the linkage
  does not have, and reading 0 as "open" would be backwards for one of them.

So the interpolation is stated per joint, as an open coordinate and a closed
coordinate. The mirrored case is the default and produces exactly the values it
always did; a geared hand states its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp

__all__ = ["Gripper"]


@wp.kernel
def _write_gripper_targets(
    command: wp.array[wp.float32],
    gripper_coords: wp.array[wp.int32],
    open_q: wp.array[wp.float32],
    closed_q: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    """Interpolate each driven finger joint between its open and closed value."""
    i = wp.tid()
    c = wp.clamp(command[0], 0.0, 1.0)
    joint_target_q[gripper_coords[i]] = open_q[i] + (closed_q[i] - open_q[i]) * c


class Gripper:
    """Driven finger joints behind one scalar, 0 open .. 1 closed.

    The command lives in a device array so expanding it into the finger targets
    can be captured alongside the IK solve: a host-side write would be frozen
    at capture time and the hand would stop responding.

    Args:
        model: The finalized model.
        coords: The driven finger joints' indices into the target vector.
        open_width: Finger travel at full open [m], for the mirrored two-joint
            default. Ignored once ``open_q`` is given.
        open_q: Per joint in ``coords``, the coordinate at full open. Empty uses
            the mirrored convention ``(+open_width, -open_width)``, which needs
            exactly two joints.
        closed_q: Per joint, the coordinate at full closed. Empty uses zeros,
            which is the mirrored convention's closed position.
        span: Fingertip separation at full open [m], which for a geared linkage
            is not derivable from its joint travel. 0 uses ``2 * open_width``.
        action_closed_at_high: Whether the **high** end of a normalized gripper
            command closes this hand.

            Read only by the action spaces, which are where a policy's number
            arrives. Everything else here speaks this package's own 0-open to
            1-closed command and is unaffected. It exists because a benchmark's
            controller maps a normalized action onto the joint's own range, so
            which end shuts is whatever the joint's limits say -- and a hand
            whose fingers *open* with increasing travel is opposite to one whose
            fingers rotate shut.
        width_coords: Two coordinates whose difference *is* the separation, for
            a hand whose jaws are a mirrored pair but whose *driven* joints are
            not both of them.

            The arena YAM is exactly that: two finger joints on mirrored ranges,
            one actuator, an equality carrying the other. Only the driven one may
            be written a target -- so ``coords`` is one joint and the mirrored
            branch below cannot fire -- but the pair still measures the gap
            exactly, and falling back to the linear interpolation would change
            what ``width`` and :meth:`closed_fraction` report on a robot a
            physics reference is calibrated against. Naming both here keeps the
            drive correct and the measurement exact.
    """

    def __init__(
        self,
        model: newton.Model,
        coords: Sequence[int],
        open_width: float = 0.04,
        *,
        open_q: Sequence[float] = (),
        closed_q: Sequence[float] = (),
        span: float = 0.0,
        width_coords: Sequence[int] = (),
        action_closed_at_high: bool = True,
    ) -> None:
        if not len(coords):
            raise ValueError("a Gripper drives at least one finger joint, got none")
        if not len(open_q) and len(coords) != 2:
            raise ValueError(
                f"a Gripper with no open_q drives exactly two mirrored finger joints, got {len(coords)}. "
                "A geared hand states open_q/closed_q per joint."
            )
        if len(open_q) and len(open_q) != len(coords):
            raise ValueError(f"open_q has {len(open_q)} values for {len(coords)} finger joints")
        if len(closed_q) and len(closed_q) != len(coords):
            raise ValueError(f"closed_q has {len(closed_q)} values for {len(coords)} finger joints")

        self.model = model
        self.coords = [int(c) for c in coords]
        self.action_closed_at_high = bool(action_closed_at_high)
        self.open_width = float(open_width)
        self.mirrored = not len(open_q)
        self.width_coords = [int(c) for c in width_coords]
        if self.width_coords and len(self.width_coords) != 2:
            raise ValueError(
                f"width_coords names the two coordinates whose difference is the gap, got {len(width_coords)}"
            )
        self.open_q = (
            np.asarray([self.open_width, -self.open_width], dtype=np.float32)
            if self.mirrored
            else np.asarray(open_q, dtype=np.float32)
        )
        self.closed_q = (
            np.zeros(len(self.coords), dtype=np.float32)
            if not len(closed_q)
            else np.asarray(closed_q, dtype=np.float32)
        )
        self.span = float(span) if span > 0.0 else 2.0 * self.open_width
        """Fingertip separation at full open [m]. What a grasp-fit test measures against."""

        self._fit: tuple[float, float] | None = None
        """A measured ``gap = a * command + b`` [m], or ``None`` for the default
        law. See :meth:`calibrate`."""

        self._command_host = np.zeros(1, dtype=np.float32)
        self._command = wp.zeros(1, dtype=wp.float32, device=model.device)
        self._coords = wp.array(np.asarray(self.coords, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._open = wp.array(self.open_q, dtype=wp.float32, device=model.device)
        self._closed = wp.array(self.closed_q, dtype=wp.float32, device=model.device)

    @property
    def command(self) -> float:
        """The last command written, 0 open .. 1 closed."""
        return float(self._command_host[0])

    @command.setter
    def command(self, value: float) -> None:
        self._command_host[0] = float(np.clip(value, 0.0, 1.0))
        self._command.assign(self._command_host)

    def apply(self, joint_target_q: wp.array) -> None:
        """Write the finger targets into the solver's target vector."""
        wp.launch(
            _write_gripper_targets,
            dim=len(self.coords),
            inputs=[self._command, self._coords, self._open, self._closed],
            outputs=[joint_target_q],
            device=self.model.device,
        )

    def width(self, state: newton.State) -> float:
        """Current fingertip separation [m].

        Exact for mirrored prismatic fingers, where the two coordinates *are* the
        separation. For a geared linkage there is no such identity, so it is the
        span scaled by how far the driver sits along its own travel -- which is
        the readout a teleoperator wants and not a calibrated measurement.
        """
        q = state.joint_q.numpy()
        if self.width_coords:
            return float(q[self.width_coords[0]] - q[self.width_coords[1]])
        if self.mirrored:
            return float(q[self.coords[0]] - q[self.coords[1]])
        lo, hi = float(self.open_q[0]), float(self.closed_q[0])
        if abs(hi - lo) < 1.0e-9:
            return self.span
        closed = float(np.clip((float(q[self.coords[0]]) - lo) / (hi - lo), 0.0, 1.0))
        return self.span * (1.0 - closed)

    # ------------------------------------------------------------ conversion
    @property
    def kind(self) -> str:
        """``"prismatic"`` or ``"geared"``.

        What a caller needs to know before treating travel as distance. A
        mirrored prismatic pair's joint coordinate *is* half the gap; a linkage's
        is an angle, and the relationship between the two is a curve.

        Keyed on whether the gap is *measurable* from coordinates rather than on
        how many joints are driven. A hand with one actuator and an
        equality-carried twin is still a parallel jaw, and calling it geared
        would tell every caller its travel is non-linear when it is exactly
        linear.
        """
        return "prismatic" if (self.mirrored or self.width_coords) else "geared"

    @property
    def calibrated(self) -> bool:
        """Whether someone measured this hand's gap against its command."""
        return self._fit is not None

    def calibrate(self, *, gap_per_command: float, gap_at_zero: float) -> None:
        """Install a measured ``gap = a * command + b`` [m], command 0 open..1 closed.

        The escape from the default law's one assumption. A task that has
        measured its own jaw gap against its own collider -- which is the only
        honest way to get one, since the free gap depends on whether the full
        fingertip, a trimmed blade or a fitted box is what actually collides --
        installs the fit here, and every consumer of :meth:`command_for_width`
        gets it. Without one they get a monotone approximation and
        ``robot.describe_gripper`` says so.

        Args:
            gap_per_command: The fit's slope [m per unit of command].
            gap_at_zero: The fit's intercept: the gap at command 0 [m].
        """
        if abs(float(gap_per_command)) < 1.0e-12:
            raise ValueError("a jaw-gap fit with zero slope cannot be inverted")
        self._fit = (float(gap_per_command), float(gap_at_zero))

    def width_for_command(self, command: float) -> float:
        """The jaw gap [m] this command is expected to produce.

        Expected, not measured -- :meth:`width` is the measurement. This is the
        forward direction of whatever law :meth:`command_for_width` inverts, and
        the two are guaranteed to agree with each other.
        """
        c = float(np.clip(command, 0.0, 1.0))
        if self._fit is not None:
            return self._fit[0] * c + self._fit[1]
        return self.span * (1.0 - c)

    def command_for_width(self, gap: float) -> float:
        """The command [0 open .. 1 closed] that leaves the jaws ``gap`` apart [m].

        **The one place metres become a command in this package.**

        Uninstrumented, the law is ``1 - gap / span``, which is exactly the
        arithmetic :meth:`~robosimstudio.solvers.GraspCandidate.grip_command`
        already used and nothing called. It is exact for a mirrored prismatic
        pair and a monotone approximation for a linkage; :meth:`calibrate`
        replaces it with a measurement where one exists.
        """
        if self._fit is not None:
            slope, intercept = self._fit
            return float(np.clip((float(gap) - intercept) / slope, 0.0, 1.0))
        if self.span <= 0.0:
            return 0.0
        return float(np.clip(1.0 - float(gap) / self.span, 0.0, 1.0))

    def closed_fraction(self, state: newton.State) -> float:
        """How far the fingers *actually* are from open, 0 open .. 1 closed.

        The measured counterpart of :attr:`command`, and the difference between
        them is the only signal a robot has that it is holding something: a hand
        commanded shut on a 0.07 m object stalls at about 0.18 of its stroke,
        while the same command on air reaches 1.0. Reporting the *command* as
        proprioception makes those two indistinguishable -- which reads as a
        gripper that always closed perfectly, so a graph verifying its grasp this
        way concludes it is holding air.
        """
        return float(np.clip(1.0 - self.width(state) / self.span, 0.0, 1.0))
