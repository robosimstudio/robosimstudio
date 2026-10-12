# SPDX-License-Identifier: Apache-2.0
"""Driving every world's joint targets from device arrays.

A scripted routine plans a pair of joint configurations and interpolates between
them on the host, one step at a time (see the smoothstep in the shipped tape-peel
routine). One world, one host loop, and the arithmetic is free next to the step
it precedes. At five hundred worlds it is not: the same loop becomes five hundred
Python iterations and a host-to-device write per world per step, in front of a
physics step that is one graph launch.

So the same arithmetic moves to the device and the *plan* becomes the thing the
host writes -- per world, when a motion starts, rather than per world per step.
Three pieces, and they share a shape:

* :class:`SegmentDriver` -- each world holds one joint-space segment
  ``(q0, q1, t0, t1)`` and a clock. A step advances the clocks and evaluates the
  smoothstep. A world past the end of its segment clamps to ``q1``, which is what
  makes "the motion finished, hold there" cost nothing and need no branch.
* :class:`GripperBank` -- one 0-open-to-1-closed command per world, expanded into
  finger targets by the same interpolation
  :class:`~robosimstudio.solvers.motion.gripper.Gripper` uses, over a world axis.
* :class:`WorldEnables` -- the gate arrays. A world that nothing is driving is
  not written at all, so its targets stay bit-identical rather than being
  rewritten with the same numbers.

Everything here follows the substep block's rule: kernels are launched
unconditionally and gated by device-side flags, so the whole control stage can sit
inside one captured graph while the host keeps writing plans into the arrays it
reads by pointer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

__all__ = ["GripperBank", "SegmentDriver", "WorldEnables", "write_coord_values"]


@wp.kernel
def _advance_clocks(active: wp.array[wp.int32], dt: float, clock: wp.array[wp.float32]):
    """Tick the clock of every world that has a segment running."""
    world = wp.tid()
    if active[world] != 0:
        clock[world] = clock[world] + dt


@wp.kernel
def _start_segments(
    planned: wp.array[wp.int32],
    clock: wp.array[wp.float32],
    span: wp.array[wp.float32],
    seg_t0: wp.array[wp.float32],
    seg_t1: wp.array[wp.float32],
):
    """Stamp each newly planned segment with the clock it starts at.

    On device, off the device's own clock, which is the only clock that knows
    where a world is. The host used to keep a mirror of it that only a wave
    reset wrote, so every segment planned after the first in a wave was stamped
    with ``t0 = 0`` against a clock already hundreds of steps along -- and the
    smoothstep sat at its end before the segment's first step. A fifty-waypoint
    descent measured at ninety per cent of the way to its endpoint one step
    after it was installed, which is a step input on a stiff position servo, not
    a motion.
    """
    world = wp.tid()
    if planned[world] == 0:
        return
    seg_t0[world] = clock[world]
    seg_t1[world] = clock[world] + span[world]
    planned[world] = 0


@wp.kernel
def _drive_segments(
    clock: wp.array[wp.float32],
    active: wp.array[wp.int32],
    seg_q0: wp.array2d[wp.float32],
    seg_q1: wp.array2d[wp.float32],
    seg_t0: wp.array[wp.float32],
    seg_t1: wp.array[wp.float32],
    path: wp.array3d[wp.float32],
    path_len: wp.array[wp.int32],
    path_dt: wp.array[wp.float32],
    control_dt: float,
    scene_coords: wp.array[wp.int32],
    coords_per_world: int,
    joint_target_q: wp.array[wp.float32],
):
    """Each active world along its own motion: a smoothstep segment, or a path.

    A world with a path walks its waypoints one per ``path_dt`` seconds from the
    moment the path was stamped, interpolating linearly between consecutive
    ones and holding the last -- the streaming contract the single-world facade
    keeps (one waypoint per control step), which is what makes a straight
    Cartesian line from ``plan_linear`` *stay* a straight line. The first drive
    after a stamp lands on waypoint 0, exactly as the serial stream's first step
    does.

    A world without a path smoothsteps its segment. The clamp is what retires a
    finished motion either way: past the end the interpolant sticks and the
    world holds its endpoint exactly, so nothing has to notice the end and
    switch the world off. Whoever planned the motion sees it complete in the
    next snapshot and decides what happens next.
    """
    world, k = wp.tid()
    if active[world] == 0:
        return
    n = path_len[world]
    if n > 0:
        elapsed = wp.max(clock[world] - seg_t0[world] - control_dt, 0.0)
        progress = elapsed / wp.max(path_dt[world], 1.0e-6)
        i = int(wp.floor(progress))
        if i >= n - 1:
            joint_target_q[world * coords_per_world + scene_coords[k]] = path[world, n - 1, k]
            return
        f = progress - float(i)
        joint_target_q[world * coords_per_world + scene_coords[k]] = path[world, i, k] + f * (
            path[world, i + 1, k] - path[world, i, k]
        )
        return
    span = wp.max(seg_t1[world] - seg_t0[world], 1.0e-6)
    u = wp.clamp((clock[world] - seg_t0[world]) / span, 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    joint_target_q[world * coords_per_world + scene_coords[k]] = seg_q0[world, k] + s * (
        seg_q1[world, k] - seg_q0[world, k]
    )


@wp.kernel
def _write_world_coords(
    values: wp.array2d[wp.float32],
    enable: wp.array[wp.int32],
    coords: wp.array[wp.int32],
    coords_per_world: int,
    joint_target_q: wp.array[wp.float32],
):
    """Write absolute per-world coordinate values into the enabled worlds."""
    world, k = wp.tid()
    if enable[world] == 0:
        return
    joint_target_q[world * coords_per_world + coords[k]] = values[world, k]


@wp.kernel
def _start_ramps(
    planned: wp.array[wp.int32],
    clock: wp.array[wp.float32],
    next_command: wp.array[wp.float32],
    next_span: wp.array[wp.float32],
    ramp_from: wp.array[wp.float32],
    ramp_to: wp.array[wp.float32],
    ramp_t0: wp.array[wp.float32],
    ramp_span: wp.array[wp.float32],
):
    """Begin each newly commanded ramp from wherever the hand's command is now.

    "Now" is evaluated on device off the running ramp, so a re-grip issued
    mid-ramp continues from the blend it interrupted rather than from a stale
    host copy -- which is what starting a ramp from a value nothing in the fleet
    ever wrote did: a hand closed on an object was ramped *open* on its first
    increment.
    """
    world = wp.tid()
    if planned[world] == 0:
        return
    u = wp.clamp((clock[world] - ramp_t0[world]) / wp.max(ramp_span[world], 1.0e-6), 0.0, 1.0)
    if ramp_span[world] <= 0.0:
        u = 1.0
    ramp_from[world] = ramp_from[world] + (ramp_to[world] - ramp_from[world]) * u
    ramp_to[world] = next_command[world]
    ramp_t0[world] = clock[world]
    ramp_span[world] = next_span[world]
    planned[world] = 0


@wp.kernel
def _write_gripper_bank(
    clock: wp.array[wp.float32],
    ramp_from: wp.array[wp.float32],
    ramp_to: wp.array[wp.float32],
    ramp_t0: wp.array[wp.float32],
    ramp_span: wp.array[wp.float32],
    enable: wp.array[wp.int32],
    coords: wp.array[wp.int32],
    open_q: wp.array[wp.float32],
    closed_q: wp.array[wp.float32],
    coords_per_world: int,
    joint_target_q: wp.array[wp.float32],
):
    """Expand each world's ramped 0..1 command into that world's finger targets.

    Linear in the ramp, as ``robot.set_grip``'s host loop blends: a step input
    on a stiff finger servo slams the pads, which is how a part gets ejected
    rather than held. A span of zero is the outright command.
    """
    world, i = wp.tid()
    if enable[world] == 0:
        return
    u = wp.clamp((clock[world] - ramp_t0[world]) / wp.max(ramp_span[world], 1.0e-6), 0.0, 1.0)
    if ramp_span[world] <= 0.0:
        u = 1.0
    c = wp.clamp(ramp_from[world] + (ramp_to[world] - ramp_from[world]) * u, 0.0, 1.0)
    joint_target_q[world * coords_per_world + coords[i]] = open_q[i] + (closed_q[i] - open_q[i]) * c


@wp.kernel
def _tick_ramps(clock: wp.array[wp.float32]):
    """One control step on every hand's ramp clock."""
    world = wp.tid()
    clock[world] = clock[world] + 1.0


class WorldEnables:
    """A per-world gate, mirrored on the host so a set does not read the device.

    Args:
        num_worlds: Worlds in the fleet.
        device: Where the array lives.
        default: Initial value for every world.
    """

    def __init__(self, num_worlds: int, device, *, default: int = 0) -> None:
        self.num_worlds = int(num_worlds)
        self.host = np.full(self.num_worlds, int(default), dtype=np.int32)
        self.array = wp.array(self.host, dtype=wp.int32, device=device)

    def set(self, world: int, value: bool) -> None:
        self.host[int(world)] = 1 if value else 0

    def set_all(self, values) -> None:
        self.host[:] = np.asarray(values, dtype=np.int32).reshape(-1)

    def clear(self) -> None:
        self.host[:] = 0

    def flush(self) -> None:
        self.array.assign(self.host)

    @property
    def any(self) -> bool:
        return bool(self.host.any())

    def worlds(self) -> np.ndarray:
        """Indices of the enabled worlds."""
        return np.flatnonzero(self.host)


class SegmentDriver:
    """One joint-space motion per world, interpolated on device.

    The batched form of the plan-then-interpolate pattern the scripted routines
    use: plan a pair of configurations once, then let the device walk between
    them. Each world's motion is independent, so worlds start and finish
    motions whenever their own logic says to.

    Two shapes of motion, one driver. A **segment** is a smoothstep from ``q0``
    to ``q1`` over a duration. A **path** is a list of waypoints walked one per
    ``steps_per_waypoint`` control steps with linear interpolation between them
    -- the single-world facade's streaming contract, kept because a straight
    Cartesian line is only straight if every waypoint on it is visited. Both
    are *stamped* with their start time on device, from the device's own clock,
    at :meth:`flush`: the host never reads the clock and never guesses it.

    Args:
        num_worlds: Worlds in the fleet.
        scene_coords: Which ``joint_q`` coordinates this driver writes, as
            template indices -- one arm's, typically.
        coords_per_world: The fleet model's ``joint_q`` stride.
        control_dt: Seconds per control step, for the clocks.
        device: Where the arrays live.
        max_path: Waypoints a path may carry. A longer path is resampled
            uniformly onto this many, which keeps its shape and loses only
            density it did not need.
    """

    def __init__(
        self,
        *,
        num_worlds: int,
        scene_coords: Sequence[int],
        coords_per_world: int,
        control_dt: float,
        device,
        max_path: int = 512,
    ) -> None:
        self.num_worlds = int(num_worlds)
        self.scene_coords = [int(c) for c in scene_coords]
        self.width = len(self.scene_coords)
        self.coords_per_world = int(coords_per_world)
        self.control_dt = float(control_dt)
        self.device = device
        self.max_path = max(2, int(max_path))

        shape = (self.num_worlds, self.width)
        self._q0 = np.zeros(shape, dtype=np.float32)
        self._q1 = np.zeros(shape, dtype=np.float32)
        self._span = np.zeros(self.num_worlds, dtype=np.float32)
        self._planned = np.zeros(self.num_worlds, dtype=np.int32)
        self._path = np.zeros((self.num_worlds, self.max_path, self.width), dtype=np.float32)
        self._path_len = np.zeros(self.num_worlds, dtype=np.int32)
        self._path_dt = np.full(self.num_worlds, self.control_dt, dtype=np.float32)
        self._path_dirty = False

        self.q0 = wp.array(self._q0, dtype=wp.float32, device=device)
        self.q1 = wp.array(self._q1, dtype=wp.float32, device=device)
        self.t0 = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.t1 = wp.ones(self.num_worlds, dtype=wp.float32, device=device)
        self.span = wp.array(self._span, dtype=wp.float32, device=device)
        self.planned = wp.array(self._planned, dtype=wp.int32, device=device)
        self.clock = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.path = wp.array(self._path, dtype=wp.float32, device=device)
        self.path_len = wp.array(self._path_len, dtype=wp.int32, device=device)
        self.path_dt = wp.array(self._path_dt, dtype=wp.float32, device=device)
        self.enables = WorldEnables(self.num_worlds, device)
        self.coords = wp.array(np.asarray(self.scene_coords, dtype=np.int32), dtype=wp.int32, device=device)

    # -------------------------------------------------------------- planning
    def plan(self, world: int, q0: Sequence[float], q1: Sequence[float], duration_steps: int) -> None:
        """Give one world a segment, starting at the next :meth:`flush`.

        Args:
            world: Which world.
            q0: Configuration at the start of the segment.
            q1: Configuration at the end.
            duration_steps: How many control steps to take getting there. Zero or
                one lands on ``q1`` at the next step.
        """
        index = int(world)
        self._q0[index] = np.asarray(q0, dtype=np.float32).reshape(-1)
        self._q1[index] = np.asarray(q1, dtype=np.float32).reshape(-1)
        self._span[index] = max(1, int(duration_steps)) * self.control_dt
        if self._path_len[index]:
            self._path_len[index] = 0
            self._path_dirty = True
        self._planned[index] = 1
        self.enables.host[index] = 1

    def plan_path(self, world: int, waypoints: Sequence[Sequence[float]], steps_per_waypoint: int = 1) -> int:
        """Give one world a path to walk, one waypoint per ``steps_per_waypoint`` steps.

        Returns how many control steps the walk takes to reach its last
        waypoint. A path longer than ``max_path`` is resampled uniformly onto
        ``max_path`` waypoints, first and last kept.
        """
        index = int(world)
        points = np.asarray(waypoints, dtype=np.float32).reshape(-1, self.width)
        if len(points) == 0:
            raise ValueError("a path needs at least one waypoint")
        if len(points) > self.max_path:
            picks = np.linspace(0, len(points) - 1, self.max_path).round().astype(int)
            points = points[picks]
        n = len(points)
        per = max(1, int(steps_per_waypoint))
        self._path[index, :n] = points
        self._path_len[index] = n
        self._path_dt[index] = per * self.control_dt
        self._path_dirty = True
        self._q0[index] = points[0]
        self._q1[index] = points[-1]
        self._span[index] = n * per * self.control_dt
        self._planned[index] = 1
        self.enables.host[index] = 1
        return (n - 1) * per + 1

    def stop(self, world: int) -> None:
        """Take a world off the driver. Its targets keep whatever they hold."""
        self.enables.host[int(world)] = 0

    def clear(self) -> None:
        """Retire every motion and reset the clocks -- what a wave reset does."""
        self._span[:] = 0.0
        self._planned[:] = 0
        self._path_len[:] = 0
        self._path_dirty = True
        self.enables.clear()
        self.clock.zero_()
        self.t0.zero_()
        self.t1.fill_(1.0)
        self.span.assign(self._span)
        self.planned.assign(self._planned)
        self.path_len.assign(self._path_len)
        self.enables.flush()

    def flush(self) -> None:
        """Push planned motions to the arrays the captured kernels read, and stamp them.

        The stamp is a kernel launch off the device clock, outside the captured
        graph: the host does not know what the clock says and must not pretend
        to. See :func:`_start_segments` for what pretending cost.
        """
        self.q0.assign(self._q0)
        self.q1.assign(self._q1)
        self.span.assign(self._span)
        self.planned.assign(self._planned)
        if self._path_dirty:
            self.path.assign(self._path)
            self.path_len.assign(self._path_len)
            self.path_dt.assign(self._path_dt)
            self._path_dirty = False
        self.enables.flush()
        if self._planned.any():
            wp.launch(
                _start_segments,
                dim=self.num_worlds,
                inputs=[self.planned, self.clock, self.span],
                outputs=[self.t0, self.t1],
                device=self.device,
            )
            self._planned[:] = 0

    # ---------------------------------------------------------------- driving
    def drive(self, joint_target_q: wp.array) -> None:
        """Advance the clocks and write this step's targets. Capturable."""
        wp.launch(
            _advance_clocks,
            dim=self.num_worlds,
            inputs=[self.enables.array, self.control_dt],
            outputs=[self.clock],
            device=self.device,
        )
        wp.launch(
            _drive_segments,
            dim=(self.num_worlds, self.width),
            inputs=[
                self.clock,
                self.enables.array,
                self.q0,
                self.q1,
                self.t0,
                self.t1,
                self.path,
                self.path_len,
                self.path_dt,
                self.control_dt,
                self.coords,
                self.coords_per_world,
            ],
            outputs=[joint_target_q],
            device=self.device,
        )

    # -------------------------------------------------------------- progress
    def elapsed(self) -> np.ndarray:
        """Per world, how far through its motion it is, 0 to 1. Syncs."""
        clock = self.clock.numpy()
        t0 = self.t0.numpy()
        t1 = self.t1.numpy()
        span = np.maximum(t1 - t0, 1.0e-6)
        return np.clip((clock - t0) / span, 0.0, 1.0)

    def finished(self) -> np.ndarray:
        """Per world, whether its motion has run out. Syncs once via elapsed."""
        return self.elapsed() >= 1.0

    def target(self, world: int) -> np.ndarray:
        """The configuration world ``world`` is heading for."""
        return self._q1[int(world)].copy()


class GripperBank:
    """One hand's command per world, ramped and expanded into finger targets on device.

    The batched twin of :class:`~robosimstudio.solvers.motion.gripper.Gripper`,
    built from one -- the open and closed coordinates are properties of the hand,
    identical in every world, so only the command has a world axis.

    A command may be ramped: ``set(world, value, ramp_steps=24)`` blends from
    wherever the hand's command is at the next :meth:`flush` to ``value`` over
    that many control steps, linearly, as ``robot.set_grip``'s host loop does.
    The blend runs on device and its start is stamped there, off the bank's own
    step clock, so a re-grip issued mid-ramp continues from the blend it
    interrupted.

    Args:
        num_worlds: Worlds in the fleet.
        coords: The driven finger joints' template indices.
        open_q: Per joint, the coordinate at full open.
        closed_q: Per joint, the coordinate at full closed.
        coords_per_world: The fleet model's ``joint_q`` stride.
        device: Where the arrays live.
    """

    def __init__(
        self,
        *,
        num_worlds: int,
        coords: Sequence[int],
        open_q: Sequence[float],
        closed_q: Sequence[float],
        coords_per_world: int,
        device,
    ) -> None:
        self.num_worlds = int(num_worlds)
        self.coords_per_world = int(coords_per_world)
        self.device = device
        self._command = np.zeros(self.num_worlds, dtype=np.float32)
        self._ramp = np.zeros(self.num_worlds, dtype=np.float32)
        self._planned = np.zeros(self.num_worlds, dtype=np.int32)
        self.command = wp.array(self._command, dtype=wp.float32, device=device)
        """What every world's hand is being told to reach, as last flushed."""
        self.next_span = wp.array(self._ramp, dtype=wp.float32, device=device)
        self.planned = wp.array(self._planned, dtype=wp.int32, device=device)
        self.ramp_from = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.ramp_to = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.ramp_t0 = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.ramp_span = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.clock = wp.zeros(self.num_worlds, dtype=wp.float32, device=device)
        self.coords = wp.array(np.asarray(coords, dtype=np.int32), dtype=wp.int32, device=device)
        self.open_q = wp.array(np.asarray(open_q, dtype=np.float32), dtype=wp.float32, device=device)
        self.closed_q = wp.array(np.asarray(closed_q, dtype=np.float32), dtype=wp.float32, device=device)
        self.width = len(coords)
        self.enables = WorldEnables(self.num_worlds, device, default=1)

    @classmethod
    def like(cls, gripper, *, num_worlds: int, coords_per_world: int):
        """Build from a bound single-world gripper, so one hand states its own
        interpolation once and both paths read the same numbers."""
        return cls(
            num_worlds=num_worlds,
            coords=gripper.coords,
            open_q=gripper.open_q,
            closed_q=gripper.closed_q,
            coords_per_world=coords_per_world,
            device=gripper.model.device,
        )

    def set(self, world: int, value: float, ramp_steps: int = 0) -> None:
        """Command one world's hand, 0 open to 1 closed, over ``ramp_steps`` steps."""
        index = int(world)
        self._command[index] = float(np.clip(value, 0.0, 1.0))
        self._ramp[index] = float(max(0, int(ramp_steps)))
        self._planned[index] = 1

    def set_all(self, values, ramp_steps: int = 0) -> None:
        self._command[:] = np.clip(np.asarray(values, dtype=np.float32).reshape(-1), 0.0, 1.0)
        self._ramp[:] = float(max(0, int(ramp_steps)))
        self._planned[:] = 1

    def commands(self) -> np.ndarray:
        """What every world's hand is being told to reach, without reading the device."""
        return self._command.copy()

    def flush(self) -> None:
        """Push new commands to the device and stamp their ramps off its clock."""
        self.command.assign(self._command)
        self.next_span.assign(self._ramp)
        self.planned.assign(self._planned)
        self.enables.flush()
        if self._planned.any():
            wp.launch(
                _start_ramps,
                dim=self.num_worlds,
                inputs=[self.planned, self.clock, self.command, self.next_span],
                outputs=[self.ramp_from, self.ramp_to, self.ramp_t0, self.ramp_span],
                device=self.device,
            )
            self._planned[:] = 0

    def apply(self, joint_target_q: wp.array) -> None:
        """Tick the ramp clocks and write every enabled world's finger targets. Capturable."""
        wp.launch(_tick_ramps, dim=self.num_worlds, outputs=[self.clock], device=self.device)
        wp.launch(
            _write_gripper_bank,
            dim=(self.num_worlds, self.width),
            inputs=[
                self.clock,
                self.ramp_from,
                self.ramp_to,
                self.ramp_t0,
                self.ramp_span,
                self.enables.array,
                self.coords,
                self.open_q,
                self.closed_q,
                self.coords_per_world,
            ],
            outputs=[joint_target_q],
            device=self.device,
        )


def write_coord_values(
    values: wp.array,
    enable: wp.array,
    coords: wp.array,
    coords_per_world: int,
    joint_target_q: wp.array,
    *,
    num_worlds: int,
    width: int,
    device,
) -> None:
    """Write absolute per-world joint values, gated per world.

    The batched form of
    :meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.set_joint_targets`: joint-space
    control that must not disturb the coordinates it was not given.
    """
    wp.launch(
        _write_world_coords,
        dim=(num_worlds, width),
        inputs=[values, enable, coords, coords_per_world],
        outputs=[joint_target_q],
        device=device,
    )
