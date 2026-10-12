# SPDX-License-Identifier: Apache-2.0
"""A tiny leader arm over a Feetech bus that is not one, for the feedback tests.

Shared by the modules that test :mod:`robosimstudio.robots.teleop.force_feedback`:
the signal and gates in ``test_leader_force_feedback.py``, the servo writes and
the tick in ``test_leader_force_feedback_servos.py``, the two-clock exchange
in ``test_leader_force_feedback_session.py``, and the loop
closed against a stepped scene in ``test_leader_feedback_loop.py``.

:class:`FakeBus` is the interesting half, and it is the library's own
:class:`~robosimstudio.robots.teleop.simulated.SimulatedBus` with this arm's
resting positions: it answers the ``lerobot`` ``FeetechMotorsBus`` methods the
module actually uses and **records every call in order**, because with force
feedback the order is the safety claim -- a goal and a torque limit must land
before the torque is enabled, and a tick that both disarms and arms must disarm
first. No test here opens a serial port or imports ``lerobot``.
"""

from __future__ import annotations

from contextlib import contextmanager

from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop.lerobot_controller import (
    ArmCalibration,
    GripperCalibration,
    JointCalibration,
)
from robosimstudio.robots.teleop.simulated import SimulatedBus

__all__ = ["FakeBus", "FakeController", "instant", "layout", "servos"]


def layout(*, joints: int = 2, gripper: bool = True) -> ffb.ArmLayout:
    """A tiny arm whose ticks are radians, so a target reads as the arithmetic.

    ``ticks_per_radian`` of 1 and an offset of 1000 make ``1002 ticks`` exactly
    ``2 rad``; the gripper's span is 100 ticks, so a command is a percentage.
    Small enough that an expected value can be worked out in the assertion.
    """
    calibration = ArmCalibration(
        joints=tuple(JointCalibration(name=f"j{i}", ticks_per_radian=1.0, offset_ticks=1000.0) for i in range(joints)),
        gripper=GripperCalibration(name="grip", open_ticks=0.0, closed_ticks=100.0) if gripper else None,
    )
    return ffb.ArmLayout(calibration)


class FakeBus(SimulatedBus):
    """The simulated bus, resting where :func:`layout`'s arm reads zero.

    The grip motor rests at its open endpoint rather than sharing the arm
    joints' 1000: its calibrated span here is 0..100 ticks, and a reading
    outside that clips to a full-stroke command -- which would leave every
    fixture permanently in "jaws shut on something" territory.
    """

    def __init__(self, motors, *, torque_enable=0, torque_limit=0, **kwargs):
        kwargs.setdefault("present", {motor: 0 if motor == "grip" else 1000 for motor in motors})
        super().__init__(motors, torque_enable=torque_enable, torque_limit=torque_limit, **kwargs)


class FakeController:
    """What ``LerobotFeetechController.connect`` would have returned."""

    def __init__(self, bus):
        self.bus = bus

    def close(self):
        self.bus.disconnect()


def servos(bus=None, arm=None):
    """A connected ``LeaderServos`` over a :class:`FakeBus`, no serial port.

    Returns ``(servos, bus)``. The connection is assembled rather than opened:
    :meth:`LeaderServos.connect` is the one method that opens a port, and most
    of what is worth testing happens after it.
    """
    arm = arm if arm is not None else layout()
    bus = bus if bus is not None else FakeBus(arm.motors)
    built = ffb.LeaderServos("/dev/fake", arm, ffb.ComplianceConfig())
    built.controller = FakeController(bus)
    return built, bus


@contextmanager
def instant():
    """The module's settle and pacing sleeps at zero, restored on the way out.

    They are waits for real servos to act; a table of registers has nothing to
    wait for, and a verified torque-off would otherwise cost each test 0.3 s.
    """
    names = ("TORQUE_OFF_SETTLE_S", "COMMAND_GAP_S", "EEPROM_WRITE_SETTLE_S")
    before = {name: getattr(ffb, name) for name in names}
    for name in names:
        setattr(ffb, name, 0.0)
    try:
        yield
    finally:
        for name, value in before.items():
            setattr(ffb, name, value)
