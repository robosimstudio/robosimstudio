# SPDX-License-Identifier: Apache-2.0
"""A leader arm that is not one: a servo bus over a table, and a scripted hand.

For running the teleoperation code with no arm on the desk -- the tests, and
the examples' ``--simulate``. **Exactly what it simulates, and what it does not:**

* :class:`SimulatedBus` answers the Feetech bus methods the teleop modules
  call (``sync_read``, ``read``, ``write``, ``sync_write``, ``enable_torque``,
  ``disable_torque``, ``disconnect``) from a table of registers per motor, and
  records every call in order. A write lands in the table and is returned by
  the next read. That is all a register is here.
* :class:`ScriptedHand` stands in for the operator: ``Present_Position`` is a
  fixed function of time -- the follower's home pose plus a slow sweep --
  converted to ticks through the arm's own calibration.

What is **not** simulated is everything that makes a servo a servo. A motor
whose torque is enabled does not move: the scripted hand goes where the script
says however hard the goal position pulls, so a correction is computed and
written and never felt. There is no serial timing, no dropped packet, no
EEPROM, no torque, no current. A run over this bus shows that the code path
executes and what it *would* write; it says nothing about how an arm feels,
and it is not evidence that force feedback works on hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from robosimstudio.arms import arm_slots

from .lerobot_controller import (
    ENCODER_RESOLUTION,
    ArmCalibration,
    GripperCalibration,
    JointCalibration,
    LeaderPort,
    LerobotFeetechController,
)

__all__ = ["ScriptedHand", "SimulatedBus", "connector", "direct_drive_calibration", "homes", "leaders"]

DIRECT_DRIVE_TICKS_PER_RADIAN = ENCODER_RESOLUTION / (2.0 * math.pi)
"""A direct-drive STS3215: 4096 counts per turn, no gearing between servo and joint."""


class SimulatedBus:
    """A table of servo registers that records the order they are written.

    The order is recorded because with force feedback the order is the safety
    claim: a goal and a torque limit must land before the torque is enabled,
    and a tick that both disarms and arms must disarm first. A stand-in that
    only remembered final register values could not tell a correct render from
    one that briefly energized a motor against a stale goal.

    Args:
        motors: Motor names to answer for.
        torque_enable: Starting ``Torque_Enable``, for a servo that comes up
            energized.
        torque_limit: Starting ``Torque_Limit``, for one that comes up already
            being driven by something else.
        present: Starting ``Present_Position`` per motor; 2048 where unstated.
        hand: ``hand(seconds) -> {motor: ticks}``, consulted on every read of
            ``Present_Position``. ``None`` leaves positions where they were put.
        clock: ``time.monotonic`` unless a test wants its own.
    """

    def __init__(
        self,
        motors: Sequence[str],
        *,
        torque_enable: int = 0,
        torque_limit: int = 0,
        present: Mapping[str, float] | None = None,
        hand: Callable[[float], Mapping[str, float]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.calls: list[tuple] = []
        self.hand = hand
        self.clock = clock
        self.started = clock()
        self.registers: dict[str, dict[str, Any]] = {
            motor: {
                "Torque_Enable": torque_enable,
                "Torque_Limit": torque_limit,
                "Present_Position": (present or {}).get(motor, 2048),
                "Goal_Position": (present or {}).get(motor, 2048),
                "Operating_Mode": 3,
                "P_Coefficient": 12,
                "D_Coefficient": 0,
                "I_Coefficient": 4,
            }
            for motor in motors
        }

    def _move_hand(self) -> None:
        if self.hand is None:
            return
        for motor, ticks in self.hand(self.clock() - self.started).items():
            if motor in self.registers:
                self.registers[motor]["Present_Position"] = ticks

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        self.calls.append(("read", data_name, motor))
        if data_name == "Present_Position":
            self._move_hand()
        return self.registers[motor][data_name]

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        self.calls.append(("write", data_name, motor, value))
        self.registers[motor][data_name] = value

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        names = list(self.registers) if motors is None else ([motors] if isinstance(motors, str) else list(motors))
        self.calls.append(("sync_read", data_name, tuple(names)))
        if data_name == "Present_Position":
            self._move_hand()
        return {motor: self.registers[motor][data_name] for motor in names}

    def sync_write(self, data_name, values, *, normalize=True, num_retry=0):
        self.calls.append(("sync_write", data_name, dict(values)))
        for motor, value in values.items():
            self.registers[motor][data_name] = value

    def enable_torque(self, motors=None, num_retry=0):
        names = [motors] if isinstance(motors, str) else list(motors or self.registers)
        self.calls.append(("enable_torque", tuple(names)))
        for motor in names:
            self.registers[motor]["Torque_Enable"] = 1

    def disable_torque(self, motors=None, num_retry=0):
        names = [motors] if isinstance(motors, str) else list(motors or self.registers)
        self.calls.append(("disable_torque", tuple(names)))
        for motor in names:
            self.registers[motor]["Torque_Enable"] = 0

    def disconnect(self, disable_torque=True):
        self.calls.append(("disconnect",))

    def ordered(self, *kinds) -> list[tuple]:
        """Just the calls of these kinds, in the order they were made."""
        return [call for call in self.calls if call[0] in kinds]

    def energized(self) -> list[str]:
        """Motors whose ``Torque_Enable`` is set right now."""
        return [motor for motor, table in self.registers.items() if table["Torque_Enable"]]


class ScriptedHand:
    """Where a stand-in operator's hand is at time ``t``: home, plus a slow sweep.

    The base joint swings ``sweep`` radians either side of home once every
    ``period_s`` seconds, the elbow bobs a little, and the grip opens and
    closes -- enough for every slot to move, slow enough (0.47 rad/s at the
    default) that a follower with teleoperation gains keeps up in free space.

    Args:
        calibration: The arm's calibration; the script is written in radians
            and a grip fraction and converted to ticks through it.
        home: The follower's home pose, ``[q_0 .. q_{n-1}]`` or with a grip
            entry after it.
        sweep: Amplitude of the base joint's swing [rad].
        period_s: Seconds per swing.
        bob: Amplitude of the elbow's bob [rad].
        grip: Whether the grip opens and closes; ``False`` holds it open.
    """

    def __init__(
        self,
        calibration: ArmCalibration,
        home: Sequence[float],
        *,
        sweep: float = 0.6,
        period_s: float = 8.0,
        bob: float = 0.1,
        grip: bool = True,
    ) -> None:
        self.calibration = calibration
        self.home = [float(v) for v in home][: len(calibration.joints)]
        if len(self.home) != len(calibration.joints):
            raise ValueError(f"home has {len(self.home)} joints, the calibration {len(calibration.joints)}")
        self.sweep = float(sweep)
        self.period_s = float(period_s)
        self.bob = float(bob)
        self.grip = bool(grip)

    def pose(self, t: float) -> list[float]:
        """``[q_0 .. q_{n-1}, grip]`` at time ``t`` [s]. Starts at home, at rest."""
        phase = 2.0 * math.pi * t / self.period_s
        pose = list(self.home)
        if pose:
            pose[0] += self.sweep * math.sin(phase)
        if len(pose) > 2:
            pose[2] += self.bob * 0.5 * (1.0 - math.cos(2.0 * phase))
        pose.append(0.5 * (1.0 - math.cos(2.0 * phase)) if self.grip else 0.0)
        return pose

    def __call__(self, t: float) -> dict[str, int]:
        """The same pose as raw ``Present_Position`` counts per motor."""
        pose = self.pose(t)
        ticks = {
            joint.name: int(round(joint.to_ticks(value))) % ENCODER_RESOLUTION
            for joint, value in zip(self.calibration.joints, pose, strict=False)
        }
        if self.calibration.gripper is not None:
            gripper = self.calibration.gripper
            ticks[gripper.name] = int(round(gripper.to_ticks(pose[-1]))) % ENCODER_RESOLUTION
        return ticks


def direct_drive_calibration(name: str, joints: int = 6, *, gripper: bool = True) -> ArmCalibration:
    """A made-up calibration for an arm that does not exist: direct drive, centred, positive.

    ``<name>_j0 .. <name>_j<n-1>`` and ``<name>_grip``, the names
    ``scripts/calibrate_leader_arms.py`` writes.
    """
    return ArmCalibration(
        joints=tuple(
            JointCalibration(name=f"{name}_j{i}", ticks_per_radian=DIRECT_DRIVE_TICKS_PER_RADIAN, offset_ticks=2048.0)
            for i in range(joints)
        ),
        gripper=GripperCalibration(name=f"{name}_grip", open_ticks=1600.0, closed_ticks=2400.0) if gripper else None,
    )


def leaders(arm_names: Sequence[str], joints: int = 6) -> dict[str, LeaderPort]:
    """One made-up leader per arm name, on a port called ``simulated-<name>``."""
    return {
        name: LeaderPort(
            name=name,
            port=f"simulated-{name}",
            serial=f"simulated-{name}",
            calibration=direct_drive_calibration(name, joints),
        )
        for name in arm_names
    }


def homes(env, leaders: Mapping[str, LeaderPort]) -> dict[str, list[float]]:
    """``{port: where that arm's joint targets stand now}``, for :func:`connector`.

    Read off a built scene, so each scripted hand starts where its simulated
    arm already is and the first sample moves nothing.

    Args:
        env: The built scene -- a :class:`~robosimstudio.sim.Sim`.
        leaders: ``{arm name: LeaderPort}``; a name the robot has no arm for is
            left out.
    """
    space = env.use_control("joint_position")
    targets = [float(v) for v in space.target_action(env.control)]
    per_arm = space.per_arm
    table: dict[str, list[float]] = {}
    for index, name in enumerate(arm_slots(env)):
        if name in leaders:
            table[leaders[name].port] = targets[index * per_arm : (index + 1) * per_arm - 1]
    return table


def connector(
    homes: Mapping[str, Sequence[float]], *, clock: Callable[[], float] = time.monotonic, **script: Any
) -> Callable[..., LerobotFeetechController]:
    """A ``connect(port, [calibration])`` that opens a :class:`SimulatedBus` under a :class:`ScriptedHand`.

    Drop-in for
    :meth:`~robosimstudio.robots.teleop.lerobot_controller.LerobotFeetechController.connect`
    wherever one is taken -- :class:`~robosimstudio.robots.teleop.force_feedback.Teleoperation`'s
    ``connect``, or called directly.

    Args:
        homes: ``{port: home pose}`` -- where each arm's script is centred. A
            port missing here is centred on zeros.
        clock: Passed to each bus.
        **script: Passed to each :class:`ScriptedHand` (``sweep``, ``period_s``,
            ``bob``, ``grip``).
    """

    def connect(port: str, arms: list[ArmCalibration], **_: Any) -> LerobotFeetechController:
        calibration = arms[0]
        home = homes.get(port, [0.0] * len(calibration.joints))
        hand = ScriptedHand(calibration, home, **script)
        motors = [joint.name for joint in calibration.joints]
        if calibration.gripper is not None:
            motors.append(calibration.gripper.name)
        bus = SimulatedBus(motors, present=hand(0.0), hand=hand, clock=clock)
        return LerobotFeetechController(bus, list(arms))

    return connect
