# SPDX-License-Identifier: Apache-2.0
"""A Feetech leader arm as a teleop input, alongside the studio's gizmo drag.

The studio already has one live input source: a panel writes ``joint_target_q``
each tick from a dragged gizmo, and :meth:`~robosimstudio.core.env.EnvBase.step_teleop`
advances the scene from whatever is already there. A leader arm is the same
shape of input -- a joint configuration read every tick, not an action a
policy decided -- so :meth:`LerobotFeetechController.poll` produces exactly
the vector :class:`~robosimstudio.solvers.control.action.ActionJointPosition`
already knows how to apply, and hands it to whichever control mode
``env.use_control("joint_position")`` returns rather than writing targets
itself. That keeps calibration the only thing this module owns.

:func:`build_action` is the pure half: servo ticks in, one
``[q_0 .. q_{n-1}, grip]`` per arm out, with no bus and no env. Test it there.
``lerobot`` itself is imported lazily, inside :meth:`LerobotFeetechController.connect`,
so a checkout that never touches real servos never pays for it, and an install
without the ``[lerobot-feetech]`` extra still imports this module.

Reading a leader is one direction. :mod:`~robosimstudio.robots.teleop.force_feedback`
is the other: it writes goals back to the same servos, and the tick conversions
it needs -- :meth:`JointCalibration.to_ticks`, :meth:`GripperCalibration.to_ticks`,
:func:`raw_delta` -- live here beside their forward maps.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.arms import arm_slots

__all__ = [
    "ENCODER_RESOLUTION",
    "ArmCalibration",
    "GripperCalibration",
    "JointCalibration",
    "LeaderError",
    "LeaderMismatch",
    "LeaderPort",
    "LeaderUnavailable",
    "LerobotFeetechController",
    "apply_arm_actions",
    "build_action",
    "check_leaders",
    "load_calibration",
    "open_leader",
    "parse_leader_ports",
    "raw_delta",
    "resolve_leaders",
    "serial_from_port",
]


ENCODER_RESOLUTION = 4096
"""Counts per turn of an STS3215, and the modulus every conversion below wraps at."""


def raw_delta(ticks: float, zero: float, resolution: int = ENCODER_RESOLUTION) -> float:
    """The *shortest* signed displacement from ``zero``, in counts.

    A rotary encoder's count is a position on a circle, and subtracting two of
    them is the displacement only while neither has crossed the seam at
    0/4095. A joint whose travel straddles that seam reads a jump of a whole
    turn -- 2*pi rad, 360 degrees -- from one tick to the next, which a leader
    loop renders as the simulated arm slamming across its range, and a velocity
    estimate to match.

    ``(ticks - zero + half) % resolution - half``, the convention a reference
    teleoperation rig for the same servos uses. For a joint parked away from
    the seam it equals the plain subtraction; at the seam it is the only right
    answer.
    """
    half = resolution / 2.0
    return (float(ticks) - float(zero) + half) % resolution - half


def serial_from_port(port: str) -> str:
    """The adapter's own serial number, out of a ``.../ch340-<serial>`` path.

    Calibration is keyed by serial, never by role (see :func:`load_calibration`),
    and the udev rule this repository ships puts the serial in the device path
    -- so this reads it back rather than querying the USB device over a
    connection that may not be open yet. A port that does not follow that
    convention (another adapter, a bare ``/dev/ttyACM0``, a test double) still
    works, less legibly: its whole basename becomes the key, and the
    calibration file has to use that same string.
    """
    base = port.rsplit("/", 1)[-1]
    return base.rsplit("-", 1)[-1] if "-" in base else base


@dataclass(frozen=True)
class JointCalibration:
    """One servo's mapping from a Feetech tick count to a sim joint radian.

    ``radians = sign * (ticks - offset_ticks) / ticks_per_radian``. All three
    are per-servo facts -- centre, direction, and counts-per-turn -- that a
    calibration pass against the sim's own home pose measures once per joint.

    Args:
        name: The motor's name, as read back from :meth:`sync_read` -- the key
            this calibration looks itself up by.
        ticks_per_radian: The servo's counts per radian of the sim joint.
        sign: ``-1`` when the servo turns opposite the sim joint's convention.
        offset_ticks: The tick count at the sim joint's zero.
    """

    name: str
    ticks_per_radian: float
    sign: float = 1.0
    offset_ticks: float = 0.0

    def to_radians(self, ticks: float) -> float:
        """This joint's angle [rad] from one raw reading.

        Through :func:`raw_delta` rather than a subtraction, so a joint whose
        travel crosses the encoder seam does not read a full turn of error.
        Away from the seam the two agree exactly.
        """
        return self.sign * raw_delta(ticks, self.offset_ticks) / self.ticks_per_radian

    def to_ticks(self, radians: float) -> float:
        """The exact inverse of :meth:`to_radians`: a joint angle as a tick count.

        Reading a leader is one direction; *commanding* one is the other, and
        force feedback needs both -- a ``Goal_Position`` is written in the
        servo's own counts, and the correction it renders was computed in the
        sim's radians. Stated here rather than at the call site so the two
        directions cannot drift apart: a calibration whose forward map is
        edited and whose inverse is not would read the arm correctly and pull
        it somewhere else.

        Not wrapped: the caller wraps onto the encoder's turn where it writes
        a goal (:meth:`~robosimstudio.robots.teleop.force_feedback.ArmLayout.ticks_of`).
        """
        return self.offset_ticks + radians * self.ticks_per_radian / self.sign


@dataclass(frozen=True)
class GripperCalibration:
    """One gripper servo's mapping from ticks to :attr:`Gripper.command`.

    ``Gripper.command`` is 0 open .. 1 closed, so this is a linear map between
    two measured tick counts rather than a radian conversion.

    Args:
        name: The motor's name, as read back from :meth:`sync_read`.
        open_ticks: The tick count with the leader's jaws fully open.
        closed_ticks: The tick count with the leader's jaws fully closed.
    """

    name: str
    open_ticks: float
    closed_ticks: float

    def to_command(self, ticks: float) -> float:
        """0 open .. 1 closed from one raw reading.

        Wrapped for the reason :meth:`JointCalibration.to_radians` is, and it
        bites harder here: a hand's two endpoints are often only a few hundred
        counts apart, so a span that happens to straddle the encoder seam comes
        out the long way round -- the wrong sign and several times the
        magnitude, which reads as a gripper that runs backwards and saturates.
        """
        span = raw_delta(self.closed_ticks, self.open_ticks)
        if span == 0.0:
            return 0.0
        return float(np.clip(raw_delta(ticks, self.open_ticks) / span, 0.0, 1.0))

    def to_ticks(self, command: float) -> float:
        """The inverse of :meth:`to_command`, over the calibrated span.

        Inverse on ``[0, 1]``, which is where :meth:`to_command` lands: it
        clips, so the round trip is exact for a command in range and saturates
        outside it rather than commanding the jaws past an endpoint somebody
        measured. The span is the same shortest-way displacement
        :meth:`to_command` divides by, so a hand whose endpoints straddle the
        encoder seam inverts the way it reads; the caller wraps the result onto
        the encoder's turn. See :meth:`JointCalibration.to_ticks` for why the
        inverse lives beside the forward map.
        """
        c = float(np.clip(command, 0.0, 1.0))
        return self.open_ticks + c * raw_delta(self.closed_ticks, self.open_ticks)


@dataclass(frozen=True)
class ArmCalibration:
    """One arm's joints, in the sim's own joint order, plus its gripper.

    The order of :attr:`joints` is what :func:`build_action` emits -- it must
    match the order :class:`~robosimstudio.solvers.control.action.ActionJointPosition`
    was built with for this arm.
    """

    joints: tuple[JointCalibration, ...]
    gripper: GripperCalibration | None = None


def build_action(ticks: dict[str, float], arms: list[ArmCalibration]) -> np.ndarray:
    """``[q_0 .. q_{n-1}, grip]`` per arm, in :attr:`ArmCalibration` order.

    Matches the layout :class:`~robosimstudio.solvers.control.action.ActionJointPosition`
    both applies and reads back, so the result is a valid ``joint_position``
    action for exactly the arms named in ``arms``.

    Args:
        ticks: One raw reading per motor name, as :meth:`sync_read` returns.
        arms: Calibration for each arm, in the action space's own arm order.
    """
    parts: list[float] = []
    for arm in arms:
        parts.extend(joint.to_radians(ticks[joint.name]) for joint in arm.joints)
        parts.append(arm.gripper.to_command(ticks[arm.gripper.name]) if arm.gripper is not None else 0.0)
    return np.asarray(parts, dtype=np.float32)


def load_calibration(path: str | Path) -> dict[str, ArmCalibration]:
    """One :class:`ArmCalibration` per key, from a measured calibration file.

    The numbers themselves -- a servo's centre, direction and counts per turn --
    are facts about one physical leader arm that nothing in this repository can
    know, so they live in a JSON file a session points at rather than in code::

        {
          "5AE6080553": {
            "label": "right YAM leader",
            "joints": [{"name": "r_shoulder", "ticks_per_radian": 651.9,
                        "sign": -1.0, "offset_ticks": 2048.0}, ...],
            "gripper": {"name": "r_grip", "open_ticks": 2048.0, "closed_ticks": 3000.0}
          },
          "5AE6080669": {...}
        }

    **Keyed by the leader's own serial number, not a role like "left"/"right".**
    A servo's mechanical mounting -- and so its calibration -- is a fact about
    one physical unit, not about which desk or which side it happens to sit on
    today; a role-keyed file goes silently wrong the moment two units swap
    places, or when it is shared with someone whose "left" is a different
    physical arm entirely. ``label`` is optional and purely for a human
    skimming the file -- callers resolve the key from the connected device's
    own serial (:func:`serial_from_port`, over the ``ch340-<serial>`` udev
    naming), never from ``label``.

    ``sign`` and ``offset_ticks`` default the way :class:`JointCalibration`
    does; ``gripper`` is omitted for an arm with none. Keys this reader does
    not know -- a ``note``, a joint's ``scale_source`` -- are a calibration
    tool's own bookkeeping and are ignored here.
    """
    data = json.loads(Path(path).read_text())
    arms: dict[str, ArmCalibration] = {}
    for name, spec in data.items():
        joints = tuple(
            JointCalibration(
                name=j["name"],
                ticks_per_radian=float(j["ticks_per_radian"]),
                sign=float(j.get("sign", 1.0)),
                offset_ticks=float(j.get("offset_ticks", 0.0)),
            )
            for j in spec["joints"]
        )
        gripper_spec = spec.get("gripper")
        gripper = (
            GripperCalibration(
                name=gripper_spec["name"],
                open_ticks=float(gripper_spec["open_ticks"]),
                closed_ticks=float(gripper_spec["closed_ticks"]),
            )
            if gripper_spec is not None
            else None
        )
        arms[name] = ArmCalibration(joints=joints, gripper=gripper)
    return arms


class LeaderError(Exception):
    """Something about a leader arm stopped a session, and the message is one sentence.

    The base of every error the teleop modules raise on purpose, so a script
    can print the sentence and stop::

        except LeaderError as exc:
            raise SystemExit(str(exc)) from None
    """


class LeaderUnavailable(LeaderError, RuntimeError):
    """A leader arm cannot be used, and the message is the one sentence that says why.

    A missing extra, a missing or mismatched calibration file, a port with
    nothing on it, a bus that would not open. There is nothing in a stack trace
    that would help somebody whose arm is simply not plugged in.
    """


class LeaderMismatch(LeaderError, ValueError):
    """The leaders named do not fit this robot; nothing was connected."""


@dataclass(frozen=True)
class LeaderPort:
    """One leader arm, resolved: which arm it drives, where it is, and its calibration.

    Args:
        name: The simulated arm it drives -- the robot's own arm name.
        port: Its serial device.
        serial: The adapter serial its calibration was looked up by.
        calibration: That calibration.
    """

    name: str
    port: str
    serial: str
    calibration: ArmCalibration


def parse_leader_ports(spec: str) -> dict[str, str]:
    """``name:port`` pairs, comma separated, as ``{name: port}``.

    ``right:/dev/ch340-5AE6080553,left:/dev/ch340-5AE6080669`` -- the name is
    the robot's own arm name and the port is that arm's own bus. A bimanual
    leader is two buses, so it is two pairs.

    Raises:
        LeaderUnavailable: If an entry is not ``name:port``.
    """
    ports: dict[str, str] = {}
    for pair in str(spec).split(","):
        if not pair.strip():
            continue
        name, _, port = pair.partition(":")
        if not name.strip() or not port.strip():
            raise LeaderUnavailable(f"leader entry {pair.strip()!r} is not name:port, e.g. right:/dev/ch340-<serial>")
        ports[name.strip()] = port.strip()
    return ports


def resolve_leaders(
    ports: Mapping[str, str], calibration_path: str | Path, *, check_ports: bool = True
) -> dict[str, LeaderPort]:
    """Look each leader's calibration up by its serial, and check it can be opened at all.

    Everything that can be known without opening a bus, in the order somebody
    setting a rig up meets it: a calibration file, an entry in it for each
    adapter's serial, a device on each port, and the Feetech driver. The first
    one missing is reported, in one sentence.

    Args:
        ports: ``{arm name: serial device}``, as :func:`parse_leader_ports` returns.
        calibration_path: The JSON :func:`load_calibration` reads.
        check_ports: Whether each device must exist. ``False`` for a port that
            is not a path on this machine.

    Raises:
        LeaderUnavailable: With the one thing to plug in, install or pass.
    """
    if not ports:
        raise LeaderUnavailable(
            "no leader arm was named: plug the leader in and pass --leader <arm>:<port>, "
            "e.g. --leader right:/dev/ch340-<serial>"
        )
    path = Path(calibration_path)
    if not path.is_file():
        raise LeaderUnavailable(
            f"calibration file {path} does not exist: write one with scripts/calibrate_leader_arms.py "
            f"--out {path}, or point at the file you have"
        )
    try:
        calibration = load_calibration(path)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LeaderUnavailable(f"calibration file {path} could not be read ({type(exc).__name__}: {exc})") from exc
    leaders: dict[str, LeaderPort] = {}
    for name, port in ports.items():
        serial = serial_from_port(port)
        if serial not in calibration:
            raise LeaderUnavailable(
                f"{name}: {path} has no calibration for serial {serial!r} (it has {sorted(calibration)}): "
                f"a calibration belongs to one physical arm, so calibrate this one with "
                f"scripts/calibrate_leader_arms.py"
            )
        if check_ports and not os.path.exists(port):
            raise LeaderUnavailable(
                f"{name}: no device at {port}: plug the leader arm in and power it "
                f"(scripts/udev/99-ch340-serial.rules is what names it /dev/ch340-<serial>)"
            )
        leaders[name] = LeaderPort(name=name, port=port, serial=serial, calibration=calibration[serial])
    try:
        driver = importlib.util.find_spec("lerobot")
    except (ImportError, ValueError):
        driver = None
    if driver is None:
        raise LeaderUnavailable(
            "the Feetech bus driver is not installed: install the lerobot-feetech extra "
            "(from a checkout: `uv sync --extra lerobot-feetech`)"
        )
    return leaders


def check_leaders(env, calibrations: Mapping[str, ArmCalibration | LeaderPort]) -> None:
    """Refuse leaders that do not match this robot, before anything is connected.

    Two ways a calibration and a scene can disagree, and both are silent at
    runtime: a name this robot has no arm for (the leader would simply never be
    applied), and a joint count that is not the arm's (its slice would drive
    the wrong joints).

    Args:
        env: The built scene -- a :class:`~robosimstudio.sim.Sim`.
        calibrations: Per arm name, its calibration or its :class:`LeaderPort`.

    Raises:
        LeaderMismatch: Naming the arm and what the robot has instead.
    """
    slots = arm_slots(env)
    missing = [name for name in calibrations if name not in slots]
    if missing:
        raise LeaderMismatch(f"this robot has no arm named {missing} -- it has {list(slots)}")
    per_arm = env.use_control("joint_position").per_arm
    for name, entry in calibrations.items():
        joints = len(getattr(entry, "calibration", entry).joints)
        if joints + 1 != per_arm:
            raise LeaderMismatch(
                f"calibration for {name!r} has {joints} joints plus a grip, "
                f"but this robot's arm takes {per_arm - 1} plus a grip"
            )


def open_leader(leader: LeaderPort, connect: Any = None) -> Any:
    """Open one leader's bus, or say in one sentence why it would not open.

    Args:
        leader: Which port, and its calibration.
        connect: ``connect(port, [calibration])``. Defaults to
            :meth:`LerobotFeetechController.connect`, which is where ``lerobot``
            is imported; a simulated leader passes its own.

    Raises:
        LeaderUnavailable: If the driver is missing or the bus did not open.
    """
    connect = connect if connect is not None else LerobotFeetechController.connect
    try:
        return connect(leader.port, [leader.calibration])
    except ModuleNotFoundError as exc:
        raise LeaderUnavailable(
            "the Feetech bus driver is not installed: install the lerobot-feetech extra "
            "(from a checkout: `uv sync --extra lerobot-feetech`)"
        ) from exc
    except Exception as exc:
        raise LeaderUnavailable(
            f"could not open the leader on {leader.port} ({type(exc).__name__}: {exc}): check that the arm "
            f"is powered, that every servo answers, and that nothing else has the port open"
        ) from exc


def apply_arm_actions(env, poses: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Apply one ``[q_0 .. q_{n-1}, grip]`` per named arm as a single ``joint_position`` action.

    ``ActionJointPosition.apply`` zips against *every* arm the control was
    built with, so an arm with no leader still needs a slice of the vector: its
    own current target, read back with ``target_action``, which makes this a
    no-op for that arm rather than a fight over it. That is what lets half a
    bimanual leader drive one arm while the other keeps its gizmo.

    Main thread only: it writes Warp arrays.

    Args:
        env: The live scene -- a :class:`~robosimstudio.sim.Sim`.
        poses: ``{arm name: pose}``. An arm missing here is left alone; a name
            the robot does not have is ignored.

    Returns:
        The combined action that was applied, in the control's own arm order.
    """
    space = env.use_control("joint_position")
    action = np.array(space.target_action(env.control), dtype=np.float32)
    per_arm = space.per_arm
    for index, name in enumerate(arm_slots(env)):
        pose = poses.get(name)
        if pose is not None:
            action[index * per_arm : (index + 1) * per_arm] = np.asarray(pose, dtype=np.float32)
    space.apply(action, env.control, env.state)
    return action


@dataclass
class LerobotFeetechController:
    """Polls a Feetech bus and drives the studio's ``joint_position`` control.

    Args:
        bus: Anything with ``sync_read(data_name) -> dict[str, float]`` and
            ``disconnect()`` -- a ``lerobot.motors.feetech.FeetechMotorsBus``,
            or a fake in a test.
        arms: Calibration for each arm this controller drives, in the same
            arm order the scene's ``joint_position`` control was built with.
    """

    bus: Any
    arms: list[ArmCalibration] = field(default_factory=list)

    @classmethod
    def connect(cls, port: str, arms: list[ArmCalibration], **bus_kwargs: Any) -> LerobotFeetechController:
        """Open the serial connection and return a controller bound to it.

        Args:
            port: The leader arm's serial device, e.g. ``/dev/ttyACM0``.
            arms: As on the class.
            **bus_kwargs: Passed through to ``FeetechMotorsBus``.
        """
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        motors = {}
        for arm in arms:
            for joint in arm.joints:
                motors[joint.name] = Motor(id=len(motors) + 1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100)
            if arm.gripper is not None:
                motors[arm.gripper.name] = Motor(
                    id=len(motors) + 1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100
                )
        bus = FeetechMotorsBus(port=port, motors=motors, **bus_kwargs)
        bus.connect()
        return cls(bus, arms)

    def read_action(self) -> np.ndarray:
        """Read the leader arm's own arms, with nothing applied.

        The half of :meth:`poll` that a caller driving *several* controllers
        into one shared, multi-arm action space needs on its own:
        ``ActionJointPosition.apply`` zips against every arm the space was
        built with, so a single-arm controller cannot call it with just its
        own slice -- the caller has to read every controller first, concatenate
        them into the space's own arm order, and apply once.
        """
        # normalize=False: the bus's default normalises Present_Position into
        # the servo's configured range (-100..100 or 0..100), and the
        # calibration above -- ticks_per_radian, offset_ticks -- is measured
        # against the raw counts underneath that, not the normalised units.
        ticks = self.bus.sync_read("Present_Position", normalize=False)
        return build_action(ticks, self.arms)

    def poll(self, env) -> np.ndarray:
        """Read the leader arm and apply it as a ``joint_position`` action.

        Only correct when this controller's own :attr:`arms` are *every* arm
        the scene's ``joint_position`` control was built with -- a single-arm
        controller among several has to go through :meth:`read_action` and let
        the caller apply the combined vector once; see there.

        Args:
            env: The live environment. Read for its active control mode and
                written through it, exactly as the gizmo-drag panel does.

        Returns:
            The action that was applied, for a caller that also wants to log it.
        """
        action = self.read_action()
        env.use_control("joint_position").apply(action, env.control, env.state)
        return action

    def close(self) -> None:
        """Release the serial connection. Idempotent."""
        self.bus.disconnect()
