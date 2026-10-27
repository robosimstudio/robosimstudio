# SPDX-License-Identifier: Apache-2.0
"""Calibrate a Feetech leader arm against the sim's own joint convention.

lerobot ships no calibration data for these servos, nor even for its own
reference robots -- ``ticks_per_radian``, direction and offset all depend on
exactly how each horn was pressed onto its shaft, which varies unit to unit
even on identical hardware. What it ships instead is a *procedure*: move to a
known pose, capture the raw ticks, repeat.

Here that procedure is a per-joint least-squares fit rather than a two-point
read: for each joint the operator walks the leader arm's physical joint
through a spread of angles around the sim's own home pose (``ArmIndex.home_q``)
and captures a raw ``Present_Position`` reading at each. Fitting
``ticks = slope * radians + intercept`` per joint recovers
:class:`~robosimstudio.robots.teleop.lerobot_controller.JointCalibration`'s
``ticks_per_radian`` (``abs(slope)``), ``sign`` (``slope``'s own sign) and
``offset_ticks`` (the intercept) together -- so the sign is read off real
data rather than guessed, and a spread of points catches a bad read a single
pose would not. The gripper is a simple two-point open/closed capture instead,
matching :class:`~.GripperCalibration`'s own shape.

A leader arm's raw tick count depends only on that joint's own servo -- moving
one joint does not move its neighbours' encoders -- so the rest of the arm's
posture during a joint's sweep does not matter.

Writes the JSON :func:`~robosimstudio.robots.teleop.lerobot_controller.load_calibration`
reads: one entry per **leader's own serial number** (read back from the
``--leader`` port, which is expected to be the ``ch340-<serial>`` path the
repository's udev rule produces), not the arm's role name. The calibration is
a fact about one physical servo's mounting, not about which role a session
gave it; keying by role would go silently wrong the moment two units swap
desks. The role is kept alongside as a ``label``, purely for a human skimming
the file. Within an entry, joints are in ascending servo-id order, which
matters beyond readability --
:meth:`~robosimstudio.robots.teleop.lerobot_controller.LerobotFeetechController.connect`
assigns servo ids ``1..N`` by the calibration's own joint order, so writing
them out of id order would silently talk to the wrong servo.

**The whole workflow lives in the viser page, not the terminal.** Watching a
3D view and typing in a terminal at the same time is its own source of
confusion -- losing track of which prompt is live, missing context that
scrolled away. So the browser itself carries the state: a translucent ghost
of every moving shape at the *target* pose; the real arm mesh tracking the
leader continuously; a live readout of degrees, the joint's own (frequently
nonzero) home angle, and raw ticks; a *Capture* button and an *Invert*
button; and a running list of every joint already calibrated, with its
offset. The terminal only takes the bootstrap arguments below and reports
where the file landed at the end.

The simulated arm is a bare scene -- ``yam_bimanual`` in an empty ``bench_075``
cell unless ``--embodiment`` and ``--cell`` say otherwise. No task is involved;
the calibration is between a leader and a robot, and the robot is the same one
``examples/09_teleop/`` drives.

Usage::

    uv run python scripts/calibrate_leader_arms.py \\
        --leader left:/dev/ch340-<left-serial>,right:/dev/ch340-<right-serial> \\
        --out scripts/calibration/local/yam_leader.json

``scripts/calibration/local/`` is git-ignored, so a real calibration never
overwrites ``scripts/calibration/yam_leader.json``, the tracked sample, and the
examples read it from there unless ``--calibration`` says otherwise.

Partial recalibration
---------------------
The full sweep is 5 points per joint plus two gripper poses -- 64 captures for
a bimanual pair, which is the right price the first time and the wrong one when
only the zero has moved. Two flags narrow it, and both **require** an existing
entry for that leader's serial, because both keep numbers from it:

``--zero-only``
    One capture per arm, with every joint held at the sim's own home pose at
    once, rewriting ``offset_ticks`` alone. ``ticks_per_radian`` and ``sign``
    are kept from the existing fit, which is what makes this sound rather than
    a shortcut: those two describe the gear ratio and which way the horn was
    pressed onto the shaft, and neither moves unless the arm is disassembled.
    The zero does -- a slipped horn, a re-seated servo, a belt re-tensioned.
    Inverting :meth:`~...JointCalibration.to_radians` at a known angle gives it
    in one reading: ``offset_ticks = ticks - sign * ticks_per_radian * home``.

    Because it keeps a scale rather than measuring one, it refuses an entry
    that carries the bare ``DEFAULT_TICKS_PER_RADIAN`` seed on every joint and
    declares no ``scale_source`` -- nothing there vouches for the number it
    would keep. A fitted entry says where its scale came from and is accepted.

``--gripper-only``
    The two-point open/closed capture alone, keeping every joint. 2 captures.

Neither writes a joint or gripper it did not just measure, and an arm named in
``--leader`` with no existing entry is refused by name rather than silently
swept, since there is nothing for a partial pass to keep.

Safety
------
Torque is taken off every motor when its bus opens, and the bus is left that
way: the only register this tool writes is that torque-off. It never touches a
servo's EEPROM. The only file it writes is the calibration JSON, after every
capture, and it rewrites exactly the entries for the arms named in
``--leader``: another arm's entry is carried through as it was read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.robots.teleop.lerobot_controller import LeaderError, parse_leader_ports, serial_from_port

SWEEP_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
"""Degrees off the sim's home angle to sample per joint. Small enough that a
leader arm's own mechanical stops rarely get in the way; wide enough that
five points make a fit, not a guess."""

DEFAULT_TICKS_PER_RADIAN = 4096.0 / (2.0 * math.pi)
"""4096 ticks/revolution off the STS3215 datasheet, direct-drive (no gearing
between servo and joint). Only a seed for the live mirror before a joint has
two of its own points to fit -- never written to the output file by the sweep.

Written exactly rather than as the 651.9 it rounds to, because the --zero-only
guard compares against it and a hand-written file may carry either spelling."""

SEED_TICKS_PER_RADIAN_TOL = 1e-2
"""How close to the seed still counts as *being* the seed.

Loose enough to catch the same constant transcribed to one decimal by hand,
which is how it tends to appear in a file nobody fitted; far tighter than the
gap to any real fit."""

POLL_S = 0.1
"""How often the main loop re-reads the bus and redraws."""

GHOST_COLOR = np.array([70, 220, 110], dtype=np.uint8)
GHOST_OPACITY = 0.35
GHOST_PREFIX = "/calibration/ghost"

DRIVER_MISSING = (
    "the Feetech bus driver is not installed: run with `uv run --extra lerobot-feetech`, or "
    "pip install 'robosimstudio[lerobot-feetech] @ git+https://github.com/robosimstudio/robosimstudio'"
)
"""What a run without the ``lerobot-feetech`` extra is told."""

DEFAULT_OUT = Path("scripts/calibration/local/yam_leader.json")
"""Where the examples read their calibration from unless told otherwise. Git-ignored:
the tracked ``scripts/calibration/yam_leader.json`` is a sample for arms that are not yours."""


class Poser:
    """Drives one coordinate of ``sim``'s joint state and redraws it.

    Everything else in ``joint_q`` stays at the ``baseline`` captured right
    after the build -- the pose the robot calls home -- so a sweep of one joint
    moves only that joint in the browser, exactly as it should only move that
    joint on the physical leader.

    Also owns the ghost: a translucent duplicate of every *moving* shape in
    the scene (the arms and grippers, not the bench -- that never moves and a
    duplicate of it says nothing), redrawn at the target pose. The browser
    then shows two things at once -- the real arm mesh, tracking wherever the
    leader actually is, and a solid copy of the model standing at the pose it
    needs to reach.

    Built from :class:`~newton.viewer.ViewerViser`'s own per-shape batches
    (``viewer._shape_instances``) rather than a public API, because there is
    no public one for a second pose of the same model -- ``log_state`` draws
    exactly one. Each batch's mesh is already registered under
    ``viewer._meshes``; the ghost only needs a *second* set of transforms for
    it, computed the same way :meth:`show` computes the live ones, pushed to
    a parallel ``/calibration/ghost/...`` name so it never collides with the
    live draw's own ``/model/...`` one.
    """

    def __init__(self, sim, viewer) -> None:
        self.env = sim
        self.viewer = viewer
        self.baseline = sim.state.joint_q.numpy().copy()
        self.server = viewer._server
        # Static batches (furniture) hold still either way, so ghosting them
        # would only draw a second, identical copy on top of the first --
        # moving shapes only.
        self._moving = [s for s in viewer._shape_instances.values() if not s.static]
        self._ghost_handles: dict[str, Any] = {}

    def show(self, overrides: dict[int, float]) -> None:
        import newton

        joint_q = self.baseline.copy()
        for coord, value in overrides.items():
            joint_q[coord] = value
        self.env.state.joint_q.assign(joint_q)
        newton.eval_fk(self.env.model, self.env.state.joint_q, self.env.state.joint_qd, self.env.state)
        self.viewer.begin_frame(self.env.sim_time)
        self.viewer.log_state(self.env.state)
        self.viewer.end_frame()

    def show_ghost(self, overrides: dict[int, float]) -> None:
        """Redraw the ghost at the given target pose.

        A scratch pose, computed and discarded -- it never reaches
        :meth:`show`'s own state, so it does not disturb whatever the live
        loop drew last through ``log_state``.
        """
        import newton

        joint_q = self.baseline.copy()
        for coord, value in overrides.items():
            joint_q[coord] = value
        scratch = self.env.state
        live_q = scratch.joint_q.numpy().copy()  # what show() last left behind
        scratch.joint_q.assign(joint_q)
        newton.eval_fk(self.env.model, scratch.joint_q, scratch.joint_qd, scratch)

        viewer = self.viewer
        for shapes in self._moving:
            shapes.update(scratch, world_offsets=viewer.world_offsets, layer_xform=viewer.layer.xform)
            xf = shapes.world_xforms.numpy()
            if not len(xf):
                continue
            mesh_data = viewer._meshes.get(viewer._qualify(shapes.mesh))
            if mesh_data is None:
                continue
            scales = shapes.scales.numpy().astype(np.float32) if shapes.scales is not None else None
            name = f"{GHOST_PREFIX}{shapes.name}"
            handle = self.server.scene.add_batched_meshes_simple(
                name,
                vertices=mesh_data["points"],
                faces=mesh_data["indices"],
                batched_positions=xf[:, :3].astype(np.float32),
                batched_wxyzs=viewer._quats_xyzw_to_wxyz(xf[:, 3:7]),
                batched_scales=scales,
                batched_colors=np.tile(GHOST_COLOR, (len(xf), 1)),
                opacity=GHOST_OPACITY,
                lod="off",
                cast_shadow=False,
                receive_shadow=False,
            )
            self._ghost_handles[name] = handle

        # Restore: eval_fk just overwrote body_q for the scratch pose, and the
        # next call the live loop makes is show(), which reassigns joint_q
        # from scratch anyway -- but until then the viewer's last frame should
        # still read as whatever was actually drawn, not this detour.
        scratch.joint_q.assign(live_q)
        newton.eval_fk(self.env.model, scratch.joint_q, scratch.joint_qd, scratch)

    def hide_ghost(self) -> None:
        for handle in self._ghost_handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._ghost_handles = {}


def _read_many_ticks(bus, motor_names: list[str]) -> dict[str, float]:
    """One bus round-trip for several motors, instead of one read per motor.

    ``normalize=False``: the calibration this writes is measured against raw
    ticks, not the bus's own normalised units.
    """
    return {
        name: float(value)
        for name, value in bus.sync_read("Present_Position", normalize=False, motors=motor_names).items()
    }


def _fit_joint(radians: list[float], ticks: list[float]) -> tuple[float, float, float, float]:
    """``(ticks_per_radian, sign, offset_ticks, residual)`` from sampled points."""
    slope, intercept = np.polyfit(radians, ticks, 1)
    fitted = slope * np.asarray(radians) + intercept
    residual = float(np.max(np.abs(np.asarray(ticks) - fitted)))
    sign = 1.0 if slope >= 0 else -1.0
    return abs(float(slope)), sign, float(intercept), residual


def zeroed_offset(ticks: float, *, sign: float, ticks_per_radian: float, angle: float) -> float:
    """``offset_ticks`` that makes ``ticks`` read as ``angle``.

    The exact inverse of
    :meth:`~robosimstudio.robots.teleop.lerobot_controller.JointCalibration.to_radians`,
    stated here rather than inline so the round trip is checkable: feeding the
    result back through ``to_radians`` with the same ``ticks`` returns
    ``angle``, whatever the sign.
    """
    if not ticks_per_radian:
        raise ValueError("ticks_per_radian must be non-zero")
    return float(ticks) - float(sign) * float(ticks_per_radian) * float(angle)


def scale_is_vouched(joints: list[dict]) -> bool:
    """Whether a calibration's scale may be kept by a re-zero.

    A joint sitting exactly on the direct-drive seed with no ``scale_source``
    is a number nobody measured; one the sweep fitted says ``"fitted"``, and
    an entry converted from another record says where its scale came from.
    """
    for joint in joints:
        on_seed = abs(float(joint.get("ticks_per_radian", 0.0)) - DEFAULT_TICKS_PER_RADIAN) < SEED_TICKS_PER_RADIAN_TOL
        if on_seed and not joint.get("scale_source"):
            return False
    return True


def connect_bus(port: str, joint_names: list[str], gripper_name: str | None):
    """Open a Feetech bus with ids ``1..N`` in joint order, the gripper last.

    The same id assignment
    :meth:`~robosimstudio.robots.teleop.lerobot_controller.LerobotFeetechController.connect`
    makes from a finished calibration, so the file this writes talks to the
    servos it was measured on. Opening is all this does: the torque-off is
    :class:`Calibrator`'s, on whatever bus it was handed.
    """
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ModuleNotFoundError as exc:
        raise SystemExit(DRIVER_MISSING) from exc

    motors = {
        name: Motor(id=i + 1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100)
        for i, name in enumerate(joint_names)
    }
    if gripper_name is not None:
        motors[gripper_name] = Motor(id=len(joint_names) + 1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()
    return bus


def build_scene(embodiment: str, cell: str):
    """The robot the leader is calibrated against, in an otherwise empty cell."""
    import robosimstudio as rss

    scene = rss.Scene(cell=cell, name="calibration")
    scene.add(rss.robot(embodiment))
    return scene.build(control_mode="joint_position")


def plan_for(sim, ports: dict[str, str]) -> list[tuple[str, str, Any]]:
    """``(arm name, port, ArmIndex)`` per leader, refusing a name this robot lacks."""
    arms = sim.scene.embodiment.arms
    missing = [name for name in ports if name not in arms]
    if missing:
        raise SystemExit(f"this robot has no arm named {missing} -- it has {sorted(arms)}")
    return [(name, port, arms[name]) for name, port in ports.items()]


class Calibrator:
    """Drives the whole workflow from the viser GUI -- arms, joints, gripper.

    Follows the same threading contract the studio's own panels do: viser fires
    GUI callbacks on its own server threads, so they only set plain flags
    (:attr:`capture_requested`, :attr:`invert_requested`,
    :attr:`start_requested`); :meth:`tick`, called from the main loop, is the
    only place that touches the bus, the model, or the GUI's own text.

    A small explicit state machine (:attr:`phase`) rather than blocking nested
    loops: a GUI button click cannot block a call stack the way ``input()``
    blocked one, so each phase is a ``_tick_*`` method that reads one bus
    sample, updates the display, and either stays put or advances -- exactly
    one step per call.

    Args:
        sim: The built scene.
        poser: Draws it, and the ghost.
        plan: ``(arm name, port, ArmIndex)`` per leader, from :func:`plan_for`.
        out: The calibration file, read for resume and rewritten after every capture.
        mode: ``"full"``, ``"zero"`` or ``"gripper"`` -- see the module docstring.
        connect: ``connect(port, joint_names, gripper_name) -> bus``, which only
            opens it; torque is taken off here, after it returns. Defaults to
            :func:`connect_bus`, which is where ``lerobot`` is imported; a test
            passes a bus over a table of registers.
    """

    def __init__(
        self, sim, poser: Poser, plan: list[tuple[str, str, Any]], out: Path, *, mode: str = "full", connect=None
    ) -> None:
        if mode not in ("full", "zero", "gripper"):
            raise ValueError(f"mode must be full, zero or gripper, not {mode!r}")
        self.env = sim
        self.poser = poser
        self.plan = plan
        self.out = Path(out)
        self.mode = mode
        self._connect = connect if connect is not None else connect_bus
        self.calibration: dict[str, dict] = {}
        if self.out.exists():
            try:
                self.calibration = json.loads(self.out.read_text())
            except (json.JSONDecodeError, OSError):
                self.calibration = {}

        self.start_requested = False
        self.capture_requested = False
        self.invert_requested = False
        self.redo_requested = False

        self._arm_i = 0
        self._bus: Any = None
        self._joint_names: list[str] = []
        self._gripper_name = ""
        self._has_gripper = False
        self._index: Any = None
        self._arm_name = ""
        self._serial = ""
        self._joint_i = 0
        self._point_i = 0
        self._coord = 0
        self._home_rad = 0.0
        self._target_rad = 0.0
        self._radians: list[float] = []
        self._ticks: list[float] = []
        self._prior: list[tuple[float, float]] = []
        self._assumed_sign = 1.0
        self._anchor_ticks: float | None = None
        self._joints_out: list[dict] = []
        self._existing_entry: dict = {}
        """The whole prior entry, so a partial pass preserves what it does not write."""
        self._open_overrides: dict[int, float] = {}
        self._closed_overrides: dict[int, float] = {}
        self._open_ticks = 0.0
        self._results_text = "_no joints calibrated yet_\n"

        # full:    waiting -> joint -> gripper_open -> gripper_closed -> (next arm) -> done
        # zero:    waiting -> zero  -> (next arm) -> done
        # gripper: waiting -> gripper_open -> gripper_closed -> (next arm) -> done
        self.phase = "waiting"
        self._build_gui()

    # ------------------------------------------------------------------ GUI
    def _build_gui(self) -> None:
        gui = self.poser.server.gui
        with gui.add_folder("Leader Arm Calibration"):
            self._status = gui.add_text("Status", initial_value="press Start to begin", disabled=True)
            self._target_text = gui.add_text("Target", initial_value="", disabled=True)
            self._live_text = gui.add_text("Live", initial_value="", disabled=True)
            start = gui.add_button("Start")
            start.on_click(lambda _: setattr(self, "start_requested", True))
            capture = gui.add_button("Capture")
            capture.on_click(lambda _: setattr(self, "capture_requested", True))
            invert = gui.add_button("Invert direction")
            invert.on_click(lambda _: setattr(self, "invert_requested", True))
            redo = gui.add_button("Redo last joint", hint="Discard the most recently captured joint and re-sweep it.")
            redo.on_click(lambda _: setattr(self, "redo_requested", True))
            self._home_offset = gui.add_number(
                "Home offset (deg)",
                initial_value=0.0,
                step=1.0,
                hint="Shifts the live preview's assumed starting position, before this joint has 2 "
                "captured points to fit from instead. Use it if the leader wasn't actually at home "
                "when this joint's sweep began -- has no effect once the real fit takes over, and no "
                "bearing on the saved calibration either way.",
            )
            self._results = gui.add_markdown(self._results_text)

    # ----------------------------------------------------------------- loop
    def tick(self) -> None:
        if self.phase == "waiting":
            if self.start_requested:
                self.start_requested = False
                self._enter_arm()
            return
        if self.redo_requested:
            self.redo_requested = False
            self._redo_last_joint()
            return
        if self.phase == "joint":
            self._tick_joint()
        elif self.phase == "zero":
            self._tick_zero()
        elif self.phase == "gripper_open":
            self._tick_gripper("open")
        elif self.phase == "gripper_closed":
            self._tick_gripper("close")
        # "done": nothing left to do; the driver in main() stops calling tick().

    @property
    def done(self) -> bool:
        return self.phase == "done"

    # ------------------------------------------------------------ per phase
    def _advance_arm(self) -> None:
        self._arm_i += 1
        if self._arm_i < len(self.plan):
            self._enter_arm()
        else:
            self.phase = "done"
            self._status.value = "all arms calibrated"
            self._target_text.value = ""
            self._live_text.value = ""

    def _enter_arm(self) -> None:
        arm_name, port, index = self.plan[self._arm_i]
        self._arm_name = arm_name
        self._serial = serial_from_port(port)
        self._index = index
        joint_count = len(index.arm_coords)
        self._joint_names = [f"{arm_name}_j{i}" for i in range(joint_count)]
        self._has_gripper = bool(index.gripper_coords)
        self._gripper_name = f"{arm_name}_grip"

        # Resume: keyed by the leader's own serial, not the role name -- a role
        # is only "which job this session gave the device", and the calibration
        # is a fact about the device itself. An arm this file already has every
        # joint (and gripper, if this pairing has one) for is done -- skip it
        # without touching its bus. A partial entry seeds _joints_out so
        # already-captured joints are not re-swept.
        existing = self.calibration.get(self._serial) or {}
        existing_joints = existing.get("joints") or []
        fully_done = len(existing_joints) == joint_count and (not self._has_gripper or "gripper" in existing)
        if self.mode == "full" and fully_done:
            self._status.value = f"{arm_name} ({self._serial}): already fully calibrated in {self.out.name}, skipping"
            self._advance_arm()
            return
        if self.mode != "full" and len(existing_joints) != joint_count:
            # Refused rather than promoted to a full sweep: a partial pass is
            # defined by what it keeps, and there is nothing here to keep.
            raise SystemExit(
                f"--{self.mode}-only needs {arm_name} ({self._serial}) already calibrated in {self.out}, "
                f"but that entry has {len(existing_joints)} of {joint_count} joints. "
                f"Run the full sweep for this arm first."
            )
        if self.mode == "zero" and not scale_is_vouched(existing_joints):
            # --zero-only keeps the scale it finds, and for any joint whose home
            # angle is nonzero the new offset is ``ticks - sign * tpr * home``,
            # so a scale nobody stands behind puts a wrong offset straight into
            # the file. Refused rather than warned: it writes after every
            # capture, so a warning arrives too late.
            raise SystemExit(
                f"--zero-only refuses {arm_name} ({self._serial}): a joint's ticks_per_radian is exactly the "
                f"{DEFAULT_TICKS_PER_RADIAN:.1f} seed and declares no scale_source, so nothing vouches for that "
                f"scale and re-zeroing would keep it. Run the full sweep for this arm."
            )

        gripper_name = self._gripper_name if self._has_gripper else None
        self._bus = self._connect(port, list(self._joint_names), gripper_name)
        # Slack before anything is read: the operator is about to move this arm
        # by hand, and a servo holding position would fight them. Here rather
        # than in connect_bus, so it holds for whatever opened the bus.
        self._bus.disable_torque([*self._joint_names, *([gripper_name] if gripper_name else [])], num_retry=3)
        self._existing_entry = dict(existing)
        self._joints_out = list(existing_joints)
        self._joint_i = len(self._joints_out)
        self._status.value = f"{arm_name} ({port}) connected, torque off" + (
            f" -- resuming after {len(self._joints_out)} already-saved joint(s)"
            if self._joints_out and self.mode == "full"
            else ""
        )
        if self.mode == "zero":
            self._enter_zero()
        elif self.mode == "gripper" or self._joint_i >= len(self._joint_names):
            # Also the landing place for a *resumed full* run whose file already
            # held every joint but no gripper: _enter_joint would index home_q
            # past its end. Every joint is done; the gripper is what is left.
            if self._has_gripper:
                self._enter_gripper()
            else:
                self._finish_arm(gripper=None)
        else:
            self._enter_joint()

    def _redo_last_joint(self) -> None:
        """Discard the most recently captured joint of the current arm and re-sweep it.

        Only meaningful mid-sweep, with the bus already open -- a no-op before
        any joint has been captured, in a partial mode, or once the run is done.
        """
        if self._bus is None or not self._joints_out or self.mode != "full":
            return
        self._joints_out.pop()
        self._joint_i -= 1
        if self._results_text.count("\n- **") > 0:
            # Drop the last result line (added by _finish_joint) so it isn't
            # left stale once the redo produces a new one.
            head, _, _ = self._results_text.rpartition("\n- **")
            self._results_text = head + "\n" if head else "_no joints calibrated yet_\n"
            self._results.content = self._results_text
        self._autosave()
        self._enter_joint()

    # ---------------------------------------------------------------- sweep
    def _enter_joint(self) -> None:
        self._point_i = 0
        self._radians = []
        self._ticks = []
        self._prior = []
        self._assumed_sign = 1.0
        self._anchor_ticks = None
        self._home_offset.value = 0.0
        self.phase = "joint"
        self._enter_point()

    def _enter_point(self) -> None:
        i = self._joint_i
        home_rad = float(self._index.home_q[i])
        offset_deg = SWEEP_DEG[self._point_i]
        target_rad = home_rad + math.radians(offset_deg)
        self._coord = self._index.arm_coords[i]
        self._home_rad = home_rad
        self._target_rad = target_rad
        self.poser.show_ghost({self._coord: target_rad})
        self._target_text.value = (
            f"{self._arm_name} / {self._joint_names[i]}  ({i + 1}/{len(self._joint_names)}, "
            f"point {self._point_i + 1}/{len(SWEEP_DEG)}): target {math.degrees(target_rad):+.1f} deg "
            f"(home {math.degrees(home_rad):+.1f}, {offset_deg:+.0f} off)"
        )
        self._status.value = "move the leader into the ghost -- Capture when it matches, Invert if it's backwards"

    def _done_joint_overrides(self, readings: dict[str, float]) -> dict[int, float]:
        """This arm's already-calibrated joints, mapped through their own real fit.

        :meth:`Poser.show` only moves the coordinates it is given; every joint
        left out sits frozen at the sim's own home pose. Without this, the
        rendered arm during joint 3's sweep would show joints 0-2 exactly
        where the build put them -- not a preview of the leader's actual,
        current state. Their raw ticks are read alongside the joint actually
        being swept, in the one :func:`_read_many_ticks` call.
        """
        overrides = {}
        for j, joint in enumerate(self._joints_out):
            t = readings[joint["name"]]
            overrides[self._index.arm_coords[j]] = (
                joint["sign"] * (t - joint["offset_ticks"]) / joint["ticks_per_radian"]
            )
        return overrides

    def _tick_joint(self) -> None:
        motor_name = self._joint_names[self._joint_i]
        done_names = [j["name"] for j in self._joints_out]
        readings = _read_many_ticks(self._bus, [*done_names, motor_name])
        ticks = readings[motor_name]
        if len(self._prior) >= 2:
            radians, ticks_so_far = zip(*self._prior, strict=True)
            tpr, sign, offset, _ = _fit_joint(list(radians), list(ticks_so_far))
            live_rad = sign * (ticks - offset) / tpr
        else:
            if self._anchor_ticks is None:
                self._anchor_ticks = ticks
            home_rad = self._home_rad + math.radians(self._home_offset.value)
            live_rad = home_rad + self._assumed_sign * (ticks - self._anchor_ticks) / DEFAULT_TICKS_PER_RADIAN
        overrides = self._done_joint_overrides(readings)
        overrides[self._coord] = live_rad
        self.poser.show(overrides)
        delta_deg = math.degrees(live_rad) - math.degrees(self._home_rad)
        self._live_text.value = (
            f"{math.degrees(live_rad):+.1f} deg  (Δ {delta_deg:+.1f} from home)  raw ticks {ticks:.0f}"
        )

        if self.invert_requested:
            self.invert_requested = False
            # Not re-anchored: the leader has not moved, so mirroring the
            # same, unchanged delta around the same anchor is what makes the
            # flip visibly snap instead of silently doing nothing.
            self._assumed_sign = -self._assumed_sign
            self._status.value = f"flipped -- now assuming {'inverted' if self._assumed_sign < 0 else 'normal'}"
            return
        if self.capture_requested:
            self.capture_requested = False
            self._radians.append(self._target_rad)
            self._ticks.append(ticks)
            self._prior.append((self._target_rad, ticks))
            self._point_i += 1
            if self._point_i < len(SWEEP_DEG):
                self._enter_point()
            else:
                self._finish_joint()

    def _finish_joint(self) -> None:
        tpr, sign, offset, residual = _fit_joint(self._radians, self._ticks)
        name = self._joint_names[self._joint_i]
        warn = "  ⚠ high residual, consider redoing this joint" if residual > 40.0 else ""
        self._joints_out.append(
            {
                "name": name,
                "ticks_per_radian": tpr,
                "sign": sign,
                "offset_ticks": offset,
                # This joint's scale came from its own five points. See
                # scale_is_vouched for what reads it.
                "scale_source": "fitted",
            }
        )
        self._results_text += (
            f"- **{name}**: ticks/rad={tpr:.1f}  sign={sign:+.0f}  offset={offset:.1f} ticks  "
            f"(residual {residual:.1f}){warn}\n"
        )
        self._results.content = self._results_text
        self._autosave()
        self._joint_i += 1
        if self._joint_i < len(self._joint_names):
            self._enter_joint()
        elif self._has_gripper:
            self._enter_gripper()
        else:
            self._finish_arm(gripper=None)

    # -------------------------------------------------------------- re-zero
    def _enter_zero(self) -> None:
        """Stand the whole arm's ghost at home: one capture, every joint at once."""
        home = {coord: float(q) for coord, q in zip(self._index.arm_coords, self._index.home_q, strict=True)}
        self.poser.show_ghost(home)
        degrees = ", ".join(f"{math.degrees(float(q)):+.0f}" for q in self._index.home_q)
        self._target_text.value = f"{self._arm_name}: hold the whole arm at home ({degrees} deg)"
        self._status.value = "match the ghost with every joint at once -- Capture to set the zeros"
        self.phase = "zero"

    def _tick_zero(self) -> None:
        """Every joint's zero, from its existing scale and one reading at home.

        The arm is *shown* through its existing calibration rather than through
        the new offsets, which is the point: what the operator is lining up is
        the old zero's error, and hiding that behind a corrected preview would
        make every joint look already right.
        """
        readings = _read_many_ticks(self._bus, list(self._joint_names))
        joints = self._joints_out
        live = {}
        for i, joint in enumerate(joints):
            t = readings[self._joint_names[i]]
            live[self._index.arm_coords[i]] = joint["sign"] * (t - joint["offset_ticks"]) / joint["ticks_per_radian"]
        self.poser.show(live)
        self._live_text.value = "  ".join(
            f"j{i}{math.degrees(v) - math.degrees(float(self._index.home_q[i])):+.0f}"
            for i, v in enumerate(live.values())
        )
        if not self.capture_requested:
            return
        self.capture_requested = False
        lines = []
        for i, joint in enumerate(joints):
            before = float(joint["offset_ticks"])
            after = zeroed_offset(
                readings[self._joint_names[i]],
                sign=float(joint["sign"]),
                ticks_per_radian=float(joint["ticks_per_radian"]),
                angle=float(self._index.home_q[i]),
            )
            joint["offset_ticks"] = after
            lines.append(f"- **{joint['name']}**: offset {before:.1f} -> {after:.1f}  ({after - before:+.1f} ticks)\n")
        self._results_text += f"**{self._arm_name}** re-zeroed at home:\n" + "".join(lines)
        self._results.content = self._results_text
        self.poser.hide_ghost()
        self._finish_arm(gripper=self._existing_entry.get("gripper"))

    # -------------------------------------------------------------- gripper
    def _enter_gripper(self) -> None:
        """Stand the ghost at full open and hand the gripper over to the operator."""
        self.poser.hide_ghost()
        self.poser.show({})
        self._open_overrides = dict(zip(self._index.gripper_coords, self._index.gripper_open_q, strict=True))
        self._closed_overrides = dict(zip(self._index.gripper_coords, self._index.gripper_closed_q, strict=True))
        self.poser.show_ghost(self._open_overrides)
        self._target_text.value = f"{self._arm_name} / gripper: open fully"
        self._status.value = "open the gripper fully, matching the ghost -- Capture when it matches"
        self.phase = "gripper_open"

    def _tick_gripper(self, which: str) -> None:
        done_names = [j["name"] for j in self._joints_out]
        readings = _read_many_ticks(self._bus, [*done_names, self._gripper_name])
        ticks = readings[self._gripper_name]
        pose = dict(self._open_overrides if which == "open" else self._closed_overrides)
        pose.update(self._done_joint_overrides(readings))
        self.poser.show(pose)
        self._live_text.value = f"raw ticks {ticks:.0f}"
        if not self.capture_requested:
            return
        self.capture_requested = False
        if which == "open":
            self._open_ticks = ticks
            self.poser.show_ghost(self._closed_overrides)
            self._target_text.value = f"{self._arm_name} / gripper: close fully"
            self._status.value = "close the gripper fully, matching the ghost -- Capture when it matches"
            self.phase = "gripper_closed"
        else:
            gripper = {"name": self._gripper_name, "open_ticks": self._open_ticks, "closed_ticks": ticks}
            self._results_text += (
                f"- **gripper**: open={gripper['open_ticks']:.1f}  closed={gripper['closed_ticks']:.1f}\n"
            )
            self._results.content = self._results_text
            self._finish_arm(gripper=gripper)

    # -------------------------------------------------------------- writing
    def _entry(self, joints: list[dict], gripper: dict | None) -> dict:
        """One arm's entry. A partial pass keeps the keys it did not write.

        Keys this tool does not write are still someone's -- a ``note`` on an
        entry records where its numbers came from. A full sweep replaces the
        measurement and so retires such a note; a partial pass keeps the
        numbers it describes, so dropping it would strip their provenance.
        """
        entry: dict = {"label": self._arm_name, "joints": list(joints)}
        if self.mode != "full":
            entry.update({k: v for k, v in self._existing_entry.items() if k not in ("label", "joints", "gripper")})
        if gripper is not None:
            entry["gripper"] = gripper
        return entry

    def _autosave(self) -> None:
        """Write every fully- and partially-done arm to disk, right now.

        After every joint, not only every arm: two done joints followed by a
        closed terminal must not be lost. A partial entry -- an arm's
        ``joints`` so far, no ``gripper`` key yet -- is a valid, resumable
        prefix of the file :func:`~robosimstudio.robots.teleop.lerobot_controller.load_calibration`
        expects; it just names fewer joints than the real arm has.
        """
        partial = dict(self.calibration)
        if self._serial and self._joints_out:
            partial[self._serial] = self._entry(self._joints_out, None)
        self._write(partial)

    def _write(self, calibration: dict) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(self.out.suffix + ".tmp")
        tmp.write_text(json.dumps(calibration, indent=2) + "\n")
        tmp.replace(self.out)  # atomic: an interrupted write must not truncate a good calibration

    def _finish_arm(self, gripper: dict | None) -> None:
        self.calibration[self._serial] = self._entry(self._joints_out, gripper)
        self._joints_out = []  # already folded into self.calibration; _autosave must not double it under _serial
        self._write(self.calibration)
        self.poser.hide_ghost()
        self.poser.show({})
        try:
            # disable_torque=False: it is already off, and the bus's own
            # disconnect would spend a write per motor re-proving it.
            self._bus.disconnect(disable_torque=False)
        except Exception:
            pass
        self._bus = None
        self._advance_arm()

    def close(self) -> None:
        """Release a bus left open by an interrupted run."""
        if self._bus is not None:
            try:
                self._bus.disconnect(disable_torque=False)
            except Exception:
                pass
            self._bus = None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--leader",
        "--arms",
        dest="leader",
        required=True,
        help="<arm>:<port>[,...], e.g. left:/dev/ch340-<left-serial>,right:/dev/ch340-<right-serial>",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="calibration JSON to write (and resume from)")
    parser.add_argument("--embodiment", default="yam_bimanual", help="the registered robot the leader drives")
    parser.add_argument("--cell", default="bench_075", help="the cell to stand it in")
    partial = parser.add_mutually_exclusive_group()
    partial.add_argument(
        "--zero-only",
        action="store_true",
        help="one capture per arm at the sim's home pose, rewriting offset_ticks only and keeping "
        "each joint's existing ticks_per_radian and sign. Needs an already-calibrated entry for "
        "every arm named in --leader",
    )
    partial.add_argument(
        "--gripper-only",
        action="store_true",
        help="the two-point open/closed gripper capture alone, keeping every joint. Needs an "
        "already-calibrated entry for every arm named in --leader",
    )
    parser.add_argument("--viser-port", type=int, default=8081, help="separate from the studio's own 8080 by default")
    parser.add_argument("--share", action="store_true", help="request a public viser share URL")
    return parser


def preflight(ports: dict[str, str], out: Path, mode: str) -> None:
    """Everything that can be refused before a scene is built, in one sentence each."""
    if importlib.util.find_spec("lerobot") is None:
        raise SystemExit(DRIVER_MISSING)
    for name, port in ports.items():
        if not os.path.exists(port):
            raise SystemExit(
                f"{name}: no device at {port}: plug the leader arm in and power it "
                f"(scripts/udev/99-ch340-serial.rules is what names it /dev/ch340-<serial>)"
            )
    if mode != "full" and not out.exists():
        raise SystemExit(
            f"{out} does not exist -- there is no zero to move without a scale to move it against. "
            f"Run the full sweep first."
        )


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    mode = "zero" if args.zero_only else "gripper" if args.gripper_only else "full"
    ports = parse_leader_ports(args.leader)
    if not ports:
        raise SystemExit("no leader arm was named: pass --leader <arm>:<port>, e.g. --leader right:/dev/ch340-<serial>")
    preflight(ports, args.out, mode)

    from robosimstudio.sim.draw import aim_camera, cell_of, home_view
    from robosimstudio_studio.garments import viewer_studio

    sim = build_scene(args.embodiment, args.cell)
    viewer = viewer_studio(port=args.viser_port, share=args.share)
    calibrator = None
    try:
        plan = plan_for(sim, ports)
        viewer.set_model(sim.model)
        aim_camera(viewer, *home_view(cell_of(sim)))
        poser = Poser(sim, viewer)
        poser.show({})
        calibrator = Calibrator(sim, poser, plan, args.out, mode=mode)
        captures = {
            "full": f"{len(SWEEP_DEG)} captures per joint, then 2 for the gripper",
            "zero": "1 capture per arm, at home -- scale and sign are kept",
            "gripper": "2 captures, open and closed -- every joint is kept",
        }[mode]
        print(f"\nmode: {mode} ({captures})")
        print(f"everything happens in the browser now: http://localhost:{args.viser_port}")
        print(f"progress is saved to {args.out} after every capture, not only at the end")
        while not calibrator.done:
            calibrator.tick()
            time.sleep(POLL_S)
    finally:
        if calibrator is not None:
            calibrator.close()
        try:
            viewer.close()
        except Exception:
            pass
        sim.close()

    # Already current: the file is rewritten after every capture, including on
    # an interrupted run, so there is nothing left to do here but say where it landed.
    if calibrator is not None and calibrator.calibration:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeaderError as exc:
        raise SystemExit(str(exc)) from None
