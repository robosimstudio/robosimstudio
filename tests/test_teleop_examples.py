# SPDX-License-Identifier: Apache-2.0
"""The teleoperation examples in their hardware-free mode, stepped in virtual time.

**No real servo is involved, and nothing here shows that force feedback works on
hardware.** ``--simulate`` puts a table of registers where the Feetech bus would
be and a scripted hand where the operator would be; a motor whose torque this
code enables does not move, so the hand goes where its script says however hard
a goal pulls. What is real is everything the examples build and drive: the
shipped ``yam_bimanual`` with the teleoperation gains, the welded panel and the
20 kg crate, the follower's physics, and every line of the examples' own
wiring from a leader reading to a servo register.

Both examples are loaded by file path and run through their own ``main()`` with
``--simulate``, the way a reader without a rig runs them. Only the page is
replaced -- :func:`robosimstudio.view` for the first, ``StudioSession`` for the
second -- and the clock the scripted hand and the page read is a virtual one,
so the result does not depend on how fast this machine is: the first example's
page steps at its scene's 50 Hz, and the second's at 60 Hz with each leader loop
at 100 Hz, on a 300 Hz base.

What is claimed, per example:

* ``00_leader_arm``: it builds, hands the studio a page that takes both arms off
  their gizmos, and the simulated arm follows the scripted hand through its
  whole sweep with every joint inside the fixed deadband a feedback loop would
  judge it by -- it tracks, it does not lag;
* ``01_force_feedback``: the arm that meets nothing never has a motor enabled;
  the arm whose sweep meets the panel is stopped at 0.30 rad while its hand
  goes on to 0.60, and the correction on the blocked joint leaves the deadband,
  reaches the cap, and points back toward where the arm actually is, with the
  status line naming the motor; and with ``--no-feedback`` the free arm still
  swings its 0.60 rad and the other still stops at the panel, while nothing but
  the torque-off is ever written.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop import simulated
from tests.fixtures.leader_feedback import instant
from tests.fixtures.viser_stubs import _StubGui

TIER = "medium"
"""Builds the examples' scenes and steps them on the GPU."""

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "06_teleop"

BASE_HZ = 300
"""The virtual clock's base rate: the leader ticks every 3rd count, the page steps every 5th."""


def _load(name: str):
    """One example as a module, loaded by file path: a numbered folder is not a package."""
    module_name = "_rss_teleop_example_" + re.sub(r"\W", "_", name)
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _virtual(clock: _Clock):
    """``simulated.connector`` with this clock under every scripted hand."""
    real = simulated.connector

    def connector(homes, **script):
        return real(homes, clock=clock, **script)

    return connector


class _Arm:
    """What :class:`StudioBridge` and the first example ask of a ``PanelArm``."""

    def __init__(self) -> None:
        self.suspended = False

    def suspend(self) -> None:
        self.suspended = True

    def resume(self) -> None:
        self.suspended = False


def _measure(sim, space):
    """``[q_0 .. q_5, jaw]`` per arm, as the bridge reads the follower back."""
    from robosimstudio.arms import arm_slots

    joint_q = sim.state.joint_q.numpy()
    out = {}
    for index, name in enumerate(arm_slots(sim)):
        ik, gripper = space.arms[index]
        values = [float(joint_q[c]) for c in ik.arm_coords]
        values.append(ffb.jaw_fraction(gripper, joint_q) if gripper is not None else 0.0)
        out[name] = np.asarray(values)
    return out


class TestTheLeaderArmExample(unittest.TestCase):
    """``00_leader_arm.py --simulate``: the follower tracks the scripted hand."""

    SECONDS = 8.0
    """One full period of the scripted hand: the base out and back, the jaw shut and open twice."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio.arms import arm_slots

        clock = _Clock()
        opened: list[tuple] = []
        with (
            mock.patch.object(rss, "view", lambda sim, **kwargs: opened.append((sim, kwargs))),
            mock.patch.object(simulated, "connector", _virtual(clock)),
        ):
            _load("00_leader_arm.py").main(["--simulate"])
        ((cls.sim, kwargs),) = opened
        cls.studio = studio = _Studio(cls.sim)
        kwargs["page"](studio)
        space = cls.sim.use_control("joint_position")
        per = space.per_arm
        cls.names = arm_slots(cls.sim)
        cls.rows: list[tuple[float, dict, dict]] = []
        hz = cls.sim.request.physics.control_hz  # 50 under the example's MUJOCO_500HZ: the hand plays in sim time
        for k in range(int(cls.SECONDS * hz)):
            clock.t = k / hz
            for fn in studio.steps:
                fn()
            applied = np.asarray(space.target_action(cls.sim.control), dtype=np.float64)
            targets = {name: applied[i * per : (i + 1) * per] for i, name in enumerate(cls.names)}
            cls.sim.step_teleop()
            cls.rows.append((clock.t, targets, _measure(cls.sim, space)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _mismatch(self, name: str, slots: slice, after: float = 1.0) -> float:
        """Worst |measured - the target it was chasing| over the run, after the start-up."""
        return max(
            float(np.abs(meas[name][slots] - tgt[name][slots]).max()) for t, tgt, meas in self.rows if t >= after
        )

    def test_it_builds_the_follower_and_takes_both_arms_off_their_gizmos(self) -> None:
        self.assertEqual(sorted(self.names), ["left", "right"])
        self.assertIn("cube", self.sim.names())
        self.assertTrue(all(arm.suspended for arm in self.studio.arms.values()), "the leader has the arms now")
        self.assertEqual(len(self.studio.steps), 1, "one drive hook per step")

    def test_the_hand_moved_and_the_follower_went_with_it(self) -> None:
        for name in self.names:
            with self.subTest(arm=name):
                base = [meas[name][0] for _, _, meas in self.rows]
                self.assertGreater(max(base) - min(base), 1.0, "the base swung the scripted 0.6 rad each way")
                self.assertGreater(max(meas[name][-1] for _, _, meas in self.rows), 0.95, "the jaw shut fully on air")

    def test_every_arm_joint_stays_inside_the_fixed_deadband(self) -> None:
        """0.04 rad is the floor a feedback loop would hold still against; the
        teleoperation gains keep a 0.47 rad/s hand well inside it."""
        for name in self.names:
            with self.subTest(arm=name):
                self.assertLess(self._mismatch(name, slice(0, 6)), ffb.ComplianceConfig().deadband)

    def test_the_jaw_keeps_up_too(self) -> None:
        for name in self.names:
            with self.subTest(arm=name):
                self.assertLess(self._mismatch(name, slice(6, 7)), ffb.ComplianceConfig().max_deadband_grip)

    def test_the_scene_stayed_finite(self) -> None:
        self.assertTrue(np.isfinite(self.sim.state.joint_q.numpy()).all())


class _Studio:
    """What the first example's ``page`` callback asks of a ``StudioSession``."""

    def __init__(self, sim) -> None:
        self.env = sim
        self.arms = {"left": _Arm(), "right": _Arm()}
        self.steps: list = []

    def each_step(self, fn) -> None:
        self.steps.append(fn)


class _Page:
    """A ``StudioSession`` whose ``run`` steps the scene in virtual time.

    Everything :class:`StudioBridge` and the second example touch: ``env``,
    ``gui``, ``arms``, ``panels``, ``leader_arm``, ``quit_requested``,
    ``each_step``, ``run`` and ``close``. The leader loops are ticked here too,
    on their own 100 Hz, since their threads were never started. Where each
    simulated arm got to is read back after every step, for the run in which
    no loop forms a decision to read it from.
    """

    seconds = 12.0
    clock: _Clock
    loops: list
    log: dict[str, list]
    buses: dict
    followed: dict[str, list]

    def __init__(self, sim, config=None) -> None:
        self.env = sim
        self.gui = _StubGui()
        self.arms = {"left": _Arm(), "right": _Arm()}
        self.panels: list = []
        self.leader_arm = None
        self.quit_requested = False
        self.steps: list = []

    def each_step(self, fn) -> None:
        self.steps.append(fn)

    def run(self) -> None:
        space = self.env.use_control("joint_position")
        for loop in self.loops:
            self.buses[loop.name] = loop.servos.bus
        for count in range(int(self.seconds * BASE_HZ)):
            self.clock.t = count / BASE_HZ
            if count % 3 == 0:
                for loop in self.loops:
                    loop.tick(self.clock.t)
                    bus = loop.servos.bus
                    self.log[loop.name].append((self.clock.t, loop.decisions, bus.energized(), loop.status()))
            if count % 5 == 0:
                for fn in self.steps:
                    fn()
                if self.quit_requested:
                    return
                self.env.step_teleop()
                for name, pose in _measure(self.env, space).items():
                    self.followed[name].append(pose)

    def close(self) -> None:
        pass


def _run_force_feedback(argv: list[str], seconds: float):
    """``01_force_feedback.py`` through its own ``main``, on the fake page and the virtual clock."""
    module = _load("01_force_feedback.py")
    clock = _Clock()
    started: list = []
    log: dict[str, list] = {"left": [], "right": []}
    followed: dict[str, list] = {"left": [], "right": []}
    buses: dict = {}
    names = {"seconds": seconds, "clock": clock, "loops": started, "log": log, "buses": buses, "followed": followed}
    page = type("Page", (_Page,), names)
    with (
        instant(),
        mock.patch.object(module, "StudioSession", page),
        mock.patch.object(simulated, "connector", _virtual(clock)),
        # A bound list.append would take no self here: the lambda is what makes it a method.
        mock.patch.object(ffb.LeaderArmLoop, "start", lambda self: started.append(self)),  # noqa: PLW0108
        mock.patch.object(ffb, "StudioBridge", functools.partial(ffb.StudioBridge, clock=clock)),
    ):
        code = module.main(argv)
    return code, started, log, buses, followed


class TestTheForceFeedbackExample(unittest.TestCase):
    """``01_force_feedback.py --simulate``: the panel is felt, free space is not."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.code, cls.loops, cls.log, cls.buses, _ = _run_force_feedback(["--simulate"], seconds=12.0)
        cls.config = ffb.ComplianceConfig()

    def _live(self, name: str):
        """Ticks after the arm's gates finished their dwell, with decisions."""
        rows = self.log[name]
        first = next(
            (i for i, (_, _, _, status) in enumerate(rows) if status in ("free",) or status.startswith("pushing")), None
        )
        self.assertIsNotNone(first, f"{name} never finished its alignment dwell")
        return [row for row in rows[first:] if row[1]]

    def test_it_ran_to_the_end_and_left_torque_off(self) -> None:
        self.assertEqual(self.code, 0)
        self.assertEqual(sorted(loop.name for loop in self.loops), ["left", "right"])
        for name, bus in self.buses.items():
            with self.subTest(arm=name):
                self.assertEqual(bus.energized(), [], "every motor is off on the way out")

    def test_the_arm_that_meets_nothing_never_has_a_motor_enabled(self) -> None:
        self.assertEqual(self.buses["left"].ordered("enable_torque"), [])
        rows = self._live("left")
        self.assertGreater(len(rows), 800)
        self.assertTrue(all(d.effective == 0.0 and not d.active for _, ds, _, _ in rows for d in ds))
        base = [ds[0].follower for _, ds, _, _ in rows]
        self.assertGreater(max(base) - min(base), 1.0, "and it did swing the whole 0.6 rad each way")

    def test_the_panel_stops_the_other_arm_and_the_correction_points_back_at_it(self) -> None:
        """The scripted hand swings the base inward past the panel; the arm cannot
        follow, so the mismatch on that joint is negative, leaves the deadband,
        and the goal written leads the hand back toward the arm.

        Two things hold for a hand at rest and not for one on its way back out,
        and the sweep shows both. The motor stays energized down to a release
        threshold three quarters of the way up the deadband its own speed sets,
        and between the two the correction is zero and the goal is the hand's
        own position: the last four ticks of each contact. And the mismatch is
        taken against the pose the arm was chasing, so a goal the cap behind a
        retreating hand can sit a few milliradians past the arm (4.2 at most
        here); that it sits *between* the two is claimed where the hand is
        still, at the top of the swing.
        """
        config = self.config
        held = [ds[0] for _, ds, live, _ in self._live("right") if "right_j0" in live]
        self.assertGreater(len(held), 300, "the base motor was energized through both contacts (measured 370 ticks)")
        for base in held:
            self.assertTrue(base.active)
            self.assertGreater(base.leader, 0.3, "the hand is inward, past where the panel stops the arm")
            self.assertAlmostEqual(base.follower, 0.30, delta=0.02, msg="the arm is stopped at the panel")
            self.assertLess(base.mismatch, -config.deadband, "the arm is short of the hand")
            if abs(base.mismatch) > base.deadband:
                self.assertLess(base.effective, 0.0, "past the deadband its speed set, the correction points back")
                self.assertLess(base.target, base.leader, "the goal leads the hand toward the arm")
                self.assertLessEqual(base.leader - base.target, config.max_error + 1e-9, "by at most the cap")
            else:
                self.assertLess(base.mismatch, -config.release_at(base.speed), "not yet back through the release")
                self.assertEqual(base.effective, 0.0, "inside that deadband nothing is rendered")
                self.assertEqual(base.target, base.leader, "and the goal is the hand's own position")
        easing = [d for d in held if d.effective == 0.0]
        self.assertLess(len(easing), 20, "measured 8: the release band is the last four ticks of each contact")
        deep = [d for d in held if abs(d.mismatch) > config.max_error]
        self.assertGreater(len(deep), 150, "for most of the contact the arm was more than the cap behind (229 ticks)")
        self.assertEqual({d.effective for d in deep}, {-config.max_error}, "and there every correction is capped")
        peak = max(held, key=lambda d: d.leader)
        self.assertAlmostEqual(peak.leader, 0.6, delta=0.005, msg="the top of the scripted 0.6 rad swing")
        self.assertLess(peak.speed, 0.05, "where the hand is still, so this is not lag")
        self.assertAlmostEqual(peak.target, peak.leader - config.max_error)
        self.assertLess(peak.follower, peak.target, "the goal sits between the arm, stopped at 0.30 rad,")
        self.assertLess(peak.target, peak.leader, "and the hand")

    def test_the_status_line_names_the_motor_that_is_pushing_back(self) -> None:
        pushing = [status for _, _, _, status in self.log["right"] if status.startswith("pushing back")]
        self.assertGreater(len(pushing), 100)
        self.assertTrue(
            any("right_j0" in status for status in pushing), "the base joint, while the side is on the panel"
        )
        self.assertTrue(any("right_grip" in status for status in pushing), "and the jaw the panel pushes shut")
        self.assertFalse(any("left" in status for status in pushing))

    def test_swinging_away_from_the_panel_is_free(self) -> None:
        """The other half of the same sweep, through empty space: nothing is energized."""
        rows = [(live, ds) for _, ds, live, _ in self._live("right") if ds[0].leader < -0.05]
        self.assertGreater(len(rows), 300)
        self.assertTrue(all(live == [] for live, _ in rows))
        self.assertTrue(all(d.effective == 0.0 for _, ds in rows for d in ds))

    def test_arming_wrote_the_goal_and_the_limit_before_enabling_torque(self) -> None:
        bus = self.buses["right"]
        first = next(i for i, call in enumerate(bus.calls) if call[0] == "enable_torque" and "right_j0" in call[1])
        before = [call for call in bus.calls[:first] if call[0] == "write" and call[2] == "right_j0"]
        self.assertEqual(before[-2][:3], ("write", "Goal_Position", "right_j0"))
        self.assertEqual(before[-1], ("write", "Torque_Limit", "right_j0", self.config.torque_limit))


class TestTheNoFeedbackRun(unittest.TestCase):
    """``--no-feedback``, the run the documentation says to take first."""

    EEPROM = ("Operating_Mode", "P_Coefficient", "D_Coefficient", "I_Coefficient")

    @classmethod
    def setUpClass(cls) -> None:
        cls.code, cls.loops, cls.log, cls.buses, cls.followed = _run_force_feedback(
            ["--simulate", "--no-feedback"], seconds=4.0
        )

    def test_nothing_is_energized_and_no_eeprom_register_is_touched(self) -> None:
        self.assertEqual(self.code, 0)
        for name, bus in self.buses.items():
            with self.subTest(arm=name):
                self.assertEqual(bus.ordered("enable_torque"), [])
                registers = {call[1] for call in bus.calls if call[0] in ("write", "sync_write")}
                self.assertFalse(registers & {*self.EEPROM, "Goal_Position"}, registers)
                self.assertEqual(registers - {"Torque_Limit"}, set(), "only the torque-off's zeroed limit is written")

    def test_the_status_says_so_and_the_arm_still_follows(self) -> None:
        """Half a period of the scripted hand: both bases 0.6 rad inward and
        home again. The free arm goes the whole way; the other is stopped by
        the panel where the feedback run stops it, with nothing to feel it by."""
        for name in ("left", "right"):
            with self.subTest(arm=name):
                statuses = {status for _, _, _, status in self.log[name]}
                self.assertEqual(statuses, {"reading only (no motor is energized)"})
                self.assertFalse(any(decisions for _, decisions, _, _ in self.log[name]), "no correction is formed")
                self.assertGreater(max(pose[-1] for pose in self.followed[name]), 0.95, "the jaw shut with the hand's")
                self.assertAlmostEqual(self.followed[name][-1][0], 0.0, delta=0.02, msg="and the base came home")
        self.assertGreater(max(pose[0] for pose in self.followed["left"]), 0.59, "measured 0.600: the whole swing")
        self.assertAlmostEqual(
            max(pose[0] for pose in self.followed["right"]), 0.30, delta=0.02, msg="measured 0.301: at the panel"
        )


if __name__ == "__main__":
    unittest.main()
