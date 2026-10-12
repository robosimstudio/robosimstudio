# SPDX-License-Identifier: Apache-2.0
"""Leader force feedback with the loop closed in software, against a stepped scene.

**No real servo is involved, and nothing here shows that force feedback works on
hardware.** The leader is a table of registers (``SimulatedBus``) whose
``Present_Position`` follows a trajectory this file writes down; a motor whose
torque this code enables does not move, so the "hand" goes where the script
says however hard the goal pulls. What is real is everything on the other side:
the shipped ``yam_bimanual`` with the teleoperation gains, MuJoCo contact against
welded geometry, :class:`StudioBridge` applying the leader pose and reading the
follower back, :class:`LeaderArmLoop` deciding, and :class:`LeaderServos`
writing registers.

So this is the strongest claim available without the rig, and it is a claim
about what the code *computes*:

* a leader moved through free space, slowly and briskly, drags the simulated arm
  with every slot's correction at exactly zero -- no ``enable_torque`` is ever
  written, on either arm. The six arm joints use at most 0.17 of their live
  deadband doing it and the jaw 0.61 of its own;
* the same free pass with the page stepping irregularly, 13 to 23 ms apart,
  still energizes nothing -- and does energize the jaw motor when each sample
  is compared with the hand's pose at that instant instead of with the pose
  the arm was chasing, which is why the first is the default;
* a simulated arm stopped by a welded wall produces a correction on the blocked
  joint that leaves the deadband, capped, pointing back toward where the arm
  actually is, written goal-then-limit-then-enable; and lets go when the leader
  comes back;
* a jaw pushed shut by that wall, and a jaw stalled on a welded bar, are both
  reported on the grip slot, with opposite signs;
* the other arm, in the same scene at the same time, stays dark.

The two clocks are stepped in virtual time in a fixed order -- the leader at
100 Hz, the page at 60 Hz, on a 300 Hz base -- so nothing depends on how fast
this machine is. That a real STS3215 then renders the registers written here as
a force that feels like the scene is the rig checklist's to show
(``examples/06_teleop/README.md``, its rig checklist).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop.follower import YAM_FOLLOWER, YAM_TELEOP_ARM, YAM_TELEOP_GRIP
from robosimstudio.robots.teleop.lerobot_controller import (
    ArmCalibration,
    GripperCalibration,
    JointCalibration,
    LerobotFeetechController,
)
from robosimstudio.robots.teleop.simulated import DIRECT_DRIVE_TICKS_PER_RADIAN, SimulatedBus
from tests.fixtures.leader_feedback import instant

TIER = "medium"
"""Builds two scenes and steps them on the GPU."""

BASE_HZ = 300
"""The virtual clock's base rate: the leader ticks every 3rd count, the page steps every 5th."""

HOME = (0.025, -0.31, 0.371)
"""The right hand's tool-centre point at home, in the bench's work-surface frame [m]."""


def _calibration(name: str, sign: float) -> ArmCalibration:
    """A direct-drive leader; ``sign`` is which way every encoder counts.

    The right arm is given ``-1``, as every joint of the sample calibration in
    ``scripts/calibration/yam_leader.json`` has, so a goal written in ticks runs
    the opposite way to the angle it stands for and a sign slip in the inverse
    map would show.
    """
    return ArmCalibration(
        joints=tuple(
            JointCalibration(
                name=f"{name}_j{i}", ticks_per_radian=DIRECT_DRIVE_TICKS_PER_RADIAN, sign=sign, offset_ticks=2048.0
            )
            for i in range(6)
        ),
        gripper=GripperCalibration(name=f"{name}_grip", open_ticks=1600.0, closed_ticks=2400.0),
    )


def _smooth(t: float, start: float, duration: float) -> float:
    """0 before ``start``, 1 after ``start + duration``, a raised cosine between: starts and ends at rest."""
    if t <= start:
        return 0.0
    if t >= start + duration:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * (t - start) / duration))


def _there_and_back(t: float, start: float, duration: float) -> float:
    """0 -> 1 -> 0 over ``duration`` from ``start``, at rest at both ends and at the top."""
    if t <= start or t >= start + duration:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (t - start) / duration))


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Page:
    """The four attributes :class:`StudioBridge` needs, over a built scene."""

    def __init__(self, sim) -> None:
        self.env = sim
        self.arms: dict = {}
        self.panels: list = []
        self.leader_arm = None
        self.quit_requested = False

    def each_step(self, fn) -> None:
        del fn


class _Rig:
    """Two simulated leaders, the real bridge and a real scene, stepped in virtual time.

    Args:
        sim: The built scene.
        scripts: ``{arm name: script}``, where ``script(t)`` returns the offset
            from home ``[dq_0 .. dq_5, grip]`` the stand-in hand is at.
    """

    def __init__(self, sim, scripts) -> None:
        from robosimstudio.arms import arm_slots

        self.sim = sim
        self.clock = _Clock()
        self.said: list[str] = []
        space = sim.use_control("joint_position")
        home = np.asarray(space.target_action(sim.control), dtype=np.float64)
        self.names = arm_slots(sim)
        self.config = ffb.ComplianceConfig()
        self.loops: dict[str, ffb.LeaderArmLoop] = {}
        self.buses: dict[str, SimulatedBus] = {}
        self.layouts: dict[str, ffb.ArmLayout] = {}
        self.log: dict[str, list[tuple]] = {name: [] for name in self.names}
        """Per arm, per leader tick: ``(t, decisions, energized motors, their goal registers)``."""
        for index, name in enumerate(self.names):
            layout = ffb.ArmLayout(_calibration(name, -1.0 if name == "right" else 1.0))
            arm_home = home[index * space.per_arm : (index + 1) * space.per_arm]
            bus = SimulatedBus(layout.motors, hand=self._hand(layout, arm_home, scripts[name]), clock=self.clock)
            servos = ffb.LeaderServos(
                f"simulated-{name}",
                layout,
                self.config,
                connect=lambda port, arms, bus=bus: LerobotFeetechController(bus, arms),
            )
            servos.connect(configure=True)
            self.layouts[name], self.buses[name] = layout, bus
            self.loops[name] = ffb.LeaderArmLoop(name, servos, self.config, notify=self.said.append)
        self.bridge = ffb.StudioBridge(_Page(sim), self.loops, clock=self.clock)
        self.count = 0
        self._next_page = 0
        self._pages = 0

    @staticmethod
    def _hand(layout, home, script):
        def ticks(t: float) -> dict[str, int]:
            offset = script(t)
            pose = [float(home[i]) + float(offset[i]) for i in range(layout.width - 1)] + [float(offset[-1])]
            return {layout.motor_of(i): layout.ticks_of(i, pose[i]) for i in range(layout.width)}

        return ticks

    def run_until(self, t_end: float, gaps: tuple[int, ...] = (5,)) -> None:
        """Advance both clocks to ``t_end`` [s], logging every leader tick's decisions.

        Args:
            t_end: Virtual time to stop at.
            gaps: Base counts between page steps, cycled. ``(5,)`` is a steady
                60 Hz; anything else is a page that steps unevenly.
        """
        while self.count / BASE_HZ < t_end:
            self.clock.t = self.count / BASE_HZ
            if self.count % 3 == 0:
                for name, loop in self.loops.items():
                    loop.tick(self.clock.t)
                    bus = self.buses[name]
                    live = bus.energized()
                    goals = {motor: bus.registers[motor]["Goal_Position"] for motor in live}
                    self.log[name].append((self.clock.t, loop.decisions, live, goals))
            if self.count >= self._next_page:
                self.bridge.pump()
                if self.bridge.fault is not None:
                    raise self.bridge.fault
                self.sim.step_teleop()
                self._next_page = self.count + gaps[self._pages % len(gaps)]
                self._pages += 1
            self.count += 1

    def window(self, name: str, t0: float, t1: float):
        """The logged ticks of one arm with ``t0 <= t < t1`` that carry decisions."""
        return [row for row in self.log[name] if t0 <= row[0] < t1 and row[1]]

    def enables(self, name: str) -> list[tuple]:
        return self.buses[name].ordered("enable_torque")


def _free(t: float) -> list[float]:
    """Every slot moving through empty space: once slowly, once briskly, then still.

    The base joint swings *outward* -- away from the wall -- 0.5 rad and back,
    the elbow and the three wrist joints move with it, and the jaw closes fully
    on air and opens again. 4 s for the slow pass (0.39 rad/s at the base) and
    1.6 s for the brisk one (0.98 rad/s, and 1.96 stroke/s on the jaw).
    """
    s = _there_and_back(t, 1.5, 4.0) + _there_and_back(t, 6.0, 1.6)
    return [-0.5 * s, 0.2 * s, 0.2 * s, 0.4 * s, -0.4 * s, 0.4 * s, 1.0 * s]


SETTLED, FREE_END, HOLD_START, HOLD_END, BACK_HOME = 1.5, 8.0, 10.0, 11.5, 14.0
"""The timeline [s]: settle at home, the free pass, into the wall, hold, back out."""


def _into_the_wall(t: float) -> list[float]:
    """:func:`_free`, then the base joint swung 0.6 rad *inward*, held, and brought back."""
    offset = _free(t)
    offset[0] += 0.6 * (_smooth(t, FREE_END, HOLD_START - FREE_END) - _smooth(t, HOLD_END, 2.0))
    return offset


class TestFreeSpaceStaysFreeAndAWallIsFelt(unittest.TestCase):
    """One scene, both arms driven: the right meets a welded wall, the left never meets anything."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="feedback_loop")
        scene.add(rss.robot(YAM_FOLLOWER, arm=YAM_TELEOP_ARM, gripper=YAM_TELEOP_GRIP))
        # A welded wall 140 mm inboard of the right hand, at the hand's own
        # height: the base joint's inward swing brings the hand's side onto it.
        scene.add(rss.slab(name="wall", half=(0.10, 0.01, 0.10), at=(HOME[0] - 0.025, HOME[1] + 0.14, HOME[2])))
        cls.sim = scene.build(control_mode="joint_position")
        with instant():
            cls.rig = _Rig(cls.sim, {"left": _into_the_wall, "right": _into_the_wall})
            cls.rig.run_until(BACK_HOME + 1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    # ---------------------------------------------------------------- startup
    def test_each_arm_serves_the_dwell_before_it_is_live(self) -> None:
        """Nothing may energize off the first samples: the jaws are still
        opening from where the scene spawned them, a mismatch a quarter of the
        stroke wide that is not contact."""
        for name in self.rig.names:
            with self.subTest(arm=name):
                first = self.rig.log[name][1][1]
                self.assertGreater(abs(first[-1].mismatch), 0.2, "the jaw really does start away from the leader's")
                self.assertEqual(self.rig.said.count(f"[{name}] alignment held; contact feedback is live again."), 1)
                live = self.rig.window(name, 1.0, SETTLED)
                self.assertTrue(live and all(not d.active for _, ds, _, _ in live for d in ds))

    # ------------------------------------------------------------- free space
    def test_in_free_space_every_correction_is_exactly_zero(self) -> None:
        for name in self.rig.names:
            with self.subTest(arm=name):
                rows = self.rig.window(name, SETTLED, FREE_END)
                self.assertGreater(len(rows), 600)
                self.assertTrue(all(d.effective == 0.0 and not d.active for _, ds, _, _ in rows for d in ds))
                self.assertTrue(all(energized == [] for _, _, energized, _ in rows))

    def test_in_free_space_no_torque_register_is_enabled(self) -> None:
        """What a motor would have seen: nothing. Checked on the bus, not on the decisions."""
        for name in self.rig.names:
            with self.subTest(arm=name):
                until = self.rig.window(name, FREE_END, FREE_END + 0.02)[0]
                before = [row for row in self.rig.log[name] if row[0] <= until[0]]
                self.assertTrue(all(energized == [] for _, _, energized, _ in before))

    def test_the_free_pass_was_a_real_motion_and_the_follower_kept_up(self) -> None:
        """Not vacuous: the arm travelled, briskly, and the mismatch stayed
        under the threshold its own speed set."""
        for name in self.rig.names:
            with self.subTest(arm=name):
                rows = self.rig.window(name, SETTLED, FREE_END)
                base = [ds[0] for _, ds, _, _ in rows]
                self.assertGreater(max(d.follower for d in base) - min(d.follower for d in base), 0.45)
                self.assertGreater(max(d.speed for d in base), 0.8, "the brisk pass passed 0.8 rad/s")
                self.assertGreater(max(ds[-1].follower for _, ds, _, _ in rows), 0.95, "the jaw closed on air, fully")
                arm = max(abs(d.mismatch) / d.deadband for _, ds, _, _ in rows for d in ds[:-1])
                self.assertLess(
                    arm, 0.3, "the six arm joints used under a third of their live deadband (measured 0.17)"
                )
                jaw = max(abs(ds[-1].mismatch) / ds[-1].deadband for _, ds, _, _ in rows)
                self.assertLess(jaw, 0.8, "the jaw stayed inside its own (measured 0.61)")

    # --------------------------------------------------------------- the wall
    def _held(self, name: str):
        """The last tick of the hold: the leader 0.6 rad inward and at rest."""
        return self.rig.window(name, HOLD_START, HOLD_END)[-1][1]

    def test_the_wall_stops_the_simulated_arm_short_of_the_leader(self) -> None:
        base = self._held("right")[0]
        self.assertLess(base.follower, base.leader - 0.2, "the base joint is blocked a long way short")
        self.assertGreater(base.follower, base.leader - 0.45, "after travelling part of the way to the wall")
        self.assertLess(base.speed, 0.01, "and the hand is still, so this is not lag")

    def test_the_blocked_joint_s_correction_leaves_the_deadband_capped_and_pointing_back(self) -> None:
        base = self._held("right")[0]
        self.assertTrue(base.active)
        self.assertLess(base.mismatch, -self.rig.config.deadband)
        self.assertEqual(base.effective, -self.rig.config.max_error, "capped: the goal leads the hand by 0.2 rad")
        self.assertAlmostEqual(base.target, base.leader - self.rig.config.max_error)
        self.assertLess(base.follower, base.target, "the goal sits between the arm and the hand")
        self.assertLess(base.target, base.leader)

    def test_the_goal_in_the_servo_s_register_is_that_target_in_its_own_counts(self) -> None:
        """Through a calibration whose encoders count backwards: the goal is
        *more* ticks than the hand's position and still less angle."""
        _, decisions, energized, goals = self.rig.window("right", HOLD_START, HOLD_END)[-1]
        base, joint = decisions[0], self.rig.layouts["right"].joints[0]
        self.assertIn("right_j0", energized)
        one_tick = 1.0 / DIRECT_DRIVE_TICKS_PER_RADIAN
        self.assertAlmostEqual(joint.to_radians(goals["right_j0"]), base.target, delta=one_tick)
        self.assertGreater(goals["right_j0"], joint.to_ticks(base.leader), "sign -1: a smaller angle is a larger count")
        retargets = [
            call
            for call in self.rig.buses["right"].calls
            if call[0] == "sync_write" and call[1] == "Goal_Position" and "right_j0" in call[2]
        ]
        self.assertGreater(len(retargets), 100, "re-targeted every tick while the joint stayed in contact")

    def test_arming_wrote_the_goal_then_the_limit_then_enabled_torque(self) -> None:
        bus = self.rig.buses["right"]
        first = next(i for i, call in enumerate(bus.calls) if call[0] == "enable_torque" and "right_j0" in call[1])
        before = [call for call in bus.calls[:first] if call[0] == "write" and call[2] == "right_j0"]
        self.assertEqual(before[-2][:3], ("write", "Goal_Position", "right_j0"))
        self.assertEqual(before[-1], ("write", "Torque_Limit", "right_j0", self.rig.config.torque_limit))

    def test_the_jaw_the_wall_pushes_shut_is_felt_on_the_grip_slot(self) -> None:
        """The hand's inboard finger meets the wall first and is pushed closed
        against an open command: the follower's jaw is *more* closed than the
        leader's, so the correction closes the leader's trigger."""
        grip = self._held("right")[-1]
        self.assertGreater(grip.follower, 0.5)
        self.assertLess(grip.leader, 0.01)
        self.assertTrue(grip.active)
        self.assertEqual(grip.effective, self.rig.config.max_error)

    def test_the_joints_the_wall_does_not_load_stay_dark(self) -> None:
        """A sideways push is carried by the base joint and the jaw; the two
        pitch joints and the three wrist joints never leave their deadbands."""
        rows = self.rig.window("right", FREE_END, BACK_HOME + 1.0)
        for slot in range(1, 6):
            with self.subTest(slot=slot):
                self.assertTrue(all(not ds[slot].active for _, ds, _, _ in rows))
        motors = {motor for call in self.rig.enables("right") for motor in call[1]}
        self.assertEqual(motors, {"right_j0", "right_grip"})

    def test_backing_away_lets_go(self) -> None:
        bus = self.rig.buses["right"]
        self.assertEqual(bus.energized(), [], "nothing is energized once the leader is home")
        last_on = max(i for i, call in enumerate(bus.calls) if call[0] == "enable_torque")
        offs = {call[1][0] for call in bus.calls[last_on:] if call[0] == "disable_torque"}
        self.assertEqual(offs, {"right_j0", "right_grip"}, "each released by an acknowledged torque-off")
        final = self.rig.log["right"][-1][1]
        self.assertTrue(all(not d.active and d.effective == 0.0 for d in final))
        self.assertEqual(self.rig.loops["right"].status(), "free")

    def test_the_other_arm_swung_through_the_same_motion_and_felt_nothing(self) -> None:
        """The control: the left arm's inward 0.6 rad is outward for it, into empty space."""
        self.assertEqual(self.rig.enables("left"), [])
        rows = self.rig.window("left", SETTLED, BACK_HOME + 1.0)
        self.assertTrue(all(d.effective == 0.0 for _, ds, _, _ in rows for d in ds))
        base = [ds[0] for _, ds, _, _ in rows]
        self.assertGreater(max(d.follower for d in base), 0.55, "it did reach the 0.6 rad the right arm could not")

    def test_nothing_faulted_and_the_scene_stayed_finite(self) -> None:
        self.assertIsNone(self.rig.bridge.fault)
        self.assertTrue(all(loop.fault is None for loop in self.rig.loops.values()))
        self.assertTrue(np.isfinite(self.sim.state.joint_q.numpy()).all())
        self.assertFalse(any("suspended" in line for line in self.rig.said), "the page never went stale")


class TestAPageThatStepsUnevenlyIsNotFeltAsContact(unittest.TestCase):
    """Why a follower sample is compared with the pose it was chasing.

    The free pass again, nothing in reach, with the page stepping at uneven
    intervals the way a machine with other work to do delivers them. Once with
    each comparison; the scenes and the schedule are otherwise identical.
    """

    GAPS = (4, 7, 5, 6, 4, 5, 7, 6, 5, 4, 6, 7)
    """Base counts between page steps, cycled: 13.3 to 23.3 ms, 18.3 ms on average."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.energized: dict[str, int] = {}
        cls.worst: dict[str, float] = {}
        for compare in ffb.COMPARISONS:
            scene = rss.Scene(cell="bench_075", name="feedback_jitter")
            scene.add(rss.robot(YAM_FOLLOWER, arm=YAM_TELEOP_ARM, gripper=YAM_TELEOP_GRIP))
            sim = scene.build(control_mode="joint_position")
            try:
                with instant():
                    rig = _Rig(sim, {"left": _free, "right": _free})
                    for loop in rig.loops.values():
                        loop.compare = compare
                    rig.run_until(FREE_END, gaps=cls.GAPS)
                rows = [row for name in rig.names for row in rig.window(name, SETTLED, FREE_END)]
                cls.energized[compare] = sum(1 for _, _, live, _ in rows if live)
                cls.worst[compare] = max(abs(d.mismatch) / d.deadband for _, ds, _, _ in rows for d in ds)
            finally:
                sim.close()

    def test_against_the_pose_it_was_chasing_nothing_is_energized(self) -> None:
        self.assertEqual(self.energized["applied"], 0)
        self.assertLess(self.worst["applied"], 1.0, "measured 0.84, on the jaw")

    def test_against_the_hand_s_pose_at_that_instant_the_jaw_motor_is(self) -> None:
        """Not a requirement -- a record of what the other form does, so that
        if it ever stops being true the default can be argued again."""
        self.assertGreater(self.energized["live"], 0, "measured: 36 of the 1300 ticks, the jaw motor each time")
        self.assertGreater(self.worst["live"], 1.0, "measured 1.24, on the jaw")


GRIP_SHUT, GRIP_HELD, GRIP_OPEN = 3.5, 4.5, 7.0
"""The second timeline [s]: both jaws shut by 3.5, held to 4.5, open again by 6.5."""

BAR_THICKNESS = 0.024
"""The welded bar between the right hand's jaws [m]: a quarter of the 95 mm stroke."""


def _squeeze(t: float) -> list[float]:
    """Both hands still at home; the jaw closes over two seconds, holds, and opens."""
    return [0.0] * 6 + [_smooth(t, 1.5, 2.0) - _smooth(t, GRIP_HELD, 2.0)]


class TestAJawShutOnSomethingSolidStallsAndAJawShutOnAirDoesNot(unittest.TestCase):
    """The reason the follower's jaw is reported as measured rather than as commanded."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="feedback_grip")
        scene.add(rss.robot(YAM_FOLLOWER, arm=YAM_TELEOP_ARM, gripper=YAM_TELEOP_GRIP))
        # A welded square bar standing up through the right hand's open jaws.
        half = BAR_THICKNESS / 2.0
        scene.add(rss.slab(name="bar", half=(half, half, 0.20), at=(HOME[0], HOME[1], 0.20)))
        cls.sim = scene.build(control_mode="joint_position")
        with instant():
            cls.rig = _Rig(cls.sim, {"left": _squeeze, "right": _squeeze})
            cls.rig.run_until(GRIP_OPEN + 0.5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _held(self, name: str):
        return self.rig.window(name, GRIP_SHUT, GRIP_HELD)[-1][1]

    def test_the_jaw_on_the_bar_stalls_at_the_bar_s_width(self) -> None:
        grip = self._held("right")[-1]
        self.assertGreater(grip.leader, 0.99, "the leader's jaw is shut")
        self.assertAlmostEqual(grip.follower, 1.0 - BAR_THICKNESS / 0.095, delta=0.03)

    def test_the_stall_is_rendered_as_a_pull_back_open(self) -> None:
        grip = self._held("right")[-1]
        self.assertTrue(grip.active)
        self.assertLess(grip.mismatch, -0.2)
        self.assertEqual(grip.effective, -self.rig.config.max_error)
        held = self.rig.window("right", GRIP_SHUT, GRIP_HELD)[-1][2]
        self.assertEqual(held, ["right_grip"], "the jaw motor, and only the jaw motor, is energized during the hold")

    def test_the_goal_opens_the_leader_s_jaw_part_way_and_stays_inside_its_stroke(self) -> None:
        _, decisions, _, goals = self.rig.window("right", GRIP_SHUT, GRIP_HELD)[-1]
        grip, gripper = decisions[-1], self.rig.layouts["right"].gripper
        self.assertAlmostEqual(grip.target, grip.leader - self.rig.config.max_error, delta=0.01)
        self.assertAlmostEqual(gripper.to_command(goals["right_grip"]), grip.target, delta=0.01)
        low, high = sorted((gripper.open_ticks, gripper.closed_ticks))
        written = [
            call[2]["right_grip"]
            for call in self.rig.buses["right"].calls
            if call[0] == "sync_write" and call[1] == "Goal_Position" and "right_grip" in call[2]
        ]
        self.assertTrue(written and all(low <= goal <= high for goal in written), "never past a measured endpoint")

    def test_the_arm_joints_of_the_gripping_hand_stay_dark(self) -> None:
        motors = {motor for call in self.rig.enables("right") for motor in call[1]}
        self.assertEqual(motors, {"right_grip"})

    def test_the_jaw_on_air_reaches_full_travel_and_no_motor_is_ever_enabled(self) -> None:
        grip = self._held("left")[-1]
        self.assertGreater(grip.follower, 0.99)
        self.assertEqual(grip.effective, 0.0)
        self.assertEqual(self.rig.enables("left"), [])

    def test_opening_the_jaw_lets_go(self) -> None:
        self.assertEqual(self.rig.buses["right"].energized(), [])
        self.assertEqual(self.rig.loops["right"].status(), "free")
        self.assertIsNone(self.rig.bridge.fault)


if __name__ == "__main__":
    unittest.main()
