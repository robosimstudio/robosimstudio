# SPDX-License-Identifier: Apache-2.0
"""Leader force feedback: what reaches the servos, and one leader thread's tick.

The other half of :mod:`robosimstudio.robots.teleop.force_feedback`, where the
decisions from ``test_leader_force_feedback.py`` meet a bus:

* the **rendering** -- the order goals, torque limits and torque enables reach
  a bus, which is where a mistake energizes a motor against a stale goal;
* the **preflight** -- refusing a leader that does not match the robot before
  any bus is opened;
* the **loop** -- one leader thread's tick, and what a stale simulation sample
  does to it. A page that is paused, rebuilding a scene, or gone publishes
  nothing, and force rendered against a mismatch that old would push the
  operator's hand toward where the simulated arm used to be.

The two-clock exchange with the page and the whole session's way in and out are
in ``test_leader_force_feedback_session.py``.

**No servo is involved.** The bus is ``FakeBus``, a table of registers that
records calls in order, because with force feedback the order *is* the safety
claim. What these prove is which registers this code writes and when; that a
real STS3215 then does what its register says is the rig checklist's to show
(``examples/06_teleop/README.md``, its rig checklist).
"""

from __future__ import annotations

import unittest

from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop.lerobot_controller import LeaderUnavailable
from tests.fixtures.leader_feedback import FakeBus, FakeController
from tests.fixtures.leader_feedback import layout as _layout
from tests.fixtures.leader_feedback import servos as _servos
from tests.fixtures.leader_session import FakeEnv as _FakeEnv
from tests.fixtures.leader_session import FakeSession as _FakeSession
from tests.fixtures.leader_session import InstantWaits
from tests.fixtures.leader_session import Loop as _Loop

TIER = "fast"
"""No GPU, no scene, no serial port."""


class _Instant(InstantWaits, unittest.TestCase):
    """A test case whose torque-off settles and command pacing cost nothing."""


class TestTheServoRendering(_Instant):
    """What actually reaches the bus, and in what order."""

    def test_configuring_leaves_the_arm_slack_and_parked_where_it_stands(self):
        servos, bus = _servos()
        servos._configure(servos.layout.motors)
        self.assertTrue(servos.armed)
        for motor in servos.layout.motors:
            self.assertEqual(bus.registers[motor]["Torque_Enable"], 0, "configured is not energized")
            self.assertEqual(bus.registers[motor]["Torque_Limit"], 0)
            self.assertEqual(bus.registers[motor]["Operating_Mode"], 0)
            self.assertEqual(bus.registers[motor]["P_Coefficient"], 30)
            parked = 0 if motor == "grip" else 1000
            self.assertEqual(bus.registers[motor]["Goal_Position"], parked, "parked at the measured pose")

    def test_the_originals_are_kept_for_the_way_out(self):
        servos, _ = _servos()
        servos._configure(servos.layout.motors)
        self.assertEqual(servos.saved["j0"]["P_Coefficient"], 12)
        self.assertEqual(servos.saved["j0"]["Operating_Mode"], 3)

    def test_a_second_configure_does_not_save_the_feedback_gains_as_originals(self):
        """After a suspend/resume the live values are ours, not the servo's."""
        servos, _ = _servos()
        servos._configure(servos.layout.motors)
        servos._configure(servos.layout.motors)
        self.assertEqual(servos.saved["j0"]["P_Coefficient"], 12)

    def test_arming_against_a_servo_that_came_up_hot_is_refused(self):
        """A nonzero torque limit at startup means something else is driving."""
        layout = _layout()
        servos, _ = _servos(FakeBus(layout.motors, torque_limit=500), layout)
        with self.assertRaises(ffb.ServoFault):
            servos._refuse_nonzero_limits(layout.motors)

    def test_a_servo_that_will_not_let_go_is_a_fault(self):
        layout = _layout()
        bus = FakeBus(layout.motors)

        def _stuck(motors=None, num_retry=0):
            bus.calls.append(("disable_torque", motors))

        bus.registers["j1"]["Torque_Enable"] = 1
        bus.disable_torque = _stuck
        servos, _ = _servos(bus, layout)
        with self.assertRaises(ffb.ServoFault) as caught:
            servos._torque_off(layout.motors, passes=1)
        self.assertIn("j1", str(caught.exception))

    def test_an_unreadable_torque_gate_counts_as_still_energized(self):
        """The one wrong answer this check must never give is a false "off"."""
        layout = _layout()
        bus = FakeBus(layout.motors)

        def _blind(data_name, motor, *, normalize=True, num_retry=0):
            raise ConnectionError("no status packet")

        bus.read = _blind
        servos, _ = _servos(bus, layout)
        self.assertEqual(servos._still_energized(layout.motors), layout.motors)

    def test_rendering_disarms_before_it_arms(self):
        """Never add torque on a tick that is also removing some."""
        servos, bus = _servos()
        proposal = ffb.propose(
            servos.config,
            servos.layout,
            leader=[0.0, 0.0, 0.0],
            follower=[0.0, 0.4, 0.0],
            gates=[
                ffb.JointGate(active=True, neutralized=True),
                ffb.JointGate(active=False, neutralized=True),
                ffb.JointGate(active=False, neutralized=True),
            ],
            now=0.0,
            neutral_dwell_s=0.5,
        )
        servos.render(proposal)
        order = [call[0] for call in bus.calls]
        self.assertIn("enable_torque", order)
        self.assertLess(order.index("disable_torque"), order.index("enable_torque"))

    def test_arming_writes_the_goal_and_the_limit_before_enabling_torque(self):
        """Acknowledged, and in that order: enabling against a stale goal is
        the failure this pays two round trips to avoid."""
        servos, bus = _servos()
        proposal = ffb.propose(
            servos.config,
            servos.layout,
            leader=[0.0, 0.0, 0.0],
            follower=[0.4, 0.0, 0.0],
            gates=[ffb.JointGate(active=False, neutralized=True) for _ in range(3)],
            now=0.0,
            neutral_dwell_s=0.5,
        )
        servos.render(proposal)
        writes = [call for call in bus.calls if call[0] in ("write", "enable_torque")]
        self.assertEqual(writes[0][:3], ("write", "Goal_Position", "j0"))
        self.assertEqual(writes[1][:3], ("write", "Torque_Limit", "j0"))
        self.assertEqual(writes[1][3], servos.config.torque_limit)
        self.assertEqual(writes[2], ("enable_torque", ("j0",)))

    def test_the_goal_written_is_never_further_than_the_cap_from_the_hand(self):
        """However far the simulated arm is, the servo is asked for 0.2 rad."""
        servos, bus = _servos()
        proposal = ffb.propose(
            servos.config,
            servos.layout,
            leader=[0.0, 0.0, 0.0],
            follower=[3.0, 0.0, 0.0],
            gates=[ffb.JointGate(active=False, neutralized=True) for _ in range(3)],
            now=0.0,
            neutral_dwell_s=0.5,
        )
        servos.render(proposal)
        goal = next(call for call in bus.calls if call[:3] == ("write", "Goal_Position", "j0"))
        self.assertEqual(goal[3], 1000, "1000 ticks is 0 rad here, and 0.2 rad rounds to no whole tick")
        self.assertAlmostEqual(proposal.decisions[0].target, servos.config.max_error)

    def test_a_continuing_joint_rides_one_sync_write(self):
        """Already energized against a recent goal; the fast path is safe."""
        servos, bus = _servos()
        proposal = ffb.propose(
            servos.config,
            servos.layout,
            leader=[0.0, 0.0, 0.0],
            follower=[0.4, 0.0, 0.0],
            gates=[ffb.JointGate(active=True, neutralized=True) for _ in range(3)],
            now=0.0,
            neutral_dwell_s=0.5,
        )
        servos.render(proposal)
        goals = [call for call in bus.calls if call[0] == "sync_write" and call[1] == "Goal_Position"]
        self.assertEqual(len(goals), 1, "one broadcast carries every continuing goal")
        self.assertEqual(set(goals[0][2]), {"j0"}, "only the joint still in contact")
        self.assertNotIn("enable_torque", [call[0] for call in bus.calls])
        self.assertNotIn("write", [call[0] for call in bus.calls], "and no acknowledged write is paid for")

    def test_closing_restores_the_registers_it_changed(self):
        servos, bus = _servos()
        servos._configure(servos.layout.motors)
        servos.close()
        self.assertEqual(bus.registers["j0"]["P_Coefficient"], 12)
        self.assertEqual(bus.registers["j0"]["Operating_Mode"], 3)
        self.assertEqual(bus.registers["j0"]["Torque_Enable"], 0)
        self.assertEqual(bus.calls[-1], ("disconnect",))

    def test_closing_is_idempotent(self):
        servos, _ = _servos()
        servos._configure(servos.layout.motors)
        servos.close()
        servos.close()


class TestConnecting(_Instant):
    """The one method that opens a port, driven through its ``connect`` seam."""

    def _servos(self, bus=None, *, connect=None):
        layout = _layout()
        bus = bus if bus is not None else FakeBus(layout.motors)
        built = ffb.LeaderServos(
            "/dev/fake", layout, ffb.ComplianceConfig(), connect=connect or (lambda port, arms: FakeController(bus))
        )
        return built, bus

    def test_connecting_takes_torque_off_before_anything_else_is_written(self):
        """An arm that came up energized is made slack first, and proven so."""
        servos, bus = self._servos(FakeBus(_layout().motors, torque_enable=1))
        servos.connect(configure=True)
        first_write = next(i for i, call in enumerate(bus.calls) if call[0] in ("write", "sync_write"))
        first_off = next(i for i, call in enumerate(bus.calls) if call[0] == "disable_torque")
        self.assertLess(first_off, first_write)
        self.assertEqual(bus.energized(), [])
        self.assertTrue(servos.armed)

    def test_without_feedback_no_eeprom_register_is_written(self):
        """The dry run: torque off and verified, and the gains left alone."""
        servos, bus = self._servos()
        servos.connect(configure=False)
        written = {call[1] for call in bus.calls if call[0] in ("write", "sync_write")}
        self.assertEqual(written, {"Torque_Limit"}, "zeroing the torque limit is the only write")
        self.assertFalse(servos.armed)
        self.assertEqual(bus.registers["j0"]["P_Coefficient"], 12)

    def test_a_missing_driver_is_one_sentence_naming_the_extra(self):
        def no_driver(port, arms):
            raise ModuleNotFoundError("No module named 'lerobot'")

        servos, _ = self._servos(connect=no_driver)
        with self.assertRaises(LeaderUnavailable) as caught:
            servos.connect()
        self.assertIn("robosimstudio[lerobot-feetech]", str(caught.exception))

    def test_a_port_that_will_not_open_is_one_sentence_naming_the_port(self):
        def dead(port, arms):
            raise ConnectionError("could not open port")

        servos, _ = self._servos(connect=dead)
        with self.assertRaises(LeaderUnavailable) as caught:
            servos.connect()
        self.assertIn("/dev/fake", str(caught.exception))
        self.assertIn("powered", str(caught.exception))


class TestPreflight(_Instant):
    """A leader that does not match the robot is refused while torque is off."""

    def _bridge(self, slots, per_arm, names):
        servos, _ = _servos()
        loops = {name: _Loop(servos) for name in names}
        return ffb.StudioBridge(_FakeSession(_FakeEnv(slots, per_arm)), loops)

    def test_an_arm_this_robot_does_not_have_is_refused(self):
        """Otherwise the leader is simply never applied, silently."""
        bridge = self._bridge(("left", "right"), 3, ["middle"])
        with self.assertRaises(ffb.LeaderMismatch) as caught:
            bridge.preflight()
        self.assertIn("middle", str(caught.exception))
        self.assertIn("left", str(caught.exception), "and it says what the robot does have")

    def test_a_calibration_of_the_wrong_width_is_refused(self):
        """A mismatched slice would drive the wrong joints of the right arm."""
        bridge = self._bridge(("left",), 7, ["left"])
        with self.assertRaises(ffb.LeaderMismatch) as caught:
            bridge.preflight()
        self.assertIn("2 joints plus a grip", str(caught.exception))
        self.assertIn("6 plus a grip", str(caught.exception))

    def test_a_matching_leader_passes(self):
        self._bridge(("left", "right"), 3, ["left"]).preflight()

    def test_an_arm_with_no_leader_is_not_checked(self):
        """Half a bimanual rig is a supported configuration, not an error."""
        self._bridge(("left", "right"), 3, []).preflight()


class TestTheLoop(_Instant):
    """The exchange between the two clocks, and what staleness does to it."""

    def setUp(self):
        super().setUp()
        self.servos, self.bus = _servos()

    def test_a_tick_publishes_what_it_read_for_the_page_to_apply(self):
        loop = _Loop(self.servos, feedback=False)
        self.bus.registers["j0"]["Present_Position"] = 1002
        loop.tick(0.0)
        self.assertAlmostEqual(loop.latest_leader()[0], 2.0)

    def test_no_feedback_never_touches_a_torque_register(self):
        loop = _Loop(self.servos, feedback=False)
        loop.publish_follower([5.0, 5.0, 1.0], 0.0)
        loop.tick(0.0)
        self.assertEqual({call[0] for call in self.bus.calls}, {"sync_read"}, "a read, and nothing written")

    def _live(self, loop, follower=(0.4, 0.0, 0.0), at=0.0):
        """Drive one rendered tick, so the loop is out of its initial suspension."""
        loop.gates = [ffb.JointGate(active=False, neutralized=True) for _ in range(3)]
        loop.publish_follower(list(follower), at)
        loop.tick(at + 0.01)
        self.assertFalse(loop.suspended)
        return loop

    def test_a_stale_follower_sample_suspends_rather_than_pulling(self):
        """The paused page. The mismatch is huge and must render nothing."""
        loop = self._live(_Loop(self.servos))
        self.bus.calls.clear()
        loop.publish_follower([2.0, 2.0, 1.0], 0.0)
        loop.tick(1.0)
        self.assertTrue(loop.suspended)
        self.assertNotIn("enable_torque", [call[0] for call in self.bus.calls])
        self.assertEqual(self.bus.energized(), [], "and the joint that was live is let go")

    def test_a_suspend_puts_every_joint_back_behind_the_dwell(self):
        loop = self._live(_Loop(self.servos))
        loop.tick(1.0)
        self.assertTrue(all(not gate.neutralized for gate in loop.gates))
        self.assertEqual(loop.decisions, [])

    def test_staying_suspended_costs_the_bus_nothing_beyond_the_read(self):
        """A paused page must not drop this loop to the rate of a torque-off.

        Disarming again every tick would sleep and read back every motor to
        re-confirm what it already did -- turning a 100 Hz loop into about four,
        for as long as somebody left the page paused.
        """
        loop = self._live(_Loop(self.servos))
        loop.tick(1.0)
        self.bus.calls.clear()
        loop.tick(1.1)
        loop.tick(1.2)
        self.assertEqual(
            {call[0] for call in self.bus.calls},
            {"sync_read"},
            "reading the leader is all a suspended tick may do",
        )

    def test_the_stale_limit_scales_with_the_page_s_own_cadence(self):
        """A heavy scene must not be mistaken for a stalled one.

        A contact-rich scene can spend 300 ms per step in physics alone, so a
        flat 250 ms floor would call a healthy page stalled on nearly every step
        and thrash the arm between suspended and re-arming.
        """
        loop = _Loop(self.servos, stale_after_s=0.25, stale_factor=3.0)
        self.assertEqual(loop.stale_limit(), 0.25, "no cadence yet, so the floor applies")
        for i in range(5):
            loop.publish_follower([0.0, 0.0, 0.0], i * 0.40)
        self.assertAlmostEqual(loop.stale_limit(), 1.20, places=6, msg="3 x a 400 ms step")

    def test_a_fast_page_keeps_the_floor(self):
        """Scaling a 5 ms cadence would drop the arm 15 ms after a pause."""
        loop = _Loop(self.servos, stale_after_s=0.25, stale_factor=3.0)
        for i in range(5):
            loop.publish_follower([0.0, 0.0, 0.0], i * 0.005)
        self.assertEqual(loop.stale_limit(), 0.25)

    def test_an_absence_does_not_teach_the_loop_that_slow_is_normal(self):
        """The gap across a pause is the thing being detected, not cadence."""
        loop = _Loop(self.servos, stale_after_s=0.25, stale_factor=3.0)
        for i in range(4):
            loop.publish_follower([0.0, 0.0, 0.0], i * 0.10)
        loop.publish_follower([0.0, 0.0, 0.0], 60.0)
        self.assertAlmostEqual(loop.stale_limit(), 0.30, places=6, msg="still 3 x 100 ms")

    def test_a_disarm_that_fails_is_retried_rather_than_remembered_as_done(self):
        """``suspended`` is committed only once the torque actually came off."""
        loop = self._live(_Loop(self.servos))
        self.servos.disarm = lambda: (_ for _ in ()).throw(ffb.ServoFault("bus died"))
        with self.assertRaises(ffb.ServoFault):
            loop.tick(1.0)
        self.assertFalse(loop.suspended, "so the next tick tries again")

    def test_a_fresh_sample_is_rendered(self):
        loop = _Loop(self.servos)
        loop.gates = [ffb.JointGate(active=False, neutralized=True) for _ in range(3)]
        loop.publish_follower([0.4, 0.0, 0.0], 0.0)
        loop.tick(0.05)
        self.assertFalse(loop.suspended)
        self.assertEqual([call for call in self.bus.calls if call[0] == "enable_torque"], [("enable_torque", ("j0",))])

    def test_a_failed_render_leaves_the_old_gates_so_the_disarm_is_retried(self):
        """The two-phase commit: believing torque is off when it is not is the
        one state this may never reach."""
        loop = _Loop(self.servos)
        armed = [ffb.JointGate(active=True, neutralized=True) for _ in range(3)]
        loop.gates = list(armed)
        self.servos.render = lambda proposal: (_ for _ in ()).throw(ffb.ServoFault("bus died"))
        loop.publish_follower([0.0, 0.0, 0.0], 0.0)
        with self.assertRaises(ffb.ServoFault):
            loop.tick(0.05)
        self.assertTrue(all(gate.active for gate in loop.gates), "still considered live")

    def test_nothing_energizes_off_the_first_samples_after_a_connect(self):
        """A fresh loop is behind the dwell: a large mismatch at startup -- the
        leader held somewhere the simulated arm is not -- renders nothing."""
        loop = _Loop(self.servos)
        for i in range(10):
            loop.publish_follower([1.0, 1.0, 1.0], i * 0.01)
            loop.tick(i * 0.01 + 0.005)
        self.assertNotIn("enable_torque", [call[0] for call in self.bus.calls])
        self.assertEqual(loop.status(), "waiting for alignment")

    def test_the_readout_names_the_motor_that_is_pushing_back(self):
        loop = self._live(_Loop(self.servos))
        self.assertEqual(loop.status(), "pushing back: j0 +0.40")
        loop.publish_follower([0.0, 0.0, 0.0], 0.02)
        loop.tick(0.03)
        self.assertEqual(loop.status(), "free")
        self.assertEqual(_Loop(self.servos, feedback=False).status(), "reading only (no motor is energized)")

    def test_the_hand_s_own_speed_raises_the_threshold_it_is_judged_against(self):
        """Two ticks 10 ms apart, the hand 0.01 rad further on: 1 rad/s, and a
        lag-sized 0.07 rad mismatch that a still hand would feel is not felt."""
        loop = _Loop(self.servos)
        loop.gates = [ffb.JointGate(active=False, neutralized=True) for _ in range(3)]
        loop.SPEED_SMOOTHING = 1.0  # the unfiltered speed, so two samples are enough
        loop.publish_follower([0.0, 0.0, 0.0], 0.0)
        loop.tick(0.0)
        self.bus.registers["j0"]["Present_Position"] = 1001  # 1 rad in 10 ms is far past the ceiling's knee
        self.bus.registers["j1"]["Present_Position"] = 1000
        loop.publish_follower([1.0 + 0.10, 0.07, 0.0], 0.01)
        loop.tick(0.01)
        moving, still = loop.decisions[0], loop.decisions[1]
        self.assertAlmostEqual(moving.deadband, loop.config.max_deadband, msg="capped: fast contact is still felt")
        self.assertEqual(moving.effective, 0.0, "0.10 rad behind at speed is inside the scaled deadband")
        self.assertAlmostEqual(still.deadband, loop.config.deadband)
        self.assertGreater(still.effective, 0.0, "the joint that is not moving keeps the floor")


if __name__ == "__main__":
    unittest.main()
