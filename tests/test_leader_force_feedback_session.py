# SPDX-License-Identifier: Apache-2.0
"""Leader force feedback: the two-clock exchange, and the session's way in and out.

Where ``test_leader_force_feedback_servos.py`` stops at one leader thread's
tick, this module is that thread beside a page that steps on a clock of its own:

* the **comparison** -- which leader pose a follower sample is measured
  against: the one the arm was chasing, or the hand's pose right now;
* the **exchange** -- one page step: the leader's pose in, the measured arm and
  what it was chasing back out;
* the **thread** -- the loop running for real, and what a fault on it does;
* the **session** -- :class:`Teleoperation`'s way in and, above all, its way
  out: every arm slack and every register restored whatever went wrong.

**No servo is involved.** The bus is ``FakeBus``, a table of registers that
records calls in order; see ``tests/fixtures/leader_feedback.py``.
"""

from __future__ import annotations

import threading
import time
import unittest

from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop.lerobot_controller import LeaderPort, LeaderUnavailable
from tests.fixtures.leader_feedback import FakeBus, FakeController
from tests.fixtures.leader_feedback import layout as _layout
from tests.fixtures.leader_feedback import servos as _servos
from tests.fixtures.leader_session import FakeArmPanel as _FakeArmPanel
from tests.fixtures.leader_session import FakeEnv as _FakeEnv
from tests.fixtures.leader_session import FakeSession as _FakeSession
from tests.fixtures.leader_session import InstantWaits
from tests.fixtures.leader_session import Loop as _Loop

TIER = "fast"
"""No GPU, no scene, no serial port; the one real thread is joined inside its test."""


class _Instant(InstantWaits, unittest.TestCase):
    """A test case whose torque-off settles and command pacing cost nothing."""


class TestWhatASampleIsComparedAgainst(_Instant):
    """``compare``: the pose the arm was chasing, or the hand's pose right now."""

    def setUp(self):
        super().setUp()
        self.servos, self.bus = _servos()

    def _loop(self, compare):
        loop = _Loop(self.servos, compare=compare)
        loop.gates = [ffb.JointGate(active=False, neutralized=True) for _ in range(3)]
        return loop

    def test_a_hand_that_moved_on_after_the_sample_is_not_contact(self):
        """The arm is exactly where it was told to be; the hand is 0.3 rad on."""
        self.bus.registers["j0"]["Present_Position"] = 1000.3  # ticks are radians in this layout
        loop = self._loop("applied")
        loop.publish_follower([0.0, 0.0, 0.0], 0.0, reference=[0.0, 0.0, 0.0])
        loop.tick(0.01)
        self.assertEqual(loop.decisions[0].effective, 0.0)
        self.assertEqual(self.bus.energized(), [])

    def test_the_live_form_reads_the_same_sample_as_a_gap(self):
        self.bus.registers["j0"]["Present_Position"] = 1000.3
        loop = self._loop("live")
        loop.publish_follower([0.0, 0.0, 0.0], 0.0, reference=[0.0, 0.0, 0.0])
        loop.tick(0.01)
        self.assertLess(loop.decisions[0].effective, 0.0)
        self.assertEqual(self.bus.energized(), ["j0"])

    def test_an_arm_short_of_the_pose_it_was_chasing_is_contact_either_way(self):
        for compare in ffb.COMPARISONS:
            with self.subTest(compare=compare):
                servos, bus = _servos()
                loop = _Loop(servos, compare=compare)
                loop.gates = [ffb.JointGate(active=False, neutralized=True) for _ in range(3)]
                bus.registers["j0"]["Present_Position"] = 1000.3
                loop.publish_follower([0.0, 0.0, 0.0], 0.0, reference=[0.3, 0.0, 0.0])
                loop.tick(0.01)
                self.assertEqual(bus.energized(), ["j0"])
                self.assertAlmostEqual(
                    loop.decisions[0].target, 0.3 - 0.2, msg="built on the hand, pulled back by the cap"
                )

    def test_a_sample_with_no_reference_yet_falls_back_to_the_hand(self):
        """The first sample of a session was chasing nothing this module applied."""
        self.bus.registers["j0"]["Present_Position"] = 1000.3
        loop = self._loop("applied")
        loop.publish_follower([0.0, 0.0, 0.0], 0.0)
        loop.tick(0.01)
        self.assertEqual(self.bus.energized(), ["j0"])

    def test_an_unknown_comparison_is_refused(self):
        with self.assertRaises(ValueError):
            _Loop(self.servos, compare="psychic")


class _FakeIK:
    def __init__(self, coords):
        self.arm_coords = list(coords)


class _FakeJaw:
    """One driven finger travelling 0.04 -> 0, and a carried one nothing reads."""

    coords = (2,)
    open_q = (0.04,)
    closed_q = (0.0,)

    def closed_fraction(self, state):
        raise AssertionError("the gap between the pads is not what the grip slot compares")


class _FakeArray:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class _BridgeEnv:
    """One arm of two joints and a jaw, whose ``joint_q`` a test sets directly."""

    def __init__(self):
        self.joint_q = [0.0, 0.0, 0.04, -0.04]
        self.state = type("State", (), {"joint_q": _FakeArray(self.joint_q)})()
        self.control = object()
        self.applied = []
        self.scene = type("Scene", (), {"embodiment": type("Emb", (), {"iks": {"left": None}})()})()
        env = self

        class _Space:
            per_arm = 3
            arms = ((_FakeIK((0, 1)), _FakeJaw()),)

            def target_action(self, control):
                return [0.0, 0.0, 0.0]

            def apply(self, action, control, state):
                env.applied.append([float(v) for v in action])

        self.space = _Space()

    def use_control(self, mode):
        return self.space


class TestTheBridgeExchange(_Instant):
    """One page step: the leader's pose in, the measured arm and what it was chasing back out."""

    def setUp(self):
        super().setUp()
        self.servos, self.bus = _servos()
        self.loop = _Loop(self.servos)
        self.env = _BridgeEnv()
        self.clock = [0.0]
        self.bridge = ffb.StudioBridge(_FakeSession(self.env), {"left": self.loop}, clock=lambda: self.clock[0])

    def _hand_at(self, ticks):
        self.bus.registers["j0"]["Present_Position"] = ticks
        self.loop.tick(self.clock[0])

    def test_the_leader_pose_is_applied_as_one_action(self):
        self._hand_at(1000.5)
        self.bridge.pump()
        self.assertEqual(self.env.applied, [[0.5, 0.0, 0.0]])

    def test_the_first_sample_was_chasing_nothing_and_says_so(self):
        self._hand_at(1000.5)
        self.bridge.pump()
        self.assertIsNone(self.loop._take_follower().reference)

    def test_each_later_sample_carries_the_pose_applied_the_step_before(self):
        self._hand_at(1000.5)
        self.bridge.pump()
        self.clock[0] = 0.02
        self._hand_at(1000.9)
        self.bridge.pump()
        sample = self.loop._take_follower()
        self.assertEqual(sample.reference, [0.5, 0.0, 0.0], "not the 0.9 applied a moment ago")
        self.assertEqual(sample.at, 0.02)

    def test_the_jaw_is_measured_on_the_driven_finger(self):
        """Half its travel is half shut, whatever the carried finger is doing."""
        self.env.joint_q[2] = 0.02
        self.env.joint_q[3] = -0.04  # the carried finger has not moved at all
        self._hand_at(1000.0)
        self.bridge.pump()
        self.assertAlmostEqual(self.loop._take_follower().values[-1], 0.5)

    def test_an_arm_with_no_leader_reading_yet_is_left_where_it_is(self):
        self.bridge.pump()
        self.assertEqual(self.env.applied, [[0.0, 0.0, 0.0]], "its own current target, written back")


class TestTheThread(_Instant):
    """The loop on its own clock: it starts, faults and stops with torque off."""

    def test_stopping_the_thread_leaves_every_motor_slack(self):
        servos, bus = _servos()
        loop = _Loop(servos, rate_hz=200.0)
        loop.gates = [ffb.JointGate(active=False, neutralized=True) for _ in range(3)]
        loop.start()
        deadline = time.monotonic() + 2.0
        while not bus.energized() and time.monotonic() < deadline:
            loop.publish_follower([0.4, 0.0, 0.0], time.monotonic())
            time.sleep(0.002)
        self.assertEqual(bus.energized(), ["j0"], "the thread armed the joint in contact")
        loop.stop()
        self.assertEqual(bus.energized(), [], "and its own way out disarmed it")
        self.assertIsNone(loop.fault)

    def test_a_run_of_bus_glitches_is_a_fault_and_the_arm_goes_slack(self):
        servos, bus = _servos()
        real = servos.read_ticks
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] > 3:
                raise ConnectionError("no status packet")
            return real()

        servos.read_ticks = flaky
        loop = _Loop(servos, rate_hz=500.0)
        loop.start()
        loop._thread.join(timeout=2.0)
        self.assertFalse(loop._thread.is_alive())
        self.assertIsInstance(loop.fault, ffb.ServoFault)
        self.assertIn("bus glitches", str(loop.fault))
        self.assertEqual(bus.energized(), [])
        self.assertEqual(loop.status(), "fault: ServoFault")

    def test_one_dropped_packet_is_ridden_through(self):
        servos, _ = _servos()
        real = servos.read_ticks
        state = {"n": 0}

        def once():
            state["n"] += 1
            if state["n"] == 2:
                raise ConnectionError("no status packet")
            return real()

        servos.read_ticks = once
        loop = _Loop(servos, rate_hz=500.0, feedback=False)
        loop.start()
        deadline = time.monotonic() + 2.0
        while state["n"] < 6 and time.monotonic() < deadline:
            time.sleep(0.002)
        loop.stop()
        self.assertIsNone(loop.fault)
        self.assertGreaterEqual(state["n"], 6)

    def test_stopping_a_thread_that_never_started_is_not_an_error(self):
        servos, _ = _servos()
        _Loop(servos).stop()


class TestTheBridgeLetsGo(_Instant):
    def test_a_released_bridge_stops_driving_and_hands_the_arm_back(self):
        """A page keeps its ``each_step`` hooks; a bridge whose leaders are gone
        must not go on applying the last pose a dead thread published."""
        servos, _ = _servos()
        arm = _FakeArmPanel()
        session = _FakeSession(_FakeEnv(("left",), 3), {"left": arm})
        bridge = ffb.StudioBridge(session, {"left": _Loop(servos)})
        bridge._env = session.env
        bridge.driven = ["left"]
        arm.suspend()
        bridge.release()
        self.assertFalse(arm.suspended)
        calls = []
        bridge._pump = lambda: calls.append(1)
        bridge.pump()
        self.assertEqual(calls, [])

    def test_a_loop_fault_asks_the_page_to_stop(self):
        servos, _ = _servos()
        loop = _Loop(servos)
        loop.fault = ffb.ServoFault("bus died")
        session = _FakeSession(_FakeEnv(("left",), 3))
        bridge = ffb.StudioBridge(session, {"left": loop})
        bridge.pump()
        self.assertTrue(session.quit_requested)
        self.assertIs(bridge.fault, loop.fault)


def _leaders(*names, joints=2):
    return {
        name: LeaderPort(
            name=name, port=f"/dev/fake-{name}", serial=name, calibration=_layout(joints=joints).calibration
        )
        for name in names
    }


class TestTeleoperation(_Instant):
    """The whole session's way in, and its way out."""

    def setUp(self):
        super().setUp()
        self.buses: dict[str, FakeBus] = {}
        self.said: list[str] = []

    def _connect(self, port, arms):
        bus = FakeBus(_layout(joints=len(arms[0].joints)).motors)
        self.buses[port] = bus
        return FakeController(bus)

    def _session(self, slots=("left", "right"), per_arm=3):
        return _FakeSession(_FakeEnv(slots, per_arm), {name: _FakeArmPanel() for name in slots})

    def _teleop(self, session, leaders, **kwargs):
        kwargs.setdefault("connect", self._connect)
        return ffb.Teleoperation(session, leaders, notify=self.said.append, **kwargs)

    def test_entering_connects_configures_hooks_and_starts_each_arm(self):
        session = self._session()
        with self._teleop(session, _leaders("left", "right")) as teleop:
            self.assertEqual(set(teleop.loops), {"left", "right"})
            self.assertEqual(len(session.hooks), 1, "one hook on the page for both arms")
            self.assertTrue(all(loop._thread.is_alive() for loop in teleop.loops.values()))
            for bus in self.buses.values():
                self.assertEqual(bus.registers["j0"]["P_Coefficient"], 30, "feedback gains installed")
                self.assertEqual(bus.energized(), [], "and nothing energized by connecting")
            self.assertEqual(set(teleop.status()), {"left", "right"})
        self.assertEqual(teleop.exit_code, 0)
        self.assertIn("Stopped; leader torque is off.", "".join(self.said))

    def test_leaving_puts_every_register_back_and_every_motor_slack(self):
        session = self._session()
        with self._teleop(session, _leaders("left", "right")):
            pass
        for bus in self.buses.values():
            self.assertEqual(bus.energized(), [])
            for motor in bus.registers:
                self.assertEqual(bus.registers[motor]["P_Coefficient"], 12)
                self.assertEqual(bus.registers[motor]["Operating_Mode"], 3)
                self.assertEqual(bus.registers[motor]["Torque_Limit"], 0)
            self.assertEqual(bus.calls[-1], ("disconnect",))
        self.assertEqual(session.closed, 1, "the page is closed with it")
        self.assertFalse(any(t.name.startswith("leader-") and t.is_alive() for t in threading.enumerate()))

    def test_without_feedback_no_gain_is_written_and_no_motor_is_ever_enabled(self):
        session = self._session()
        with self._teleop(session, _leaders("left"), feedback=False) as teleop:
            loop = teleop.loops["left"]
            deadline = time.monotonic() + 2.0
            while loop.latest_leader() is None and time.monotonic() < deadline:
                time.sleep(0.002)
            loop.publish_follower([3.0, 3.0, 1.0], time.monotonic())
            time.sleep(0.05)
        bus = self.buses["/dev/fake-left"]
        self.assertNotIn("enable_torque", [call[0] for call in bus.calls])
        written = {call[1] for call in bus.calls if call[0] in ("write", "sync_write")}
        self.assertEqual(written, {"Torque_Limit"})
        self.assertIn("no motor will be energized", "".join(self.said))

    def test_a_leader_the_robot_has_no_arm_for_is_refused_before_any_bus_is_opened(self):
        session = self._session(slots=("left", "right"))
        with self.assertRaises(ffb.LeaderMismatch), self._teleop(session, _leaders("middle")):
            self.fail("the body must not run")
        self.assertEqual(self.buses, {}, "no port was opened")
        self.assertEqual(session.closed, 1, "and the page is closed rather than left serving")

    def test_a_second_arm_that_fails_to_connect_closes_the_first(self):
        """A partial connect must not leave one arm holding feedback gains."""

        def connect(port, arms):
            if port.endswith("right"):
                raise ConnectionError("could not open port")
            return self._connect(port, arms)

        session = self._session()
        with (
            self.assertRaises(LeaderUnavailable) as caught,
            self._teleop(session, _leaders("left", "right"), connect=connect),
        ):
            self.fail("the body must not run")
        self.assertIn("/dev/fake-right", str(caught.exception))
        left = self.buses["/dev/fake-left"]
        self.assertEqual(left.registers["j0"]["P_Coefficient"], 12, "the gains it installed were put back")
        self.assertEqual(left.energized(), [])
        self.assertEqual(left.calls[-1], ("disconnect",))
        self.assertEqual(session.hooks, [], "and nothing was hooked onto the page")

    def test_an_arm_that_faults_while_connecting_is_itself_closed(self):
        """The arm whose own connect raised has a bus open too."""

        def hot(port, arms):
            controller = self._connect(port, arms)
            real_read = controller.bus.read

            def read(data_name, motor, **kwargs):
                return 500 if data_name == "Torque_Limit" else real_read(data_name, motor, **kwargs)

            controller.bus.read = read
            return controller

        session = self._session()
        with self.assertRaises(ffb.ServoFault) as caught, self._teleop(session, _leaders("left"), connect=hot):
            self.fail("the body must not run")
        self.assertIn("nonzero torque limits", str(caught.exception))
        self.assertEqual(self.buses["/dev/fake-left"].calls[-1], ("disconnect",))

    def test_ctrl_c_is_how_a_session_ends_and_is_not_an_error(self):
        session = self._session()
        with self._teleop(session, _leaders("left")) as teleop:
            raise KeyboardInterrupt
        self.assertEqual(teleop.exit_code, 0)
        self.assertEqual(self.buses["/dev/fake-left"].energized(), [])

    def test_an_exception_in_the_caller_s_code_still_tears_down_and_then_propagates(self):
        session = self._session()
        with self.assertRaises(ZeroDivisionError), self._teleop(session, _leaders("left")) as teleop:
            raise ZeroDivisionError
        self.assertEqual(teleop.exit_code, 1)
        bus = self.buses["/dev/fake-left"]
        self.assertEqual(bus.registers["j0"]["P_Coefficient"], 12)
        self.assertEqual(bus.energized(), [])

    def test_a_fault_on_a_leader_thread_is_the_session_s_exit_code(self):
        session = self._session()
        with self._teleop(session, _leaders("left")) as teleop:
            loop = teleop.loops["left"]
            loop.servos.read_ticks = lambda: (_ for _ in ()).throw(ConnectionError("no status packet"))
            loop._thread.join(timeout=2.0)
            teleop.bridge.pump()
            self.assertTrue(session.quit_requested, "the page is asked to stop")
        self.assertEqual(teleop.exit_code, 1)
        self.assertIsInstance(teleop.fatal, ffb.ServoFault)
        self.assertEqual(self.buses["/dev/fake-left"].energized(), [])

    def test_a_trace_is_written_on_the_way_out(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            session = self._session()
            with self._teleop(session, _leaders("left"), trace=path) as teleop:
                loop = teleop.loops["left"]
                deadline = time.monotonic() + 2.0
                while len(teleop.trace.ticks["left"]) < 5 and time.monotonic() < deadline:
                    loop.publish_follower([0.0, 0.0, 0.0], time.monotonic())
                    time.sleep(0.002)
            doc = json.loads(path.read_text())
            self.assertEqual(doc["joints"], {"left": ["j0", "j1", "grip"]})
            self.assertEqual(set(doc["arms"]["left"]), set(ffb.Trace.TICK_FIELDS))
            self.assertGreaterEqual(len(doc["arms"]["left"]["t"]), 5)

    def test_a_rate_out_of_range_is_refused_before_anything_is_opened(self):
        with self.assertRaises(ValueError):
            self._teleop(self._session(), _leaders("left"), rate_hz=0.0)
        self.assertEqual(self.buses, {})


if __name__ == "__main__":
    unittest.main()
