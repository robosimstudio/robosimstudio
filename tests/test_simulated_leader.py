# SPDX-License-Identifier: Apache-2.0
"""The simulated leader: a servo bus over a table, and a scripted hand.

What the examples' ``--simulate`` and the feedback tests stand on, so what it
is has to be exactly what its module says: registers that return what was
written, a hand that goes where the script says, and nothing that behaves like
a motor.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.robots.teleop import simulated
from robosimstudio.robots.teleop.lerobot_controller import LerobotFeetechController

TIER = "fast"
"""No GPU, no scene, no serial port."""

HOME = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestTheBus(unittest.TestCase):
    def test_a_register_returns_what_was_written_to_it(self) -> None:
        bus = simulated.SimulatedBus(["a", "b"])
        bus.write("Goal_Position", "a", 1234)
        bus.sync_write("Torque_Limit", {"a": 5, "b": 6})
        self.assertEqual(bus.read("Goal_Position", "a"), 1234)
        self.assertEqual(bus.sync_read("Torque_Limit"), {"a": 5, "b": 6})

    def test_every_call_is_recorded_in_order(self) -> None:
        bus = simulated.SimulatedBus(["a"])
        bus.write("Goal_Position", "a", 1)
        bus.enable_torque("a")
        bus.disable_torque(["a"])
        self.assertEqual(
            [call[0] for call in bus.calls], ["write", "enable_torque", "disable_torque"], "the order is the claim"
        )
        self.assertEqual(bus.ordered("enable_torque"), [("enable_torque", ("a",))])

    def test_enabling_torque_does_not_move_anything(self) -> None:
        """The one thing this is not: a motor. A goal is a number in a table."""
        bus = simulated.SimulatedBus(["a"], present={"a": 2000})
        bus.write("Goal_Position", "a", 3000)
        bus.enable_torque("a")
        self.assertEqual(bus.energized(), ["a"])
        self.assertEqual(bus.sync_read("Present_Position", ["a"]), {"a": 2000})

    def test_the_hand_moves_the_positions_and_only_on_a_read(self) -> None:
        clock = _Clock()
        bus = simulated.SimulatedBus(["a"], hand=lambda t: {"a": 2000 + int(100 * t)}, clock=clock)
        clock.t = 1.5
        self.assertEqual(bus.sync_read("Present_Position"), {"a": 2150})


class TestTheScriptedHand(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = simulated.direct_drive_calibration("right")
        self.hand = simulated.ScriptedHand(self.calibration, HOME)

    def test_it_starts_at_home_with_the_jaw_open(self) -> None:
        np.testing.assert_allclose(self.hand.pose(0.0), [*HOME, 0.0], atol=1e-12)

    def test_the_base_joint_swings_the_stated_amplitude_each_way(self) -> None:
        base = [self.hand.pose(t / 50.0)[0] for t in range(int(50 * self.hand.period_s))]
        self.assertAlmostEqual(max(base), self.hand.sweep, places=3)
        self.assertAlmostEqual(min(base), -self.hand.sweep, places=3)

    def test_the_jaw_closes_fully_and_opens_again(self) -> None:
        jaw = [self.hand.pose(t / 50.0)[-1] for t in range(int(50 * self.hand.period_s))]
        self.assertAlmostEqual(max(jaw), 1.0, places=3)
        self.assertAlmostEqual(min(jaw), 0.0, places=6)

    def test_it_is_slow_enough_for_a_follower_to_keep_up(self) -> None:
        """0.47 rad/s at the base: well under the speed the free pass in
        ``test_leader_feedback_loop.py`` shows a follower tracking."""
        dt = 0.01
        speed = max(
            abs(self.hand.pose(t * dt + dt)[0] - self.hand.pose(t * dt)[0]) / dt
            for t in range(int(self.hand.period_s / dt))
        )
        self.assertLess(speed, 0.5)

    def test_the_ticks_read_back_as_the_pose_through_the_same_calibration(self) -> None:
        ticks = self.hand(1.7)
        pose = self.hand.pose(1.7)
        for joint, value in zip(self.calibration.joints, pose, strict=False):
            self.assertAlmostEqual(joint.to_radians(ticks[joint.name]), value, delta=0.002)
        self.assertAlmostEqual(self.calibration.gripper.to_command(ticks["right_grip"]), pose[-1], delta=0.002)

    def test_a_home_of_the_wrong_length_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            simulated.ScriptedHand(self.calibration, (0.0, 1.0))


class TestConnecting(unittest.TestCase):
    def test_one_made_up_leader_per_arm_six_joints_and_a_grip(self) -> None:
        leaders = simulated.leaders(("left", "right"))
        self.assertEqual(set(leaders), {"left", "right"})
        self.assertEqual(leaders["right"].port, "simulated-right")
        self.assertEqual(len(leaders["right"].calibration.joints), 6)
        self.assertEqual(leaders["right"].calibration.gripper.name, "right_grip")

    def test_the_connector_is_a_drop_in_for_the_real_connect(self) -> None:
        """Same call, same return type: a controller whose ``read_action`` is
        the hand's pose in the action layout."""
        clock = _Clock()
        leader = simulated.leaders(("right",))["right"]
        connect = simulated.connector({leader.port: HOME}, clock=clock)
        controller = connect(leader.port, [leader.calibration])
        self.assertIsInstance(controller, LerobotFeetechController)
        np.testing.assert_allclose(controller.read_action(), [*HOME, 0.0], atol=0.002)
        clock.t = 2.0  # a quarter of the default period: the base at its full swing, the jaw shut
        action = controller.read_action()
        self.assertAlmostEqual(float(action[0]), 0.6, delta=0.002)
        self.assertAlmostEqual(float(action[-1]), 1.0, delta=0.002)
        controller.close()
        self.assertEqual(controller.bus.calls[-1], ("disconnect",))


if __name__ == "__main__":
    unittest.main()
