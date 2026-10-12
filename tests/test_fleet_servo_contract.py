# SPDX-License-Identifier: Apache-2.0
"""The fleet's servo is the single-world servo, keyword for keyword.

The connector's pose verbs are written against
:class:`~robosimstudio_policy_agent.servo.CartesianServo` and are not told
which servo they hold: ``move_cartesian_until_contact`` passes
``stop_on_stall``, ``go_to_pose`` reads ``outcome`` off the return. A fleet
servo with a narrower signature made the first a ``TypeError`` in every world of
every batched run, and one returning ``None`` made the second report
``reached: False`` on every move that arrived. These tests hold the two servos
to each other without a device, so the drift is caught where it is typed.
"""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio_policy_agent.fleet.intents import IntentHandle
from robosimstudio_policy_agent.fleet.solution import _FleetServo
from robosimstudio_policy_agent.fleet.world_facade import WorldFacade
from robosimstudio_policy_agent.servo import POS_TOLERANCE, CartesianServo

TIER = "fast"

POSE = {"position": {"x": 0.5, "y": -0.3, "z": 0.8}, "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}}


class _Facade:
    """Records what the servo asked for and answers like the fleet does."""

    def __init__(self, result=None) -> None:
        self.calls: list[dict] = []
        self.result = result

    def servo_to_pose(self, position, quat_xyzw, **kwargs):
        self.calls.append({"position": np.asarray(position), "quat": np.asarray(quat_xyzw), **kwargs})
        return self.result


class TestTheSignaturesAgree(unittest.TestCase):
    def test_every_serial_keyword_is_accepted_by_the_fleet(self):
        serial = inspect.signature(CartesianServo.move).parameters
        fleet = inspect.signature(_FleetServo.move).parameters
        missing = [name for name in serial if name not in fleet]
        self.assertEqual(missing, [], f"the fleet servo drops keywords the connector passes: {missing}")

    def test_defaults_agree(self):
        serial = inspect.signature(CartesianServo.move).parameters
        fleet = inspect.signature(_FleetServo.move).parameters
        for name, parameter in serial.items():
            if name in ("self", "pose"):
                continue
            with self.subTest(keyword=name):
                self.assertEqual(fleet[name].default, parameter.default)


class TestTheReturnIsTheSerialOutcome(unittest.TestCase):
    def test_the_outcome_dict_comes_back(self):
        facade = _Facade({"outcome": "reached", "remaining_distance": 0.001, "steps": 17})
        result = _FleetServo(facade).move(POSE, arm_id=0)
        self.assertEqual(result["outcome"], "reached")
        self.assertEqual(result["steps"], 17)

    def test_the_stall_arguments_reach_the_motion(self):
        facade = _Facade({"outcome": "contact", "remaining_distance": 0.02, "steps": 30})
        result = _FleetServo(facade).move(
            POSE, arm_id=1, max_steps=120, stop_on_stall=True, stall_steps=7, stall_epsilon=0.001
        )
        self.assertEqual(result["outcome"], "contact")
        call = facade.calls[0]
        self.assertTrue(call["stop_on_stall"])
        self.assertEqual(call["stall_steps"], 7)
        self.assertAlmostEqual(call["stall_epsilon"], 0.001)
        self.assertEqual(call["arm_id"], 1)
        self.assertEqual(call["max_steps"], 120)

    def test_the_serial_default_ball_is_the_fleet_default_ball(self):
        facade = _Facade({"outcome": "reached", "remaining_distance": 0.0, "steps": 1})
        _FleetServo(facade).move(POSE)
        self.assertAlmostEqual(facade.calls[0]["pos_tolerance"], POS_TOLERANCE)
        _FleetServo(facade).move(POSE, pos_tolerance=0.0015)
        self.assertAlmostEqual(facade.calls[1]["pos_tolerance"], 0.0015)

    def test_roll_honouring_travels_with_the_move(self):
        facade = _Facade({"outcome": "reached", "remaining_distance": 0.0, "steps": 1})
        _FleetServo(facade, honours_roll=True).move(POSE)
        self.assertTrue(facade.calls[0]["honours_roll"])
        _FleetServo(facade, honours_roll=False).move(POSE)
        self.assertFalse(facade.calls[1]["honours_roll"])

    def test_a_motion_that_never_ended_is_a_timeout_not_none(self):
        result = _FleetServo(_Facade(None)).move(POSE)
        self.assertEqual(result["outcome"], "timeout")

    def test_a_bad_speed_scale_is_refused_as_the_serial_servo_refuses_it(self):
        with self.assertRaises(ValueError):
            _FleetServo(_Facade({})).move(POSE, speed_scale=0.0)


class TestAHandleCarriesItsResult(unittest.TestCase):
    def test_wait_returns_what_the_fleet_completed_with(self):
        handle = IntentHandle()
        handle.complete(result={"outcome": "timeout"}, steps=4)
        self.assertEqual(handle.wait(timeout=0.1), {"outcome": "timeout"})

    def test_a_joint_move_completes_with_nothing(self):
        handle = IntentHandle()
        handle.complete(steps=2)
        self.assertIsNone(handle.wait(timeout=0.1))


class _Snapshot:
    def __init__(self) -> None:
        self.step = 3
        self.obs = {
            "proprio": {
                "right/joint_q": np.zeros(6),
                "right/ee_pose": np.array([0.5, 0.0, 0.8, 0, 0, 0, 1.0]),
                "right/gripper": 0.25,
            }
        }
        self.body_q = np.zeros((4, 7))
        self.frames = {"overhead": {"rgb": np.zeros((2, 2, 3))}}
        self.info = {}


class _Bus:
    def __init__(self) -> None:
        self.snapshot = _Snapshot()
        self.rendered = 0
        self.parks = None

    def batchable(self, kind: str) -> bool:
        return True

    def batched(self, kind, payload, timeout=None):
        self.rendered += 1
        return self.snapshot.frames


class TestProprioceptionDoesNotRender(unittest.TestCase):
    """A hand's pose is read off the snapshot; only a look photographs the fleet."""

    def _facade(self):
        bus = _Bus()
        fleet = SimpleNamespace(cameras=(), max_steps=100, num_worlds=1)
        facade = WorldFacade(0, fleet=fleet, bus=bus, intents=None, arm_names=["right"])
        return facade, bus

    def test_get_proprioception_never_asks_the_rig(self):
        facade, bus = self._facade()
        obs = facade.get_proprioception()
        self.assertEqual(bus.rendered, 0)
        np.testing.assert_allclose(obs["robot_cartesian_pos_0"][:3], [0.5, 0.0, 0.8])
        self.assertAlmostEqual(obs["robot_joint_pos_0"][-1], 0.75)  # the runtime's open fraction
        self.assertNotIn("overhead", obs)

    def test_get_observation_still_looks(self):
        facade, bus = self._facade()
        obs = facade.get_observation()
        self.assertEqual(bus.rendered, 1)
        self.assertIn("overhead", obs)

    def test_the_connector_prefers_the_narrow_reader(self):
        from robosimstudio_policy_agent.connector import _ProprioceptiveFeedback

        facade, bus = self._facade()
        feedback = _ProprioceptiveFeedback()
        feedback.facade = facade
        feedback.env = facade
        feedback._robot_feedback()
        self.assertEqual(bus.rendered, 0, "motion feedback must not photograph the fleet")


if __name__ == "__main__":
    unittest.main()
