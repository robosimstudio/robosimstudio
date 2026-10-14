# SPDX-License-Identifier: Apache-2.0
"""A planned line does not report an earlier line's arrival as its own,
describe_planner reports the planner's honours_roll, and cuRobo plans a move
too short to have a direction free rather than returning nothing."""

from __future__ import annotations

import unittest

from robosimstudio_policy_agent.ik import IKBackend
from robosimstudio_policy_agent.motion_tools import MotionTools
from robosimstudio_policy_agent.planners.curobo import MotionBackendCuRobo

TIER = "fast"
"""Backends built bare with stubbed members: no GPU, no cuRobo, no scene."""


def _pose(x: float) -> dict:
    return {"position": {"x": x, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class _Planner:
    honours_roll = True
    last_refusal = ""

    def plan_linear(self, start, end, **kwargs):
        return {"waypoints": [{"positions": [0.0]}]}


def _ik_with(planner) -> IKBackend:
    ik = object.__new__(IKBackend)
    ik.planner = planner
    ik.prefer_planned = True
    ik.honours_roll = False
    ik._own_refusal = ""
    ik.last_position_error = 0.0123  # an earlier solved line's arrival
    return ik


class TestPlannedLineArrival(unittest.TestCase):
    def test_a_planner_answer_does_not_carry_the_last_lines_error(self):
        result = MotionTools(_ik_with(_Planner())).plan_linear(_pose(0.1), start=_pose(0.0))
        self.assertTrue(result["planned"])
        self.assertIsNone(result["position_error_m"])


class TestDescribePlanner(unittest.TestCase):
    def test_honours_roll_is_the_planners(self):
        self.assertTrue(MotionTools(_ik_with(_Planner())).describe_planner()["honours_roll"])

    def test_without_a_planner_it_is_the_solvers(self):
        self.assertFalse(MotionTools(_ik_with(None)).describe_planner()["honours_roll"])


class TestCuroboShortMove(unittest.TestCase):
    def test_a_rotation_in_place_is_planned_free(self):
        planner = object.__new__(MotionBackendCuRobo)
        calls = []
        planner._import = lambda: (None, None, None, None, None, None)
        planner._orientation_held = lambda *a, **k: False

        def plan(position, quat, arm_id, seed, criteria, num_waypoints, **kwargs):
            calls.append(criteria)
            return {"waypoints": ["free"]}

        planner._plan = plan
        self.assertEqual(planner.plan_linear(_pose(0.0), _pose(0.0)), {"waypoints": ["free"]})
        self.assertEqual(calls, [None])


if __name__ == "__main__":
    unittest.main()
