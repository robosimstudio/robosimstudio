# SPDX-License-Identifier: Apache-2.0
"""``robot.*`` introspection reads the home posture, and device reads go through the bus."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_policy_agent import robot_tools
from robosimstudio_policy_agent.robot_tools import RobotTools

TIER = "fast"

HOME = (np.array([0.4, 0.0, 0.35]), np.array([1.0, 0.0, 0.0, 0.0]))


class _Arm:
    """An arm whose live TCP is low (just grasped) and whose home is high."""

    home_pose = HOME

    def ee_pose(self, state):
        raise AssertionError("the live pose is not the home pose")


class _Bus:
    def __init__(self) -> None:
        self.inside = False
        self.calls = 0

    def call(self, fn, *args, **kwargs):
        self.calls += 1
        self.inside = True
        try:
            return fn(*args, **kwargs)
        finally:
            self.inside = False


def _tools(bus=None) -> RobotTools:
    hand = SimpleNamespace(
        tip_reach=0.02,
        housing_rise=0.05,
        pad_thickness=0.01,
        tcp_offset=(0.0, 0.0, 0.1),
        tcp_offset_rot=(0.0, 0.0, 0.0, 1.0),
    )
    handle = SimpleNamespace(arm_coords=[0, 1], arm_dofs=[0, 1], home_q=[0.0, 0.0])
    model = SimpleNamespace(
        joint_limit_lower=SimpleNamespace(numpy=lambda: np.array([-1.0, -1.0])),
        joint_limit_upper=SimpleNamespace(numpy=lambda: np.array([1.0, 1.0])),
    )
    env = SimpleNamespace(
        handles=SimpleNamespace(arms={"arm": handle}, surface_z=0.0),
        scene=SimpleNamespace(embodiment=SimpleNamespace(hands={"arm": hand})),
        model=model,
        state=SimpleNamespace(joint_q=SimpleNamespace(numpy=lambda: np.array([0.1, 0.2]))),
        arm=lambda name="": _Arm(),
    )
    tools = RobotTools(env, bus or _Bus())
    tools._arm = lambda arm_id, role: "arm"
    return tools


class HomePoseTest(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("arm_id_of", "arm_role", "arm_order"):
            patcher = mock.patch.object(robot_tools, name, return_value=0 if name == "arm_id_of" else ())
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_describe_workspace_transport_is_the_home_height(self) -> None:
        out = _tools().describe_workspace()
        self.assertAlmostEqual(out["transport_z"], 0.35)
        self.assertEqual(out["transport_z_source"], "home_tcp")

    def test_describe_arm_reports_the_home_pose(self) -> None:
        out = _tools().describe_arm()
        self.assertAlmostEqual(out["home_pose"]["position"]["z"], 0.35)


class ForwardKinematicsOnBusTest(unittest.TestCase):
    def test_fk_runs_inside_bus_call(self) -> None:
        bus = _Bus()
        tools = _tools(bus)

        class _FkArm(_Arm):
            def fk_ee(self, q):
                assert bus.inside, "fk_ee ran off the bus"
                return np.array([float(q[0]), 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

        tools.env.arm = lambda name="": _FkArm()
        out = tools.forward_kinematics([[0.5, 0.0]])
        self.assertEqual(bus.calls, 1)
        self.assertAlmostEqual(out["poses"][0]["position"]["x"], 0.5)
        self.assertEqual(out["joint_config"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
