# SPDX-License-Identifier: Apache-2.0
"""Small contracts of the policy agent's tools, held without a simulator.

Each case is one promise a tool's docstring or protocol makes and an earlier
version broke: a relative grip preset that silently used half the span, a wait
whose timeout was per step, servo verbs that raised where their sibling routed,
joint moves whose protocol said ``bool`` and returned ``None``, a stale graph
thread writing into the next episode, and a provenance line naming the branch
twice.
"""

from __future__ import annotations

import subprocess
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import robosimstudio_policy_agent.robot_tools as robot_tools
import robosimstudio_policy_agent.skills as skills
from robosimstudio.core.bus import BusTimeout
from robosimstudio_policy_agent.connector import _routed_servo_move
from robosimstudio_policy_agent.facade import GapEnvFacade
from robosimstudio_policy_agent.graph_solution import SolutionGraph
from robosimstudio_policy_agent.robot_tools import RobotTools

TIER = "fast"


class _Gripper:
    def command_for_width(self, width):
        return float(np.clip(1.0 - width / 0.08, 0.0, 1.0))

    def width_for_command(self, command):
        return 0.08 * (1.0 - command)


class _RampingFacade:
    def __init__(self) -> None:
        self.commands: list[float] = []

    def set_grip(self, opening, *, ramp_steps, settle_steps, arm_id):
        del ramp_steps, settle_steps, arm_id
        self.commands.append(opening)


def _grip_tools() -> RobotTools:
    tools = RobotTools.__new__(RobotTools)
    tools.env = types.SimpleNamespace()
    tools.facade = _RampingFacade()
    tools._arm = lambda arm_id, role: "arm"
    tools._gripper = lambda name: _Gripper()
    tools.describe_gripper = lambda arm_id, role: {
        "span_m": 0.08,
        "min_grasp_width_m": 0.0,
        "finger": {"clearance_m": 0.002},
    }
    return tools


class TestRelativeGripPresets(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(robot_tools, "arm_id_of", lambda env, name: 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pinch_and_grip_take_the_object_width_beside_them(self):
        tools = _grip_tools()
        pinch = tools.set_grip(preset="pinch", object_width_m=0.03)
        self.assertAlmostEqual(pinch["requested_width_m"], 0.03 + 2 * 0.002)
        grip = tools.set_grip(preset="grip", object_width_m=0.03, squeeze_m=0.004)
        self.assertAlmostEqual(grip["requested_width_m"], 0.026)

    def test_a_relative_preset_without_an_object_width_refuses(self):
        for preset in ("pinch", "grip"):
            with self.assertRaisesRegex(ValueError, "object_width_m"):
                _grip_tools().set_grip(preset=preset)

    def test_an_absolute_preset_still_takes_nothing_else(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _grip_tools().set_grip(preset="open", object_width_m=0.03)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _grip_tools().set_grip(preset="pinch", object_width_m=0.03, width_m=0.05)


class _SlowBus:
    """Steps once per ``wait_for_step``, taking ``delay`` seconds each time."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.snapshot = types.SimpleNamespace(step=0)

    def wait_for_step(self, *, after, timeout):
        if timeout < self.delay:
            time.sleep(timeout)
            raise BusTimeout("quiet")
        time.sleep(self.delay)
        self.snapshot = types.SimpleNamespace(step=after + 1)


class TestWaitIsOneDeadline(unittest.TestCase):
    def test_the_timeout_bounds_the_whole_wait_not_each_step(self):
        tools = RobotTools.__new__(RobotTools)
        tools.bus = _SlowBus(0.02)
        start = time.monotonic()
        out = tools.wait_steps(steps=1000, timeout_s=0.2)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0)
        self.assertLess(out["waited"], 1000)
        self.assertGreater(out["waited"], 0)

    def test_a_wait_inside_its_budget_counts_every_step(self):
        tools = RobotTools.__new__(RobotTools)
        tools.bus = _SlowBus(0.0)
        self.assertEqual(tools.wait_steps(steps=5, timeout_s=5.0), {"waited": 5})


class TestServoVerbsRouteUnreachable(unittest.TestCase):
    def test_unreachable_is_an_outcome_not_a_raise(self):
        def move():
            raise RuntimeError("servo: pose unreachable after 3 consecutive solve failures")

        out = _routed_servo_move(move)
        self.assertEqual(out["outcome"], "unreachable")
        self.assertFalse(out["reached"])

    def test_the_servo_result_comes_back_with_reached(self):
        self.assertTrue(_routed_servo_move(lambda: {"outcome": "reached", "steps": 4})["reached"])
        contact = _routed_servo_move(lambda: {"outcome": "contact", "steps": 4})
        self.assertEqual(contact["outcome"], "contact")
        self.assertFalse(contact["reached"])

    def test_any_other_error_still_raises(self):
        def move():
            raise RuntimeError("servo: LM solve returned non-finite joints")

        with self.assertRaises(RuntimeError):
            _routed_servo_move(move)


class _InlineBus:
    def call(self, fn, *a, **kw):
        return fn(*a, **kw)


def _facade(errors: list[float]) -> GapEnvFacade:
    facade = GapEnvFacade.__new__(GapEnvFacade)
    facade.bus = _InlineBus()
    facade.apply_joint_action = lambda goal, arm_id=0: None
    facade._arm = lambda arm_id: "arm"
    readings = iter(errors)
    facade._joint_error = lambda name, goal: next(readings)
    return facade


class TestJointMovesSayWhetherTheyArrived(unittest.TestCase):
    def test_converged(self):
        self.assertIs(_facade([1.0, 0.001]).move_to_joints_blocking(np.zeros(2), max_steps=5), True)

    def test_out_of_budget(self):
        self.assertIs(_facade([1.0] * 5).move_to_joints_blocking(np.zeros(2), max_steps=5), False)

    def test_fire_and_forget_is_not_known_to_arrive(self):
        self.assertIs(_facade([]).move_to_joints_blocking(np.zeros(2), max_steps=0), False)

    def test_a_trajectory_reports_its_settle(self):
        facade = _facade([1.0, 0.001])
        self.assertIs(facade.stream_joint_trajectory([[0.0, 0.0], [0.1, 0.1]], settle_max_steps=5), True)
        self.assertIs(facade.stream_joint_trajectory([]), True)


class TestAStaleGraphThreadWritesNothing(unittest.TestCase):
    def test_writes_from_a_superseded_generation_are_dropped(self):
        policy = SolutionGraph.__new__(SolutionGraph)
        policy._generation = 2
        policy._result = policy._error = None
        policy.trace = {}
        policy._recording_node = ""
        policy._on_demand_observations = False
        policy.connector = object()
        policy.skills = []
        policy.perception = ""
        policy.episode_trace_dir = None
        policy.graph_path = "g"
        policy.env = types.SimpleNamespace(episode=types.SimpleNamespace(instruction=""))

        gate = threading.Event()

        def execute(graph, **kwargs):
            gate.wait(5.0)
            raise RuntimeError("late")

        fake_gap = types.ModuleType("gap")
        fake_gap.execute = execute
        with mock.patch.dict("sys.modules", {"gap": fake_gap}):
            stale = threading.Thread(target=policy._run_graph, args=(1,))
            stale.start()
            gate.set()
            stale.join(5.0)
        self.assertIsNone(policy._error)
        self.assertEqual(policy.trace, {})

        policy._on_graph_node_start("n", True, 1)
        self.assertEqual(policy._recording_node, "")
        policy._on_graph_node_start("n", True, 2)
        self.assertEqual(policy._recording_node, "n")


class TestSiblingProvenance(unittest.TestCase):
    def test_branch_and_sha_are_asked_separately(self):
        answers = {"--abbrev-ref": "main\n", "--short": "abc1234\n"}

        def run(cmd, **kwargs):
            del kwargs
            return subprocess.CompletedProcess(cmd, 0, stdout=answers[cmd[4]], stderr="")

        with (
            mock.patch.object(skills, "ORS_SIBLING", Path(__file__).parent),
            mock.patch("subprocess.run", run),
        ):
            line = skills._sibling_state()
        self.assertIn("main@abc1234", line)


if __name__ == "__main__":
    unittest.main()
