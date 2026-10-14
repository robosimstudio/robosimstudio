# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the batched graph policy's per-wave state and observations.

* A graph thread that outlived its wave writes nothing into the reused ``_World``.
* A graph name that is neither shipped nor a directory raises, as the serial path does.
* Observing every world reads each gripper bank's commands once, not once per world.
"""

from __future__ import annotations

import threading
import types
import unittest
from unittest import mock

import numpy as np

from robosimstudio_policy_agent.fleet import solution as fleet_solution
from robosimstudio_policy_agent.fleet.obs_format import world_observation, world_observations

TIER = "fast"


def _policy(state):
    policy = fleet_solution.SolutionGraphBatched.__new__(fleet_solution.SolutionGraphBatched)
    policy.worlds = [state]
    policy.parks = types.SimpleNamespace(adopt=lambda world: None)
    policy.perception = ""
    policy.graph_path = "g"
    policy.skills = []
    return policy


def _run(policy, *, current: bool, outcome):
    state = policy.worlds[0]

    def execute(graph, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    fake_gap = types.ModuleType("gap")
    fake_gap.execute = execute
    with mock.patch.dict("sys.modules", {"gap": fake_gap}):
        thread = threading.Thread(target=policy._run_graph, args=(0, ""))
        # The next wave has started its own thread, or this one is still the world's.
        state.thread = thread if current else threading.Thread(target=lambda: None)
        thread.start()
        thread.join(5.0)


class TestAStaleWorldThreadWritesNothing(unittest.TestCase):
    def test_a_stale_error_is_dropped(self):
        state = fleet_solution._World()
        _run(_policy(state), current=False, outcome=RuntimeError("late"))
        self.assertIsNone(state.error)
        self.assertEqual(state.trace, {})

    def test_a_stale_result_is_dropped(self):
        state = fleet_solution._World()
        _run(_policy(state), current=False, outcome={"ok": True})
        self.assertIsNone(state.result)

    def test_the_current_thread_still_records(self):
        state = fleet_solution._World()
        _run(_policy(state), current=True, outcome=RuntimeError("now"))
        self.assertIsInstance(state.error, RuntimeError)
        self.assertIn("traceback", state.trace)
        state = fleet_solution._World()
        _run(_policy(state), current=True, outcome={"ok": True})
        self.assertEqual(state.result, {"ok": True})


class TestAnUnknownGraphNameRaises(unittest.TestCase):
    def test_typo_is_refused(self):
        def graph_path(name):
            raise KeyError(name)

        task = types.SimpleNamespace(graph_path=graph_path, graphs=[("gt", "graphs/weave")], full_name="suite/task")
        fleet = types.SimpleNamespace(task=task)
        with self.assertRaises(ValueError) as caught:
            fleet_solution.SolutionGraphBatched._resolve_graph(fleet, "/no/such/graph-weave", registries=[])
        self.assertIn("'gt'", str(caught.exception))


class _Bank:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.calls = 0

    def commands(self):
        self.calls += 1
        return self.values.copy()


class TestObservingEveryWorldReadsTheBankOnce(unittest.TestCase):
    def test_one_read_per_bank(self):
        num_worlds = 6
        bank = _Bank(np.linspace(0.0, 1.0, num_worlds))
        embodiment = types.SimpleNamespace(
            arms={"right": types.SimpleNamespace(arm_coords=list(range(7)), ee_body=0)},
            grippers={},
            hands={"right": types.SimpleNamespace(tcp_offset=(0.0, 0.0, 0.0))},
        )
        fleet = types.SimpleNamespace(
            scene=types.SimpleNamespace(embodiment=embodiment),
            grippers={"right": bank},
            episodes=[None] * num_worlds,
            num_worlds=num_worlds,
        )
        body = np.zeros((1, 7))
        body[0, 6] = 1.0
        snapshot = types.SimpleNamespace(coords=lambda world: np.zeros(8), body=lambda world: body)
        observed = world_observations(fleet, snapshot)
        self.assertEqual(bank.calls, 1)
        self.assertEqual(len(observed), num_worlds)
        for world, obs in enumerate(observed):
            self.assertAlmostEqual(obs["proprio"]["right/gripper_command"], bank.values[world])
            alone = world_observation(fleet, world, snapshot)
            self.assertAlmostEqual(alone["proprio"]["right/gripper_command"], bank.values[world])


if __name__ == "__main__":
    unittest.main()
