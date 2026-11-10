# SPDX-License-Identifier: Apache-2.0
"""A world nothing can close and nothing reads is scored on the step it closes, and only then.

Scoring was 19 % of a cable_routing/L2 wave's wall time, 1.7 ms per world per
step, for a verdict that can only matter on a world's last step when the
document does not end on success, the goal has no history and the graph cannot
see the evaluator. These pin the three conditions and that a world scored late
closes the way it would have: same step, same kind, same verdict.
"""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio_bench.goals.combinators import AllOf, AnyOf, HoldFor, Leaf, Not, Once, WorstOf, holds_history
from robosimstudio_bench.goals.context import EvalResult
from robosimstudio_bench.goals.reward import Fraction, Progress, Scored
from robosimstudio_bench.run import pool
from robosimstudio_bench.run.fleet import RunnerFleet, WorldSlot

TIER = "fast"
"""No GPU. Gates every commit."""


def _leaf(value: bool = True, *, calls: list | None = None) -> Leaf:
    def measure(_ctx):
        if calls is not None:
            calls.append(1)
        return EvalResult(success=value, reward=float(value), metrics={"m": 1.0})

    return Leaf(fn=measure)


class TestHoldsHistory(unittest.TestCase):
    """An allowlist, so a node added later is treated as stateful until someone says otherwise."""

    def test_the_pure_shapes_over_pure_leaves_hold_none(self):
        tree = WorstOf(of=(AllOf(of=(_leaf(), _leaf())), AnyOf(of=(Not(of=_leaf()), _leaf()))))
        self.assertFalse(holds_history(tree))

    def test_a_temporal_node_anywhere_holds_history(self):
        self.assertTrue(holds_history(Once(of=_leaf())))
        self.assertTrue(holds_history(HoldFor(of=_leaf(), seconds=1.0)))
        self.assertTrue(holds_history(AllOf(of=(_leaf(), Once(of=_leaf())))))

    def test_a_smoothed_leaf_holds_history(self):
        """A measure's window is declared by the hook that empties it."""
        leaf = _leaf()
        leaf.on_reset = lambda: None
        self.assertTrue(holds_history(leaf))
        self.assertTrue(holds_history(AllOf(of=(_leaf(), leaf))))

    def test_an_unknown_node_is_assumed_to(self):
        class Mine(AllOf):
            pass

        self.assertTrue(holds_history(Mine(of=(_leaf(),))))


class TestScored(unittest.TestCase):
    def test_memoryless_asks_of_the_goal_and_the_reward_not_of_observe(self):
        self.assertTrue(Scored(goal=_leaf(), observe=(Once(of=_leaf()),)).memoryless)
        self.assertFalse(Scored(goal=Once(of=_leaf())).memoryless)
        self.assertFalse(Scored(goal=_leaf(), reward=Progress(of=Once(of=_leaf()), key="m", gate=1.0)).memoryless)
        self.assertTrue(Scored(goal=_leaf(), reward=Progress(of=_leaf(), key="m", gate=1.0)).memoryless)
        self.assertTrue(Scored(goal=_leaf(), reward=Fraction()).memoryless)

    def test_advance_keeps_only_observe_history(self):
        goal_calls, pure_calls, once_calls = [], [], []
        scored = Scored(
            goal=_leaf(calls=goal_calls),
            observe=(_leaf(calls=pure_calls), Once(of=_leaf(calls=once_calls))),
        )
        self.assertTrue(scored.advances)
        scored.advance(SimpleNamespace())
        self.assertEqual((len(goal_calls), len(pure_calls), len(once_calls)), (0, 0, 1))

    def test_nothing_to_advance_says_so(self):
        self.assertFalse(Scored(goal=_leaf(), observe=(_leaf(),)).advances)

    def test_an_observed_latch_seen_only_on_a_skipped_step_reaches_the_row(self):
        """``(once ...)`` in ``:observe`` is what makes :meth:`advance` necessary."""
        readings = iter([True, False])

        def touch(_ctx):
            return EvalResult(success=next(readings), reward=0.0, metrics={})

        scored = Scored(goal=_leaf(), observe=(Once(of=Leaf(fn=touch), label="observe.0"),))
        scored.advance(SimpleNamespace())
        metrics = scored(SimpleNamespace()).metrics
        self.assertTrue(metrics["observe.0"])
        self.assertTrue(metrics["latched"])
        self.assertFalse(metrics["holding_now"])


class TestCanWait(unittest.TestCase):
    """Every condition is necessary: any one of them false scores every step, as before."""

    def _runner(self, *, recorder=None, on_tick=None) -> RunnerFleet:
        runner = RunnerFleet.__new__(RunnerFleet)
        runner.recorder = recorder
        runner.on_wave_tick = on_tick
        return runner

    def _slot(self, *, terminate=False, evaluator=None) -> WorldSlot:
        return WorldSlot(
            world=0,
            episode=SimpleNamespace(episode_id=1),
            task=SimpleNamespace(terminate_on_success=terminate),
            evaluator=evaluator if evaluator is not None else Scored(goal=_leaf()),
        )

    POLICY = SimpleNamespace(reads_verdicts=False)
    FLEET = SimpleNamespace(task=None)

    def test_all_conditions_met(self):
        self.assertTrue(self._runner()._verdict_can_wait(self._slot(), self.FLEET, self.POLICY))

    def test_a_document_that_ends_on_success_scores_every_step(self):
        self.assertFalse(self._runner()._verdict_can_wait(self._slot(terminate=True), self.FLEET, self.POLICY))

    def test_a_policy_that_can_read_the_verdict_scores_every_step(self):
        policy = SimpleNamespace(reads_verdicts=True)
        self.assertFalse(self._runner()._verdict_can_wait(self._slot(), self.FLEET, policy))

    def test_a_policy_that_does_not_say_is_assumed_to_read(self):
        self.assertFalse(self._runner()._verdict_can_wait(self._slot(), self.FLEET, SimpleNamespace()))

    def test_a_stateful_goal_scores_every_step(self):
        slot = self._slot(evaluator=Scored(goal=Once(of=_leaf())))
        self.assertFalse(self._runner()._verdict_can_wait(slot, self.FLEET, self.POLICY))

    def test_a_hand_written_evaluator_scores_every_step(self):
        slot = self._slot(evaluator=lambda ctx: EvalResult(success=False, reward=0.0))
        self.assertFalse(self._runner()._verdict_can_wait(slot, self.FLEET, self.POLICY))

    def test_a_recording_or_a_live_view_scores_every_step(self):
        """Both read ``slot.info`` as the episode runs."""
        self.assertFalse(self._runner(recorder=object())._verdict_can_wait(self._slot(), self.FLEET, self.POLICY))
        self.assertFalse(self._runner(on_tick=print)._verdict_can_wait(self._slot(), self.FLEET, self.POLICY))

    def test_a_tick_that_reads_nothing_does_not_hold_the_verdict(self):
        """A pool worker's watchdog stamp ticks on every step of every cell."""
        runner = self._runner(on_tick=print)
        runner.tick_reads_verdicts = False
        self.assertTrue(runner._verdict_can_wait(self._slot(), self.FLEET, self.POLICY))

    def test_the_pool_worker_says_its_tick_reads_nothing(self):
        """Read off the source: building a worker's runner opens a fleet."""
        source = inspect.getsource(pool._FleetWorker.run)
        stamp = source.index("self.runner.on_wave_tick = self._tick")
        self.assertIn("self.runner.tick_reads_verdicts = False", source[stamp:])
        tick = inspect.getsource(pool._FleetWorker._tick)
        self.assertNotIn("_slots.", tick)
        self.assertNotIn("slots[", tick)

    def test_a_parked_world_has_nothing_to_wait_for(self):
        self.assertFalse(self._runner()._verdict_can_wait(WorldSlot(world=0), self.FLEET, self.POLICY))


class _Fleet:
    """Just enough of a FleetEnv for one wave: a step clock and an evaluate that counts."""

    eval_every = 1
    num_worlds = 2
    task = SimpleNamespace(full_name="t/v", terminate_on_success=False)
    model = SimpleNamespace(device="cuda:0")

    def __init__(self) -> None:
        self.step_count = 0
        self.evaluated: list[tuple[int, int]] = []
        self.contexts: list[tuple[int, int]] = []
        self.diverge_at: dict[int, int] = {}

    def reset_wave(self, *_args) -> None:
        self.step_count = 0

    def snapshot(self):
        diverged = np.array([self.diverge_at.get(w, 10**9) <= self.step_count for w in range(self.num_worlds)])
        return SimpleNamespace(step=self.step_count, diverged=diverged)

    def settled_body_q(self, _world):
        return np.zeros((1, 7))

    def context(self, world, snapshot, *, settled_body_q):
        self.contexts.append((world, snapshot.step))
        return SimpleNamespace(world=world, step=snapshot.step)

    def evaluate(self, world, evaluator, snapshot, *, settled_body_q):
        self.evaluated.append((world, snapshot.step))
        return evaluator(self.context(world, snapshot, settled_body_q=settled_body_q))

    def flush(self) -> None:
        pass

    def step(self) -> None:
        self.step_count += 1


class _Policy:
    name = "stub"

    def __init__(self, *, reads: bool = False, requests: dict | None = None) -> None:
        self.reads_verdicts = reads
        self.requests = requests or {}
        self.step = 0

    def reset_wave(self, slots) -> None:
        pass

    def act(self, live) -> None:
        pass

    def on_step(self, snapshot) -> None:
        self.step += 1

    def termination_requests(self) -> dict:
        return {world: why for world, (at, why) in self.requests.items() if self.step >= at}

    def row_extras(self, _world) -> dict:
        return {}


class TestTheWave(unittest.TestCase):
    """The runner's own loop, on stubs: when the evaluator runs and how a world closes."""

    def _run(self, *, policy, fleet=None, limit=20, observe=()):
        fleet = fleet or _Fleet()
        runner = RunnerFleet.__new__(RunnerFleet)
        runner.job = SimpleNamespace(max_steps=limit, solution_kwargs=(), renderer="")
        runner.recorder = None
        runner.on_wave_tick = None
        runner.num_worlds = fleet.num_worlds
        runner.rows = []
        runner.verbose = False
        runner.summary = SimpleNamespace(waves=[], rows=[])
        runner.store = SimpleNamespace(append_result=lambda row: None)
        runner._evaluator = lambda task, episode: Scored(goal=_leaf(False), observe=tuple(observe))
        runner._check_graph = lambda slots: None
        members = [(fleet.task, SimpleNamespace(episode_id=w, seed=w, instruction="")) for w in range(fleet.num_worlds)]
        runner._run_wave(fleet, policy, 0, members)
        return fleet, {row["episode_id"]: row for row in runner.rows}

    def test_a_request_is_scored_on_that_step_and_no_other(self):
        policy = _Policy(requests={0: (5, "graph finished"), 1: (9, "graph finished")})
        fleet, rows = self._run(policy=policy)
        self.assertEqual(sorted(fleet.evaluated), [(0, 5), (1, 9)])
        self.assertEqual((rows[0]["episode_length"], rows[1]["episode_length"]), (5, 9))
        self.assertEqual(rows[0]["close_kind"], "policy_request")
        self.assertEqual(rows[0]["termination_reason"], "graph finished")

    def test_a_world_that_runs_out_of_budget_is_scored_on_its_last_step(self):
        fleet, rows = self._run(policy=_Policy(), limit=7)
        self.assertEqual(sorted(fleet.evaluated), [(0, 7), (1, 7)])
        self.assertEqual(rows[1]["close_kind"], "truncated")
        self.assertEqual(rows[1]["episode_length"], 7)

    def test_a_diverged_world_closes_on_the_step_it_diverged(self):
        fleet = _Fleet()
        fleet.diverge_at = {1: 4}
        fleet, rows = self._run(policy=_Policy(requests={0: (6, "graph finished")}), fleet=fleet)
        self.assertEqual(rows[1]["close_kind"], "diverged")
        self.assertEqual(rows[1]["episode_length"], 4)
        self.assertEqual(rows[1]["metrics"], {"diverged": True})
        self.assertEqual(fleet.evaluated, [(0, 6)])

    def test_a_policy_that_reads_is_scored_on_every_step(self):
        fleet, rows = self._run(policy=_Policy(reads=True, requests={0: (3, "x"), 1: (3, "x")}))
        self.assertEqual(sorted(fleet.evaluated), [(w, s) for w in (0, 1) for s in (1, 2, 3)])
        self.assertEqual(rows[0]["episode_length"], 3)

    def test_the_row_is_the_one_every_step_scoring_writes(self):
        """Scored late and scored always close at one step, with one verdict and the same metrics."""
        make = lambda reads: _Policy(reads=reads, requests={0: (5, "x"), 1: (8, "y")})  # noqa: E731
        _, late = self._run(policy=make(False), observe=(Once(of=_leaf(), label="observe.0"),))
        _, always = self._run(policy=make(True), observe=(Once(of=_leaf(), label="observe.0"),))
        for episode in (0, 1):
            for key in ("success", "reward", "episode_length", "close_kind", "termination_reason", "metrics"):
                self.assertEqual(late[episode][key], always[episode][key], key)

    def test_an_observed_latch_is_advanced_on_the_steps_between(self):
        fleet, _ = self._run(
            policy=_Policy(requests={0: (3, "x"), 1: (3, "x")}), observe=(Once(of=_leaf(), label="observe.0"),)
        )
        # The context each step's `advance` was built from, then the closing pass's own.
        # Step 0 is the evaluator's reset, which reads one too.
        stepped = sorted(context for context in fleet.contexts if context[1] > 0)
        self.assertEqual(stepped, [(w, s) for w in (0, 1) for s in (1, 2, 3)])


if __name__ == "__main__":
    unittest.main()
