# SPDX-License-Identifier: Apache-2.0
"""A reward masked to the episode and labelled apart from the goal, and the
gauntlet reading a document the way the loader and the env do."""

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import evaluate
from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.goals.reward import Gated, Progress
from robosimstudio_bench.taskdoc.ast import Condition, Instance, Ref, TaskDoc
from robosimstudio_bench.taskdoc.parse import parse_problem
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"

# The package re-exports `run` as `gauntlet`, which shadows the module.
gauntlet = importlib.import_module("robosimstudio_bench.goals.gauntlet")

TWO = """
(define (problem two)
  (:objects a b - block)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (and (>= (containment a square) 1.0) (>= (containment b square) 1.0)))
  (:reward (progress (containment REWARDED square) (:gate 1.0) (:gated-by (<= (speed GATED) 0.02)))))
"""


def _fake_bind(condition, doc, catalog, task=""):
    """A leaf that remembers which object it was built about."""
    del doc, catalog, task
    name = condition.args[0].base
    ok = condition.head != "speed"
    return Leaf(fn=lambda ctx: (ok, 0.5, {"fraction": 0.5}), about=(condition.head, (name,)))


def _evaluator(rewarded, gated, active):
    doc = parse_problem(parse_sexp(TWO.replace("REWARDED", rewarded).replace("GATED", gated)))
    with mock.patch.object(evaluate, "bind", _fake_bind):
        return evaluate.make_evaluator(doc, active=active)


class TestRewardIsMaskedToTheEpisode(unittest.TestCase):
    def test_a_measure_about_a_parked_object_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside this episode's active set"):
            _evaluator("b", "b", active={"a"})

    def test_a_gate_about_a_parked_object_is_dropped(self):
        scored = _evaluator("a", "b", active={"a"})
        self.assertIsInstance(scored.reward, Progress)
        self.assertEqual(scored.reward.of.about, ("containment", ("a",)))

    def test_an_active_gate_is_kept(self):
        scored = _evaluator("a", "a", active={"a"})
        self.assertIsInstance(scored.reward, Gated)
        self.assertEqual(scored.reward.by.about, ("speed", ("a",)))

    def test_unmasked_without_an_active_set(self):
        scored = _evaluator("b", "b", active=None)
        self.assertIsInstance(scored.reward, Gated)
        self.assertEqual(scored.reward.of.of.about, ("containment", ("b",)))


class TestRewardNodesHaveTheirOwnLabels(unittest.TestCase):
    def test_reward_verdicts_do_not_collide_with_the_goal(self):
        scored = _evaluator("a", "a", active=None)
        self.assertEqual(scored.reward.of.of.label, "reward")
        self.assertEqual(scored.reward.by.label, "reward.gated_by")
        metrics = scored(None).metrics
        self.assertIs(metrics["reward"], True)
        self.assertIs(metrics["reward.gated_by"], False)
        self.assertIs(metrics["goal"], True)


class TestGauntletStartReadsTheActiveSet(unittest.TestCase):
    def test_a_parked_objects_init_clause_is_not_checked(self):
        doc = TaskDoc(
            name="t",
            objects=(Instance("a", "block"), Instance("b", "block")),
            init=(Condition("on-table", (Ref("a"),)), Condition("on-table", (Ref("b"),))),
            goal=Condition("lifted", (Ref("a"),)),
        )

        def fake_bind(c, doc, asset_of, task=""):
            ok = c.head == "on-table" and c.args[0].base == "a"
            return Leaf(fn=lambda ctx: (ok, float(ok), {}))

        env = SimpleNamespace(
            reset=lambda: None,
            make_context=object,
            episode=SimpleNamespace(active=frozenset({"a"})),
        )
        with (
            mock.patch.object(gauntlet, "bind", fake_bind),
            mock.patch.object(gauntlet, "Snapshot", lambda ctx: SimpleNamespace(ctx=ctx)),
        ):
            stage = gauntlet._start(doc, env, "t")
        self.assertEqual(stage.status, "pass", stage.findings)


class TestGauntletReadsTheSuiteAndProfile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "demo.suite").write_text(
            "(define (suite demo)\n  (:staging (:profile fake@v1))\n  (:fixtures bin - box))\n"
        )
        self.task = root / "one.task"
        self.task.write_text(
            "(define (problem one)\n  (:suite demo)\n  (:objects tool - hammer)\n  (:goal (In tool bin)))\n"
        )

    def tearDown(self):
        self.dir.cleanup()

    def test_read_merges_the_sibling_suite(self):
        stage, doc = gauntlet._read(self.task)
        self.assertEqual(stage.status, "pass", stage.findings)
        self.assertIn("bin", {f.name for f in doc.objects})
        self.assertEqual(str(dict(doc.staging).get("profile")), "fake@v1")

    def test_bind_falls_back_to_the_documents_profile(self):
        _, doc = gauntlet._read(self.task)
        with mock.patch(
            "robosimstudio_bench.taskbuild.staging.get_profile",
            side_effect=lambda name: SimpleNamespace(bindings=None) if name == "fake@v1" else None,
        ) as get:
            stage, _ = gauntlet._bind(doc, "")
        get.assert_called_once_with("fake@v1")
        self.assertEqual(stage.data.get("profile"), "fake@v1")


class TestVacuousThreshold(unittest.TestCase):
    @staticmethod
    def _flags(op, value):
        return [f.kind for f in gauntlet._vacuous([Condition("distance", (Ref("a"),), op=op, value=value)])]

    def test_strictly_positive_is_not_vacuous(self):
        self.assertEqual(self._flags(">", 0.0), [])

    def test_the_vacuous_ones_are_still_flagged(self):
        self.assertEqual(self._flags(">=", 0.0), ["vacuous-threshold"])
        self.assertEqual(self._flags(">", -0.1), ["vacuous-threshold"])


if __name__ == "__main__":
    unittest.main()
