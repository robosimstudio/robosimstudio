# SPDX-License-Identifier: Apache-2.0
"""A native pairing admitted however its hand is spelled; a reward bound with the
rest of the document; a comparison around a verdict refused; a suite's name kept
as written; and a variant's staging merged into its suite's clause by clause."""

import unittest
from unittest import mock

from robosimstudio.robots.registry import get_embodiment, hands_of
from robosimstudio.specs import Binding, TaskRequirements, TaskSpec
from robosimstudio_bench.goals.combinators import Leaf
from robosimstudio_bench.goals.gauntlet import _reward_measure
from robosimstudio_bench.goals.goal import build_goal
from robosimstudio_bench.goals.reward import Gated, build_reward, reading_key
from robosimstudio_bench.taskbuild.matrix import admits, declared_key
from robosimstudio_bench.taskdoc import VAB_INDUSTRIAL_V1, apply_bindings, merge_suite, parse_problem, parse_suite
from robosimstudio_bench.taskdoc.ast import Condition
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"


def _suite(text):
    return parse_suite(parse_sexp(text))


def _problem(text):
    return parse_problem(parse_sexp(text))


class TestAdmitsTheRobotsOwnHand(unittest.TestCase):
    def test_naming_the_own_hand_is_the_native_pairing(self):
        task = TaskSpec(name="t", scene="bench_0728", embodiment="panda", requires=TaskRequirements(monolithic=True))
        (own,) = hands_of(get_embodiment("panda"))
        self.assertEqual(admits(task, "bench_0728", "panda", ""), (True, ""))
        self.assertEqual(admits(task, "bench_0728", "panda", own), (True, ""))

    def test_another_pairing_is_still_refused(self):
        task = TaskSpec(name="t", scene="bench_0728", embodiment="panda", requires=TaskRequirements(monolithic=True))
        ok, why = admits(task, "bench_0728", "panda_robotiq", "")
        self.assertFalse(ok)
        self.assertIn("one import", why)


class TestTheOwnHandFindsTheDeclaredBinding(unittest.TestCase):
    def _task(self):
        native = Binding(scene="bench_0728", embodiment="panda", graphs=("pick",))
        return TaskSpec(
            name="t",
            scene="bench_0728",
            embodiment="panda",
            bindings=(native,),
            requires=TaskRequirements(monolithic=True),
        )

    def test_the_own_hand_is_the_declared_raw_key(self):
        task = self._task()
        (own,) = hands_of(get_embodiment("panda"))
        self.assertEqual(declared_key(task, "bench_0728", "panda", own), ("bench_0728", "panda", ""))
        self.assertIs(task.binding_for(*declared_key(task, "bench_0728", "panda", own)), task.bindings[0])

    def test_empty_arguments_are_the_tasks_own(self):
        self.assertEqual(declared_key(self._task()), ("bench_0728", "panda", ""))

    def test_an_undeclared_pairing_is_none(self):
        self.assertIsNone(declared_key(self._task(), "bench_0728", "panda_robotiq", ""))
        self.assertIsNone(declared_key(self._task(), "bench_0728", "no_such_robot", ""))

    def test_the_fleet_base_takes_the_own_hand(self):
        from robosimstudio.specs import replace
        from robosimstudio_bench.run import fleet as run_fleet

        base = self._task()
        (own,) = hands_of(get_embodiment("panda"))
        instance = replace(base, name="instance", gripper=own, fleet_base="base")
        with mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=base):
            built = run_fleet.fleet_build_task(instance)
        self.assertEqual((built.scene, built.embodiment), ("bench_0728", "panda"))
        self.assertIs(built.binding_for(), built.bindings[0])


class TestApplyBindingsBindsTheReward(unittest.TestCase):
    def test_measure_and_mask_are_bound(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin) (:goal (In hammer bin))"
            " (:reward (progress (In hammer bin) (:gate 1.0) (:gated-by (In hammer bin)))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)
        measure, gate, mask = bound.reward["progress"]
        self.assertEqual(measure, [">=", ["containment", "hammer", "bin"], 1.0])
        self.assertEqual(gate, [":gate", 1.0])
        self.assertEqual(mask, [":gated-by", [">=", ["containment", "hammer", "bin"], 1.0]])
        parsed = parse_condition(mask[1])
        self.assertEqual((parsed.head, parsed.op, parsed.value), ("containment", ">=", 1.0))
        self.assertEqual(parsed, bound.goal)
        # The original is untouched.
        self.assertEqual(doc.reward["progress"][0], ["In", "hammer", "bin"])

    def test_the_bound_reward_reads_the_atom_not_the_comparison(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin) (:goal (In hammer bin))"
            " (:reward (progress (In hammer bin) (:gate 1.0))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)

        def make_node(form):
            return Leaf(fn=lambda ctx: (True, 1.0, {}))

        clause = build_reward(bound.reward, make_node)
        self.assertEqual(clause.key, reading_key("containment"))
        self.assertNotEqual(clause.key, ">=")
        self.assertEqual(_reward_measure(bound.reward), ("containment", ("hammer", "bin")))

    def test_the_gated_bound_reward_reads_the_atom(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin) (:goal (In hammer bin))"
            " (:reward (progress (In hammer bin) (:gate 1.0) (:gated-by (In hammer bin)))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)
        clause = build_reward(bound.reward, lambda form: Leaf(fn=lambda ctx: (True, 1.0, {})))
        self.assertIsInstance(clause, Gated)
        self.assertEqual(clause.of.key, reading_key("containment"))

    def test_bound_inside_connectives_and_modals(self):
        doc = _problem(
            "(define (problem p) (:objects hammer bin)"
            " (:reward (progress (and (hold-for 2 (In hammer bin)) (lifted hammer)) (:gate 1.0))))"
        )
        bound = apply_bindings(doc, VAB_INDUSTRIAL_V1)
        self.assertEqual(
            bound.reward["progress"][0],
            ["and", ["hold-for", 2, [">=", ["containment", "hammer", "bin"], 1.0]], ["lifted", "hammer"]],
        )


class TestComparisonAroundAVerdict(unittest.TestCase):
    def test_refused_for_once_hold_for_after(self):
        for text in (
            "(>= (once (fold-score towel)) 0.8)",
            "(>= (hold-for 2 (fold-score towel)) 0.8)",
            "(>= (after (grasped towel) (fold-score towel)) 0.8)",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_condition(parse_sexp(text)[0])

    def test_peak_still_takes_one(self):
        condition = parse_condition(parse_sexp("(<= (peak (tension film)) 12.0)")[0])
        self.assertEqual((condition.modal, condition.op, condition.value), ("peak", "<=", 12.0))

    def test_build_goal_refuses_one_that_got_through(self):
        inner = Condition(head="fold-score")
        condition = Condition(head="once", modal="once", children=(inner,), op=">=", value=0.8)
        with self.assertRaises(ValueError):
            build_goal(condition, lambda *_: None)


class TestSuiteName(unittest.TestCase):
    def test_kept_as_written(self):
        suite = _suite("(define (suite MySuite))")
        self.assertEqual(suite.name, "MySuite")
        doc = merge_suite(_problem("(define (problem v1) (:suite MySuite))"), suite)
        self.assertEqual(doc.suite, "MySuite")


class TestStagingMergesClauseByClause(unittest.TestCase):
    def test_variant_count_keeps_suite_sampler(self):
        suite = _suite(
            "(define (suite s) (:staging (:scene bench_0728)"
            " (:episodes (:count 50) (:sampler foo) (:model-key-fields layout))))"
        )
        doc = merge_suite(_problem("(define (problem v1) (:staging (:episodes (:count 5))))"), suite)
        self.assertEqual(
            doc.staging["episodes"],
            [[":sampler", "foo"], [":model-key-fields", "layout"], [":count", 5]],
        )
        self.assertEqual(doc.staging["scene"], "bench_0728")

    def test_a_value_with_a_head_is_replaced_whole(self):
        suite = _suite("(define (suite s) (:staging (:embodiment yam_bimanual (:work-arm left))))")
        doc = merge_suite(_problem("(define (problem v1) (:staging (:embodiment panda)))"), suite)
        self.assertEqual(doc.staging["embodiment"], "panda")


if __name__ == "__main__":
    unittest.main()
