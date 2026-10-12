# SPDX-License-Identifier: Apache-2.0
"""The goal language's semantics, pinned exactly.

Driven by a **scripted tape** -- a sequence of contexts with controlled
``sim_time`` and a leaf whose verdict the test dictates. No solver, no scene,
no GPU. That is what makes it possible to assert things a rollout never could:
that a latch releases on ``reset`` and not before, that an interruption restarts
a hold, that a peak survives a dip.

Two of these tests exist because the bug they catch is otherwise invisible:

* ``once`` must latch **success only** and pass reward through. Both shipped
  latching evaluators report a live reward under a latched verdict, so a ``once``
  that froze both would diverge from every recorded trace on the descending edge
  -- and would still look perfectly correct in any test that only checked the
  final step.
* ``hold_for`` counts **simulated seconds**, not steps. Running the same logical
  tape at two different rates is the only way that distinction is ever tested,
  and getting it wrong produces a task that silently means something different
  at a different control rate.
"""

from __future__ import annotations

import unittest

from robosimstudio.scene.state import WorldState
from robosimstudio_bench.goals.combinators import (
    After,
    AllOf,
    AnyOf,
    HoldFor,
    Integral,
    Leaf,
    Node,
    Not,
    Once,
    Peak,
    WorstOf,
    combinator_names,
)
from robosimstudio_bench.goals.context import EvalResult, Evaluator

TIER = "fast"
"""No GPU. Gates every commit."""


def ctx(sim_time: float = 0.0, step: int = 0) -> WorldState:
    """A context carrying only what a combinator reads.

    The task/episode/handles/state fields are the environment's business and no
    combinator touches them, so a scripted context leaves them empty rather than
    building a scene to satisfy a type.
    """
    return WorldState(handles=None, state=None, step=step, sim_time=sim_time)


class Scripted(Node):
    """A leaf whose verdict the test dictates, step by step."""

    def __init__(self, script: list[tuple[bool, float]], label: str = "leaf") -> None:
        self.script = script
        self.label = label
        self.calls = 0

    def evaluate(self, _ctx: WorldState) -> EvalResult:
        success, reward = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return EvalResult(success=success, reward=reward, metrics={"value": reward})


def const(success: bool, reward: float = 0.0, label: str = "") -> Leaf:
    """A leaf that always says the same thing."""
    return Leaf(fn=lambda _c: EvalResult(success=success, reward=reward), label=label)


class TestTheOperators(unittest.TestCase):
    """``&``, ``|`` and ``~`` build the same nodes a document lowers to.

    The point of writing a goal in Python is that it stays as closed a grammar
    as the document one: these operators can only build nodes that already
    exist, so every reader of the tree is unchanged.
    """

    def test_they_build_the_same_nodes_a_document_lowers_to(self) -> None:
        a, b = const(True, label="a"), const(False, label="b")
        self.assertIsInstance(a & b, AllOf)
        self.assertIsInstance(a | b, AnyOf)
        self.assertIsInstance(~a, Not)

    def test_a_chain_flattens_rather_than_nesting(self) -> None:
        """Not cosmetic -- see the next test for what nesting would cost."""
        a, b, c = const(True), const(True), const(False)
        self.assertEqual(len((a & b & c).of), 3)
        self.assertEqual(len((a | b | c).of), 3)

    def test_nesting_would_change_the_reward_not_just_the_shape(self) -> None:
        """``AllOf``'s reward is the *share* of children satisfied, so a nested
        chain scores differently from a flat one over the same leaves. Two of
        three holding is 2/3 flat and 1/2 nested -- the same verdict, a
        different number, and the flat one is what a reader of ``a & b & c``
        expects."""
        leaves = (const(True), const(True), const(False))
        flat = leaves[0] & leaves[1] & leaves[2]
        nested = AllOf(of=(AllOf(of=leaves[:2]), leaves[2]))
        self.assertAlmostEqual(flat(ctx()).reward, 2.0 / 3.0)
        self.assertAlmostEqual(nested(ctx()).reward, 1.0 / 2.0)

    def test_a_labelled_node_is_not_absorbed(self) -> None:
        """Labels are assigned when a tree is built, so a labelled ``AllOf`` is
        one somebody already committed to and a chain leaves it whole."""
        committed = AllOf(of=(const(True), const(True)), label="goal.0")
        joined = committed & const(False)
        self.assertEqual(len(joined.of), 2)
        self.assertIs(joined.of[0], committed)

    def test_the_operators_agree_with_the_verdicts_they_stand_for(self) -> None:
        self.assertTrue((const(True) & const(True))(ctx()).success)
        self.assertFalse((const(True) & const(False))(ctx()).success)
        self.assertTrue((const(False) | const(True))(ctx()).success)
        self.assertTrue((~const(False))(ctx()).success)


class TestAGoalRefusesToBeABoolean(unittest.TestCase):
    """The trap operator overloading always sets, guarded the way SQLAlchemy and
    pandas guard it."""

    def test_truthiness_raises(self) -> None:
        with self.assertRaises(TypeError):
            bool(const(True) & const(True))

    def test_a_chained_comparison_cannot_silently_discard_half_the_goal(self) -> None:
        """``0.2 <= x <= 0.8`` is ``(0.2 <= x) and (x <= 0.8)``, and ``and``
        takes the truth of the left operand -- so without this guard the second
        half is dropped and the goal scores something plausible and wrong."""

        class Measure(Node):
            def __ge__(self, other):
                return const(True)

            def __le__(self, other):
                return const(True)

            def evaluate(self, _ctx):
                return EvalResult(success=True, reward=1.0)

        with self.assertRaises(TypeError):
            _ = 0.2 <= Measure() <= 0.8


class TestTheSetIsClosed(unittest.TestCase):
    def test_exactly_these_nine(self) -> None:
        """The firewall. Adding a combinator requires editing this list and
        saying why -- which is the point: expressivity grows by minting atoms,
        which are code and get tested and calibrated, never by growing the
        grammar.

        **This list grew once, from eight to nine.** ``not`` was added because
        BDDL has it, LIBERO problems use it, and this stack reads those files --
        so the parser would otherwise accept a form the compiler could not lower,
        which is a footgun aimed at whoever imports a task.

        That is a different justification from the other eight, each of which was
        forced by a shipped evaluator, and it is the only other kind that should
        ever qualify: **a form the input language already has.** Wanting to
        express something new is what minting an atom is for.
        """
        self.assertEqual(
            combinator_names(),
            ["after", "all", "any", "hold_for", "integral", "not", "once", "peak", "worst_of"],
        )

    def test_every_combinator_is_an_evaluator(self) -> None:
        """A compiled goal tree must drop in exactly where a hand-written
        evaluator goes today, so nothing downstream changes."""
        leaf = const(True)
        built = {
            "all": AllOf(of=(leaf,)),
            "any": AnyOf(of=(leaf,)),
            "not": Not(of=leaf),
            "worst_of": WorstOf(of=(leaf,)),
            "once": Once(of=leaf),
            "hold_for": HoldFor(of=leaf, seconds=1.0),
            "after": After(event=leaf, of=leaf),
            "peak": Peak(of=leaf),
            "integral": Integral(of=leaf),
        }
        self.assertEqual(sorted(built), combinator_names(), "every registered combinator needs a case here")
        for name, node in built.items():
            with self.subTest(combinator=name):
                self.assertIsInstance(node, Evaluator)
                self.assertIsInstance(node(ctx()), EvalResult)


class TestConjunction(unittest.TestCase):
    def test_reward_is_the_satisfied_share(self) -> None:
        """What makes a half-sorted bin legible as 0.5 rather than a failure."""
        node = AllOf(of=(const(True), const(True), const(False), const(False)))
        result = node(ctx())
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.reward, 0.5)

    def test_all_of_everything_succeeds(self) -> None:
        result = AllOf(of=(const(True), const(True)))(ctx())
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.reward, 1.0)

    def test_any_takes_the_best_child(self) -> None:
        result = AnyOf(of=(const(False, 0.2), const(True, 0.9)))(ctx())
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.reward, 0.9)

    def test_children_metrics_are_namespaced(self) -> None:
        """Two creases must not overwrite each other's numbers."""
        node = AllOf(of=(Scripted([(True, 0.9)], "far"), Scripted([(False, 0.4)], "near")))
        metrics = node(ctx()).metrics
        self.assertAlmostEqual(metrics["far.value"], 0.9)
        self.assertAlmostEqual(metrics["near.value"], 0.4)

    def test_a_namespace_is_applied_once_however_deep_the_tree(self) -> None:
        """A label is an absolute path, so a grandchild arrives already named.

        Prefixing it again compounds: the shirt's creases were reporting at
        `goal.0.1.goal.0.1.0.score`, one repetition per level. Only that suite
        has a goal deep enough to show it -- everything else is a clause or two
        under one connective, where the first prefix is also the last. It cost the
        shirt the one reading its own document calls the diagnostic: which crease
        is the weak one.
        """
        inner = AllOf(of=(Scripted([(True, 0.9)], "a.0"), Scripted([(False, 0.4)], "a.1")), label="a")
        metrics = AllOf(of=(inner,), label="")(ctx()).metrics
        self.assertIn("a.0.value", metrics)
        self.assertEqual([key for key in metrics if key.count("a.") > 1], [])


class TestWorstOf(unittest.TestCase):
    def test_scores_the_weakest_member(self) -> None:
        """The shirt: two beautiful folds and one that missed must not average
        their way to a pass."""
        node = WorstOf(of=(Scripted([(True, 0.94)], "far"), Scripted([(True, 0.42)], "hem")))
        result = node(ctx())
        self.assertAlmostEqual(result.reward, 0.42)

    def test_names_the_weak_member(self) -> None:
        """The difference between '0.42' and 'the hem crease is the one that failed'."""
        node = WorstOf(of=(Scripted([(True, 0.94)], "far"), Scripted([(True, 0.42)], "hem")))
        self.assertEqual(node(ctx()).metrics["worst"], "hem")


class TestOnce(unittest.TestCase):
    def test_latches_success(self) -> None:
        """A towel brushed by a retreating forearm has still been folded."""
        leaf = Scripted([(False, 0.1), (True, 0.9), (False, 0.3)])
        node = Once(of=leaf)
        self.assertFalse(node(ctx()).success)
        self.assertTrue(node(ctx()).success)
        self.assertTrue(node(ctx()).success, "a latched verdict must survive the condition lapsing")

    def test_reward_passes_through_live(self) -> None:
        """The asymmetry that would otherwise break two ports silently: both
        shipped latching evaluators report a live reward under a latched verdict."""
        leaf = Scripted([(True, 0.9), (False, 0.3)])
        node = Once(of=leaf)
        self.assertAlmostEqual(node(ctx()).reward, 0.9)
        second = node(ctx())
        self.assertTrue(second.success, "success stays latched")
        self.assertAlmostEqual(second.reward, 0.3, msg="reward must NOT latch")

    def test_reports_whether_it_holds_now(self) -> None:
        node = Once(of=Scripted([(True, 1.0), (False, 0.0)]))
        node(ctx())
        metrics = node(ctx()).metrics
        self.assertTrue(metrics["latched"])
        self.assertFalse(metrics["holding_now"])


class TestHoldFor(unittest.TestCase):
    def test_counts_simulated_seconds_not_steps(self) -> None:
        """The same tape at two rates. A hold stated in steps silently means
        something different the moment a task runs at another rate, and this is
        the only test that can tell the difference."""
        for dt, expected_step in ((0.1, 50), (0.2, 25)):
            with self.subTest(dt=dt):
                node = HoldFor(of=const(True), seconds=5.0)
                fired = None
                for i in range(1, 80):
                    if node(ctx(sim_time=i * dt)).success:
                        fired = i
                        break
                self.assertIsNotNone(fired)
                # First sample at or past 5.0 s of holding. The clock starts on
                # the first satisfied step, so the hold completes one step later.
                self.assertEqual(fired, expected_step + 1)

    def test_an_interruption_restarts_the_clock(self) -> None:
        """'Held' is a claim about continuity. Cloth relaxes: a fold that looks
        perfect at release and springs back a second later has not held."""
        leaf = Scripted([(True, 1.0)] * 4 + [(False, 0.0)] + [(True, 1.0)] * 40)
        node = HoldFor(of=leaf, seconds=1.0)
        for i in range(1, 5):
            node(ctx(sim_time=i * 0.5))
        node(ctx(sim_time=2.5))  # the interruption
        self.assertFalse(node(ctx(sim_time=3.0)).success, "the clock must restart, not resume")
        node(ctx(sim_time=3.5))
        self.assertTrue(node(ctx(sim_time=4.0)).success)

    def test_reports_how_long_it_has_held(self) -> None:
        node = HoldFor(of=const(True), seconds=4.0)
        node(ctx(sim_time=1.0))
        metrics = node(ctx(sim_time=3.0)).metrics
        self.assertAlmostEqual(metrics["held_s"], 2.0)
        self.assertAlmostEqual(metrics["hold_progress"], 0.5)


class TestAfter(unittest.TestCase):
    def test_does_not_evaluate_the_child_before_the_event(self) -> None:
        """'Not yet asked' and 'asked and false' are different facts, and a
        report that conflates them misleads whoever reads it."""
        child = Scripted([(True, 1.0)])
        node = After(event=Scripted([(False, 0.0), (True, 1.0)], "release"), of=child)
        result = node(ctx())
        self.assertFalse(result.success)
        self.assertEqual(child.calls, 0, "the child must not be evaluated before the event")
        self.assertEqual(result.metrics["awaiting"], "release")

    def test_scores_the_child_once_the_event_fires(self) -> None:
        child = Scripted([(True, 0.8)])
        node = After(event=Scripted([(False, 0.0), (True, 1.0)]), of=child)
        node(ctx())
        result = node(ctx())
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.reward, 0.8)

    def test_the_event_latches(self) -> None:
        """The cable is released once; that it is no longer *being* released does
        not un-ask the question."""
        node = After(event=Scripted([(True, 1.0), (False, 0.0), (False, 0.0)]), of=const(True))
        node(ctx())
        self.assertTrue(node(ctx()).success)
        self.assertTrue(node(ctx()).success)


class TestPeakAndIntegral(unittest.TestCase):
    def test_peak_survives_a_dip(self) -> None:
        """Film peel: a clean peel and one that tears the film are identical at
        the final step and differ only in a tension the episode passed through."""
        leaf = Scripted([(True, 0.0)] * 3, "load")
        leaf.script = [(True, 1.0), (True, 9.0), (True, 2.0)]
        node = Peak(of=leaf, key="value", op="<=", value=5.0)
        self.assertTrue(node(ctx()).success)
        self.assertFalse(node(ctx()).success, "the spike must fail the budget")
        result = node(ctx())
        self.assertFalse(result.success, "and must keep failing it after the dip")
        self.assertAlmostEqual(result.metrics["peak"], 9.0)

    def test_integral_accumulates_against_simulated_time(self) -> None:
        """The dose rather than the spike -- and per-second, so it means the same
        thing at any control rate."""
        leaf = Scripted([(True, 2.0)] * 5, "load")
        node = Integral(of=leaf, key="value", op="<=", value=3.0)
        node(ctx(sim_time=0.0))  # first sample only starts the clock
        node(ctx(sim_time=1.0))  # + 2.0 * 1.0
        result = node(ctx(sim_time=1.5))  # + 2.0 * 0.5
        self.assertAlmostEqual(result.metrics["integral"], 3.0)
        self.assertTrue(result.success)
        self.assertFalse(node(ctx(sim_time=2.0)).success)


class TestReset(unittest.TestCase):
    def test_reset_clears_every_stateful_node(self) -> None:
        """Generated over the registry, so a ninth combinator cannot skip it.

        An evaluator that does not fully reset makes episode ``i``'s verdict
        depend on episode ``i-1``, which is the least reproducible bug a
        benchmark can have.
        """
        cases = {
            "once": lambda: Once(of=Scripted([(True, 1.0), (False, 0.0)])),
            "hold_for": lambda: HoldFor(of=const(True), seconds=1.0),
            "after": lambda: After(event=Scripted([(True, 1.0), (False, 0.0)]), of=const(True)),
            "peak": lambda: Peak(of=Scripted([(True, 9.0), (True, 0.0)]), key="value", op="<=", value=5.0),
            "integral": lambda: Integral(of=const(True, 2.0), key="", op="<=", value=1.0),
            "all": lambda: AllOf(of=(const(True),)),
            "any": lambda: AnyOf(of=(const(True),)),
            "not": lambda: Not(of=const(True)),
            "worst_of": lambda: WorstOf(of=(const(True),)),
        }
        self.assertEqual(sorted(cases), combinator_names(), "every combinator needs a reset case")

        for name, build in cases.items():
            with self.subTest(combinator=name):
                node = build()
                first = [node(ctx(sim_time=t)).success for t in (0.0, 1.0, 2.0)]
                node.reset(ctx())
                if isinstance(node, Scripted):  # pragma: no cover - defensive
                    continue
                for child in node.children():
                    if isinstance(child, Scripted):
                        child.calls = 0
                again = [node(ctx(sim_time=t)).success for t in (0.0, 1.0, 2.0)]
                self.assertEqual(first, again, "a reset node must score the same sequence again")


if __name__ == "__main__":
    unittest.main()
