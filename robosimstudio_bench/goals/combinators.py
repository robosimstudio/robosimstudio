# SPDX-License-Identifier: Apache-2.0
"""The closed set of ways to combine measures into a goal.

Dissecting the shipped evaluators turns up a clean split. The **measures** are
not compositional -- a winding angle, SoftGym's fold error, a peel's peak tension
are irreducibly code, and pretending otherwise produces a framework nobody can
express a real task in. But the **goals over the measures** are compositional,
and shallowly: conjunction, a worst-case over a set, and five temporal forms
cover every one of the seventeen variants.

So this set is **closed**. Expressivity grows by minting atoms -- which are code,
tested, calibrated and reviewed -- never by adding a combinator. That is the
firewall that keeps a task language from silently becoming a bad programming
language, and the golden list in the test suite is what enforces it: adding one
requires editing a golden and explaining why.

Each of these, and what forced it:

======================  =================================================
``all`` / ``any``       conjunction / disjunction. Every suite.
``not``                 negation. **Forced by the input language, not by a
                        suite** -- see :class:`Not`. The only other kind of
                        justification that should ever add to this set.
``worst_of``            min over a set. The shirt, scored on its worst crease.
``once``                latch. The towel: a fold that has happened has happened.
``hold_for``            must hold for N **simulated seconds**. The towel and
                        shirt's released-and-settled, GarmentLab's own 5 s.
``after``               evaluate only after an event. The cable wrap, scored on
                        turns *retained after release*.
``peak`` / ``integral`` temporal max / accumulation. Film peel's peak adhesive
                        tension -- a clean peel and one that tears the film are
                        the same footage otherwise.
======================  =================================================

Two names a reader may expect are deliberately absent. **``fraction`` is not a
combinator**: reward is often not "the share of goal predicates" -- insertion
depth over its gate, an inverse mate gap, a progress term masked to zero by a
tension budget -- so reward is its own
clause over the same terms rather than a node type. A node still reports its
natural partial credit here, and the reward clause may override it entirely.
**``for_each`` is compile-time sugar**: a quantifier over a class expands to
``all`` of its members before anything runs, so it is the compiler's business,
not a runtime node.

One semantic is worth stating loudly because getting it wrong breaks two ports
silently. **``once`` latches success and passes reward through.** Both the tape
and the cloth evaluators latch their verdict while continuing to report a live
reward, so a ``once`` that latched both would freeze the reward at the instant of
success and every golden trace would diverge on the descending edge.

Every node satisfies the existing ``Evaluator`` protocol -- ``reset(ctx)`` plus
``__call__(ctx) -> EvalResult`` -- so a compiled goal tree drops into the
environment exactly where a hand-written evaluator goes today, and nothing
downstream changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from robosimstudio.assetlib.semantics.states import COMPARISONS

from .context import EvalResult, WorldState

__all__ = [
    "COMBINATORS",
    "After",
    "AllOf",
    "AnyOf",
    "HoldFor",
    "Integral",
    "Leaf",
    "Node",
    "Not",
    "Once",
    "Peak",
    "WorstOf",
    "combinator_names",
]


class Node:
    """A goal-tree node: an ``Evaluator`` that composes.

    Subclasses implement :meth:`evaluate`; :meth:`__call__` wraps it so every
    node contributes its own verdict to ``metrics`` under its label. That
    per-node decomposition is what makes a zero-reward run legible -- "the wrap
    held but the cable never came to rest" instead of "0.0".
    """

    label: str = ""

    def reset(self, ctx: WorldState) -> None:
        """Clear per-episode state. Overridden by every stateful node."""
        for child in self.children():
            child.reset(ctx)

    def children(self) -> tuple[Node, ...]:
        """Sub-nodes, for reset and for walking the tree."""
        return ()

    def evaluate(self, ctx: WorldState) -> EvalResult:
        raise NotImplementedError

    def __call__(self, ctx: WorldState) -> EvalResult:
        result = self.evaluate(ctx)
        if self.label:
            result.metrics.setdefault(self.label, bool(result.success))
        return result

    # ----------------------------------------------------------- operators
    #
    # `&`, `|` and `~` build the *same* frozen nodes a document lowers to, so a
    # goal written in Python and one parsed from BDDL are one tree and every
    # reader of it -- `outline()`, `placements()`, `leaves()`, the gauntlet's
    # lint, the studio's goal panel -- is unchanged. The vocabulary stays as
    # closed as it is now: these operators can only build nodes that already
    # exist.

    def __and__(self, other: Node) -> Node:
        """``a & b`` -- every child holds.

        **Chains flatten, and that is not cosmetic.** ``AllOf``'s reward is the
        *share* of its children satisfied, so nesting ``(a & b) & c`` scores 1/2
        where ``a & b & c`` scores 2/3 -- the same verdict but a different
        reward, and the shape a reader would not expect. Only an unlabelled
        ``AllOf`` is absorbed: labels are assigned when the tree is built, so a
        labelled one is a node somebody already committed to.
        """
        if isinstance(self, AllOf) and not self.label:
            return AllOf(of=(*self.of, other))
        return AllOf(of=(self, other))

    def __or__(self, other: Node) -> Node:
        """``a | b`` -- any child holds. Flattens for the same reason."""
        if isinstance(self, AnyOf) and not self.label:
            return AnyOf(of=(*self.of, other))
        return AnyOf(of=(self, other))

    def __invert__(self) -> Node:
        """``~a`` -- the child does not hold."""
        return Not(of=self)

    def __bool__(self) -> bool:
        """Always raises.

        A tree built by operator overloading is otherwise silently collapsed by
        ``if goal:`` -- and worse by a chained comparison, where Python turns
        ``0.2 <= x <= 0.8`` into ``(0.2 <= x) and (x <= 0.8)`` and ``and`` takes
        the truth of the left operand. That would evaluate a *goal* as a boolean
        and quietly discard half the expression. SQLAlchemy and pandas both
        guard this; the cost of not guarding it is a goal that scores something
        plausible and wrong.
        """
        raise TypeError(
            f"a goal is a tree, not a boolean: {type(self).__name__} cannot be used in "
            "`if`, `and`, `or`, or a chained comparison. Use `&`, `|` and `~`, and write "
            "`(a >= 1) & (b <= 2)` rather than `1 <= a <= 2`."
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _merge(results: Sequence[EvalResult], labels: Sequence[str]) -> dict[str, Any]:
        """Children's metrics, namespaced by label so two creases do not collide.

        Keys that already carry a name are left alone. A child's *own* verdict
        key, because every node records its verdict under its own label and
        prefixing that again produces ``goal.0.goal.0`` -- unreadable, and a
        caller looking for a clause's verdict at the name the clause was given
        would not find it. Anything already *under* that label, for the same
        reason one level further down. And any key containing a ``/``, because
        that is the binder's mark for a key that names its own subject:
        ``containment/hammer_0/bin2_hammer_region`` is already unique and already
        says what it is, where
        ``goal.2.containment/hammer_0/bin2_hammer_region`` only says it twice.

        What does need the namespace is a bare measure name -- two creases both
        reporting ``value``, two clauses both reporting ``fraction`` -- and those
        are exactly the keys that carry no subject of their own.

        **The "already under it" case was missing, and it compounds.** A label is
        an absolute path, so a grandchild's key reaches its grandparent already
        namespaced and was namespaced again, once per level: the shirt's three
        creases reported at ``goal.0.1.goal.0.1.0.score``. Only that suite has a
        goal deep enough to show it -- everywhere else is a clause or two under
        one connective, where the first prefix is also the last -- and it cost the
        shirt exactly the reading its own document calls the diagnostic, which is
        which crease is the weak one.
        """
        out: dict[str, Any] = {}
        for label, result in zip(labels, results, strict=False):
            for key, value in result.metrics.items():
                keep = not label or key == label or key.startswith(f"{label}.") or "/" in key
                out[key if keep else f"{label}.{key}"] = value
        return out


@dataclass
class Leaf(Node):
    """A measure, wrapped as a node.

    The seam between the compositional half of a goal and the irreducible half.
    ``fn`` is whatever actually measures -- an atom bound to its operands in the
    compiled case, a hand-written closure in a test -- and it returns either an
    ``EvalResult`` or a ``(success, reward, metrics)`` triple.

    Keeping this deliberately dumb is what lets the combinators be tested against
    scripted sequences with no simulator at all.

    Attributes:
        fn: What measures.
        label: Where this sits in the goal tree.
        about: ``(atom, operand names)`` -- what this leaf is *about*, as opposed
            to how it computes. Set by the binder and empty for a hand-written
            closure. It exists because a scripted policy legitimately needs to
            know which object goes in which region, and the goal is where that
            fact lives: before this, both the tool-sort routine and its tool
            registry read the pairs out of ``TaskSpec.metadata``, an untyped side
            channel carrying a copy of something the goal already stated. A copy
            can disagree with the original, and one that does produces a policy
            solving a task nobody is scoring.
    """

    fn: Callable[[WorldState], Any]
    label: str = ""
    about: tuple[str, tuple[str, ...]] = ("", ())
    on_reset: Callable[[], None] | None = None
    """Called when the episode resets, for a measure that carries state.

    Measures are meant to be pure functions of a snapshot and nearly all are.
    The exception is a *smoothed* one -- a reading taken as the largest over a
    short window, because the instrument is noisy -- and that window has to be
    emptied between episodes or the first steps of one are scored on the last
    steps of the previous. This is the hook that empties it, and it is on the
    leaf rather than in the combinator set because it is measurement state, not
    goal state: nothing about the task changes when a buffer clears.
    """

    def reset(self, ctx: WorldState) -> None:
        super().reset(ctx)
        if self.on_reset is not None:
            self.on_reset()

    def evaluate(self, ctx: WorldState) -> EvalResult:
        got = self.fn(ctx)
        if isinstance(got, EvalResult):
            # Copied rather than passed through: a node may namespace its
            # children's metrics, and mutating a result the caller still holds
            # is how one crease's numbers end up under another's label.
            return EvalResult(success=got.success, reward=got.reward, metrics=dict(got.metrics))
        if isinstance(got, tuple):
            success, reward = got[0], got[1]
            metrics = got[2] if len(got) > 2 else {}
            return EvalResult(success=bool(success), reward=float(reward), metrics=dict(metrics or {}))
        return EvalResult(success=bool(got), reward=float(bool(got)))


@dataclass
class AllOf(Node):
    """Every child holds.

    Reward is the *share* of children satisfied, which is what makes a
    half-sorted bin legible as 0.5 rather than simply a failure -- BDDL's own
    reward, and what ``all_satisfied`` already computes.
    """

    of: tuple[Node, ...]
    label: str = ""

    def children(self) -> tuple[Node, ...]:
        return self.of

    def evaluate(self, ctx: WorldState) -> EvalResult:
        results = [child(ctx) for child in self.of]
        if not results:
            return EvalResult(success=False, reward=0.0)
        satisfied = sum(1 for r in results if r.success)
        return EvalResult(
            success=satisfied == len(results),
            reward=satisfied / len(results),
            metrics=self._merge(results, [c.label for c in self.of]),
        )


@dataclass
class AnyOf(Node):
    """At least one child holds. Reward is the best child's."""

    of: tuple[Node, ...]
    label: str = ""

    def children(self) -> tuple[Node, ...]:
        return self.of

    def evaluate(self, ctx: WorldState) -> EvalResult:
        results = [child(ctx) for child in self.of]
        if not results:
            return EvalResult(success=False, reward=0.0)
        return EvalResult(
            success=any(r.success for r in results),
            reward=max(r.reward for r in results),
            metrics=self._merge(results, [c.label for c in self.of]),
        )


@dataclass
class Not(Node):
    """The child does not hold.

    A ninth combinator, and worth saying why it exists when the set is supposed
    to be closed. It is **earned by the dialect, not by wanting expressivity**:
    ``Not`` is standard BDDL, LIBERO problems use it, and this stack reads those
    files. Refusing it would mean the parser accepting a form the compiler then
    could not lower -- a footgun aimed squarely at whoever imports a task.

    That is a different justification from the other eight, which were each
    forced by a shipped evaluator, and it is the only kind of justification that
    should ever add to this set: a form the *input language* already has. Wanting
    to say something new is what minting an atom is for.

    Reward is the complement, which is the only defensible reading -- a condition
    half-satisfied is half-violated.
    """

    of: Node
    label: str = ""

    def children(self) -> tuple[Node, ...]:
        return (self.of,)

    def evaluate(self, ctx: WorldState) -> EvalResult:
        inner = self.of(ctx)
        return EvalResult(
            success=not inner.success,
            reward=1.0 - inner.reward,
            metrics=dict(inner.metrics),
        )


@dataclass
class WorstOf(Node):
    """Every member holds, and the score is the worst of them.

    Not the same statement as :class:`All`, and the shirt is why: three creases
    each scored on their own quality, reported on the weakest, so a garment with
    two beautiful folds and one that missed does not average its way to a pass.
    """

    of: tuple[Node, ...]
    label: str = ""

    def children(self) -> tuple[Node, ...]:
        return self.of

    def evaluate(self, ctx: WorldState) -> EvalResult:
        results = [child(ctx) for child in self.of]
        if not results:
            return EvalResult(success=False, reward=0.0)
        worst = min(range(len(results)), key=lambda i: results[i].reward)
        metrics = self._merge(results, [c.label for c in self.of])
        # Which member is the weak one, by name -- the difference between "0.42"
        # and "the hem crease is the one that failed".
        metrics["worst"] = self.of[worst].label or str(worst)
        return EvalResult(
            success=all(r.success for r in results),
            reward=results[worst].reward,
            metrics=metrics,
        )


@dataclass
class Once(Node):
    """Success, once achieved, stays achieved. Reward stays live.

    A towel that has been folded and is then brushed by a retreating forearm has
    still been folded, and scoring only the final step would make episode length
    part of the measurement.

    The asymmetry between success and reward is deliberate and load-bearing: the
    shipped tape and cloth evaluators both latch their verdict while continuing
    to report the live quantity, so latching reward too would freeze it at the
    instant of success and diverge from every recorded trace on the way down.
    """

    of: Node
    label: str = ""
    _latched: bool = field(default=False, init=False)

    def children(self) -> tuple[Node, ...]:
        return (self.of,)

    def reset(self, ctx: WorldState) -> None:
        super().reset(ctx)
        self._latched = False

    def evaluate(self, ctx: WorldState) -> EvalResult:
        inner = self.of(ctx)
        self._latched = self._latched or bool(inner.success)
        metrics = dict(inner.metrics)
        metrics["latched"] = bool(self._latched)
        metrics["holding_now"] = bool(inner.success)
        return EvalResult(success=self._latched, reward=inner.reward, metrics=metrics)


@dataclass
class HoldFor(Node):
    """The child must hold continuously for ``seconds`` of **simulated** time.

    Simulated seconds, never control steps: the requirement is a physical one
    about the world settling, and a step count silently means something different
    the moment a task runs at a different rate. Cloth relaxes -- a fold looks
    perfect the instant the fingers open and can spring halfway back a second
    later -- so an evaluator sampling the instant of release scores the pose the
    robot passed through rather than the one it left behind.

    Any interruption restarts the clock. That is the point: "held" is a claim
    about continuity.
    """

    of: Node
    seconds: float
    label: str = ""
    _since: float = field(default=-1.0, init=False)

    def children(self) -> tuple[Node, ...]:
        return (self.of,)

    def reset(self, ctx: WorldState) -> None:
        super().reset(ctx)
        self._since = -1.0

    def evaluate(self, ctx: WorldState) -> EvalResult:
        inner = self.of(ctx)
        now = float(ctx.sim_time)
        if not inner.success:
            self._since = -1.0
        elif self._since < 0.0:
            self._since = now
        held = 0.0 if self._since < 0.0 else max(0.0, now - self._since)
        metrics = dict(inner.metrics)
        metrics["held_s"] = held
        metrics["hold_progress"] = min(1.0, held / self.seconds) if self.seconds > 0 else 1.0
        return EvalResult(
            success=bool(inner.success and held >= self.seconds),
            reward=inner.reward,
            metrics=metrics,
        )


@dataclass
class After(Node):
    """The child is scored only once ``event`` has happened.

    The cable wrap is the reason: a coil is only wrapped if it is *still* wrapped
    once the hand lets go, so the turns that count are the ones retained after
    release. Before the event the child is not evaluated at all -- it contributes
    nothing rather than contributing a failure, because "not yet asked" and
    "asked and false" are different facts and a report that conflates them
    misleads whoever reads it.

    The event latches on its rising edge: once it has happened it has happened,
    even if the condition that signalled it stops holding.
    """

    event: Node
    of: Node
    label: str = ""
    _fired: bool = field(default=False, init=False)

    def children(self) -> tuple[Node, ...]:
        return (self.event, self.of)

    def reset(self, ctx: WorldState) -> None:
        super().reset(ctx)
        self._fired = False

    def evaluate(self, ctx: WorldState) -> EvalResult:
        if not self._fired:
            self._fired = bool(self.event(ctx).success)
        if not self._fired:
            return EvalResult(success=False, reward=0.0, metrics={"awaiting": self.event.label or "event"})
        inner = self.of(ctx)
        metrics = dict(inner.metrics)
        metrics["fired"] = True
        return EvalResult(success=inner.success, reward=inner.reward, metrics=metrics)


def _reading(inner: EvalResult, key: str, word: str) -> float:
    """The quantity ``peak`` / ``integral`` thresholds: metric ``key``, or the leaf's reward for none.

    Raises:
        KeyError: When ``key`` names a metric the measure did not report -- the
            reward clause's rule (``goals/reward.py::_quantity``). Falling back
            to the leaf's normalised reward thresholded a different number that
            looked just as plausible.
    """
    if not key:
        return float(inner.reward)
    if key not in inner.metrics:
        raise KeyError(f"{word} reads {key!r}, but the measure reported {sorted(inner.metrics)}")
    return float(inner.metrics[key])


@dataclass
class Peak(Node):
    """The largest value a measure reached this episode, thresholded.

    Film peel is the case that needs it: a clean peel and one that tears the film
    look identical at the final step, and differ only in a tension the episode
    passed through. A measure that is only ever read at the end cannot see it.
    """

    of: Node
    key: str = ""
    op: str = "<="
    value: float = 0.0
    label: str = ""
    _peak: float = field(default=float("-inf"), init=False)

    def children(self) -> tuple[Node, ...]:
        return (self.of,)

    def reset(self, ctx: WorldState) -> None:
        super().reset(ctx)
        self._peak = float("-inf")

    def evaluate(self, ctx: WorldState) -> EvalResult:
        inner = self.of(ctx)
        current = _reading(inner, self.key, "peak")
        self._peak = max(self._peak, current)
        metrics = dict(inner.metrics)
        metrics["peak"] = self._peak
        return EvalResult(
            success=COMPARISONS[self.op](self._peak, float(self.value)),
            reward=inner.reward,
            metrics=metrics,
        )


@dataclass
class Integral(Node):
    """A measure accumulated over simulated time, thresholded.

    The dose rather than the spike -- how much of something happened, not how bad
    the worst instant was. Accumulated against ``sim_time`` for the same reason
    :class:`HoldFor` counts seconds: a per-step sum means something different at a
    different control rate.
    """

    of: Node
    key: str = ""
    op: str = "<="
    value: float = 0.0
    label: str = ""
    _total: float = field(default=0.0, init=False)
    _last_t: float = field(default=-1.0, init=False)

    def children(self) -> tuple[Node, ...]:
        return (self.of,)

    def reset(self, ctx: WorldState) -> None:
        super().reset(ctx)
        self._total = 0.0
        self._last_t = -1.0

    def evaluate(self, ctx: WorldState) -> EvalResult:
        inner = self.of(ctx)
        now = float(ctx.sim_time)
        current = _reading(inner, self.key, "integral")
        if self._last_t >= 0.0:
            self._total += current * max(0.0, now - self._last_t)
        self._last_t = now
        metrics = dict(inner.metrics)
        metrics["integral"] = self._total
        return EvalResult(
            success=COMPARISONS[self.op](self._total, float(self.value)),
            reward=inner.reward,
            metrics=metrics,
        )


COMBINATORS: dict[str, type[Node]] = {
    "all": AllOf,
    "any": AnyOf,
    "not": Not,
    "worst_of": WorstOf,
    "once": Once,
    "hold_for": HoldFor,
    "after": After,
    "peak": Peak,
    "integral": Integral,
}
"""The frozen set. Pinned by a golden in the test suite, so growing it is a
deliberate act with an explanation attached rather than a convenience somebody
reached for at 2am."""


def combinator_names() -> list[str]:
    """The combinator names, sorted -- what the closed-set golden compares."""
    return sorted(COMBINATORS)
