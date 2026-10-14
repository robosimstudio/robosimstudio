# SPDX-License-Identifier: Apache-2.0
"""Reward, which is not the goal.

The tempting design says reward is the share of goal predicates satisfied, and
that a goal tree therefore scores an episode by itself. It fits sorting tasks
-- the share of ``In`` predicates satisfied -- and little else. The kinds of
task this stack is built for score reward in other shapes:

=====================  ========================================================
Kind of task           What its reward is
=====================  ========================================================
sorting, folding       the share of goal predicates satisfied (``fraction``)
insertion by depth     ``clip(depth / gate, 0, 1)`` -- a quantity over a gate,
                       the goal being a conjunction the depth is only part of
mating by gap          ``clip(1 - gap / gate, 0, 1)`` -- the same shape read
                       from the other end, because the measure *shrinks* toward
                       success (``shrinking``)
peeling                ``min(1, broken / gate)`` -- and it keeps moving after
                       the verdict latches (``progress``)
peeling a film         the same, multiplied by a 0/1 mask: progress counts only
                       while the film is still within tolerance (``gated-by``)
=====================  ========================================================

So a task document states its reward separately, in one of exactly four forms,
and this module is those four. The set is closed for the same reason the
combinator set is: a fifth shape means a task nobody can compare against the
others.

**Reward passes through a latch.** ``once`` freezes *success*; the quantity goes
on moving. Freezing reward at the instant of success would hide what happened
after it -- a peel that kept going, a fold that came undone. :class:`Scored` takes success from the goal tree and reward from the
clause, and those really are two different reads of the same step.

**A measure reports its quantity by name, not as its reward.** A progress clause
names the metric key it reads (``(progress (broken-fraction tape) (:gate 0.95))``
reads ``broken-fraction``), because a leaf's ``reward`` is a normalised
satisfaction and the thing being divided by a gate is a raw number. Reading the
wrong one would produce a reward that looked plausible and was normalised twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robosimstudio_bench.taskdoc.parse import COMPARISONS
from robosimstudio_bench.taskdoc.typecheck import _GATE_WORDS, _MASK_WORDS

from .combinators import Node
from .context import EvalResult, WorldState, finite_reward

__all__ = ["Fraction", "Gated", "Progress", "RewardClause", "Scored", "build_reward", "measured_atom", "reading_key"]


def _clip(value: float) -> float:
    """``value`` clamped to [0, 1], with NaN read as 0.0 (see :func:`finite_reward`)."""
    value = finite_reward(value)
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


@dataclass
class Fraction:
    """The goal tree's own reward: the share of its conjuncts satisfied.

    The default, and what a sorting or folding task uses. Written as a
    clause rather than assumed, so that a document which says nothing about
    reward and one which says ``(:reward (fraction))`` compile to the same thing
    and mean the same thing.
    """

    def score(self, verdict: EvalResult, ctx: WorldState) -> tuple[float, dict[str, Any]]:
        del ctx
        return _clip(float(verdict.reward)), {}


@dataclass
class Progress:
    """A quantity over a gate: ``clip(v / gate)``, or ``clip(1 - v / gate)``.

    Both directions, because a measure may grow toward success (an insertion
    depth, a tape's broken-bond fraction) or shrink toward it (a connector's
    mating gap). One class, two words at the call site: ``progress``
    and ``shrinking``.

    Two words rather than ``(progress ... (:invert true))``, and the choice is
    close. A missing flag silently inverts the meaning; so does the wrong word,
    so neither is safer. What decides it is the reading: ``(shrinking (mate-gap
    plug) (:gate 0.021))`` says what it does, where a reward whose direction
    hangs on a boolean three tokens later has to be read twice.

    Attributes:
        of: The node that computes the quantity. Evaluated every step, and its
            metrics are kept -- an uncalibrated number nobody records is a
            number nobody can calibrate.
        key: Which of its metrics carries the raw quantity.
        gate: The value at which progress reads 1.0. Not necessarily the value at
            which the goal succeeds: a film peel gates reward at its success
            fraction while its goal also demands the film stayed within
            tolerance, and conflating them would make a torn film look finished.
        invert: Whether the measure shrinks toward success.
    """

    of: Node
    key: str
    gate: float
    invert: bool = False

    def score(self, verdict: EvalResult, ctx: WorldState) -> tuple[float, dict[str, Any]]:
        del verdict
        result = self.of(ctx)
        raw = _quantity(result, self.key)
        if not self.gate > 0.0:
            # Unreachable from a document: ``_progress`` refuses a gate <= 0 at
            # build time. A clause constructed directly with one gets the same
            # refusal here rather than a constant reward.
            raise ValueError(f"progress gate must be > 0, got {self.gate!r}")
        share = raw / self.gate
        return _clip(1.0 - share if self.invert else share), dict(result.metrics)

    def nodes(self) -> tuple[Node, ...]:
        return (self.of,)


@dataclass
class Gated:
    """Progress, multiplied by a 0/1 mask.

    A film peel's shape, and the reason it exists is worth stating: a film
    peeled 80% of the way and then torn is **not** 0.8 of a success, and a reward
    that merely dipped would let a policy trade tearing against speed. The mask
    is a verdict; the thing masked is progress.
    """

    of: Progress
    by: Node

    def score(self, verdict: EvalResult, ctx: WorldState) -> tuple[float, dict[str, Any]]:
        value, metrics = self.of.score(verdict, ctx)
        masking = self.by(ctx)
        metrics.update(masking.metrics)
        metrics.setdefault("reward_gate", bool(masking.success))
        return value * (1.0 if masking.success else 0.0), metrics

    def nodes(self) -> tuple[Node, ...]:
        return (*self.of.nodes(), self.by)


RewardClause = Fraction | Progress | Gated
"""The four shapes, closed. ``Progress`` covers two of them via ``invert``."""


def build_reward(
    section: dict[str, Any],
    make_node: Callable[[Any], Node | None],
    *,
    make_mask: Callable[[Any], Node | None] | None = None,
) -> RewardClause:
    """Build the clause a document's ``:reward`` section names.

    Reads the parsed section, which is the ``_plain`` shape -- ``(progress
    (broken-fraction tape) (:gate 0.95))`` arrives as
    ``{"progress": [<measure form>, [":gate", 0.95]]}``.

    Args:
        section: The parsed ``:reward``. Empty means :class:`Fraction`, because a
            document that says nothing about reward gets BDDL's own.
        make_node: Builds the node for a measure form -- the same maker the goal
            tree uses, so a measure means the same thing in both places. It
            answers ``None`` for a form the episode masked away (every clause
            of it about a parked object), and a measure masked away is refused.
        make_mask: Builds the node for a ``(:gated-by CLAUSE)``; ``make_node``
            when not given. Separate so the mask's per-node verdicts can be
            labelled apart from the measure's. A mask masked away to ``None``
            drops the gate, as a goal clause about a parked object is dropped.

    Raises:
        ValueError: On any shape outside the four. Refused rather than
            approximated: a reward that silently means something adjacent to what
            it says is a benchmark measuring the wrong thing, and unlike a goal it
            fails *quietly*, as a number that looks plausible.
    """
    if not section:
        return Fraction()

    if "fraction" in section:
        return Fraction()

    for word, invert in (("progress", False), ("shrinking", True)):
        if word in section:
            return _progress(section[word], make_node, make_mask or make_node, invert=invert, word=word)

    raise ValueError(
        f"reward names {sorted(section)}, and the clause set is closed: "
        "(fraction), (progress MEASURE (:gate V)), (shrinking MEASURE (:gate V)), "
        "and either progress form with an added (:gated-by CLAUSE)"
    )


def _progress(
    body: Any,
    make_node: Callable[[Any], Node | None],
    make_mask: Callable[[Any], Node | None],
    *,
    invert: bool,
    word: str,
) -> RewardClause:
    """One progress clause: a measure form, a gate, and optionally a mask.

    The mask reads as a modifier of the progress rather than as a wrapper around
    it -- ``(progress (broken-fraction film) (:gate 0.6) (:gated-by (intact
    film)))`` -- because that is how the thing is said out loud, and a reward
    clause that has to be read inside-out is one nobody checks.
    """
    parts = body if isinstance(body, list) else [body]
    measure = None
    gate = None
    mask = None
    for part in parts:
        keyword = str(part[0]).lstrip(":") if isinstance(part, list) and part else ""
        if keyword in _GATE_WORDS:
            if len(part) != 2:
                raise ValueError(f"({word} ...): (:gate V) takes one number, got {part[1:]!r}")
            gate = float(part[1])
            if not gate > 0.0:
                # Refused here, at build time: ``Progress.score`` divides by it
                # every step, and a zero or negative gate makes the reward a
                # constant (or a sign-flipped share) that looks like a result.
                raise ValueError(f"({word} ...): (:gate V) must be > 0, got {part[1]!r}")
        elif keyword in _MASK_WORDS:
            if len(part) != 2:
                raise ValueError(f"({word} ...): (:gated-by CLAUSE) takes one clause, got {part[1:]!r}")
            mask = part[1]
        elif measure is None:
            measure = part
        else:
            raise ValueError(f"({word} ...) measures one quantity, and was given a second: {part!r}")

    if measure is None:
        raise ValueError(f"({word} ...) names no measure, so there is nothing to be making progress on")
    if gate is None:
        # Required, not defaulted to 1.0. Every shipped suite divides by a
        # calibrated number -- a bore depth, a bond count, a travel -- and a
        # silent 1.0 would turn a raw millimetre reading into a reward of 1.0 on
        # the first step of the run.
        raise ValueError(f"({word} ...) has no (:gate V), and progress toward an unstated gate is not a quantity")

    atom = measured_atom(measure)
    head = atom[0] if isinstance(atom, list) and atom else atom
    of = make_node(measure)
    if of is None:
        raise ValueError(
            f"({word} ...) measures {measure!r}, which is about objects outside this episode's active set, "
            "so there is nothing to be making progress on"
        )
    clause = Progress(of=of, key=reading_key(str(head)), gate=gate, invert=invert)
    by = make_mask(mask) if mask is not None else None
    return Gated(of=clause, by=by) if by is not None else clause


def measured_atom(measure: Any) -> Any:
    """The atom call a progress measure reads, under any comparison around it.

    A binding with an op rewrites ``(In hammer bin)`` to ``(>= (containment
    hammer bin) 1.0)``; the parsed leaf is still the ``containment`` atom, so the
    quantity is its reading, and taking the head verbatim keyed the reward on
    ``>=``, which no atom reports.
    """
    if (
        isinstance(measure, list)
        and len(measure) == 3
        and str(measure[0]) in COMPARISONS
        and isinstance(measure[1], list)
        and measure[1]
    ):
        return measure[1]
    return measure


def reading_key(head: str) -> str:
    """Which metric key a measure's name refers to.

    ``atom:key`` selects, and a bare name means the atom's primary return.
    Resolved here rather than taken as the head verbatim, because a head is an
    *atom* name and the metrics are keyed by what the atom *reports* --
    ``insertion`` reports ``depth``, and reading the head as a key looked for a
    metric nobody emits.

    Public, and shared with :func:`~robosimstudio_bench.goals.goal.build_goal`,
    because ``peak`` and ``integral`` need the same answer for the quantity they
    threshold. Two copies of this rule is the shape of drift ``about_key``'s
    docstring records the cost of: every reader raises on a key the measure did
    not report, so a second copy that drifted would fail every episode.
    """
    from robosimstudio.assetlib.semantics.atoms import ATOMS

    name, _, selector = head.replace("-", "_").partition(":")
    if selector:
        return selector
    # A measure nothing has registered falls back to its own name. Not an error:
    # the structural half of this module is tested against three-line fakes with
    # no catalog at all, which is the whole reason building a clause and binding
    # one are separable.
    return ATOMS.get(name).primary if name in ATOMS else name


def _quantity(result: EvalResult, key: str) -> float:
    """One metric of a measure's result, as a number.

    Raises:
        KeyError: When the named metric is absent. Loud rather than defaulting to
            zero -- a silent zero is a reward that reads "no progress at all" for
            an entire run, which looks exactly like a policy that never moved.
    """
    if key not in result.metrics:
        raise KeyError(f"reward reads {key!r}, but the measure reported {sorted(result.metrics)}")
    return float(result.metrics[key])


@dataclass
class Scored:
    """A goal tree and a reward clause, as the one thing that scores an episode.

    Satisfies the ``Evaluator`` protocol, so it drops into ``TaskSpec.evaluator``
    exactly where a hand-written evaluator goes today.

    Attributes:
        goal: Decides success. Its own reward is what a ``Fraction`` clause reads.
        reward: Decides reward. Evaluated every step, independently.
        observe: Recorded every step and gating nothing -- the quantities nobody
            has calibrated yet. They are in ``metrics`` and never in ``success``,
            which is the whole distinction: reporting an uncalibrated number is
            how it gets calibrated, thresholding one is how a benchmark ends up
            measuring an arbitrary constant.
        doc: The document this was built from, carried rather than discarded.
            Nothing here reads it -- scoring is entirely the three fields above --
            but it is what lets a *describer* say ``hammer_0`` is a rigid body
            affording a ``grasp_box``, instead of a position and nothing else.
            Kept on the evaluator because that is the object a live scene already
            holds a reference to; the alternative was a second path from a
            running env back to its own definition.
        leaf: Builds the measure for one clause -- the same maker the goal was
            built with. Carried for the same reason ``doc`` is: verifying a
            task's ``:init`` needs to measure a constraint the way the goal
            would, and rebuilding a binder from outside would be a second path to
            the same answer, free to disagree with the first. ``None`` for an
            evaluator assembled without one, which makes :meth:`check_start`
            report nothing rather than fail.
        active: The objects the episode this was built for runs, or ``None``
            for every one. What :meth:`check_start` masks the document's
            ``:init`` by, so a parked object's start constraint -- stated for
            the pose it would have had -- is not checked against its park.
    """

    goal: Node
    reward: RewardClause = field(default_factory=Fraction)
    observe: tuple[Node, ...] = ()
    doc: Any = None
    leaf: Callable[[Any], Node] | None = None
    active: frozenset[str] | None = None

    def reset(self, ctx: WorldState) -> None:
        self.goal.reset(ctx)
        for node in (*_clause_nodes(self.reward), *self.observe):
            node.reset(ctx)

    def check_start(self, ctx: WorldState):
        """Whether the settled world is where the document said it would start.

        Two questions from one read: did every ``:init`` constraint survive the
        settle, and is the goal *already* satisfied. Both matter at the same
        moment and an author fixing one usually needs the other, which is why
        :class:`~robosimstudio_bench.goals.initial.InitReport` answers both.

        Called after settling, never before -- the arrangement has not happened
        yet at build time and every constraint would fail for a reason that is
        nobody's fault.

        Returns:
            An ``InitReport``, or ``None`` when this evaluator was built without
            a document or a leaf maker and there is nothing to check against.
            ``None`` is "no answer", distinct from a report saying the init holds.
        """
        if self.doc is None or self.leaf is None:
            return None
        from robosimstudio.assetlib.semantics.snapshot import Snapshot

        from .initial import check_init

        return check_init(self.doc, self.leaf, Snapshot(ctx=ctx), active=self.active)

    def __call__(self, ctx: WorldState) -> EvalResult:
        verdict = self.goal(ctx)
        metrics: dict[str, Any] = dict(verdict.metrics)

        for node in self.observe:
            for key, value in node(ctx).metrics.items():
                metrics.setdefault(key, value)

        # The clause evaluates its own measures and hands back what they
        # reported, rather than being asked for a number and then re-run for its
        # metrics. Running a node twice per step is not merely wasteful: `once`
        # and `hold-for` carry state, so a second call advances a clock that only
        # one step passed on, and a hold would come out satisfied in half the
        # time it was asked to hold for.
        reward, measured = self.reward.score(verdict, ctx)
        for key, value in measured.items():
            metrics.setdefault(key, value)
        return EvalResult(success=bool(verdict.success), reward=reward, metrics=metrics)


def _clause_nodes(clause: RewardClause) -> tuple[Node, ...]:
    """Every node a clause owns, for reset. ``Fraction`` owns none."""
    return clause.nodes() if hasattr(clause, "nodes") else ()
