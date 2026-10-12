# SPDX-License-Identifier: Apache-2.0
"""``:init`` -- what must be true when the episode starts, and checking it is.

Standard BDDL's init is a list of facts about the starting state, and every
harness quietly does two different things with it: *arrange* the scene so they
hold, and *assume* they hold thereafter. Splitting those is the whole content of
this module.

**Init is verified, not asserted.** A constraint is checked after the scene
settles, with the same atom that would check it as a goal -- because settling is
physics and physics does not always agree. A hammer placed inside ``bin1_region``
can bounce out of it during the settle, and a task that assumed its init held
would run an episode whose starting state it has misdescribed, then attribute the
result to the policy. Verified, that is a refusal before the first step.

**A goal already true at init measures nothing**, and this is the single most
useful lint for an agent-authored task. It is also the one that is easiest to get
subtly wrong: ``hold_for(5s, X)`` is *false* on the first step even when ``X``
holds, so a naive check calls the task non-trivial and the trap stays open. The
temporal wrappers are stripped before asking, so the question is "would this be
satisfied", not "is it satisfied yet".

**What is not here.** Actually placing objects -- sampling poses, pinning a cable
to a gripper, laying a shirt flat -- belongs to the compiler, which owns the
scene lifecycle. This says what must be true and can say whether it is; it does
not move anything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio_bench.taskdoc.ast import Condition, TaskDoc

from .combinators import Node
from .goal import build_goal, mask_condition, mask_goal, maskable_objects

__all__ = ["TEMPORAL", "InitReport", "build_init", "check_init", "starts_solved", "strip_temporal"]

TEMPORAL = ("once", "hold-for", "after")
"""Modals whose verdict depends on time having passed.

Stripped before asking whether a goal is already satisfied. ``peak`` and
``integral`` are absent deliberately: both report a quantity accumulated so far,
and on the first step that quantity is the current one -- so they answer the
question honestly without help, and stripping them would change what is asked.
"""


@dataclass(frozen=True)
class InitReport:
    """Whether the world started where the task said it would.

    Attributes:
        holds: Whether every init constraint is satisfied.
        failed: The constraints that are not, by their clause label.
        details: Per clause, what the measure actually reported -- so a failure
            says "containment 0.0, wanted >= 1.0" rather than "init failed".
        trivial: Whether the *goal* is already satisfied at the start. Reported
            alongside because both are answered from the same settled snapshot,
            and because an author fixing one usually needs to know the other.
    """

    holds: bool = True
    failed: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    trivial: bool = False

    def __bool__(self) -> bool:
        return self.holds and not self.trivial

    def summary(self) -> str:
        """One line, phrased for whoever has to act on it."""
        if self.trivial and not self.holds:
            return f"init does not hold ({', '.join(self.failed)}) and the goal is already satisfied"
        if self.trivial:
            return "the goal is already satisfied at the start, so this task measures nothing"
        if not self.holds:
            return f"init does not hold after settling: {', '.join(self.failed)}"
        return "init holds and the goal is not yet satisfied"


def strip_temporal(condition: Condition) -> Condition:
    """The same clause with its time-dependent wrappers removed.

    ``(hold-for 5.0 (and A B))`` becomes ``(and A B)``; ``(once X)`` becomes
    ``X``; ``(after E X)`` becomes ``(and E X)``. Applied recursively, so a
    conjunction of latched clauses is stripped throughout.

    This exists for exactly one question -- "would the goal be satisfied by this
    starting state" -- and asking it without stripping gets the answer wrong in
    the dangerous direction. A ``hold-for`` reports false on the first step
    however true its body is, so an unstripped check declares every latched task
    non-trivial and lets the one broken task through.

    ``after`` keeps its event, and dropping it was a measured false alarm: the
    event is *content* -- the sequence "press, then released" starts solved only
    if the start is both pressed and released, which no state is -- where a
    plain strip reduced it to "released?", true of every scene at rest, and
    called the sponge press trivial on the strength of it.
    """
    if condition.modal == "after" and condition.children and isinstance(condition.modal_arg, Condition):
        return Condition(
            head="and",
            children=(strip_temporal(condition.modal_arg), strip_temporal(condition.children[0])),
        )
    if condition.modal in TEMPORAL and condition.children:
        return strip_temporal(condition.children[0])
    if condition.children:
        return Condition(
            head=condition.head,
            args=condition.args,
            op=condition.op,
            value=condition.value,
            modal=condition.modal,
            modal_arg=condition.modal_arg,
            children=tuple(strip_temporal(child) for child in condition.children),
        )
    return condition


def build_init(
    doc: TaskDoc, make_leaf: Callable[[Condition], Node], *, active: Iterable[str] | None = None
) -> tuple[Node, ...]:
    """One node per init constraint, built in the **check** direction.

    Checking, not sampling. An init clause is a constraint to be verified after
    the scene settles, using the atom that would check it as a goal. Under an
    ``active`` set the constraints about parked objects are dropped -- they
    describe a start the episode does not have.
    """
    clauses = doc.init if active is None else _masked(doc.init, active, doc)
    return tuple(build_goal(clause, make_leaf, path=f"init.{i}") for i, clause in enumerate(clauses))


def starts_solved(
    doc: TaskDoc, make_leaf: Callable[[Condition], Node], snapshot: Snapshot, *, active: Iterable[str] | None = None
) -> bool:
    """Whether the goal is already satisfied by the starting state.

    Temporal wrappers are stripped first; see :func:`strip_temporal` for why that
    is the difference between a lint that works and one that always passes.
    """
    goal = mask_goal(doc, active)
    if goal is None:
        return False
    node = build_goal(strip_temporal(goal), make_leaf, path="goal")
    node.reset(snapshot.ctx)
    return bool(node(snapshot.ctx).success)


def _masked(clauses: tuple[Condition, ...], active: Iterable[str], doc: TaskDoc) -> tuple[Condition, ...]:
    objects = maskable_objects(doc)
    keep = frozenset(str(n) for n in active)
    return tuple(c for c in (mask_condition(c, keep, objects) for c in clauses) if c is not None)


def check_init(
    doc: TaskDoc,
    make_leaf: Callable[[Condition], Node],
    snapshot: Snapshot,
    *,
    active: Iterable[str] | None = None,
) -> InitReport:
    """Verify a settled scene against what the task said its start would be.

    Args:
        doc: The task.
        make_leaf: Builds the measure for one clause -- the same maker the goal
            uses, so an init constraint and a goal clause mean the same thing.
        snapshot: The world **after settling**. Before settling is the wrong
            moment: the arrangement has not happened yet, and every constraint
            would fail for a reason that is nobody's fault.
        active: The objects the episode runs, or ``None`` for every one. A
            constraint about a parked object is not checked, and the goal is
            asked whether it starts solved with those clauses masked, exactly
            as the evaluator scores it.

    Returns:
        A report, rather than a raise. An authoring loop wants the whole list --
        an agent fixing one violated constraint per round trip is the experience
        this exists to avoid.
    """
    failed: list[str] = []
    details: dict[str, Any] = {}

    for node in build_init(doc, make_leaf, active=active):
        node.reset(snapshot.ctx)
        result = node(snapshot.ctx)
        label = node.label or "init"
        details[label] = {k: v for k, v in result.metrics.items() if k != label}
        if not result.success:
            failed.append(label)

    return InitReport(
        holds=not failed,
        failed=tuple(failed),
        details=details,
        trivial=starts_solved(doc, make_leaf, snapshot, active=active),
    )
