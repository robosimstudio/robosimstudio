# SPDX-License-Identifier: Apache-2.0
"""Binding an upstream dialect's predicates to this stack's measures.

An imported BDDL problem says `(In hammer_0 bin2_hammer_region)`. This stack has
no atom called ``In``; it has ``containment``, which returns a fraction. Some
table has to say which is which, and the interesting part is that the table is
**provenance, not plumbing**.

Here is why. BDDL's ``In`` is a box test: it asks whether an object's position
falls inside a rectangle and a height window, and **a carried object satisfies
it** -- an episode can score on a tool still between the fingers, fifteen
centimetres up. This stack also has ``resting_on``, which requires a contact
force from the destination with the object at rest, and that is a materially
stronger claim about the same English sentence.

Binding ``In`` to the stronger one would silently make an imported task *harder
than the benchmark it came from*, and every number reported against it would be
incomparable with every published number. So:

**A reproduction-tier import binds the source's semantics.** Comparability with
the benchmark outranks rigor. Promoting a task to native means rebinding *and
re-labelling the tier* -- scored differently, labelled differently. Each binding
below carries the rationale for its choice, because six months on nobody
reconstructs this from a mapping table.

The second thing a binding does is bridge a type difference. Upstream predicates
are boolean; several of our measures are graded. ``In`` becomes
``(>= (containment x r) 1.0)`` -- the comparison is part of the binding, not
something a task author writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .ast import Condition, TaskDoc
from .parse import COMPARISONS, CONNECTIVES, MODALS
from .typecheck import _GATE_WORDS, _MASK_WORDS, _REWARD_CLAUSES

__all__ = ["VAB_INDUSTRIAL_V1", "BindingTable", "PredicateBinding", "apply_bindings"]


@dataclass(frozen=True)
class PredicateBinding:
    """One source predicate, and what it means here.

    Attributes:
        atom: The measure it binds to.
        op: Comparison used to turn a boolean predicate into a graded one.
            Empty leaves it a bare predicate.
        value: What the measure is compared against.
        rationale: **Why this binding and not a stronger or weaker one.**
            Required in spirit: a mapping without it is a decision nobody can
            audit, and the decisions here change what published numbers mean.
        stronger_alternative: A measure that would have been a defensible choice
            and was deliberately *not* taken. Recorded so a later reader can see
            the fork rather than assume there was none.
    """

    atom: str
    op: str = ""
    value: float = 0.0
    rationale: str = ""
    stronger_alternative: str = ""


@dataclass(frozen=True)
class BindingTable:
    """Every binding for one upstream dialect, at one provenance tier."""

    name: str
    tier: str
    bindings: dict[str, PredicateBinding] = field(default_factory=dict)
    notes: str = ""

    def get(self, predicate: str) -> PredicateBinding | None:
        """Look one up, case-insensitively -- the dialect writes ``In`` and ``on``."""
        lowered = predicate.lower()
        for source, binding in self.bindings.items():
            if source.lower() == lowered:
                return binding
        return None


VAB_INDUSTRIAL_V1 = BindingTable(
    name="vab_industrial@v1",
    tier="reproduction",
    notes=(
        "For the VAB-industrial tool-sort problems, whose region literals this repo already "
        "transcribes to the digit. Bound to reproduce that benchmark's scoring, not to improve on it."
    ),
    bindings={
        "In": PredicateBinding(
            atom="containment",
            op=">=",
            value=1.0,
            rationale=(
                "BDDL's In is a box test that a CARRIED object satisfies -- an episode can score on "
                "a tool still between the fingers 15 cm above the bin. Bound to the box test on "
                "purpose: this is a reproduction of that benchmark, and a task scored more strictly "
                "than the numbers it is compared against is not a reproduction of anything."
            ),
            stronger_alternative="resting_on",
        ),
        "On": PredicateBinding(
            atom="containment",
            op=">=",
            value=1.0,
            rationale=(
                "Same box test as In in this dialect's usage -- the tool-sort problems state "
                "surfaces as regions rather than as a distinct support relation."
            ),
            stronger_alternative="resting_on",
        ),
    },
)


def _bind_condition(condition: Condition, table: BindingTable) -> Condition:
    """Rewrite one clause through a table, leaving what it does not cover."""
    if condition.children:
        argument = condition.modal_arg
        if isinstance(argument, Condition):
            argument = _bind_condition(argument, table)
        return replace(
            condition,
            modal_arg=argument,
            children=tuple(_bind_condition(child, table) for child in condition.children),
        )

    binding = table.get(condition.head)
    if binding is None:
        # Unbound predicates are left ALONE, not dropped and not guessed at. The
        # typechecker will report them as unresolved, which names the gap; a
        # binding step that quietly deleted what it did not understand would
        # produce a task that typechecks and means less than it says.
        return condition

    if binding.op:
        return replace(condition, head=binding.atom, op=binding.op, value=binding.value)
    return replace(condition, head=binding.atom)


def _bind_form(form: Any, table: BindingTable) -> Any:
    """:func:`_bind_condition` on the plain list shape a ``:reward`` arrives in.

    The reward section is ``_plain``, not conditions, so its forms are rewritten
    as written: the same leaves bound the same way, into a form
    :func:`~.parse.parse_condition` reads back as what :func:`_bind_condition`
    would have produced.
    """
    if not isinstance(form, list) or not form:
        return form
    head = str(form[0])
    folded = head.lower()
    if folded.replace("-", "_") in CONNECTIVES:
        return [form[0], *(_bind_form(child, table) for child in form[1:])]
    if folded in MODALS:
        # `hold-for` takes a duration before its body, and that is not a clause.
        keep = 2 if folded == "hold-for" else 1
        return [*form[:keep], *(_bind_form(child, table) for child in form[keep:])]
    if head in COMPARISONS and len(form) == 3 and isinstance(form[1], list) and form[1]:
        measure = form[1]
        if str(measure[0]).lower() in MODALS:
            return [head, _bind_form(measure, table), form[2]]
        binding = table.get(str(measure[0]))
        if binding is None:
            return form
        if binding.op:
            # The binding's comparison replaces the author's, as it does on a
            # parsed leaf that already carried one.
            return [binding.op, [binding.atom, *measure[1:]], binding.value]
        return [head, [binding.atom, *measure[1:]], form[2]]
    binding = table.get(head)
    if binding is None:
        return form
    if binding.op:
        return [binding.op, [binding.atom, *form[1:]], binding.value]
    return [binding.atom, *form[1:]]


def _bind_reward(reward: dict[str, Any], table: BindingTable) -> dict[str, Any]:
    """The measure and the ``(:gated-by CLAUSE)`` of each progress clause, bound.

    The gate is a number and ``fraction`` names no measure, so both pass through.
    """
    out = dict(reward)
    for word in _REWARD_CLAUSES:
        body = out.get(word)
        if not isinstance(body, list):
            continue
        parts = []
        seen_measure = False
        for part in body:
            keyword = str(part[0]).lstrip(":") if isinstance(part, list) and part else ""
            if keyword in _MASK_WORDS and len(part) == 2:
                parts.append([part[0], _bind_form(part[1], table)])
            elif keyword not in _GATE_WORDS + _MASK_WORDS and not seen_measure and isinstance(part, list):
                seen_measure = True
                parts.append(_bind_form(part, table))
            else:
                parts.append(part)
        out[word] = parts
    return out


def apply_bindings(doc: TaskDoc, table: BindingTable) -> TaskDoc:
    """Rewrite a parsed document's predicates through a binding table.

    Returns a new document; the original is untouched, so an importer can show
    both and a reviewer can see exactly what the binding changed.

    The table's identity and tier are recorded in the result's provenance,
    because a task's numbers only mean something alongside the semantics they
    were scored under.
    """
    provenance = dict(doc.provenance)
    provenance["bound_by"] = table.name
    provenance.setdefault("tier", table.tier)

    return replace(
        doc,
        init=tuple(_bind_condition(c, table) for c in doc.init),
        goal=_bind_condition(doc.goal, table) if doc.goal is not None else None,
        observe=tuple(_bind_condition(c, table) for c in doc.observe),
        reward=_bind_reward(doc.reward, table),
        provenance=provenance,
    )
