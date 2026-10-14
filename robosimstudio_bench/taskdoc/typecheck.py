# SPDX-License-Identifier: Apache-2.0
"""Checking a parsed task against what the world actually contains.

Parsing asks "is this file well formed". This asks the second question: **do the
things it names exist, and do they afford what it asks of them.** They are
separate on purpose -- a schema error and a missing asset are different failures
that a reader, and an authoring agent, act on differently.

**The catalog is injected, not imported.** The asset library sits *above* the
toolkit, so this module cannot reach for it; it takes anything satisfying
:class:`Catalog` instead. That keeps the layer law intact and has a second
benefit worth more than the first: the checker is a pure function of a document
and a vocabulary, so its tests need no assets registered, no scene and no GPU.

**Problems are typed, and one type is special.** A :attr:`CATALOG_GAP` means the
world does not contain something the task needs -- no hook asset, no ``loop``
feature on the object somebody wants to hang. That is not a mistake in the file.
It is the single most useful thing an authoring loop can be told, because it
converts "the task failed" into "the catalog needs a hook", which is an action.
Everything else is a mistake in the file, and says so.

Structural typing throughout. An atom does not require an object to *be* a kind
of thing; it requires the object to *have* a feature of some kind. Anything
carrying a ``loop`` satisfies ``threaded``'s first operand -- scissors, a mug, a
keyring -- with nothing declaring an interface or inheriting anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .ast import Condition, Ref, TaskDoc

__all__ = [
    "ARITY",
    "CATALOG_GAP",
    "DIRECTION",
    "MALFORMED",
    "UNRESOLVED",
    "Catalog",
    "CheckProblem",
    "check",
]

CATALOG_GAP = "catalog-gap"
"""The world lacks something the task needs. Not a mistake in the file."""
UNRESOLVED = "unresolved"
"""The task names something the catalog has never heard of."""
ARITY = "arity"
"""An atom was given the wrong number of operands."""
DIRECTION = "direction"
"""An atom used in a direction it does not support."""
MALFORMED = "malformed"
"""Structurally wrong in a way parsing could not have caught."""


@runtime_checkable
class Catalog(Protocol):
    """What the checker needs to know about the world.

    Deliberately four methods. A protocol that grew to mirror the asset library
    would couple the two, and the point of injecting it is that they stay apart.
    """

    def has_asset(self, name: str) -> bool: ...

    def features_of(self, asset: str) -> dict[str, str]:
        """Feature name to kind, for one asset."""
        ...

    def atom(self, name: str) -> Any | None:
        """The atom's spec, or ``None`` if nothing by that name is catalogued."""
        ...

    def states_of(self, asset: str) -> tuple[str, ...]:
        """Named states this asset declares, usable as bare predicates."""
        ...


@dataclass(frozen=True)
class CheckProblem:
    """One thing wrong, phrased for whoever has to act on it."""

    kind: str
    detail: str
    where: str = ""

    def __str__(self) -> str:
        return f"[{self.kind}] {self.where + ': ' if self.where else ''}{self.detail}"

    @property
    def is_gap(self) -> bool:
        """Whether this is the world's fault rather than the file's."""
        return self.kind == CATALOG_GAP


def _resolve_ref(ref: Ref, doc: TaskDoc, catalog: Catalog, where: str) -> tuple[list[CheckProblem], str]:
    """Check one reference, and report the feature kind it lands on.

    Returns ``(problems, kind)`` -- ``kind`` empty when the reference names a
    whole object rather than one of its features, which is legitimate and common.
    """
    out: list[CheckProblem] = []

    # A region declared by the problem file is a name in its own right: BDDL
    # states regions per problem, so `bin2_hammer_region` is not a feature of
    # anything and must resolve here before the instance lookup.
    if any(r.name == ref.base for r in doc.regions) and not ref.path:
        return out, "region"

    try:
        instance = doc.instance(ref.base)
    except KeyError:
        out.append(CheckProblem(MALFORMED, f"{ref} names {ref.base!r}, which the task never declares", where))
        return out, ""

    asset = instance.type
    if not asset:
        # An untyped object is legal BDDL and simply cannot be checked further.
        # Silence here rather than a warning: the corpus is full of them, and a
        # checker that cries about the normal case gets muted.
        return out, ""

    if not catalog.has_asset(asset):
        out.append(
            CheckProblem(
                CATALOG_GAP,
                f"{ref} needs asset {asset!r}, which the catalog does not have. "
                "Add it, or point the task at something that exists.",
                where,
            )
        )
        return out, ""

    if not ref.path:
        return out, ""

    features = catalog.features_of(asset)
    # Only the FIRST path element is checked against the asset's own features.
    # Deeper paths walk into parts (a drawer's interior), which needs the part's
    # own declaration -- not modelled yet, and claiming to check it would be
    # worse than saying nothing.
    head = ref.path[0]
    if head not in features:
        out.append(
            CheckProblem(
                CATALOG_GAP,
                f"{ref}: asset {asset!r} has no feature {head!r}. It offers {sorted(features)}.",
                where,
            )
        )
        return out, ""
    return out, features[head]


def _check_condition(
    condition: Condition, doc: TaskDoc, catalog: Catalog, where: str, *, direction: str
) -> list[CheckProblem]:
    """Check one clause and everything under it."""
    out: list[CheckProblem] = []

    if condition.children:
        argument = condition.modal_arg
        if isinstance(argument, Condition):
            out += _check_condition(argument, doc, catalog, f"{where}.{condition.modal}", direction=direction)
        for index, child in enumerate(condition.children):
            label = f"{where}.{condition.modal or condition.head}[{index}]"
            out += _check_condition(child, doc, catalog, label, direction=direction)
        if condition.modal == "hold-for" and not isinstance(condition.modal_arg, (int, float)):
            out.append(CheckProblem(MALFORMED, f"(hold-for ...) needs a duration in seconds, got {argument!r}", where))
        return out

    kinds: list[str] = []
    for ref in condition.args:
        problems, kind = _resolve_ref(ref, doc, catalog, where)
        out += problems
        kinds.append(kind)

    # `atom:key` selects one of a multi-valued measure's returns, and the
    # SELECTOR is not part of the atom's name -- `bind` has always split it here
    # (`condition.head.partition(":")`) and the checker had not.
    head, _, selector = condition.head.partition(":")
    atom = catalog.atom(head)
    if atom is None:
        # The head may be a word the ASSET defines rather than an atom. Two
        # shapes, and both must be allowed here:
        #
        #   (adhered tape)                  a semantic state -- boolean
        #   (<= (bond-fraction tape) 0.05)  a published state -- a quantity
        #
        # A published state exists precisely to be thresholded, so the
        # comparison form is the *common* one.
        if condition.args:
            first = condition.args[0]
            try:
                instance = doc.instance(first.base)
            except KeyError:
                instance = None
            if instance is not None and instance.type and catalog.has_asset(instance.type):
                if condition.head in catalog.states_of(instance.type):
                    return out
        out.append(
            CheckProblem(
                UNRESOLVED,
                f"nothing named {condition.head!r} is catalogued as an atom, and no operand declares it as a state",
                where,
            )
        )
        return out

    # A selector that names nothing the atom returns. Worth checking precisely
    # because splitting it above made the whole construct resolvable: `insertion`
    # reports depth, lat and mate_gap, and `insertion:lateral` would otherwise
    # bind clean and read a metric nobody emits.
    # An atom that declares no returns refuses every selector, as bind does; a
    # catalog object with no ``returns`` attribute at all is not checked.
    returns = tuple(getattr(atom, "returns", ()) or ())
    if selector and hasattr(atom, "returns") and selector not in returns:
        out.append(
            CheckProblem(
                UNRESOLVED,
                f"atom {head!r} reports {list(returns)}, and {selector!r} is not among them",
                where,
            )
        )

    if not atom.supports(direction):
        out.append(
            CheckProblem(
                DIRECTION,
                f"atom {condition.head!r} does not support {direction!r}; it supports {list(atom.directions)}",
                where,
            )
        )

    required = [a for a in atom.args if not a.optional]
    if len(condition.args) < len(required):
        out.append(
            CheckProblem(
                ARITY,
                f"atom {condition.head!r} needs {len(required)} operand(s), got {len(condition.args)}",
                where,
            )
        )
        return out

    # Structural typing: the operand must HAVE a feature of an accepted kind.
    # Nothing declares an interface; a mug and a pair of scissors both satisfy a
    # `loop` operand because both annotate one.
    for spec, ref, kind in zip(atom.args, condition.args, kinds, strict=False):
        if spec.kinds and kind and kind not in spec.kinds:
            out.append(
                CheckProblem(
                    MALFORMED,
                    f"atom {condition.head!r} wants a {list(spec.kinds)} for {spec.name!r}, but {ref} is a {kind!r}",
                    where,
                )
            )
    return out


def check(doc: TaskDoc, catalog: Catalog) -> list[CheckProblem]:
    """Every problem with a task, given a catalog. Empty means it typechecks.

    Returns problems rather than raising, and returns *all* of them: an author
    fixing one error at a time through six round trips is the experience this
    exists to avoid, and an agent revising a document wants the whole list.
    """
    out: list[CheckProblem] = []

    for region in doc.regions:
        if region.target and not any(o.name == region.target for o in doc.objects):
            out.append(
                CheckProblem(
                    MALFORMED,
                    f"region {region.name!r} hangs off {region.target!r}, which the task never declares",
                    "regions",
                )
            )

    for index, condition in enumerate(doc.init):
        # Checked in the CHECK direction, not "sample". An init clause is a
        # constraint that gets *verified* after the scene settles, using the same
        # atom that would check it as a goal -- so it needs that atom to support
        # checking, not sampling. Only an explicit layout family (`pose ~ grid`)
        # samples, and that is a different construct.
        out += _check_condition(condition, doc, catalog, f"init[{index}]", direction="check")

    if doc.goal is None:
        out.append(CheckProblem(MALFORMED, "the task states no goal, so nothing can score it", "goal"))
    else:
        out += _check_condition(doc.goal, doc, catalog, "goal", direction="check")

    out += _check_reward(doc, catalog)

    for index, condition in enumerate(doc.observe):
        out += _check_condition(condition, doc, catalog, f"observe[{index}]", direction="observe")

    return out


_REWARD_CLAUSES = ("progress", "shrinking")
"""The reward forms that name a measure. ``fraction`` names none -- it reads the
goal tree it is scored beside -- so there is nothing here to resolve."""

_MASK_WORDS = ("gated-by", "gated_by")
_GATE_WORDS = ("gate", "gate-at")
"""The spellings of a progress clause's gate and mask, written once here and
imported by ``goals/reward.py`` (which builds them) and ``goals/gauntlet.py``
(which reads past them to the measure), so a new spelling reaches all three."""


def _check_reward(doc: TaskDoc, catalog: Catalog) -> list[CheckProblem]:
    """Check the measure a ``:reward`` clause names, and its mask.

    The checker used to visit ``:init``, ``:goal`` and ``:observe`` and stop
    there, which let a reward name a measure nothing implements: it passed lint
    and raised at build, one layer past the tool whose entire job is to say what
    is wrong with the file. An authoring agent got a stack trace where it should
    have got a catalog gap.

    Two directions, because a reward clause reads its two operands differently.
    The measure is read for a **value** -- ``observe``, the same as ``:observe``.
    The mask is a clause that has to be true or false -- ``check``, the same as
    the goal. Collapsing them would report a perfectly good progress measure as a
    direction error for not affording a verdict.

    The section arrives as the ``_plain`` shape rather than as conditions, so the
    forms are parsed here. A form that will not parse is left to
    :func:`~robosimstudio_bench.goals.reward.build_reward`, which refuses
    the whole closed clause set with a better message than a checker walking a
    dict could give.
    """
    from .parse import parse_condition

    out: list[CheckProblem] = []
    for word in _REWARD_CLAUSES:
        body = doc.reward.get(word)
        if body is None:
            continue
        parts = body if isinstance(body, list) else [body]
        seen_measure = False
        for part in parts:
            keyword = str(part[0]).lstrip(":") if isinstance(part, list) and part else ""
            if keyword in _GATE_WORDS and len(part) == 2 and not _positive(part[1]):
                out.append(
                    CheckProblem(MALFORMED, f"(:{keyword} V) must be a number > 0, got {part[1]!r}", f"reward.{word}")
                )
            elif keyword in _MASK_WORDS and len(part) == 2 and isinstance(part[1], list):
                out += _visit_reward_form(part[1], doc, catalog, f"reward.{word}.gated-by", "check", parse_condition)
            elif keyword not in _GATE_WORDS + _MASK_WORDS and not seen_measure and isinstance(part, list):
                seen_measure = True
                out += _visit_reward_form(part, doc, catalog, f"reward.{word}", "observe", parse_condition)
    return out


def _positive(value: Any) -> bool:
    """Whether a gate value is a number above zero -- what ``Progress`` divides by."""
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _visit_reward_form(
    form: Any, doc: TaskDoc, catalog: Catalog, where: str, direction: str, parse: Any
) -> list[CheckProblem]:
    """One reward operand, checked if it parses as a clause."""
    try:
        condition = parse(form)
    except (ValueError, TypeError):
        return []
    return _check_condition(condition, doc, catalog, where, direction=direction)
