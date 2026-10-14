# SPDX-License-Identifier: Apache-2.0
"""Asking the scene a question, in the vocabulary the goal is written in.

A goal is judgment: a clause, a threshold, a temporal operator, and a verdict at
the end of an episode. An **agent** wants the same vocabulary pointed differently
-- *is the hammer on the bench right now* -- with no threshold and no episode.
That is this module.

**One measurement path, two callers.** The binder and this both resolve operands
the same way, read them in the form the atom asked for, and dispatch to the same
handler. A second implementation would drift, and the drift would be the worst
kind: an agent reasoning about a world slightly different from the one being
graded, with nothing to show which was wrong. So a query is a bound clause minus
the binding.

**No threshold, and that is a design statement rather than an omission.** What
counts as *enough* containment is the task's, and an agent asking "how much" and
getting back a boolean somebody else chose has been told less than it asked. The
raw returns come back unchanged. Rendering them belongs to the consumer, not
the simulator.

**No smoothing either**, and this one is a caveat rather than a choice. Smoothing
is a property of a *bound clause* -- a window over control steps -- and a single
question has no window. For a contact reading that means an answer can catch the
solver's contact set mid-churn, so the instantaneous values are labelled as such.
An agent that needs the settled answer asks twice, or reads the goal's own
metrics.

**Asking a whole family** is the other half of what makes this usable. An agent
holding two object names rarely knows which relation it wants; it wants to know
what is true. ``query(snapshot, "SpatialRelation", "hammer", "bench")`` evaluates
every spatial relation that admits the pair and returns what each says.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.atom import leaves_of
from robosimstudio.assetlib.semantics.atoms import get_atom
from robosimstudio.assetlib.semantics.handlers import dispatch
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.scene.entity import entity_layouts
from robosimstudio_bench.taskdoc.ast import Ref, TaskDoc
from robosimstudio_bench.taskdoc.parse import parse_ref

from .bind import BindError, Operand, read, resolve_operand

__all__ = ["QueryRefused", "applicable", "query"]


class QueryRefused(LookupError):
    """This question cannot be asked of these things.

    A distinct type because the two failures behind it are worth telling apart
    by a caller that is deciding what to try next: an atom nobody implements for
    this pair of substrates is a *catalog* answer -- the thing does not afford
    the question -- while a reference naming nothing is a mistake in the asking.
    Both name what *is* available, because a refusal that lists the alternatives
    is the most useful thing an authoring loop can be told.
    """


def _default_catalog(name: str) -> AssetRecord | None:
    from robosimstudio.assetlib.registry import ASSETS

    try:
        return ASSETS.get(name)
    except KeyError:
        return None


def _layouts(snapshot: Any, operands: Sequence[Operand]) -> tuple[tuple[str, ...], ...]:
    """Per operand, what its live entity's capabilities add to its layout chain."""
    entities = getattr(snapshot, "entities", None) or {}
    return tuple(entity_layouts(entities.get(o.ref.base)) for o in operands)


def _operand(reference: str, doc: TaskDoc | None, catalog: Callable[[str], AssetRecord | None]) -> Operand:
    """One reference, resolved as far as what is known allows.

    With a document, exactly as a goal clause resolves it -- so a query about
    ``port.bore`` reaches the same feature the goal does. Without one, the
    reference degrades to an untyped operand, which is enough for readers that
    take a handle by name.
    """
    ref: Ref = parse_ref(reference)
    if doc is None:
        return Operand(ref=ref)
    return resolve_operand(ref, doc, catalog)


def query(
    snapshot: Snapshot | Any,
    atom: str,
    *references: str,
    doc: TaskDoc | None = None,
    catalog: Callable[[str], AssetRecord | None] | None = None,
    task: str = "",
) -> dict[str, Any]:
    """Ask one atom -- or one family -- about these references, right now.

    Args:
        snapshot: A frozen read of the scene, or an ``WorldState`` to wrap in
            one. Read-only, so asking cannot perturb what is being asked about.
        atom: A registered atom name, or a family name such as
            ``"SpatialRelation"`` to ask every relation that admits the operands.
            Spelled as a document spells it: ``contact-force`` is accepted, and
            ``insertion:depth`` narrows the answer to that one return.
        *references: ``"hammer"``, ``"port.bore"``, ``"cable.point(0.63)"`` --
            the same reference grammar a document is written in.
        doc: The task document, when there is one. Without it, references
            resolve untyped, which is enough for whole-object readers.
        catalog: How to look up an asset record. Defaults to the registry.
        task: The task scope to dispatch under; the document's name when not
            given, as :func:`~.evaluate.make_evaluator` binds the goal.

    Returns:
        For an atom, the handler's own structured returns. For a family,
        ``{"family": name, "readings": {atom: {...}}}`` holding one entry per
        applicable atom.

    Raises:
        QueryRefused: If nothing implements the atom for these operands, or the
            name is neither an atom nor a family.
    """
    frozen = snapshot if isinstance(snapshot, Snapshot) else Snapshot(ctx=snapshot)
    lookup = catalog or _default_catalog
    # The document's own spelling, normalised the way :func:`bind` does it:
    # `contact-force` is the registered `contact_force`, and `insertion:depth`
    # selects one of the measure's returns. Refusing either would make an agent
    # ask in a different vocabulary from the one its goal is written in.
    measure, _, selector = atom.replace("-", "_").partition(":")

    if not selector and leaves_of(measure):
        readings: dict[str, Any] = {}
        for leaf in leaves_of(measure):
            if len(leaf.args) != len(references):
                continue
            try:
                readings[leaf.name] = query(frozen, leaf.name, *references, doc=doc, catalog=lookup, task=task)
            except (LookupError, BindError, ValueError):  # QueryRefused and KeyError are LookupErrors
                # An atom this pair does not afford is not an error at family
                # level -- it is the answer to "what can I ask", arrived at by
                # asking. ``BindError`` is in the list for the same reason: a
                # reader that cannot read an operand in the form an atom wants
                # is that atom declining the pair, one step later. Individually
                # every one of these still refuses, loudly.
                continue
        return {"family": atom, "readings": readings}

    try:
        spec = get_atom(measure)
    except LookupError as exc:
        raise QueryRefused(str(exc)) from None
    if selector and selector not in spec.returns:
        raise QueryRefused(
            f"atom {atom!r}: {measure!r} reports {list(spec.returns)}, and {selector!r} is not among them"
        )

    operands = [_operand(reference, doc, lookup) for reference in references]
    forms = [arg.reads or "position" for arg in spec.args] + ["position"] * len(operands)
    kinds = tuple(o.kind for o in operands)
    substrates = tuple(o.substrate for o in operands)
    # The task and asset scopes as :func:`bind` passes them, so a task- or
    # asset-scoped override answers a query the same way it scores the goal.
    owner = next((o.asset for o in operands if o.asset), "")
    scope = task or (doc.name if doc is not None else "")
    try:
        handler = dispatch(measure, kinds, substrates, layouts=_layouts(frozen, operands), asset=owner, task=scope)
    except LookupError as exc:
        # ``LookupError``, not ``KeyError``: dispatch refuses an unimplemented
        # pairing with a plain ``LookupError`` (``semantics/handlers.py``), which
        # the narrower catch let escape past the ``QueryRefused`` promised above.
        raise QueryRefused(
            f"{atom!r} is not implemented for {list(zip(substrates, kinds, strict=False))}: {exc}"
        ) from None

    values = [read(frozen, operand, form) for operand, form in zip(operands, forms, strict=False)]
    reported = handler.fn(frozen, *values) if spec.reads_scene else handler.fn(*values)
    # A scalar return is keyed by the primary, as :func:`bind` reports it.
    returns = dict(reported) if isinstance(reported, Mapping) else {spec.primary or measure: reported}
    if not selector:
        return returns
    if selector not in returns:
        raise QueryRefused(f"atom {atom!r}: the handler reported {sorted(returns)}, not {selector!r}")
    return {selector: returns[selector]}


def applicable(family: str, *operands: Operand) -> list[str]:
    """Which atoms in a family admit these operands, by name.

    What ``sim.list_relations`` answers with. Cheaper than asking each and
    catching the refusals, and it is the honest form of "what can I ask about
    these two things".
    """
    kinds = tuple(o.kind for o in operands)
    substrates = tuple(o.substrate for o in operands)
    out: list[str] = []
    for leaf in leaves_of(family):
        if len(leaf.args) != len(operands):
            continue
        try:
            dispatch(leaf.name, kinds, substrates)
        except LookupError:
            continue
        out.append(leaf.name)
    return out
