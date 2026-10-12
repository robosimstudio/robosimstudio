# SPDX-License-Identifier: Apache-2.0
"""One asset's interface, parsed on its own terms.

:mod:`~robosimstudio.assetlib.semantics` states the rule this module exists to
serve: objects carry features and handlers freely, **in the Python sense of
"interface" -- duck-typed, per object, discovered rather than declared.** There is
no ``Graspable``, no roster, nothing to implement. What an
:class:`~robosimstudio.assetlib.record.AssetRecord` declares *is* the interface.

The consequence is the whole reason for this file. **No two assets need have
anything in common beyond the substrate.** The towel offers three particle sets
and a fold template; the receptacle offers a bore with an axis, a mouth and a
depth; the tape offers a bond field and ten published quantities; the cabinet
offers a joint, two regions and a handle. None of those is a specialisation of
another and none can be rendered by the same code path -- so a reader that wants
to *show* an asset cannot walk a fixed shape. It has to take each declaration and
parse it against the kinds that declaration itself names.

**Three tiers, three parses**, which is where most of the work is:

``literal``
    Constants. Paired field by field against the kind's declared payload, so each
    number arrives with the meaning its kind gives it, marked as a point, a
    direction or neither. Missing required fields are reported rather than
    guessed.
``recipe``
    An expression in the closed selector language, canonicalised through the
    selector's own :func:`~..semantics.selectors.selector_to_dict` -- so axis
    names read as axis names and a derived constant does not print as binary
    noise.
``resolver``
    A versioned reference to registered code. There is nothing to show but the
    key, and showing the key is the honest answer.

**An unregistered kind is legal.** :func:`~..semantics.kinds.is_registered_kind`
returning ``False`` means a bespoke per-object feature -- exactly the freedom the
duck-typed reading grants -- and it is reported as uncontracted rather than as an
error, because the only thing missing is a statement of what its handlers owe.

Nothing here validates; :meth:`~..record.AssetRecord.problems` is where invalid
lives. This reports what an asset says about itself in the form something can
render, and reports the gaps as gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .record import AssetRecord
from .semantics.features import FeatureSpec
from .semantics.kinds import get_kind, is_registered_kind
from .semantics.selectors import round_floats, selector_to_dict

__all__ = ["Facet", "Interface", "Member", "facet", "interface"]

POINT = "point"
"""A payload field holding a position in the body frame: rotate and translate."""

DIRECTION = "direction"
"""A payload field holding a direction in the body frame: rotate only."""


@dataclass(frozen=True, kw_only=True)
class Member:
    """One field of one literal feature, beside what its kind says it means.

    Attributes:
        name: The payload key.
        value: What the asset declared for it.
        meaning: The kind's own prose for this field, empty when the kind does
            not name it.
        required: Whether the kind lists it under ``payload`` rather than
            ``optional``.
        frame: :data:`POINT`, :data:`DIRECTION`, or ``""`` for a scalar or a
            value already in whatever frame it is used in. What a reader needs to
            know before drawing the number anywhere -- translating a direction
            turns a unit vector into a position, which is the mistake
            :class:`~..semantics.kinds.FeatureKind` declares these fields to
            prevent.
        declared: Whether the kind names this field at all. ``False`` is a
            payload key the kind does not define -- refused by
            :meth:`~..semantics.features.FeatureSpec.problems`, and surfaced here
            rather than dropped so a reader sees the same thing the checker does.
    """

    name: str
    value: Any
    meaning: str = ""
    required: bool = False
    frame: str = ""
    declared: bool = True


@dataclass(frozen=True, kw_only=True)
class Facet:
    """One feature of one asset, parsed against the kind it names.

    Attributes:
        name: How the feature is referenced -- the ``bore`` in ``port.bore``.
        kind: Its feature kind.
        tier: One of :data:`~..semantics.features.TIERS`, and which of
            :attr:`members`, :attr:`select` and :attr:`resolver` carries content.
        contracted: Whether the kind is registered and therefore says what its
            handlers owe. ``False`` is a bespoke per-object kind, which is legal.
        summary: The kind's one-line description, empty when uncontracted.
        contracts: The obligations every handler on this kind owes, as prose.
        substrates: Which substrates may carry this kind, empty for any.
        members: The literal payload, field by field. Empty on the other tiers.
        select: The canonical selector form, on a ``recipe``. ``None`` otherwise.
        resolver: The registered code key, on a ``resolver``. ``""`` otherwise.
        missing: Required fields the payload does not supply. Reported rather
            than filled: a ``loop`` with no ``normal`` is not a loop with a
            default normal.
        extra: Payload keys the kind does not define.
        pins_mesh: Whether resolving this depends on the exact mesh layout, so a
            reader knows the annotation rots on a re-mesh unless
            :attr:`mesh_sha` is set.
        mesh_sha: The rest layout it was authored against.
        provenance: Where its numbers came from.
    """

    name: str
    kind: str
    tier: str
    contracted: bool
    summary: str = ""
    contracts: tuple[str, ...] = ()
    substrates: tuple[str, ...] = ()
    members: tuple[Member, ...] = ()
    select: dict[str, Any] | None = None
    resolver: str = ""
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    pins_mesh: bool = False
    mesh_sha: str = ""
    provenance: str = ""

    @property
    def points(self) -> tuple[Member, ...]:
        """Members holding a body-frame position."""
        return tuple(m for m in self.members if m.frame == POINT)

    @property
    def directions(self) -> tuple[Member, ...]:
        """Members holding a body-frame direction."""
        return tuple(m for m in self.members if m.frame == DIRECTION)

    @property
    def drawable(self) -> bool:
        """Whether this carries geometry something could draw in a scene.

        A ``literal`` with at least one point or direction has a place in the
        world. A ``template`` carrying only step counts does not, and neither
        does a recipe until it is resolved against a rest layout -- which needs a
        built scene and is therefore not this layer's answer to give.
        """
        return bool(self.points or self.directions)


@dataclass(frozen=True, kw_only=True)
class Interface:
    """Everything one asset declares about itself, in a form something can show.

    Attributes:
        record: The declaration, carried rather than copied. Its ``states``,
            ``published`` and ``sim`` are already typed and need no parsing, so
            they are read straight off it; duplicating them here would be a
            second copy free to disagree with the first.
        facets: Its features, each parsed against its own kind. This is the part
            that needed parsing, because this is the part where two assets have
            nothing in common.
    """

    record: AssetRecord
    facets: tuple[Facet, ...] = ()

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def substrate(self) -> str:
        return self.record.substrate

    def facet(self, name: str) -> Facet:
        """One parsed feature by name.

        Raises:
            KeyError: Naming what this asset does offer, for the same reason
                :meth:`~..record.AssetRecord.feature` does -- the common outcome
                is that somebody asked for something the object does not have.
        """
        for parsed in self.facets:
            if parsed.name == name:
                return parsed
        raise KeyError(f"asset {self.name!r} has no feature {name!r}; it offers {[f.name for f in self.facets]}")

    def of_kind(self, kind: str) -> tuple[Facet, ...]:
        """Every parsed feature of one kind."""
        return tuple(parsed for parsed in self.facets if parsed.kind == kind)

    @property
    def uncontracted(self) -> tuple[str, ...]:
        """Kinds this asset names that no registered kind describes.

        Legal, and worth showing: it is the difference between a feature whose
        handlers owe something stated and one whose handlers owe nothing, and
        only one of those has a contract test behind it.
        """
        return tuple(dict.fromkeys(parsed.kind for parsed in self.facets if not parsed.contracted))


def interface(record: AssetRecord) -> Interface:
    """Parse one asset's declaration into a renderable view.

    Args:
        record: The declaration. Read only -- no registry lookups beyond the kind
            table, no scene, no model, so this answers for an asset that is in no
            task.

    Returns:
        The view, with features in the asset's own declaration order. Order is
        the author's and is kept: a garment lists the panel a fold is scored on
        before the strips a hand takes, which is the order somebody reads them in.
    """
    return Interface(record=record, facets=tuple(facet(spec) for spec in record.features))


def facet(spec: FeatureSpec) -> Facet:
    """Parse one feature against the kind it names.

    Three tiers and three different answers -- see the module docstring. The tier
    is read from the feature rather than inferred from the payload, because a
    recipe whose expression happens to be a bare constant is still a recipe: the
    tier names the mechanism the compiler must run, not the shape of what is in
    it.
    """
    contracted = is_registered_kind(spec.kind)
    kind = get_kind(spec.kind) if contracted else None

    members: tuple[Member, ...] = ()
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    select: dict[str, Any] | None = None
    resolver = ""

    if spec.tier == "recipe":
        # `select`, or the payload itself -- both spellings are in the corpus and
        # `FeatureSpec.problems` accepts either, so this reads it the same way
        # rather than inventing a third convention.
        select = selector_to_dict(spec.payload.get("select", spec.payload))
    elif spec.tier == "resolver":
        resolver = str(spec.payload.get("resolver", ""))
    else:
        members = _members(spec, kind)
        if kind is not None:
            missing = tuple(kind.missing(spec.payload))
            extra = tuple(kind.unknown(spec.payload))

    return Facet(
        name=spec.name,
        kind=spec.kind,
        tier=spec.tier,
        contracted=contracted,
        summary=kind.summary if kind is not None else "",
        contracts=tuple(kind.contracts) if kind is not None else (),
        substrates=tuple(kind.substrates) if kind is not None else (),
        members=members,
        select=select,
        resolver=resolver,
        missing=missing,
        extra=extra,
        pins_mesh=spec.references_discretisation,
        mesh_sha=spec.mesh_sha,
        provenance=spec.provenance,
    )


def _members(spec: FeatureSpec, kind: Any) -> tuple[Member, ...]:
    """One literal payload, field by field, in the kind's declaration order.

    The kind's order and not the payload's, so two assets carrying a ``cavity``
    read the same way down the page whichever order their authors happened to
    write the keys in. Anything the kind does not name follows, marked
    undeclared -- last, because it is the exceptional case and because the
    checker refuses it.

    Floats go through :func:`~..semantics.selectors.round_floats`, the same
    canonicalisation :func:`~..semantics.features.feature_to_dict` applies and for
    the reason stated there: ``WIDTH / 6`` is exactly ``0.06999999999999999``, and
    a readout that says so is one nobody trusts.
    """
    payload = round_floats(dict(spec.payload))
    if kind is None:
        return tuple(Member(name=key, value=value, declared=False) for key, value in payload.items())

    out: list[Member] = []
    for names, required in ((kind.payload, True), (kind.optional, False)):
        for key, meaning in names.items():
            if key not in payload:
                continue
            out.append(
                Member(
                    name=key,
                    value=payload[key],
                    meaning=meaning,
                    required=required,
                    frame=POINT if key in kind.points else (DIRECTION if key in kind.directions else ""),
                )
            )
    named = set(kind.payload) | set(kind.optional)
    out += [Member(name=key, value=value, declared=False) for key, value in payload.items() if key not in named]
    return tuple(out)
