# SPDX-License-Identifier: Apache-2.0
"""Feature kinds: where recurrence lives.

A **feature** is a typed geometric attribute annotated on one object -- the
scissors' hole, the hanger's rod, the shirt's hem strip, a bin's interior. Every
feature carries a **kind**, and the kind is the whole reason this layer buys
anything.

Consider annotating a pair of scissors so it can hang on a hanger. The natural
thing to write is not ``scissors_hole``; it is ``kind: loop``. And the handler
that decides whether the scissors are hanging is not written against scissors --
it is ``threaded(loop, rod)``, written against the geometry. So a mug handle, a
tape roll, a keyring and a washer all hang on that hanger for free, the moment
somebody annotates their loop. **Object categories never generalise; geometric
feature kinds almost always do.** That asymmetry is why this stack is
prototype-style rather than class-based, and why the open set lives here rather
than in a hierarchy of capability interfaces.

Three rules keep the set honest:

* **A kind is a plain string.** There is no enum, because an enum is a roster and
  the roster is exactly what design rule 16 forbids. An asset may annotate a
  feature of any kind it likes.
* **A registered kind carries a contract**, and its handlers are tested against
  it. Registration is what earns a kind its generality -- not its existence.
* **An unregistered kind is legal but uncontracted.** It resolves, it can be
  referenced, and the typechecker *warns* rather than refusing. This is what
  preserves per-object bespoke features when nothing recurs; the promotion path
  (bespoke used twice -> register the kind, write its contract) mirrors atom
  minting.

A kind exists only if it is probe- or geometry-backed. That is the anti-ontology
rule: no kind is declared because a committee thought objects ought to have one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robosimstudio.registry import Registry

__all__ = [
    "KINDS",
    "FeatureKind",
    "get_kind",
    "is_registered_kind",
    "list_kinds",
    "register_kind",
]


@dataclass(frozen=True, kw_only=True)
class FeatureKind:
    """A kind of feature, and the contract its handlers owe.

    Attributes:
        name: The kind, as it appears in a sidecar (``loop``, ``rod``, ``region``).
        summary: What it is, in one line. Read by the catalog and by an agent
            deciding whether to annotate one.
        payload: Field names a feature of this kind must carry, and what each
            means. Checked at resolve time -- a ``loop`` without a ``normal`` is
            not a loop, and finding that out at annotation time beats finding it
            out when a goal silently never fires.
        optional: Payload fields that may be present.
        contracts: Named obligations every handler dispatching on this kind must
            satisfy, as prose. The test suite generates one check per entry, so a
            contract that is written is a contract that is enforced.
        substrates: Which substrates can carry this kind, empty for any. A
            ``bond_field`` on a rigid body is a mis-annotation worth catching.
        points: Payload fields holding a **position in the body frame**, which a
            reader must rotate *and* translate to reach world coordinates.
        directions: Payload fields holding a **direction in the body frame**,
            which a reader must rotate and must **not** translate.
        provenance: Where the kind came from -- which suite first needed it, or
            which benchmark defines it.

    **Why the frame conversion is declared per kind rather than done per
    handler.** A feature is authored in its body's frame -- a bore's mouth is
    where it is on the port, not where the port happens to be standing -- and
    every measure over it needs world coordinates. Doing that conversion inside
    each handler means every handler that ever dispatches on ``cavity`` has to
    get the same rotation right, and one that forgets to translate produces a
    bore at the world origin: a number, plausible, and wrong by however far the
    part is from the origin.

    Splitting points from directions is the part that is easy to get wrong in the
    other direction. An axis is a direction and translating it turns a unit
    vector into a position, which then normalises to something pointing at the
    body rather than along the bore.
    """

    name: str
    summary: str
    payload: dict[str, str] = field(default_factory=dict)
    optional: dict[str, str] = field(default_factory=dict)
    contracts: tuple[str, ...] = ()
    substrates: tuple[str, ...] = ()
    points: tuple[str, ...] = ()
    directions: tuple[str, ...] = ()
    provenance: str = ""

    def missing(self, payload: dict[str, Any]) -> list[str]:
        """Required payload fields ``payload`` does not supply."""
        return sorted(set(self.payload) - set(payload))

    def unknown(self, payload: dict[str, Any]) -> list[str]:
        """Payload fields this kind does not define.

        Refused rather than ignored, for the reason ``materials._build`` gives
        about unknown material fields: silently dropping a typo'd key produces a
        feature that quietly kept a default, and that gets blamed on the solver.
        """
        return sorted(set(payload) - set(self.payload) - set(self.optional))


KINDS: Registry[FeatureKind] = Registry("feature kind", ("robosimstudio.domains",))
"""Every registered kind.

The general ones register from :mod:`.library`, which the asset library imports.
A kind only one kind of thing can carry -- a position along a chain, by arc
length -- registers from that thing's domain pack, beside the layout it is
stated on; so this scans the domain packages on its first lookup, as the entity
and material registries do, and whether a kind is registered does not depend on
whether anything happened to import its pack first."""


def register_kind(kind: FeatureKind) -> FeatureKind:
    """Register a feature kind under its own name."""
    return KINDS.register(kind.name, kind, origin=__name__)


def get_kind(name: str) -> FeatureKind:
    """Look one up by name. Raises for an unregistered kind."""
    return KINDS.get(name)


def is_registered_kind(name: str) -> bool:
    """Whether ``name`` has a registered contract.

    ``False`` is not an error: a bespoke per-object feature kind is legitimate
    and the typechecker only warns. What it cannot do is claim the generality a
    registered kind has, because no contract says what its handlers owe.
    """
    return name in KINDS


def list_kinds() -> list[str]:
    """Every registered kind, sorted."""
    return KINDS.names()
