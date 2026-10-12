# SPDX-License-Identifier: Apache-2.0
"""What an asset declares it needs: a domain, contact rules by role, and what it is to a foreign engine.

The declaration half of physics composition, as data. An asset says *what it
is* and *what it touches* (:class:`SimRequirements`) and, when a host engine
will integrate it, *what shape of thing it is* (:class:`HostActor`). It never
names a solver: the arrangement (:mod:`~robosimstudio.specs.arrangement`) says
which backend serves each domain, and the composer derives the rest.

One of the four modules :mod:`~robosimstudio.specs.physics` was split into.
It reads the registries (a declared domain is checked against ``DOMAINS``) and
the arrangement's :class:`~robosimstudio.specs.arrangement.BackendSpec` for the
per-asset override, and nothing else. :data:`~robosimstudio.specs.registries.ROLES`,
the word list a rule names, lives beside ``DomainInfo`` for the reason given
there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .arrangement import BackendSpec
from .registries import DOMAINS

__all__ = [
    "COUPLINGS",
    "ContactRule",
    "HostActor",
    "SimRequirements",
    "contacts_with",
    "sim_to_dict",
]


# ---------------------------------------------------------------------------
# Interaction vocabulary: what an asset touches, by role
# ---------------------------------------------------------------------------

COUPLINGS = ("two_way", "one_way", "off")
"""How a cross-island contact exchanges momentum: both ways, rigid-to-deformable
only (the rigid side never feels the deformable), or not at all."""


@dataclass(frozen=True, kw_only=True)
class ContactRule:
    """One thing an asset collides with.

    Attributes:
        with_: A role from :data:`~robosimstudio.specs.registries.ROLES` or an entity name.
        coupling: One of :data:`COUPLINGS`. Only meaningful across islands --
            two things in one solver always couple both ways.
    """

    with_: str
    coupling: str = "two_way"

    def __post_init__(self) -> None:
        if self.coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}, got {self.coupling!r}")
        if not self.with_:
            raise ValueError("a contact rule needs a role or an entity name")


def contacts_with(*roles: str, coupling: str = "two_way") -> tuple[ContactRule, ...]:
    """``contacts_with("surface", "hands")`` -> two rules with one coupling."""
    return tuple(ContactRule(with_=role, coupling=coupling) for role in roles)


@dataclass(frozen=True, kw_only=True)
class SimRequirements:
    """What one asset needs of the simulation to behave like itself.

    The declaration half of physics composition: an asset says *what it is* and
    *what it touches*, in its own terms, and the arrangement says which solver
    serves that. A towel says "I am cloth, hands must be able to push me, and I
    rest on the surface and the fixtures"; it never says ``SolverVBD(iterations=40)``.

    Attributes:
        domain: A registered physics domain. The library ships ``"rigid"``,
            ``"articulated"``, ``"rod"``, ``"cloth"`` and ``"static"``; a
            content pack registers its own the same way, and the composer treats
            it on the same terms -- see :func:`register_domain`.
        contacts: What the asset collides with, by role or name, and how the
            contact couples across islands. Empty means the composer's default
            for the domain.
        contributes_hooks: Named substep stages this asset installs -- a bond
            field, a pin, a latch -- so a manifest can state the stage set as
            part of the configuration.
        self_contact: Whether the asset must collide with itself. A folded
            sheet that passes through itself is not folded.
        backend: A per-asset backend override, or ``None`` for the
            arrangement's choice for the domain. The permanent escape hatch:
            an asset whose physics needs a particular solver states it here,
            versioned by whatever arrangement carries it.
        notes: Why a number is what it is.
    """

    domain: str = "rigid"
    contacts: tuple[ContactRule, ...] = ()
    contributes_hooks: tuple[str, ...] = ()
    self_contact: bool = False
    backend: BackendSpec | None = None
    notes: str = ""

    @property
    def family(self) -> str:
        """The domain, under the name earlier declarations used."""
        return self.domain

    def contact(self, with_: str) -> ContactRule | None:
        """The rule naming ``with_``, or ``None``."""
        return next((r for r in self.contacts if r.with_ == with_), None)

    @property
    def hand_coupling(self) -> str:
        """How the hands couple to this asset: a coupling, or ``"off"`` when nothing says."""
        hands = self.contact("hands")
        return hands.coupling if hands is not None else "off"

    def problems(self) -> list[str]:
        """Everything structurally wrong with this declaration."""
        out: list[str] = []
        if self.domain not in DOMAINS:
            out.append(f"domain {self.domain!r} is not registered (registered: {sorted(DOMAINS)})")
        for contact in self.contacts:
            if contact.with_ == "self" and not self.self_contact:
                out.append("a contact rule names 'self' but self_contact is False; say one thing")
        return out


@dataclass(frozen=True, kw_only=True)
class HostActor:
    """What an entity is, to a foreign engine that will integrate it.

    The declaration a host-engine island needs beyond the model's own arrays.
    The composer supplies the particle range -- it is the only party that knows
    where the entity landed -- and this supplies the two things only the entity
    knows: **what shape of thing this is** and the scalars its engine will want.

    Attributes:
        kind: A registered actor kind --
            :func:`~robosimstudio.engines.geometry.register_actor_kind`. The
            two the tree ships are ``"shell"`` (a membrane: triangles) and
            ``"tet"`` (a volume). The set is open, which is the point: a rod is
            neither, and before it was open a rod could not be simulated by a
            foreign engine at all.
        params: JSON-able scalars, recorded verbatim into the manifest's
            host-island record -- which is how a repro bundle rebuilds the same
            actor from the model and the manifest alone. Anything derivable from
            the model belongs in the model: a tet body's elasticity is inverted
            back out of ``tet_materials`` rather than restated here, so there is
            exactly one statement of what the body is made of.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def record(self, *, name: str, particle_start: int, particle_count: int) -> dict[str, Any]:
        """This declaration plus the range the composer resolved, as plain JSON."""
        return {
            "name": str(name),
            "kind": str(self.kind),
            "particle_start": int(particle_start),
            "particle_count": int(particle_count),
            "params": dict(self.params),
        }


def sim_to_dict(spec: SimRequirements) -> dict[str, Any]:
    """Requirements as plain JSON-able data, for the catalog and the dump."""
    out: dict[str, Any] = {"domain": spec.domain}
    if spec.contacts:
        out["contacts"] = [{"with": r.with_, "coupling": r.coupling} for r in spec.contacts]
    if spec.contributes_hooks:
        out["contributes_hooks"] = list(spec.contributes_hooks)
    if spec.self_contact:
        out["self_contact"] = True
    if spec.backend is not None:
        out["backend"] = {
            "solver": spec.backend.solver,
            "params": dict(spec.backend.params),
            "substeps": spec.backend.substeps,
        }
    if spec.notes:
        out["notes"] = spec.notes
    return out
