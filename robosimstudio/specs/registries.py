# SPDX-License-Identifier: Apache-2.0
"""The registries: domains and state layouts, engine bundles, backends, couplers, schemas, and what an engine consumes.

Every extension point of physics composition is a registry keyed by a string,
and this is where they all live -- :data:`DOMAINS`, :data:`SUBSTRATES`,
:data:`ENGINES`, :data:`BACKENDS`, :data:`COUPLERS` and the :data:`ACTOR_CONSUMES`
table -- each with the record type it holds and the ``register_*`` / ``get_*``
pair that fills and reads it. Nothing here imports a solver: a backend is
recorded by dotted path, and resolving one is the composer's job.

One of the four modules :mod:`~robosimstudio.specs.physics` was split into, and
the one the other three depend on: a declared domain is checked against
:data:`DOMAINS`, a stated backend is frozen with the same helper a registered
one is. Two things about what is *not* here:

* The library's three coupler **declarations** close this module (their
  builders are ``robosimstudio.scene.physics.couplers``'s, keyed by the same
  names). The Newton solver registrations do not: they live in
  ``robosimstudio.engines.newton``, the same shape as the foreign-engine
  bundles, and arrive through the same :data:`ENGINES` scan -- so no alias is
  registered by importing the IR, and :func:`get_backend` finds ``"mujoco"``
  exactly as it finds ``"superdex"``.
* :data:`ROLES` is the interaction vocabulary's word list and is stated here
  rather than beside ``ContactRule``, because :class:`DomainInfo` validates a
  domain's role and default contacts against it and ``declare`` imports from
  this module, not the other way round.

``SUBSTRATES``, ``register_substrate``, ``layout_chain``, ``reads_as`` and
``ENGINES`` are public and imported by name across the tree, and stay outside
``__all__`` as they were: the flat ``robosimstudio.specs`` namespace is the IR,
and these are its mechanism.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from robosimstudio.registry import Registry

from .params import REBUILD, RECAPTURE, ParamUI, ui

if TYPE_CHECKING:
    from .arrangement import BackendSpec

__all__ = [
    "ACTOR_CONSUMES",
    "BACKENDS",
    "COUPLERS",
    "DOMAINS",
    "ROLES",
    "STATE_KINDS",
    "BackendInfo",
    "CouplerInfo",
    "DomainInfo",
    "ParamRule",
    "ParamSchema",
    "actor_consumes",
    "declare_consumes",
    "engine_consumes",
    "get_backend",
    "get_coupler",
    "get_domain",
    "list_domains",
    "register_backend",
    "register_coupler",
    "register_domain",
    "rule",
]


# ---------------------------------------------------------------------------
# Domains: what kinds of state a thing in the scene carries
# ---------------------------------------------------------------------------

STATE_KINDS = ("bodies", "joints", "particles", "shapes")
"""The kinds of model state an entity can own, and therefore a solver entry can be given."""

SUBSTRATES: Registry[tuple[str, ...]] = Registry("state layout", same=operator.eq)
"""The state layouts a *measure* reads, each mapped to its **fallback chain**.

A layout is the shape of state a handler knows how to take: one pose, a pose plus
joint coordinates, an arc-length chain, particles over a rest layout. It is a
different axis from the domain physics integrates, and the two disagree across
the corpus -- a welded peg is ``domain="static"`` and reads as ``"rigid"``,
because nothing integrates it and it still has a pose to measure against.

**This was a closed four-tuple, and the closure was the wrong mechanism for a
sound reason.** The argument for it was that these are the shapes a handler knows
how to take, so a new one would be a layout nothing could measure -- which is
true, and is exactly what the chain fixes. A layout now declares what it may also
be *read as* when nothing more specific answers, so opening the set cannot strand
a kind: ``("volume", "cloth")`` is a tetrahedral body that would rather be
measured as a volume and will settle for a particle cloud.

The closure was also already being evaded. Two of the six shipped domains read as
something they are not: ``static`` reads as ``rigid`` (honest -- a fixture really
is one pose) and ``soft`` read as ``cloth`` (not honest -- a tetrahedral body is
not particles over a rest layout, and an early extension example taught the
same workaround for ``granular``).

The head of a chain is the layout itself and is **minted by being declared**;
every later member must already be registered, so a typo in the tail is refused
where it is written. A typo in the head mints a layout nothing dispatches to;
the way to catch that is to check that every registered layout reaches at
least one handler through its chain.
"""


def register_substrate(name: str, chain: Sequence[str] = (), *, origin: str = "") -> tuple[str, ...]:
    """Declare a state layout, and what it may also be read as.

    Args:
        name: The layout. Becomes the head of its own chain.
        chain: Layouts to fall back to, most specific first, when nothing
            implements a measure for this one. Each must already be registered.
        origin: Who declared it, for the duplicate message.

    Returns:
        The full chain, head first.

    Raises:
        ValueError: On a fallback nothing has registered -- which is the check
            the closed tuple used to perform, kept and moved rather than dropped.
    """
    tail = tuple(entry for entry in chain if entry != name)
    unknown = [entry for entry in tail if entry not in SUBSTRATES]
    if unknown:
        raise ValueError(
            f"state layout {name!r} falls back to {unknown}, which nothing has registered "
            f"(registered: {SUBSTRATES.names()}). A fallback must already exist -- the head of a "
            "chain is minted by declaring it, the tail is not."
        )
    return SUBSTRATES.register(name, (name, *tail), origin=origin or name)


for _layout in ("rigid", "articulated", "cloth"):
    register_substrate(_layout, origin="robosimstudio.specs.registries")
# Only the layouts the library's own handlers take. A layout that belongs to one
# kind of thing -- an arc-length chain, a volume -- is minted by that kind's
# ``register_domain(reads=...)``, in its own pack, beside the handlers that read
# it: a layout registered here with its handlers somewhere else is a layout
# nothing can measure on a checkout that lacks the pack.


def layout_chain(name: str, *, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """``name`` and everything it may also be read as, most specific first.

    An unregistered name comes back as its own one-entry chain: the typecheck and
    the goal binder each refuse an unknown layout in their own words, and refusing
    it twice here would turn a legible message into a stack trace from a lookup.

    ``extra`` appends further layouts *after* the name's own chain -- what an
    entity's capabilities mint, per :attr:`robosimstudio.scene.entity.Entity.layouts`.
    After, so the domain's answer still wins wherever it has one, and a
    capability's is the fallback. An empty ``name`` with extras is a thing with
    no record at all that its class nonetheless says something about.
    """
    base: tuple[str, ...] = SUBSTRATES.get(name) if name in SUBSTRATES else ((name,) if name else ())
    out = list(base)
    for layout in extra:
        for entry in SUBSTRATES.get(layout) if layout in SUBSTRATES else (layout,):
            if entry and entry not in out:
                out.append(entry)
    return tuple(out)


def reads_as(substrate: str, required: str) -> bool:
    """Whether a thing on ``substrate`` may be measured as ``required``.

    **The one rule for substrate compatibility**, so that every consumer answers
    it the same way. A layout is compatible with itself and with everything down
    its chain: a ``volume`` is a ``volume``, and it is a ``cloth`` too because it
    said it would settle for one.

    Three places ask this and used to ask it with ``==`` -- the goal binder, the
    handler dispatch, and the per-kind allowlist that says a ``set`` belongs on a
    cloth. Three copies of a comparison is how two of them end up disagreeing,
    and the disagreement is silent: an asset that validates but cannot bind, or
    binds and is scored by something else.
    """
    return not required or required in layout_chain(substrate)


# The interaction vocabulary's word list, stated here rather than beside
# ``ContactRule`` because ``DomainInfo`` below validates a domain's role and
# default contacts against it -- and ``declare`` imports from this module, not
# the other way round.
ROLES = ("surface", "fixtures", "objects", "deformables", "hands", "robot", "self", "world")
"""What a contact or attachment rule may name instead of an entity:

* ``surface`` -- the cell's own furniture (the table) and the ground;
* ``fixtures`` -- the task's declared statics (pegs, posts, a box);
* ``objects`` -- the task's free rigid bodies;
* ``deformables`` -- every cloth, rod and soft body;
* ``hands`` -- the embodiment's coupling surfaces (pads, fingers, the tool link);
* ``robot`` -- every body of the embodiment;
* ``self`` -- the asset itself (self-contact);
* ``world`` -- the world frame, for an attachment.
"""


@dataclass(frozen=True, kw_only=True)
class DomainInfo:
    """One physics domain: a name and the kinds of model state its entities carry.

    Attributes:
        name: The domain, as assets declare it (``"cloth"``, ``"rod"``).
        state: Which of :data:`STATE_KINDS` an entity of this domain owns. This
            is what lets the composer hand a solver the right index lists
            without knowing the solver: a ``"cloth"`` owns particles, a ``"rod"``
            owns bodies and joints.
        summary: One line for a reader.
        role: Which of :data:`ROLES` an entity of this domain plays in the
            interaction graph. **Declared rather than matched.**
        contacts: The roles this domain touches when an asset declares nothing.
        world_fixed: Whether its entities own no dynamics and sit on the world.
            True for scenery, and the last of the three things the composer
            matched ``domain == "static"`` for.
    """

    name: str
    state: tuple[str, ...]
    reads: tuple[str, ...]
    summary: str = ""
    role: str = "objects"
    contacts: tuple[str, ...] = ()
    world_fixed: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"domain {self.name!r} plays role {self.role!r}, which is not one of {ROLES}")
        unknown_contacts = [r for r in self.contacts if r not in ROLES]
        if unknown_contacts:
            raise ValueError(
                f"domain {self.name!r} declares contact with {unknown_contacts}, which are not roles. "
                f"The roles are {ROLES}."
            )
        unknown = [kind for kind in self.state if kind not in STATE_KINDS]
        if unknown:
            raise ValueError(f"domain {self.name!r} names state kinds {unknown}; the kinds are {STATE_KINDS}")
        # Stated as a bare string by every domain that reads as exactly one thing,
        # which is most of them; normalised here so ``reads[0]`` is always the
        # layout and the rest is always the fallback.
        if isinstance(self.reads, str):
            object.__setattr__(self, "reads", (self.reads,))
        if not self.reads:
            raise ValueError(
                f"domain {self.name!r} names no state layout. A domain says which layout a measure "
                "sees for it -- a welded fixture integrates nothing and still reads as 'rigid'."
            )
        unregistered = [layout for layout in self.reads if layout not in SUBSTRATES]
        if unregistered:
            raise ValueError(
                f"domain {self.name!r} reads as {unregistered}, which nothing has registered "
                f"(registered: {SUBSTRATES.names()}). register_domain() mints the head of a chain "
                "for you; a fallback has to exist already."
            )


ENGINES: Registry[None] = Registry("engine bundle", ("robosimstudio.engines",))
"""Foreign-engine bundles, scanned so their backends register themselves.

It holds nothing --- a backend goes in :data:`BACKENDS`, and this exists only so
that :func:`get_backend` has a package to scan when an alias it has never heard
of turns up. That is the same trick :data:`DOMAINS` plays for a domain bundle,
and it is here for the same reason: ``GENERIC`` names ``superdex`` for the soft
domain, and nothing guarantees anyone imported the bundle that registers it.

Separate from ``DOMAINS`` because **an engine is not a domain**. While SuperDex
was the only foreign engine it registered from ``domains/soft``, which was
defensible --- that domain can be simulated by nothing else. It stops being
defensible for an engine that serves cloth, soft *and* rod: such an engine
belongs to none of them, and hanging it off whichever domain happened to need it
first is how a registry acquires a wrong shape it then keeps.
"""

DOMAINS: Registry[DomainInfo] = Registry("domain", ("robosimstudio.domains",), same=operator.eq)
"""Registered domains, by name.

Scanned rather than listed. Each built-in used to register here, next to the type
that describes it; each now registers in its own bundle under
``robosimstudio.domains``, beside the entity that builds it and the measures that
read it -- so a domain is one place rather than three.

The scan is what makes that safe. A measure has to exist before a goal binds
(``lower()`` -> ``bind()`` -> ``dispatch``), and this layer cannot import the one
holding the bundles. :meth:`Registry.load` imports them on the first lookup,
inside a function, which is the same lazy shape the four sanctioned cyclic edges
already use.

A task registers its own the same way, from its ``assets/`` package -- which
became true when ``taskbuild`` began installing a populator here. It was not
before: this registry scanned a package and had no populator, so a domain
declared by content registered only if something happened to import that task
first, which is a coincidence rather than a mechanism."""


def register_domain(
    name: str,
    *,
    state: tuple[str, ...],
    reads: str | Sequence[str],
    summary: str = "",
    role: str = "objects",
    contacts: tuple[str, ...] = (),
    world_fixed: bool = False,
) -> DomainInfo:
    """Register a physics domain. Identical re-registration is accepted.

    Args:
        name: The domain, as assets declare it.
        state: Which of :data:`STATE_KINDS` an entity of this domain owns.
        reads: Which state layout a *measure* sees it as. Stated rather than
            defaulted because it is not derivable from ``state``: ``rod`` and
            ``rigid`` own the same kinds and are read completely differently.

            A bare string is a layout that stands alone. A sequence is a
            **chain**, most specific first: the head is this domain's own layout
            and is minted here, and the rest is what a measure may fall back to
            when nothing implements it for the head. ``("volume", "cloth")``
            says "measure me as a volume if you can, as a particle cloud if you
            cannot" -- which is what lets a genuinely new layout arrive without
            being unmeasurable on the day it does.
        summary: One line for a reader.
    """
    # The head is minted before the record is built, because ``DomainInfo``
    # validates against the registry rather than writing to it: a dataclass whose
    # constructor registers is one that cannot be built to ask a question about.
    chain = (reads,) if isinstance(reads, str) else tuple(reads)
    register_substrate(chain[0], chain[1:], origin=name)
    info = DomainInfo(
        name=name,
        state=tuple(state),
        reads=chain,
        summary=summary,
        role=role,
        contacts=tuple(contacts),
        world_fixed=world_fixed,
    )
    return DOMAINS.register(name, info, origin=name)


def get_domain(name: str) -> DomainInfo:
    return DOMAINS.get(name)


def list_domains() -> list[str]:
    return DOMAINS.names()


# ---------------------------------------------------------------------------
# Parameter schemas: what a solver kwarg is, and what changing it costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ParamRule:
    """How one backend or coupler parameter is presented and applied.

    Attributes:
        ui: The presentation and the change tier, exactly as a spec field
            carries it in ``field(metadata=ui(...))``.
        apply: How a change lands on a running program. ``"rebuild"`` means it
            is consumed at construction and needs a fresh scene;
            ``"attr:<name>"`` means the solver keeps it as a plain attribute
            that may be reassigned (RECAPTURE: a captured graph baked the old
            value); ``"custom:<name>"`` names a coupler- or solver-specific
            applier the composer knows.
    """

    ui: ParamUI
    apply: str = "rebuild"


ParamSchema = Mapping[str, ParamRule]
"""Parameter name -> rule, for one backend alias or coupler kind."""


def rule(*, apply: str = "rebuild", tier: str = REBUILD, **ui_kwargs: Any) -> ParamRule:
    """One schema entry. ``rule(tier=RECAPTURE, apply="attr:iterations", range=(1, 100))``."""
    return ParamRule(ui=ui(tier=tier, **ui_kwargs)[next(iter(ui(tier=tier, **ui_kwargs)))], apply=apply)


# ---------------------------------------------------------------------------
# Backends: solvers by alias or dotted path
# ---------------------------------------------------------------------------


def _frozen_params(params: Mapping[str, Any] | tuple[tuple[str, Any], ...]) -> tuple[tuple[str, Any], ...]:
    """Parameters as a sorted, hashable tuple of pairs.

    Sorted so two arrangements stating the same parameters in a different order
    compare equal, which is what lets the registries accept a redundant copy.
    Values must be hashable: a list becomes a tuple, a dict is refused.
    """
    items = dict(params).items()
    out: list[tuple[str, Any]] = []
    for key, raw in sorted(items):
        value = tuple(raw) if isinstance(raw, list) else raw
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"parameter {key!r} must be hashable (got {type(value).__name__}); specs are frozen"
            ) from exc
        out.append((str(key), value))
    return tuple(out)


@dataclass(frozen=True, kw_only=True)
class BackendInfo:
    """One registered solver backend.

    Attributes:
        alias: The short name an arrangement uses (``"vbd"``), or the dotted
            path itself for an unregistered solver.
        cls: Dotted path of the :class:`newton.solvers.SolverBase` subclass.
            A string, so registering a backend imports nothing.
        supports: Which :data:`STATE_KINDS` the solver can own. Empty means
            unknown, which the composer treats as "anything" and leaves to the
            solver's own constructor to refuse.
        schema: Per-parameter presentation and tiers, or ``None``.
        consumes: The :class:`HostActor` parameter names this engine reads, or
            empty to say nothing. Declared so a parameter an entity states and the
            engine never looks at is a warning at compose naming both, rather
            than a stiffness that silently became a default. Only meaningful
            for a host engine; a Newton solver takes its material off the model.
        summary: One line for a reader.
    """

    alias: str
    cls: str
    supports: tuple[str, ...] = ()
    schema: ParamSchema | None = None
    summary: str = ""
    consumes: tuple[str, ...] = ()
    maximal_coordinates: bool = False
    """The solver integrates bodies in maximal coordinates and leaves ``joint_q``
    to be mirrored back by ``eval_ik`` -- VBD and XPBD. A registered fact, so
    the composer asks it here rather than keeping an alias list."""
    native_contacts: bool = False
    """The solver generates its own contacts (MuJoCo), so the composer routes
    none of the scene's contact pairs to it."""
    defaults: tuple[tuple[str, Any], ...] = ()
    """Constructor kwargs applied under whatever an arrangement states.

    What this stack has always handed the solver when an arrangement said
    nothing -- MuJoCo's sparse Jacobian and its 2048 ceilings, VBD's
    millimetre-scale self-contact radii -- so a registered alias means the
    same physics it meant before the registry existed. Recorded into the
    manifest as the *effective* parameters, never hidden. A dotted-path
    backend has none: what it is handed is exactly what was stated."""
    particle_defaults: tuple[tuple[str, Any], ...] = ()
    """Defaults applied only when the island owns particles -- VBD's
    self-contact settings make no sense over a rod alone."""
    host_engine: bool = False
    """Whether this backend is a foreign engine stepped on the host.

    Registered knowledge, like :attr:`supports`, so the composer can ask
    without importing the class: a host island is split out before the Newton
    couplers, its solver's ``step`` is a no-op inside the captured graph, and
    the real work runs as an eager per-control-step stage. False for every
    solver that integrates Newton's own state on the device."""

    def __eq__(self, other: object) -> bool:  # schemas are mappings; compare them by content
        if not isinstance(other, BackendInfo):
            return NotImplemented
        return (
            self.alias,
            self.cls,
            self.supports,
            dict(self.schema or {}),
            self.summary,
            self.defaults,
            self.particle_defaults,
            self.host_engine,
            self.consumes,
            self.maximal_coordinates,
            self.native_contacts,
        ) == (
            other.alias,
            other.cls,
            other.supports,
            dict(other.schema or {}),
            other.summary,
            other.defaults,
            other.particle_defaults,
            other.host_engine,
            other.consumes,
            other.maximal_coordinates,
            other.native_contacts,
        )

    def effective(self, spec: BackendSpec, *, has_particles: bool = True) -> BackendSpec:
        """``spec`` with this alias's defaults filled in underneath what it states."""
        merged = dict(self.defaults)
        if has_particles:
            merged.update(dict(self.particle_defaults))
        merged.update(spec.kwargs())
        return replace(spec, params=_frozen_params(merged))

    __hash__ = None  # type: ignore[assignment]


BACKENDS: Registry[BackendInfo] = Registry("backend", same=operator.eq)
"""Registered solver backends, by alias.

Empty until a bundle registers into it. Newton's own solvers arrive from
``robosimstudio.engines.newton`` through the :data:`ENGINES` scan
:func:`get_backend` runs on a miss, the same door the foreign engines use --
so ask :func:`get_backend`, which scans, rather than testing membership here
before anything has."""

ACTOR_CONSUMES: dict[str, tuple[str, ...]] = {}
"""``"engine:kind"`` -> the :class:`HostActor` parameter names that engine's
builder for that kind reads. Declared by ``register_actor_builder(...,
consumes=)`` beside the builder, because that is where the reads are: a shell
builder reads ``thickness`` and never ``friction``, and an engine-wide union
cannot say so. Held here rather than in ``engines`` because the composer asks
it, and the composer names no engine."""


def declare_consumes(engine: str, kind: str, keys: Sequence[str]) -> None:
    """Record what ``engine``'s ``kind`` builder reads. Empty declares nothing."""
    if keys:
        ACTOR_CONSUMES[f"{engine}:{kind}"] = tuple(keys)


def actor_consumes(engine: str, kind: str) -> tuple[str, ...] | None:
    """What ``engine``'s builder for ``kind`` reads, or ``None`` when it declares nothing."""
    return ACTOR_CONSUMES.get(f"{engine}:{kind}")


def engine_consumes(engine: str) -> tuple[str, ...]:
    """The union of what every builder ``engine`` registered reads."""
    prefix = f"{engine}:"
    return tuple(sorted({key for name, keys in ACTOR_CONSUMES.items() if name.startswith(prefix) for key in keys}))


def register_backend(
    alias: str,
    *,
    cls: str | type,
    supports: tuple[str, ...] = (),
    schema: ParamSchema | None = None,
    summary: str = "",
    defaults: Mapping[str, Any] | None = None,
    particle_defaults: Mapping[str, Any] | None = None,
    host_engine: bool = False,
    consumes: Sequence[str] = (),
    maximal_coordinates: bool = False,
    native_contacts: bool = False,
) -> BackendInfo:
    """Register a solver under an alias. Identical re-registration is accepted.

    Args:
        alias: The name arrangements will use.
        cls: The solver class, or its dotted path. A class is recorded by its
            dotted path so the registry itself holds no import.
        supports: State kinds the solver can integrate; empty leaves it unknown.
        schema: Optional parameter schema for the studio and for live tuning.
        summary: One line for a reader.
        defaults: Constructor kwargs applied under what an arrangement states.
        particle_defaults: Defaults applied only to an island that owns particles.
        host_engine: Whether this backend is a foreign engine stepped on the
            host rather than a Newton solver stepped on the device -- see
            :attr:`BackendInfo.host_engine`.
    """
    path = cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"
    info = BackendInfo(
        alias=alias,
        cls=path,
        supports=tuple(supports),
        schema=schema,
        summary=summary,
        defaults=_frozen_params(dict(defaults or {})),
        particle_defaults=_frozen_params(dict(particle_defaults or {})),
        host_engine=bool(host_engine),
        consumes=tuple(consumes),
        maximal_coordinates=bool(maximal_coordinates),
        native_contacts=bool(native_contacts),
    )
    return BACKENDS.register(alias, info, origin=path)


def get_backend(name: str) -> BackendInfo:
    """The backend an arrangement named: a registered alias, or a dotted path.

    A dotted path needs no registration -- that is the extensibility contract --
    so one resolves to an anonymous :class:`BackendInfo` with no schema and
    unknown support.
    """
    if name in BACKENDS:
        return BACKENDS[name]
    if "." in name:
        return BackendInfo(alias=name, cls=name)
    # An alias a bundle registers exists only once that bundle has been
    # imported, and nothing guarantees anyone imported it: GENERIC names
    # "superdex" for the soft domain, and an arm-only scene never touches
    # either bundle on its own. These two scans are the registries that *do*
    # know where bundles live, so a miss loads them once and retries before
    # concluding the name is wrong. Engines first: that is where a backend
    # belongs, and a domain bundle registering one is the exception. Newton's
    # own solvers are a bundle too (``robosimstudio.engines.newton``), so this
    # is the path "mujoco" takes as well: nothing is registered by importing
    # the IR, and an arm-only scene resolves its solver the same way a soft
    # one does -- which is why the engine scan is checked before the domain
    # scan runs, not after both: the domain bundles import the asset library,
    # and a scene with no deformable in it should not pay for that to learn
    # where MuJoCo is.
    ENGINES.load()
    if name in BACKENDS:
        return BACKENDS[name]
    DOMAINS.load()
    if name in BACKENDS:
        return BACKENDS[name]
    raise KeyError(f"unknown backend {name!r}: neither a registered alias ({BACKENDS.names()}) nor a dotted path")


# ---------------------------------------------------------------------------
# Couplers: how islands exchange momentum
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CouplerInfo:
    """One registered coupling algorithm.

    Attributes:
        method: The name an arrangement uses (``"proxy"``, ``"admm"``).
        schema: Per-parameter presentation and tiers, or ``None``.
        summary: One line for a reader.
    """

    method: str
    schema: ParamSchema | None = None
    summary: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CouplerInfo):
            return NotImplemented
        return (self.method, dict(self.schema or {}), self.summary) == (
            other.method,
            dict(other.schema or {}),
            other.summary,
        )

    __hash__ = None  # type: ignore[assignment]


COUPLERS: Registry[CouplerInfo] = Registry("coupler", same=operator.eq)
"""Registered coupler kinds, by name. The *builders* live a layer up, in
``robosimstudio.scene.physics.couplers``, keyed by the same names."""


def register_coupler(method: str, *, schema: ParamSchema | None = None, summary: str = "") -> CouplerInfo:
    """Declare a coupler kind. Its builder is registered beside the composer."""
    return COUPLERS.register(method, CouplerInfo(method=method, schema=schema, summary=summary))


def get_coupler(method: str) -> CouplerInfo:
    return COUPLERS.get(method)


# ---------------------------------------------------------------------------
# Built-in schemas and registrations
# ---------------------------------------------------------------------------
#
# The couplers' half. The Newton solvers' half -- five ``register_backend``
# calls and the two schemas beside them -- is ``robosimstudio.engines.newton``.

PROXY_SCHEMA: dict[str, ParamRule] = {
    "mode": rule(choices=("staggered", "lagged"), group="coupling"),
    "iterations": rule(range=(1.0, 8.0), step=1.0, group="coupling", doc="Relaxation passes per substep."),
    "mass_scale": rule(tier=RECAPTURE, apply="custom:proxy_mass_scale", log=True, range=(1.0, 1.0e3), group="coupling"),
    "one_way_mass_scale": rule(log=True, range=(1.0e2, 1.0e9), group="coupling"),
    "collide_interval": rule(range=(1.0, 8.0), step=1.0, group="cadence"),
    "contact_matching": rule(choices=("disabled", "latest", "sticky"), group="contact"),
    "proxy_relaxation": rule(range=(0.1, 2.0), group="coupling"),
    "proxy_joints": rule(group="coupling", doc="Keep the hands' joints as proxy joints in the destination."),
}

ADMM_SCHEMA: dict[str, ParamRule] = {
    "iterations": rule(range=(1.0, 50.0), step=1.0, group="coupling"),
    "rho": rule(log=True, range=(1e-3, 1e3), group="coupling"),
    "gamma": rule(range=(0.0, 10.0), group="coupling"),
    "baumgarte": rule(range=(0.0, 1.0), group="coupling"),
    "rigid_contact_matching": rule(choices=("disabled", "latest", "sticky"), group="contact"),
}

register_coupler("proxy", schema=PROXY_SCHEMA, summary="Mass-scaled virtual proxies of one island's bodies in another.")
register_coupler("admm", schema=ADMM_SCHEMA, summary="Linearised ADMM over cross-island contacts and attachments.")
register_coupler(
    "one_way",
    summary=(
        "A rigid island leading a deformable one as a moving boundary: the deformable collides with the rigid "
        "bodies (full-surface contact allowed) and never pushes back."
    ),
)
