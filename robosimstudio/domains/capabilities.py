# SPDX-License-Identifier: Apache-2.0
"""How a kind's state is *held*, and where a pack publishes what a kind can be *asked*.

A domain answers one question and is quietly asked two. It picks the solver, and
it decides what the thing is to everybody else -- which measures bind, which
actor a foreign engine is handed, which affordances reach it. Those are
different questions, and they are kept apart:

``domain``
    Which model state a thing owns, and therefore which backend serves it.
    Declared by ``register_domain``.

**capability**
    Which geometry a thing can show, and therefore which measures and which
    affordances it gets, *whatever its domain*. A mixin with one abstract
    geometry method and everything else derived from it; its ``layouts`` are
    what :func:`~robosimstudio.scene.entity.entity_layouts` unions into the
    chain a measure dispatches on, so a thing of one domain can be read as
    another's layout because its class says it can show that geometry.

**No capability is defined in this module, and that is the rule rather than a
gap.** A capability has to be earned by a kind that needs it: more than one
implementation that disagree about everything internal, and real behaviour
derived from the one method they share. So it ships in that kind's **domain
pack**, beside the entities that implement it -- and a shape with no second user
is a guess about the future, which is why none is declared here on spec.

What is here is where a pack **publishes** one. :func:`register_capability`
makes a pack's mixin importable as
``robosimstudio.domains.capabilities.<Name>``, so an entity in another pack, or
in nobody's, declares it without naming the pack that ships it. A capability
whose pack is not installed is simply not an attribute of this module.

The mixin that *is* here is the other axis, from the storage side:
:class:`ParticleMixin` is the particle lifecycle, which is about *where state
lives* rather than about what may be asked of it. :func:`host_bridge` finds the
foreign engine's bridge for the island an entity landed in.
"""

from __future__ import annotations

import operator
from typing import Any, ClassVar

import newton
import numpy as np

from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.registry import Registry

__all__ = ["CAPABILITIES", "ParticleMixin", "host_bridge", "register_capability"]


CAPABILITIES: Registry[type] = Registry("capability", ("robosimstudio.domains",), same=operator.is_)
"""Capability mixins published by the domain packs that ship them, by class name.

Filled by :func:`register_capability` while a pack is imported, and read by this
module's ``__getattr__`` -- so ``from robosimstudio.domains.capabilities import
<Name>`` resolves a capability to whichever installed pack ships it.

A registry like the entity and material ones, and for the same reason: it scans
the domain packages on its first lookup, so what is published does not depend on
whether anything happened to import the pack before asking. One name is one
class -- the same class again is accepted, a different one under a taken name is
refused with both origins."""


def register_capability(cls: type) -> type:
    """Publish a capability a domain pack ships, importable from this module by its class name.

    Called by the pack, at module scope, beside the mixin it defines. Usable as
    a decorator.

    Args:
        cls: The capability mixin. Its ``__name__`` is the name it is published
            under, and it should state ``layouts``: the state layouts a class
            declaring it may additionally be read as.
    """
    return CAPABILITIES.register(cls.__name__, cls, origin=cls.__module__)


def __getattr__(name: str) -> Any:
    """Resolve a capability a pack published (:pep:`562`), scanning the packs first.

    Only names this module does not define reach here, and a capability is
    registered while its pack is imported -- so the lookup is the registry's,
    which imports the domain packages once and names none of them.
    """
    if name.startswith("_"):
        # A private or dunder probe (``__path__``, ``_pytestfixturefunction``) is never a capability,
        # and must not cost a scan of every domain package to be told so.
        raise AttributeError(name)
    if name in CAPABILITIES:
        return CAPABILITIES.get(name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. Defined here: {sorted(__all__)}; published by the "
        f"installed domain packs: {CAPABILITIES.names()}. A capability ships in the domain pack that "
        "earned it, and is importable from here only while that pack is installed."
    )


class ParticleMixin:
    """**My state is particles.** The lifecycle every particle-backed kind shares.

    Storage, not capability: this says where a thing's state lives, not what may
    be asked of it. One kind may be particle-backed in one implementation and
    body-backed in another and answer the same capability in both; a thing may
    be particle-backed and declare a capability a pack publishes. The two axes
    are exactly ``DomainInfo.state`` against the layout a measure reads, which
    the physics vocabulary already keeps apart.

    Only the five members that were **byte-identical** where they overlapped are
    here -- verified by comparing the three implementations' method bodies rather
    than by eye. ``__init__``, ``build``, ``host_actor``, ``on_scene_ready``,
    ``resolve_sets``, ``set_indices``, ``settle`` and ``update_probe`` genuinely
    differ between a sheet, a foam block and a centerline, and stay with them.
    """

    particles: np.ndarray
    probe: ParticleProbe | None
    sets: ClassVar[dict[str, np.ndarray]] = {}
    """Named subsets of :attr:`particles`. Class-level default so a kind with no
    named sets -- a bare centerline -- still publishes a handle."""
    tube_radius: float = 0.0
    """The radius a renderer sweeps a centerline by, or 0 for a kind drawn from
    its triangles. A centerline owns no triangle, so without it the path tracer
    drew a simulated, measured curve as nothing at all."""

    def bind(self, model: newton.Model) -> None:
        """Take this thing's particle range off the model and open a probe per world.

        The range is the **template's**, like every index an entity resolves at
        bind; a replicated model holds ``W`` copies of it at the particle
        stride, and a probe built on the first block would answer every world
        with the first world's particles. So there is one probe per world, each
        over its own block, and :attr:`probe` stays the first of them -- which is
        what a single-world caller has always been handed and still is.
        """
        super().bind(model)  # type: ignore[misc]
        self.particles = self.range.particle_indices  # type: ignore[attr-defined]
        worlds = getattr(self, "worlds", None)
        count = int(getattr(worlds, "num_worlds", 1) or 1)
        stride = int(getattr(worlds, "particles_per_world", 0) or 0)
        self.probes = []  # type: ignore[attr-defined]
        for world in range(count):
            probe = ParticleProbe(model, world=world)
            probe.add_set(self.spec.name, self.particles + world * stride)  # type: ignore[attr-defined]
            self.probes.append(probe)  # type: ignore[attr-defined]
        self.probe = self.probes[0]  # type: ignore[attr-defined]

    def add_set_every_world(self, key: str, indices: Any) -> None:
        """Register a named particle set on every world's probe.

        *indices* are the template's, as every index an entity resolves is; a
        replicated model holds ``W`` copies of them at the particle stride, so
        each probe gets its own world's copy. Adding the set to :attr:`probe`
        alone -- which is what this replaced -- left worlds 1..W-1 knowing only
        the base set, and a goal asking for a named one got ``unknown particle
        set 'towel/moving'`` from the first world past zero.
        """
        import numpy as _np

        probes = getattr(self, "probes", None) or ([self.probe] if getattr(self, "probe", None) else [])
        worlds = getattr(self, "worlds", None)
        stride = int(getattr(worlds, "particles_per_world", 0) or 0)
        base = _np.asarray(indices, dtype=_np.int64)
        for world, probe in enumerate(probes):
            if probe is not None:
                probe.add_set(key, base + world * stride)

    def probe_for(self, world: int = 0) -> Any:
        """This thing's probe for one world. The only one, at one world."""
        probes = getattr(self, "probes", None) or [getattr(self, "probe", None)]
        return probes[int(world) % len(probes)] if probes[0] is not None else None

    def handle(self, model: newton.Model) -> Any:
        """This thing's particles and named sets, for renderers and goals.

        ``triangles`` come off the model through the same reader the render path
        uses, so a sheet draws from the handle rather than from a second walk of
        the connectivity. A kind with no triangles -- a centerline -- gets an
        empty array and is drawn as what it is.
        """
        from robosimstudio.core.render.visual import cloth_triangles
        from robosimstudio.scene.handles import DeformableHandle

        particles, triangles = cloth_triangles(model, self.particles)
        return DeformableHandle(
            name=self.name,  # type: ignore[attr-defined]
            particles=particles,
            sets=dict(self.sets),
            triangles=triangles,
            tube_radius=float(self.tube_radius),
        )

    def snapshot_rest(self, pipeline: Any) -> None:
        """Record what a reset returns to: the drape this thing settled into.

        **Every world's probe, not just the first.** :meth:`bind` opens one
        probe per world over that world's own block and keeps the first as
        :attr:`probe`; baselining only that one left worlds 1..W-1 with none,
        and a goal that measures against the settled drape -- ``mirror_pairs``,
        which is how a fold is scored -- raised "no settled baseline" on the
        first world past zero it was asked about. The settle steps every world
        at once, so there is a baseline to take for each of them here.
        """
        state = pipeline.state_0
        self.rest = {  # type: ignore[attr-defined]
            "particle_q": state.particle_q.numpy().copy(),
            "particle_qd": state.particle_qd.numpy().copy(),
        }
        probes = getattr(self, "probes", None) or ([self.probe] if self.probe is not None else [])
        for probe in probes:
            if probe is None:
                continue
            probe.update(state)
            probe.snapshot_baseline()

    def join_settle(self, pipeline: Any, run: tuple[int, int] | None = None) -> None:
        """Join the scene's one particle settle, then re-baseline every thing in it.

        A settle steps the whole scene and a reset restores the last whole-array
        snapshot, so the particle things in a scene share one record on the
        pipeline, ``particle_settles``: a *run* (``(frames, substeps)``) one of
        them already ran is not run again, and each that has snapshotted
        snapshots again after one that is. ``run=None`` joins without settling,
        as a kind that does not settle does.
        """
        settles = getattr(pipeline, "particle_settles", None)
        if settles is None:
            settles = pipeline.particle_settles = {"runs": set(), "things": []}
        if run is not None and run not in settles["runs"]:
            settles["runs"].add(run)
            self.settle(pipeline)  # type: ignore[attr-defined]
        settles["things"].append(self)
        for thing in settles["things"]:
            thing.snapshot_rest(pipeline)

    def reset(self, pipeline: Any, episode: Any, *, world: int = 0) -> None:
        """Put the particles back where :meth:`snapshot_rest` found them.

        The rest field covers the *whole* particle array, replicated or not --
        a snapshot taken after the settle already holds every world's copy --
        so this restores the fleet in one assign and the world it is told about
        makes no difference. Accepted anyway, because the phase carries it.
        """
        del episode, world
        if not self.rest:  # type: ignore[attr-defined]
            return
        for state in (pipeline.state_0, pipeline.state_1):
            state.particle_q.assign(self.rest["particle_q"])  # type: ignore[attr-defined]
            state.particle_qd.assign(self.rest["particle_qd"])  # type: ignore[attr-defined]

    def observe(self, state: newton.State, step: int | None = None, *, readback: Any = None) -> dict[str, float]:
        """A summary of this thing's particles, never the particles themselves.

        A cloth of thousands of particles at control rate is gigabytes an
        episode, and a policy reasoning about a garment does not need every
        vertex."""
        return self.update_probe(state, step, readback=readback).summary(self.spec.name)  # type: ignore[attr-defined]


def host_bridge(pipeline: Any, entity_name: str) -> Any | None:
    """The foreign engine's bridge simulating ``entity_name``'s island, or ``None``.

    Duck-typed against the program the pipeline was bound to: the composed
    solver exposes ``solver(island_name)`` and a host solver carries a
    ``bridge``. ``None`` -- quietly -- when the arrangement put this domain on a
    Newton solver instead: the entity is then an ordinary deformable and nothing
    engine-side applies.

    Both routes to the island's own solver are tried, and the second is not a
    belt-and-braces: ``solver_for`` answers with the *composed* solver on an
    uncoupled program, which is what a scene holding nothing but the host island
    is -- a sponge on a bench and no robot. That answer carries no ``bridge``,
    so a scene of exactly one engine used to report having no engine, and every
    reader of this (the entity's own ``bridge``, the studio's handles, the
    contact probes) quietly got ``None`` on the one scene shape the engine is
    easiest to demonstrate on.
    """
    program = getattr(pipeline, "program", None)
    if program is None:
        return None
    island_of = getattr(program, "island_of", None)
    island = None
    if callable(island_of):
        try:
            island = island_of(entity_name)
        except (KeyError, LookupError):
            island = None
    if island is None:
        return None
    solver_for = getattr(program, "solver_for", None)
    solver = solver_for(entity_name) if callable(solver_for) else None
    bridge = getattr(solver, "bridge", None)
    if bridge is not None:
        return bridge
    solver_of = getattr(getattr(program, "solver", None), "solver", None)
    solver = solver_of(island.name) if callable(solver_of) else None
    return getattr(solver, "bridge", None)
