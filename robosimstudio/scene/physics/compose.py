# SPDX-License-Identifier: Apache-2.0
"""The composer: from bound entities and an arrangement to a steppable program.

:func:`compose` runs once per compile, after every entity is bound and before
the substep loop exists:

1. the **interaction graph** -- what is there, what it owns, what it touches;
2. the **islands** -- one solver per distinct backend, the robot's first;
3. the **channels** -- for every two-way or one-way contact between a rigid
   island and a deformable one, the rigid bodies that must be proxied: the
   hands for a ``hands`` rule, the whole object for an ``objects`` one;
4. the **routing** -- which shape pairs each pipeline detects, once;
5. the **solver** -- bare, proxy-coupled or ADMM-coupled, with its pipelines;
6. the **schedule** -- the mirror stage for maximal-coordinate islands, and the
   slots entities contribute their kernels to;
7. the **manifest** -- all of the above as data.

What comes back is a :class:`PhysicsProgram`, which the compiler hands to the
pipeline and to every entity that needs to ask which solver owns it.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.schedule import Schedule, Stage, SubstepHooks
from robosimstudio.specs.params import REBUILD, RECAPTURE
from robosimstudio.specs.physics import (
    ArrangementSpec,
    ChannelRecord,
    HostActor,
    IslandRecord,
    PhysicsManifest,
    actor_consumes,
    get_backend,
    get_coupler,
    get_domain,
)

from .backends import effective_backend, resolve_backend_class
from .couplers import Channel, CouplerBuild, SolverCosim, build_coupled
from .errors import CompositionError, MaterialLossWarning
from .interactions import InteractionGraph, build_graph
from .islands import Island, partition
from .routing import Routing, route
from .world import REPLICATED_HOSTS, expand_indices, expand_shapes

__all__ = ["Channel", "CompositionError", "Island", "PhysicsProgram", "compose"]


@dataclass
class PhysicsProgram:
    """Everything the composer built, and how to drive it.

    Attributes:
        model: The model being stepped.
        arrangement: What it was built from.
        graph: The interaction graph.
        islands: The solver islands, in stepping order.
        channels: The coupling channels.
        routing: The pair lists.
        solver: The one solver the pipeline steps -- bare or coupled.
        collision: The main collision pipeline, or ``None``.
        contacts: Its contact buffer, or ``None``.
        schedule: The substep program entities append to.
        hooks: The phase lists over :attr:`schedule`, for entities.
        manifest: All of it as data.
        worlds: The world map when replicated.
    """

    model: newton.Model
    arrangement: ArrangementSpec
    graph: InteractionGraph
    islands: list[Island]
    channels: list[Channel]
    routing: Routing
    solver: Any
    collision: Any
    contacts: Any
    schedule: Schedule
    hooks: SubstepHooks
    manifest: PhysicsManifest
    worlds: Any = None
    warnings: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- lookup
    @property
    def coupled(self) -> bool:
        return len(self.islands) > 1

    @property
    def collide_every(self) -> int:
        return int(self.arrangement.collision.collide_every)

    def island_of(self, name: str) -> Island:
        for island in self.islands:
            if island.owns(name):
                return island
        raise KeyError(f"no island owns entity {name!r}; islands: {[(i.name, i.entities) for i in self.islands]}")

    def island(self, name: str) -> Island:
        for island in self.islands:
            if island.name == name:
                return island
        raise KeyError(f"no island {name!r}; has {[i.name for i in self.islands]}")

    def solver_for(self, name: str) -> Any:
        """The solver stepping entity ``name``: the entry's when coupled, the bare one otherwise."""
        island = self.island_of(name)
        if not self.coupled:
            return self.solver
        return self.solver.solver(island.name)

    def solver_of_island(self, name: str) -> Any:
        return self.solver.solver(name) if self.coupled else self.solver

    def view_for(self, name: str) -> Any:
        """The model view entity ``name`` is solved in (the model itself when bare)."""
        if not self.coupled:
            return self.model
        return self.solver.view(self.island_of(name).name)

    def solvers(self) -> dict[str, Any]:
        if not self.coupled:
            return {self.islands[0].name: self.solver}
        return {island.name: self.solver.solver(island.name) for island in self.islands}

    @property
    def mujoco_solver(self) -> Any | None:
        """The MuJoCo solver, bare or inside the coupler, for a contact probe. ``None`` when there is none."""
        for island in self.islands:
            if island.backend.solver == "mujoco":
                return self.solver_of_island(island.name)
        return None

    # --------------------------------------------------------------- control
    @property
    def hosts(self) -> dict[str, Any]:
        """The host-engine solvers by island name; empty when there are none.

        The seam through which a foreign engine's *own* state is driven —
        attached to the pipeline before anything settles, told to record its
        rest once everything has, and restored on reset. None of that can be an
        entity's job: an entity owns the particle arrays, and a foreign engine's
        rest is its own state, which no entity can snapshot for it. The soft
        domain did it anyway while it was the only host engine, and a cloth
        routed to that same engine silently got none of it.
        """
        return dict(getattr(self.solver, "hosts", {}) or {})

    def notify(self, flags: int) -> None:
        """Tell every solver that part of the model changed."""
        if not flags:
            return
        self.solver.notify_model_changed(int(flags))

    def reset(self, state: newton.State, *, flags: Any = None, world_mask: Any = None) -> None:
        """Reset every solver's internal buffers (warm starts, contact anchors)."""
        reset = getattr(self.solver, "reset", None)
        if reset is None:
            return
        try:
            if flags is None:
                reset(state, world_mask=world_mask)
            else:
                reset(state, world_mask=world_mask, flags=flags)
        except TypeError:
            reset(state)

    def apply(self, path: str, value: Any) -> str:
        """Apply one parameter change to the running program, returning its tier.

        Args:
            path: ``"backend:<island or domain>:<param>"`` or ``"coupling:<param>"``.
            value: The new value.

        Returns:
            The tier the change cost: ``"live"``, ``"recapture"`` or
            ``"rebuild"`` (meaning nothing was applied; the scene must be
            recompiled with the new spec).
        """
        kind, _, rest = path.partition(":")
        if kind == "coupling":
            info = get_coupler(self.arrangement.coupling.method)
            param = rest
            schema = info.schema or {}
            rule = schema.get(param)
            if rule is None or rule.apply == "rebuild":
                return REBUILD
            if rule.apply == "custom:proxy_mass_scale":
                return self._apply_proxy_mass_scale(float(value))
            return REBUILD
        if kind == "backend":
            target, _, param = rest.partition(":")
            island = next((i for i in self.islands if i.name == target or target in i.domains), None)
            if island is None:
                raise KeyError(f"no island or domain {target!r}")
            info = get_backend(island.backend.solver)
            rule = (info.schema or {}).get(param)
            if rule is None or rule.apply == "rebuild":
                return REBUILD
            solver = self.solver_of_island(island.name)
            if rule.apply.startswith("attr:"):
                setattr(solver, rule.apply.split(":", 1)[1], value)
                return rule.ui.tier
            if rule.apply == "custom:mujoco_opt":
                opt = getattr(getattr(solver, "mjw_model", None), "opt", None)
                if opt is None or not hasattr(opt, param):
                    return REBUILD
                setattr(opt, param, type(getattr(opt, param))(value))
                return RECAPTURE
            return REBUILD
        raise KeyError(f"unknown parameter path {path!r}")

    def _apply_proxy_mass_scale(self, value: float) -> str:
        mappings = getattr(self.solver, "_proxy_mappings", None)
        refresh = getattr(self.solver, "_apply_proxy_effective_masses", None)
        if not mappings or refresh is None:
            return REBUILD
        for mapping in mappings:
            if mapping.mass_scale > 1.0e5:
                continue  # a one-way proxy keeps its own scale
            mapping.mass_scale = float(value)
        refresh()
        return RECAPTURE

    # -------------------------------------------------------------- problems
    def problems(self) -> list[str]:
        """What the composition could not honour -- unrouted contacts, unresolved roles."""
        return [*self.graph.unresolved, *self.routing.unrouted, *self.warnings]

    def summary(self) -> str:
        lines = [f"physics: {self.manifest.coupler} over {len(self.islands)} island(s)"]
        for island in self.islands:
            lines.append(
                f"  island {island.name}: {island.backend.solver} x{island.backend.substeps} -> "
                f"{list(island.entities)} ({island.bodies.size} bodies, {island.joints.size} joints, "
                f"{island.particles.size} particles, {island.shapes.size or 'derived'} shapes)"
            )
        for channel in self.channels:
            pairs = "view pairs" if channel.pairs is None else f"{len(channel.pairs)} pairs"
            lines.append(
                f"  channel {channel.source} -> {channel.destination}: {channel.bodies.size} bodies "
                f"({channel.coupling}, {pairs}) from {list(channel.entities)}"
            )
        if self.routing.main_pairs is not None:
            lines.append(f"  main pipeline: {len(self.routing.main_pairs)} pairs")
        for problem in self.problems():
            lines.append(f"  ! {problem}")
        return "\n".join(lines)


def _channels(graph: InteractionGraph, islands: list[Island], worlds, model: newton.Model) -> list[Channel]:
    """Proxy channels: for every cross-island edge, the rigid side's bodies into the deformable side."""
    by_entity = {node.name: next((i for i in islands if i.owns(node.name)), None) for node in graph.nodes.values()}
    gathered: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in graph.edges:
        a, b = graph.nodes[edge.a], graph.nodes[edge.b]
        ia, ib = by_entity[a.name], by_entity[b.name]
        if ia is None or ib is None or ia is ib:
            continue
        # The deformable island is the destination; the other island's bodies are proxied.
        if ia.is_deformable and not ib.is_deformable:
            dest, src, src_node, hands_only = ia, ib, b, edge.hands_only
        elif ib.is_deformable and not ia.is_deformable:
            dest, src, src_node, hands_only = ib, ia, a, edge.hands_only and False
            hands_only = edge.hands_only
        else:
            continue  # two rigid islands or two deformable ones: ADMM's business, or unrouted
        bodies = src_node.hand_bodies if hands_only else src_node.bodies
        if bodies.size == 0:
            continue
        key = (src.name, dest.name, edge.coupling)
        slot = gathered.setdefault(key, {"bodies": [], "entities": set()})
        slot["bodies"].append(bodies)
        slot["entities"].add(src_node.name)

    shape_body = model.shape_body.numpy() if model.shape_count else np.zeros(0, np.int32)
    joint_child = model.joint_child.numpy() if model.joint_count else np.zeros(0, np.int32)
    out: list[Channel] = []
    for (src, dest, coupling), slot in gathered.items():
        bodies = np.unique(np.concatenate(slot["bodies"])).astype(np.int32)
        if worlds is not None and worlds.num_worlds > 1:
            bodies = expand_indices(bodies, worlds.bodies_per_world, worlds.num_worlds)
        shapes = np.flatnonzero(np.isin(shape_body, bodies)).astype(np.int32)
        joints = (
            np.flatnonzero(np.isin(joint_child, bodies)).astype(np.int32) if joint_child.size else np.zeros(0, np.int32)
        )
        out.append(
            Channel(
                source=src,
                destination=dest,
                bodies=bodies,
                shapes=shapes,
                joints=joints,
                coupling=coupling,
                entities=tuple(sorted(slot["entities"])),
            )
        )
    # A one-way and a two-way channel to the same destination are both allowed:
    # Newton takes one Proxy per mapping and the mass scales differ.
    out.sort(key=lambda c: (c.source, c.destination, c.coupling != "two_way"))
    return out


def _host_recipe(
    island: Island,
    host_channels: list[Channel],
    graph: InteractionGraph,
    model: newton.Model,
    arrangement: ArrangementSpec,
    worlds=None,
) -> dict:
    """The build recipe for one host-engine island, as plain JSON-able data.

    Everything a foreign engine needs beyond the model's own arrays: gravity,
    one actor record per owned entity (its contiguous particle range plus
    whatever the entity declares through ``host_actor``), and
    one mirror record per Newton shape the island's contact rules reach --
    channel shapes with the channel's coupling, static and world-fixed shapes
    as fixed boundary. Recorded verbatim into the manifest, which is how a
    repro bundle rebuilds the same foreign scene with the model and the
    manifest alone.

    **Replicated, every record carries the world it belongs to.** A foreign
    engine builds one scene from the whole model and has no world axis to pair
    contacts within, so the records alone are not enough: sixteen sponges and
    sixteen grippers at the same coordinates would be one pile colliding with
    itself. ``world`` is what lets the engine put each world somewhere of its
    own -- see :class:`~robosimstudio.engines.superdex.bridge.SuperdexBridge`,
    which adds a per-world translation on the way in and takes it off on the way
    out.

    Args:
        worlds: The stride map when the model is replicated, or ``None``. With
            it, one actor record per entity *per world*, and one mirror record
            per shape per world -- including a world-fixed shape, which is one
            shape in Newton and has to become one collider per world here
            because each world's copy sits at a different place in the host
            scene.
    """
    replicated = worlds is not None and worlds.num_worlds > 1
    world_count = int(worlds.num_worlds) if replicated else 1

    def spread(shapes) -> np.ndarray:
        """Template shape indices as every world's copy of them."""
        if not replicated:
            return np.asarray(shapes, dtype=np.int32)
        return expand_shapes(shapes, worlds.shapes_per_world, worlds.num_worlds, model.shape_count)

    def world_of(shape: int) -> int | None:
        """Which world a replicated shape index belongs to, or ``None`` if global.

        ``None`` is a shape that sits past every world's block -- added to the
        model after replication -- and is therefore one piece of geometry every
        world shares. Shared in Newton, where the worlds are superimposed; not
        shareable here, where they are not.
        """
        if not replicated:
            return 0
        limit = worlds.shapes_per_world * worlds.num_worlds
        return int(shape) // worlds.shapes_per_world if int(shape) < limit else None

    actors: list[dict] = []
    tet_rows = model.tet_indices.numpy().reshape(-1, 4) if model.tet_count else np.zeros((0, 4), np.int64)
    for node in island.nodes:
        particles = np.asarray(node.particles, dtype=np.int64)
        if particles.size == 0:
            continue
        start, count = int(particles.min()), int(particles.size)
        if not np.array_equal(particles, np.arange(start, start + count)):
            raise CompositionError(
                f"host island {island.name!r}: entity {node.name!r} owns a non-contiguous particle range, "
                "which a host actor cannot address as one block"
            )
        declare = getattr(node.entity, "host_actor", None)
        declared = declare() if callable(declare) else None
        if declared is None:
            # The entity said nothing, so the model says it instead: tetrahedra
            # are a volumetric body, triangles alone are a shell. Stated in
            # state-shape terms, never domain names, which the composer must not
            # know -- and kept as the default so an entity that declares nothing
            # is composed exactly as it was before a declaration existed. It can
            # only ever answer "tet or shell"; anything else has to be declared.
            owns_tets = bool(tet_rows.size) and bool(
                ((tet_rows.min(axis=1) >= start) & (tet_rows.max(axis=1) < start + count)).any()
            )
            declared = HostActor(kind="tet" if owns_tets else "shell")
        _warn_lost_material(node, declared, island)
        # The template's range, offset into each world's block. A replicated
        # model holds ``W`` copies of this entity's particles at that stride, and
        # an engine that built one actor from the first block would step one
        # world's sponge and leave the rest of the wave holding a rest pose.
        per_world = int(worlds.particles_per_world) if replicated else 0
        for world in range(world_count):
            record = declared.record(
                name=node.name if world_count == 1 else f"{node.name}#w{world}",
                particle_start=start + world * per_world,
                particle_count=count,
            )
            record["world"] = world
            record["entity"] = node.name
            actors.append(record)

    shape_body = model.shape_body.numpy() if model.shape_count else np.zeros(0, np.int32)
    shape_flags = model.shape_flags.numpy() if model.shape_count else np.zeros(0, np.int32)
    mirrors: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add_mirror(shape: int, *, two_way: bool, world: int | None = None) -> None:
        if world is None:
            world = world_of(shape)
        if world is None:
            # A shape past every world's block: one piece of geometry in Newton,
            # shared because the worlds are superimposed. Here they are not, so
            # every world needs its own collider cut from the same shape.
            for each in range(world_count):
                add_mirror(shape, two_way=two_way, world=each)
            return
        if (shape, world) in seen:
            return
        seen.add((shape, world))
        # Only geometry the Newton world itself would collide. A body's shape
        # list also carries sites and visual meshes -- a Panda hand brings a
        # grip-site marker sphere and three axis cylinders, one of them twenty
        # metres long -- and Newton's own pipelines skip them by this flag.
        # Mirrored, they are phantom colliders: the foam gets pressed by
        # markers no render draws, which measures as force and looks like
        # penetration.
        if not int(shape_flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            return
        mirrors.append(
            {
                "shape": int(shape),
                "body": int(shape_body[shape]),
                "two_way": bool(two_way),
                "world": int(world),
            }
        )

    for channel in host_channels:
        if channel.destination != island.name:
            continue
        for shape in channel.shapes:
            add_mirror(int(shape), two_way=channel.coupling == "two_way")

    # Statics the contact rules reach, and the world-fixed shapes the compiler
    # adds after every entity (the ground plane) -- the same set a Newton
    # deformable island is given, mirrored as fixed boundary instead.
    hosted = set(island.entities)
    for edge in graph.edges:
        if (edge.a in hosted) == (edge.b in hosted):
            continue
        other = graph.nodes[edge.b if edge.a in hosted else edge.a]
        if get_domain(other.domain).world_fixed:
            for shape in spread(other.shapes):
                add_mirror(int(shape), two_way=False)
    if model.shape_count:
        for shape in np.flatnonzero(shape_body < 0):
            add_mirror(int(shape), two_way=False)

    # Lists, not tuples, so the record IS its own JSON round trip.
    return {
        "island": island.name,
        "gravity": [0.0, 0.0, float(arrangement.gravity)],
        "control_hz": float(arrangement.control_hz),
        "num_worlds": world_count,
        "actors": actors,
        "mirrors": mirrors,
    }


def _warn_lost_material(node: Any, declared: HostActor, island: Any) -> None:
    """Say, once per actor, what this engine will never read.

    Two sides of one seam. **Engine side**: the backend declared ``consumes``
    and the actor states a parameter outside it -- Genesis reads five keys and a
    cloth's ``friction``, ``particle_radius`` and ``self_contact`` are not among
    them. **Entity side**: the spec's material states a field away from its
    default that never appears in the actor at all -- every shipped fabric
    states ``tri_ke`` and ``edge_ke``, none states ``shell_young``, and so
    packing tape and bath towelling reached both foreign engines at the same
    ``E = 1e5``. Measured; nothing said so; this says so.
    """
    engine = island.backend.solver
    info = get_backend(engine)
    # The kind's own builder first -- a shell builder never reads ``friction``,
    # and only it can say so -- the engine-wide declaration as the fallback.
    consumes = actor_consumes(engine, declared.kind) or info.consumes
    if consumes:
        unread = sorted(set(declared.params) - set(consumes))
        if unread:
            warnings.warn(
                f"{node.name!r}: {engine}'s {declared.kind} builder does not read {unread} of the parameters "
                f"its actor states (it reads {sorted(consumes)}), so those have no effect on this engine.",
                MaterialLossWarning,
                stacklevel=3,
            )
    material = getattr(getattr(node.entity, "spec", None), "material", None)
    if material is None or not dataclasses.is_dataclass(material):
        return
    default = type(material)()
    stated = [f.name for f in dataclasses.fields(material) if getattr(material, f.name) != getattr(default, f.name)]
    # Fields that travel some other way than the actor's params -- a tet body's
    # Young and Poisson go into ``model.tet_materials`` and the bridge inverts
    # them back out -- are not lost, and saying so sends a reader hunting for a
    # bug in a modulus that is in fact applied.
    carried = set(getattr(type(material), "via_model", ()))
    lost = [name for name in stated if name not in declared.params and name not in carried]
    if not lost:
        return
    unset = [f.name for f in dataclasses.fields(material) if getattr(material, f.name) is None]
    hint = f" Knobs this material offers and leaves unset: {unset}." if unset else ""
    warnings.warn(
        f"{node.name!r}: {len(lost)} stated material field(s) never reach {engine}: {lost}. "
        f"What the engine is handed: {sorted(declared.params)}.{hint}",
        MaterialLossWarning,
        stacklevel=3,
    )


def compose(
    model: newton.Model,
    parts: dict[str, Any],
    embodiment: Any | None,
    arrangement: ArrangementSpec,
    *,
    scene_statics=(),
    worlds=None,
    verbose: bool = False,
) -> PhysicsProgram:
    """Compose the solvers for a bound scene.

    Args:
        model: The model the solvers step -- the replicated one for a fleet.
        parts: Every built entity by name (statics, objects, deformables, extras).
        embodiment: The robot's entity, or ``None``.
        arrangement: The staged arrangement.
        scene_statics: Names of the cell's own statics (they play ``surface``).
        worlds: The world map when replicated; entities are bound to the template.
        verbose: Print the summary.

    Raises:
        CompositionError: When the arrangement cannot honour the declarations.
    """
    index_model = model
    graph = build_graph(index_model, parts, embodiment, arrangement, scene_statics=scene_statics)
    islands = partition(graph, arrangement, model, worlds=worlds)
    channels = _channels(graph, islands, worlds, model)

    # Host-engine islands leave the Newton story here. Their channels become
    # the foreign scene's kinematic mirror lists, their edges are hidden from
    # the routing (a pair a foreign engine detects is not a pair any Newton
    # pipeline may also detect -- and a deformable x host edge is a declared
    # limitation, not an unrouted one), and every predicate below -- the
    # coupler check, one_way, the explicit broad phase, the manifest's coupler
    # -- is computed over the Newton islands that remain.
    host_islands = [i for i in islands if i.is_host_engine]
    newton_islands = [i for i in islands if not i.is_host_engine]
    host_names = {i.name for i in host_islands}
    hosted_entities = {n.name for i in host_islands for n in i.nodes}
    host_channels = [c for c in channels if c.destination in host_names]
    channels = [c for c in channels if c.destination not in host_names and c.source not in host_names]
    if host_islands and worlds is not None and worlds.num_worlds > 1:
        # Refused by *engine*, not by the fact of being hosted. A foreign engine
        # builds one scene from the whole model and has no world axis to pair
        # contacts within, so the worlds have to be laid out somewhere of their
        # own inside it -- which is a property of the bridge and is what
        # ``REPLICATED_HOSTS`` asserts about the ones that do it.
        unready = sorted({i.backend.solver for i in host_islands if i.backend.solver not in REPLICATED_HOSTS})
        if unready:
            raise CompositionError(
                f"host-engine islands do not replicate: {unready} cannot be built at "
                f"{worlds.num_worlds} worlds. Their bridge would put every world's actors at the same "
                "coordinates in one scene, where they would collide with each other. Run single-world, "
                "or stage a Newton backend for those domains."
            )

    if len(newton_islands) > 1:
        try:
            get_coupler(arrangement.coupling.method)
        except KeyError as exc:
            raise CompositionError(str(exc)) from exc

    world_shapes = (
        np.flatnonzero(model.shape_body.numpy() < 0).astype(np.int32) if model.shape_count else np.zeros(0, np.int32)
    )
    route_graph = graph
    if host_islands:
        route_graph = InteractionGraph(
            nodes=graph.nodes,
            edges=[e for e in graph.edges if e.a not in hosted_entities and e.b not in hosted_entities],
            self_contact=graph.self_contact,
            unresolved=graph.unresolved,
        )
    routing = route(route_graph, newton_islands, channels, model, world_shapes=world_shapes, worlds=worlds)
    for channel in channels:
        channel.pairs = routing.channel_pairs.get((channel.source, channel.destination))

    collision = None
    contacts = None
    spec = arrangement.collision
    one_way = len(newton_islands) > 1 and arrangement.coupling.method == "one_way"
    if one_way:
        # The one-way coupler collides between its two solves, on the poses the
        # rigid solver has just produced, so it owns the pipeline: an executor
        # colliding before the solve would hand the deformable last substep's arm.
        routing.main_pairs = None
    if routing.main_pairs is not None:
        kwargs: dict[str, Any] = {
            "soft_contact_margin": float(spec.soft_contact_margin),
            "enable_rigid_soft_full_surface_contact": bool(spec.full_surface_contact),
            "contact_matching": spec.contact_matching,
        }
        if spec.rigid_contact_max > 0:
            model.rigid_contact_max = int(spec.rigid_contact_max)
            kwargs["rigid_contact_max"] = int(spec.rigid_contact_max)
        if spec.broad_phase == "explicit" and len(newton_islands) > 1:
            # An explicit list is exact and honours the builder's filters (it was
            # cut from the model's own table). A lone island keeps Newton's own
            # broad phase over the full table, which is the same set.
            kwargs["broad_phase"] = "explicit"
            if len(routing.main_pairs):
                kwargs["shape_pairs_filtered"] = wp.array(routing.main_pairs, dtype=wp.vec2i, device=model.device)
        elif spec.broad_phase != "explicit":
            kwargs["broad_phase"] = spec.broad_phase
        collision = newton.CollisionPipeline(model, **kwargs)
        contacts = collision.contacts()

    newton_solver = None
    if newton_islands:
        ctx = CouplerBuild(
            model=model,
            arrangement=arrangement,
            islands=newton_islands,
            channels=channels,
            spec=arrangement.coupling,
            collision=spec,
        )
        newton_solver = build_coupled(ctx)
        if one_way:
            contacts = newton_solver.contacts
    host_solvers: dict[str, Any] = {}
    host_recipes: dict[str, dict] = {}
    for island in host_islands:
        recipe = _host_recipe(island, host_channels, graph, model, arrangement, worlds=worlds)
        host_recipes[island.name] = recipe
        cls = resolve_backend_class(island.backend)
        params = effective_backend(island.backend, has_particles=True).kwargs()
        host_solvers[island.name] = cls(model, recipe=recipe, substeps=int(island.backend.substeps), **params)
    solver = SolverCosim(model, newton_solver, host_solvers) if host_solvers else newton_solver
    prepare = getattr(solver, "prepare_contacts", None)
    if prepare is not None and contacts is not None:
        prepare(contacts)

    schedule = Schedule()
    hooks = SubstepHooks(schedule)
    coupled = len(islands) > 1 or bool(host_solvers)
    entry_states = hasattr(solver, "entry_state")
    # ``solver(name)`` is the coupled solvers' accessor; a coupler kind a task
    # registered need not offer one, and a bare solver has nothing to look up.
    lookup = getattr(solver, "solver", None) if coupled else None
    for island in islands:
        inner = lookup(island.name) if callable(lookup) else (None if coupled else solver)
        if inner is None:
            continue
        if getattr(inner, "particle_enable_self_contact", False) and hasattr(inner, "rebuild_bvh"):
            # Once per control step, outside the graph: rebuilding a BVH
            # reallocates, which a replay cannot do. On the entry's own state
            # when the coupler keeps one -- the detector was built over the
            # view's particles -- and on the shared state otherwise.
            def rebuild(state, dt, _inner=inner, _name=island.name, _views=entry_states) -> None:
                del dt
                _inner.rebuild_bvh(solver.entry_state(_name) if _views else state)

            schedule.add(Stage(name=f"rebuild_bvh:{island.name}", fn=rebuild, phase="before"))
    for name, host in host_solvers.items():
        # The co-simulation tick: eager, once per control step, after the
        # substeps and before the batched read-back -- ``after`` rather than
        # ``before`` because VBD's velocity write-back is not gated on the
        # particle flags and would zero a foreign island's ``particle_qd``
        # between an earlier write and the snapshot.
        schedule.add(
            Stage(
                name=f"cosim:{name}",
                fn=host.host_tick,
                phase="after",
                reads=frozenset({"body_q"}),
                writes=frozenset({"particle_q", "particle_qd"}),
            )
        )
        force_stage = getattr(host, "rigid_force_stage", None)
        if callable(force_stage):
            # Wrench feedback: a device kernel every substep, inside the graph,
            # reading the persistent array the tick rewrote outside it.
            schedule.add(
                Stage(name=f"cosim_force:{name}", fn=force_stage, phase="rigid_force", writes=frozenset({"body_f"}))
            )
    mirror = any(island.maximal_coordinates for island in islands)
    if mirror:

        def recover(state, dt, _model=model) -> None:
            del dt
            newton.eval_ik(_model, state, state.joint_q, state.joint_qd)

        schedule.add(
            Stage(
                name="mirror_joint_ik",
                fn=recover,
                phase="mirror",
                reads=frozenset({"body_q"}),
                writes=frozenset({"joint_q", "joint_qd"}),
            )
        )

    if one_way:
        # The one-way coupler mirrors nothing: the deformable simply collides
        # against the poses the rigid solver produced. Dropping the channels
        # keeps the program honest about what was built.
        channels = []
    manifest = PhysicsManifest(
        arrangement=arrangement,
        islands=tuple(
            IslandRecord(
                name=i.name,
                backend=effective_backend(i.backend, has_particles=bool(i.particles.size)),
                # A host island records its backend *alias*, never a class
                # path: a repro bundle may not name the exporting project, and
                # the bundle runner keeps a self-contained twin per alias.
                solver_class=i.backend.solver if i.is_host_engine else i.solver_class,
                domains=i.domains,
                entities=i.entities,
                bodies=tuple(int(v) for v in i.bodies),
                joints=tuple(int(v) for v in i.joints),
                particles=tuple(int(v) for v in i.particles),
                shapes=tuple(int(v) for v in i.shapes),
                host_engine=i.is_host_engine,
                host_recipe=host_recipes.get(i.name),
            )
            for i in islands
        ),
        channels=tuple(
            ChannelRecord(
                source=c.source,
                destination=c.destination,
                bodies=tuple(int(v) for v in c.bodies),
                coupling=c.coupling,
                mass_scale=float(
                    arrangement.coupling.get("mass_scale", 1.0)
                    if c.coupling == "two_way"
                    else arrangement.coupling.get("one_way_mass_scale", 1.0e6)
                ),
                mode=str(arrangement.coupling.get("mode", "staggered")),
                collide_interval=int(arrangement.coupling.get("collide_interval", 1)),
                pairs=None if c.pairs is None else tuple((int(a), int(b)) for a, b in c.pairs),
                entities=c.entities,
            )
            for c in channels
        ),
        coupler="none" if len(newton_islands) <= 1 else arrangement.coupling.method,
        main_pairs=None if routing.main_pairs is None else tuple((int(a), int(b)) for a, b in routing.main_pairs),
        world_count=int(worlds.num_worlds) if worlds is not None else 1,
        mirror_joint_ik=mirror,
        stages=schedule.names(),
    )
    program = PhysicsProgram(
        model=model,
        arrangement=arrangement,
        graph=graph,
        islands=islands,
        channels=channels,
        routing=routing,
        solver=solver,
        collision=collision,
        contacts=contacts,
        schedule=schedule,
        hooks=hooks,
        manifest=manifest,
        worlds=worlds,
    )
    hooks.program = program
    if verbose:
        print(program.summary())
    return program
