# SPDX-License-Identifier: Apache-2.0
"""Partition the interaction graph into solver islands.

An island is one Newton solver over a compacted view of the shared model. Every
non-static node resolves a backend -- its own declared override, else the
arrangement's entry for its domain, else the ``"*"`` fallback -- and nodes
whose backends are *equal* share an island. That equality is the whole
mechanism by which a towel and a rope end up in one VBD: the arrangement names
``backend("vbd", ...)`` for ``"cloth"`` and ``"rod"`` with the same parameters,
and nothing else has to be said.

Statics own no dynamics. Their **shapes** are given to every deformable island
there is, so a rod can rest on the table it was laid on and a heap beside it
on the same table (an entry that lists shapes sees only those; one that lists
none sees its bodies' shapes plus every world-fixed shape -- Newton's rule, and
what lets the robot keep seeing the table through the coupled solver's
filter). A welded static's bodies and joints go to the first island that can
carry bodies, so no joint is left unowned -- an unowned joint is disabled in
every view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import newton
import numpy as np

from robosimstudio.specs.physics import BACKENDS, ArrangementSpec, BackendSpec, get_backend, get_domain

from .backends import solver_class_path
from .devices import on_device
from .errors import CompositionError
from .interactions import InteractionGraph, Node
from .world import expand_indices, expand_shapes

__all__ = ["Island", "partition"]


@dataclass
class Island:
    """One solver over the state of some entities.

    Attributes:
        name: The coupler's entry name.
        backend: What solves it.
        solver_class: The resolved dotted path.
        nodes: The entities it owns.
        bodies, joints, particles, shapes: Global model indices, world-expanded.
            ``shapes`` empty means "derived" (the rigid island's rule).
        extra_shapes: Static and world-fixed shapes this island was also given.
    """

    name: str
    backend: BackendSpec
    solver_class: str
    nodes: list[Node] = field(default_factory=list)
    bodies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    joints: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    particles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    extra_shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    @property
    def domains(self) -> tuple[str, ...]:
        out: list[str] = []
        for node in self.nodes:
            if node.domain not in out:
                out.append(node.domain)
        return tuple(out)

    @property
    def entities(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def is_robot(self) -> bool:
        return any("robot" in node.roles for node in self.nodes)

    @property
    def is_deformable(self) -> bool:
        """Whether this island is the *soft side* of a coupling.

        Two ways to be: a node here plays the ``deformables`` role, or the
        island's backend can integrate particles at all. The second is what was
        missing. ``compose._channels`` decides proxy direction by this predicate,
        and with roles alone a VBD island holding only a free rigid body -- role
        ``objects`` -- read as a *rigid* island: no channel was built, the body
        was orphaned, and a cube in such an island sat frozen at its spawn pose
        with ``body_qd`` exactly zero. Keying on the solver's registered
        capability is what lets ``GENERIC`` put everything but the robot on VBD
        and have a cube fall beside a rope beside an arm.

        Registered knowledge, not a literal list of aliases: a dotted-path
        solver nobody registered has an empty ``supports`` and falls back to the
        roles, which is the behaviour this had before.
        """
        if any(get_domain(node.domain).role == "deformables" for node in self.nodes):
            return True
        return "particles" in get_backend(self.backend.solver).supports

    @property
    def is_host_engine(self) -> bool:
        """Whether this island's backend is a foreign engine stepped on the host.

        Registered knowledge, exactly as :attr:`is_deformable` reads
        ``supports``: the composer splits these islands out before the Newton
        couplers, gives their solver a no-op ``step`` inside the graph, and
        schedules the real work as an eager per-control-step stage. A dotted
        path nobody registered is not one -- a host engine has to say so.
        """
        return get_backend(self.backend.solver).host_engine

    @property
    def maximal_coordinates(self) -> bool:
        """Whether ``joint_q`` has to be mirrored from ``body_q`` after each step."""
        if self.joints.size == 0:
            return False
        info = get_backend(self.backend.solver)
        if info.alias in BACKENDS:
            return info.maximal_coordinates  # a registered fact
        return "." in info.alias  # an anonymous dotted solver is assumed maximal

    def owns(self, name: str) -> bool:
        return any(node.name == name for node in self.nodes)

    def uses_native_contacts(self) -> bool:
        """Whether this island's solver generates its own contacts (MuJoCo's default)."""
        return get_backend(self.backend.solver).native_contacts and bool(self.backend.get("use_mujoco_contacts", True))


def _alias_name(spec: BackendSpec) -> str:
    alias = spec.solver
    if "." in alias:
        tail = alias.rsplit(".", 1)[-1]
        alias = tail.removeprefix("Solver").lower() or tail.lower()
    return alias


def partition(
    graph: InteractionGraph,
    arrangement: ArrangementSpec,
    model: newton.Model,
    *,
    worlds=None,
) -> list[Island]:
    """Islands for the graph under the arrangement, the robot's first.

    Args:
        graph: The interaction graph over the template model.
        arrangement: Which backend serves which domain.
        model: The model the solvers will step (the replicated one for a fleet),
            for the shape count.
        worlds: The :class:`~robosimstudio.core.worlds.WorldMap` when replicated.

    Raises:
        CompositionError: When a non-static entity has no backend, or its
            backend is known not to support the state its domain carries.
    """
    groups: dict[BackendSpec, Island] = {}
    statics: list[Node] = []
    num_worlds = int(worlds.num_worlds) if worlds is not None else 1

    for node in graph.nodes.values():
        if get_domain(node.domain).world_fixed:
            statics.append(node)
            continue
        spec = node.sim.backend or arrangement.backend_for(node.domain)
        if spec is None:
            raise CompositionError(
                f"entity {node.name!r} is a {node.domain!r} and the arrangement names no backend for that "
                f"domain (it serves {list(arrangement.domains())}). Add a backend for {node.domain!r}, or a "
                "'*' fallback."
            )
        # Before grouping, so two entities whose specs are equal stay in one
        # island once both carry the device's kwargs.
        spec = on_device(
            spec,
            model.device,
            owner=node.name,
            domain=node.domain,
            declared=node.sim.backend is not None,
            num_worlds=num_worlds,
        )
        info = get_backend(spec.solver)
        if info.supports:
            needed = [k for k in get_domain(node.domain).state if k != "shapes" and node.has(k)]
            missing = [k for k in needed if k not in info.supports]
            if missing:
                raise CompositionError(
                    f"entity {node.name!r} ({node.domain}) owns {needed}, but backend {spec.solver!r} "
                    f"integrates only {list(info.supports)} -- it cannot own {missing}."
                )
        island = groups.get(spec)
        if island is None:
            island = Island(name=_alias_name(spec), backend=spec, solver_class=solver_class_path(spec))
            groups[spec] = island
        island.nodes.append(node)

    islands = list(groups.values())
    if not islands:
        # A model built by hand, with no entity layer over it -- a bench rig, a
        # material strip. One island from the fallback backend, owning nothing
        # explicitly, which for a single solver means the whole model.
        spec = arrangement.backend_for("*")
        if spec is None:
            raise CompositionError(
                "the scene declares no entity with dynamics and the arrangement names no '*' backend, "
                "so there is nothing to solve and nothing to solve it with"
            )
        spec = on_device(spec, model.device, owner="the model", domain="*", num_worlds=num_worlds)
        if not (model.body_count or model.joint_count or model.particle_count):
            # Distinct from the case above, and worth its own sentence: there the
            # arrangement was short an entry, here the *scene* has nothing in it
            # that moves. Handing that model to a solver fails inside the solver
            # -- MuJoCo says "the model must have at least one joint to be able
            # to convert it", which is true and is about a conversion, not about
            # a scene of nothing but furniture. Assembling the scenery first and
            # looking at it is an ordinary thing to try.
            names = sorted(n.name for n in graph.nodes.values())
            raise CompositionError(
                f"this scene has nothing that moves: {names or 'no entities at all'} -- every one of them is "
                "welded to the world, so there are no bodies, no joints and no particles to integrate. Add "
                "something dynamic: rss.box, rss.sheet, rss.sponge, or a robot. A scene of only scenery cannot "
                "be built at all today, not even to look at -- the solver is constructed during the compile, "
                "so there is no argument to build() that skips it."
            )
        return [Island(name=_alias_name(spec), backend=spec, solver_class=solver_class_path(spec))]

    # The robot's island steps first: it is the proxy *source* in every channel.
    islands.sort(key=lambda island: (not island.is_robot, island.name))
    seen: dict[str, int] = {}
    for island in islands:
        count = seen.get(island.name, 0)
        seen[island.name] = count + 1
        if count:
            island.name = f"{island.name}_{count + 1}"

    # Ownership: template indices first, expanded per world at the end.
    for island in islands:
        island.bodies = np.unique(np.concatenate([n.bodies for n in island.nodes] or [np.zeros(0, np.int32)])).astype(
            np.int32
        )
        island.joints = np.unique(np.concatenate([n.joints for n in island.nodes] or [np.zeros(0, np.int32)])).astype(
            np.int32
        )
        island.particles = np.unique(
            np.concatenate([n.particles for n in island.nodes] or [np.zeros(0, np.int32)])
        ).astype(np.int32)

    # Statics: shapes to every deformable island; bodies/joints to the first
    # body-carrying island. A host-engine island is skipped for the shapes even
    # though it is deformable -- its statics are mirrored into the foreign scene
    # from its own recipe, and a VBD island beside it still needs the table.
    # *Every* deformable island rather than the first: an entry that lists
    # shapes sees only those, so a second deformable island (a heap on its own
    # solver beside a towel on VBD) given none would rest on nothing.
    deformables = [i for i in islands if i.is_deformable and not i.is_host_engine]
    body_carrier = next((i for i in islands if i.is_robot), None) or next(
        (i for i in islands if "bodies" in (get_backend(i.backend.solver).supports or ("bodies",))), islands[0]
    )
    static_shapes = [n.shapes for n in statics] or [np.zeros(0, np.int32)]
    static_bodies = [n.bodies for n in statics] or [np.zeros(0, np.int32)]
    static_joints = [n.joints for n in statics] or [np.zeros(0, np.int32)]
    if statics:
        body_carrier.bodies = np.unique(np.concatenate([body_carrier.bodies, *static_bodies])).astype(np.int32)
        body_carrier.joints = np.unique(np.concatenate([body_carrier.joints, *static_joints])).astype(np.int32)
    if len(islands) > 1:
        for deformable in deformables:
            own = [n.shapes for n in deformable.nodes] or [np.zeros(0, np.int32)]
            deformable.extra_shapes = np.unique(np.concatenate(static_shapes)).astype(np.int32)
            deformable.shapes = np.unique(np.concatenate([*own, deformable.extra_shapes])).astype(np.int32)

    if worlds is not None and worlds.num_worlds > 1:
        for island in islands:
            island.bodies = expand_indices(island.bodies, worlds.bodies_per_world, worlds.num_worlds)
            island.joints = expand_indices(island.joints, worlds.joints_per_world, worlds.num_worlds)
            island.particles = expand_indices(island.particles, worlds.particles_per_world, worlds.num_worlds)
            if island.shapes.size:
                island.shapes = expand_shapes(
                    island.shapes, worlds.shapes_per_world, worlds.num_worlds, model.shape_count
                )
            if island.extra_shapes.size:
                island.extra_shapes = expand_shapes(
                    island.extra_shapes, worlds.shapes_per_world, worlds.num_worlds, model.shape_count
                )
    elif deformables and len(islands) > 1:
        # World-fixed shapes the compiler adds after every entity -- the ground
        # plane -- sit past every entity's range and must be visible to the rod.
        on_world = (
            np.flatnonzero(model.shape_body.numpy() < 0).astype(np.int32)
            if model.shape_count
            else np.zeros(0, np.int32)
        )
        for deformable in deformables:
            deformable.shapes = np.unique(np.concatenate([deformable.shapes, on_world])).astype(np.int32)
            deformable.extra_shapes = np.unique(np.concatenate([deformable.extra_shapes, on_world])).astype(np.int32)
    return islands
