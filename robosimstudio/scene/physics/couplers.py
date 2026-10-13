# SPDX-License-Identifier: Apache-2.0
"""Build the coupled solver for a set of islands and channels.

The builders are registered by kind under the same names the
:mod:`robosimstudio.specs.physics` coupler registry declares, so a task may
register a new algorithm beside its arrangement:
``register_coupler_builder("mine", build)`` with
``build(ctx: CouplerBuild) -> SolverBase``. Two ship here: Newton's proxy
coupler (``"proxy"``), which mirrors one island's bodies into another as
mass-scaled virtual proxies, and its ADMM coupler (``"admm"``), which solves
cross-island contacts as constraints. One island needs no coupler at all and
gets the bare solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.registry import Registry, same_source
from robosimstudio.specs.physics import ArrangementSpec, CollisionSpec, CouplerSpec

from .backends import build_solver
from .errors import CompositionError
from .islands import Island

__all__ = [
    "COUPLER_BUILDERS",
    "Channel",
    "CouplerBuild",
    "SolverCosim",
    "SolverOneWay",
    "build_coupled",
    "register_coupler_builder",
]


@dataclass
class Channel:
    """One coupling channel: some bodies of ``source`` proxied into ``destination``.

    Attributes:
        source: Island that owns the bodies.
        destination: Island that sees them as proxies.
        bodies: Global body indices (world-expanded).
        shapes: The shapes on those bodies.
        joints: Joints among those bodies, kept as proxy joints when asked.
        coupling: ``"two_way"`` or ``"one_way"``.
        entities: Which entities the bodies came from.
        pairs: Explicit shape pairs for the channel's pipeline, or ``None``.
    """

    source: str
    destination: str
    bodies: np.ndarray
    shapes: np.ndarray
    joints: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    coupling: str = "two_way"
    entities: tuple[str, ...] = ()
    pairs: np.ndarray | None = None


@dataclass
class CouplerBuild:
    """Everything a coupler builder is handed."""

    model: newton.Model
    arrangement: ArrangementSpec
    islands: list[Island]
    channels: list[Channel]
    spec: CouplerSpec
    collision: CollisionSpec

    def entries(self, entry_cls) -> list:
        """One ``SolverCoupled.Entry`` per island."""
        out = []
        for island in self.islands:
            out.append(
                entry_cls(
                    name=island.name,
                    solver=(
                        lambda view, spec=island.backend, particles=bool(island.particles.size): build_solver(
                            spec, view, has_particles=particles
                        )
                    ),
                    bodies=[int(b) for b in island.bodies],
                    joints=[int(j) for j in island.joints],
                    particles=[int(p) for p in island.particles],
                    shapes=[int(s) for s in island.shapes],
                    substeps=int(island.backend.substeps),
                )
            )
        return out


COUPLER_BUILDERS: Registry[Callable[[CouplerBuild], Any]] = Registry("coupler builder", same=same_source)
"""Coupler kind -> builder. Populated below and by tasks."""


def register_coupler_builder(kind: str, build: Callable[[CouplerBuild], Any]) -> None:
    """Register the builder for a coupler kind. Identical re-registration is accepted.

    Two tasks each shipping their own copy of one coupler is the case that has
    to keep working, so an identical copy is accepted and a genuine clash is an
    error naming both origins.

    Raises:
        ValueError: If ``kind`` is taken by a different builder.
    """
    COUPLER_BUILDERS.register(kind, build, origin=build.__module__)


def _pipeline_factory(ctx: CouplerBuild, pairs: np.ndarray | None, *, particles: bool = False):
    """The collision pipeline one channel's destination solve is stepped with.

    Args:
        ctx: What the coupler is being built from.
        pairs: Explicit shape pairs, or ``None`` for Newton's own broad phase.
        particles: Whether the destination owns particles. **An explicit pair
            list is refused for one that does**, and the reason is measured: a
            pair list is a list of *shape* pairs, and handing one to the
            pipeline restricts the soft pass with it -- a towel dropped onto a
            cord in the same island fell straight through, cord unmoved, with
            every ``COLLIDE_PARTICLES`` flag set and 1825 rigid pairs routed.
            The view has already hidden everything the entry does not own, so
            the broad phase over it sees the same set the list was cut from and
            the only thing given up is the builder's own filters -- which apply
            to rigid pairs, which is what the list was for.
    """
    collision = ctx.collision
    matching = ctx.spec.get("contact_matching", collision.contact_matching)
    explicit = pairs is not None and len(pairs) and not particles

    def make(view):
        kwargs: dict[str, Any] = {
            "soft_contact_margin": float(collision.soft_contact_margin),
            "enable_rigid_soft_full_surface_contact": bool(collision.full_surface_contact),
            "contact_matching": matching,
        }
        if explicit:
            kwargs["broad_phase"] = "explicit"
            kwargs["shape_pairs_filtered"] = wp.array(
                np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=view.device
            )
        if collision.rigid_contact_max > 0:
            kwargs["rigid_contact_max"] = int(collision.rigid_contact_max)
        return newton.CollisionPipeline(view, **kwargs)

    return make


def build_proxy(ctx: CouplerBuild) -> Any:
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

    if len(ctx.islands) > 2:
        raise CompositionError(
            f"the proxy coupler couples at most two islands and this scene has {len(ctx.islands)} "
            f"({[i.name for i in ctx.islands]}); stage coupler('admm') or merge backends so two domains share one."
        )
    params = ctx.spec.kwargs()
    mode = str(params.get("mode", "staggered"))
    mass_scale = float(params.get("mass_scale", 1.0))
    one_way_scale = float(params.get("one_way_mass_scale", 1.0e6))
    interval = int(params.get("collide_interval", 1))
    relaxation = float(params.get("proxy_relaxation", 1.0))
    relaxation_mode = str(params.get("proxy_relaxation_mode", "fixed"))
    proxy_joints = bool(params.get("proxy_joints", False))

    by_name = {island.name: island for island in ctx.islands}
    proxies = []
    for channel in ctx.channels:
        scale = mass_scale if channel.coupling == "two_way" else one_way_scale
        destination = by_name.get(channel.destination)
        particles = bool(destination is not None and destination.particles.size)
        proxies.append(
            SolverCoupledProxy.Proxy(
                source=channel.source,
                destination=channel.destination,
                bodies=[int(b) for b in channel.bodies],
                joints=[int(j) for j in channel.joints] if proxy_joints else (),
                mass_scale=scale,
                mode=mode,
                proxy_relaxation=relaxation,
                proxy_relaxation_mode=relaxation_mode,
                collision_pipeline=_pipeline_factory(ctx, channel.pairs, particles=particles),
                collide_interval=interval,
            )
        )
    return SolverCoupledProxy(
        model=ctx.model,
        entries=ctx.entries(SolverCoupled.Entry),
        coupling=SolverCoupledProxy.Config(proxies=proxies, iterations=int(params.get("iterations", 1))),
    )


def build_admm(ctx: CouplerBuild) -> Any:
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

    params = ctx.spec.kwargs()
    pairs = [SolverCoupledADMM.ContactPair(source=c.source, destination=c.destination) for c in ctx.channels]
    if not pairs:
        pairs = SolverCoupledADMM.auto_detect_contact_pairs(ctx.entries(SolverCoupled.Entry))
    known = {
        "iterations",
        "rho",
        "gamma",
        "baumgarte",
        "joint_stiffness",
        "joint_damping",
        "joint_angular_stiffness",
        "joint_angular_damping",
        "joint_proximal_bodies",
        "joint_proximal_mass_scale",
        "rigid_contact_matching",
        "contact_matching_pos_threshold",
        "contact_matching_normal_dot_threshold",
        "contact_matching_force_scale",
    }
    config = SolverCoupledADMM.Config(**{k: v for k, v in params.items() if k in known}, contact_pairs=pairs)
    return SolverCoupledADMM(model=ctx.model, entries=ctx.entries(SolverCoupled.Entry), coupling=config)


class SolverOneWay:
    """Two solvers over one model, the rigid one leading: a kinematic mirror, no feedback.

    One solver with the standard ``step`` signature: the rigid solver steps with
    its own contacts, the collision pipeline runs over its new body poses, and
    the deformable solver steps with ``integrate_with_external_rigid_solver=True``
    -- the robot is a moving boundary it never pushes back on.

    VBD reads the new poses off ``state_out`` and the previous ones off
    ``state_in``, and its friction is the particle's slip against the difference:
    how far the pad under a particle moved this substep. So ``state_in`` keeps
    the previous poses for the solve, and gets the new ones only while the
    collision pipeline runs, which must find this substep's arm. Handing VBD the
    new pose twice made every finger stand still to friction: a closed jaw
    lifting a towel's edge dragged it back down, and kept 0 of the 15 particles
    it pinched; with the motion it keeps all 15 through a 100 mm lift.

    Why it exists beside the proxy coupler rather than instead of it: Newton's
    proxy harvest consumes per-particle contact records only, and refuses
    full-surface (edge/face) soft contacts outright -- which are what let a pad
    narrower than a mesh cell hold a sheet. A cloth a jaw must pinch therefore
    couples one way with full-surface contact, or two ways without it; the
    arrangement says which, and both are data.

    Args:
        model: The shared model.
        rigid: The rigid solver, built over the whole model.
        soft: The deformable solver, built over the whole model in external-rigid mode.
        collision: The pipeline the deformable solver's contacts come from.
        names: ``(rigid island name, deformable island name)``.
    """

    def __init__(self, model, rigid, soft, collision, names: tuple[str, str]) -> None:
        self.model = model
        self.rigid = rigid
        self.soft = soft
        self.collision = collision
        self.contacts = collision.contacts()
        self.names = names
        self.device = model.device
        # Allocated here, not on the first step: a step may be captured into a graph.
        self._body_q_prev = wp.clone(model.body_q) if model.body_count else None

    def solver(self, name: str):
        return {self.names[0]: self.rigid, self.names[1]: self.soft}[name]

    def view(self, name: str):
        del name
        return self.model

    def entry_names(self) -> tuple[str, ...]:
        return tuple(self.names)

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        del contacts
        if self._body_q_prev is None:
            self.rigid.step(state_in, state_out, control, None, dt)
            self.collision.collide(state_in, self.contacts)
            self.soft.step(state_in, state_out, control, self.contacts, dt)
            return
        wp.copy(self._body_q_prev, state_in.body_q)
        self.rigid.step(state_in, state_out, control, None, dt)
        # The pipeline collides ``state_in``: this substep's arm against the particles where they start.
        wp.copy(state_in.body_q, state_out.body_q)
        self.collision.collide(state_in, self.contacts)
        # And VBD's friction is the pad's motion from ``state_in`` to ``state_out``, so put it back.
        wp.copy(state_in.body_q, self._body_q_prev)
        self.soft.step(state_in, state_out, control, self.contacts, dt)

    def notify_model_changed(self, flags) -> None:
        for inner in (self.rigid, self.soft):
            inner.notify_model_changed(int(flags))

    def reset(self, state, world_mask=None, flags=None) -> None:
        for inner in (self.rigid, self.soft):
            reset = getattr(inner, "reset", None)
            if reset is None:
                continue
            try:
                reset(state, world_mask=world_mask, flags=flags) if flags is not None else reset(
                    state, world_mask=world_mask
                )
            except TypeError:
                reset(state)


def build_one_way(ctx: CouplerBuild) -> Any:
    """A rigid island leading a deformable one, with the deformable island's contacts from a full pipeline."""
    if len(ctx.islands) != 2:
        raise CompositionError(
            f"the one_way coupler joins exactly one rigid island to one deformable island, got "
            f"{[i.name for i in ctx.islands]}"
        )
    deformable = next((i for i in ctx.islands if i.is_deformable), None)
    rigid = next((i for i in ctx.islands if i is not deformable), None)
    if deformable is None or rigid is None or rigid.is_deformable:
        raise CompositionError("the one_way coupler needs one rigid island and one deformable island")
    rigid_solver = build_solver(rigid.backend, ctx.model, has_particles=False)
    soft_spec = deformable.backend.with_(integrate_with_external_rigid_solver=True)
    soft_solver = build_solver(soft_spec, ctx.model, has_particles=bool(deformable.particles.size))
    collision = ctx.collision
    kwargs: dict[str, Any] = {
        "soft_contact_margin": float(collision.soft_contact_margin),
        "enable_rigid_soft_full_surface_contact": bool(collision.full_surface_contact),
    }
    if collision.contact_matching != "disabled":
        kwargs["contact_matching"] = collision.contact_matching
    pipeline = newton.CollisionPipeline(ctx.model, **kwargs)
    return SolverOneWay(ctx.model, rigid_solver, soft_solver, pipeline, (rigid.name, deformable.name))


class SolverCosim:
    """The Newton side of a scene, plus its host-engine islands, as one solver.

    Not a coupler in the arrangement's vocabulary and never staged as one: the
    composer wraps automatically whenever a host-engine island exists, so a
    scene gains a sponge without its staged ``proxy``/``one_way`` coupling
    changing meaning. Inside the captured graph this is exactly the Newton
    solver it wraps -- a host solver's ``step`` is a no-op by contract -- and
    the host engines advance in the eager ``after`` stages the composer
    schedules beside it.

    Args:
        model: The shared model.
        newton_solver: The composed Newton-side solver (bare or coupled), or
            ``None`` when every island is a host island.
        hosts: Host solvers by island name.
    """

    def __init__(self, model, newton_solver, hosts: dict[str, Any]) -> None:
        self.model = model
        self.newton_solver = newton_solver
        self.hosts = dict(hosts)
        self.device = getattr(model, "device", None)

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        if self.newton_solver is not None:
            self.newton_solver.step(state_in, state_out, control, contacts, dt)

    def notify_model_changed(self, flags) -> None:
        for inner in (self.newton_solver, *self.hosts.values()):
            if inner is not None:
                inner.notify_model_changed(int(flags))

    def reset(self, state, world_mask=None, flags=None) -> None:
        for inner in (self.newton_solver, *self.hosts.values()):
            reset = getattr(inner, "reset", None)
            if reset is None:
                continue
            try:
                reset(state, world_mask=world_mask, flags=flags) if flags is not None else reset(
                    state, world_mask=world_mask
                )
            except TypeError:
                reset(state)

    def solver(self, name: str):
        if name in self.hosts:
            return self.hosts[name]
        inner = self.newton_solver
        if inner is None:
            raise KeyError(f"no island {name!r}; host islands: {sorted(self.hosts)}")
        lookup = getattr(inner, "solver", None)
        return lookup(name) if callable(lookup) else inner

    def view(self, name: str):
        if name in self.hosts:
            return self.model
        inner = self.newton_solver
        lookup = getattr(inner, "view", None)
        return lookup(name) if callable(lookup) else self.model

    def entry_names(self) -> tuple[str, ...]:
        inner = self.newton_solver
        names = getattr(inner, "entry_names", None)
        newton_names = tuple(names()) if callable(names) else ()
        return (*newton_names, *self.hosts)

    def __getattr__(self, name: str):
        # The incidental solver surface -- ``entry_state``, ``prepare_contacts``,
        # ``contacts``, the proxy tuning hooks -- belongs to the Newton side;
        # forwarding it wholesale is what lets the wrapper stand wherever the
        # inner solver stood. Only reached for names not set in __init__.
        inner = self.__dict__.get("newton_solver")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def build_bare(ctx: CouplerBuild) -> Any:
    """One island: the solver over the whole model, no coupler."""
    if len(ctx.islands) != 1:
        raise CompositionError(f"a bare solver needs exactly one island, got {[i.name for i in ctx.islands]}")
    island = ctx.islands[0]
    # A bare island owns the whole model, so its particles are the model's --
    # its own index list is empty when it came from the entity-free path.
    particles = bool(island.particles.size) or (not island.nodes and int(ctx.model.particle_count) > 0)
    return build_solver(island.backend, ctx.model, has_particles=particles)


def build_coupled(ctx: CouplerBuild) -> Any:
    """The solver for the islands: bare for one, the arrangement's coupler for more."""
    if len(ctx.islands) == 1:
        return build_bare(ctx)
    method = ctx.spec.method
    if method not in COUPLER_BUILDERS:
        raise CompositionError(f"no builder for coupler {method!r}; registered: {COUPLER_BUILDERS.names()}")
    return COUPLER_BUILDERS[method](ctx)


register_coupler_builder("proxy", build_proxy)
register_coupler_builder("admm", build_admm)
register_coupler_builder("one_way", build_one_way)
