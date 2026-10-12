# SPDX-License-Identifier: Apache-2.0
"""The interaction graph: what is in the scene, what state it owns, what it touches.

A :class:`Node` per built entity -- its domain, its declared requirements, the
roles it plays (the table is ``surface``, a peg is ``fixtures``, a hand is both
``robot`` and ``hands``) and the model indices it owns. An :class:`Edge` per
contact rule, resolved from roles to nodes. The assets declare the rules; a
domain that declares none gets the defaults below, which reproduce what every
shipped scene did before anything was declared: a deformable touches the
surface, the fixtures, the objects, the hands and the other deformables; a
rigid object touches the surface, the fixtures, the robot and the other
objects; a robot touches the furniture and the objects.

Joints and shapes are **derived from bodies against the model** rather than
asked of the entity: a joint belongs to whoever owns its child body, a shape to
whoever owns the body it sits on. That is what makes an unknown entity
composable -- it reports bodies and particles, the two things only it knows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np

from robosimstudio.specs.physics import DOMAINS, ROLES, ArrangementSpec, ContactRule, SimRequirements, get_domain

from .errors import CompositionError

__all__ = ["Edge", "InteractionGraph", "Node", "build_graph", "default_contacts"]


def default_contacts(domain: str) -> tuple[ContactRule, ...]:
    """What a domain touches when its asset declares nothing.

    Read off the domain's own registration rather than looked up in a table
    here. The table enumerated the six domains the library happens to ship, so a
    domain registered by content touched **nothing** by default and no message
    said so -- a scene composed, stepped, and silently never collided.
    """
    return tuple(ContactRule(with_=role) for role in get_domain(domain).contacts)


_STRENGTH = {"off": 0, "one_way": 1, "two_way": 2}


@dataclass
class Node:
    """One built entity in the interaction graph.

    Attributes:
        name: The entity's name.
        entity: The entity object.
        domain: Its physics domain.
        sim: Its declaration (the spec's, or the domain default).
        roles: Which of :data:`~robosimstudio.specs.physics.ROLES` it plays.
        bodies, joints, particles, shapes: Global template-model indices it owns.
        hand_bodies: For a robot, the bodies that may touch deformables.
        hand_shapes: The shapes on those bodies.
        world_fixed: Whether its shapes sit on the world (a static).
    """

    name: str
    entity: Any
    domain: str
    sim: SimRequirements
    roles: frozenset[str]
    bodies: np.ndarray
    joints: np.ndarray
    particles: np.ndarray
    shapes: np.ndarray
    hand_bodies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    hand_shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    world_fixed: bool = False

    @property
    def state(self) -> tuple[str, ...]:
        return get_domain(self.domain).state

    def has(self, kind: str) -> bool:
        return {"bodies": self.bodies, "joints": self.joints, "particles": self.particles, "shapes": self.shapes}[
            kind
        ].size > 0


@dataclass(frozen=True)
class Edge:
    """One contact between two nodes. ``hands_only`` restricts ``b`` to its coupling surfaces."""

    a: str
    b: str
    coupling: str = "two_way"
    hands_only: bool = False
    source: str = "default"

    @property
    def key(self) -> tuple[str, str, bool]:
        return (*sorted((self.a, self.b)), self.hands_only)


@dataclass
class InteractionGraph:
    nodes: dict[str, Node]
    edges: list[Edge]
    self_contact: set[str]
    unresolved: list[str] = field(default_factory=list)
    """Rules that named a role nothing in this scene plays -- reported, not refused:
    a cloth that declares contact with ``objects`` in a scene without any is not wrong."""

    def by_role(self, role: str) -> list[Node]:
        return [n for n in self.nodes.values() if role in n.roles]

    def edges_of(self, name: str) -> list[Edge]:
        return [e for e in self.edges if name in (e.a, e.b)]

    def robot(self) -> Node | None:
        return next((n for n in self.nodes.values() if "robot" in n.roles), None)


def _joints_of(model: newton.Model, bodies: np.ndarray, joint_child: np.ndarray) -> np.ndarray:
    if bodies.size == 0 or model.joint_count == 0:
        return np.zeros(0, dtype=np.int32)
    owned = np.isin(joint_child, bodies)
    return np.flatnonzero(owned).astype(np.int32)


def _shapes_of(bodies: np.ndarray, shape_body: np.ndarray, hint: np.ndarray) -> np.ndarray:
    on_bodies = (
        np.flatnonzero(np.isin(shape_body, bodies)).astype(np.int32) if bodies.size else np.zeros(0, dtype=np.int32)
    )
    return np.unique(np.concatenate([on_bodies, np.asarray(hint, dtype=np.int32)])).astype(np.int32)


def _requirements(entity: Any) -> SimRequirements:
    """The entity's declaration: the spec's ``sim`` when it carries one, else the class default."""
    declared = getattr(getattr(entity, "spec", None), "sim", None)
    if isinstance(declared, SimRequirements):
        return declared
    asked = getattr(entity, "sim_requirements", None)
    if callable(asked):
        out = asked()
        if isinstance(out, SimRequirements):
            return out
    domain = getattr(entity, "domain", "rigid")
    return SimRequirements(domain=domain)


def _owned(entity: Any, method: str, fallback) -> np.ndarray:
    fn = getattr(entity, method, None)
    values = fn() if callable(fn) else fallback()
    return np.asarray(list(values), dtype=np.int32).reshape(-1)


def build_graph(
    model: newton.Model,
    parts: dict[str, Any],
    embodiment: Any | None,
    arrangement: ArrangementSpec,
    *,
    scene_statics: Iterable[str] = (),
) -> InteractionGraph:
    """Nodes and edges for a bound scene.

    Args:
        model: The template model every entity is bound to.
        parts: Every built entity by name, the compiler's ``parts`` map.
        embodiment: The robot's entity, or ``None``.
        arrangement: For its ``overrides``.
        scene_statics: Names of the statics that are the *cell's* (the table),
            which play ``surface``; every other static plays ``fixtures``.
    """
    shape_body = model.shape_body.numpy() if model.shape_count else np.zeros(0, dtype=np.int32)
    joint_child = model.joint_child.numpy() if model.joint_count else np.zeros(0, dtype=np.int32)
    scene_statics = set(scene_statics)

    nodes: dict[str, Node] = {}
    entities = dict(parts)
    # By identity, not by key: a caller that already put the robot in ``parts``
    # under its own name must not get a second node owning the same bodies --
    # two nodes over one body is two islands claiming it, which the coupled
    # solver refuses much later and much less legibly.
    if embodiment is not None and not any(part is embodiment for part in entities.values()):
        entities["embodiment"] = embodiment

    for name, entity in entities.items():
        sim = _requirements(entity)
        domain = sim.domain
        if domain not in DOMAINS:
            raise CompositionError(
                f"entity {name!r} declares physics domain {domain!r}, which is not registered "
                f"(registered: {sorted(DOMAINS)}). register_domain() it beside the entity class."
            )
        rng = getattr(entity, "range", None)
        bodies = _owned(entity, "owned_bodies", lambda r=rng: r.body_indices if r is not None else ())
        particles = _owned(entity, "owned_particles", lambda r=rng: r.particle_indices if r is not None else ())
        hint = _owned(entity, "owned_shapes", lambda r=rng: r.shape_indices if r is not None else ())
        joints = _joints_of(model, bodies, joint_child)
        shapes = _shapes_of(bodies, shape_body, hint)

        # The role is the domain's own, declared at ``register_domain``. One
        # promotion survives and belongs here rather than there: a fixture the
        # *cell* owns is the work surface, which is a fact about the scene and
        # not about the kind of thing.
        info = get_domain(domain)
        role = "surface" if info.world_fixed and name in scene_statics else info.role
        roles: set[str] = {role}

        node = Node(
            name=name,
            entity=entity,
            domain=domain,
            sim=sim,
            roles=frozenset(roles),
            bodies=bodies,
            joints=joints,
            particles=particles,
            shapes=shapes,
            world_fixed=bool(info.world_fixed and bodies.size == 0),
        )
        # No domain guard: an entity offers hands by *having* ``coupling_bodies``,
        # and a hand is a hand whether the thing carrying it registered itself as
        # ``articulated`` or as something nobody here has heard of.
        hands = getattr(entity, "coupling_bodies", None)
        hand_bodies = np.asarray(list(hands()) if callable(hands) else [], dtype=np.int32)
        node.hand_bodies = hand_bodies
        node.hand_shapes = _shapes_of(hand_bodies, shape_body, np.zeros(0, dtype=np.int32))
        if hand_bodies.size:
            node.roles = frozenset(roles | {"hands"})
        nodes[name] = node

    graph = InteractionGraph(nodes=nodes, edges=[], self_contact=set())
    edges: dict[tuple[str, str, bool], Edge] = {}

    def add(a: str, b: str, coupling: str, hands_only: bool, source: str) -> None:
        edge = Edge(a=a, b=b, coupling=coupling, hands_only=hands_only, source=source)
        have = edges.get(edge.key)
        if (
            have is None
            or source == "override"
            or (have.source != "override" and _STRENGTH[coupling] > _STRENGTH[have.coupling])
        ):
            edges[edge.key] = edge

    def resolve(node: Node, rule: ContactRule, source: str) -> None:
        target = rule.with_
        if target == "self":
            graph.self_contact.add(node.name)
            return
        if target == "world":
            return
        if target in nodes:
            if target != node.name:
                add(node.name, target, rule.coupling, False, source)
            return
        if target not in ROLES:
            graph.unresolved.append(f"{node.name}: contact with {target!r}, which is neither a role nor an entity")
            return
        if target == "hands":
            robot = graph.robot()
            if robot is None or robot.hand_bodies.size == 0:
                if source != "default":
                    graph.unresolved.append(f"{node.name}: contact with hands, but the scene has no coupling surfaces")
                return
            add(node.name, robot.name, rule.coupling, True, source)
            return
        targets = [n for n in graph.by_role(target) if n.name != node.name]
        if not targets and source != "default":
            # A *declared* rule naming a role nobody plays is worth a line; a
            # domain default naming one is just a scene without that kind of thing.
            graph.unresolved.append(f"{node.name}: contact with {target}, which nothing in this scene plays")
        for other in targets:
            add(node.name, other.name, rule.coupling, False, source)

    for node in nodes.values():
        rules = node.sim.contacts or default_contacts(node.domain)
        source = "declared" if node.sim.contacts else "default"
        for rule in rules:
            resolve(node, rule, source)
        if node.sim.self_contact:
            graph.self_contact.add(node.name)

    for target, value in arrangement.overrides:
        if not isinstance(value, ContactRule):
            continue
        subjects = [nodes[target]] if target in nodes else graph.by_role(target)
        if not subjects:
            graph.unresolved.append(f"override for {target!r} names nothing in this scene")
        for node in subjects:
            resolve(node, value, "override")

    graph.edges = [e for e in edges.values() if e.coupling != "off"]
    return graph
