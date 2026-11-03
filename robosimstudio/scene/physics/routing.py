# SPDX-License-Identifier: Apache-2.0
"""Which shape pairs each collision pipeline detects.

A pair is detected in exactly one place. The rule is Newton's: when an island
is the *destination* of a proxy channel, its solver is stepped with the
channel's own pipeline and never sees the outer contacts, so that pipeline must
carry everything the island touches -- its own pairs, the statics it rests on,
and the proxied bodies. The main pipeline then serves what is left: a rigid
island that asked for Newton contacts (``use_mujoco_contacts=False``), or a
lone deformable island with no channel at all.

Every list is cut from ``model.shape_contact_pairs`` rather than enumerated by
hand, because that table already honours every
``add_shape_collision_filter_pair`` the entities made -- and an explicit broad
phase handed a hand-made list silently re-admits every filtered pair. The
cable-routing cell has 392 of them, the working arm's blades against the table, and
re-admitted they were the floor the hand stopped on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import newton
import numpy as np

from robosimstudio.specs.physics import get_domain

from .interactions import InteractionGraph, Node
from .islands import Island
from .world import expand_shapes

__all__ = ["Routing", "filtered_pairs", "route"]


def filtered_pairs(model: newton.Model, rows) -> np.ndarray:
    """``rows`` less anything the *builder* filtered, as an ``[n, 2]`` int32 array.

    An explicit broad phase is a list of pairs to **check**, and Newton reaches
    for ``model.shape_contact_pairs`` -- which honours every
    ``add_shape_collision_filter_pair`` -- only when no list is handed to it.
    """
    rows = np.asarray(list(rows), dtype=np.int32).reshape(-1, 2)
    keep = [row for row in rows.tolist() if not model.shape_collision_filter_contains(int(row[0]), int(row[1]))]
    return np.asarray(keep, dtype=np.int32).reshape(-1, 2)


@dataclass
class Routing:
    """The pair lists, by pipeline.

    Attributes:
        main_pairs: Rows for the main pipeline, or ``None`` for no main pipeline.
        channel_pairs: Rows per ``(source, destination)`` channel, or ``None``
            when the channel's pipeline should use its view's own pair set.
        unrouted: Edges no pipeline serves, in words.
    """

    main_pairs: np.ndarray | None = None
    channel_pairs: dict[tuple[str, str], np.ndarray | None] = field(default_factory=dict)
    unrouted: list[str] = field(default_factory=list)


def _island_of(islands: list[Island], name: str) -> Island | None:
    return next((i for i in islands if i.owns(name)), None)


def _rows_between(pairs: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows of ``pairs`` with one shape in ``a`` and the other in ``b``."""
    if pairs.size == 0 or a.size == 0 or b.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    in_a0 = np.isin(pairs[:, 0], a)
    in_b1 = np.isin(pairs[:, 1], b)
    in_a1 = np.isin(pairs[:, 1], a)
    in_b0 = np.isin(pairs[:, 0], b)
    return pairs[(in_a0 & in_b1) | (in_a1 & in_b0)]


def _rows_within(pairs: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    if pairs.size == 0 or shapes.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return pairs[np.isin(pairs[:, 0], shapes) & np.isin(pairs[:, 1], shapes)]


def _shapes_for(node: Node, hands_only: bool) -> np.ndarray:
    return node.hand_shapes if hands_only else node.shapes


def route(
    graph: InteractionGraph,
    islands: list[Island],
    channels: list,
    model: newton.Model,
    *,
    world_shapes: np.ndarray,
    worlds=None,
) -> Routing:
    """Cut the model's pair table into per-pipeline lists.

    Args:
        graph: The interaction graph.
        islands: The islands.
        channels: The channels (``.source``, ``.destination``, ``.bodies``, ``.shapes``).
        model: The model being stepped, for its pair table.
        world_shapes: Shapes on the world body (the ground plane and any static
            added outside an entity's range), visible to every pipeline.
        worlds: The stride map when the model is replicated, or ``None``.

            **Load bearing, and its absence was silent.** The pair table cut from
            here is the *replicated* model's and holds every world's pairs; the
            sets doing the cutting come from the interaction graph, whose nodes
            hold the **template's** shape indices because that is the model every
            entity bound to. Cut one against the other and every list comes back
            holding world 0's pairs and nothing else -- so world 0 has contacts
            and no other world does.

            Measured on a cable-routing scene at three worlds, before this
            argument existed: worlds 1 and 2 were identical to world 0 at reset,
            diverged from the first step, and their cables were 19 m below the
            bench by step 120 -- free fall, because nothing was holding them up.
            Nothing reported anything: the states stayed finite, the episodes
            ran their budget, and the rows said what rows say.
    """
    pairs = (
        model.shape_contact_pairs.numpy().astype(np.int32).reshape(-1, 2)
        if model.shape_count
        else np.zeros((0, 2), np.int32)
    )
    routing = Routing()
    destinations = {c.destination for c in channels}
    replicated = worlds is not None and worlds.num_worlds > 1

    def spread(shapes: np.ndarray) -> np.ndarray:
        """Template shape indices, as every world's copy of them.

        Only for sets that come from the graph. A channel's shapes are derived
        from the replicated model's ``shape_body`` and are already world
        complete -- putting them through here would drop every world past the
        first, since ``expand_shapes`` reads anything at or above one world's
        stride as global.
        """
        if not replicated:
            return shapes
        return expand_shapes(shapes, worlds.shapes_per_world, worlds.num_worlds, model.shape_count)

    static_shapes = spread(
        np.unique(
            np.concatenate(
                [n.shapes for n in graph.nodes.values() if get_domain(n.domain).world_fixed] + [world_shapes]
            )
        ).astype(np.int32)
    )

    main_rows: list[np.ndarray] = []
    channel_rows: dict[tuple[str, str], list[np.ndarray]] = {(c.source, c.destination): [] for c in channels}
    needs_main = False

    def island_for(node: Node) -> Island | None:
        return _island_of(islands, node.name)

    for edge in graph.edges:
        a, b = graph.nodes[edge.a], graph.nodes[edge.b]
        ia, ib = island_for(a), island_for(b)
        sa, sb = spread(_shapes_for(a, False)), spread(_shapes_for(b, edge.hands_only))
        rows = _rows_between(pairs, sa, sb)
        # Which side is the deformable destination, if any.
        dest = None
        other = None
        for node, island in ((a, ia), (b, ib)):
            if island is not None and island.name in destinations:
                dest, other = island, (b if node is a else a)
        if dest is not None:
            channel = next((c for c in channels if c.destination == dest.name), None)
            key = (channel.source, channel.destination)
            io = island_for(other)
            if io is dest or io is None or get_domain(other.domain).world_fixed:
                channel_rows[key].append(rows)
            elif io is not None and io.name == channel.source:
                # Only the proxied bodies' shapes are visible in the destination.
                proxied = _rows_between(pairs, spread(_shapes_for(dest_node(graph, dest), False)), channel.shapes)
                hand_rows = _rows_between(pairs, sa if a is other else sb, channel.shapes) if False else proxied
                channel_rows[key].append(hand_rows)
            else:
                routing.unrouted.append(
                    f"{a.name} x {b.name}: {other.name} is in island {io.name!r}, which is not proxied into "
                    f"{dest.name!r}; add a channel or declare the contact off"
                )
            continue
        # Neither end is a channel destination: the main pipeline, if the
        # islands involved take Newton contacts.
        natives = [i for i in (ia, ib) if i is not None and i.uses_native_contacts()]
        if ia is not None and ib is not None and ia is ib and ia.uses_native_contacts():
            continue  # MuJoCo detects its own
        if natives and (ia is None or ib is None or ia is ib):
            continue  # a rigid island's static contacts, detected natively
        if natives and ia is not None and ib is not None and ia is not ib:
            routing.unrouted.append(
                f"{a.name} x {b.name}: {ia.name!r} and {ib.name!r} are different islands with no channel between them"
            )
            continue
        needs_main = True
        main_rows.append(rows)

    # Self-contact of rigid deformables (rods) and statics inside a destination.
    for dest_name in destinations:
        dest = next(i for i in islands if i.name == dest_name)
        channel = next(c for c in channels if c.destination == dest_name)
        key = (channel.source, channel.destination)
        own = spread(
            np.unique(np.concatenate([n.shapes for n in dest.nodes] or [np.zeros(0, np.int32)])).astype(np.int32)
        )
        channel_rows[key].append(_rows_within(pairs, own))
        channel_rows[key].append(_rows_between(pairs, own, static_shapes))
        channel_rows[key].append(_rows_between(pairs, own, channel.shapes))

    if needs_main or (len(islands) == 1 and not islands[0].uses_native_contacts()):
        # A lone island that does not detect its own contacts gets Newton's own
        # broad phase over the whole model rather than an enumerated list: the
        # list would be the same set, and building it costs a pass over every
        # pair for nothing.
        rows = (
            np.unique(np.concatenate(main_rows or [np.zeros((0, 2), np.int32)]), axis=0)
            if main_rows
            else np.zeros((0, 2), np.int32)
        )
        routing.main_pairs = rows.astype(np.int32)
    for key, chunks in channel_rows.items():
        rows = np.concatenate(chunks or [np.zeros((0, 2), np.int32)]).reshape(-1, 2)
        routing.channel_pairs[key] = np.unique(rows, axis=0).astype(np.int32) if rows.size else rows.astype(np.int32)
    return routing


def dest_node(graph: InteractionGraph, dest: Island) -> Node:
    """A synthetic node holding every shape of a destination island, for routing."""
    shapes = np.unique(np.concatenate([n.shapes for n in dest.nodes] or [np.zeros(0, np.int32)])).astype(np.int32)
    first = dest.nodes[0]
    return Node(
        name=dest.name,
        entity=None,
        domain=first.domain,
        sim=first.sim,
        roles=frozenset(),
        bodies=dest.bodies,
        joints=dest.joints,
        particles=dest.particles,
        shapes=shapes,
    )
