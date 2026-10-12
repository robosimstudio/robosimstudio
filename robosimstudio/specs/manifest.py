# SPDX-License-Identifier: Apache-2.0
"""The manifest: what the composer built for one scene, as data a bundle can carry.

The composer in ``robosimstudio.scene.physics`` reads what the assets declared
and what the arrangement staged, derives everything else, and records what it
built here: the solver islands (:class:`IslandRecord`), the coupling channels
between them (:class:`ChannelRecord`) and the arrangement they came from, as one
:class:`PhysicsManifest` that round-trips through plain JSON. A repro bundle
carries it, the studio shows it, and neither needs an alias table to read it.

One of the four modules :mod:`~robosimstudio.specs.physics` was split into. It
depends on the arrangement's types and on nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from .arrangement import ArrangementSpec, BackendSpec, CollisionSpec, backend, coupler

__all__ = [
    "ChannelRecord",
    "IslandRecord",
    "PhysicsManifest",
]


# ---------------------------------------------------------------------------
# The manifest: what the composer built, as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class IslandRecord:
    """One solver island: a backend over a set of entities' state.

    Attributes:
        name: The entry name the coupler knows it by.
        backend: The backend that solves it.
        solver_class: Dotted path of the solver class, resolved.
        domains: The domains it serves.
        entities: The entities it owns, by name.
        bodies, joints, particles, shapes: Global model indices it owns
            (world-expanded when the model is replicated). ``shapes`` empty
            means "derived from the bodies plus every world-fixed shape".
    """

    name: str
    backend: BackendSpec
    solver_class: str
    domains: tuple[str, ...]
    entities: tuple[str, ...]
    bodies: tuple[int, ...] = ()
    joints: tuple[int, ...] = ()
    particles: tuple[int, ...] = ()
    shapes: tuple[int, ...] = ()
    host_engine: bool = False
    """Whether this island is a foreign engine stepped on the host. When true,
    ``solver_class`` is the backend *alias*, never a dotted path -- a repro
    bundle may not name the exporting project, and the runner keeps its own
    self-contained twin per alias."""
    host_recipe: Any = None
    """The host island's build recipe -- gravity, per-entity actor records,
    mirror shape records -- as plain JSON-able data. What the bridge consumes,
    recorded so a bundle rebuilds the same foreign scene from the model and
    this manifest alone. ``None`` for every Newton island."""


@dataclass(frozen=True, kw_only=True)
class ChannelRecord:
    """One coupling channel between two islands.

    Attributes:
        source: Island the proxied bodies belong to.
        destination: Island that sees them as proxies.
        bodies: The proxied global body indices.
        coupling: ``"two_way"`` or ``"one_way"``.
        mass_scale: Proxy mass scale applied in the destination.
        mode: ``"staggered"`` or ``"lagged"``.
        collide_interval: How often the channel's own pipeline re-detects.
        pairs: The explicit shape pairs routed to the channel's pipeline, or
            ``None`` when it uses the destination view's own pair set.
        entities: Which entities the proxied bodies came from.
    """

    source: str
    destination: str
    bodies: tuple[int, ...]
    coupling: str = "two_way"
    mass_scale: float = 1.0
    mode: str = "staggered"
    collide_interval: int = 1
    pairs: tuple[tuple[int, int], ...] | None = None
    entities: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PhysicsManifest:
    """What the composer built for one scene, as data a bundle can carry.

    Attributes:
        arrangement: The arrangement it was built from.
        islands: The solver islands, in stepping order.
        channels: The coupling channels.
        coupler: ``"none"`` (one island, a bare solver), ``"proxy"``,
            ``"admm"`` or a registered kind.
        main_pairs: Shape pairs routed to the main collision pipeline, or
            ``None`` for no main pipeline.
        world_count: How many worlds the model holds.
        mirror_joint_ik: Whether the schedule mirrors maximal-coordinate
            islands back into ``joint_q`` after each substep.
        stages: The named stages entities contributed, by phase.
    """

    arrangement: ArrangementSpec
    islands: tuple[IslandRecord, ...]
    channels: tuple[ChannelRecord, ...] = ()
    coupler: str = "none"
    main_pairs: tuple[tuple[int, int], ...] | None = None
    world_count: int = 1
    mirror_joint_ik: bool = False
    stages: tuple[tuple[str, str], ...] = ()

    def island(self, name: str) -> IslandRecord:
        for island in self.islands:
            if island.name == name:
                return island
        raise KeyError(f"no island {name!r}; has {[i.name for i in self.islands]}")

    def island_of(self, entity: str) -> IslandRecord:
        for island in self.islands:
            if entity in island.entities:
                return island
        raise KeyError(f"no island owns {entity!r}")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able data. Index lists stay as lists; an exporter may move them to arrays."""

        def back(spec: BackendSpec) -> dict[str, Any]:
            return {"solver": spec.solver, "params": dict(spec.params), "substeps": int(spec.substeps)}

        arrangement = {
            "control_hz": float(self.arrangement.control_hz),
            "substeps": int(self.arrangement.substeps),
            "gravity": float(self.arrangement.gravity),
            "use_cuda_graph": bool(self.arrangement.use_cuda_graph),
            "backends": {domain: back(spec) for domain, spec in self.arrangement.backends},
            "coupling": {"method": self.arrangement.coupling.method, "params": dict(self.arrangement.coupling.params)},
            "collision": {f.name: getattr(self.arrangement.collision, f.name) for f in fields(CollisionSpec)},
        }
        return {
            "arrangement": arrangement,
            "coupler": self.coupler,
            "world_count": int(self.world_count),
            "mirror_joint_ik": bool(self.mirror_joint_ik),
            "stages": [list(pair) for pair in self.stages],
            "islands": [
                {
                    "name": i.name,
                    "backend": back(i.backend),
                    "solver_class": i.solver_class,
                    "domains": list(i.domains),
                    "entities": list(i.entities),
                    "bodies": list(i.bodies),
                    "joints": list(i.joints),
                    "particles": list(i.particles),
                    "shapes": list(i.shapes),
                    "host_engine": bool(i.host_engine),
                    "host_recipe": i.host_recipe,
                }
                for i in self.islands
            ],
            "channels": [
                {
                    "source": c.source,
                    "destination": c.destination,
                    "bodies": list(c.bodies),
                    "coupling": c.coupling,
                    "mass_scale": float(c.mass_scale),
                    "mode": c.mode,
                    "collide_interval": int(c.collide_interval),
                    "pairs": None if c.pairs is None else [list(p) for p in c.pairs],
                    "entities": list(c.entities),
                }
                for c in self.channels
            ],
            "main_pairs": None if self.main_pairs is None else [list(p) for p in self.main_pairs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PhysicsManifest:
        def back(node: Mapping[str, Any]) -> BackendSpec:
            return backend(node["solver"], substeps=int(node.get("substeps", 1)), **dict(node.get("params", {})))

        node = data["arrangement"]
        arrangement = ArrangementSpec(
            control_hz=float(node["control_hz"]),
            substeps=int(node["substeps"]),
            gravity=float(node.get("gravity", -9.81)),
            use_cuda_graph=bool(node.get("use_cuda_graph", True)),
            backends=tuple((domain, back(spec)) for domain, spec in node["backends"].items()),
            coupling=coupler(node["coupling"]["method"], **dict(node["coupling"].get("params", {}))),
            collision=CollisionSpec(**dict(node.get("collision", {}))),
        )
        islands = tuple(
            IslandRecord(
                name=i["name"],
                backend=back(i["backend"]),
                solver_class=i["solver_class"],
                domains=tuple(i.get("domains", ())),
                entities=tuple(i.get("entities", ())),
                bodies=tuple(int(v) for v in i.get("bodies", ())),
                joints=tuple(int(v) for v in i.get("joints", ())),
                particles=tuple(int(v) for v in i.get("particles", ())),
                shapes=tuple(int(v) for v in i.get("shapes", ())),
                host_engine=bool(i.get("host_engine", False)),
                host_recipe=i.get("host_recipe"),
            )
            for i in data.get("islands", ())
        )
        channels = tuple(
            ChannelRecord(
                source=c["source"],
                destination=c["destination"],
                bodies=tuple(int(v) for v in c.get("bodies", ())),
                coupling=c.get("coupling", "two_way"),
                mass_scale=float(c.get("mass_scale", 1.0)),
                mode=c.get("mode", "staggered"),
                collide_interval=int(c.get("collide_interval", 1)),
                pairs=None if c.get("pairs") is None else tuple((int(a), int(b)) for a, b in c["pairs"]),
                entities=tuple(c.get("entities", ())),
            )
            for c in data.get("channels", ())
        )
        main_pairs = data.get("main_pairs")
        return cls(
            arrangement=arrangement,
            islands=islands,
            channels=channels,
            coupler=data.get("coupler", "none"),
            main_pairs=None if main_pairs is None else tuple((int(a), int(b)) for a, b in main_pairs),
            world_count=int(data.get("world_count", 1)),
            mirror_joint_ik=bool(data.get("mirror_joint_ik", False)),
            stages=tuple((str(a), str(b)) for a, b in data.get("stages", ())),
        )
