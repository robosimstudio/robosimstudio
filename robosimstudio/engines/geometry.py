# SPDX-License-Identifier: Apache-2.0
"""Newton's collision shapes, read into a form any engine can build from.

The half of a mirror that is not about any particular engine. A *mirror* is a
Newton rigid shape rebuilt inside a foreign engine so that a deformable it owns
has something to press against; the deformable never lives in two engines, but
the boundary it touches does. What each engine does with the geometry differs --
SuperDex wants a shape handle and an SDF bake, Genesis wants a morph and a
material -- but *reading it off the model* is the same work twice, and this is
where it happens once.

Two things are kept apart on purpose. :attr:`ShapeGeometry.kind` is the shape's
**primitive intent**, so an engine that has a native box or cylinder can use it
and keep an exact surface; :meth:`ShapeGeometry.mesh` is the tessellation, for
the kinds an engine lacks. Tessellating eagerly would charge every engine for a
mesh it may not want -- SuperDex evaluates planes and spheres implicitly and
would throw the triangles away.

Everything here is derived from the **finalized** ``newton.Model``, never from
specs. That is the rule the host-engine recipe already follows, and it is what
lets a repro bundle rebuild a mirror from the model and the manifest alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from newton import GeoType

from robosimstudio.core.math import young_poisson_from_lame
from robosimstudio.registry import Registry
from robosimstudio.specs.physics import actor_consumes, declare_consumes, engine_consumes

__all__ = [
    "ACTOR_BUILDERS",
    "ACTOR_KINDS",
    "ActorGeometry",
    "ActorReader",
    "ShapeGeometry",
    "actor_builder",
    "actor_consumes",
    "actor_geometry",
    "builders_for",
    "engine_consumes",
    "iter_mirrors",
    "read_shape_geometry",
    "register_actor_builder",
    "register_actor_kind",
    "shape_arrays",
]

_POS = slice(0, 3)
_XYZW = slice(3, 7)

_KINDS: dict[int, str] = {
    int(GeoType.PLANE): "plane",
    int(GeoType.SPHERE): "sphere",
    int(GeoType.BOX): "box",
    int(GeoType.CAPSULE): "capsule",
    int(GeoType.CYLINDER): "cylinder",
    int(GeoType.CONE): "cone",
    int(GeoType.MESH): "mesh",
    int(GeoType.CONVEX_MESH): "mesh",
}
"""Newton's ``GeoType`` as a name an engine adapter can switch on.

A name rather than the enum so that an adapter never imports Newton's geometry
module to ask what it is looking at, and so that both mesh types collapse to the
one thing an engine can do with either.
"""


def shape_arrays(model: Any) -> dict[str, np.ndarray]:
    """The five shape arrays a mirror is built from, pulled to the host once."""
    return {
        "type": model.shape_type.numpy(),
        "scale": model.shape_scale.numpy(),
        "transform": model.shape_transform.numpy(),
        "body": model.shape_body.numpy(),
        "mu": model.shape_material_mu.numpy(),
    }


@dataclass(frozen=True)
class ShapeGeometry:
    """One Newton collision shape, as an engine-neutral description.

    Attributes:
        shape: Newton shape index.
        body: Newton body index; ``-1`` for world-fixed.
        kind: One of ``plane sphere box capsule cylinder cone mesh``.
        scale: Newton's shape scale — a radius, a half-extent, a half-height,
            per kind. **Half** measures, as Newton stores them.
        local: The shape's body-local transform ``[7]``, xyz + xyzw.
        mu: The shape's Coulomb friction coefficient.
        two_way: Whether contact against this mirror is harvested back onto the
            Newton body. Comes from the contact rule, not from the geometry.
        source: ``model.shape_source[shape]`` for a mesh, else ``None``.
        world: Which world of a replicated model this shape is being mirrored
            for. 0 when nothing replicated it.

            Carried through from the record rather than derived from ``shape``,
            because one shape can be mirrored more than once: a world-fixed
            bench is one shape in Newton and one collider per world in a host
            engine, since the worlds share a place there and must not.
    """

    shape: int
    body: int
    kind: str
    scale: np.ndarray
    local: np.ndarray
    mu: float
    two_way: bool = False
    source: Any = None
    world: int = 0

    @property
    def moving(self) -> bool:
        """Whether a body carries this shape, so a tick has to drive it."""
        return self.body >= 0

    def mesh(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(vertices, faces)`` in the shape's local frame, or ``None``.

        Tessellated on demand: an engine with a native primitive for this kind
        should use that instead and keep an exact surface. Coarse counts are
        deliberate for the analytic kinds -- an engine that needs triangles for
        a capsule is going to bake an SDF or a barrier from them, and contact
        quality then follows that resolution rather than this one.

        A mesh comes out of ``model.shape_source`` exactly as collision saw it,
        so a decomposed fingertip mirrors as its decomposition rather than as
        the visual hull it was decomposed to escape.
        """
        import trimesh

        scale = self.scale
        if self.kind == "box":
            mesh = trimesh.creation.box(extents=2.0 * scale)
        elif self.kind == "capsule":
            mesh = trimesh.creation.capsule(radius=float(scale[0]), height=2.0 * float(scale[1]), count=(12, 12))
        elif self.kind == "cylinder":
            mesh = trimesh.creation.cylinder(radius=float(scale[0]), height=2.0 * float(scale[1]), sections=24)
        elif self.kind == "cone":
            mesh = trimesh.creation.cone(radius=float(scale[0]), height=2.0 * float(scale[1]), sections=24)
            # trimesh puts the base at z=0 and the apex at 2h; Newton's cone is
            # centred, base at -h and apex at +h (``add_shape_cone``).
            mesh.apply_translation((0.0, 0.0, -float(scale[1])))
        elif self.kind == "sphere":
            mesh = trimesh.creation.icosphere(radius=float(scale[0]), subdivisions=2)
        elif self.kind == "plane":
            return None
        elif self.kind == "mesh":
            if self.source is None:
                return None
            verts = np.asarray(getattr(self.source.vertices, "numpy", lambda: self.source.vertices)(), dtype=np.float64)
            faces = np.asarray(
                getattr(self.source.indices, "numpy", lambda: self.source.indices)(), dtype=np.int64
            ).reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=verts * scale, faces=faces, process=False)
        else:
            return None
        return (
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3),
        )


def read_shape_geometry(
    model: Any,
    shape: int,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    body: int | None = None,
    two_way: bool = False,
) -> ShapeGeometry | None:
    """One Newton shape, described. ``None`` for a kind nothing can mirror.

    Args:
        arrays: The output of :func:`shape_arrays`, when a caller is walking
            many shapes and would rather not pull the model to the host once
            per shape.
        body: Override for the owning body — the mirror record's, which is the
            one the recipe stated.
    """
    arrays = shape_arrays(model) if arrays is None else arrays
    index = int(shape)
    kind = _KINDS.get(int(arrays["type"][index]))
    if kind is None:
        return None
    return ShapeGeometry(
        shape=index,
        body=int(arrays["body"][index]) if body is None else int(body),
        kind=kind,
        scale=np.asarray(arrays["scale"][index], dtype=np.float64),
        local=np.asarray(arrays["transform"][index], dtype=np.float64),
        mu=float(arrays["mu"][index]),
        two_way=bool(two_way),
        source=model.shape_source[index] if kind == "mesh" else None,
    )


def iter_mirrors(model: Any, records) -> list[ShapeGeometry]:
    """The host recipe's ``mirrors`` list, read off the model.

    Records are ``{"shape": int, "body": int, "two_way": bool}`` — exactly what
    the manifest's host-island record carries, which is why this takes the raw
    dicts rather than a type the manifest would then have to know about.
    Anything unmirrorable is dropped, as a skip rather than an error: the
    composer already refused the cases that are mistakes.
    """
    arrays = shape_arrays(model)
    out: list[ShapeGeometry] = []
    for record in records:
        geom = read_shape_geometry(
            model,
            int(record["shape"]),
            arrays=arrays,
            body=record.get("body"),
            two_way=bool(record.get("two_way", False)),
        )
        if geom is not None:
            out.append(replace(geom, world=int(record.get("world", 0))))
    return out


@dataclass(frozen=True)
class ActorGeometry:
    """One deformable actor, read off the model by its particle range.

    The other half of what a mirror needs: the deformable itself. A host recipe
    names a contiguous particle range and a kind, and the connectivity has to be
    *cut* out of the model's global triangle or tetrahedron table -- rows wholly
    inside the range, re-indexed to start at zero. Two engines wanted the same
    cut, which is why it is here.

    Attributes:
        kind: A registered actor kind. ``"shell"`` (a membrane: triangles) and
            ``"tet"`` (a volume) are the two registered here; a domain pack
            registers its own beside them (a centerline of ordered nodes is
            one such).
        rest: ``(count, 3)`` world coordinates. Already world -- the entity baked
            its pose into the builder -- so an actor placed at identity has local
            coordinates that *are* world, and every read-back is direct.
        tris / tets: connectivity, re-indexed local to the range.
        young / poisson: for a ``tet``, inverted back out of
            ``model.tet_materials``; ``None`` for a shell, whose moduli are the
            island backend's rather than the model's (a cloth spec speaks its
            solver's stiffnesses, not Young and Poisson).
        params: the entity's :class:`~robosimstudio.specs.physics.HostActor` params, verbatim.
    """

    name: str
    kind: str
    start: int
    count: int
    rest: np.ndarray
    tris: np.ndarray | None = None
    tets: np.ndarray | None = None
    young: float | None = None
    poisson: float | None = None
    params: dict = field(default_factory=dict)


ActorReader = Callable[[Any, "ActorGeometry"], "ActorGeometry"]
"""``(model, base) -> ActorGeometry``: fill in what only this kind's layout says.

``base`` already carries the name, the kind, the particle range, the rest
positions and the declared params -- everything derivable without knowing what
kind of thing it is. A reader adds the rest: the connectivity cut out of the
model's global tables, and whatever material the model itself states."""

ACTOR_KINDS: Registry[ActorReader] = Registry("actor kind", ("robosimstudio.domains",))
"""How each kind of deformable actor is read off the model.

**Open, and that is the whole point of it existing.** This was an ``if kind ==
"shell" ... else`` for as long as there were two kinds, and the cost was not the
branch: it was that the set of things a foreign engine could simulate was closed
at two, both of them particle-meshes. A centerline is neither -- an *ordered*
run of particles with no triangle and no tetrahedron in it -- so a curve could
not be handed to an engine that has a first-class formulation for one, however
good that formulation was.

Registered by the module that knows the layout, which for a kind the library
does not ship is a domain pack's own module. So this scans the domain packages
on its first lookup, exactly as the entity and material registries do: a kind a
pack ships is registered while the pack is imported, nothing here may name a
pack, and what is registered must not depend on whether anything happened to
import one before asking. The scan is a string and runs inside a lookup, so
importing this module still names nothing above it."""


ACTOR_BUILDERS: Registry[Callable[..., Any]] = Registry("actor builder", ("robosimstudio.domains",))
"""``"<engine>:<kind>"`` -> how that engine builds that kind of actor.

The other half of :data:`ACTOR_KINDS`. Reading a kind off the Newton model was
open -- ``register_actor_kind`` -- and *building* it was not: each engine bundle
kept a ``BUILDERS`` class dict, so handing SuperDex a kind it had never seen
meant editing ``engines/superdex/bridge.py``. The refusal message even said so.
An engine still says what it can build; it says it here, where a third party
can say it too -- a domain pack registers how an engine builds *its* kind from
the pack, and a lookup scans the packs first, as :data:`ACTOR_KINDS` does.
"""


def register_actor_builder(
    engine: str, kind: str, build: Callable[..., Any], *, consumes: Sequence[str] = (), summary: str = ""
) -> Callable[..., Any]:
    """Register how ``engine`` builds one kind of actor, and what that builder reads.

    Args:
        engine: The backend alias the bridge is registered under (``"superdex"``).
        kind: What an entity's :class:`~robosimstudio.specs.physics.HostActor`
            names, and what :func:`register_actor_kind` reads.
        build: The bridge's own calling convention.
        consumes: The actor parameter names this builder reads. The composer
            warns, naming the engine, the kind and the key, when an actor
            states one outside this set -- a material that silently became a
            default is the failure this exists to make loud. Empty declares
            nothing, and the engine-wide ``register_backend(consumes=...)``
            answers instead.
        summary: One line, for a reader of the registry.
    """
    del summary
    declare_consumes(engine, kind, consumes)
    return ACTOR_BUILDERS.register(f"{engine}:{kind}", build, origin=getattr(build, "__module__", ""))


def actor_builder(engine: str, kind: str) -> Callable[..., Any] | None:
    """The registered builder, or ``None`` -- the bridge names what it *does* build."""
    try:
        return ACTOR_BUILDERS.get(f"{engine}:{kind}")
    except KeyError:
        return None


def builders_for(engine: str) -> list[str]:
    """Every kind ``engine`` can build, for a refusal that names them."""
    prefix = f"{engine}:"
    return sorted(name[len(prefix) :] for name in ACTOR_BUILDERS.names() if name.startswith(prefix))


def register_actor_kind(kind: str, read: ActorReader, *, summary: str = "") -> ActorReader:
    """Register how one kind of actor is read off the model.

    Args:
        kind: What an entity's :class:`~robosimstudio.specs.physics.HostActor`
            names. Every engine that can build this kind switches on the same
            string, and one that cannot says so rather than guessing.
        read: The reader.
        summary: One line, for a reader of the registry.
    """
    del summary
    return ACTOR_KINDS.register(kind, read, origin=getattr(read, "__module__", ""))


def _read_shell(model: Any, base: ActorGeometry) -> ActorGeometry:
    """A membrane: the triangles wholly inside the range, re-indexed to zero."""
    start, stop = base.start, base.start + base.count
    rows = model.tri_indices.numpy().reshape(-1, 3) if model.tri_count else np.zeros((0, 3), np.int64)
    inside = (rows.min(axis=1) >= start) & (rows.max(axis=1) < stop) if len(rows) else np.zeros(0, bool)
    tris = rows[inside] - start
    if not len(tris):
        raise ValueError(f"shell actor {base.name!r} owns no triangles in [{start}, {stop})")
    return replace(base, tris=tris)


def _read_tet(model: Any, base: ActorGeometry) -> ActorGeometry:
    """A volume: the tetrahedra inside the range, plus the elasticity the model states."""
    start, stop = base.start, base.start + base.count
    rows = model.tet_indices.numpy().reshape(-1, 4) if model.tet_count else np.zeros((0, 4), np.int64)
    inside = (rows.min(axis=1) >= start) & (rows.max(axis=1) < stop) if len(rows) else np.zeros(0, bool)
    tets = rows[inside] - start
    if not len(tets):
        raise ValueError(f"soft actor {base.name!r} owns no tetrahedra in [{start}, {stop})")
    # Elasticity is inverted back out of the model rather than taken from the
    # spec, so there is exactly one statement of what the body is made of and a
    # bundle rebuilds the same material from the model and the manifest alone.
    materials = model.tet_materials.numpy()
    k_mu, k_lambda = (float(v) for v in materials[np.flatnonzero(inside)[0]][:2])
    young, poisson = young_poisson_from_lame(k_mu, k_lambda)
    return replace(base, tets=tets, young=young, poisson=poisson)


register_actor_kind("shell", _read_shell, summary="A membrane: triangles with thickness-integrated stiffness.")
register_actor_kind("tet", _read_tet, summary="A volume: tetrahedra, with the model's own Lame pair.")


def actor_geometry(model: Any, record: dict) -> ActorGeometry:
    """One recipe actor record, read off the finalized model.

    Raises:
        ValueError: when the range owns no connectivity of its declared kind.
            That is a composition mistake rather than a skippable one -- an
            actor with no triangles is an entity whose particles the composer
            claimed and the builder never wired.
        KeyError: when nothing has registered the declared kind, naming what is
            registered. An unknown kind is refused rather than defaulted,
            because the two defaults available -- shell and tet -- would both
            silently build the wrong thing.
    """
    start = int(record["particle_start"])
    count = int(record["particle_count"])
    kind = str(record.get("kind", "tet"))
    name = str(record.get("name", f"{kind}:{start}"))
    rest = np.asarray(model.particle_q.numpy()[start : start + count], dtype=np.float64)
    base = ActorGeometry(name, kind, start, count, rest, params=dict(record.get("params", {})))
    return ACTOR_KINDS.get(kind)(model, base)
