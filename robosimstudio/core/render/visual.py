# SPDX-License-Identifier: Apache-2.0
"""The backend-free scene description the visual-matching renderers consume.

A :class:`VisualPart` is a drawable mesh, the Newton body it hangs off, and a
local 4x4 -- and deliberately nothing about how any of it is rasterised. That
is the seam two renderers hang off: a second renderer of the *same* scene
description is worth something, because it can disagree about the picture; a
second re-authoring of the scene would not be.

It also carries a *dynamic* case, which is what makes cloth work here at all.
A deformable has no body to hang off, so :class:`ClothPart` rebuilds its mesh
from ``particle_q`` each frame over fixed triangle indices. The rigid parts
move by pose; the cloth part moves by geometry.

Nothing in this module knows which parts a scene has. The caller passes a list
in -- the toolkit layer builds one from its scene handles and asset paths -- which
is what keeps core free of any knowledge about robots, tasks or asset layouts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np

from ..math import transform_matrix

__all__ = [
    "TUBE_SIDES",
    "ClothPart",
    "ShapeMesh",
    "VisualPart",
    "cloth_triangles",
    "flatten_scene",
    "model_visual_parts",
    "shape_geometry",
    "shape_meshes",
    "tube_points",
    "tube_triangles",
]


@dataclass
class VisualPart:
    """One drawable mesh and where it hangs.

    Attributes:
        mesh: A ``trimesh.Trimesh`` or ``trimesh.Scene``. Typed loosely because
            core must not import trimesh at module level.
        body: Newton body index, or -1 for world-fixed.
        local: 4x4 in that body's frame.
        name: Label, used for material heuristics and for debugging which part
            of a composite is misplaced.
    """

    mesh: Any
    body: int = -1
    local: np.ndarray = field(default_factory=lambda: np.eye(4))
    name: str = ""

    def world_matrix(self, body_q: np.ndarray | None) -> np.ndarray:
        """This part's world 4x4 for one step's body transforms."""
        if self.body < 0 or body_q is None:
            return self.local
        return transform_matrix(body_q[self.body]) @ self.local


@dataclass
class ClothPart:
    """A deformable drawn from live particle positions.

    Attributes:
        particles: Indices into ``particle_q`` this surface's vertices are.
        triangles: Triangle indices *into* :attr:`particles`, shape ``[n, 3]``.
            When a :attr:`thickness` is set these become the closed shell's
            triangles at construction; the declared surface is kept aside for
            the per-frame normals.
        name: Label.
        color: Flat RGB in [0, 1] for backends with no material to sample.
        opacity: Alpha in (0, 1]. Below 1 the surface renders translucent --
            a polymer film is a thing you see the substrate through. Backends
            that cannot blend simply draw it opaque.
        double_sided: Whether to draw back faces. Cloth is a zero-thickness
            surface, so a single-sided draw shows holes wherever it curls.
        thickness: Visual thickness [m]. Zero draws the surface as-is. Set on
            a membrane whose fabric has a real thickness the solver geometry
            cannot carry, so an exported video shows the same solid cloth the
            studio does -- see :mod:`robosimstudio.core.shell`. A surface that
            cannot be shelled (non-manifold) quietly stays a film, which is
            the same fallback the studio takes.
    """

    particles: np.ndarray
    triangles: np.ndarray
    name: str = "cloth"
    color: tuple[float, float, float] = (0.85, 0.85, 0.88)
    opacity: float = 1.0
    double_sided: bool = True
    thickness: float = 0.0

    def __post_init__(self) -> None:
        self._surface = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)
        if self.thickness > 0.0:
            from ..shell import shell_faces

            try:
                self.triangles = shell_faces(self._surface, len(np.asarray(self.particles).reshape(-1)))
            except ValueError:
                self.thickness = 0.0

    def vertices(self, particle_q: np.ndarray) -> np.ndarray:
        """This cloth's drawn vertices for one step [m].

        The gathered surface, or -- with a thickness -- both faces of its
        shell, matching the topology :attr:`triangles` was rewritten to.
        """
        verts = np.asarray(particle_q, dtype=np.float64)[self.particles]
        if self.thickness > 0.0:
            from ..shell import shell_vertices

            return shell_vertices(verts, self._surface, self.thickness)
        return verts


def cloth_triangles(model: newton.Model, particles: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """The triangles of one deformable, re-indexed to a local vertex numbering.

    ``Model.tri_indices`` numbers every triangle in the scene against the global
    particle array. A renderer wants one mesh per deformable with vertices
    starting at 0, so this selects the triangles whose corners are all in
    ``particles`` and rewrites their indices.

    Args:
        model: The finalized model.
        particles: The deformable's particle indices, as a scene handle records
            them.

    Returns:
        ``(particles, triangles)`` -- the (possibly reordered) particle index
        array and triangles into it, shape ``[n, 3]``.
    """
    idx = np.unique(np.asarray(particles, dtype=np.int64))
    if model.tri_count == 0 or not len(idx):
        return idx.astype(np.int32), np.zeros((0, 3), dtype=np.int32)
    tris = np.asarray(model.tri_indices.numpy(), dtype=np.int64).reshape(-1, 3)
    keep = np.all(np.isin(tris, idx), axis=1)
    tris = tris[keep]
    # Map global particle index -> position in idx, in one pass.
    lookup = np.full(int(idx.max()) + 1, -1, dtype=np.int64)
    lookup[idx] = np.arange(len(idx))
    return idx.astype(np.int32), lookup[tris].astype(np.int32)


TUBE_SIDES = 12
"""Vertices round each ring of a drawn centerline: the path tracer's and the page's."""


def tube_triangles(nodes: int, sides: int = TUBE_SIDES) -> np.ndarray:
    """The faces of a tube over ``nodes`` rings of ``sides``, for :func:`tube_points`' vertices.

    Two triangles per quad between consecutive rings, then a fan closing each
    end on the two centre vertices :func:`tube_points` appends. Fixed for a
    centerline's life: only the vertices move.
    """
    ring, k = np.meshgrid(np.arange(nodes - 1), np.arange(sides), indexing="ij")
    a, b = ring * sides + k, ring * sides + (k + 1) % sides
    quads = np.stack([a, a + sides, b, b, a + sides, b + sides], axis=-1).reshape(-1, 3)
    k = np.arange(sides)
    last = (nodes - 1) * sides
    start = np.stack([np.full(sides, nodes * sides), k, (k + 1) % sides], axis=-1)
    end = np.stack([np.full(sides, nodes * sides + 1), last + (k + 1) % sides, last + k], axis=-1)
    caps = np.stack([start, end], axis=1).reshape(-1, 3)
    return np.concatenate([quads, caps]).astype(np.int32)


def tube_points(centre: np.ndarray, radius: float, sides: int = TUBE_SIDES) -> tuple[np.ndarray, np.ndarray]:
    """A tube of ``radius`` round a polyline: ring vertices and outward normals, then the two end centres.

    How a centerline is drawn -- a rope whose particles are its nodes and which
    owns no triangle, given a radius by its deformable handle's ``tube_radius``.
    The ring frames are parallel-transported along the curve, so the tube does
    not twist or pinch where the cable turns. ``[nodes * sides + 2, 3]`` each,
    for :func:`tube_triangles`.
    """
    c = np.asarray(centre, dtype=np.float64)
    n = len(c)
    tangent = np.gradient(c, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(tangent[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(tangent[0], helper)
    u /= np.linalg.norm(u)
    frames_u = np.empty((n, 3))
    for i in range(n):
        if i:
            u = u - np.dot(u, tangent[i]) * tangent[i]
            u /= max(np.linalg.norm(u), 1.0e-12)
        frames_u[i] = u
    frames_v = np.cross(tangent, frames_u)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.cos(angles)[None, :, None] * frames_u[:, None, :] + np.sin(angles)[None, :, None] * frames_v[:, None, :]
    points = (c[:, None, :] + radius * ring).reshape(-1, 3)
    normals = ring.reshape(-1, 3)
    points = np.concatenate([points, c[:1], c[-1:]], axis=0)
    normals = np.concatenate([normals, -tangent[:1], tangent[-1:]], axis=0)
    return points, normals


def model_visual_parts(
    model: newton.Model,
    *,
    exclude_shapes: Sequence[int] = (),
    plane_extent: float = 100.0,
) -> list[VisualPart]:
    """One part per visible shape the finalized model itself carries.

    The scene description for a task whose appearance lives *in the model*
    rather than in separately loadable mesh files. An MJCF export carries its
    visual geoms -- meshes with colours and texture paths, coloured primitives
    -- through Newton's importer onto ``shape_source``, ``shape_color`` and
    ``shape_scale``, and this reads them back the same way the interactive
    viewers do. A caller-supplied asset list can only ever cover the entities
    somebody wrote specs for; this covers what the model was actually built
    from, robot included.

    Args:
        model: The finalized model.
        exclude_shapes: Shape indices something else already draws -- the
            entities a caller placed authored meshes for, or the parts a
            visual-matching composite supplies from a photograph.
        plane_extent: Half-extent stand-in for a plane whose scale claims
            infinity. Anything beyond the far clip is gone anyway; this just
            has to outrun every camera.

    Returns:
        Rigid parts, one per drawn shape, placed by the shape's own transform.
    """
    import trimesh

    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_xform = model.shape_transform.numpy()
    colors = model.shape_color.numpy() if model.shape_color is not None else None
    labels = list(getattr(model, "shape_label", None) or ())
    excluded = {int(s) for s in np.asarray(exclude_shapes, dtype=np.int64).reshape(-1)}
    visible = int(newton.ShapeFlags.VISIBLE)

    parts: list[VisualPart] = []
    for s in range(model.shape_count):
        if s in excluded or not (int(shape_flags[s]) & visible):
            continue
        geom = _shape_trimesh(trimesh, model.shape_source[s], int(shape_type[s]), shape_scale[s], plane_extent)
        if geom is None:
            continue
        color = colors[s] if colors is not None else (0.5, 0.5, 0.5)
        _apply_shape_visual(trimesh, geom, model.shape_source[s], color)
        parts.append(
            VisualPart(
                mesh=geom,
                body=int(shape_body[s]),
                local=transform_matrix(shape_xform[s]),
                name=labels[s] if s < len(labels) else f"shape_{s}",
            )
        )
    return parts


def shape_geometry(
    src: Any, geo_type: int, scale: np.ndarray, plane_extent: float = 100.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """One model shape as bare ``(vertices, faces)``, dimensions baked in.

    Primitive meshing goes through :meth:`newton.Mesh.create_\\*` so the
    conventions -- half-extent boxes, Z-axis capsules and cylinders, radius in
    ``scale[0]`` -- are Newton's own rather than a re-derivation of them.

    Split out of :func:`_shape_trimesh` because trimesh is an optional
    dependency and callers need the geometry without it: the studio's picker
    intersects rays against it, and the antipodal sampler reads its faces.

    Returns:
        Vertices ``[n, 3]`` and triangle indices ``[m, 3]``, or ``None`` for a
        shape with no drawable geometry (an unmeshed type, or a mesh whose
        source did not survive the import).
    """
    kind = newton.GeoType(geo_type)
    if kind in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
        if src is None:
            return None
        vertices = np.asarray(src.vertices, dtype=np.float64) * np.asarray(scale[:3], dtype=np.float64)
        return vertices, np.asarray(src.indices, dtype=np.int64).reshape(-1, 3)

    radius, half = float(scale[0]), float(scale[1]) if len(scale) > 1 else 0.0
    if kind == newton.GeoType.SPHERE:
        mesh = newton.Mesh.create_sphere(radius, compute_inertia=False)
    elif kind == newton.GeoType.BOX:
        mesh = newton.Mesh.create_box(float(scale[0]), float(scale[1]), float(scale[2]), compute_inertia=False)
    elif kind == newton.GeoType.CAPSULE:
        mesh = newton.Mesh.create_capsule(radius, half, up_axis=newton.Axis.Z, compute_inertia=False)
    elif kind == newton.GeoType.CYLINDER:
        mesh = newton.Mesh.create_cylinder(radius, half, up_axis=newton.Axis.Z, compute_inertia=False)
    elif kind == newton.GeoType.CONE:
        mesh = newton.Mesh.create_cone(radius, half, up_axis=newton.Axis.Z, compute_inertia=False)
    elif kind == newton.GeoType.ELLIPSOID:
        mesh = newton.Mesh.create_ellipsoid(float(scale[0]), float(scale[1]), float(scale[2]), compute_inertia=False)
    elif kind == newton.GeoType.PLANE:
        width = float(scale[0]) if scale[0] > 0.0 else plane_extent
        length = float(scale[1]) if len(scale) > 1 and scale[1] > 0.0 else plane_extent
        mesh = newton.Mesh.create_plane(width, length, compute_inertia=False)
    else:
        return None
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3),
    )


@dataclass(frozen=True)
class ShapeMesh:
    """One shape's triangles, in the shape's own frame.

    Attributes:
        index: Model shape index.
        body: The body it hangs off, or -1 for a world-fixed shape.
        local: 4x4 placing the shape within that body.
        vertices: ``[n, 3]``, dimensions already baked in.
        faces: Triangle indices ``[m, 3]``.
        lo: Per-axis minimum of :attr:`vertices`, for a cheap cull.
        hi: Per-axis maximum.
    """

    index: int
    body: int
    local: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def shape_meshes(model: newton.Model, *, skip_planes: bool = True, flag: int | None = None) -> list[ShapeMesh]:
    """Every visible shape of a model as bare triangles. Gathered once.

    The geometry two host-side queries need and neither of them should be
    re-deriving: the studio's picker intersects rays against it, and the
    antipodal grasp sampler
    (the antipodal grasp sampler) searches it for
    opposed faces.

    Args:
        model: The finalized model.
        skip_planes: Drop ``GeoType.PLANE``. A ground plane claims a hundred
            metres, so it swallows every click that missed everything else and
            it is never a thing anybody means to pick or grasp.
        flag: Which :class:`newton.ShapeFlags` bit a shape has to carry, as an
            int. Defaults to ``VISIBLE``, which is what a picker and a mesh
            search want.

            ``COLLIDE_SHAPES`` is the other one worth asking for, and it is a
            *different set of shapes*, not a filter on the same one. Being drawn
            and stopping a finger are separate facts and a scene may carry
            either without the other: an object imported from a bare OBJ is drawn as a mesh
            whose collider bit is off and collides as a substitute the compiler
            fitted with its visual bit off, so the two gathers share no shape at
            all on it, and the tool-sorting export's visual geoms carry ``contype="0"``
            -- drawn, never touched. Whether the substitute's *bounds* match the
            drawing depends on which one it is, and it has been both: a hull
            agrees with the mesh, and the fitted box those objects carried until
            this week stood up to 28.6 mm proud of it. Anything asking "what
            would the hand hit" wants this set rather than the drawn one, and
            asking the drawn one is right exactly when the answer is going to be
            shown to a person.

    Returns:
        One entry per shape carrying ``flag`` that meshes to something. Shapes
        whose geometry did not survive the import are left out.
    """
    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_xform = model.shape_transform.numpy()
    wanted = int(newton.ShapeFlags.VISIBLE) if flag is None else int(flag)

    meshes: list[ShapeMesh] = []
    for s in range(model.shape_count):
        if not (int(shape_flags[s]) & wanted):
            continue
        if skip_planes and newton.GeoType(int(shape_type[s])) == newton.GeoType.PLANE:
            continue
        geometry = shape_geometry(model.shape_source[s], int(shape_type[s]), shape_scale[s])
        if geometry is None:
            continue
        vertices, faces = geometry
        if len(vertices) == 0 or len(faces) == 0:
            continue
        meshes.append(
            ShapeMesh(
                index=s,
                body=int(shape_body[s]),
                local=transform_matrix(shape_xform[s]),
                vertices=vertices,
                faces=faces,
                lo=vertices.min(axis=0),
                hi=vertices.max(axis=0),
            )
        )
    return meshes


def _shape_trimesh(trimesh_mod: Any, src: Any, geo_type: int, scale: np.ndarray, plane_extent: float) -> Any | None:
    """A ``trimesh.Trimesh`` of one model shape, dimensions baked in."""
    geometry = shape_geometry(src, geo_type, scale, plane_extent)
    if geometry is None:
        return None
    vertices, faces = geometry
    return trimesh_mod.Trimesh(vertices=vertices, faces=faces, process=False)


def _apply_shape_visual(trimesh_mod: Any, geom: Any, src: Any, color: Any) -> None:
    """Give one shape's trimesh the appearance the model recorded for it.

    A texture with matching UVs becomes a textured material; everything else
    gets the shape's flat colour. Both land as ``TextureVisuals`` over a
    ``PBRMaterial`` because that is the one form each backend's material path
    already reads -- a bare ``ColorVisuals`` renders pybullet's default grey.
    """
    image = _texture_pil(getattr(src, "texture", None)) if src is not None else None
    uv = getattr(src, "uvs", None) if src is not None else None
    if uv is None and src is not None:
        uv = getattr(src, "_uvs", None)
    if image is not None and uv is not None and len(uv) == len(geom.vertices):
        material = trimesh_mod.visual.material.PBRMaterial(
            baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.7
        )
        geom.visual = trimesh_mod.visual.TextureVisuals(uv=np.asarray(uv, dtype=np.float64), material=material)
        return
    rgba = np.clip(np.append(np.asarray(color, dtype=np.float64).reshape(-1)[:3], 1.0), 0.0, 1.0)
    material = trimesh_mod.visual.material.PBRMaterial(baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.7)
    geom.visual = trimesh_mod.visual.TextureVisuals(material=material)


def _texture_pil(texture: Any) -> Any | None:
    """The texture as a PIL image, from the path or array the importer kept."""
    if texture is None:
        return None
    try:
        from PIL import Image

        if isinstance(texture, str):
            return Image.open(texture).convert("RGB")
        array = np.asarray(texture)
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    except Exception:
        return None
    return None


def flatten_scene(mesh: Any) -> list[tuple[Any, np.ndarray]]:
    """Split a ``trimesh.Scene`` into ``(Trimesh, local 4x4)`` pairs.

    A single ``Trimesh`` comes back as one pair at identity, so callers do not
    branch on which of the two an asset loaded as.
    """
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        out = []
        for node in mesh.graph.nodes_geometry:
            xform, geom_name = mesh.graph[node]
            geom = mesh.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh):
                out.append((geom, np.asarray(xform, dtype=np.float64)))
        return out
    return [(mesh, np.eye(4))]
