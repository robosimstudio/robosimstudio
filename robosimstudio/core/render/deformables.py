# SPDX-License-Identifier: Apache-2.0
"""A scene's deformables as meshes and points a renderer with no particle pass of its own can draw.

The tiled tracer draws particles natively and the path tracer logs each
deformable through its viewer; a rasteriser or an outside renderer (``gl``,
``blender``) is handed meshes and has to be told what a cloth, a rope or a
heap looks like. This is that statement, read off the scene's
:class:`~robosimstudio.scene.handles.DeformableHandle` table the way the path
tracer reads it:

* a handle with triangles (a sheet, a bag, a soft body's surface) is a mesh of
  fixed topology whose vertices are its particles;
* a handle with a ``tube_radius`` (a rope's centerline) is a tube swept along
  its particles (:func:`~robosimstudio.core.render.visual.tube_points`);
* a particle body (:func:`~robosimstudio.core.render.particles.draws_points`)
  is the surface its handle grows over its particles, for a renderer that can
  redraw a mesh of changing topology every frame (``surfaces=True``), and
  otherwise its particles as spheres at ``point_radius``;
* nothing else, as the path tracer draws a scene: a proxy particle a coupling
  adds is in no handle and not in the picture.

With no handle table at all -- a renderer built by hand over a bare model --
the deformables are drawn as the tiled tracer draws them: the model's
triangles (``model.tri_indices``) as one mesh, and each active particle no
triangle uses as a sphere at its own radius.

The topology is fixed at construction, so a renderer allocates once and moves
vertices each frame (:meth:`DeformableMesh.vertices`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .palette import DEFORMABLE
from .particles import draws_points
from .visual import TUBE_SIDES, tube_points, tube_triangles

__all__ = ["DeformableMesh", "Drawables", "ParticlePoints", "ParticleSurface", "deformable_drawables", "vertex_normals"]


def vertex_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted smooth normals, so a sheet does not render faceted. Unit length; 0 for an unused vertex."""
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    n = np.zeros_like(p)
    if len(t):
        face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        corners = t.reshape(-1)
        for axis in range(3):  # a scatter-add by bincount: np.add.at is an order of magnitude slower
            n[:, axis] = np.bincount(corners, weights=np.repeat(face[:, axis], 3), minlength=len(p))[: len(p)]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-12)


@dataclass
class DeformableMesh:
    """One deformable drawn as a mesh of fixed topology.

    Attributes:
        name: The deformable's scene name (``""`` for the whole model's triangles).
        particles: Indices into ``particle_q``: the vertices of a sheet, the nodes of a tube.
        triangles: ``int32 [k, 3]`` into this mesh's own vertices.
        color: Display RGB in [0, 1], as the palette states colours.
        tube_radius: Above 0, the mesh is a tube of this radius round ``particles``.
    """

    name: str
    particles: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float] = DEFORMABLE
    tube_radius: float = 0.0
    sides: int = TUBE_SIDES

    @property
    def vertex_count(self) -> int:
        """How many vertices :meth:`vertices` returns, every frame."""
        n = len(self.particles)
        return n * self.sides + 2 if self.tube_radius > 0.0 else n

    def positions(self, particle_q: np.ndarray) -> np.ndarray:
        """This frame's vertices alone, ``float64 [vertex_count, 3]``, for a renderer that makes its own normals."""
        q = self._rows(particle_q)
        return tube_points(q, self.tube_radius, self.sides)[0] if self.tube_radius > 0.0 else q

    def vertices(self, particle_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """This frame's ``(vertices, normals)``, ``float64 [vertex_count, 3]`` each, in world metres."""
        if self.tube_radius > 0.0:
            return tube_points(self._rows(particle_q), self.tube_radius, self.sides)
        q = self._rows(particle_q)
        return q, vertex_normals(q, self.triangles)

    def _rows(self, particle_q: np.ndarray) -> np.ndarray:
        """This mesh's particles' positions: the rows gathered, then cast."""
        return np.asarray(np.asarray(particle_q)[self.particles], dtype=np.float64)


@dataclass
class ParticlePoints:
    """Particles drawn as spheres.

    Attributes:
        name: The deformable's scene name, or ``""`` for particles no handle names.
        particles: Indices into ``particle_q``.
        radii: One radius per particle [m].
        color: Display RGB in [0, 1].
    """

    name: str
    particles: np.ndarray
    radii: np.ndarray
    color: tuple[float, float, float] = DEFORMABLE

    def positions(self, particle_q: np.ndarray) -> np.ndarray:
        """This frame's sphere centres, ``float64 [n, 3]``."""
        return np.asarray(np.asarray(particle_q)[self.particles], dtype=np.float64)


@dataclass
class ParticleSurface:
    """A particle body drawn as the surface its handle grows over its particles, every frame anew.

    Attributes:
        name: The deformable's scene name.
        particles: Indices into ``particle_q``.
        surface: The handle's ``surface`` callable (see
            :class:`~robosimstudio.scene.handles.DeformableHandle`).
        color: Display RGB in [0, 1].
    """

    name: str
    particles: np.ndarray
    surface: Any
    color: tuple[float, float, float] = DEFORMABLE

    def mesh(self, particle_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """This frame's ``(vertices float32 [m, 3], triangles int32 [k, 3])``; both empty when there is none."""
        from .particles import surface_mesh

        verts, tris, _ = surface_mesh(self.surface, np.asarray(particle_q)[self.particles])
        return np.asarray(verts, dtype=np.float32).reshape(-1, 3), np.asarray(tris, dtype=np.int32).reshape(-1, 3)


@dataclass
class Drawables:
    """Every deformable of a model, as meshes, as points, and as surfaces regrown each frame."""

    meshes: list[DeformableMesh] = field(default_factory=list)
    points: list[ParticlePoints] = field(default_factory=list)
    surfaces: list[ParticleSurface] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.meshes or self.points or self.surfaces)


def _particle_radii(model: Any, index: np.ndarray) -> np.ndarray:
    radius = getattr(model, "particle_radius", None)
    if radius is None:
        return np.full(len(index), 0.005)
    values = np.asarray(radius.numpy() if hasattr(radius, "numpy") else radius, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full(len(index), float(values[0]))
    return values[index]


def deformable_drawables(
    model: Any,
    deformables: Mapping[str, Any] | None = None,
    colors: Mapping[str, Sequence[float]] | None = None,
    *,
    surfaces: bool = False,
) -> Drawables:
    """The model's deformables as :class:`DeformableMesh`, :class:`ParticlePoints` and :class:`ParticleSurface`.

    Args:
        model: The finalized model.
        deformables: ``name -> DeformableHandle`` (a scene's ``handles.deformables``), or ``None``.
        colors: ``name -> RGB`` for those (``Sim``'s ``deformable_colors``); a name left out is the palette's.
        surfaces: Whether the caller can draw a mesh whose topology changes every frame: a particle body
            whose handle offers a ``surface`` (a liquid, a dough) is then that surface, as the path tracer
            draws it, rather than its grains.
    """
    out = Drawables()
    count = int(getattr(model, "particle_count", 0) or 0)
    if not count:
        return out
    colors = dict(colors or {})
    for name, handle in (deformables or {}).items():
        particles = np.asarray(getattr(handle, "particles", ()), dtype=np.int64).reshape(-1)
        if not len(particles) or particles.min() < 0 or particles.max() >= count:
            continue  # a handle into another model's particles: nothing of it is drawn rather than the wrong ones
        color = tuple(float(v) for v in colors.get(name, DEFORMABLE))[:3]
        triangles = np.asarray(getattr(handle, "triangles", ()), dtype=np.int64).reshape(-1, 3)
        tube = float(getattr(handle, "tube_radius", 0.0) or 0.0)
        if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(particles)):
            import warnings

            warnings.warn(
                f"deformable {name!r}: its triangles index past its {len(particles)} particles (global indices "
                "where local ones belong?); it is not drawn",
                stacklevel=2,
            )
            continue
        if len(triangles):
            # only the particles a triangle uses: a soft body's inner nodes are not on its surface
            used, local = np.unique(triangles, return_inverse=True)
            out.meshes.append(DeformableMesh(name, particles[used], local.reshape(-1, 3).astype(np.int32), color))
        elif tube > 0.0 and len(particles) >= 2:
            out.meshes.append(DeformableMesh(name, particles, tube_triangles(len(particles)), color, tube))
        elif draws_points(handle) and surfaces and getattr(handle, "surface", None) is not None:
            out.surfaces.append(ParticleSurface(name, particles, handle.surface, color))
        elif draws_points(handle):
            radii = np.full(len(particles), float(handle.point_radius))
            out.points.append(ParticlePoints(name, particles, radii, color))
        # a handle that says nothing about its look (no triangles, tube or point radius) is not drawn, as the path
        # tracer does not draw it
    if deformables is not None:
        return out  # a scene's handle table: its deformables and nothing else, as the path tracer draws them
    covered = np.zeros(count, dtype=bool)
    # no handle table (a renderer built by hand over a bare model): the model's triangles as one mesh, as the
    # tiled tracer draws them, and the active particles no triangle uses as spheres
    tri = getattr(model, "tri_indices", None)
    tris = np.asarray(tri.numpy() if hasattr(tri, "numpy") else (tri if tri is not None else ()), dtype=np.int64)
    tris = tris.reshape(-1, 3)
    tris = tris[(tris.min(axis=1) >= 0) & (tris.max(axis=1) < count)] if len(tris) else tris
    if len(tris):
        used, local = np.unique(tris, return_inverse=True)
        out.meshes.append(DeformableMesh("", used, local.reshape(-1, 3).astype(np.int32)))
        covered[used] = True
    flags = getattr(model, "particle_flags", None)
    if flags is not None:  # a parked or inactive particle is not drawn
        import newton

        flags = np.asarray(flags.numpy() if hasattr(flags, "numpy") else flags).reshape(-1)
        if len(flags) == count:
            covered |= (flags & int(newton.ParticleFlags.ACTIVE)) == 0
    rest = np.flatnonzero(~covered)
    if len(rest):
        out.points.append(ParticlePoints("", rest, _particle_radii(model, rest)))
    return out
