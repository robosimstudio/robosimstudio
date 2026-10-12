# SPDX-License-Identifier: Apache-2.0
"""Turning a click in the 3D view into a thing in the scene.

viser hands a click over as a *ray* -- an origin and a direction in world space
-- and nothing more, because the browser draws the scene while the server holds
the physics. So the intersection happens here, on the host, against the geometry
the model was actually built from.

Newton has a picker, but it is not public API and it is built for the OpenGL
viewer's right-drag: it applies a force, holds device state, and synchronizes
twice per call. ``ViewerViser.apply_forces`` is a documented no-op, which is why
the studio pushes bodies with :class:`~robosimstudio.core.ObjectDragger`
instead. What is wanted here is only the question answered -- *what did I click*
-- so this asks it directly, in numpy, and leaves force to the draggers.

Two kinds of thing are pickable and they are found differently:

* A **shape** has a surface. Cull by bounding box, then intersect the survivors
  triangle by triangle, so a click that passes between two objects picks the one
  the ray actually enters rather than whichever happens to have the nearer
  centre.
* A **particle** has no surface at all -- a deformable is a cloud -- so the hit
  is the particle nearest the ray, and it is only a hit within a tolerance.
  Without that a click on empty sky would seize whatever cloth was least far
  from the horizon.

Everything is host-side and runs at click rate, not step rate: a scene's
triangles are gathered once, lazily, and reused until the model is replaced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import newton
import numpy as np

from robosimstudio.core.math import transform_matrix
from robosimstudio.core.render.visual import ShapeMesh, shape_meshes

__all__ = ["ScenePicker", "ShapeHit"]


@dataclass(frozen=True)
class ShapeHit:
    """What a ray met first.

    Attributes:
        shape: Model shape index.
        body: The body it hangs off, or -1 for a world-fixed shape.
        point: Where the ray entered the surface [m].
        distance: How far along the ray that was [m].
    """

    shape: int
    body: int
    point: np.ndarray
    distance: float


@dataclass(frozen=True)
class ParticleHit:
    """The particle nearest a ray.

    Attributes:
        index: Model particle index.
        point: The particle's own position, not the point on the ray -- a grab
            wants to start where the cloth *is* [m].
        distance: How far along the ray the closest approach was [m].
        miss: Perpendicular distance from the ray to the particle [m], which is
            how far off the aim was.
    """

    index: int
    point: np.ndarray
    distance: float
    miss: float


class ScenePicker:
    """Ray queries against one compiled model.

    Args:
        model: The finalized model. A model swap (an episode that recompiles)
            invalidates the cache; call :meth:`invalidate` or build another.
        particle_tolerance: How far a ray may pass from a particle and still
            count as having hit it [m].
        max_triangles: Shapes with more triangles than this are picked on their
            bounding box alone. A scanned grocery mesh can carry tens of
            thousands of triangles, and a click is not worth a full sweep of
            them when the box already answers "which object".
    """

    def __init__(self, model: newton.Model, *, particle_tolerance: float = 0.05, max_triangles: int = 20000) -> None:
        self.model = model
        self.particle_tolerance = float(particle_tolerance)
        self.max_triangles = int(max_triangles)
        self._shapes: list[ShapeMesh] | None = None

    def invalidate(self) -> None:
        """Drop the cached geometry. Cheap; it is rebuilt on the next query."""
        self._shapes = None

    # ------------------------------------------------------------------ shapes
    def shape_at(self, state: newton.State, origin: Sequence[float], direction: Sequence[float]) -> ShapeHit | None:
        """The first shape a ray enters, or ``None``."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            return None
        direction = direction / norm

        # A scene may have no rigid bodies at all -- a tuning rig is a cloth, a
        # ground plane and nothing else -- and Newton leaves ``body_q`` unset
        # there rather than allocating an empty array. Only body-attached shapes
        # need it, so the world-fixed ones are still pickable without it.
        body_q = None if state.body_q is None else state.body_q.numpy()
        best: ShapeHit | None = None
        for shape in self._geometry():
            if shape.body >= 0 and body_q is None:
                continue
            world = shape.local if shape.body < 0 else transform_matrix(body_q[shape.body]) @ shape.local
            rot, pos = world[:3, :3], world[:3, 3]
            # Into the shape's own frame, where the cached box and triangles
            # live. The transform is rigid, so the inverse rotation is its
            # transpose and the ray keeps its parameterisation -- a distance
            # measured in here is the same distance out there.
            local_origin = rot.T @ (origin - pos)
            local_dir = rot.T @ direction
            if not _hits_box(local_origin, local_dir, shape.lo, shape.hi):
                continue
            if len(shape.faces) > self.max_triangles:
                distance = _box_entry(local_origin, local_dir, shape.lo, shape.hi)
            else:
                distance = _hits_triangles(local_origin, local_dir, shape.vertices, shape.faces)
            if distance is None or (best is not None and distance >= best.distance):
                continue
            best = ShapeHit(shape=shape.index, body=shape.body, point=origin + direction * distance, distance=distance)
        return best

    def _geometry(self) -> list[ShapeMesh]:
        """Every visible, meshable shape in its own frame. Gathered once.

        Shared with the antipodal sampler rather than gathered twice -- see
        :func:`~robosimstudio.core.render.visual.shape_meshes`, which also
        explains why the ground plane is left out.
        """
        if self._shapes is None:
            self._shapes = shape_meshes(self.model)
        return self._shapes

    # --------------------------------------------------------------- particles
    def particle_at(
        self,
        state: newton.State,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        candidates: Sequence[int] | None = None,
    ) -> ParticleHit | None:
        """The particle nearest the ray, within :attr:`particle_tolerance`.

        Args:
            state: The state to read particle positions from.
            origin: Ray origin [m].
            direction: Ray direction; need not be normalized.
            candidates: Restrict to these particle indices -- one cloth's own,
                where a scene holds several. ``None`` considers every particle.
        """
        if not self.model.particle_count:
            return None
        positions = state.particle_q.numpy()
        pool = (
            np.arange(len(positions), dtype=np.int64) if candidates is None else np.asarray(candidates, dtype=np.int64)
        )
        if len(pool) == 0:
            return None

        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            return None
        direction = direction / norm

        points = positions[pool].astype(np.float64)
        relative = points - origin
        along = relative @ direction
        perpendicular = np.linalg.norm(relative - np.outer(along, direction), axis=1)
        # Behind the camera is not clickable, and a particle at the origin is
        # not a pick either.
        perpendicular = np.where(along > 0.0, perpendicular, np.inf)
        nearest = int(np.argmin(perpendicular))
        if not np.isfinite(perpendicular[nearest]) or perpendicular[nearest] > self.particle_tolerance:
            return None
        return ParticleHit(
            index=int(pool[nearest]),
            point=points[nearest],
            distance=float(along[nearest]),
            miss=float(perpendicular[nearest]),
        )


def _hits_box(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Slab test. An axis the ray is parallel to contributes infinities, which
    ``nanmin``/``nanmax`` absorb along with the 0*inf the degenerate case makes."""
    return _box_entry(origin, direction, lo, hi) is not None


def _box_entry(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float | None:
    """Where the ray enters the box, or ``None`` if it misses."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / direction
        t0 = (lo - origin) * inverse
        t1 = (hi - origin) * inverse
    near = float(np.nanmax(np.minimum(t0, t1)))
    far = float(np.nanmin(np.maximum(t0, t1)))
    if far < max(near, 0.0):
        return None
    return max(near, 0.0)


def _hits_triangles(
    origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray, eps: float = 1.0e-9
) -> float | None:
    """Möller-Trumbore over every triangle at once, double sided.

    Double sided because a click should land on the inside of a bin's wall as
    readily as the outside, and a mesh whose winding did not survive its import
    should still be selectable.
    """
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0

    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(det) <= eps
    inverse = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))

    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inverse
    t = np.einsum("ij,ij->i", edge2, qvec) * inverse

    hit = ~parallel & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps)
    if not hit.any():
        return None
    return float(t[hit].min())
