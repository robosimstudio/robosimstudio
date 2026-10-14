# SPDX-License-Identifier: Apache-2.0
"""How close two sets of shapes come, over the pairs the model still tests.

The question a planner has to answer before it commands anything: *if I put the
arm here, does it hit the other arm?* Newton can answer it after the fact --
step, then read the contacts out of the solver
(:class:`~robosimstudio.core.probes.contact.ContactProbe`) -- and after the
fact is too late for a plan that has to be rejected or moved.

So this is the same question asked of geometry alone, at any joint
configuration, with no solver and no stepping. Two things make it honest rather
than a second opinion:

* **It reads the model's own collision filtering.** A pair the model filters is
  not a collision however close it comes -- a gripper's two pads meet by design
  -- so the minimum is taken over the pairs the broad phase would actually
  test. On a robot whose shapes are all filtered against each other the answer
  is ``inf``, which is the truth: those parts cannot collide, so no plan needs
  to keep them apart. That is what lets a caller run the same search whether or
  not the pairs are live and get the old behaviour for free when they are not.
* **Every collider becomes a sphere swept along a segment**, which is exact for
  the capsules and spheres an arm's colliders overwhelmingly are, and a tight
  over-estimate for a box (its longest axis, thickened by the other two
  half-extents' diagonal). Over-estimating shrinks the reported gap, so the
  measure errs towards saying two parts are closer than they are -- the safe
  direction for a clearance gate. Anything else falls back to its bounding
  sphere.

Measured against the physics on the bimanual fold -- 215 samples through a
carry that ends with the two forearms crossed -- "gap at or below 2 mm" and
"the solver reports contact between these arms" agreed on 214 of them.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp

__all__ = ["ClearanceProbe"]

_EPS = 1.0e-12


def _segment_segment(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Closest distance between every segment of one set and of the other.

    Ericson's ``ClosestPtSegmentSegment``, vectorised to an ``[n, m]`` table.
    Degenerate (zero-length) segments are points and fall out of the same
    formula through the ``a``/``e`` guards, which is why a sphere needs no
    special case beyond Ericson's point-against-segment branch.
    """
    d1, d2 = q1 - p1, q2 - p2
    r = p1[:, None, :] - p2[None, :, :]
    a = np.einsum("ik,ik->i", d1, d1)[:, None]
    e = np.einsum("jk,jk->j", d2, d2)[None, :]
    b = np.einsum("ik,jk->ij", d1, d2)
    c = np.einsum("ijk,ik->ij", r, d1)
    f = np.einsum("ijk,jk->ij", r, d2)

    a_ok, e_ok = a > _EPS, e > _EPS
    denom = a * e - b * b
    s = np.where(denom > _EPS, np.clip((b * f - c * e) / np.where(denom > _EPS, denom, 1.0), 0.0, 1.0), 0.0)
    t = np.where(e_ok, (b * s + f) / np.where(e_ok, e, 1.0), 0.0)
    # Where the first pass put t outside the segment, clamp it and re-solve s
    # against that end -- the standard two-pass fix-up.
    s = np.where(t < 0.0, np.clip(-c / np.where(a_ok, a, 1.0), 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / np.where(a_ok, a, 1.0), 0.0, 1.0), s)
    # Second segment a point: t is already 0, and s is the first segment's
    # closest point to it -- denom is 0 there, so the first pass left s at 0.
    s = np.where(a_ok & ~e_ok, np.clip(-c / np.where(a_ok, a, 1.0), 0.0, 1.0), s)
    s = np.where(a_ok, s, 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest_1 = p1[:, None, :] + d1[:, None, :] * s[..., None]
    closest_2 = p2[None, :, :] + d2[None, :, :] * t[..., None]
    return np.linalg.norm(closest_1 - closest_2, axis=-1)


def _rotate(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = quat[:3], quat[3]
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def _rotate_all(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """:func:`_rotate` over matching stacks of quaternions and vectors."""
    u, w = quat[:, :3], quat[:, 3:4]
    uv = np.einsum("ij,ij->i", u, v)[:, None]
    uu = np.einsum("ij,ij->i", u, u)[:, None]
    return 2.0 * uv * u + (w * w - uu) * v + 2.0 * w * np.cross(u, v)


class ClearanceProbe:
    """Closest approach between two shape sets, at any joint configuration.

    One instance per built model; the per-shape geometry is read once. Queries
    are numpy over a few dozen rows plus, for the joint-space form, one forward
    kinematics evaluation.

    Args:
        model: The finalized model.
    """

    def __init__(self, model: newton.Model) -> None:
        self.model = model
        self.shape_body = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.collides = np.asarray([(int(f) & collide) != 0 for f in flags])
        """Whether each shape takes part in shape-shape collision at all."""
        self.filtered: set[tuple[int, int]] = {
            (int(a), int(b)) for a, b in getattr(model, "shape_collision_filter_pairs", ())
        }
        """The pairs the broad phase will not test, as ``(low, high)``."""
        self._local_a, self._local_b, self.radius = self._swept_spheres(model)
        self._testable: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray] = {}
        """Per shape-set pair, which entries of the table the broad phase tests.

        Cached because a planner asks the same two arms about hundreds of
        configurations and the filter set does not move between them -- built
        once at a thousand-odd set lookups instead of once per query."""
        # The scratch state is allocated on first use, not here: a caller that
        # already has body poses -- a probe watching the live scene -- never
        # needs one, and building one is what would make this class require a
        # whole model rather than the six arrays it actually reads.
        self._state = None
        self._joint_q: wp.array | None = None

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _swept_spheres(model: newton.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every shape as a segment in its body's frame, plus a radius."""
        n = model.shape_count
        xform = model.shape_transform.numpy()
        types = model.shape_type.numpy()
        scale = model.shape_scale.numpy()
        fallback = model.shape_collision_radius.numpy()
        first = np.zeros((n, 3))
        last = np.zeros((n, 3))
        radius = np.zeros(n)
        for shape in range(n):
            origin = np.asarray(xform[shape][:3], dtype=np.float64)
            quat = np.asarray(xform[shape][3:7], dtype=np.float64)
            kind = int(types[shape])
            size = np.asarray(scale[shape], dtype=np.float64)
            if kind == int(newton.GeoType.SPHERE):
                first[shape] = last[shape] = origin
                radius[shape] = size[0]
            elif kind == int(newton.GeoType.CAPSULE):
                # Newton's capsule is (radius, half-height) about local +z.
                half = _rotate(quat, np.array([0.0, 0.0, size[1]]))
                first[shape], last[shape] = origin - half, origin + half
                radius[shape] = size[0]
            elif kind == int(newton.GeoType.BOX):
                # The longest axis as the segment, the other two as a radius:
                # a capsule that contains the box, tight on a flat pad and only
                # loose at the corners.
                longest = int(np.argmax(size[:3]))
                axis = np.zeros(3)
                axis[longest] = size[longest]
                half = _rotate(quat, axis)
                first[shape], last[shape] = origin - half, origin + half
                radius[shape] = float(np.hypot(*[size[i] for i in range(3) if i != longest]))
            elif kind == int(newton.GeoType.MESH):
                segment = ClearanceProbe._mesh_segment(model, shape, origin, quat, size)
                if segment is None:
                    first[shape] = last[shape] = origin
                    radius[shape] = float(fallback[shape])
                else:
                    first[shape], last[shape], radius[shape] = segment
            else:
                # A cone, a plane: its own bounding sphere. Conservative
                # and never wrong about a collision, only about how much room
                # there is when there is no collision.
                first[shape] = last[shape] = origin
                radius[shape] = float(fallback[shape])
        return first, last, radius

    @staticmethod
    def _mesh_segment(model, shape: int, origin, quat, size):
        """A mesh's swept sphere from its own vertices, or ``None``.

        The fallback ``(shape origin, shape_collision_radius)`` pair is not a
        bound: the radius is not measured about the shape origin, so nothing
        guarantees the sphere contains the mesh — and on a robot whose links
        are all meshes the probe was **anti-conservative**, the one direction
        a clearance probe must never be. Measured on a USB-C insertion scene:
        `sim.clearance("arm:support", "bulkhead")` reported **+44 mm of room
        with the pads physically pinching the barrel** (per-pair table: the
        fingertip mesh read +115 mm while touching), so a policy's honest
        contact check failed 12/12 on a brace the evaluator's own tape shows
        seated. The same mismatch read -228 mm for two arms that never
        touched. Deriving the segment and radius from the mesh's actual
        vertices (principal axis through the centroid; radius the largest
        vertex distance to that segment) makes the capsule contain the mesh
        by construction, so the reported gap is once again a floor on the
        true gap.
        """
        source = getattr(model, "shape_source", None)
        mesh = source[shape] if source is not None and shape < len(source) else None
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or len(vertices) == 0:
            return None
        v = np.asarray(vertices, dtype=np.float64) * np.asarray(size[:3], dtype=np.float64)
        v = origin + _rotate_all(np.repeat(quat[None, :], len(v), axis=0), v)
        centroid = v.mean(axis=0)
        centred = v - centroid
        # Principal direction of the vertex cloud; degenerate clouds (a flat
        # or point-like mesh) fall back to a plain bounding sphere.
        _, singular, basis = np.linalg.svd(centred, full_matrices=False)
        if singular[0] < 1.0e-9:
            return centroid, centroid, float(np.linalg.norm(centred, axis=1).max())
        axis = basis[0]
        t = centred @ axis
        first = centroid + axis * float(t.min())
        last = centroid + axis * float(t.max())
        # Radius: the largest perpendicular distance from any vertex to the
        # segment's line — every projection lands inside [t.min, t.max] by
        # construction, so this capsule contains every vertex (and, for a
        # convex collision mesh, the whole surface).
        on_segment = centroid + np.outer(t, axis)
        radius = float(np.linalg.norm(v - on_segment, axis=1).max())
        return first, last, radius

    def _world(self, shapes: np.ndarray, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = np.array(self._local_a[shapes], dtype=np.float64)
        last = np.array(self._local_b[shapes], dtype=np.float64)
        bodies = self.shape_body[shapes]
        # A shape welded to the world is already world-framed and stays put.
        moved = bodies >= 0
        if moved.any():
            pose = np.asarray(body_q, dtype=np.float64)[bodies[moved]]
            position, quat = pose[:, :3], pose[:, 3:7]
            first[moved] = position + _rotate_all(quat, first[moved])
            last[moved] = position + _rotate_all(quat, last[moved])
        return first, last

    def _testable_pairs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Which entries of the ``[len(a), len(b)]`` table the broad phase tests."""
        key = (tuple(int(s) for s in a), tuple(int(s) for s in b))
        mask = self._testable.get(key)
        if mask is None:
            mask = np.ones((len(a), len(b)), dtype=bool)
            for i, sa in enumerate(a):
                for j, sb in enumerate(b):
                    if sa == sb or (min(int(sa), int(sb)), max(int(sa), int(sb))) in self.filtered:
                        mask[i, j] = False
            self._testable[key] = mask
        return mask

    # -------------------------------------------------------------- queries
    def _colliding(self, shapes: Sequence[int]) -> np.ndarray:
        idx = np.asarray([int(s) for s in shapes], dtype=np.int64)
        return idx[self.collides[idx]] if len(idx) else idx

    def live_pairs(self, shapes_a: Sequence[int], shapes_b: Sequence[int]) -> int:
        """How many pairs across the two sets the broad phase would test."""
        a, b = self._colliding(shapes_a), self._colliding(shapes_b)
        return 0 if not len(a) or not len(b) else int(self._testable_pairs(a, b).sum())

    def closest(self, shapes_a: Sequence[int], shapes_b: Sequence[int], body_q: np.ndarray) -> tuple[float, int, int]:
        """The closest live pair across the two sets, at these body poses.

        Returns:
            ``(gap, shape_a, shape_b)`` -- the separation [m], negative where
            the two overlap, and which shapes made it. ``(inf, -1, -1)`` when no
            pair across the sets is live, which is the answer for two parts the
            model has filtered against each other.
        """
        a, b = self._colliding(shapes_a), self._colliding(shapes_b)
        if not len(a) or not len(b):
            return float("inf"), -1, -1
        first_a, last_a = self._world(a, body_q)
        first_b, last_b = self._world(b, body_q)
        gaps = _segment_segment(first_a, last_a, first_b, last_b)
        gaps -= self.radius[a][:, None] + self.radius[b][None, :]
        gaps[~self._testable_pairs(a, b)] = np.inf
        i, j = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
        gap = float(gaps[i, j])
        return (gap, int(a[i]), int(b[j])) if np.isfinite(gap) else (float("inf"), -1, -1)

    def between(self, shapes_a: Sequence[int], shapes_b: Sequence[int], body_q: np.ndarray) -> float:
        """:meth:`closest`'s gap alone [m]."""
        return self.closest(shapes_a, shapes_b, body_q)[0]

    def body_q(self, joint_q: Sequence[float]) -> np.ndarray:
        """Body poses at a joint configuration, without touching the scene.

        Forward kinematics into this probe's own state, so a planner can ask
        about a configuration the robot is not in and has no intention of
        passing through if the answer is bad.
        """
        if self._state is None:
            self._state = self.model.state()
            self._joint_q = wp.zeros(self.model.joint_coord_count, dtype=wp.float32, device=self.model.device)
        self._joint_q.assign(np.asarray(joint_q, dtype=np.float32))
        newton.eval_fk(self.model, self._joint_q, self.model.joint_qd, self._state)
        return self._state.body_q.numpy()

    def at(self, joint_q: Sequence[float], shapes_a: Sequence[int], shapes_b: Sequence[int]) -> float:
        """:meth:`between` at a joint configuration [m]."""
        return self.between(shapes_a, shapes_b, self.body_q(joint_q))
