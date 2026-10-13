# SPDX-License-Identifier: Apache-2.0
"""An oriented-bounding-box grasp heuristic over Newton geometry.

A port of VAB-industrial's ``compute_obb`` / ``grasp_from_obb`` /
``top_down_grasp_quat``, which are that benchmark's own scripted grasp policy.
Keeping the heuristic identical means a teleop assist, a scripted routine and a
graph node all propose the grasp the benchmark's baseline would, so a pose that
works here is a pose that works there.

The box is the body-frame AABB over a body's shapes -- the reference's
``compute_obb`` takes ``R`` from the *body* orientation and boxes the geoms in
that frame. Rigid bodies keep that box for the life of the scene, so
:class:`ObbGrasp` is computed once at build time and only transformed per query.

Two caveats the reference documents apply unchanged: ``handle_frac`` offsets
*away* from the centre of mass, which suits a hammer and inverts on a
screwdriver; and an object wider than the gripper span cannot be grasped at the
centroid at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import newton
import numpy as np

from robosimstudio.core.math import quat_multiply, quat_to_mat

__all__ = ["ObbGrasp", "compute_obb", "grasp_from_obb", "top_down_grasp_quat"]

# Unit-cube corner signs, for expanding a centre + half-extents into 8 points.
_CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)


@dataclass(frozen=True)
class ObbGrasp:
    """A body-frame bounding box, and what it implies about grasping the body."""

    body: int
    """Index into ``State.body_q``."""
    center_local: np.ndarray
    """Box centre in the body's own frame [m], shape ``[3]``."""
    half: np.ndarray
    """Box half-extents [m], shape ``[3]``."""
    long_axis: int
    """Index of the longest axis -- the one a top-down grasp aligns to."""
    handle_sign: float
    """Which way along :attr:`long_axis` the light (graspable) end lies."""
    gripper_span: float
    """Finger travel the fit test is made against [m]."""

    @property
    def length(self) -> float:
        """Extent along the long axis [m] -- the direction the fingers straddle."""
        return float(2.0 * self.half[self.long_axis])

    @property
    def grasp_width(self) -> float:
        """Extent the fingers must close across at the centroid [m].

        A top-down grasp lines the hand up with the long axis, so the fingers
        close along the widest remaining axis.
        """
        return float(2.0 * np.delete(self.half, self.long_axis).max())

    @property
    def fits_gripper(self) -> bool:
        """Whether :attr:`grasp_width` fits :attr:`gripper_span`.

        Conservative by construction: it boxes the whole object, so a thin
        handle inside a wide box reads as too wide. A claw hammer is exactly
        that case -- 13 cm across the claw, with a 3.6 cm handle the reference
        baseline grasps every time by sliding along it with ``handle_frac``.
        Read a ``False`` as "not graspable at the centroid", not "not
        graspable".
        """
        return self.grasp_width < self.gripper_span


def _shape_corners_local(builder: newton.ModelBuilder, shape: int) -> np.ndarray:
    """The 8 corners of one shape's bounding box, in its body's frame.

    Meshes are special-cased for the reason the reference documents: falling
    back to an isotropic size collapses a mesh to a cube and erases the long
    axis the grasp keys off.
    """
    geo_type = newton.GeoType(builder.shape_type[shape])
    scale = np.asarray(builder.shape_scale[shape], dtype=np.float64)

    if geo_type == newton.GeoType.BOX:
        center, half = np.zeros(3), scale[:3]
    elif geo_type == newton.GeoType.SPHERE:
        center, half = np.zeros(3), np.full(3, scale[0])
    elif geo_type in (newton.GeoType.CAPSULE, newton.GeoType.CYLINDER, newton.GeoType.CONE):
        radius, half_height = float(scale[0]), float(scale[1])
        # Capsule ends are hemispherical, so they add a radius to the length.
        axial = half_height + (radius if geo_type == newton.GeoType.CAPSULE else 0.0)
        center, half = np.zeros(3), np.array([radius, radius, axial])
    elif geo_type == newton.GeoType.ELLIPSOID:
        center, half = np.zeros(3), scale[:3]
    elif geo_type == newton.GeoType.MESH or geo_type == getattr(newton.GeoType, "CONVEX_MESH", newton.GeoType.MESH):
        # A convex hull is a mesh whose vertices are its own bound; the drawn-hull
        # collider every tool-sorting variant stands on is one.
        mesh = builder.shape_source[shape]
        if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
            return np.zeros((0, 3))
        verts = np.asarray(mesh.vertices, dtype=np.float64) * scale[:3]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    else:
        return np.zeros((0, 3))

    tf = builder.shape_transform[shape]
    pos = np.array([tf[0], tf[1], tf[2]], dtype=np.float64)
    rot = quat_to_mat([tf[3], tf[4], tf[5], tf[6]])
    return pos[None, :] + (center[None, :] + _CORNER_SIGNS * half) @ rot.T


def compute_obb(
    builder: newton.ModelBuilder,
    body: int,
    com_local: Sequence[float],
    *,
    gripper_span: float = 0.08,
) -> ObbGrasp:
    """Box the shapes of a body the gripper can hit, in the body's frame, and pick the graspable end.

    The colliding shapes, when the body has any -- what the fingers close on
    -- and every non-site shape only on a body that collides with nothing. A
    ``drawn_hull`` collider keeps the export's fitted box on the body as a
    non-colliding shape beside the hull it collides with, and that box carries
    a general rotation the upright body does not: boxing it too turned a
    36 x 72 x 78 mm tape measure into 107 x 109 x 46 (the box's corners, swept
    into the body's axes), so the oracle reported a width no gripper spans,
    slid the grasp along the wrong axis, and closed on nothing -- in every
    level-1 sorting scene the tape appears in. All 197 objects of that tier
    are boxed that way; the hand-written variants' boxes are axis-aligned and
    unchanged by this.

    Args:
        builder: The builder the shapes still live on, i.e. pre-``finalize``.
        body: Body index to box.
        com_local: The body's centre of mass in its own frame [m], which decides
            which end is the heavy head and which is the handle.
        gripper_span: Finger travel the fit test is made against [m].
    """
    site, collide = int(newton.ShapeFlags.SITE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    shapes = [
        s
        for s in range(builder.shape_count)
        if builder.shape_body[s] == body and not (int(builder.shape_flags[s]) & site)
    ]
    colliding = [s for s in shapes if int(builder.shape_flags[s]) & collide]
    corners = [_shape_corners_local(builder, s) for s in (colliding or shapes)]
    corners = [c for c in corners if len(c)]
    if not corners:
        raise ValueError(f"body {body} has no boxable shapes")
    pts = np.concatenate(corners, axis=0)

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center_local = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    long_axis = int(np.argmax(half))

    # The COM sits towards the heavy head, so the handle is the other way.
    com = np.asarray(com_local, dtype=np.float64)
    head_sign = np.sign(com[long_axis] - center_local[long_axis])
    handle_sign = -(head_sign if head_sign != 0 else 1.0)

    return ObbGrasp(
        body=int(body),
        center_local=center_local,
        half=half,
        long_axis=long_axis,
        handle_sign=float(handle_sign),
        gripper_span=float(gripper_span),
    )


def grasp_from_obb(
    obb: ObbGrasp, body_transform: Sequence[float], handle_frac: float = 0.0
) -> tuple[np.ndarray, float]:
    """World grasp point and handle yaw for an object at a given pose.

    Args:
        obb: The body-frame box, from :func:`compute_obb`.
        body_transform: The body's current pose as ``(px, py, pz, qx, qy, qz, qw)``.
        handle_frac: How far along the long axis to slide the grasp point from
            the centroid towards the light end. The reference defaults its runs
            to 0.0 -- its scripts *expose* 0.5 as an argparse default, but the
            baseline numbers in its README are recorded at
            ``--handle_frac 0.0``. The offset suits a hammer and inverts on a
            screwdriver.

    Returns:
        The world grasp position [m], and the yaw [rad] of the long axis in the
        table plane.
    """
    pos = np.asarray(body_transform[:3], dtype=np.float64)
    rot = quat_to_mat(body_transform[3:7])

    grasp_local = obb.center_local.copy()
    grasp_local[obb.long_axis] += obb.handle_sign * handle_frac * obb.half[obb.long_axis]
    grasp_pos = pos + rot @ grasp_local

    long_axis_world = rot[:, obb.long_axis]
    return grasp_pos, float(np.arctan2(long_axis_world[1], long_axis_world[0]))


def top_down_grasp_quat(home_quat_xyzw: Sequence[float], handle_yaw: float) -> np.ndarray:
    """Yaw the home (downward-pointing) gripper so the fingers cross the handle.

    The yaw is wrapped into [-90, 90) degrees -- the reference's own
    ``(yaw + pi/2) % pi - pi/2``, kept identical: the hand is symmetric under a
    180-degree flip, so the wrapped yaw grasps identically while keeping the
    last wrist joint away from its limit.
    """
    yaw = (float(handle_yaw) + np.pi / 2.0) % np.pi - np.pi / 2.0
    half = yaw / 2.0
    return quat_multiply([0.0, 0.0, np.sin(half), np.cos(half)], home_quat_xyzw)
