# SPDX-License-Identifier: Apache-2.0
"""Host-side quaternion and transform helpers, in one place.

Every module that talks to a front end, a renderer or an asset format needs the
same handful of conversions. They live here instead.

Conventions, which differ by neighbour and are the whole reason this module
exists:

* **Warp and Newton** store quaternions **xyzw**, and a transform as the seven
  floats ``(px, py, pz, qx, qy, qz, qw)`` -- that is what ``State.body_q`` rows
  are. This module's unsuffixed functions take that layout.
* **MuJoCo, USD and the graph-policy contract** are **wxyz**. :func:`quat_wxyz_to_xyzw` and its
  inverse are the only places that conversion should happen.
* **viser** is wxyz as well, so the teleop panels go through the same pair.

Everything is float64 numpy on the host. Device-side rotation belongs in a Warp
kernel using ``wp.quat_rotate``, not here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "IDENTITY_QUAT",
    "compose_xforms",
    "compute_angle_between",
    "lame_from_young_poisson",
    "look_at_matrix",
    "mat_to_quat",
    "quat_from_axis_angle",
    "quat_inverse",
    "quat_multiply",
    "quat_slerp",
    "quat_to_axis_angle",
    "quat_to_mat",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "quat_yaw",
    "transform_matrix",
    "transform_point",
    "xform_from_matrix",
    "young_poisson_from_lame",
]

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
"""The identity rotation, xyzw."""


def quat_xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    """Reorder a Warp/Newton quaternion into the wxyz order MuJoCo and viser use."""
    x, y, z, w = (float(v) for v in q)
    return np.array([w, x, y, z])


def quat_wxyz_to_xyzw(q: Sequence[float]) -> np.ndarray:
    """Reorder a MuJoCo/USD/viser quaternion into Warp's xyzw order."""
    w, x, y, z = (float(v) for v in q)
    return np.array([x, y, z, w])


def quat_to_mat(q_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion, shape ``[3, 3]``.

    Normalizes as it goes, so a quaternion that has drifted off the unit sphere
    (which read-back float32 poses routinely have) still yields an orthogonal
    matrix rather than one that quietly scales.
    """
    x, y, z, w = (float(v) for v in q_xyzw)
    n = x * x + y * y + z * z + w * w
    if n < 1.0e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ]
    )


def mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """xyzw quaternion of a 3x3 rotation matrix.

    The rotation-only face of :func:`xform_from_matrix`, sharing its
    trace-stable branch selection -- a 180-degree yaw has ``w == 0``, so
    assuming ``w`` is the largest component fails exactly at the grasp
    orientations half the tasks use.
    """
    m = np.eye(4)
    m[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return xform_from_matrix(m)[3:7]


def quat_multiply(a_xyzw: Sequence[float], b_xyzw: Sequence[float]) -> np.ndarray:
    """Hamilton product ``a * b`` of two xyzw quaternions."""
    ax, ay, az, aw = (float(v) for v in a_xyzw)
    bx, by, bz, bw = (float(v) for v in b_xyzw)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_inverse(q_xyzw: Sequence[float]) -> np.ndarray:
    """Inverse of an xyzw quaternion (conjugate over squared norm)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    n = x * x + y * y + z * z + w * w
    if n < 1.0e-12:
        return IDENTITY_QUAT.copy()
    return np.array([-x, -y, -z, w]) / n


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    """xyzw quaternion for a rotation of ``angle`` [rad] about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1.0e-12:
        return IDENTITY_QUAT.copy()
    half = 0.5 * float(angle)
    return np.append(a / n * np.sin(half), np.cos(half))


def quat_slerp(q0_xyzw: Sequence[float], q1_xyzw: Sequence[float], t: float) -> np.ndarray:
    """Shortest-arc interpolation between two xyzw quaternions.

    ``q1`` is negated first when the two point into opposite hemispheres, so the
    result turns the short way round -- without that, half of all inputs
    interpolate the 300-degree way to a 60-degree difference. Near-parallel
    inputs fall back to a normalized lerp, where the ``sin(theta)`` division
    loses precision.
    """
    a = np.asarray(q0_xyzw, dtype=np.float64)
    b = np.asarray(q1_xyzw, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + float(t) * (b - a)
    else:
        theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        q = (np.sin((1.0 - float(t)) * theta) * a + np.sin(float(t) * theta) * b) / np.sin(theta)
    return q / np.linalg.norm(q)


def quat_to_axis_angle(q_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation vector of an xyzw quaternion: the axis scaled by the angle [rad].

    The inverse of :func:`quat_from_axis_angle`, and the form a Cartesian
    orientation *error* is naturally expressed in -- three numbers that go to
    zero together, which is what a controller can put a gain on.

    The double cover is resolved towards the short way round: a quaternion and
    its negation are the same rotation, so ``w`` is flipped positive first and
    the angle comes back in ``[0, pi]``. The reference's own ``quat2axisangle``
    omits that flip and returns angles up to ``2*pi`` for a negative ``w``,
    which is invisible where it is used (the error quaternion between two nearby
    frames has ``w`` near +1) and a sign inversion where it is not -- a servo
    handed one would take the long way round through its joint limits.
    """
    q = np.asarray(q_xyzw, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1.0e-12:
        return np.zeros(3)
    x, y, z, w = q / n
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1.0e-9:
        # Below this the small-angle limit is more accurate than the division:
        # the rotation vector is 2 * the vector part, to float precision.
        return 2.0 * np.array([x, y, z])
    return np.array([x, y, z]) * (2.0 * float(np.arccos(np.clip(w, -1.0, 1.0))) / sin_half)


def quat_yaw(q_xyzw: Sequence[float]) -> float:
    """Heading of an xyzw quaternion's +x axis in the world xy plane [rad]."""
    x_axis = quat_to_mat(q_xyzw)[:, 0]
    return float(np.arctan2(x_axis[1], x_axis[0]))


def transform_matrix(xform: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous matrix of one ``body_q``-style row ``(p, q_xyzw)``."""
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(xform[3:7])
    m[:3, 3] = [float(v) for v in xform[:3]]
    return m


def xform_from_matrix(m: np.ndarray) -> np.ndarray:
    """The ``(px, py, pz, qx, qy, qz, qw)`` row a 4x4 matrix represents.

    The rotation is read back through the trace-stable branch selection, not by
    assuming ``w`` is the largest component -- a 180-degree yaw (which is where
    half the camera rigs sit) has ``w == 0``.
    """
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s, 0.25 / s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1.0e-12))
        q = np.zeros(4)
        q[3] = (r[k, j] - r[j, k]) / s
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    return np.append(np.asarray(m, dtype=np.float64)[:3, 3], q / np.linalg.norm(q))


def compose_xforms(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """``a * b`` for two ``(p, q_xyzw)`` rows: ``b`` given in ``a``'s frame, in the world's."""
    return np.asarray(xform_from_matrix(transform_matrix(a) @ transform_matrix(b)), dtype=np.float64)


def transform_point(xform: Sequence[float], point: Sequence[float]) -> np.ndarray:
    """Map a point through one ``body_q``-style row."""
    return np.asarray(xform[:3], dtype=np.float64) + quat_to_mat(xform[3:7]) @ np.asarray(point, dtype=np.float64)


def compute_angle_between(x1: Sequence[float], x2: Sequence[float]) -> float:
    """Angle between two vectors [rad], with degenerate inputs reported as pi.

    Transcribed from SimplerEnv's ``common.compute_angle_between``, which the
    grasp tests call with contact impulses that are frequently zero; treating a
    zero vector as maximally misaligned is what makes those tests fail closed.
    """
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1.0e-12 or nb < 1.0e-12:
        return float(np.pi)
    return float(np.arccos(np.clip(np.dot(a / na, b / nb), -1.0, 1.0)))


def look_at_matrix(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world 4x4 looking from ``eye`` at ``target``.

    The returned basis is the vision convention -- ``+x`` right, ``+y`` down,
    ``+z`` forward along the view direction -- which is what
    :class:`~robosimstudio.core.cameras.CameraSpec` stores and what a pinhole
    ``K`` projects through. Renderers that want OpenGL's ``-z``-forward basis
    flip the last two columns themselves.
    """
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    n = np.linalg.norm(forward)
    if n < 1.0e-12:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / n
    up_v = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(forward, up_v / np.linalg.norm(up_v)))) > 0.999:
        up_v = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_v)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, forward, eye
    return m


# ---------------------------------------------------------------------------
# Elasticity: the one pair of constants two engines state differently
# ---------------------------------------------------------------------------
# Here rather than beside the soft domain because it is not a fact about
# tetrahedral bodies. Newton stores a Lame pair in ``model.tet_materials``;
# every foreign engine this stack co-simulates with asks for Young and Poisson
# instead, so the round trip is what lets a bridge read a body's material off
# the finalized model rather than off the spec that built it -- which is what
# makes a repro bundle rebuildable from the model and the manifest alone.


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    """``(mu, lambda)`` [Pa] from Young's modulus [Pa] and Poisson ratio.

    The standard isotropic relations. ``poisson`` must be below 0.5 -- at the
    incompressible limit lambda diverges, and a solver asked for it returns
    numbers that mean nothing.
    """
    e, nu = float(young), float(poisson)
    if not e > 0.0:
        raise ValueError(f"young must be > 0 Pa, got {e}")
    if not -1.0 < nu < 0.5:
        raise ValueError(f"poisson must be in (-1, 0.5), got {nu}")
    mu = e / (2.0 * (1.0 + nu))
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def young_poisson_from_lame(mu: float, lam: float) -> tuple[float, float]:
    """``(young, poisson)`` back from the Lame pair. Inverse of the above."""
    mu, lam = float(mu), float(lam)
    if not mu > 0.0:
        raise ValueError(f"mu must be > 0 Pa, got {mu}")
    e = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return e, nu
