# SPDX-License-Identifier: Apache-2.0
"""Pose arithmetic for a kinematic mirror, in the form the inner loop wants.

These are the hot-path twins of the general helpers, and the duplication is
deliberate where it exists. A mirror drive composes one transform and
interpolates another *per mirror, per engine substep*, so this runs a few
hundred times a control step: :func:`compose` is the direct quaternion form of
:func:`robosimstudio.scene.frames.compose`, which goes through two 4x4 matrices
to say the same thing, and :func:`lerp_pose` is a normalised lerp rather than
:func:`~robosimstudio.core.math.quat_slerp` because the endpoints are one
substep apart and the difference between the two is far below what a contact
solver can tell.

Every quaternion here is ``xyzw``, which is Newton's convention and therefore
this package's. An engine that speaks ``wxyz`` converts at its own boundary and
nowhere else; a swap that leaks past that boundary produces a plausible-looking
wrong rotation rather than an error, which is the reason this paragraph exists.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.core.math import quat_multiply

__all__ = ["compose", "compose_many", "lerp_pose", "lerp_poses", "quat_rotate"]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the ``xyzw`` quaternion ``q``."""
    x, y, z, w = q
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def compose(body: np.ndarray, local: np.ndarray) -> np.ndarray:
    """``body ∘ local`` -- a shape's world pose from its body's.

    Both are ``[x, y, z, qx, qy, qz, qw]``; so is the result.
    """
    out = np.empty(7)
    out[:3] = body[:3] + quat_rotate(body[3:7], local[:3])
    out[3:7] = quat_multiply(body[3:7], local[3:7])
    return out


def lerp_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear position, normalised-lerp orientation, sign-matched.

    The sign match is load-bearing rather than tidy: ``q`` and ``-q`` are the
    same rotation, and lerping between the two representations sweeps the long
    way round -- a mirror that took the long way round between two substeps
    would sweep a contact surface through the deformable it is supposed to be
    resting against.
    """
    out = np.empty(7)
    out[:3] = (1.0 - t) * a[:3] + t * b[:3]
    qb = b[3:7] if np.dot(a[3:7], b[3:7]) >= 0.0 else -b[3:7]
    q = (1.0 - t) * a[3:7] + t * qb
    norm = float(np.linalg.norm(q))
    out[3:7] = q / norm if norm > 0.0 else b[3:7]
    return out


# ---------------------------------------------------------------- the many
# The same two operations over a whole mirror set at once. A drive composes and
# interpolates once per mirror per substep, so the scalar forms above run
# ``mirrors x substeps`` times a control step -- measured at 2808 x 2 on a
# three-world surface-preparation scene, where the pose arithmetic and the engine calls
# behind it were together 67 per cent of the host tick against 2 per cent for
# the solve itself. Each call is microseconds; it is the *count* that costs.
#
# Written beside the scalar forms rather than replacing them because both have
# callers and the pair is the point: a reader checking the batched arithmetic
# has the one-pose version directly above it to check it against.


def compose_many(bodies: np.ndarray, locals_: np.ndarray) -> np.ndarray:
    """``body ∘ local`` for a whole set. ``[n, 7]`` from ``[n, 7]`` and ``[n, 7]``."""
    bodies = np.asarray(bodies, dtype=np.float64).reshape(-1, 7)
    locals_ = np.asarray(locals_, dtype=np.float64).reshape(-1, 7)
    q, v = bodies[:, 3:7], locals_[:, :3]
    u, w = q[:, :3], q[:, 3:4]
    uv = np.einsum("ij,ij->i", u, v)[:, None]
    uu = np.einsum("ij,ij->i", u, u)[:, None]
    out = np.empty((len(bodies), 7))
    out[:, :3] = bodies[:, :3] + 2.0 * uv * u + (w * w - uu) * v + 2.0 * w * np.cross(u, v)

    # The Hamilton product, written out: one temporary per term rather than one
    # per quaternion, which is what keeps this a handful of array ops.
    ax, ay, az, aw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    b = locals_[:, 3:7]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out[:, 3] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 4] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 5] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 6] = aw * bw - ax * bx - ay * by - az * bz
    return out


def lerp_poses(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """:func:`lerp_pose` for a whole set, including its sign match.

    The sign match is per row and is load-bearing for the same reason it is in
    the scalar form: a mirror that took the long way round between two substeps
    would sweep a contact surface through the deformable it is resting against.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 7)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 7)
    out = np.empty_like(a)
    out[:, :3] = (1.0 - t) * a[:, :3] + t * b[:, :3]

    qa, qb = a[:, 3:7], b[:, 3:7]
    flip = np.where(np.einsum("ij,ij->i", qa, qb) >= 0.0, 1.0, -1.0)[:, None]
    q = (1.0 - t) * qa + t * (flip * qb)
    norm = np.linalg.norm(q, axis=1)[:, None]
    # A zero norm is two opposite quaternions at exactly t = 0.5; the scalar
    # form falls back to the endpoint and so does this.
    out[:, 3:7] = np.where(norm > 0.0, q / np.where(norm > 0.0, norm, 1.0), qb)
    return out
