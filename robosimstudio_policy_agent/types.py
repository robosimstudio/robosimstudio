# SPDX-License-Identifier: Apache-2.0
"""The wire format: ``gap_core.types`` dicts from numpy, and back.

A ``Vec3`` is ``{"x", "y", "z"}``, a ``Quaternion`` is ``{"w", "x", "y", "z"}``
-- wxyz, where Newton and every core solver speak xyzw -- and an ``Se3Pose``
is ``{"position": Vec3, "rotation": Quaternion}``. Every tool that answers a
pose and every backend that reads one goes through here, so the one place the
component order flips is the one place it is written down.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.math import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

__all__ = ["quat_wxyz", "se3", "se3_arrays", "se3_of_xform", "vec3"]


def vec3(v: Any) -> dict[str, float]:
    """A ``Vec3`` from anything indexable by 0, 1, 2."""
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def quat_wxyz(q_xyzw: Any) -> dict[str, float]:
    """A ``Quaternion`` (wxyz) from an xyzw quaternion."""
    w = quat_xyzw_to_wxyz(np.asarray(q_xyzw, dtype=np.float64))
    return {"w": float(w[0]), "x": float(w[1]), "y": float(w[2]), "z": float(w[3])}


def se3(position: Any, quat_xyzw: Any) -> dict[str, Any]:
    """An ``Se3Pose`` from a position and an xyzw quaternion."""
    return {"position": vec3(position), "rotation": quat_wxyz(quat_xyzw)}


def se3_of_xform(xform: Any) -> dict[str, Any]:
    """An ``Se3Pose`` from a ``body_q`` row ``(x, y, z, qx, qy, qz, qw)``."""
    return se3(xform[:3], xform[3:7])


def se3_arrays(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(position [3], quat xyzw [4])`` from an ``Se3Pose``."""
    p, q = pose["position"], pose["rotation"]
    return (
        np.array([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64),
        np.asarray(quat_wxyz_to_xyzw([q["w"], q["x"], q["y"], q["z"]]), dtype=np.float64),
    )
