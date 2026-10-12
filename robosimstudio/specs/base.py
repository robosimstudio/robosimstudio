# SPDX-License-Identifier: Apache-2.0
"""The primitive types every other spec is written in.

Tuples rather than dataclasses on purpose: a pose is three floats and four, it
is hashable, it costs nothing to copy, and it is what the solver's arrays hold
anyway. Anything richer would be a conversion at every boundary.
"""

from __future__ import annotations

__all__ = ["IDENTITY", "Quat", "Vec3", "Xform"]


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
"""xyzw, as Warp and Newton store it."""
Xform = tuple[float, float, float, float, float, float, float]
"""``(px, py, pz, qx, qy, qz, qw)``."""

IDENTITY: Xform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
