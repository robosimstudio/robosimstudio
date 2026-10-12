# SPDX-License-Identifier: Apache-2.0
"""The ``articulated`` domain: links joined by driven joints.

The registration only. What *builds* one is the robot's entity, and that lives in
``robots`` -- a robot is the one thing in the scene whose declaration and whose
builder are worth keeping together, because adding a robot should be a matter of
saying which joints are an arm rather than of writing a builder.
"""

from __future__ import annotations

from robosimstudio.specs.physics import register_domain

__all__: list[str] = []

DOMAIN = register_domain(
    "articulated",
    state=("bodies", "joints", "shapes"),
    reads="articulated",
    role="robot",
    contacts=("surface", "fixtures", "objects"),
    summary="A robot: links joined by driven joints.",
)

from . import measures as _measures  # noqa: E402, F401 -- registers containment, height and speed
