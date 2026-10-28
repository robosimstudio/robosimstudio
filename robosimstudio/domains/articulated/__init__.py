# SPDX-License-Identifier: Apache-2.0
"""The ``articulated`` domain: links joined by joints.

Two kinds of thing are built in it. The robot, whose entity lives in ``robots``
-- the one thing in the scene whose declaration and builder are worth keeping
together, because adding a robot should be a matter of saying which joints are an
arm rather than of writing a builder. And a jointed **object** -- a cabinet, a
door, a box with a lid -- which is :class:`~.entity.EntityArticulated`, imported
from a USD stage and registered as ``articulated_object``. The domain's role is
the robot's; the object entity says ``objects`` for itself.
"""

from __future__ import annotations

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import register_domain

from .entity import EntityArticulated
from .make import articulated

__all__ = ["EntityArticulated", "articulated"]

DOMAIN = register_domain(
    "articulated",
    state=("bodies", "joints", "shapes"),
    reads="articulated",
    role="robot",
    contacts=("surface", "fixtures", "objects"),
    summary="Links joined by joints: a robot, or a jointed object such as a cabinet.",
)

register_entity("articulated_object")(EntityArticulated)

from . import measures as _measures  # noqa: E402, F401 -- registers containment, height, speed and opening
