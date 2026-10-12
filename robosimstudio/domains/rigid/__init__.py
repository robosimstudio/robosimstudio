# SPDX-License-Identifier: Apache-2.0
"""The ``rigid`` domain: what it is, what builds it, what reads it.

One bundle per physical kind. Importing this registers the domain and everything
that comes with it -- the entity that builds one, and any measure that only makes
sense over it -- so a scene and a goal cannot disagree about whether this kind of
thing exists.

Reached through the ``DOMAINS`` registry's package scan rather than by anyone
importing it: a domain has to be registered before a goal binds, and the layer
below cannot name the layer that holds it.
"""

from __future__ import annotations

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import register_domain

from . import measures as _measures  # noqa: F401 -- registers the rigid measures
from .declare import Body
from .entity import EntityRigid
from .make import body, box, capsule, cylinder, sphere

__all__ = [
    "Body",
    "EntityRigid",
    "body",
    "box",
    "capsule",
    "cylinder",
    "sphere",
]

DOMAIN = register_domain(
    "rigid",
    state=("bodies", "joints", "shapes"),
    reads="rigid",
    role="objects",
    contacts=("surface", "fixtures", "objects", "robot", "deformables"),
    summary="A free rigid body on a free joint.",
)

# The generic name for this kind of thing, so a spec can ask for it without a
# task directory having bound one. A task's own spelling stays the task's; the
# registry accepts an identical second registration under another name.
register_entity("rigid")(EntityRigid)
