# SPDX-License-Identifier: Apache-2.0
"""The ``static`` domain: what it is, what builds it, what reads it.

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

from .entity import EntityArena, EntityStatic
from .make import post, slab
from .posts import EntityPost, fixture_shape_cfg

__all__ = [
    "EntityArena",
    "EntityPost",
    "EntityStatic",
    "fixture_shape_cfg",
    "post",
    "slab",
]

DOMAIN = register_domain(
    "static",
    state=("shapes",),
    reads="rigid",
    role="fixtures",
    contacts=(),
    world_fixed=True,
    summary="World-fixed scenery: a table, a post, a bin.",
)

# The generic name for this kind of thing, so a spec can ask for it without a
# task directory having bound one. A task's own spelling stays the task's; the
# registry accepts an identical second registration under another name.
register_entity("static")(EntityStatic)
register_entity("arena")(EntityArena)
# What ``rss.post`` names. Registered here, beside the class, so a post builds
# in a scene that holds nothing but scenery and whatever stands on it.
register_entity("post")(EntityPost)
