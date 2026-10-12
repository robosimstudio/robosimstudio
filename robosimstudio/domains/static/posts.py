# SPDX-License-Identifier: Apache-2.0
"""A welded upright cylinder: the post, and the contact settings it is built with.

:class:`~robosimstudio.specs.StaticSpec`'s procedural shape is a box, and a
thing wound round a box is a different scene from a thing wound round a post.
:class:`EntityPost` is the entity :func:`~robosimstudio.domains.static.make.post`
names: it reads the spec's ``half_extents`` as ``(radius, _, half_height)`` and
says so loudly when they are missing.

It is scenery like any other static -- welded, placed by the episode, on a body
of its own per world in a replicated build -- and it lives here, beside
:class:`~robosimstudio.domains.static.entity.EntityStatic`, so that a scene
standing a post up needs nothing but this domain. A kind that bears against
posts (a rope would) reuses this class and :func:`fixture_shape_cfg` rather than
the other way round.

The module is ``posts`` and not ``post`` because the package already exports a
*function* called ``post``: importing a submodule binds its name on the package,
so a ``post.py`` here would replace the constructor with a module the moment it
was imported, and ``rss.post(...)`` would fail with "module is not callable".
"""

from __future__ import annotations

import newton

from .entity import EntityStatic, PlacedAtBuild

__all__ = ["EntityPost", "fixture_shape_cfg"]


def fixture_shape_cfg(builder: newton.ModelBuilder, *, collides: bool = True):
    """The contact settings a post is built with, for every fixture meant to answer as one does.

    ``ke = 1e4``, ``kd = 0.1``, ``mu = 2.0``, ``gap = 0.01`` -- and the one place
    they are written down. :class:`EntityPost` takes them from here, and so does
    any fixture a task draws beside a post for the same thing to bear against:
    two fixtures that answered contact differently would be a difference in the
    result that is not a difference in the task, and a second copy of four
    numbers is how that difference gets introduced by accident.

    Args:
        builder: The builder whose ``default_shape_cfg`` is copied.
        collides: False for a shape that is there to be looked at -- a bolt
            head, a screw boss -- and admitting one to the contact set only
            gives a fingertip somewhere new to catch.
    """
    cfg = builder.default_shape_cfg.copy()
    cfg.ke = 1.0e4
    cfg.kd = 0.1
    cfg.mu = 2.0
    cfg.gap = 0.01
    cfg.has_shape_collision = bool(collides)
    return cfg


class EntityPost(PlacedAtBuild, EntityStatic):
    """A vertical cylinder: ``half_extents = (radius, -, half_height)``."""

    claims_body = True
    """``build`` below claims its per-world body before drawing.

    Without it :meth:`~robosimstudio.domains.static.entity.EntityStatic.places_per_world`
    answers no for this class -- it overrides ``build`` -- and a wave keeps every
    world's post at the build episode's pose while the model key says the boards
    may be shared."""

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        # The episode's pose, and in a replicated build a body to carry it per
        # world: ``_claim_body`` does both, and a ``build`` override that skipped
        # it would leave every sampler that places a board moving the seats and
        # leaving the fixtures.
        self._claim_body(builder, episode)
        del episode
        spec = self.spec
        if spec.half_extents is None:
            raise ValueError(f"post {spec.name!r} needs half_extents=(radius, _, half_height)")
        radius, _, half_height = (float(v) for v in spec.half_extents)
        # ``_shape_frame``, not ``spec.xform``: the episode may have placed this
        # fixture, and reading the declared pose here moves the SEAT and leaves
        # the POST behind -- exactly the split ``episode_xform`` warns about.
        # Measured before the fix: three episodes with sampled board yaws of
        # +0.071, -0.124 and +0.230 rendered the posts at pixel-identical
        # centroids, while their seat boxes moved up to 14 mm.
        #
        # It answers the world frame on a world-hung build and the post's own body
        # at identity on a replicated one, which is the one line that has to
        # differ between the two builds.
        body, xform = self._shape_frame()
        cfg = fixture_shape_cfg(builder)
        builder.add_shape_cylinder(
            body=body,
            xform=xform,
            radius=radius,
            half_height=half_height,
            cfg=cfg,
            label=spec.name,
        )
        # Display is applied by Entity.build_into, once the range has closed.
