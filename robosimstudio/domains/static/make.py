# SPDX-License-Identifier: Apache-2.0
"""Constructors for scenery: free functions that return a spec.

Scenery is welded to the world and integrates nothing, so it is a different spec
from a free body and therefore different constructors. That is deliberate rather
than an oversight: one ``box()`` returning either an
:class:`~robosimstudio.specs.ObjectSpec` or a
:class:`~robosimstudio.specs.StaticSpec` depending on a flag would be a function
whose *return type* changes with an argument, which is exactly the hidden
knowledge this tier exists to remove. A thing that falls and a thing that does
not are not the same thing with a switch.

Poses here are **world coordinates**, not the work-surface frame: a cell's own
furniture is what defines that frame, and a place is not positioned relative to
itself.
"""

from __future__ import annotations

from robosimstudio.specs import StaticSpec, Vec3, Xform
from robosimstudio.specs.asset import ShapeSpec

__all__ = ["post", "slab"]

_IDENTITY: Xform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _xform(at: Vec3 | Xform) -> Xform:
    values = tuple(float(v) for v in at)
    if len(values) == 7:
        return values
    if len(values) == 3:
        return (*values, 0.0, 0.0, 0.0, 1.0)
    raise ValueError(f"a pose is 3 numbers or 7 (x y z qx qy qz qw), got {len(values)}")


def slab(*, name: str = "table", half: Vec3 = (0.4, 0.6, 0.375), at: Vec3 | Xform = _IDENTITY, **kwargs) -> StaticSpec:
    """A welded box: a table, a wall, a bin's side.

    ``at`` is the box's **centre**, so a 750 mm work surface built from a slab
    750 mm thick sits at ``z = 0.375``. Stating the centre rather than the top is
    what the builder wants and what every shipped bench already does; the
    alternative is a constructor that is right until somebody changes the
    thickness.
    """
    return StaticSpec(name=name, shape=ShapeSpec(kind="box", half_extents=half), xform=_xform(at), **kwargs)


def post(
    *, name: str = "post", radius: float = 0.02, half_height: float = 0.06, at: Vec3 | Xform = _IDENTITY, **kwargs
) -> StaticSpec:
    """A welded upright cylinder: a peg, a bollard, a spindle to wind something round.

    Built by the ``"post"`` entity,
    :class:`~robosimstudio.domains.static.posts.EntityPost`, which this domain
    registers itself -- so a post needs no other kind of thing to be installed.
    """
    return StaticSpec(
        name=name,
        entity="post",
        shape=ShapeSpec(kind="cylinder", radius=radius, half_height=half_height),
        half_extents=(radius, radius, half_height),
        xform=_xform(at),
        **kwargs,
    )
