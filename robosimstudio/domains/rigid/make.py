# SPDX-License-Identifier: Apache-2.0
"""Constructors for a rigid body: free functions that return a spec.

Beside :mod:`~robosimstudio.domains.rigid.declare`, and doing a different job.
``declare`` builds an :class:`~robosimstudio.assetlib.record.AssetRecord` -- what
a thing *is*, and what a goal may ask about it. This builds an
:class:`~robosimstudio.specs.ObjectSpec` -- put one of these *here*, for the
compiler. A corpus asset needs both; a cube you are dropping on a bench to see
whether the bench works needs only the second, and having to write the first is
what made the hello-world five files long.

**Nothing here is privileged.** Every one of these is a free function returning a
value you hold, and one you write in your own module reaches ``Scene.add`` on
identical terms::

    def die(*, at, pip=0.004):
        return rss.box(name="die", half=(0.008,) * 3, at=at, color=(0.9, 0.9, 0.85))

That is the idiom this package already wrote down: *"A constructor returns a
value you hold, the way pyroki's do; what you do with it is the task's
business."* A method on ``Scene`` would have been a closed set.
"""

from __future__ import annotations

from robosimstudio.specs import AssetSpec, ObjectSpec, Vec3
from robosimstudio.specs.asset import ShapeSpec

__all__ = ["body", "box", "capsule", "cylinder", "sphere"]

_ORIGIN: Vec3 = (0.0, 0.0, 0.0)


def box(*, name: str = "box", half: Vec3 = (0.025, 0.025, 0.025), at: Vec3 = _ORIGIN, **kwargs) -> ObjectSpec:
    """A rectangular free body. ``half`` is half-extents, so a 50 mm cube is ``(0.025,) * 3``.

    ``at`` is stated in the **work-surface frame** of whatever cell the scene
    stands in, which is what lets the same spec move to a taller bench. Extra
    keyword arguments go straight to :class:`~robosimstudio.specs.ObjectSpec` --
    ``mass``, ``color``, ``friction``, ``static``.
    """
    return ObjectSpec(name=name, shape=ShapeSpec(kind="box", half_extents=half), pose=at, **kwargs)


def sphere(*, name: str = "sphere", radius: float = 0.025, at: Vec3 = _ORIGIN, **kwargs) -> ObjectSpec:
    """A spherical free body."""
    return ObjectSpec(name=name, shape=ShapeSpec(kind="sphere", radius=radius), pose=at, **kwargs)


def capsule(
    *, name: str = "capsule", radius: float = 0.015, half_height: float = 0.04, at: Vec3 = _ORIGIN, **kwargs
) -> ObjectSpec:
    """A capsule free body: a cylinder with hemispherical caps, along its own z."""
    return ObjectSpec(
        name=name, shape=ShapeSpec(kind="capsule", radius=radius, half_height=half_height), pose=at, **kwargs
    )


def cylinder(
    *, name: str = "cylinder", radius: float = 0.02, half_height: float = 0.05, at: Vec3 = _ORIGIN, **kwargs
) -> ObjectSpec:
    """A cylindrical free body, along its own z."""
    return ObjectSpec(
        name=name, shape=ShapeSpec(kind="cylinder", radius=radius, half_height=half_height), pose=at, **kwargs
    )


def body(*, name: str, asset: str | AssetSpec, at: Vec3 = _ORIGIN, scale: float = 1.0, **kwargs) -> ObjectSpec:
    """A free body whose geometry comes from a file.

    ``asset`` may be a path -- resolved against the asset roots the same way
    every other path is -- or an :class:`~robosimstudio.specs.AssetSpec` when you
    need its other fields (a collision proxy, a texture, a collider
    approximation). ``scale`` is ignored when you pass a spec, which carries its
    own.
    """
    spec = asset if isinstance(asset, AssetSpec) else AssetSpec(path=str(asset), scale=scale)
    return ObjectSpec(name=name, asset=spec, pose=at, **kwargs)
