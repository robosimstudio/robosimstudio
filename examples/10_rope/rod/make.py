# SPDX-License-Identifier: Apache-2.0
"""Constructors for a 1-D deformable: free functions that return a spec.

Beside :mod:`~robosimstudio.domains.rod.declare`, which builds the
:class:`~robosimstudio.assetlib.record.AssetRecord` that says what a rod *is*.
This builds the :class:`~robosimstudio.domains.rod.RodSpec` that puts one in a
scene.
"""

from __future__ import annotations

from robosimstudio.specs import Vec3

from .entity import RodSpec

__all__ = ["rod"]


def rod(
    *,
    name: str = "rod",
    length: float = 0.9,
    radius: float = 0.011,
    at: Vec3 = (0.0, 0.0, 0.0),
    along: Vec3 = (0.0, 1.0, 0.0),
    segments: int = 28,
    **kwargs,
) -> RodSpec:
    """A capsule chain on cable joints: a cable, a rope, a suture, a hose.

    Args:
        length: End to end [m].
        radius: Capsule radius [m]. Segment spacing must stay above ``2 *
            radius``, so a short dense rod is the combination to watch.
        at: Where the **first** capsule goes, in the cell's **work-surface
            frame**, like every rigid constructor here: ``EntityRod.build`` runs
            ``start`` through ``Entity.placed`` before handing it to ``add_rod``,
            so a rod laid at ``z = radius`` rests on the bench rather than on the
            floor 750 mm below it.

            **It is an end, not the middle.** The chain runs ``length`` along
            ``along`` from this point, so centring a rod on the bench means
            offsetting by half its length -- ``examples/10_rope/00_cable.py`` writes
            ``at=(0.0, -length / 2, radius + 0.06)`` for exactly that reason.
        along: Direction the chain runs in. Normalised for you.
        segments: Capsules. 28 is the shipped cable's; more is smoother and
            slower, and past roughly 40 the per-body contact buffer becomes the
            thing to watch -- see
            :data:`~robosimstudio.specs.physics.GENERIC`.

    A rod needs a soft solver. Served by MuJoCo it does not mis-solve quietly --
    it dies inside MuJoCo's own importer on a body mapping -- and ``GENERIC``
    names ``vbd`` for it, which is why the default arrangement is a table rather
    than one entry.
    """
    return RodSpec(name=name, length=length, radius=radius, start=at, direction=along, segments=segments, **kwargs)
