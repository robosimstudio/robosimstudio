# SPDX-License-Identifier: Apache-2.0
"""Containment: is it in there, and how much of it.

One atom, two implementations, and the pair is the clearest example in the stack
of a rule that sounds abstract until you write it down: **the hardest implementer
sets the contract.**

Asked "is the hammer in the bin", a rigid body answers yes or no. Asked "is the
shirt in the bin", a garment cannot -- part of it is over the rim, always, and
"how much" is the only honest answer. Two returns types would be the obvious
design and it is the wrong one: a goal written against the rigid version would
break the day somebody pointed it at cloth, and every caller would grow a branch.

So containment returns a **fraction, everywhere**. The rigid case is the easy
case of that question, not a different one -- it returns 0.0 or 1.0 and loses
nothing. The cloth case returns what it actually knows. A goal thresholds a
number either way and reads the same.

That is the contract the ``region`` kind declares, and the contract tests
generate a check for it over every handler registered against the kind, so an
implementation added later inherits the obligation without anybody remembering to
extend a list.

Regions arrive here **already in world coordinates**. A region anchored to a
drawer rides the drawer, and applying that pose is the resolver's job; deciding
what fraction is inside a box is this module's. Split that way, the interesting
half is testable against hand-derived numbers with no scene.
"""

from __future__ import annotations

import numpy as np

from ..semantics.atoms import ArgSpec, AtomSpec, register_atom

__all__ = ["containment", "region_bounds"]


def region_bounds(region: dict, origin=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A region's world corners, given where its frame currently is.

    Args:
        region: ``{"xy_ranges": [x0, y0, x1, y1], "z_range": [lo, hi]}``;
            ``z_range`` defaults to BDDL's 0 to 0.15 m, as the lowering's does.
        origin: Where the region's frame sits now [m]. For a world-anchored
            region this is the origin; for one riding a drawer it is the drawer.

    Returns:
        ``(lo, hi)`` world corners.

    The z window starts at the frame's own origin rather than below it, which is
    the convention that makes "in the bin" mean what people expect: an object
    held *above* an open bin is not in it, and one resting on the bin floor is.
    """
    x0, y0, x1, y1 = (float(v) for v in region["xy_ranges"])
    z0, z1 = (float(v) for v in region.get("z_range", (0.0, 0.15)))
    base = np.asarray(origin, dtype=np.float64)
    lo = base + np.array([min(x0, x1), min(y0, y1), min(z0, z1)])
    hi = base + np.array([max(x0, x1), max(y0, y1), max(z0, z1)])
    return lo, hi


def containment(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """What fraction of ``points`` lies inside the world box ``[lo, hi]``.

    A single point in gives 0.0 or 1.0; a cloud gives the share. That is the
    whole trick -- one function, and the rigid case is the cloud of size one.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if not len(array):
        return 0.0
    inside = np.all((array >= lo) & (array <= hi), axis=1)
    return float(inside.mean())


register_atom(
    AtomSpec(
        name="containment",
        summary="What fraction of a thing lies inside a region.",
        fn=containment,
        args=(ArgSpec(name="thing"), ArgSpec(name="region", kinds=("region",), reads="region")),
        directions=("check", "observe"),
        returns=("fraction",),
        provenance=(
            "Generalises the rigid region test and the cloth particle-fraction test into one "
            "measure. BDDL's own region convention for the box; the fraction is what makes it "
            "answerable for a garment."
        ),
    )
)
