# SPDX-License-Identifier: Apache-2.0
"""Coverage: how much of one thing another thing is over.

**Three questions that sound alike and are not**, which is why this is its own
atom rather than an option on an existing one:

``containment``
    What fraction of A's points are inside B's *box*. A volume test, and the one
    BDDL's ``In`` binds to. Says nothing about area: a crumpled sheet entirely
    inside a bin scores 1.
``cloth_coverage``
    How spread out A is, against **A's own** settled footprint. A measure of
    flatness. Says nothing about where A is.
``covers`` (here)
    How much of **B's** top-down footprint is under A. The question a protective
    film over a panel asks, and the one neither of the others can answer -- a
    film peeled halfway is still entirely "in" the panel's box and still as flat
    as it ever was.

**Measured on an occupancy grid, not on a bounding rectangle**, and for the
reason :meth:`~robosimstudio.core.probes.particle.ParticleProbe.covered_area`
gives at length: a rectangle is defined by four extreme points, so a sheet lying
in a ball with one corner flicked out covers nearly the same rectangle as one
lying flat. A task scored on the rectangle is solved by flicking the corner out,
and will be.

The cell size is the resolution of the answer and wants to be a few times the
mesh spacing -- below that the gaps *between* particles read as uncovered, far
above it two different arrangements round to the same number. It is a parameter
for the same reason it is one there: that ratio is a property of the cloth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..semantics.atom import SpatialRelation
from ..semantics.atoms import ArgSpec, AtomCalibration

__all__ = ["CELL", "Covers", "cells_of", "overlap"]

CELL = 0.01
"""Grid square edge [m]. The probe's own default, named here so the two agree."""


def cells_of(points: np.ndarray, cell: float = CELL) -> set[tuple[int, int]]:
    """Which xy grid squares a cloud of points occupies.

    Packed as integer pairs rather than as a float grid: the question is which
    squares, and two points in one square are one square however close they are
    to its edge.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    finite = array[np.isfinite(array).all(axis=1)]
    if finite.size == 0:
        return set()
    keys = np.floor(finite[:, :2] / cell).astype(np.int64)
    return {(int(a), int(b)) for a, b in keys}


def overlap(cover: np.ndarray, target: np.ndarray, cell: float = CELL) -> dict[str, Any]:
    """What share of ``target``'s squares are also ``cover``'s.

    The denominator is the *target's* footprint, which is the whole point: a
    sheet twice the size of the panel it protects still covers the panel exactly
    once, and a fraction over 1 would mean the question was asked backwards.
    """
    target_cells = cells_of(target, cell)
    if not target_cells:
        return {"fraction": 0.0, "area_m2": 0.0}
    shared = target_cells & cells_of(cover, cell)
    return {"fraction": len(shared) / len(target_cells), "area_m2": len(shared) * cell * cell}


class Covers(SpatialRelation):
    """What share of one thing's footprint another thing lies over, from above.

    Top-down, and that is a real restriction worth stating rather than
    discovering: this answers "is the panel covered", not "is the box wrapped".
    A sheet draped down a side contributes nothing, because from above it is not
    over anything. The tasks that ask this -- a film on a panel, a towel on a
    table -- are all top-down, and the day one is not, the honest move is a
    second atom that names its projection rather than a flag on this one.
    """

    name = "covers"
    summary = "What fraction of a surface's top-down footprint is covered by something else."
    args = (ArgSpec(name="cover", reads="handle"), ArgSpec(name="target", reads="handle"))
    returns = ("fraction", "area_m2")
    directions = ("check", "observe")
    cost = "expensive"
    reads_scene = True
    provenance = (
        "Minted here. The occupancy-grid method is core/probes/particle.py's covered_area, which "
        "measures a set against its own settled footprint; this measures one set against another's, "
        "which is what a peel is scored on and what containment cannot see."
    )
    calibration = AtomCalibration(
        positives="a film at rest over its panel reads 0.95-1.00 before the peel starts",
        negatives="the same film after a completed peel reads under 0.05, and beside the panel reads 0.00",
        gate=None,
        bench="peel/panel at reset and at termination; a hand-written 2x2 grid fixture for the arithmetic",
        notes=(
            "No gate: what counts as covered is the task's. The peel documents score the fraction "
            "directly rather than thresholding it, which is what a graded measure is for."
        ),
    )


def footprint(snapshot, name: str) -> np.ndarray:
    """The points whose shadow is this thing's footprint.

    Two shapes of answer, because two shapes of thing. A deformable *is* its
    particles, so its footprint is where they are. A rigid body is a pose and a
    box, so its footprint is that box's corners at that pose -- eight points,
    which is enough to fill the squares between them only because the grid is
    coarse relative to the box. A body smaller than a cell reads as one square,
    which is the honest answer at this resolution rather than a rounding error.
    """
    # ``snapshot.particles`` is a dict of probes **keyed by deformable**, one
    # per entity -- not a single probe with a ``sets`` table. Asking the dict
    # for ``.sets`` returns the default every time, so this branch was
    # unreachable and every cloth fell through to ``handles.entity``, which
    # holds objects and statics and raised. That is why no shipped document
    # scores a `covers` with a cloth operand: it could not.
    probe = (snapshot.ctx.particles or {}).get(name)
    if probe is not None:
        # The probe carries a set under its own entity's name (``EntityCloth``
        # registers one at bind); ``None`` asks for every particle it owns,
        # which is the same answer for a sheet with no declared subsets.
        return np.asarray(probe.points(name if name in probe.sets else None), dtype=np.float64)
    handle = snapshot.ctx.handles.entity(name)
    centre = np.asarray(handle.position(snapshot.body_q), dtype=np.float64)
    obb = getattr(handle, "obb", None)
    if obb is None:
        return centre.reshape(1, 3)
    half = np.asarray(obb.half, dtype=np.float64)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return centre + signs * half


def _covers(snapshot, cover, target) -> dict[str, Any]:
    """Both operands arrive as handles and each is read as whatever it is.

    One function, registered for each pair it has been calibrated on. Not one
    unconstrained registration: a handler that names no substrate matches
    everything, which makes precedence meaningless and -- more to the point here
    -- would silently answer for a *rod*, measuring a cable's eight box corners
    as though they were a sheet. The pairs below are the ones where the footprint
    of the thing is the footprint this computes.
    """
    return overlap(footprint(snapshot, cover), footprint(snapshot, target))
