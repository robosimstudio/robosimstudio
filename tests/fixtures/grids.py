# SPDX-License-Identifier: Apache-2.0
"""Synthetic geometry: sheets and chains with known answers.

Promoted from ``TestFoldMetric``'s ``setUp``, which built a flat grid, a real
``ParticleProbe`` and a stand-in context in order to score the production fold
measure against hand-derived numbers. That pattern turned out to be the right
one for the whole semantic layer, so it lives here rather than in one test class.

The layouts are chosen so that a wrong implementation and a right one *disagree*:

* :func:`sheet` -- flat, the baseline every material-frame measure is stated
  against.
* :func:`ragged_sheet` -- a sleeve pushed out on one band. A boundary-tracing
  selector and a rectangle selector agree on a square sheet and differ here,
  which is the whole reason the shirt's flap selector is not a plain edge band.
* :func:`folded` -- the moving half reflected onto the standing half, one sheet
  thickness up. The ideal a fold measure should score near 1.
* :func:`dragged` -- the degenerate solution: halve the extent, stack nothing.
  Any measure without a drag penalty scores this as a fold, which is exactly why
  the drag term exists.
* :func:`helix` -- an analytic winding, so turns can be checked against a number
  somebody derived rather than against whatever the simulator produced.
"""

from __future__ import annotations

import numpy as np

__all__ = ["chain", "dragged", "folded", "helix", "ragged_sheet", "sheet"]


def sheet(*, cells: int = 21, size: float = 0.30, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """A flat square sheet, ``[cells**2, 3]``, centred where it is laid out.

    Defaults to the towel's own 0.30 m so a number derived here means something
    against the shipped suite, and to the origin because most measures are stated
    in the sheet's own frame.

    ``center`` exists because a *garment's* sets are not: a cloth's rest layout
    is the world one -- ``ClothSpec.pose`` is consumed as world coordinates --
    so the shirt's creases sit at its placement plus their offset, and a selector
    checked against an origin-centred sheet would be checked against a garment
    nobody lays out.
    """
    axis = np.linspace(-size / 2.0, size / 2.0, cells)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([gx.ravel() + center[0], gy.ravel() + center[1], np.zeros(gx.size)], axis=1)


def ragged_sheet(*, cells: int = 21, size: float = 0.30, reach: float = 0.04, at: float = 0.06) -> np.ndarray:
    """A sheet with a sleeve pushed out on one band along y.

    The layout that separates a boundary from a rectangle. A shirt's side flap is
    not a rectangle -- the sleeve reaches about 3 cm further out than the shoulder
    or the hem corner -- so a plain extreme band selects only the sleeve tip: a
    4 cm patch on a 31 cm flap. A selector that traces the boundary picks the
    whole edge; one that does not picks the tip, and on a square sheet the two are
    indistinguishable.
    """
    rest = sheet(cells=cells, size=size)
    band = np.abs(rest[:, 1] - at) < (size / cells)
    rest[band, 0] += reach
    return rest


def folded(rest: np.ndarray, *, axis: int = 1, origin: float = 0.0, thickness: float = 0.001) -> np.ndarray:
    """``rest`` with the half beyond ``origin`` reflected onto the other.

    ``thickness`` lifts the carried layer, because a real fold cannot put a
    particle where its mirror already is -- the carried layer comes to rest *on
    top of* the standing one, one sheet thickness up. A measure that forgets this
    caps a geometrically perfect fold below 1.
    """
    out = rest.copy()
    moving = rest[:, axis] > origin
    out[moving, axis] = 2.0 * origin - out[moving, axis]
    out[moving, 2] += thickness
    return out


def dragged(rest: np.ndarray, *, axis: int = 1, distance: float | None = None) -> np.ndarray:
    """``rest`` shoved sideways: the degenerate solution a fold measure must reject.

    Halves the sheet's extent along ``axis`` while stacking nothing. Every measure
    without a drag penalty scores this as a fold.
    """
    if distance is None:
        distance = float(rest[:, axis].max() - rest[:, axis].min()) / 2.0
    out = rest.copy()
    out[:, axis] -= distance
    return out


def chain(*, segments: int = 28, length: float = 0.9144, start: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """A straight chain of segment centres, ``[segments, 3]``.

    The cable's own 28 segments over 0.9144 m by default, so a winding measured
    here is measured on the shipped geometry.
    """
    s = np.linspace(0.0, length, segments)
    out = np.zeros((segments, 3))
    out[:, 0] = np.asarray(start)[0] + s
    out[:, 1] = np.asarray(start)[1]
    out[:, 2] = np.asarray(start)[2]
    return out


def helix(*, turns: float = 2.0, radius: float = 0.05, segments: int = 28, centre=(0.0, 0.0), rise: float = 0.0):
    """An analytic winding about a vertical axis at ``centre``, ``[segments, 3]``.

    Exists so a winding measure can be checked against a number somebody derived
    rather than against whatever a simulator happened to produce. Signed: reverse
    ``turns`` and a correct measure reports the same magnitude, which is what
    makes a figure-eight cancel and a hank dumped beside the peg score zero.
    """
    angle = np.linspace(0.0, 2.0 * np.pi * turns, segments)
    out = np.zeros((segments, 3))
    out[:, 0] = centre[0] + radius * np.cos(angle)
    out[:, 1] = centre[1] + radius * np.sin(angle)
    out[:, 2] = np.linspace(0.0, rise, segments)
    return out
