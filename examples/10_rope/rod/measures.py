# SPDX-License-Identifier: Apache-2.0
"""Measures over 1-D bodies: winding, straightness.

:func:`winding_turns` is a **port, not a rewrite**. It is the cable suite's own
``cable_winding_turns``, transcribed line for line, because the arithmetic is
already right and the reason it is right is subtle enough to be worth preserving
rather than re-deriving: walk the chain, and for each consecutive pair of
segments near the axis accumulate the *signed* angle their midpoints subtend
about it. The magnitude of that sum is how many turns are actually around the
axis. A hank dumped beside it scores zero; a figure-eight cancels itself. Those
two properties are what "wrapped" needs to mean and what a distance-based measure
cannot express.

Two things generalise in the move:

The measure was written about a peg and is stated here about an **axis**. Nothing
in the arithmetic knew it was a peg -- it needed a vertical line and a radius of
interest -- so the same handler now serves a bollard, a spool, a doorknob or a
hook, and does so without a line of new code the day one of those is annotated.

Its gate carries its **calibration** as data rather than as a docstring. The
number was earned: every wrap the shipped policy produces retains 0.55-0.96 turns
after release, while every non-wrap outcome ever traced -- drag-pasts, slipped
grasps, a two-lap coil springing open -- measures 0.25 or less. Half a turn sits
in the middle of that valley, which a drag-past cannot fake by a factor of three.
Recorded in the atom so the mint gate can *require* such evidence instead of
hoping for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.core.math import transform_point

__all__ = ["NEAR_RADIUS", "WRAP_GATE_TURNS", "straightness", "winding_turns"]

NEAR_RADIUS = 0.10
"""A segment takes part in the winding while within this of the axis, in the
plane normal to it [m]. The reference's own."""

WRAP_GATE_TURNS = 0.5
"""Turns retained after release that count as wrapped [-].

Calibrated into a measured separation rather than onto a wish -- see the atom's
own calibration block, which is the machine-readable form of this note.
"""


def winding_turns(positions: np.ndarray, axis_xy, *, near_radius: float = NEAR_RADIUS) -> tuple[float, int]:
    """Net turns of a chain about a vertical axis, and how many segments engage.

    Args:
        positions: Segment centres in world coordinates, ``[n, 3]``, in chain
            order. Order is load-bearing: the measure walks consecutive pairs,
            so a shuffled array reports nonsense rather than an error.
        axis_xy: Where the axis crosses the plane, ``(x, y)`` [m].
        near_radius: How close a segment must be to take part [m].

    Returns:
        ``(turns, engaged)`` -- the magnitude of the net winding in turns, and how
        many consecutive pairs contributed.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rel = positions[:, :2] - np.asarray(axis_xy, dtype=np.float64)
    dist = np.linalg.norm(rel, axis=1)
    near = dist < float(near_radius)
    total = 0.0
    engaged = 0
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    for i in range(len(positions) - 1):
        if near[i] and near[i + 1]:
            delta = angles[i + 1] - angles[i]
            # Wrapped to (-pi, pi] before accumulating, so a lap that crosses the
            # branch cut adds a small angle rather than a full turn.
            total += float(np.arctan2(np.sin(delta), np.cos(delta)))
            engaged += 1
    return abs(total) / (2.0 * np.pi), engaged


def straightness(positions: np.ndarray) -> float:
    """How straight a chain is: end-to-end distance over path length, in ``[0, 1]``.

    1.0 is a straight line; a coil tends toward 0. The obvious measure, and
    deliberately *not* a fit to a line -- a chain bent into a wide arc is not
    straight, and a residual-to-best-fit measure would call it nearly so.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 1.0
    path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    if path <= 1e-12:
        return 1.0
    return float(np.linalg.norm(positions[-1] - positions[0]) / path)


# ----------------------------------------------------------------- catalogue
register_atom(
    AtomSpec(
        name="winding_turns",
        summary="Net turns of a 1-D body wound about an axis, signed so a figure-eight cancels.",
        fn=winding_turns,
        args=(
            ArgSpec(name="body", substrates=("rod",)),
            ArgSpec(name="about", kinds=("axis", "rod"), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("turns", "engaged"),
        unit="turns",
        provenance="Ported line-for-line from cable_route's cable_winding_turns; the reference ships no predicate.",
        calibration=AtomCalibration(
            positives="0.55-0.96 turns retained after release, over every wrap the shipped routine or graph produces",
            negatives="drag-pasts, slipped grasps and a two-lap coil springing open all measure 0.25 or less",
            gate=WRAP_GATE_TURNS,
            bench="cable_route/port_easy rollouts",
            notes=(
                "Half a turn sits in the middle of the measured valley: physically it means the rod "
                "is bent around the axis and staying there, which a drag-past cannot fake by a factor "
                "of three. Run-to-run variance is real -- the graph thread's timing changes the drag "
                "dynamics lap to lap -- which is why the gate is calibrated to a spread and not a mean."
            ),
        ),
    )
)

register_atom(
    AtomSpec(
        name="straightness",
        summary="How straight a 1-D body is: end-to-end distance over path length.",
        fn=straightness,
        args=(ArgSpec(name="body", substrates=("rod",)),),
        directions=("check", "observe"),
        returns=("straightness",),
        provenance="Designed here; the reference ships no shape predicate.",
    )
)


@handler(
    "winding_turns",
    kinds=("", "axis"),
    substrates=("rod", ""),
    summary="Signed angle accumulated along the chain about a declared axis.",
)
def _winding_about_axis(positions: np.ndarray, axis: dict, *, near_radius: float = NEAR_RADIUS) -> dict:
    """Dispatch entry: a chain's positions against an ``axis`` feature in world."""
    origin = np.asarray(axis["origin"], dtype=np.float64)
    turns, engaged = winding_turns(positions, origin[:2], near_radius=near_radius)
    return {"turns": turns, "engaged": engaged}


@handler(
    "winding_turns",
    kinds=("", "rod"),
    substrates=("rod", ""),
    summary="The same measure about a rod's own axis, so a peg needs no separate annotation.",
)
def _winding_about_rod(positions: np.ndarray, rod: dict, *, near_radius: float = NEAR_RADIUS) -> dict:
    """A rod *is* an axis for this purpose -- its midline, in world."""
    ends = np.asarray(rod["axis"], dtype=np.float64)
    centre = ends.mean(axis=0)
    turns, engaged = winding_turns(positions, centre[:2], near_radius=near_radius)
    return {"turns": turns, "engaged": engaged}


@handler(
    "geometry",
    kinds=("",),
    substrates=("rod",),
    summary="The chain's ordered points in world, and its path length -- a rod's geometry is its centreline.",
)
def _rod_geometry(chain: dict) -> dict[str, Any]:
    """Where the whole rod is right now.

    ``at`` carries the polyline (the ``geometry`` contract's world-frame
    payload), ``length_m`` its path length, ``n_points`` how many points the
    chain publishes. This is the read a generic graph needs to weave a cable
    without the suite's own perception: sample ``at``, or ask for one point by
    arc length with ``<name>.point(s)``.
    """
    points = chain.get("points") or []
    return {
        "at": points,
        "frame": "world",
        "length_m": float(chain.get("length_m", 0.0)),
        "n_points": int(chain.get("n_points", len(points))),
    }


@handler(
    "geometry",
    kinds=("point",),
    substrates=("rod",),
    summary="One point of the chain by normalised arc length, in world.",
)
def _rod_point_geometry(point: dict) -> dict[str, Any]:
    return {"at": point.get("at"), "frame": "world", "s": float(point.get("s", 0.5))}


@handler("straightness", substrates=("rod",), summary="End-to-end over path length.")
def _straightness(positions: np.ndarray) -> dict:
    return {"straightness": straightness(positions)}


# ------------------------------------------------- the generic three, for a rod
#
# ``speed``, ``height`` and ``tcp_distance`` are registered for every substrate
# in the modules that mint them, and a chain wants all three read differently.
# The reason is one property, stated three ways: a 1-D body's answer to each of
# these questions is an **extreme** over its segments, not an average or a
# centroid.
#
# Every one of them is transcribed from ``cable_route``'s own evaluator, which
# computed exactly these three reductions inline before there was a vocabulary to
# put them in -- ``mean(|v_i|)``, ``max(z_i)`` and ``min(|p_i - tcp|)``. They are
# ported rather than re-derived for the reason ``winding_turns`` was: the
# arithmetic is already right, and its rightness is not obvious.


@handler("speed", kinds=("",), substrates=("rod",), summary="Mean of the segment speeds -- not the centroid's.")
def _rod_speed(velocity: np.ndarray) -> dict[str, Any]:
    """How fast a rod is moving: the mean of ``|v_i|``, not ``|mean(v_i)|``.

    The difference is the whole reason this handler exists, and it is not
    subtle. A cable being dragged around a peg has segments travelling in
    opposite directions; the mean *vector* cancels to nearly nothing and the
    generic reading calls a rod mid-wrap "at rest". The mean of the *magnitudes*
    cannot cancel, so it reads what an observer would.

    The chain velocity reader already refuses to average -- "what 'moving' means
    for a chain is the measure's business" -- and this is that decision, made
    once, here.
    """
    array = np.atleast_2d(np.asarray(velocity, dtype=np.float64))
    return {"speed": float(np.mean(np.linalg.norm(array[:, :3], axis=1)))}


@handler("height", kinds=("",), substrates=("rod",), summary="The highest segment -- 'is all of it down'.")
def _rod_height(positions: np.ndarray) -> dict[str, Any]:
    """How high a rod sits: its **highest** segment [m].

    "Has the cable been laid down" is a question about the part still in the
    air, and a mean answers it wrongly in the dangerous direction -- a 0.91 m
    rod held up by its midpoint reads about a third of its lift. The shipped
    evaluator asked for ``max(z)`` and it was right to.
    """
    array = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    return {"height": float(array[:, 2].max())}


@handler(
    "tcp_distance",
    kinds=("",),
    substrates=("rod",),
    summary="Distance to the NEAREST segment -- a hand on any part of the rod is a hand on the rod.",
)
def _rod_tcp_distance(snapshot, positions: np.ndarray) -> dict[str, Any]:
    """How far the work arm's tool centre is from the closest point of a rod.

    The nearest segment, because release is a claim about the whole body: a hand
    that has let go of the middle but is still closed on one end has not let go.
    Infinity with no arm, for the reason the rigid handler gives -- a scene with
    no hand is holding nothing, and a fail-closed measure must fail toward
    "released" rather than away from it.
    """
    handles = snapshot.handles
    name = handles.work_arm or next(iter(handles.arms), "")
    arm = handles.arms.get(name)
    if arm is None or arm.ee_body < 0 or arm.ee_body >= len(snapshot.body_q):
        return {"distance": float("inf")}
    tcp = transform_point(snapshot.body_q[arm.ee_body], arm.ee_offset)
    array = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    return {"distance": float(np.min(np.linalg.norm(array - tcp, axis=1)))}


# ---------------------------------------------------------------------------
# A chain against a region
# ---------------------------------------------------------------------------
#
# They are here now for the reason ``height`` and ``speed`` already were: a
# rod's answers are its own, and the module that knows why should be the one
# that says so.

from robosimstudio.assetlib.vocabulary.regions import containment, region_bounds


@handler(
    "containment",
    kinds=("", "region"),
    substrates=("rod", ""),
    summary="The share of a chain's segments inside the box.",
)
def _rod_in_region(positions, region: dict, *, origin=(0.0, 0.0, 0.0)) -> dict:
    lo, hi = region_bounds(region, origin)
    return {"fraction": containment(positions, lo, hi)}


@handler(
    "containment",
    kinds=("point", "region"),
    substrates=("rod", ""),
    summary="One arc-length point of a chain against the box -- 0 or 1, like the rigid case.",
)
def _rod_point_in_region(position, region: dict, *, origin=(0.0, 0.0, 0.0)) -> dict:
    """A named place along a rod, not the rod.

    Registered separately from the whole-chain handler rather than left to fall
    into it. Both would compute the same number -- the fraction of a cloud of one
    -- but the two questions are different and a dispatch table that says so is
    the point of having one: "is the cable in the lane" is a share, "is the
    cable's midpoint in the lane" is a yes or a no.
    """
    lo, hi = region_bounds(region, origin)
    return {"fraction": containment(np.atleast_2d(position), lo, hi)}
