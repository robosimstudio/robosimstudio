# SPDX-License-Identifier: Apache-2.0
"""Mating geometry: threading a rod through a loop.

The handler the whole prototype-style object model was argued over. A pair of
scissors hangs on a rail because the scissors carry a ``loop`` and the rail
carries a ``rod`` -- not because either declares an interface, inherits anything,
or knows the other exists. Annotate a loop on a mug handle, a keyring or a washer
and every one of them hangs on that rail with nothing written on either side.
Categories never generalise; geometry does, and this is where that stops being a
claim.

**Why the test is geometric rather than contact-based.** The obvious
implementation is "are the hole's contact points touching the rod", and it is
wrong twice over. Attributing contacts to a *subregion* of a mesh is brittle --
the machinery exists, since bodies carry convex decompositions, but the
attribution is fiddly and fails quietly. Worse, contact cannot distinguish
hanging from *leaning*: scissors resting against a rail touch it exactly as much
as scissors hung on it. Transforming the loop's disc by its body pose and asking
whether the rod's axis segment crosses it is exact, cheap, and cannot be faked by
an object merely resting nearby.

Contact still gets a say, but at body level and in a different atom:
``supported_by`` asks whether the hanger is bearing the load and the object has
come to rest. Threading says the geometry is right; support says gravity agrees.
A goal wants both, and keeping them separate is what makes a failure legible --
"through the loop but still held" is a different sentence from "resting on it but
not through".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atom import Relation
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler

__all__ = ["encloses", "feature_geometry", "threaded"]


def encloses(loop: dict, rod: dict) -> dict:
    """Whether a finite rod centreline lies inside a loop opening.

    Unlike :func:`threaded`, this does not require the rod to intersect the
    loop's exact mathematical plane.  A hanging tool may tilt after release
    while the hook remains visibly inside its ring.  The closest point on the
    finite rod segment is therefore compared with the usable opening radius.
    """
    centre = np.asarray(loop["center"], dtype=np.float64)
    inner_r = float(loop["inner_r"])
    # An opt-in, per loop, declared on the feature record: a calibrated radial
    # allowance for a ring whose usable opening is not its modelled inner
    # radius. A wrench ring reads a few millimetres inside its CAD bore once
    # hung, and scissor bows are not circles at all. Absent, the opening is the
    # modelled one, which is what every pre-level document measured.
    enclosure_tolerance = float(loop.get("enclosure_tolerance", 0.0))
    ends = np.asarray(rod["axis"], dtype=np.float64)
    p0, p1 = ends[0], ends[1]
    direction = p1 - p0
    span_sq = float(np.dot(direction, direction))
    if span_sq <= 1e-24:
        raise ValueError("rod axis has zero length; a point cannot lie inside a ring as a rod")

    along = float(np.clip(np.dot(centre - p0, direction) / span_sq, 0.0, 1.0))
    closest = p0 + along * direction
    offset = float(np.linalg.norm(closest - centre))
    clearance = float(inner_r + enclosure_tolerance - offset)
    return {
        "encloses": bool(clearance >= 0.0),
        "offset": offset,
        "clearance": clearance,
        "at": [float(v) for v in closest],
    }


register_atom(
    AtomSpec(
        name="encloses",
        summary="Whether a finite rod centreline lies within a closed loop's opening radius.",
        fn=encloses,
        args=(
            ArgSpec(name="loop", kinds=("loop",), reads="feature"),
            ArgSpec(name="rod", kinds=("rod",), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("encloses", "offset", "clearance"),
        unit="m",
        calibration=AtomCalibration(
            positives=(
                "A hung tool reads +1.7 mm to +6.3 mm of clearance. Measured on the recorded hangs "
                "of the tool-hanging suite's wrenches (level 2: +2.08, +1.66 mm) and scissors (+5.42, "
                "+6.28 mm), each planted and settled 120 steps -- the same fixtures the suite's scene "
                "tests verify, on 2026-08-17."
            ),
            negatives=(
                "The same tools resting on the bench at spawn read -440 to -450 mm: the hook is "
                "most of a half-metre away, which is what the do-nothing floor looks like. The "
                "*nearest* true negative is more interesting and comes from a hung scissor -- its "
                "second handle bow, the one not on the hook, reads -13 to -26 mm while the first "
                "reads +5 to +6 mm. So the valley either side of the gate is 1.7 mm on the pass "
                "side and 13 mm on the fail side, with nothing measured in between."
            ),
            gate=0.0,
            bench="the tool-hanging suite's recorded settled poses: its wrenches and scissors scenes",
            notes=(
                "The gate is zero because the measure is a signed geometric margin rather than a "
                "score: inner radius minus the rod centreline's closest offset. What the "
                "calibration establishes is not where to put a threshold but that the populations "
                "do not straddle the one the geometry already implies -- and the hung-scissor "
                "reading is why the level-2 scissors scene states its goal as an `Or` over both bows "
                "instead of naming one: on a real hang, one bow encloses and the other does not."
            ),
        ),
        provenance=(
            "Defined for freely hanging rings that may tilt away from their authored feature plane. "
            "The finite rod centreline may be offset from the ring centre, but its closest point "
            "must remain within the ring's inner radius. Hook thickness is deliberately not part "
            "of this centreline-membership check."
        ),
    )
)


@handler(
    "encloses",
    kinds=("loop", "rod"),
    summary="Closest point on a finite rod centreline against a loop's usable opening.",
)
def _encloses(loop: dict, rod: dict) -> dict:
    return encloses(loop, rod)


def threaded(loop: dict, rod: dict) -> dict:
    """Whether a rod's axis passes through a loop's disc, in world coordinates.

    Args:
        loop: ``{"center", "normal", "inner_r"}`` in **world** coordinates.
        rod: ``{"axis": [p0, p1], "radius"}`` in **world** coordinates.

    Returns:
        ``{"threaded", "offset", "clearance", "crossings", "at"}`` -- the verdict,
        how far off the disc's centre the crossing lands [m], how much room is
        left between the rod's surface and the loop's rim [m], whether the axis
        crosses the loop's plane at all, and where the crossing is.

        ``offset`` and ``clearance`` are reported whether or not the verdict is
        true, because a near miss and a wild one are different situations and a
        bare boolean cannot say which happened. Negative clearance means the rod
        is too fat for the loop -- geometrically impossible rather than merely
        unachieved, which is worth telling an author apart from a failed attempt.

        **``at`` is where the rod's axis *line* crosses the loop's plane, not
        where the loop is.** For a rod lying nearly parallel to the plane the
        crossing is arbitrarily far away -- a screwdriver resting on a bench
        read its rack hole at x = -18 m -- and ``offset`` is that same
        distance. A caller that wants the loop's own position asks
        :func:`feature_geometry` (atom ``geometry``) for the loop.
    """
    centre = np.asarray(loop["center"], dtype=np.float64)
    normal = np.asarray(loop["normal"], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("loop normal is degenerate; a disc with no orientation has no inside")
    normal = normal / norm
    inner_r = float(loop["inner_r"])

    ends = np.asarray(rod["axis"], dtype=np.float64)
    p0, p1 = ends[0], ends[1]
    direction = p1 - p0
    span = float(np.linalg.norm(direction))
    rod_r = float(rod.get("radius", 0.0))
    if span <= 1e-12:
        raise ValueError("rod axis has zero length; a point cannot be threaded through anything")

    # Where the infinite line crosses the loop's plane. Parallel-and-not-coplanar
    # means no crossing at all, which is a clean 'no' rather than a special case.
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1e-12:
        return {
            "threaded": False,
            "offset": float("inf"),
            "clearance": inner_r - rod_r,
            "crossings": False,
            "at": None,
        }

    t = float(np.dot(centre - p0, normal) / denominator)
    hit = p0 + t * direction
    offset = float(np.linalg.norm(hit - centre))

    # The rod is a *segment*, not an infinite line: hanging past the end of a
    # rail is not hanging. Crossing the plane beyond either end does not count.
    within_span = -1e-9 <= t <= 1.0 + 1e-9
    # The rod's own thickness has to fit through the hole alongside the offset.
    fits = offset + rod_r <= inner_r
    return {
        "threaded": bool(within_span and fits),
        "offset": offset,
        "clearance": float(inner_r - rod_r - offset),
        "crossings": bool(within_span),
        "at": [float(v) for v in hit],
    }


register_atom(
    AtomSpec(
        name="threaded",
        summary="Whether something rod-shaped passes through a closed loop.",
        fn=threaded,
        args=(
            ArgSpec(name="loop", kinds=("loop",), reads="feature"),
            ArgSpec(name="rod", kinds=("rod",), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("threaded", "offset", "clearance", "crossings"),
        unit="m",
        provenance=(
            "Designed here for the hanging fixtures. Geometric rather than contact-based on purpose: "
            "contact cannot tell hanging from leaning, and subregion attribution fails quietly."
        ),
    )
)


@handler(
    "threaded",
    kinds=("loop", "rod"),
    summary="Rod axis segment against the loop's disc -- exact, and unfakeable by resting nearby.",
)
def _threaded(loop: dict, rod: dict) -> dict:
    return threaded(loop, rod)


def feature_geometry(feature: dict) -> dict:
    """Where a declared feature is right now, in world coordinates.

    The honest form of "where is it" for a part that has no body of its own.
    A pegboard hook's shank, a rack's hole and a receptacle's bore are welded
    into scenery: ``sim.get_object_pose`` has nothing to read, and the relation
    atoms answer a different question -- :func:`encloses` reports the closest
    point *on the rod* to the loop's centre, :func:`threaded` reports where the
    rod's axis *line* crosses the loop's plane. Read as a location, the second
    put a rack hole eighteen metres off the bench, and a policy steered there.

    Args:
        feature: The declared payload, already placed in the world by the
            binder's ``feature`` reader: a loop's ``center``/``normal``/
            ``inner_r``, a rod's ``axis`` endpoints and ``radius``, a bore's
            mouth and axis -- whatever the kind declares, points translated and
            rotated, directions rotated, scalars untouched.

    Returns:
        The same payload, as plain lists, plus ``at`` -- one representative
        point (the centre where the payload has one, else the midpoint of its
        axis, else its origin) -- and ``frame``, always ``"world"``, so the
        answer says what it is in.
    """
    out: dict[str, Any] = {}
    for key, value in feature.items():
        array = _coordinates(value)
        out[key] = value if array is None else np.round(array, 6).tolist()
    at: list[float] | None = None
    for key in ("center", "mouth", "center_local", "origin"):
        point = _coordinates(out.get(key))
        if point is not None and point.shape == (3,):
            at = out[key]
            break
    axis = _coordinates(out.get("axis"))
    if at is None and axis is not None and axis.ndim == 2 and len(axis) >= 2:
        at = np.round(axis.mean(axis=0), 6).tolist()
    out["at"] = at
    out["frame"] = "world"
    return out


def _coordinates(value: Any) -> np.ndarray | None:
    """A payload entry as coordinates, or ``None`` if it does not hold any.

    Asked of the value rather than of its type: a payload carries points,
    directions, names and scalars, and only the first two are worth placing.
    ``np.asarray`` answers by trying -- a name refuses to become floats, a
    scalar becomes a zero-dimensional array and is left alone -- which keeps
    this dispatching on what an entry *has* rather than on what it *is*.
    """
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return None if array.ndim == 0 else array


register_atom(
    AtomSpec(
        name="geometry",
        summary="Where a declared feature is right now: its own payload, placed in world coordinates.",
        fn=feature_geometry,
        args=(ArgSpec(name="feature", reads="feature"),),
        directions=("observe",),
        returns=("at", "frame"),
        unit="m",
        cost="cheap",
        provenance=(
            "Minted for the welded fixtures the hanging and racking suites hang things on. Every "
            "relation atom answers a question *between* two things; none answered 'where is this "
            "part', and the two that carry an `at` mean different points by it. This is the "
            "binder's own world-frame feature read, handed back as data."
        ),
        calibration=AtomCalibration(
            positives=(
                "The tool-hanging suite's level-1 wrenches at spawn: the ring's `at` lies 0.170 m from wrench_1's "
                "body origin along its long axis (the asset declares it at [0.17, 0, 0]), and the hook "
                "shank's `axis` runs 0.061 m along world -x from x=0.880 to x=0.819 -- where `encloses.at` "
                "also puts the closest rod point for a ring resting on the bench."
            ),
            negatives=(
                "Its level-2 screwdrivers at spawn: `threaded.at` for the same resting screwdriver reads "
                "x = -18 m, the crossing of a near-parallel axis line with the hole's plane -- the reading "
                "this atom exists to stop being mistaken for a position."
            ),
            gate=None,
            bench="asked of a built scene: where a named feature is, read back off the live pose",
            notes="A reading, never a verdict: there is no threshold to sit anywhere, and it is observe-only.",
        ),
    )
)


@handler(
    "geometry",
    kinds=("",),
    substrates=("rigid",),
    summary="A declared feature's payload, placed in the world by its body's pose or its static transform.",
)
def _geometry(feature: dict) -> dict:
    return feature_geometry(feature)


class Alignment(Relation):
    """The angle between two declared directions.

    **Unsigned, and that is the measurement rather than a convenience.** A plug
    nose-up and the same plug nose-down are the same failure to mate, so the
    reading folds at 90 degrees -- the same reasoning ``insertion``'s tilt term
    already uses, and the reason this does not simply return a signed angle and
    let each task decide.

    **Not a replacement for ``insertion``'s tilt.** That one is reported inside a
    mating report and is contractually never gated, because a cavity's contract
    says so: how square the plug is arriving tells you *why* an insertion is
    stuck, and thresholding it would refuse approaches that work. This is the
    standalone question -- are these two directions parallel -- for the cases
    that are not a mating at all: a hung tool's shank against a peg, a poured
    bottle against vertical.

    The signed cosine rides along, because the fold throws away which side and
    something occasionally wants it back.
    """

    name = "alignment"
    summary = "The unsigned angle between two declared directions [rad]."
    args = (
        ArgSpec(name="axis_a", kinds=("axis", "cavity", "rod"), reads="feature"),
        ArgSpec(name="axis_b", kinds=("axis", "cavity", "rod"), reads="feature"),
    )
    returns = ("angle", "cos")
    unit = "rad"
    directions = ("check", "observe")
    provenance = (
        "Minted here. The arithmetic is insertion's tilt term, lifted out of the mating report so "
        "that a task which wants only the angle does not have to declare a bore to get one."
    )
    calibration = AtomCalibration(
        positives="a plug held square to its port reads under 0.05 rad over a settled hold",
        negatives="the 15-degree approach the usbc rig refuses reads 0.26 rad",
        gate=None,
        bench="the USB-C insertion scene's approach poses; unit fixtures over hand-rotated quaternions",
        notes=(
            "No gate: how square is square enough is a property of the fit, and the two shipped "
            "fits differ by an order of magnitude in clearance."
        ),
    )


def _direction_of(feature: dict) -> np.ndarray:
    """A unit direction out of whichever feature kind was handed over.

    Three kinds carry one, and each spells it differently because each was named
    for what it is rather than for this: an ``axis`` has a ``direction``, a
    ``cavity`` has its insertion ``axis``, and a ``rod`` has two endpoints whose
    difference is the direction. Read here rather than normalised into the kinds,
    because a kind's payload is its own contract and this is one consumer of it.
    """
    if "direction" in feature:
        vector = np.asarray(feature["direction"], dtype=np.float64)
    elif "axis" in feature:
        vector = np.asarray(feature["axis"], dtype=np.float64)
    elif "start" in feature and "end" in feature:
        vector = np.asarray(feature["end"], dtype=np.float64) - np.asarray(feature["start"], dtype=np.float64)
    else:
        raise KeyError(f"feature declares no direction; it has {sorted(feature)}")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("a zero-length direction has no angle")
    return vector / norm


def alignment(first: dict, second: dict) -> dict[str, Any]:
    """The unsigned angle between two features' directions [rad]."""
    cosine = float(np.dot(_direction_of(first), _direction_of(second)))
    return {"angle": float(np.arccos(np.clip(abs(cosine), -1.0, 1.0))), "cos": cosine}


for _a in ("axis", "cavity", "rod"):
    for _b in ("axis", "cavity", "rod"):
        Alignment.implement(
            kinds=(_a, _b),
            summary=f"The angle between {_a}'s direction and {_b}'s.",
        )(alignment)
