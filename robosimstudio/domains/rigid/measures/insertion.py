# SPDX-License-Identifier: Apache-2.0
"""Insertion: how far in, and how far off axis.

Two numbers about one relationship, and they have to be reported together
because either alone scores a miss. Depth measured without a lateral guard
counts a plug that slid *past* the port face as fully inserted -- it is beyond
the mouth plane along the axis, and nothing in that arithmetic knows it is
40 mm to the left. So this reports both, and a task states both gates.

**The bore's geometry comes from the port, not from the task.** Where the mouth
plane is, which way the axis runs and how deep the bore goes are facts about the
receptacle, so they live on its ``cavity`` feature and arrive here as a payload.
That is the difference between an atom and a task constant: point this at a
different connector and the numbers change with the part, not with the file that
mentions it.

**Depth is measured to the tab tip through the live orientation.** A tilted plug's
tip is not at its centre plus an offset along the world axis, and the difference
is the whole margin on a 13 mm bore: at 10 degrees of tilt a 6 mm tab is nearly
a millimetre short of where the naive reading puts it, which is 8% of the gate.

Transcribed from the USB-C insertion reference's own ``measure()``. Named that
way rather than by the module that currently holds it, because an asset that
knows which *task* uses it has the dependency backwards -- the measure is a fact
about connectors, and the task is one caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.core.math import quat_to_mat

__all__ = ["insertion"]


def insertion(
    pose: np.ndarray,
    cavity: dict[str, Any],
    *,
    tip_offset: float = 0.0,
) -> dict[str, float]:
    """Depth past the mouth and lateral offset from the axis, in metres.

    Args:
        pose: The inserted part's world pose, ``[x, y, z, qx, qy, qz, qw]``.
        cavity: The bore, **already in world coordinates**: ``axis`` (unit),
            ``mouth`` (a point on the mouth plane), ``depth`` (bore length).
            Resolving the receptacle's own frame is the reader's job, as it is
            for a region -- deciding what is inside a bore is this function's.
        tip_offset: How far the measured tip sits along the part's own +z from
            its origin [m]. Zero measures the origin, which is what a part with
            no declared tip gets.

    Returns:
        ``depth`` -- tip past the mouth plane along the axis, negative outside.
        ``lat`` -- distance from the bore axis in the plane normal to it.
        ``mate_gap`` -- travel still to go before the tip reaches the bore floor.
        ``tilt`` -- angle between the part's own +z and the bore axis [rad],
        unsigned, in ``[0, pi/2]``.

    ``mate_gap`` is reported because one shipped suite scores on it rather than
    on depth: ``rj45`` rewards ``clip(1 - gap / travel)``, a measure that shrinks
    toward success. Deriving it here rather than in each task keeps the two
    readings arithmetically consistent -- ``depth + mate_gap`` is the bore depth,
    always, and a task that computed one from the other could drift.
    """
    position = np.asarray(pose[:3], dtype=np.float64)
    axis = np.asarray(cavity["axis"], dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    mouth = np.asarray(cavity["mouth"], dtype=np.float64)
    depth_of_bore = float(cavity.get("depth", 0.0))

    # The tip through the live orientation, not along the world axis.
    rotation = quat_to_mat(np.asarray(pose[3:7], dtype=np.float64))
    tip = position + rotation @ np.array([0.0, 0.0, float(tip_offset)])

    depth = float(np.dot(tip - mouth, axis))
    offset = tip - mouth
    lateral = offset - float(np.dot(offset, axis)) * axis
    # The part's own insertion axis against the bore's. Unsigned, because a
    # plug hanging nose-up and one hanging nose-down are the same failure.
    part_axis = rotation @ np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(abs(float(np.dot(part_axis, axis))), -1.0, 1.0))
    return {
        "depth": depth,
        "lat": float(np.linalg.norm(lateral)),
        "mate_gap": depth_of_bore - depth,
        "tilt": float(np.arccos(cosine)),
    }


register_atom(
    AtomSpec(
        name="insertion",
        summary="How far a part has entered a bore, and how far it sits off the bore axis.",
        args=(
            ArgSpec(name="part", reads="pose"),
            ArgSpec(name="bore", kinds=("cavity",), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("depth", "lat", "mate_gap", "tilt"),
        unit="m",
        cost="cheap",
        provenance=(
            "The USB-C insertion reference's own measure(), plus `tilt` [rad]. Tilt was "
            "deliberately absent -- 'no shipped goal gates on them' -- and that was true "
            "until a goal needed it: depth is an AXIAL PROJECTION, so a plug that pivots "
            "in an oversized bore reports as a plug that stayed in. Measured on "
            "the USB-C insertion scene, the released plug swings from 0.99 to 10.9 degrees "
            "within eighteen frames of the jaws opening and scores reward 1.00, because "
            "depth and lat both still pass. Roll is still not reported: nothing gates it."
        ),
        calibration=AtomCalibration(
            positives=(
                "A seated USB-C plug reads 10.4-13.0 mm of depth at under 1 mm of lateral "
                "offset -- 0.8 of a 13 mm bore is the gate the shipped suite uses."
            ),
            negatives=(
                "A plug that slid past the port face reads a large positive depth at a lateral "
                "offset of tens of millimetres, which is exactly why depth alone is not a "
                "verdict. A plug still in transport reads a negative depth."
            ),
            gate=None,
            bench="the USB-C insertion scene",
            notes=(
                "No gate on purpose. The depth that counts as inserted is a fraction of the "
                "part's own bore, and the lateral guard is the mouth's half-diagonal -- both "
                "are the connector's dimensions, so a task states them from the part it uses."
            ),
        ),
    )
)


@handler(
    "insertion",
    kinds=("", "cavity"),
    substrates=("rigid", "rigid"),
    summary="A rigid part against a rigid receptacle's declared bore.",
)
def _rigid_insertion(pose, cavity: dict[str, Any]) -> dict[str, float]:
    """The tip offset comes from the cavity's own payload.

    Which reads oddly until you ask whose fact it is: *how far along the plug its
    tab reaches* is a property of the plug, but the only thing that cares is the
    bore it goes into, and a bore that accepted parts with different tip offsets
    would be measuring different things under one name. Declaring it beside the
    bore keeps the pair together.
    """
    return insertion(pose, cavity, tip_offset=float(cavity.get("tip_offset", 0.0)))
