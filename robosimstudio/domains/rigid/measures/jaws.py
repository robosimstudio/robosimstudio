# SPDX-License-Identifier: Apache-2.0
"""What the hand is doing: holding something, or far enough away to have let go.

A parallel gripper closing on a part gives one signal and it takes **two numbers
to read**: the fingers were commanded past the part's width, so holding it means
the *measured* position trails the *command* -- stalled -- at a gap that matches
the part.

Neither number alone suffices, and both failures were observed rather than
reasoned about:

* A **measured gap below the width** is also what empty jaws read after reaching
  a deep command. The part squirted out during the close and the fingers sailed
  on to their target; one such episode carried an empty gripper through a whole
  insertion stroke and reported a seat.
* A **command-versus-measured lag** alone is any fast finger move. Every close
  shows one in transit.

So: stalled, *and* at the part's own dimension.

**The gap comes from the hand and the width from the part.** How a finger
coordinate maps to pad separation is a property of the gripper --
:meth:`~robosimstudio.specs.GripperSpec.jaw_gap`, derived from its span
and travel. How wide the thing is is a property of the thing, and arrives on its
``grasp_box``. A measure that hard-coded either would be a measure of one
scene.

**Both tolerances belong here, unlike the placement thresholds.** ``contact_force``
states no gate because Arena uses two different ones for its two task shapes;
these two are servo tracking noise and fit residual, which are facts about the
instrument and the same whatever is being gripped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.core.math import transform_point

__all__ = ["GAP_BAND", "STALL_MARGIN", "jaw_state"]

STALL_MARGIN = 2.0e-4
"""How far the measured fingers must trail their command to count as stalled [m].

Servo tracking noise at rest is an order of magnitude below this.
"""

GAP_BAND = 1.5e-3
"""How close the measured gap must sit to the part's width to count [m].

A stall *at the part's dimension*, not a transient lag mid-travel. Sized above
the residual of the span-and-travel derivation, which agrees with a
forward-kinematic fit to about 1e-5.
"""


register_atom(
    AtomSpec(
        name="gripped",
        summary="The jaws are stalled at the part's own width, so they are holding it.",
        args=(ArgSpec(name="part", kinds=("grasp_box",), reads="feature"),),
        directions=("check", "observe"),
        returns=("gripped", "gap", "lag"),
        unit="m",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "The signal a real linear gripper has, in place of a force sensor it does not. "
            "Both halves and both of their failure modes were measured on a USB-C insertion "
            "rig; see this module's docstring."
        ),
        calibration=AtomCalibration(
            positives=(
                "A held 6.5 mm plug reads a gap within a millimetre of its width with the "
                "fingers trailing their command by more than 0.2 mm, held for the whole "
                "transport."
            ),
            negatives=(
                "Empty jaws that reached a deep command read a gap far below the part's width "
                "with no lag at all -- which is why the gap alone is not the signal. A finger "
                "move in transit shows a lag at a gap that matches nothing."
            ),
            gate=None,
            bench="the USB-C insertion scene",
            notes=(
                "The two tolerances are on the atom rather than the task: they are servo noise "
                "and fit residual, which are properties of the instrument and do not change "
                "with what is being gripped."
            ),
        ),
    )
)


@handler(
    "gripped",
    kinds=("grasp_box",),
    substrates=("rigid",),
    summary="A parallel jaw's stall-at-width test against a declared grasp box.",
)
def _jaw_stall(snapshot, box: dict[str, Any]) -> dict[str, Any]:
    """Reads the work arm's finger coordinates against their commanded targets.

    Fails **closed** -- reporting not-gripped rather than raising -- when the
    scene carries no commanded targets, the arm declares no finger
    coordinates, or the hand is geared (no linear gap to compare with the
    part's width; :meth:`GripperSpec.jaw_gap` refuses it too). That matches the contact predicates: a task may legitimately be
    scored in a scene with no gripper, and a measure that raised there would
    make an unrelated task uncompilable.
    """
    context = snapshot.ctx
    handles = snapshot.handles
    # By role, never by a "right" literal: the two YAM cells order their arms
    # oppositely, so a literal names a different hand on each.
    name = handles.work_arm or next(iter(handles.arms), "")
    arm = handles.arms.get(name)
    if arm is None or not arm.gripper_coords or not len(context.joint_target_q):
        return {"gripped": False, "gap": float("nan"), "lag": float("nan")}

    joint_q = context.state.joint_q.numpy()
    measured = [float(joint_q[c]) for c in arm.gripper_coords]
    commanded = [float(context.joint_target_q[c]) for c in arm.gripper_coords]

    # Both as travel towards closed, so the lag is positive when the command
    # is further shut than the fingers got -- whichever way the joints run. A
    # raw ``measured - commanded`` is that only on a hand that closes towards
    # zero; on a 2F-85 (open at 0, closing upward) it is negative on every stall.
    lag = arm.closing_travel(commanded) - arm.closing_travel(measured)
    try:
        gap = arm.jaw_gap(measured)
    except ValueError:
        # A geared hand has no linear gap to compare with the part's width;
        # fail closed, as with no fingers, rather than compare a wrong curve.
        return {"gripped": False, "gap": float("nan"), "lag": float(lag)}
    width = _width_across(box)
    held = bool(lag > STALL_MARGIN and abs(gap - width) < GAP_BAND)
    return {"gripped": held, "gap": float(gap), "lag": float(lag)}


def _width_across(box: dict[str, Any]) -> float:
    """The part's width across the closing direction [m].

    A ``grasp_box`` states half-extents in its own frame, and the jaws close on
    the *narrowest* of them -- which is what a parallel gripper does when the
    approach is free to choose a yaw, and what every shipped grasp planner here
    already assumes.
    """
    extents = np.asarray(box.get("half", ()), dtype=np.float64)
    if extents.size == 0:
        raise ValueError("a grasp_box needs 'half' extents for its width to be read")
    return 2.0 * float(np.min(extents))


def jaw_state(ctx, part: str) -> tuple[bool, float]:
    """``(held, gap)`` for a live context, from the part's declared grasp box.

    The convenience form of the ``gripped`` atom, for a graph tool that wants the
    numbers without building a goal clause -- and calling the same handler the
    goal calls, so the two cannot disagree about whether the jaws are holding
    anything.
    """
    from robosimstudio.assetlib.registry import ASSETS
    from robosimstudio.assetlib.semantics.snapshot import Snapshot

    handle = ctx.handles.entity(part)
    record = ASSETS.get(handle.category) if handle.category else None
    boxes = record.features_of_kind("grasp_box") if record is not None else ()
    if not boxes:
        return False, float("nan")
    reported = _jaw_stall(Snapshot(ctx=ctx), boxes[0].payload)
    return bool(reported["gripped"]), float(reported["gap"])


register_atom(
    AtomSpec(
        name="tcp_distance",
        summary="How far a part sits from the work arm's tool-centre point [m].",
        args=(ArgSpec(name="part", reads="position"),),
        directions=("check", "observe"),
        returns=("distance",),
        unit="m",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "The observable half of 'the hand let go', for a hold that leaves no stall "
            "signal. Some grasps are virtual -- a constraint rather than a contact -- and "
            "the fingers never touch the part at all, so `gripped` has nothing to read. "
            "What released physically means there is that the constraint is off and the "
            "hand has left, and the distance is the half a measure can see."
        ),
        calibration=AtomCalibration(
            positives=(
                "A hand withdrawn after releasing an RJ45 plug reads well past a jaw span "
                "from it; the shipped task gates at 60 mm against an 85 mm span."
            ),
            negatives=(
                "A hand still holding, or hovering to re-grip, reads inside the span. This "
                "measure cannot tell those two apart -- which is why it is a *release* test "
                "and not a grasp test, and why `gripped` exists for hands that stall."
            ),
            gate=None,
            bench="rj45_insert/insert (retired; the measurement stands)",
            notes=(
                "No gate: how far is far enough depends on the hand's span and on how close "
                "the task needs it to hover, both of which are the task's to know."
            ),
        ),
    )
)


@handler("tcp_distance", kinds=("",), substrates=("rigid",), summary="Straight-line distance to the work arm's TCP.")
def _tcp_distance(snapshot, position) -> dict[str, Any]:
    """Distance from the part to the tool centre, or infinity with no arm.

    Infinity rather than an error, and rather than zero: a scene with no arm has
    nothing holding anything, so a clause reading "further than X from the hand"
    should hold. Zero would say the opposite, which is the wrong way for a
    fail-closed measure to fail.

    The arm is resolved **by role**. A ``"right"`` literal names a different hand
    on each of the two YAM cells, which list their arms in opposite orders.
    """
    handles = snapshot.handles
    name = handles.work_arm or next(iter(handles.arms), "")
    arm = handles.arms.get(name)
    if arm is None or arm.ee_body < 0 or arm.ee_body >= len(snapshot.body_q):
        return {"distance": float("inf")}
    tcp = transform_point(snapshot.body_q[arm.ee_body], arm.ee_offset)
    return {"distance": float(np.linalg.norm(np.asarray(position, dtype=np.float64) - tcp))}


register_atom(
    AtomSpec(
        name="hands_clear",
        summary="How far a part sits from the NEAREST tool-centre point of any arm [m].",
        args=(ArgSpec(name="part", reads="position"),),
        directions=("check", "observe"),
        returns=("distance",),
        unit="m",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "The bimanual release test. `tcp_distance` answers for the work arm only, "
            "which on a two-handed set-down leaves the support hand unexamined -- a strip "
            "parked with one hand still around it read released. Same operand and return "
            "as tcp_distance ON PURPOSE: the differing claim is the minimum over EVERY "
            "arm's TCP, and a goal chooses between them by which hands it means."
        ),
        calibration=AtomCalibration(
            positives=(
                "Both YAM hands retreated after a bimanual set-down read well past a jaw "
                "span from the part; the polish task gates at 100 mm against an 85 mm span."
            ),
            negatives=(
                "Either hand still at a grasp station reads inside the span; a hand "
                "hovering to re-grip is indistinguishable from one about to hold, which is "
                "why this is a *release* test and not a grasp test."
            ),
            gate=None,
            bench="polish/strip set-down",
            notes=(
                "No gate: how far is far enough depends on the hands' spans and on how "
                "close the task lets them hover, both of which are the task's to state."
            ),
        ),
    )
)


@handler("hands_clear", kinds=("",), substrates=("rigid",), summary="Min straight-line distance to any arm's TCP.")
def _hands_clear(snapshot, position) -> dict[str, Any]:
    """Distance from the part to the nearest TCP over *every* arm, or infinity
    with none -- the same fail-open direction as ``tcp_distance``, for the same
    reason: a scene with no arms has nothing holding anything, so a release
    clause should hold."""
    p = np.asarray(position, dtype=np.float64)
    best = float("inf")
    for arm in snapshot.handles.arms.values():
        if arm is None or arm.ee_body < 0 or arm.ee_body >= len(snapshot.body_q):
            continue
        tcp = transform_point(snapshot.body_q[arm.ee_body], arm.ee_offset)
        best = min(best, float(np.linalg.norm(p - tcp)))
    return {"distance": best}
