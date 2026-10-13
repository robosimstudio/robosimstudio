# SPDX-License-Identifier: Apache-2.0
"""The i2rt linear_4310, the blade-tip gripper on the composed YAM.

Two prismatic fingers with a 0.0475 m travel behind linear_4310 blade tips, both
actuated and both running the same sign -- *not* the mirrored prismatic
convention, which is why this hand states its own open and closed coordinates
despite having two joints.

Every constant here is a fact about the gripper, measured (in the reference sims)
on the built model with probe bodies:

* Blade-tip gap in metres is ``2.0 * cmd`` -- at the 0.0475 command the tips are
  94 mm apart, at 0 they touch.
* The closing axis in the gripper frame is essentially local ``-y``, fitted from
  the pad centres. The naive axis read off the tip *body origins* is 84.5 degrees
  wrong, and the origins move **apart** as the blades close, so never read
  ``|tip_left - tip_right|`` as a gap.

What is deliberately *not* here is a command-for-gap formula. The free gap a part
sees depends on which geometry actually collides -- the full fingertip, a trimmed
blade, or a fitted box -- and the source sims measured a different intercept for
each. A task that derives grip commands calibrates against its own collider at
build time (the USB-C suite's ``_calibrate_jaw_gap`` does exactly this); a
constant here would be the untrimmed number that measurably mis-squeezes a
trimmed jaw: 16.5 degrees of grasp tilt, part thrown.

**And WHICH geometry collides is not the one that is drawn, unless the
embodiment asks for it.** ``tip_left.stl`` is an open A-frame -- a pad plate on a
triangular yoke, in three separate closed shells -- and MuJoCo collides every
mesh geom as its convex hull. The hull fills the yoke in solid: 3.13x the blade's
volume, with an inner face standing ~13 mm proud of the pad. A hand left that way
grips on a surface that is neither drawn nor where the rubber is; probed on
``polish/strip`` at the instant its pick reported a grasp, all four pads were
**13-14 mm clear of the part** while their hulls were 1.2-2.3 mm inside it. That
is the discrepancy the paragraph above records: the gap formula is fitted to the
blade tips, the contact was on the hull, and the two are 13 mm apart.

:attr:`~robosimstudio.specs.robot.EmbodimentSpec.decompose_colliders` is what
puts the contact back on the blade, and ``yam_i2rt`` sets it. Measured there,
both hands closing on a 4 mm strip, per pad rather than per hand (see
``ToolsPolish.pad_forces`` for why the difference decides whether a working grasp
reads as a grasp at all)::

    finger command   3.0 mm    2.0 mm    1.0 mm    0.0 mm
    grip force       1.0 N    13.2 N    28.5 N    44.5 N

First touch at ~3.0 mm on a 4 mm part, then about 20 N per millimetre, and the
part does not move while the jaws close (0.01 mm). None of that was true of the
hull, whose faces are not parallel and whose "grip" was mostly a net push.

The names are templated. Both YAM arms carry this hand and its joints are
``right_joint7`` / ``left_joint7``, so ``{arm}`` is substituted per arm by
:meth:`GripperSpec.for_arm` -- one entry, two hands.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.specs import GripperSpec

from ..registry import register_gripper

__all__ = [
    "GRASP_SITE_ROT",
    "GRIP_CLOSE_MEASURED",
    "GRIP_LOCAL_EXT",
    "GRIP_OPEN",
    "JAW_GAP_OFFSET",
    "JAW_GAP_PER_CMD",
    "JAW_THICKNESS",
    "LINEAR_4310",
    "TCP_OFFSET",
]

TCP_OFFSET = (0.0, 0.0, -0.1347)
"""Grasp site in the ``gripper`` body's frame [m]: 134.7 mm along local -z,
between the pads. The MJCF's ``grasp_site``."""

GRASP_SITE_ROT = (1.0, 0.0, 0.0, 0.0)
"""Rotation from the ``gripper`` body frame to the grasp frame, xyzw: 180 degrees
about x, the MJCF ``grasp_site``'s own ``quat="0 1 0 0"``. The gripper *extends*
along body -z, so this flip is what makes the grasp frame's +z the approach
direction -- the convention every core solver speaks, and the rotation whose
absence from ``ArmHandle`` made the Cartesian servo measure this hand backwards."""

GRIP_OPEN = 0.0475
"""Finger joint coordinate at full open [m]. Larger command = more open."""

JAW_GAP_PER_CMD = 2.0
"""Blade-tip gap per unit of finger command: ``tip_gap = 2.0 * cmd`` [m/m].
Both fingers move, so a millimetre of command is two of opening."""
JAW_GAP_OFFSET = 0.00536
"""Blade *centroid* gap exceeds the tip gap by this much [m]. Bookkeeping for the
centroid-frame gap readings the reference sims quote; not a surface gap."""
JAW_THICKNESS = 0.02992
"""One blade's AABB extent along its slide direction [m], from the trimmed 30 mm
blade cut. Bookkeeping for the same readings."""

GRIP_LOCAL_EXT = np.array([0.0, 0.0, -1.0])
"""Gripper-local extension axis: from the wrist towards the grasp point."""
GRIP_CLOSE_MEASURED = np.array([0.0053, -1.0, 0.0]) / np.linalg.norm(np.array([0.0053, -1.0, 0.0]))
"""Gripper-local blade-closing axis, fitted from the pad centres."""

LINEAR_4310 = register_gripper(
    GripperSpec(
        name="linear_4310",
        # Explicit names rather than a prefix + finger token: the fingers are
        # joint7/joint8, and no single substring separates them from
        # joint1..joint6 under one prefix.
        finger_names=("{arm}_joint7", "{arm}_joint8"),
        # Both fingers run 0 (closed) to 0.0475 (open) on the same sign -- not
        # the mirrored prismatic convention, which is why these are stated.
        finger_open_q=(GRIP_OPEN, GRIP_OPEN),
        finger_closed_q=(0.0, 0.0),
        open_width=GRIP_OPEN,
        # Blade-tip separation at full open, measured: 2.0 * 0.0475 - 0.001.
        # Stated rather than derived because the -1 mm is real.
        span=0.09396,
        tcp_body="{arm}_gripper",
        tcp_offset=TCP_OFFSET,
        tcp_offset_rot=GRASP_SITE_ROT,
        approach_axis=tuple(float(v) for v in GRIP_LOCAL_EXT),
        close_axis=tuple(float(v) for v in GRIP_CLOSE_MEASURED),
        finger_bodies=(("{arm}_tip_left",), ("{arm}_tip_right",)),
        # 35 mm, and it disagrees with the static measurement -- deliberately.
        #
        # The cable-routing routine states this as prose: "the blade tips, 35 mm
        # below the TCP, swinging with the wrist, clipped the cable".
        # A gripper-measurement script reads 10.02 mm on both the cable-routing and
        # the USB-C insertion scenes, which is the furthest pad corner past the TCP with the
        # wrist *still*. The routine's number is about a swept volume; the
        # measurement is about a pose.
        #
        # The larger one is kept because of what this field is used for. It
        # feeds a clearance -- how high to carry, how far above an object to
        # hover -- and a clearance that is too large costs travel while one that
        # is too small drives the blades through the thing being avoided, which
        # is the failure the routine's sentence was recording. Erring toward the
        # swept number is the safe direction, and stating both is what stops the
        # next reader "correcting" it to 10 mm.
        tip_reach=0.035,
    )
)
