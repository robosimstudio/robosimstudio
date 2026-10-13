# SPDX-License-Identifier: Apache-2.0
"""The arena YAM's linear gripper, and the three ways it used to be mis-specified.

The robots_realtime arena model's hand: two prismatic fingers on mirrored ranges,
**one** of which is actuated. The other is carried by an ``<equality>``. That one
detail is behind all three defects this file exists to fix, and each failed
differently:

**One actuator, two joints.** ``assets/yam/yam_bimanual_arms.xml`` declares
``{arm}_left_finger`` on ``[-0.00205, 0.0475]`` and ``{arm}_right_finger`` on
``[-0.0475, 0.00205]``, couples them with ``<joint joint1=... joint2=...
polycoef="0 -1 0 0 0"/>``, and attaches a single ``<position>`` actuator to
``{arm}_left_finger``. The old spec selected fingers by the token ``"finger"``,
which caught both, so the undriven DOF arrived with
``JointTargetMode.NONE`` and :meth:`EntityEmbodiment.apply_actuation`'s
position-mode assertion refused to build -- but *only* on a task that states its
own gripper gains, since the check is skipped when there are none.

The fix is :attr:`GripperSpec.follower_names`, not simply dropping the second
joint. Dropping it would build, and would also move
:class:`~robosimstudio.solvers.Gripper` off its mirrored branch -- and on
*this* hand the two joints genuinely are a mirrored pair, so ``width()`` would
stop being the exact separation and start being a linear interpolation over one
joint's travel. This robot is the tape suite's native one and the tape suite is
the physics reference; changing what its gripper reports is not a thing to do as
a side effect of fixing an actuator count.

**No finger bodies.** The old spec declared none, so
:meth:`~robosimstudio.core.probes.contact.ContactProbe.check_grasp` -- which
returns ``bool(flags) and all(flags)`` over the per-side contact flags -- had no
sides and returned ``False`` for every grasp, forever. ``sim.check_grasp`` is
what the tool-sort and grocery graphs verify every grasp with, so on this robot
they would conclude they were holding air no matter what they were holding.

**Unmeasured axes.** ``close_axis`` and ``approach_axis`` were left at the class
defaults, and ``TestGripperAxes`` could not catch it: that test skips an arm with
fewer than two pad sides, and this arm declared none. Declaring the pads turns
the test on here for the first time.

The travel is the *joint's* range, not the actuator's ``ctrlrange``. The MJCF
actuator says 0.0495 and the joint stops at 0.0475, so the documented open
command was 2 mm past the hard stop -- the same 0.0475 the i2rt hand uses, which
is what the shared hardware family would suggest.
"""

from __future__ import annotations

from robosimstudio.specs import GripperSpec

from ..registry import register_gripper

__all__ = ["FINGER_OPEN", "GRASP_SITE_OFFSET", "GRASP_SITE_QUAT_XYZW", "YAM_LINEAR_ARENA"]

GRASP_SITE_OFFSET = (0.0, -0.044, 0.130)
"""The measured grip point, in the wrist frame [m] -- NOT the MJCF's site.

The MJCF declares its grasp site at ``(0, 0, 0.1347)``, on the wrist axis; the
pads are not there. Measured off the finger colliders (the millimetre gripping
nubs that stud each pad face, both arms checked and identical): the nub field
spans ``y -0.047..-0.041, z 0.120..0.139``, so the line a pinched film or tab
actually sits on is 44 mm off the wrist axis in local -y. Driving the MJCF site
to a grasp point parked the *pads* 44 mm away from it -- invisible to every
latch-carried task, since the latch grabs at the TCP wherever the metal is, and
exactly as visible as a film riding centimetres clear of the fingers the moment
anyone looked.
"""
GRASP_SITE_QUAT_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)
"""A -90 degree twist about z, so the grasp frame matches the site's."""

FINGER_OPEN = 0.0475
"""Finger travel at full open [m] -- the *joint's* upper limit.

Not the actuator's ``ctrlrange``, which says 0.0495 and which this spec used to
carry: the joint stops at 0.0475, so a full-open command was 2 mm into the stop.
"""

YAM_LINEAR_ARENA = register_gripper(
    GripperSpec(
        name="yam_linear_arena",
        # The driven joint, and the one the equality carries. See the module
        # docstring for why the follower is named rather than dropped.
        finger_names=("{arm}_left_finger",),
        follower_names=("{arm}_right_finger",),
        finger_open_q=(FINGER_OPEN,),
        finger_closed_q=(0.0,),
        open_width=FINGER_OPEN,
        tcp_body="",
        tcp_offset=GRASP_SITE_OFFSET,
        tcp_offset_rot=GRASP_SITE_QUAT_XYZW,
        # Measured, by driving the hand open to shut and reading the pad's own
        # travel in the tool frame.
        # This is the third of the three defects in the module docstring, and the
        # most quietly expensive: the axis was never stated, so it inherited the
        # class default of local +y, and the pads really slide along local **x**.
        # Ninety degrees out. Nothing noticed for as long as every grasp pose on
        # this robot came from the OBB heuristic, which yaws the home wrist about
        # world +z and reads neither axis -- but a grasp frame composed from the
        # declared axes closes the jaws *along* the object instead of across it.
        close_axis=(1.0, 0.0, 0.0),
        approach_axis=(0.0, 0.0, 1.0),
        # Measured the same way, on tape_peel/handle -- a task that leaves this
        # hand's colliders alone, which matters: a suite that retrofits the
        # fingertips would have this measure its retrofit.
        tip_reach=0.01214,
        # span is deliberately still 0, and falls back to 2 * open_width. Unlike
        # the axes and the reach above, it is not measurable from this model: the
        # pad *origins* move 83 mm -> 108 mm apart as the hand closes, so their
        # separation is not the jaw gap and subtracting it would produce a
        # confident wrong number. The derived 0.095 is at least dimensionally
        # honest for a prismatic pair; replacing it wants a gauge object closed
        # on under physics.
        #
        # housing_* and pad_* are unmeasured and stay 0 rather than borrowing the
        # i2rt hand's numbers -- a different gripper's geometry is not a default,
        # and robot.describe_gripper reports `stated: false` so a node planning
        # around a housing knows to decline rather than trust a zero.
        finger_bodies=(("{arm}_link_left_finger",), ("{arm}_link_right_finger",)),
        open_axis_signs=(1.0, -1.0),
    )
)
