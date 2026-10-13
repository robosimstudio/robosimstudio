# SPDX-License-Identifier: Apache-2.0
"""The Menagerie 2F-85, bolted to the tool-sort export's Panda flange.

The same hand and the same physics as :mod:`robotiq_2f85_menagerie` -- one
``2f85.xml``, one published 85 mm stroke, one command-to-gap curve -- and a
separate entry for the reason this package's docstring gives: a spec is keyed to
the *file* that carries the hand, and the body a mount names is the exporter's
label rather than the manufacturer's. The Menagerie entry mounts on
``wrist_3_link``, which is Universal Robots'. A Panda has no such body, and the
failure is the one :class:`GripperSpec.mount_body` documents -- ``'wrist_3_link'
matched 0 labels``.

**The mount is the export's own flange, transcribed rather than measured.** The
robosuite Panda hangs its hand off ``robot0_right_hand`` at zero offset under
``quat="0.707107 0 0 -0.707107"`` -- a quarter turn about -z -- which is exactly
what ``gripper0_right_gripper`` carries in
``assets/embodiments/panda/panda_standalone.xml``, as in the arena it was cut from.
Putting the 2F-85's ``base_mount`` at the same frame therefore stands it where
the Franka hand stood, in the frame the arm's IK and the export's home pose were
both authored against.

**Why that quarter turn is the right one, and not a convention borrowed.** The
axes were read off the built model rather than assumed. At rest the 2F-85's pads
sit at ``x = +/-49.3 mm`` and ``z = +115.7 mm`` in ``base_mount``'s frame: the
hand reaches along its own **+z** and closes along its own **x**. The hand it
replaces declares exactly that pair -- ``approach_axis=(0, 0, 1)``,
``close_axis=(1, 0, 0)`` -- so under a rotation about z alone the approach
direction is untouched and the closing axis lands where the Franka's did. A
mount that got this wrong would compile, render, and close the jaws along an
object instead of across it, which is the failure ``franka_hand.py``'s own
``close_axis`` comment records having already happened once here.

**Where its pads are is a function of how far it is closed, and a graph has to
ask.** Measured on a single-gear meshing scene (2026-09-09), the
fingertips sit past the tool-centre point along the approach axis by 5.9 mm at
full open, 11.5 mm at a 67 mm opening and 15-16 mm closed on a 55 mm gear --
the four-bar swings them down as it closes. ``tip_reach`` below is the
open-hand figure, the one a measurement of the built model reports at the rest
pose; the live one is ``robot.get_gripper``'s ``tip_reach_m``, which the
graph-policy runtime's robot tools read. And the pads are not to
be pushed onto a surface to find out: a vertical load at the fingertips folds
the passive loop (coupler and spring link off their rest, follower to its
limit -- the adaptive mode, held by a 0.5 N.m/rad spring) and the tool centre
travels another 19 mm before anything stalls.

**The gains are per radian and they are the hand's own.** Closed on a 30 g
gear held by its tooth tips: at 20/1 the pads carry it with about 10 N, 0.8 mm
into the tips, and an 80 mm lift loses 0.7 mm; at 200/10 it is 50-60 N and the
lift slips 9 mm; at 1000/30 it is 380-450 N with the gear 20 degrees over and
thrown at the lift; at the 4000/100 a prismatic hand's stanza brought here the
jaws shut in ten steps and threw the gear across the bench.

**What this hand does not inherit is a span the cell can measure.** Five of its
six joints are held by ``<connect>`` equalities rather than by tree structure,
so forward kinematics leaves the pads at the rest pose and no joint sum is a
jaw gap; it has to be measured off the pads. The 85 mm is the published stroke,
and the module it comes from swept it under physics: 84.807 mm of gap at command
zero.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, AssetSpec, GripperSpec

from ..registry import register_gripper
from .robotiq_2f85_menagerie import DRIVER_CLOSED_Q, GRIPPER_SPAN, TCP_OFFSET

__all__ = ["MOUNT_XFORM", "ROBOTIQ_2F85_PANDA"]

MOUNT_XFORM = (0.0, 0.0, 0.0, 0.0, 0.0, -0.7071067811865476, 0.7071067811865476)
"""Where ``base_mount`` sits in ``robot0_right_hand``'s frame: xyz then xyzw.

``gripper0_right_gripper``'s own placement in the export, transcribed: zero
translation under MJCF ``quat="0.707107 0 0 -0.707107"``, which is wxyz and
normalises to a quarter turn about -z. The Franka flange body already carries
the 106.5 mm standoff from ``robot0_link7``, so nothing is added here.
"""

ROBOTIQ_2F85_PANDA = register_gripper(
    GripperSpec(
        name="robotiq_2f85_panda",
        asset=AssetSpec(path="embodiments/robotiq_2f85/2f85.xml"),
        mount_body="robot0_right_hand",
        mount_xform=MOUNT_XFORM,
        # Both drivers, as the Menagerie export declares them -- the two sides
        # are coupled through a tendon and one actuator upstream, and the weld
        # re-expresses that as "command both to the same value".
        finger_names=("right_driver_joint", "left_driver_joint"),
        finger_open_q=(0.0, 0.0),
        finger_closed_q=(DRIVER_CLOSED_Q, DRIVER_CLOSED_Q),
        # Geared: the drivers are revolute and their travel is an angle, so
        # metres-to-command is an approximation rather than an exact inverse.
        linear_travel=False,
        span=GRIPPER_SPAN,
        tcp_body="base_mount",
        tcp_offset=TCP_OFFSET,
        # Measured on the built model, not inherited: the pads separate along
        # base_mount's x and the hand reaches along its z.
        approach_axis=(0.0, 0.0, 1.0),
        close_axis=(1.0, 0.0, 0.0),
        # The pads, not the silicone facings -- the facings are children with no
        # collision geometry, and check_grasp reads contacts.
        finger_bodies=(("left_pad",), ("right_pad",)),
        # Both sides open along their own local +y, so not the mirrored default.
        open_axis_signs=(1.0, 1.0),
        # The envelope, and every number below is a fact about the hardware and
        # its Menagerie export rather than about the arm carrying it, so they
        # are the Menagerie entry's, restated rather than re-measured. Mounting
        # the same file on a different flange moves the hand; it does not change
        # how thick its pads are.
        pad_half_span=0.04897,
        pad_clearance=0.004,
        pad_thickness=0.0375,
        pad_depth=0.0597,
        pad_radius=0.011,
        tip_reach=0.00442,
        housing_half=0.0424,
        housing_rise=0.0597,
        actuation=ActuationSpec(target_ke=20.0, target_kd=1.0),
    )
)
