# SPDX-License-Identifier: Apache-2.0
"""The Robotiq 2F-85 as MuJoCo Menagerie ships it -- and the first hand that moves.

The same physical gripper as :mod:`.robotiq_2f85`, and a different entry, because
the labels a spec indexes by belong to whoever exported the model. That one is
the hand welded into an Isaac grocery stage, indexed against ``panda_link7``;
this one is ``assets/robotiq_2f85/2f85.xml``, which carries its own bodies, its
own four-bar linkage and its own actuator -- and, being a file, can be bolted
onto a wrist that has never seen it.

**That is what makes it the first *attached* hand in this repo.** This one names
an :attr:`~GripperSpec.asset`, a :attr:`~GripperSpec.mount_body` and a
:attr:`~GripperSpec.mount_xform`, and ``EntityEmbodiment`` welds it on at build
time through Newton's ``add_mjcf(parent_body=...)``.

**The mount is the model's own ``attachment_site``, not a measurement.** The
UR5e declares ``<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>`` on
``wrist_3_link`` -- that is where Universal Robots' tool flange is, and Newton's
``test_robot_composer`` bolts this very hand to that very frame. Restating the
site's numbers here rather than reading the site keeps the transform legible next
to the hand it positions; ``tests`` assert the composed result rather than the
restatement.

**The command-to-gap curve, swept, so nobody calibrates this hand twice.** Driven
0 to 1 in free air under physics at 21 commands, settling each point -- forward
kinematics cannot take this measurement, because the couplers and followers are
carried by ``<connect>`` equalities rather than by tree structure and FK leaves
the pads at the rest pose. "gap" is the narrowest distance between the two pad
colliders across the closing axis; "origins" is what the pad *body* origins read,
kept because it is the number the Isaac vendoring's docstring mistook for a gap:

    command   origins     gap        width()
    0.0       98.007 mm   84.807 mm  84.309 mm
    0.2       82.977      69.280     67.350
    0.4       66.440      52.272     50.391
    0.6       48.804      34.173     33.432
    0.8       30.504      15.393     16.473
    1.0       12.521      -3.094      0.000

Measured on ``astro/wiring_sort``; ``astro/basket_pack`` gives the same origins
column to 0.01 mm, independently. The 84.807 mm at full open is the published
85 mm stroke to 0.2 mm, which is what says the corner measurement is reading the
right faces. The gap goes *negative* at the very end because the two pads have
no collision pair with each other, so the servo drives them 3 mm into overlap at
their joint limit -- real, and not a width any object can occupy.

So ``Gripper.width()``'s default law -- the span interpolated over the driven
travel -- is right to 2.63 mm worst and 1.55 mm rms across the usable stroke.
This hand is geared, and its curve does bow, but not by enough to be worth a
:meth:`~robosimstudio.solvers.Gripper.calibrate` fit: an affine fit anchored
at the measured open gap only reaches 2.25 mm worst. ``robot.set_grip``'s default
squeeze is 4 mm, so the residual is inside the number it is compared against.
Anything needing better than 2 mm out of this hand wants a table, and this sweep
is where one starts.

**What is deliberately not named: the body called ``base``.** Both this hand and
the UR5e declare one, and :func:`~robosimstudio.scene.entity.label_index`
refuses an ambiguous match rather than taking the first -- which is the behaviour
that keeps a wrong index from surfacing later as an arm moving the wrong axis. So
the tool-centre point is stated in ``base_mount``'s frame, which is unique, and
the pads are named by their own unique labels. A hand meant to travel cannot
assume its labels are the only ones in the model.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, AssetSpec, GripperSpec

from ..registry import register_gripper

__all__ = ["DRIVER_CLOSED_Q", "GRIPPER_SPAN", "MOUNT_XFORM", "ROBOTIQ_2F85_MENAGERIE", "TCP_OFFSET"]

DRIVER_CLOSED_Q = 0.8
"""``*_driver_joint`` at full close [rad] -- the ``driver`` class's own limit."""

GRIPPER_SPAN = 0.085
"""Fingertip separation at full open [m]: the 2F-85's published stroke.

The same number the Isaac entry states, and for the same reason -- it is the gap
an object has to pass through, which is what every grasp-fit test measures
against, and it is a property of the hardware rather than of an exporter.
"""

MOUNT_XFORM = (0.0, 0.1, 0.0, -0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
"""Where ``base_mount`` sits in ``wrist_3_link``'s frame: xyz then xyzw.

``ur5e.xml``'s ``attachment_site``, transcribed. Its MJCF ``quat="-1 1 0 0"`` is
wxyz and normalises to a quarter turn about -x; the same rotation appears in
Newton's composer test as ``quat_from_axis_angle(vec3(1, 0, 0), -pi / 2)``.
"""

TCP_OFFSET = (0.0, 0.0, 0.1488)
"""Tool-centre point in ``base_mount``'s frame [m].

Derived from the model rather than guessed: the ``pinch`` site sits 0.145 m along
``base``'s +z, and ``base`` sits 0.0038 m along ``base_mount``'s +z under a
quarter turn *about z*, which leaves the z axis alone. So the two offsets simply
add. Measuring a built model (driving the hand and reading the pads) is what
confirms it, and what would catch the day the linkage's rest pose moves the pinch point.
"""

ROBOTIQ_2F85_MENAGERIE = register_gripper(
    GripperSpec(
        name="robotiq_2f85_menagerie",
        asset=AssetSpec(path="embodiments/robotiq_2f85/2f85.xml"),
        mount_body="wrist_3_link",
        mount_xform=MOUNT_XFORM,
        # *Both* drivers, unlike the Isaac entry's single ``finger_joint``. This
        # model closes its two sides with two driver joints coupled through a
        # tendon and one actuator; the weld switches that actuator to a joint
        # target, so the coupling has to be re-expressed as "command both to the
        # same value" -- which is exactly what ``compose_yam_i2rt`` does with the
        # i2rt hand's equality, and what Newton's own composer test does here.
        finger_names=("right_driver_joint", "left_driver_joint"),
        finger_open_q=(0.0, 0.0),
        finger_closed_q=(DRIVER_CLOSED_Q, DRIVER_CLOSED_Q),
        # The couplers, spring links and followers are carried by <connect>
        # equalities closing two four-bar loops. Not listed at all: naming them
        # as driven would put undriven DOFs into the group apply_actuation
        # asserts over, and naming them as followers would claim a mimic
        # relationship the solver does not have.
        #
        # Geared, because the drivers are revolute and their travel is an angle.
        # Swept, the gap that angle produces is affine to within a 2.6 mm bow
        # (module docstring), so metres-to-command is a good approximation here
        # rather than an exact inverse.
        linear_travel=False,
        span=GRIPPER_SPAN,
        tcp_body="base_mount",
        tcp_offset=TCP_OFFSET,
        # Measured, and the default would have been a right angle out. The pads
        # separate along ``base``'s y, and ``base`` sits under a quarter turn
        # about z from ``base_mount`` -- which is the frame the TCP is stated in,
        # so the axis a caller sees is x. Measured on a physics close: pad
        # centres 0.09793 m apart at full open, 0.01252 m at full closed, moving
        # along tool -x. The 0.0854 m of travel is the published 0.085 stroke to
        # within 0.4 mm, which is the check that the weld drives the real linkage
        # rather than winding it up.
        close_axis=(1.0, 0.0, 0.0),
        # The pads, not the silicone facings: the facings are children with no
        # collision geometry of their own, and check_grasp reads contacts.
        finger_bodies=(("left_pad",), ("right_pad",)),
        # Both sides open along their *own* local +y, so this is not the mirrored
        # default. The same shape as the i2rt hand, whose fingers also run on one
        # sign, and measured the same way: 73.1 mm of travel each, both positive.
        open_axis_signs=(1.0, 1.0),
        # The envelope, measured on a tool-sorting scene with the UR5e from the pads' and
        # the housing's shape corners in the tool frame -- see PROVENANCE.md.
        # ``housing_half`` lands at 0.0424 against the Isaac entry's 0.0425:
        # two independent exports of one piece of hardware, agreeing to 0.1 mm,
        # which is the best evidence either number is right.
        pad_half_span=0.04897,
        pad_clearance=0.004,
        pad_thickness=0.0375,
        pad_depth=0.0597,
        pad_radius=0.011,
        tip_reach=0.00442,
        housing_half=0.0424,
        housing_rise=0.0597,
        # Newton's composer test's numbers for this hand driven by joint target
        # rather than by its own tendon actuator. The armature stays the model's
        # own 0.005, which is in the `driver` class.
        actuation=ActuationSpec(target_ke=20.0, target_kd=1.0),
    )
)
