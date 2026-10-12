# SPDX-License-Identifier: Apache-2.0
"""The Franka Panda, as the VAB tool-sort export declares it.

robosuite 1.4.0's Panda, cut out of the merged scene MJCF that benchmark
compiles so that the arm can stand in any cell. Two files, both shipped with
the library under ``assets/embodiments/panda/`` beside the 61 meshes they name:
``panda_standalone.xml`` carries the Franka hand, and
``panda_nohand_standalone.xml`` ends at the flange. They are byte copies of the
files the benchmark's collection keeps beside its tool-sort export -- vendored
because a robot the library registers has to build where that collection is
not installed. The ``README.md`` in that directory records the source and the
licence.

Nothing was re-authored on the way, which is what keeps the benchmark's arm the
benchmark's own: the bodies, joints, limits and ``robot0_`` / ``gripper0_``
labels are the export's. So everything below is *indexing* -- which joints drive
the arm, where the tool-centre point is, which bodies are the fingers.

The gains are the one thing the export cannot supply. robosuite drives its Panda
with torque motors, so it ships no position-control gains for Newton to import;
these match Newton's own Franka example, which gives a stiff-but-stable arm under
the MuJoCo solver. Force limits are left as the MJCF declared them (±80 N·m on
the first five joints, ±12 on the wrist), which keeps the arm's authority the
same as the benchmark's.
"""

from __future__ import annotations

import dataclasses

from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec

from ..registry import register_embodiment

__all__ = ["PANDA", "PANDA_HOME_Q", "PANDA_ROBOTIQ", "PANDA_ROBOTIQ_ROLL"]

PANDA_HOME_Q = (
    0.0,
    0.19634954084936207,
    0.0,
    -2.617993877991494,
    0.0,
    2.941592653589793,
    0.7853981633974483,
)
"""``robot_init_qpos`` from the export's metadata, i.e. the benchmark's own home."""

PANDA = register_embodiment(
    EmbodimentSpec(
        name="panda",
        # The arm extracted from the robosuite export, at its own base origin
        # with the mount0_* pedestal dropped (the README beside the file says
        # how it was cut). This is *not* what the tool-sort suite's native
        # pairing builds: that binding says scene_resident, so the Panda already
        # inside the arena is indexed and this file is never opened -- which
        # keeps that suite the benchmark's own export rather than a
        # re-authoring. The asset exists so the same arm can stand in another
        # cell, and it is vendored so it can do that without the collection.
        asset=AssetSpec(path="embodiments/panda/panda_standalone.xml"),
        # Base 132 mm above the workbench top, 0.6 m behind its centre and
        # 115 mm to one side: the pedestal's height, restated as a mount.
        base_offset=(-0.6, -0.115, 0.132, 0.0, 0.0, 0.0, 1.0),
        collapse_fixed_joints=False,
        hull_colliders=True,
        # The only robot here with a planner model, because it is the only one
        # whose URDF has been measured against the MJCF physics actually steps.
        # cuRobo's own franka.yml carries the collision spheres; the URDF is
        # ours, and `assets/PROVENANCE.md` records why the substitution is safe
        # -- 0.000000 mm and 0.000000 deg of FK disagreement across 64 sampled
        # configurations at every one of the eleven collision links, with the
        # hand and finger frames differing by a constant that entry states.
        curobo_config="franka.yml",
        curobo_urdf="embodiments/panda/panda_kinematics.urdf",
        urdf="embodiments/panda/panda_kinematics.urdf",
        # A flange-grasped 150 mm syringe has a long moment arm.  Stream
        # cuRobo trajectories at half speed so a large wrist reorientation
        # does not throw the object from the position-controlled fingers.
        curobo_time_scale=2.0,
        curobo_lock_joint="panda_finger_joint1",
        arms=(
            ArmSpec(
                name="right",
                joint_names=tuple(f"robot0_joint{i}" for i in range(1, 8)),
                # The wrist. The hand names gripper0_eef as its own TCP body, so
                # this is only the bolt-on point; nothing reads it today.
                ee_body_name="robot0_link7",
                gripper="franka_hand_robosuite",
                home_q=PANDA_HOME_Q,
            ),
        ),
        # Newton's own Franka example's gains. The hand's are its own, on
        # franka_hand_robosuite -- they are a fact about the gripper, and the
        # day a second arm carries one they come along with it.
        arm_actuation=ActuationSpec(target_ke=650.0, target_kd=100.0),
        ik_iterations=8,
        # Roll FREE, like PANDA_ROBOTIQ below and for the same reason: the
        # variants that stand on this entry were measured with it free. See
        # PANDA_ROBOTIQ_ROLL for what closing it is worth and what it costs.
    )
)

PANDA_ROBOTIQ = register_embodiment(
    EmbodimentSpec(
        name="panda_robotiq",
        # The same arm as PANDA above with its Franka hand cut out, so a hand
        # can be welded on instead. Both the merged scene and
        # ``panda_standalone.xml`` carry ``gripper0_*`` welded in, which is why
        # this third file exists rather than a flag: there was no hand-free
        # Panda in the tree to bolt anything to. Derived from the standalone by
        # dropping the ``gripper0_right_gripper`` subtree and every asset,
        # actuator and exclusion naming it -- 11 bodies and the 7 arm
        # actuators, ending at ``robot0_right_hand``.
        asset=AssetSpec(path="embodiments/panda/panda_nohand_standalone.xml"),
        # PANDA's, unchanged: the same arm at the same pedestal height, so a
        # scene staged against either stands the base in the same place.
        base_offset=(-0.6, -0.115, 0.132, 0.0, 0.0, 0.0, 1.0),
        collapse_fixed_joints=False,
        hull_colliders=True,
        curobo_config="franka.yml",
        curobo_urdf="embodiments/panda/panda_kinematics.urdf",
        urdf="embodiments/panda/panda_kinematics.urdf",
        # PANDA's own tuning, carried over: a flange-grasped 150 mm syringe
        # has a long moment arm regardless of which hand is bolted on or how
        # the pairing is wired, so the same half-speed reorientation applies
        # here too.
        curobo_time_scale=2.0,
        curobo_lock_joint="panda_finger_joint1",
        arms=(
            ArmSpec(
                name="right",
                joint_names=tuple(f"robot0_joint{i}" for i in range(1, 8)),
                ee_body_name="robot0_link7",
                # The one line that differs from PANDA. The hand is welded on at
                # build time here rather than indexed out of the scene, which is
                # what makes the pairing a registry entry instead of surgery on
                # every export.
                gripper="robotiq_2f85_panda",
                home_q=PANDA_HOME_Q,
            ),
        ),
        arm_actuation=ActuationSpec(target_ke=650.0, target_kd=100.0),
        ik_iterations=8,
        # Roll FREE here, and honoured by PANDA_ROBOTIQ_ROLL below. See that
        # entry for why the geometry permits closing it, and for the measurement
        # that says this pairing may not be the one that does.
    )
)
"""The tool-sort Panda wearing a 2F-85 instead of its Franka hand.

**The planner model is the arm's, and stays right.** ``franka.yml`` and
``panda_kinematics.urdf`` describe the seven-jointed arm, which this does not
touch -- the entry that PROVENANCE.md justifies is a claim about link frames,
not about what is bolted past the flange. What *does* change is the collision
geometry beyond ``robot0_right_hand``, so any planner asking about the hand's
own spheres is asking about a Franka hand that is no longer there.
"""

PANDA_ROBOTIQ_ROLL = register_embodiment(
    dataclasses.replace(PANDA_ROBOTIQ, name="panda_robotiq_roll", honour_roll=True)
)
"""The same pairing with the wrist roll closed, for a task that commands one.

**The geometry permits it.** The 2F-85's tool centre is (0, 0, 0.1488) in
``base_mount`` and the weld stands it under a pure turn about the flange normal
-- joint 7's axis -- so the offset has no component off that axis and there is
exactly one joint-7 angle for a given top-down pose. The gear-meshing suite needs that:
with the roll free the solver discarded every yaw its graph commanded, and the
recording shows the jaws closing ALONG the chain instead of across it (pad
separation 6-16 degrees off the pinion-to-gear line) with a finger descending
onto the neighbouring gear's teeth.

**It is a second entry rather than a flag on the first because the tool-sorting
suite wears this hand too, and was not measured with the roll closed.** Its
thirty drawn level-1 documents stand on ``panda_robotiq`` through
``vab_industrial_robotiq@v1``, and ``sim.get_object_obb``'s ``grasp_quat`` hands
that graph a cross-the-handle roll far from what a free wrist holds -- the
graph-policy runtime's servo measures about 84 degrees of it on the hammer.
Closed, the arm has to reach that roll or spend each waypoint's whole budget
failing to.

**What was measured (2026-09-10).** Four level-1 tool-sorting documents under ``gt``
with oracle perception, one episode each, on ONE tree with this flag as the only
variable:

    roll free    2 of 4 solved,  26 give_up_sg entries over the four
    roll closed  0 of 4 solved,  70 give_up_sg entries over the four

**The episode outcome is not the evidence; the give-up count is.** Two-of-four
against none-of-four is p about 0.4 -- nothing -- and this task's outcome moves
on its own besides: the same four documents on a different checkout at the SAME
flag value scored 4 of 4, and the suite was recorded
taking 7456, 12518 and 15561 steps on one tree, one seed, one episode index. The
give-up count is a tally over every grasp attempt in those episodes rather than
four Bernoulli trials, and 70 against 26 is the graph failing to reach the roll
it asked for, which is exactly the mechanism this flag turns on.

**So this split is the conservative reading.** ``panda_robotiq``
keeps exactly the roll setting its thirty documents were measured under, which
makes the gear delivery no-worse-than-before for a suite that is not part of it;
the gear-meshing suite, which does need the yaw, asks for it by name. The honest tension
is that ``EmbodimentSpec.honour_roll`` is documented as "a fact about the arm",
and by that reading the flag belongs on both -- the geometry is identical. If it
does, the thing to fix is the tool-sorting grasp roll -- and the evidence for
closing it is good (that field's own note measures a wiring-sort scene going
from 1 of 4 objects placed to 3 of 4). That is a measurement for the tool-sorting
row rather than for the gear delivery, and it is the reason this entry exists
instead of the flag simply moving."""
