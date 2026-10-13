# SPDX-License-Identifier: Apache-2.0
"""The Franka hand, as the robosuite tool-sort export names it.

A mirrored prismatic pair: two finger joints travelling ``+w`` and ``-w``, which
is the convention :class:`~robosimstudio.solvers.Gripper` defaults to, so
this hand states no open/closed coordinates of its own.

Keyed to the export rather than to the hardware. The joint and body labels here
-- ``gripper0_finger_joint1``, ``gripper0_leftfinger`` -- are robosuite's
prefixing, and a Franka hand out of a different exporter is a different entry
with the same geometry. The geometry is what this file is for; the names are
what tie it to one import.

It has no file of its own. It arrives welded into the arm that carries it:
``assets/embodiments/panda/panda_standalone.xml``, which the library ships, and
the tool-sorting arenas the benchmark's collection stages, where the same
labels name the same bodies.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, GripperSpec

from ..registry import register_gripper

__all__ = ["FRANKA_HAND_ROBOSUITE", "GRIPPER_OPEN"]

GRIPPER_OPEN = 0.04
"""Franka finger travel [m]: joint1 opens positive, joint2 negative."""

FRANKA_HAND_ROBOSUITE = register_gripper(
    GripperSpec(
        name="franka_hand_robosuite",
        finger_names=("gripper0_finger_joint1", "gripper0_finger_joint2"),
        open_width=GRIPPER_OPEN,
        # No span stated: a mirrored prismatic pair's fingertip separation
        # really is twice its travel, so the derivation is exact here and
        # restating 0.08 would only be a second place for it to drift.
        #
        # gripper0_eef is robosuite's TCP -- the point between the fingertips,
        # which is what a grasp pose is naturally expressed about and what the
        # IK objectives and the teleop gizmo key off. So the offset is zero:
        # the body *is* the point. Worth saying out loud, because it is why the
        # tool-sort graph's `panda_hand_to_tcp` constant subtracts a length that
        # is not in this chain.
        tcp_body="gripper0_eef",
        approach_axis=(0.0, 0.0, 1.0),
        # Measured off the model, not assumed: driving the fingers from open to
        # shut translates each pad 40 mm along the hand's local **x**, so the
        # default (0, 1, 0) is a right angle out. Nothing noticed while every
        # grasp pose came from the OBB heuristic -- it yaws the home wrist about
        # world +z and never reads these -- but a grasp frame built from them
        # closes the jaws along the object instead of across it, and the hand
        # shuts on air beside what it was aimed at. `TestGripperAxes` gates this.
        close_axis=(1.0, 0.0, 0.0),
        # Both links per side, tip last. robosuite hangs a pad body off each
        # finger -- gripper0_finger_joint1_tip off gripper0_leftfinger -- and
        # that pad is the geometry that touches an object: measured on
        # a tool-sorting scene, every contact of a grasped clamp is on the two tip
        # shapes and none is on the finger links. Naming only the links is
        # therefore a hand whose pads are outside every shape set derived from
        # it -- `sim.check_grasp` and the placement predicates read a held object
        # as untouched, and a closing-force bench reported 0 N on jaws stopped
        # dead against a clamp. Tip last because `_tip_reach` takes the last body
        # on a side as the tip-most one.
        finger_bodies=(
            ("gripper0_leftfinger", "gripper0_finger_joint1_tip"),
            ("gripper0_rightfinger", "gripper0_finger_joint2_tip"),
        ),
        open_axis_signs=(1.0, -1.0),
        # Measured on a tool-sorting scene with a gripper-measurement script: the
        # furthest pad corner sits 9.3 mm past the tool-centre point along the
        # approach axis. Small because gripper0_eef is already the point between
        # the fingertips rather than a flange above them.
        tip_reach=0.0093,
        # Stiffer than the arm so a grasp does not slip open under a carried
        # tool's weight. A property of this hand, which is why it moved here off
        # the embodiment: the day a second arm carries one, it comes along.
        #
        # **There is no force ceiling here, and the export asks for one.**
        # ``assets/embodiments/panda/panda_standalone.xml:151-152`` declares
        # ``forcelimited="true" forcerange="-20 20"`` on both finger actuators,
        # as every tool-sort arena it was cut from does.
        # Newton's MJCF importer reads a ceiling only from a *joint's*
        # ``actuatorfrcrange`` and never from an actuator's ``forcerange``, and
        # these finger joints declare none -- so the fallback stands, and it is
        # ``builder.default_joint_cfg.effort_limit = 1e6``. Read off a built
        # model: ``joint_effort_limit`` is 1e6 at every finger coordinate.
        #
        # What that costs is not an abstraction. ``preset="close"`` commands a
        # gap of exactly zero, so the fingers stall against whatever is between
        # them and the grip force settles at ``ke x standing error`` -- a number
        # nobody chose, set by the object's own width. On this hand that is 6 N
        # at 4 mm and 18 N at 12 mm, applied to a 30 g syringe.
        #
        # Adding ``effort_limit=20.0`` restores the declared value and leaves
        # the hammer-sorting scene at 8/8 while shortening it from 703 steps to 566.
        # Not taken yet: hammer is a single-goal cell at ceiling, so that proves
        # only that nothing broke, and the multi-goal variants where a grip force
        # actually decides the score are unmeasured.
        actuation=ActuationSpec(target_ke=1500.0, target_kd=60.0),
    )
)
