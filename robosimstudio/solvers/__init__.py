# SPDX-License-Identifier: Apache-2.0
"""How a robot is driven and how it decides: the solvers under every arm.

The behaviour half of the robot library. What an arm and a hand *are* -- the
embodiment and gripper specs, measured geometry, the leader-arm buses -- lives in
:mod:`robosimstudio.robots`; this package is what turns an intent into joint
targets and a frame, whatever robot it is handed.

Five folders, split by what kind of question each answers:

* :mod:`~robosimstudio.solvers.control` -- the action spaces a policy writes
  into (joint position, absolute and delta end-effector pose), single-world and
  batched, and the drivers that turn intent into per-substep targets.
* :mod:`~robosimstudio.solvers.ik` -- inverse kinematics, one module per
  backend: Newton's own IK, PyRoKi, damped least squares, and their batched
  fleet forms.
* :mod:`~robosimstudio.solvers.motion` -- what runs over time: gripper
  commands, operational-space control, and the rate a planned path is
  densified at.
* :mod:`~robosimstudio.solvers.grasp` -- where to put the jaws: an oriented-box
  heuristic and an antipodal sampler over the real mesh.
* :mod:`~robosimstudio.solvers.render` -- how a frame is computed: the RTX path
  tracer and Newton's tiled ray tracer, behind ``core/render``'s contract.

**It reads downward only.** This package imports ``specs``, ``core`` and the
registry and nothing above: it does not know what a task is, what a scene contains, or that a
graph policy exists. That is what lets an agent, a routine, a studio panel and a
graph tool all drive the same arm through the same objects.
"""

from __future__ import annotations

from robosimstudio.solvers.control.action import (
    CONTROL_MODES,
    EE_DELTA_CONVENTIONS,
    ActionEEDelta,
    ActionEEPoseAbs,
    ActionJointPosition,
    ActionOsc,
    ActionSpace,
    UnsupportedMode,
    control_mode_class,
    make_action_space,
    make_action_spaces,
    register_control_mode,
    zero_action,
)
from robosimstudio.solvers.control.batched import GripperBank, SegmentDriver, WorldEnables, write_coord_values
from robosimstudio.solvers.grasp.antipodal import AntipodalSampler, GraspCandidate, PadBox, grasp_quat
from robosimstudio.solvers.grasp.obb import ObbGrasp, compute_obb, grasp_from_obb, top_down_grasp_quat
from robosimstudio.solvers.ik.arm_ik import ArmIK
from robosimstudio.solvers.ik.batched import BatchedArmIK, RobotIKRig
from robosimstudio.solvers.motion.gripper import Gripper
from robosimstudio.solvers.motion.osc import (
    OSC_OUTPUT_MAX,
    OscController,
    OscGains,
    OscSetpoint,
    OscWaypoints,
    Waypoint,
    osc_action,
)

__all__ = [
    "CONTROL_MODES",
    "EE_DELTA_CONVENTIONS",
    "OSC_OUTPUT_MAX",
    "ActionEEDelta",
    "ActionEEPoseAbs",
    "ActionJointPosition",
    "ActionOsc",
    "ActionSpace",
    "AntipodalSampler",
    "ArmIK",
    "BatchedArmIK",
    "GraspCandidate",
    "Gripper",
    "GripperBank",
    "ObbGrasp",
    "OscController",
    "OscGains",
    "OscSetpoint",
    "OscWaypoints",
    "PadBox",
    "RobotIKRig",
    "SegmentDriver",
    "UnsupportedMode",
    "Waypoint",
    "WorldEnables",
    "compute_obb",
    "control_mode_class",
    "grasp_from_obb",
    "grasp_quat",
    "make_action_space",
    "make_action_spaces",
    "osc_action",
    "register_control_mode",
    "top_down_grasp_quat",
    "write_coord_values",
    "zero_action",
]
