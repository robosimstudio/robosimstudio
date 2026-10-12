# SPDX-License-Identifier: Apache-2.0
"""The robot library: what an arm is, how it is driven, and how it decides.

Everything about robots that is not about any particular task, gathered from the
four places it used to be scattered over -- ``core/control``, ``toolkit/embodiments``,
``toolkit/grippers`` and the IK, motion and grasp modules that had ended up
inside the graph-policy runtime because that is where they were first needed.

Seven folders, split by what kind of knowledge each holds (this said "four" while
listing six of them, which is the sort of number a roster should not carry):

* :mod:`~robosimstudio.robots.embodiments` and :mod:`~robosimstudio.robots.grippers`
  -- the robots and hands themselves, one module each, registering a spec that
  states measured geometry. **Descriptions, not behaviour.**
* :mod:`~robosimstudio.solvers.control` -- the action spaces a policy writes into,
  and the drivers that turn intent into per-substep targets.
* :mod:`~robosimstudio.solvers.ik` -- inverse kinematics, one module per backend.
* :mod:`~robosimstudio.solvers.motion` -- gripper commands, operational-space
  control, the Cartesian servo, and collision-aware planning.
* :mod:`~robosimstudio.solvers.grasp` -- where to put the jaws: an oriented-box
  heuristic and an antipodal sampler over the real mesh.
* :mod:`~robosimstudio.robots.teleop` -- **physical** input: a leader arm's servo
  ticks mapped to the same joint vector a policy would have written. One module
  per hardware bus, each importing its driver library lazily, so a checkout
  without the hardware pays nothing.

**It reads downward only.** The robot library imports ``specs`` and ``core`` and
nothing above: it does not know what a task is, what a scene contains, or that a
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
