# SPDX-License-Identifier: Apache-2.0
"""Sunday Robotics Lemi 2B (`lemi2b`) bimanual mobile manipulator."""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec

from ..registry import register_embodiment

__all__ = [
    "CANONICAL_JOINT_NAMES",
    "LEMI2B",
    "LEMI2B_AUX_HOME_Q",
    "LEMI2B_LEFT_ARM_HOME_Q",
    "LEMI2B_RIGHT_ARM_HOME_Q",
]

CANONICAL_JOINT_NAMES: tuple[str, ...] = (
    "basex",
    "basey",
    "baseth",
    "lift",
    "NY",
    "NP",
    "TP",
    "LJ1",
    "LJ2",
    "LJ3",
    "LJ4",
    "LJ5",
    "LJ6",
    "LJ7",
    "RJ1",
    "RJ2",
    "RJ3",
    "RJ4",
    "RJ5",
    "RJ6",
    "RJ7",
    "LH_PI",
    "LH_DI",
    "LH_P3F",
    "LH_D3F",
    "RH_PI",
    "RH_DI",
    "RH_P3F",
    "RH_D3F",
)
"""Canonical 29-DOF joint order from `sunday/config/common/robot_config.textproto`."""

LEMI2B_AUX_HOME_Q: dict[str, float] = {
    "basex": 0.0,
    "basey": 0.0,
    "baseth": 0.0,
    "lift": 0.85,
    "TP": 0.0,
    "NY": 0.0,
    "NP": 0.0,
    "head_camera_dof": 0.0,
}
"""Nominal base, lift, torso, and neck joint positions from `sunday/config/lemi2b/robot_config.textproto`."""

LEMI2B_LEFT_ARM_HOME_Q: tuple[float, ...] = (
    0.174533,
    -0.523599,
    0.4,
    0.523599,
    -1.169371,
    -0.261799,
    0.349066,
)

LEMI2B_RIGHT_ARM_HOME_Q: tuple[float, ...] = (
    -0.174533,
    -0.523599,
    -0.4,
    0.523599,
    1.169371,
    0.261799,
    -0.349066,
)

LEMI2B_ARM_ACTUATION = ActuationSpec(
    target_ke=750.0,
    target_kd=35.0,
    target_ke_per_dof=(750.0, 750.0, 750.0, 750.0, 100.0, 100.0, 80.0),
    target_kd_per_dof=(35.0, 35.0, 35.0, 35.0, 1.2, 1.2, 0.8),
    effort_limit=100.0,
)
"""`sunday/config/common/sim/controller_config.textproto` `STIFF` arm gains."""

LEMI2B = register_embodiment(
    EmbodimentSpec(
        name="lemi2b",
        asset=AssetSpec(path="embodiments/lemi2b/robot.xml"),
        collapse_fixed_joints=False,
        gravcomp=True,
        enable_self_collisions=False,
        cross_arm_collisions=True,
        reset_joint_q=True,
        arms=(
            ArmSpec(
                name="left",
                joint_names=("LJ1", "LJ2", "LJ3", "LJ4", "LJ5", "LJ6", "LJ7"),
                ee_body_name="left_hand_chassis",
                base_body_name="torso",
                gripper="lemi2b_left_hand",
                home_q=LEMI2B_LEFT_ARM_HOME_Q,
                home_gripper_q=(0.0, 0.0, 0.0, 0.0),
            ),
            ArmSpec(
                name="right",
                joint_names=("RJ1", "RJ2", "RJ3", "RJ4", "RJ5", "RJ6", "RJ7"),
                ee_body_name="right_hand_chassis",
                base_body_name="torso",
                gripper="lemi2b_right_hand",
                home_q=LEMI2B_RIGHT_ARM_HOME_Q,
                home_gripper_q=(0.0, 0.0, 0.0, 0.0),
            ),
        ),
        gripper="lemi2b_hand",
        arm_actuation=LEMI2B_ARM_ACTUATION,
        gripper_actuation=ActuationSpec(target_ke=100.0, target_kd=0.1, effort_limit=10.0),
        aux_home_q=tuple(LEMI2B_AUX_HOME_Q.items()),
        urdf="embodiments/lemi2b/robot.urdf",
    )
)
