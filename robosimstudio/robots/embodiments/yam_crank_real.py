# SPDX-License-Identifier: Apache-2.0
"""Simulated twin of the real bimanual YAM rig: i2rt YAM v1 arms with crank_4310 grippers.

Built by flip-benchmark's ``robot/build/build_yam_crank.py`` from the pinned i2rt (the model the real driver solves
on) and the rig's solved calibration (yam-bimanual-calib): the bases stand where the calibration puts
them, 0.40 m apart (the arena YAM's are 0.62 m apart). The arm servos are i2rt's own gains
(kp 80/80/80/10/10/10, kd 5/5/5/1.5/1.5/1.5), so the wrist tracks as softly as on the real arm.

The asset's world frame is the rig frame: origin midway between the two base origins, x forward, z up.
``base_offset`` puts that origin 0.38 m behind the work surface's centre, which is where the overhead
camera's optical axis meets the table on the real rig. That distance is a guess; measure the mat.
"""

from __future__ import annotations

from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

from ..grippers.crank_4310 import CRANK_4310
from ..registry import register_embodiment
from .yam_look import YAM_BLACK

__all__ = ["ARM_HOME", "MAT_AHEAD", "YAM_CRANK_REAL"]

MAT_AHEAD = 0.38
"""[m] Work-surface centre ahead of the base midpoint."""

ARM_HOME = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
"""Per-arm home posture [rad]: the arena YAM's, which the i2rt arm shares joint for joint."""


def _arm(side: str) -> ArmSpec:
    return ArmSpec(
        name=side,
        joint_names=tuple(f"{side}_joint{i}" for i in range(1, 7)),
        ee_body_name=f"{side}_gripper",
        gripper="crank_4310",
        home_q=ARM_HOME,
    )


# Registered only once robot/build/build_yam_crank.py has built the asset.
YAM_CRANK_REAL = None if CRANK_4310 is None else register_embodiment(
    EmbodimentSpec(
        name="yam_crank_real",
        urdf="embodiments/yam_crank_real/yam_crank_arm.urdf",
        asset=AssetSpec(path="embodiments/yam_crank_real/yam_crank_real_arms.xml"),
        base_offset=(-MAT_AHEAD, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        collapse_fixed_joints=False,
        color=YAM_BLACK,
        enable_self_collisions=False,
        decompose_colliders=("tip_left", "tip_right"),
        decompose_threshold=0.06,
        show_collision_shapes=True,
        arms=(_arm("left"), _arm("right")),
        arm_actuation=None,
        gripper_actuation=None,
        ik_iterations=8,
    )
)
