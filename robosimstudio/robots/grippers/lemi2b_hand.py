# SPDX-License-Identifier: Apache-2.0
"""The Sunday Robotics Lemi 2B two-finger (index + 3F) four-DOF hand.

Each Lemi 2B hand (`left_hand_chassis`, `right_hand_chassis`) has a fixed thumb
and two 2-DOF articulated fingers:
  - Index finger: proximal `*_PI` (`[0, 1.5708]` rad) and distal `*_DI` (`[0, 1.39626]` rad)
  - Three-finger unit: proximal `*_P3F` (`[0, 1.5708]` rad) and distal `*_D3F` (`[0, 1.39626]` rad)

Because the left and right hands in `lemi2b/robot.xml` use asymmetric joint
prefixes (`LH_*` vs `RH_*`) and link suffixes (`index_distal_2` vs `index_distal`),
both `"lemi2b_left_hand"` and `"lemi2b_right_hand"` are registered here alongside
`"lemi2b_hand"`.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec, GripperSpec

from ..registry import register_gripper

__all__ = [
    "FINGER_CLOSED_Q",
    "FINGER_OPEN_Q",
    "HAND_ACTUATION",
    "LEMI2B_HAND",
    "LEMI2B_LEFT_HAND",
    "LEMI2B_RIGHT_HAND",
]

FINGER_OPEN_Q: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
"""All four finger joints (`PI`, `DI`, `P3F`, `D3F`) at full open [rad]."""

FINGER_CLOSED_Q: tuple[float, float, float, float] = (1.5708, 1.39626, 1.5708, 1.39626)
"""All four finger joints at full close [rad], matching the MJCF joint limits."""

HAND_ACTUATION = ActuationSpec(
    target_ke=100.0,
    target_kd=0.1,
    effort_limit=10.0,
)
"""`sunday/config/common/sim/controller_config.textproto` `STIFF` hand gains."""

LEMI2B_LEFT_HAND = register_gripper(
    GripperSpec(
        name="lemi2b_left_hand",
        finger_names=("LH_PI", "LH_DI", "LH_P3F", "LH_D3F"),
        finger_open_q=FINGER_OPEN_Q,
        finger_closed_q=FINGER_CLOSED_Q,
        linear_travel=False,
        open_width=1.5708,
        span=0.11,
        tcp_body="left_hand_chassis",
        tcp_offset=(0.134252, -0.0493026, -0.0176359),
        approach_axis=(1.0, 0.0, 0.0),
        close_axis=(0.0, -1.0, 0.0),
        finger_bodies=(
            ("index_distal_2", "index_fingertip_2"),
            ("3f_distal_2", "left_3f_fingertip"),
        ),
        actuation=HAND_ACTUATION,
    )
)

LEMI2B_RIGHT_HAND = register_gripper(
    GripperSpec(
        name="lemi2b_right_hand",
        finger_names=("RH_PI", "RH_DI", "RH_P3F", "RH_D3F"),
        finger_open_q=FINGER_OPEN_Q,
        finger_closed_q=FINGER_CLOSED_Q,
        linear_travel=False,
        open_width=1.5708,
        span=0.11,
        tcp_body="right_hand_chassis",
        tcp_offset=(0.134252, 0.0493026, -0.0176359),
        approach_axis=(1.0, 0.0, 0.0),
        close_axis=(0.0, 1.0, 0.0),
        finger_bodies=(
            ("index_distal", "index_fingertip"),
            ("3f_distal", "right_3f_fingertip"),
        ),
        actuation=HAND_ACTUATION,
    )
)

LEMI2B_HAND = register_gripper(
    GripperSpec(
        name="lemi2b_hand",
        finger_names=("LH_PI", "LH_DI", "LH_P3F", "LH_D3F"),
        finger_open_q=FINGER_OPEN_Q,
        finger_closed_q=FINGER_CLOSED_Q,
        linear_travel=False,
        open_width=1.5708,
        span=0.11,
        tcp_body="{arm}_hand_chassis",
        tcp_offset=(0.134252, 0.0, -0.0176359),
        approach_axis=(1.0, 0.0, 0.0),
        close_axis=(0.0, 1.0, 0.0),
        actuation=HAND_ACTUATION,
    )
)
