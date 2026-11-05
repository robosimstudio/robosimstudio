# SPDX-License-Identifier: Apache-2.0
"""The I2RT YAM bimanual arm, and the arena constants that come with it.

Two arms of six revolute joints and two prismatic fingers each, imported from
the pre-stripped MJCF (see ``assets/PROVENANCE.md``) with its fixed joints
collapsed.

Collapsing is why :attr:`EmbodimentSpec.gravcomp_restate` is on. The MJCF gives
every arm body ``gravcomp="1"``; ``collapse_fixed_joints`` renumbers the bodies
and re-keys only the attributes that *reference* an entity, leaving the
BODY-frequency ``mujoco:gravcomp`` values keyed by pre-collapse indices. Eight
of the sixteen arm bodies then read the gate, camera and table entries, and the
position-servoed arm sags about 7 cm at reach (17 cm in transient) while every
joint target is tracked correctly -- which is the kind of failure that gets
diagnosed as bad IK. ``add_mjcf_collapsed`` fixes it properly by keeping the
remap tables; the restate is belt-and-braces, and it is *correct* here because
after the strip every surviving body is an arm body and every one carries
``gravcomp="1"``.

The table constants are restated rather than imported. They belong to the arena
this arm was authored into, and a benchmark that imports them from a reference
checkout stops being reproducible the moment that checkout moves.
"""

from __future__ import annotations

from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

from ..grippers.yam_linear_arena import YAM_LINEAR_ARENA as _HAND
from ..registry import register_embodiment

__all__ = [
    "ARM_HOME",
    "GRASP_SITE_OFFSET",
    "GRASP_SITE_QUAT_XYZW",
    "TABLE_CENTER",
    "TABLE_HALF",
    "TABLE_TOP_Z",
    "YAM_BIMANUAL",
]

TABLE_TOP_Z = 0.728
"""Height of the arena's table top [m]."""
TABLE_CENTER = (0.6, 0.0)
"""Where the table sits in the arm's frame [m]."""
TABLE_HALF = (0.30, 0.40)
"""Table half-extents in x and y [m]."""

ARM_HOME = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
"""Per-arm home posture [rad], from the scene's own home keyframe."""

# The hand's own, re-exported: these were this module's public names first.
GRASP_SITE_OFFSET = _HAND.tcp_offset
"""The MJCF's grasp site: along the wrist's +z, between the pads [m]."""
GRASP_SITE_QUAT_XYZW = _HAND.tcp_offset_rot
"""A -90 degree twist about z, so the grasp frame matches the site's."""
FINGER_OPEN = _HAND.open_width
"""YAM finger travel at full open [m] -- the joint's own limit.

Was 0.0495, the *actuator's* ``ctrlrange``, which is 2 mm past the joint's stop.
See :mod:`robosimstudio.robots.grippers.yam_linear_arena`.
"""


def _arm(side: str) -> ArmSpec:
    """One YAM arm, its joints selected by their ``left_``/``right_`` prefix.

    The prefix still selects the *arm* joints; the hand names its own two
    explicitly, which is what stopped the undriven one being swept in with them.
    """
    return ArmSpec(
        name=side,
        joint_prefix=f"{side}_",
        # The wrist, stated rather than derived. The derivation ("the last arm
        # joint's child") needs a *built* model, and the USB-C suite's jaw
        # calibration probes the asset in a throwaway builder before there is
        # one -- so a hand that names no body of its own left it with an empty
        # string to look up. Naming it is also just true, and cheap.
        ee_body_name=f"{side}_link_6",
        gripper="yam_linear_arena",
        home_q=ARM_HOME,
    )


YAM_BIMANUAL = register_embodiment(
    EmbodimentSpec(
        name="yam_bimanual",
        # This arm's chain as a URDF, for the IK solver, which checks it against
        # the MJCF the scene steps at construction and refuses drift. Transcribed
        # from yam_bimanual_arms.xml itself: the vendor's yam.urdf lists joint6
        # first and was refused with 230 mm of drift, and yam_i2rt's transcription
        # is a different build of the arm, off by up to 2.8 mm at the tool.
        urdf="embodiments/yam/yam_bimanual_arm.urdf",
        # The arms alone, at their own base origin, stripped of the arena's
        # scenery by a development-tree script (see assets/PROVENANCE.md).
        asset=AssetSpec(path="embodiments/yam/yam_bimanual_arms.xml"),
        # Mounted 0.3475 m behind the work surface's centre and 22 mm *above*
        # it: this arena's arms stand on a gantry that clears its play table.
        # The shoulder separation stays in the asset, being robot geometry.
        base_offset=(-0.3475, 0.0, 0.022, 0.0, 0.0, 0.0, 1.0),
        collapse_fixed_joints=True,
        gravcomp_restate=True,
        enable_self_collisions=False,
        # The two arms collide with each other, as two real arms on one bench
        # do; each arm stays filtered against itself, so a hand shut on nothing
        # still closes to zero instead of stalling on its own pads.
        cross_arm_collisions=True,
        arms=(_arm("left"), _arm("right")),
        # The MJCF declares its own <position> gains per actuator class, so the
        # arm is not retuned here -- only checked. The hand's servo is the hand
        # spec's (yam_linear_arena.CLAMP, the real driver's 50 N a jaw), which
        # the importer's arena spring gives way to. The arm's gains are the
        # arena's, not the hardware's: i2rt's yam_v1.yml runs kp 80/80/80/10/10/10
        # and kd 5/5/5/1.5/1.5/1.5, where the arena has kp 80 kv 6 on joints 1-3
        # and kp 30 kv 2 on the wrist -- kept, because six shipped tasks fail on
        # the hardware's softer wrist (measured in assets/PROVENANCE.md).
        arm_actuation=None,
        gripper_actuation=None,
        ik_iterations=8,
    )
)
