# SPDX-License-Identifier: Apache-2.0
"""Embodiment specs, and the grasp-frame helpers task code asks them for.

Each module registers on import; the registry loads them all.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.solvers import grasp_quat

__all__ = ["top_down_grasp_site_quat", "work_arm_spec", "work_gripper"]


def top_down_grasp_site_quat(hand, yaw_deg: float, *, close_axis=None):
    """A top-down grasp orientation for one hand, in its grasp-*site* frame, xyzw.

    Takes the *hand* rather than the arm, because all three axes it reads are the
    hand's: swap the gripper and the grasp frame moves while the arm does not.
    :func:`work_gripper` is how a suite gets one.

    ``yaw_deg`` is the world heading the jaws close along -- 0 closes across
    world +x. ``close_axis`` overrides the arm's measured closing axis, which
    the cable-routing suite needs: its keyframes were tuned against the
    reference's *pre-measurement* axis and re-deriving them against the
    corrected one closes the jaws along the cable instead of across it.

    The axis arithmetic itself is
    :func:`~robosimstudio.solvers.grasp.antipodal.grasp_quat` -- this is that function with
    the world pair fixed at "straight down, jaws across ``yaw_deg``". The grasp
    sampler asks for arbitrary world pairs through the same door, so a grasp
    frame is built one way in this package rather than two.
    """
    yaw = np.radians(float(yaw_deg))
    return grasp_quat(
        hand.approach_axis,
        hand.close_axis if close_axis is None else close_axis,
        hand.tcp_offset_rot,
        (0.0, 0.0, -1.0),
        (np.cos(yaw), np.sin(yaw), 0.0),
    )


def work_arm_spec(env):
    """The :class:`ArmSpec` of the arm this pairing works with.

    A one-liner with a name, because the alternative -- ``spec.arms[0]`` -- is
    the bug this refactor kept finding: the two YAM cells list their arms in
    opposite orders, so an index names a different hand on each robot.
    """
    spec = env.scene.embodiment.spec
    name = env.handles.work_arm
    return next((a for a in spec.arms if a.name == name), spec.arms[0])


def work_gripper(env):
    """The :class:`~robosimstudio.specs.GripperSpec` on the work arm.

    Resolved and ``{arm}``-templated, straight off the entity that already did
    the lookup -- so a suite composing a grasp frame asks the hand it was handed
    rather than importing one robot's constants.
    """
    return env.scene.embodiment.hands.get(work_arm_spec(env).name)
