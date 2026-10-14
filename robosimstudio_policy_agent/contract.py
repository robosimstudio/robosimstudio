# SPDX-License-Identifier: Apache-2.0
"""The privilege boundary, enforced as an allowlist.

The in-process executor hands a graph ground truth: object poses, grasp boxes,
particle state, the evaluator's verdict. That is the right tool for building a
graph and for generating data. It is the wrong thing to send over the bridge to
a policy that is being *evaluated*, because a policy that can ask where the
hammer is has not been evaluated on finding it.

So the boundary is not "remember not to send that". It is an allowlist: a frame
crossing the bridge may contain a fixed set of keys, and
:func:`assert_non_privileged` refuses anything else. Adding a field to the
observation is therefore a deliberate edit to this file, which is exactly the
amount of friction that decision deserves.

This package ships no bridge sender of its own; the check is the gate one has
to call on every frame before it leaves the process. It accepts the frame the
facade builds as it stands -- ``robot_joint_pos_N``, ``robot_cartesian_pos_N``,
one block per declared camera name, and ``instruction``.

What is allowed, and why each is not privileged:

* ``rgb``, ``depth`` -- what a camera would see. A real rig has these.
* ``intrinsics``, ``pose`` -- calibration. A real rig has these too; a policy
  that needs to unproject cannot without them.
* proprioception -- joint positions and gripper state. A robot knows its own
  configuration.
* ``instruction``, ``timestamp`` -- the task and the clock.

What is not: anything naming an object, a region, a bond, a particle, a
success bit or a reward.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

__all__ = ["ALLOWED_KEYS", "ALLOWED_PATTERNS", "PRIVILEGED_KEYS", "assert_non_privileged"]

ALLOWED_KEYS = frozenset(
    {
        "timestamp",
        "instruction",
        "step",
        # per-channel arm blocks
        "left",
        "right",
        "arms",
        "joint_pos",
        "joint_vel",
        "gripper",
        "gripper_position",
        "arm_valid",
        "gripper_valid",
        "ee_pose",
        # camera blocks
        "cam",
        "cameras",
        "images",
        "rgb",
        "depth",
        "depth_data",
        "intrinsics",
        "pose",
        "pose_mat",
        "name",
        "width",
        "height",
    }
)
"""Every key a frame crossing the bridge may carry, at any nesting depth."""

ALLOWED_PATTERNS = (re.compile(r"robot_joint_pos_\d+"), re.compile(r"robot_cartesian_pos_\d+"))
"""Per-arm proprioception keys, one per runtime ``arm_id``, as the facade names them."""

PRIVILEGED_KEYS = frozenset(
    {
        "body_q",
        "particle_q",
        "objects",
        "object_poses",
        "deformables",
        "regions",
        "bond_active",
        "bond_broken_fraction",
        "success",
        "reward",
        "metrics",
        "seg",
        "handles",
        "settled_body_q",
    }
)
"""Keys that name ground truth. Present for the error message -- the check is
the allowlist, since a new privileged field nobody thought to list here would
otherwise pass."""


def assert_non_privileged(frame: Any, *, path: str = "frame", cameras: Iterable[str] = ()) -> None:
    """Raise if an observation frame carries anything privileged.

    Walks the whole structure, since the offending key is usually nested inside
    a per-camera or per-arm block rather than at the top.

    Args:
        frame: The frame about to be sent.
        path: Where in the structure the walk is, for the error message.
        cameras: The scene's declared camera names. Each may key a camera block
            at the top of the frame, and nowhere else; what is inside the block
            is held to the allowlist like everything else.

    Raises:
        ValueError: Naming the first disallowed key and where it was found.
    """
    if isinstance(frame, dict):
        cameras = frozenset(cameras)
        for key, value in frame.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: non-string key {key!r} in a bridge frame")
            allowed = (
                key in ALLOWED_KEYS
                or (key in cameras and key not in PRIVILEGED_KEYS)
                or any(pattern.fullmatch(key) for pattern in ALLOWED_PATTERNS)
            )
            if not allowed:
                why = " (ground-truth state)" if key in PRIVILEGED_KEYS else ""
                patterns = ", ".join(pattern.pattern for pattern in ALLOWED_PATTERNS)
                named = f", or a declared camera name {sorted(cameras)} at the top" if cameras else ""
                # The usual way a facade camera block fails is a sender that
                # never said which names are cameras.
                hint = (
                    " If this is a camera block, pass cameras=<the scene's camera names>."
                    if path == "frame" and not cameras and key not in PRIVILEGED_KEYS
                    else ""
                )
                raise ValueError(
                    f"{path}.{key}{why} is not on the bridge allowlist. A frame crossing the "
                    f"privilege boundary may only carry {sorted(ALLOWED_KEYS)}, keys matching "
                    f"{patterns}{named}; adding to that list is a deliberate edit to "
                    f"robosimstudio_policy_agent.contract.{hint}"
                )
            assert_non_privileged(value, path=f"{path}.{key}")
    elif isinstance(frame, (list, tuple)):
        for i, item in enumerate(frame):
            assert_non_privileged(item, path=f"{path}[{i}]")
