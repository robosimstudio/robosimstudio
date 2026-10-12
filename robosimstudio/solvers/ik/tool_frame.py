# SPDX-License-Identifier: Apache-2.0
"""Where one world's tool-centre point is, from its published body rows.

Every commanded pose in this stack is stated at the **tool-centre point** in
the **grasp frame**: the arm's ``ee_body`` transform with ``ee_offset``
translated along it and ``ee_offset_rot`` composed on. That is what
:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.ee_pose` returns, what the
single-world servo measures (``CartesianServo._measure``), what the batched IK
rig is built to drive (``link_offset`` / ``link_offset_rotation``), and what the
robot's own observation publishes.

The fleet's motion layer used to read the raw ``ee_body`` row instead, in three
places -- the servo's arrival test, the rate-capped lead law, and the runtime
observation's ``robot_cartesian_pos`` -- and on every hand in the corpus the two
differ by the length of the tool: 134.7 mm and a half-turn on the i2rt YAM,
255 mm on a Robotiq 2F-85. Measured on ``gsl_surface_prep/wipe_train`` at twelve
worlds: the batched IK put the tool-centre point on its target to the
millimetre, the arrival test measured the wrist 134.7 mm away and ran every
Cartesian move out to its budget, and ``robot.get_ee_pose`` reported the same
134.7 mm and 179 degrees of error on a perfectly placed hand. The graph then
"corrected" a pose it had already reached, in every world, and aborted.

The arm handle's ``ee_offset_rot`` docstring records the single-world servo
making the same mistake once; this module is where the fleet stops making it. One function, used by everything that measures
a hand in a wave, so the frame cannot drift between the readers again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.math import quat_multiply, transform_point

__all__ = ["tcp_pose", "tool_offsets"]


def tool_offsets(fleet, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(ee_offset, ee_offset_rot_xyzw)`` for one arm of a fleet.

    Off the template solver, which is what the single-world observation reads
    them from (``iks[name].ee_pose``); the arm handle carries the same pair and
    answers when a scene has arms but no solver for one.
    """
    embodiment = fleet.scene.embodiment
    solver = getattr(embodiment, "iks", {}).get(name) if embodiment is not None else None
    if solver is not None:
        return (
            np.asarray(solver.ee_offset, dtype=np.float64),
            np.asarray(solver.ee_offset_rot, dtype=np.float64),
        )
    handle = fleet.handles.arms.get(name) if getattr(fleet, "handles", None) is not None else None
    if handle is not None:
        return (
            np.asarray(handle.ee_offset, dtype=np.float64),
            np.asarray(handle.ee_offset_rot, dtype=np.float64),
        )
    return np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def tcp_pose(body_rows: Any, ee_body: int, ee_offset: Any, ee_offset_rot: Any) -> tuple[np.ndarray, np.ndarray]:
    """The tool-centre pose ``(position, quat_xyzw)`` from template-indexed body rows.

    ``body_rows`` is one world's ``body_q`` sliced and rebased to template
    indices -- a ``FleetSnapshot.body(world)`` view, or the rows of a
    :class:`~robosimstudio.core.worlds.WorldMap` ``body_view``.
    """
    xform = np.asarray(body_rows[int(ee_body)], dtype=np.float64)
    position = np.asarray(transform_point(xform, np.asarray(ee_offset, dtype=np.float64)), dtype=np.float64)
    quat = np.asarray(quat_multiply(xform[3:7], np.asarray(ee_offset_rot, dtype=np.float64)), dtype=np.float64)
    return position, quat
