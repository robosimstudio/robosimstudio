# SPDX-License-Identifier: Apache-2.0
"""One arm's identity for a solver: its coordinates, limits, tool frame and forward kinematics.

Read off the scene once and handed to whichever solver family is asked for, so
a solver is constructed from *facts about the arm* rather than from an
environment: ``ArmKinematics.of(arm)`` takes what :class:`~robosimstudio.solvers.ik.arm_ik.ArmIK`
already knows, and a fleet hands in its own batched forward kinematics so the
same solver answers a wave.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ArmKinematics"]

FK = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""``arm_q [W, n] -> (position [W, 3], quat_xyzw [W, 4])``: the tool frame of every row."""


@dataclass
class ArmKinematics:
    """What a solver needs to know about one arm, and nothing about the scene around it.

    Attributes:
        arm_coords: ``joint_q`` indices of the arm's actuated joints, in chain order.
        arm_dofs: ``joint_qd`` indices of the same.
        ee_body: The tool body's index.
        ee_offset: Tool-centre point in that body's frame [m].
        ee_offset_rot: Rotation from that body's frame to the grasp frame, xyzw
            -- the frame whose +z is the approach direction.
        lower, upper: The arm's joint limits ``[n]``.
        home_q: The arm's coordinates at its home posture ``[n]``.
        urdf: The arm's URDF, for a solver that needs one, or empty.
        base_xform: The world pose of the arm's base body ``(x, y, z, qx, qy, qz, qw)``
            -- the URDF root's frame -- for a solver that works in it, or ``None``.
        fk: The forward kinematics that answers for this arm's rows.
        template: A full ``joint_q`` the arm's coordinates are written into
            when a solver has to walk the whole model (the serial FK does).
        arm: The :class:`ArmIK` this was read off, when it was, so a solver
            that reuses its device paths can.
    """

    arm_coords: list[int]
    arm_dofs: list[int]
    ee_body: int
    ee_offset: np.ndarray
    ee_offset_rot: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    home_q: np.ndarray
    urdf: str = ""
    base_xform: np.ndarray | None = None
    fk: FK | None = None
    template: np.ndarray | None = None
    arm: Any = field(default=None, repr=False)

    @property
    def dof(self) -> int:
        return len(self.arm_coords)

    @classmethod
    def of(
        cls,
        arm: Any,
        *,
        template: np.ndarray | None = None,
        urdf: str = "",
        fk: FK | None = None,
        home_q: np.ndarray | None = None,
        base_xform: np.ndarray | None = None,
    ) -> ArmKinematics:
        """The kinematics of an :class:`~robosimstudio.solvers.ik.arm_ik.ArmIK`.

        Args:
            arm: The solver the scene built for this arm.
            template: The full joint vector the rest of the model holds while
                this arm is solved; the arm's home posture's when not given.
            urdf: The arm's URDF path, from its embodiment.
            fk: A batched forward kinematics to answer with instead of the
                arm's own -- a fleet's, over its replicated model.
            home_q: The arm's home coordinates, from its handle; read off the
                template, or zeros, when not given.
            base_xform: The world pose of the arm's base body, when the caller knows it.
        """
        model = arm.model
        lower = np.asarray(model.joint_limit_lower.numpy(), dtype=np.float64)[list(arm.arm_dofs)]
        upper = np.asarray(model.joint_limit_upper.numpy(), dtype=np.float64)[list(arm.arm_dofs)]
        home = home_q if home_q is not None else getattr(arm, "home_q", None)
        if home is None or len(home) != len(arm.arm_coords):
            home = (
                np.zeros(len(arm.arm_coords))
                if template is None
                else np.asarray(template, dtype=np.float64)[list(arm.arm_coords)]
            )
        return cls(
            arm_coords=[int(c) for c in arm.arm_coords],
            arm_dofs=[int(d) for d in arm.arm_dofs],
            ee_body=int(arm.ee_body),
            ee_offset=np.asarray(arm.ee_offset, dtype=np.float64),
            ee_offset_rot=np.asarray(arm.ee_offset_rot, dtype=np.float64),
            lower=lower,
            upper=upper,
            home_q=np.asarray(home, dtype=np.float64),
            urdf=str(urdf),
            base_xform=None if base_xform is None else np.asarray(base_xform, dtype=np.float64).reshape(7),
            fk=fk,
            template=None if template is None else np.asarray(template, dtype=np.float64).copy(),
            arm=arm,
        )

    def full(self, arm_q: np.ndarray, template: np.ndarray | None = None) -> np.ndarray:
        """A whole-model joint vector with ``arm_q`` written into this arm's coordinates."""
        base = template if template is not None else self.template
        if base is None:
            raise ValueError("no template joint vector: the rest of the model has to be something")
        full = np.asarray(base, dtype=np.float64).copy()
        full[self.arm_coords] = np.asarray(arm_q, dtype=np.float64)
        return full

    def poses(self, arm_q: np.ndarray, template: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The tool frame of every row of ``arm_q [W, n]``: ``(position [W, 3], quat_xyzw [W, 4])``.

        The batched ``fk`` when one was given; else the arm's own, one row at a
        time, on whichever path it is currently on (device, or the host chain
        inside an :meth:`ArmIK.host_fk` scope).
        """
        rows = np.asarray(arm_q, dtype=np.float64).reshape(-1, self.dof)
        if self.fk is not None:
            pos, quat = self.fk(rows)
            return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)
        if self.arm is None:
            raise ValueError("no forward kinematics: give ArmKinematics an fk or an arm")
        positions = np.zeros((rows.shape[0], 3))
        quats = np.zeros((rows.shape[0], 4))
        for w, row in enumerate(rows):
            pos, quat = self.arm.fk_ee(self.full(row, template).astype(np.float32))
            positions[w], quats[w] = pos, quat
        return positions, quats
