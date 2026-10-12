# SPDX-License-Identifier: Apache-2.0
"""A solver for bodies that follow their joint targets exactly.

What a robot is from a deformable's point of view when nobody needs its
dynamics: a moving boundary. The solver copies ``control.joint_target_q`` into
``joint_q`` for every coordinate it owns, zeroes the rates, and runs forward
kinematics, so the bodies land where the targets say on every substep. Under a
coupled solver it owns the robot's bodies and joints like any other entry and
is mirrored into the deformable island as proxies, exactly as MuJoCo would be
-- which is what lets an arrangement swap ``backend("mujoco")`` for
``backend("kinematic")`` and change nothing else.
"""

from __future__ import annotations

from typing import Any

import newton
import warp as wp
from newton.solvers import SolverBase

try:  # the coupling mixin is experimental; a Newton without it still lets this step alone
    from newton.solvers.experimental.coupled import CouplingInterface
except Exception:  # pragma: no cover - depends on the newton build

    class CouplingInterface:  # type: ignore[no-redef]
        """Stand-in when Newton ships no coupling interface."""


__all__ = ["KinematicSolver"]


@wp.kernel
def _follow_targets(targets: wp.array[wp.float32], joint_q: wp.array[wp.float32], joint_qd: wp.array[wp.float32]):
    i = wp.tid()
    if i < targets.shape[0]:
        joint_q[i] = targets[i]
    joint_qd[i] = 0.0


class KinematicSolver(SolverBase, CouplingInterface):
    """Bodies placed by forward kinematics from their joint targets.

    Args:
        model: The model, or the coupled solver's view of it.
        follow_targets: Copy ``control.joint_target_q`` into ``joint_q`` before
            the FK. Off, the joints stay where whatever else wrote them left
            them and only the bodies are updated.
    """

    def __init__(self, model: Any, *, follow_targets: bool = True) -> None:
        super().__init__(model)
        self.follow_targets = bool(follow_targets)

    def step(self, state_in: newton.State, state_out: newton.State, control, contacts, dt: float) -> None:
        del contacts, dt
        model = self.model
        if state_out.joint_q is None or model.joint_coord_count == 0:
            return
        if state_out.joint_q is not state_in.joint_q:
            wp.copy(state_out.joint_q, state_in.joint_q)
        if state_out.joint_qd is not state_in.joint_qd and state_in.joint_qd is not None:
            wp.copy(state_out.joint_qd, state_in.joint_qd)
        targets = getattr(control, "joint_target_q", None) if control is not None else None
        if self.follow_targets and targets is not None and targets.shape[0]:
            wp.launch(
                _follow_targets,
                dim=state_out.joint_q.shape[0],
                inputs=[targets, state_out.joint_q, state_out.joint_qd],
                device=model.device,
            )
        newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

    def notify_model_changed(self, flags) -> None:
        del flags

    def update_contacts(self, contacts, state=None) -> None:
        del contacts, state
