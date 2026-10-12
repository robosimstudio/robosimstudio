# SPDX-License-Identifier: Apache-2.0
"""A snapshot: everything a measure may read at one instant, and nothing it may write.

This is a **read-only adapter over the existing** ``WorldState``, not a second
gatherer, and the distinction matters more than it looks. Reading ``body_q`` back
from device is the single largest cost in scoring, and the environment already
pays it exactly once per step and hands the array around. A snapshot that
re-gathered would double that cost and -- worse -- create a second version of the
truth that could disagree with the first.

What it adds is a **contract**:

* **Pure.** Called twice on one snapshot, a measure returns the same answer. No
  stepping, no device writes, no advancing anything.
* **Frozen.** :meth:`freeze` makes the underlying arrays non-writeable, so a
  measure that mutates its input raises instead of silently corrupting the state
  the next measure reads. Cheap, and it turns a class of bug that is nearly
  impossible to find into an exception at the line that caused it.
* **Complete.** Poses, velocities, the settled baseline, commanded targets,
  contacts, particle probes and declared published states. Everything a measure
  is allowed to want, in one place, so "what can an atom read?" has an answer
  that fits on a page.

The same object serves checking a goal, verifying an init, and a structured
observation because they are the same measures pointed differently. And because
it is a plain value, it is exactly what a recorded tape
replays: capture these fields per step on the GPU once, and every goal tree
afterwards can be re-scored on a CPU with bit-for-bit equality. That is what
turns a chaotic multi-minute equivalence check into a deterministic one, and it
is the property this class exists to protect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robosimstudio.scene.state import WorldState

__all__ = ["Snapshot"]


@dataclass(frozen=True)
class Snapshot:
    """One instant of the world, as a measure sees it.

    Attributes:
        ctx: The context this wraps. Held rather than copied -- the arrays are
            the environment's own, gathered once for this step.
    """

    ctx: WorldState

    # ------------------------------------------------------------ identity
    @property
    def step(self) -> int:
        """Control steps since reset."""
        return int(self.ctx.step)

    @property
    def sim_time(self) -> float:
        """Simulated seconds since reset [s].

        The clock every temporal combinator counts against, and deliberately not
        a step count: a hold stated in steps means something different the moment
        a task runs at another rate.
        """
        return float(self.ctx.sim_time)

    # --------------------------------------------------------------- state
    @property
    def body_q(self) -> np.ndarray:
        """Every body's pose, ``[body_count, 7]``."""
        return self.ctx.body_q

    @property
    def body_qd(self) -> np.ndarray:
        """Every body's spatial velocity, ``[body_count, 6]``, linear first."""
        return self.ctx.body_qd

    @property
    def settled(self) -> np.ndarray:
        """Poses as of the episode's start -- what "has it moved" is measured against."""
        return self.ctx.settled_body_q

    @property
    def joint_target_q(self) -> np.ndarray:
        """Commanded joint targets.

        Kept because command-versus-measurement is a real signal no single number
        carries: a gripper stalled short of its target is holding something, and
        neither the command nor the measurement alone can say so.
        """
        return self.ctx.joint_target_q

    @property
    def contacts(self) -> Any | None:
        """The contact probe, already updated, or ``None`` when the task has none."""
        return self.ctx.contacts

    @property
    def particles(self) -> dict[str, Any]:
        """Per-deformable particle probes, already updated."""
        return self.ctx.particles

    @property
    def entities(self) -> dict[str, Any]:
        """Every built entity except the robot, by name.

        The view-independent lookup a measure needs: "which entity is called
        this", asked without knowing which of the four views it landed in. It was
        spelled ``extras`` and keyed on two of them, so an entity that started
        offering a handle -- and therefore became an object -- silently stopped
        being found by every reader that looked here.
        """
        return self.ctx.entities

    @property
    def extras(self) -> dict[str, Any]:
        """What the task's own entities published, by owner.

        The channel that becomes typed published states as assets declare them;
        until then it is what it has always been, and reading it here rather than
        from the context directly is what lets that migration happen underneath a
        stable name.
        """
        return self.ctx.extras

    @property
    def handles(self) -> Any:
        """Where everything is -- objects, regions, arms, deformables."""
        return self.ctx.handles

    # ------------------------------------------------------------ contract
    def freeze(self) -> Snapshot:
        """Make the arrays non-writeable, and return self.

        Called by the test fixtures and by ``describe`` so a measure that mutates
        its input fails where it did it. Not called on the hot path: the
        environment owns these arrays and re-uses them across steps, so flipping
        the flag every step would be work for a guarantee the tests already
        establish.
        """
        rigid = (self.ctx.body_q, self.ctx.body_qd, self.ctx.settled_body_q, self.ctx.joint_target_q)
        # A deformable measure is handed the probe's arrays, never ``body_q`` --
        # so freezing only the rigid four left every cloth, sponge and
        # centerline handler free to write through its input. The environment
        # re-uses these buffers across steps, which is what makes a mutation
        # here outlive the measure that made it.
        probes = tuple(
            array
            for probe in (self.ctx.particles or {}).values()
            for array in (getattr(probe, "positions", None), getattr(probe, "velocities", None))
        )
        for array in rigid + probes:
            if isinstance(array, np.ndarray) and array.base is None:
                array.flags.writeable = False
        return self
