# SPDX-License-Identifier: Apache-2.0
"""The substep loop: one class, one program, capturable.

:class:`ScenePipeline` is what every caller holds: the two states, the control,
the contacts, the clock, the graph and its counters. It is a
:class:`~robosimstudio.core.schedule.ScheduleExecutor` bound to a *program* --
whatever composed the physics hands it: one solver (bare or coupled), the main
collision pipeline and its cadence, the schedule entities contributed -- and it
adds the one thing the executor does not know about: the batched
:class:`~robosimstudio.core.readback.Readback` that makes every host-side reader
of a step pay for one copy rather than fifteen. The program is a protocol here:
``solver``, ``collision``, ``contacts``, ``collide_every``, ``schedule``,
``hooks``, ``coupled``, ``reset(state, flags=)`` -- and ``core`` names nothing
that builds one.

There is no branch here any more. Rigid, coupled and deformable-only scenes all
run the same body -- ``clear, rigid_force, pre_solve, collide, solve, swap,
post_solve, mirror`` -- and what differs is what the composer put in ``solve``.
A MuJoCo-only scene's solver is a bare ``SolverMuJoCo``; a towel
beside an arm is a ``SolverCoupledProxy`` over two islands. The pipeline steps
either with the same line.

Everything a stage does must be a device kernel gated by a *device-side* flag
and launched unconditionally, so the whole body is capturable into a CUDA graph;
see :mod:`~robosimstudio.core.schedule` for the lifetime rules around
:meth:`invalidate_graph`.
"""

from __future__ import annotations

from typing import Any

import newton

from .readback import HostSnapshot, Readback
from .schedule import ScheduleExecutor, SubstepHooks

__all__ = ["ScenePipeline", "SubstepHooks"]


class ScenePipeline(ScheduleExecutor):
    """One steppable scene: the composed program, its states, its graph, its read-back.

    Args:
        model: The finalized model.
        program: What composed the physics: solver, collision, schedule, hooks.
        control_hz: Control rate [Hz]. One :meth:`step` advances 1/control_hz s.
        substeps: Physics substeps per control step. Must be even.
        use_cuda_graph: Whether to attempt capture at all.
        worlds: The world map when replicated, for per-world read-back strides.
        readback: Whether to batch device-to-host reads after each step. Off
            only for a caller that never reads the state back.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        program: Any,
        control_hz: float = 50.0,
        substeps: int = 4,
        use_cuda_graph: bool = True,
        worlds: Any = None,
        readback: bool = True,
    ) -> None:
        super().__init__(
            model,
            program.schedule,
            control_hz=control_hz,
            substeps=substeps,
            use_cuda_graph=use_cuda_graph,
            contacts=program.contacts,
            solver=program.solver,
            collision=program.collision,
            collide_every=program.collide_every,
        )
        self.program = program
        self.hooks: SubstepHooks = program.hooks
        self.worlds = worlds
        strides = (
            {
                "coords_per_world": worlds.coords_per_world,
                "dofs_per_world": worlds.dofs_per_world,
                "particles_per_world": worlds.particles_per_world,
                "bodies_per_world": worlds.bodies_per_world,
            }
            if worlds is not None
            else {}
        )
        self.readback: Readback | None = (
            Readback(model, self.control, num_worlds=(worlds.num_worlds if worlds is not None else 1), **strides)
            if readback
            else None
        )
        self._dirty = True
        """Whether the states were written since the last read-back was issued."""

    # ---------------------------------------------------------------- shape
    @property
    def is_coupled(self) -> bool:
        """Whether more than one solver island was composed."""
        return bool(self.program.coupled)

    @property
    def manifest(self):
        return self.program.manifest

    # ------------------------------------------------------------- read-back
    def mark_dirty(self) -> None:
        """Say the states were written outside :meth:`step`, so the next snapshot re-reads."""
        self._dirty = True

    @property
    def snapshot(self) -> HostSnapshot:
        """This step's host copy of the state, read once and shared by every caller.

        Issued by :meth:`step`; re-issued here when something wrote the states
        directly since (a reset, a teleport) and said so with :meth:`mark_dirty`.
        """
        if self.readback is None:
            raise RuntimeError("this pipeline was built without a read-back")
        if self._dirty or self.readback.step != self.step_count:
            self.readback.enqueue(self.state_0, self.step_count)
            self._dirty = False
        return self.readback.snapshot()

    def reset_finite_flags(self) -> None:
        if self.readback is not None:
            self.readback.reset_flags()

    # -------------------------------------------------------------------- run
    def step(self) -> None:
        super().step()
        if self.readback is not None and not self.settling:
            self.readback.enqueue(self.state_0, self.step_count)
            self._dirty = False
        else:
            # A settle is scene construction and nothing reads it per step; the
            # next :attr:`snapshot` re-issues against wherever it ended up.
            self._dirty = True

    def reset_solvers(self, *, flags: Any = None) -> None:
        """Clear every solver's internal buffers -- warm starts, contact anchors."""
        self.program.reset(self.state_0, flags=flags)
        self.mark_dirty()
