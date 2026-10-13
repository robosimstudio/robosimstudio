# SPDX-License-Identifier: Apache-2.0
"""The substep program as data, and the executor that captures it.

A :class:`Schedule` is an ordered list of :class:`Stage` objects grouped by
*phase*, each a device-side callable ``(state, dt) -> None``. One control period
runs the per-control-step ``before`` phase eagerly, then the captured body --
``substeps`` repetitions of the substep phases in order -- then ``after``. The
:class:`ScheduleExecutor` owns the two states, the control, the CUDA graph and
the counters, and knows nothing about which solver is in the ``solve`` stage:
whoever composed the schedule put one there.

The phases, in order, and what belongs in each:

``before``
    Once per control step, **outside** the graph. Work that reallocates or runs
    on the host: a BVH rebuild, an IK solve whose result a kernel then reads.
``clear``
    ``state.clear_forces()``. Built in.
``rigid_force``
    Adds to ``state.body_f`` before the solve: drag wrenches, a grasp latch.
``pre_solve``
    Adds to ``state.particle_f`` before the solve: adhesion bonds, drag springs.
``collide``
    The main collision pipeline. Built in, with a cadence.
``solve``
    The coupled solver (or the bare one). Built in. The executor swaps the two
    states after it.
``post_solve``
    Kinematic writes onto the state the solve just produced: particle pins,
    handle holds. A position written before the solve is undone by it.
``mirror``
    Derived state: ``eval_ik`` mirroring maximal-coordinate islands back into
    ``joint_q``.
``after``
    Once per control step, after the body: the read-back.

**Capturability is the one rule.** Everything a stage does inside the body must
be a device kernel gated by a *device-side* flag and launched unconditionally.
A host-side ``if`` around a launch works eagerly and then silently does nothing
once captured -- the launch is simply not in the graph. The *set* of stages may
still change after capture -- a teleop panel attaching a dragger, a routine
latching onto a cloth -- and a graph that predates the change would ignore the
new launch. :meth:`ScheduleExecutor.invalidate_graph` therefore drops the graph
and schedules a re-capture on the step after next: one eager step for the new
kernel to compile in (a kernel that compiles *inside* a capture fails it), then
a capture of the current stage set.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import newton
import warp as wp

from .builder import assert_even_substeps
from .streams import capturing

__all__ = ["HOOK_PHASES", "PHASES", "SUBSTEP_PHASES", "Schedule", "ScheduleExecutor", "Stage", "SubstepHooks"]

_log = logging.getLogger(__name__)
"""Where a failed capture's traceback goes: the one-line reason is printed, the
stack is there for whoever turns ``logging`` up."""

Hook = Callable[[newton.State, float], None]
"""A stage body: ``(state, dt) -> None``, launching device kernels only."""

_PHASE_TABLE: tuple[tuple[str, bool, bool], ...] = (
    # phase, runs inside a substep, an entity may hook it
    ("before", False, True),
    ("clear", True, False),
    ("rigid_force", True, True),
    ("pre_solve", True, True),
    ("collide", True, False),
    ("solve", True, False),
    ("post_solve", True, True),
    ("mirror", True, True),
    ("after", False, True),
)
"""Every phase, in execution order, with the two facts that separate the three views of it.

One table, because there were three tuples -- nine phases, the seven inside a
substep, the six an entity may append to -- each written out by hand, and an
entity author writing ``contribute_hooks`` had to work out which was which and
why ``clear`` was in one and not another. The columns say why: ``before`` and
``after`` bracket the control step rather than running per substep, and
``clear``, ``collide`` and ``solve`` are the executor's own -- the module
docstring says "Built in" beside each -- so a stage may not be appended there.
"""

PHASES = tuple(name for name, _, _ in _PHASE_TABLE)
"""Every phase, in execution order."""

SUBSTEP_PHASES = tuple(name for name, in_substep, _ in _PHASE_TABLE if in_substep)
"""The phases inside one substep, in order. ``solve`` is followed by the state swap."""

HOOK_PHASES = tuple(name for name, _, hookable in _PHASE_TABLE if hookable)
"""The phases an entity may append a hook to: :data:`PHASES` less the built-in ones.

What :class:`SubstepHooks` exposes as a list per phase. ``clear``, ``collide``
and ``solve`` are absent because the executor runs them itself.
"""


@dataclass
class Stage:
    """One unit of device work in the substep program.

    Attributes:
        name: For the manifest and for a reader of a profile.
        fn: The work, ``(state, dt) -> None``.
        phase: One of :data:`PHASES`.
        cadence: Run every this many substeps (or control steps, for ``before``
            and ``after``). 1 is every time.
        reads: What model or state arrays it reads, by name. Read by the
            executor to decide whether the eager ``after`` phase has to wait on
            the device before it runs.
        writes: What it writes.
    """

    name: str
    fn: Hook
    phase: str = "post_solve"
    cadence: int = 1
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"stage {self.name!r}: phase must be one of {PHASES}, got {self.phase!r}")
        if int(self.cadence) < 1:
            raise ValueError(f"stage {self.name!r}: cadence must be >= 1, got {self.cadence}")


class Schedule:
    """Stages by phase. Mutable up to capture; appending afterwards needs an invalidation."""

    def __init__(self) -> None:
        self.phases: dict[str, list[Stage]] = {phase: [] for phase in PHASES}

    def add(self, stage: Stage) -> Stage:
        self.phases[stage.phase].append(stage)
        return stage

    def stages(self, phase: str) -> list[Stage]:
        return self.phases[phase]

    def extend(self, other: Schedule) -> None:
        for phase in PHASES:
            self.phases[phase].extend(other.phases[phase])

    def names(self) -> tuple[tuple[str, str], ...]:
        """``(phase, name)`` for every stage, in order -- what the manifest records."""
        return tuple((phase, stage.name) for phase in PHASES for stage in self.phases[phase])

    def __bool__(self) -> bool:
        return any(self.phases[phase] for phase in PHASES)

    def __len__(self) -> int:
        return sum(len(self.phases[phase]) for phase in PHASES)


class SubstepHooks:
    """The phase lists an entity appends its kernels to.

    A view over a :class:`Schedule` that keeps the three names the entities have
    always used -- ``rigid_force``, ``pre_solve``, ``post_solve`` -- as plain
    lists of ``(state, dt)`` callables, and adds ``mirror``, ``before`` and
    ``after``. Appending a callable wraps it in a :class:`Stage` named after it.
    Every list supports ``append``, ``extend``, ``del hooks.post_solve[n:]`` and
    iteration, which is what the env's hook-lifetime bookkeeping relies on.

    Args:
        schedule: The schedule to write into. A fresh one when omitted.
        program: Whatever composed the physics, for an entity that needs to ask
            which solver owns it. ``None`` until composition binds it.
    """

    PHASES = HOOK_PHASES
    """The phases with a list on this object: :data:`HOOK_PHASES`, kept under the name readers use."""

    def __init__(self, schedule: Schedule | None = None, program: Any = None) -> None:
        self.schedule = schedule if schedule is not None else Schedule()
        self.program = program
        for phase in self.PHASES:
            setattr(self, phase, _PhaseList(self.schedule, phase))

    def extend(self, other: SubstepHooks) -> None:
        """Append every hook of ``other``, preserving contribution order."""
        for phase in self.PHASES:
            getattr(self, phase).extend(list(getattr(other, phase)))

    def add(self, phase: str, fn: Hook, *, name: str = "", cadence: int = 1) -> Stage:
        """Add a stage to any phase, with a cadence."""
        return self.schedule.add(Stage(name=name or _hook_name(fn), fn=fn, phase=phase, cadence=cadence))

    def __bool__(self) -> bool:
        return any(len(getattr(self, phase)) for phase in self.PHASES)


def _hook_name(fn: Any) -> str:
    owner = getattr(fn, "__self__", None)
    if owner is not None:
        return f"{type(owner).__name__}.{getattr(fn, '__name__', 'hook')}"
    return getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))


class _PhaseList:
    """A list-shaped view of one phase of a schedule, holding the callables."""

    def __init__(self, schedule: Schedule, phase: str) -> None:
        self._schedule = schedule
        self._phase = phase

    @property
    def _stages(self) -> list[Stage]:
        return self._schedule.phases[self._phase]

    def append(self, fn: Hook) -> None:
        self._stages.append(Stage(name=_hook_name(fn), fn=fn, phase=self._phase))

    def extend(self, fns) -> None:
        for fn in fns:
            self.append(fn)

    def __iter__(self) -> Iterator[Hook]:
        return iter([stage.fn for stage in self._stages])

    def __len__(self) -> int:
        return len(self._stages)

    def __getitem__(self, index):
        stages = self._stages[index]
        if isinstance(index, slice):
            return [stage.fn for stage in stages]
        return stages.fn

    def __delitem__(self, index) -> None:
        del self._stages[index]

    def remove(self, fn: Hook) -> None:
        """Drop the first stage running ``fn``, by identity then by equality.

        The studio's direct-manipulation path is what needs this: a click puts a
        dragger's ``apply`` into a phase and letting go takes it out again, and
        it has only the callable to name it by.

        **Equality is what does the work, and identity is the fast path.** A
        bound method is a new object on every attribute access -- ``d.apply is
        d.apply`` is False -- so an identity-only search would never find the
        hook a dragger appended. Equality does find it and finds the right one,
        since bound methods compare by ``__self__`` and ``__func__``: with two
        draggers live, ``d.apply == e.apply`` is False. Identity runs first
        because a plain function or a shared closure is matched by it without
        calling ``__eq__`` at all.

        Raises:
            ValueError: If no stage runs ``fn``, the way ``list.remove`` does. A
                silent miss would leave the hook running and the caller
                believing it had let go.
        """
        for index, stage in enumerate(self._stages):
            if stage.fn is fn:
                del self._stages[index]
                return
        for index, stage in enumerate(self._stages):
            if stage.fn == fn:
                del self._stages[index]
                return
        raise ValueError(f"no {self._phase} stage runs {_hook_name(fn)}")

    def __contains__(self, fn: Hook) -> bool:
        return any(stage.fn is fn or stage.fn == fn for stage in self._stages)

    def __bool__(self) -> bool:
        return bool(self._stages)

    def __repr__(self) -> str:
        return f"{self._phase}={[stage.name for stage in self._stages]}"


class ScheduleExecutor:
    """Runs a :class:`Schedule` over two states, eagerly or as a captured graph.

    Args:
        model: The finalized model.
        schedule: The program. The ``solve`` and ``collide`` stages are what
            the composer put there; this class adds ``clear`` and the swap.
        control_hz: Control rate [Hz]. One :meth:`step` advances 1/control_hz s.
        substeps: Physics substeps per control step. Must be even.
        use_cuda_graph: Whether to attempt capture at all.
        contacts: The main pipeline's contact buffer, or ``None``.
        solver: The solver the built-in ``solve`` step drives, or ``None`` for
            a schedule that supplies its own ``solve`` stages.
        collision: The main collision pipeline, run at the ``collide`` point
            every ``collide_every`` substeps, or ``None``.
        collide_every: The main pipeline's cadence in substeps.
    """

    def __init__(
        self,
        model: newton.Model,
        schedule: Schedule,
        *,
        control_hz: float = 50.0,
        substeps: int = 4,
        use_cuda_graph: bool = True,
        contacts: Any = None,
        solver: Any = None,
        collision: Any = None,
        collide_every: int = 1,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.device = getattr(model, "device", None)
        self.solver = solver
        """What the built-in ``solve`` step drives: ``solver.step(state_in, state_out, control, contacts, dt)``."""
        self.collision = collision
        """The main collision pipeline the built-in ``collide`` step runs, or ``None``."""
        self.collide_every = max(1, int(collide_every))
        self.control_hz = float(control_hz)
        self.substeps = assert_even_substeps(substeps)
        self.control_dt = 1.0 / self.control_hz
        self.sim_dt = self.control_dt / self.substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()
        self.contacts = contacts

        self.use_cuda_graph = bool(use_cuda_graph)
        self._graph: Any | None = None
        self._recapture = False
        self._substep_index = 0
        self.settling = False
        """Inside :meth:`substeps_at` -- a settle, which is scene construction.

        Nothing reads the state per step of a settle, so anything that costs
        something per step and is only there to be read (the batched read-back)
        skips while this is set. Sixty settle steps of a garment is 180 copies
        of twelve thousand particles for an answer nobody asks for."""

        self.capture_error: str = ""
        """The last capture attempt's exception, or empty."""
        self.steps_replayed = 0
        """Control steps that replayed a captured graph."""
        self.steps_eager = 0
        """Control steps that ran the substep block eagerly."""
        self.recapture_failures = 0
        """How many re-capture attempts failed."""

    # ------------------------------------------------------------------ shape
    @property
    def state(self) -> newton.State:
        """The state holding the current frame. Even substeps keep this stable."""
        return self.state_0

    @property
    def captured(self) -> bool:
        return self._graph is not None

    # ---------------------------------------------------------------- running
    def _run_phase(self, phase: str, state: newton.State, index: int, dt: float) -> None:
        stages = self.schedule.phases[phase]
        if not stages:
            return
        for stage in stages:
            if index % stage.cadence == 0:
                stage.fn(state, dt)

    def substep_block(self) -> None:
        """One control period of physics, eagerly. The body of the captured graph."""
        for index in range(self.substeps):
            self._substep_index = index
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self._run_phase("clear", self.state_0, index, self.sim_dt)
            self._run_phase("rigid_force", self.state_0, index, self.sim_dt)
            self._run_phase("pre_solve", self.state_0, index, self.sim_dt)
            self._run_phase("collide", self.state_0, index, self.sim_dt)
            if self.collision is not None and self.contacts is not None and index % self.collide_every == 0:
                self.collision.collide(self.state_0, self.contacts)
            if self.solver is not None:
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self._run_phase("solve", self.state_0, index, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._run_phase("post_solve", self.state_0, index, self.sim_dt)
            self._run_phase("mirror", self.state_0, index, self.sim_dt)

    def run_before(self) -> None:
        self._run_phase("before", self.state_0, self.step_count, self.control_dt)

    def run_after(self) -> None:
        self._run_phase("after", self.state_0, self.step_count, self.control_dt)

    def _after_reads_the_device(self) -> bool:
        """Whether any ``after`` stage reads state the body has just queued work on.

        ``step`` *enqueues* the substep block and returns; the device has not
        necessarily finished when ``after`` begins. A host-engine tick reads
        ``body_q`` there to drive its mirrors, so it must not begin before the
        substeps that wrote ``body_q`` have landed.

        Today that read happens to be safe by accident: the one shipped host
        bridge reads through ``state.body_q.numpy()``, and a device-to-host copy
        stalls the queue as a side effect. That is not a property to rely on --
        an engine reading device-side, or through torch on its own stream, would
        inherit a silent race whose symptom is a mirror one control step stale
        *sometimes*. So the stall is made explicit here, and paid only where it
        is owed: a stage declares what it ``reads``, and the only shipped
        ``after`` stage that declares anything is the cosim tick. A Newton-only
        scene has an empty ``after`` phase and synchronises nothing. Asked on
        every step rather than once, so a stage added after construction counts.
        """
        return any(stage.reads for stage in self.schedule.phases["after"])

    # ---------------------------------------------------------------- capture
    def warmup_and_capture(self, *, iterations: int = 2, verbose: bool = False) -> bool:
        """Compile the kernels, then capture the substep block if we can.

        Runs real physics, so the caller must restore the scene afterwards.
        Returns whether a graph was captured; ``False`` is the eager path,
        which is a supported way to run and not an error.
        """
        for _ in range(max(0, iterations)):
            self.run_before()
            self.substep_block()
        if not self.model.device.is_cuda or not self.use_cuda_graph:
            wp.synchronize()
            return False
        try:
            with capturing(self.model.device) as capture:
                self.substep_block()
            self._graph = capture.graph
            self.capture_error = ""
        except Exception as exc:
            print(f"[pipeline] CUDA graph capture failed ({type(exc).__name__}: {exc}); running eager")
            _log.debug("CUDA graph capture failed", exc_info=True)
            self.capture_error = f"{type(exc).__name__}: {exc}"
            self._graph = None
        wp.synchronize()
        if verbose:
            print(f"[pipeline] {'captured' if self._graph is not None else 'eager'}, {self.substeps} substeps")
        return self._graph is not None

    @contextmanager
    def substeps_at(self, substeps: int) -> Iterator[None]:
        """Temporarily run at another substep count, eagerly.

        For a settle, which wants a finer timestep than the run does. Any
        captured graph is set aside rather than dropped, and a pending
        re-capture waits until the run's own substep count is back -- a graph
        captured at the settle's count would be the wrong body. If the stages
        change inside the block, the graph set aside is stale and is dropped
        for a re-capture instead. The clock is *not* restored -- a settle is
        scene construction, not simulated time.
        """
        saved_substeps, saved_dt = self.substeps, self.sim_dt
        saved_graph, saved_recapture = self._graph, self._recapture
        self.substeps = assert_even_substeps(substeps)
        self.sim_dt = self.control_dt / self.substeps
        self._graph = None
        self._recapture = False
        self.settling = True
        try:
            yield
        finally:
            self.substeps, self.sim_dt = saved_substeps, saved_dt
            if self._recapture:  # invalidated inside the block
                self._graph = None
            else:
                self._graph, self._recapture = saved_graph, saved_recapture
            self.settling = False

    def invalidate_graph(self, *, recapture: bool = True) -> None:
        """Drop the captured graph after the set of stages changed.

        Args:
            recapture: Re-capture on the next :meth:`step`, once the caller has
                finished adding stages. The alternative is running eager for
                the rest of the session.
        """
        self._graph = None
        self._recapture = bool(recapture)

    def _try_recapture(self) -> None:
        self._recapture = False
        if not self.model.device.is_cuda or not self.use_cuda_graph:
            return
        try:
            with capturing(self.model.device) as capture:
                self.substep_block()
            self._graph = capture.graph
            self.capture_error = ""
        except Exception as exc:
            print(f"[pipeline] re-capture failed ({type(exc).__name__}: {exc}); running eager")
            _log.debug("CUDA graph re-capture failed", exc_info=True)
            self.capture_error = f"{type(exc).__name__}: {exc}"
            self.recapture_failures += 1
            self._graph = None

    # -------------------------------------------------------------------- run
    def step(self) -> None:
        """Advance one control period: ``before`` eagerly, the body, then ``after``."""
        self.run_before()
        if self._graph is not None:
            self.steps_replayed += 1
            wp.capture_launch(self._graph)
        else:
            self.steps_eager += 1
            self.substep_block()
            if self._recapture and not self.settling:
                self._try_recapture()
        if self._after_reads_the_device():
            # The body above only *enqueued* its work; an ``after`` stage that
            # reads what it wrote has to wait for it. See _after_reads_the_device.
            wp.synchronize_device(self.device)
        self.run_after()
        self.sim_time += self.control_dt
        self.step_count += 1

    def sync(self) -> None:
        """Block until the device has finished the work :meth:`step` queued."""
        wp.synchronize()
