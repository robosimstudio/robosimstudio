# SPDX-License-Identifier: Apache-2.0
"""The task-level schedule: what runs once per compile, per episode, per step.

:mod:`~robosimstudio.core.schedule` is the other one. That is the *substep*
program -- device work, ordered inside a captured graph, every stage the same
``(state, dt)`` shape. This is its task-level twin: the phases a scene passes
through from spec to score, in the order it passes through them.

    stage → build → bind → materials → kernels → ready → reset → observe → score → close

**One departure from the substep schedule, and it is deliberate.** There, every
stage has the same signature, so the executor can call them all. Here they do
not: ``build`` takes a builder and an episode, ``bind`` takes a finalized model,
``kernels`` takes the hook lists, ``ready`` takes a pipeline. Flattening that
into one ``fn(**context)`` would mean inventing a context object that every
phase reads different fields out of -- a second vocabulary to keep in step with
the first. So a :class:`TaskSchedule` **orders and holds**; the call site passes
the arguments its own phase takes. What it buys is that the ordering exists in
one place instead of being implied by the sequence of loops in a compiler.

**Which is the point: a phase is no longer something only an entity can reach.**
Every one of these already existed as a method on the entity base class, called
from a loop over the entities, so a component that owned no geometry could not
join in. (Named in prose rather than cross-referenced: this layer may not name
the one that defines it, which is the same rule that keeps the phases duck-typed
here rather than checked against a type.)
:func:`steps_of` discovers phases by *method name* on any object, so anything
that implements one takes part -- and anything that implements none costs
nothing.

**Some of these lists will be empty, and that is known rather than assumed.**
Counted across the corpus at the time this was written: ``build`` 6 overrides,
``bind`` 4, ``kernels`` 3, ``ready`` 2, and ``after_build``,
``apply_model_materials`` and ``close`` zero. ``close`` has no implementations
and nothing working around its absence. The vocabulary is here in full anyway,
so that a phase which acquires a first user acquires it by being named rather
than by growing a seam.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["PHASE_METHODS", "TASK_PHASES", "Step", "TaskSchedule", "steps_of"]

TASK_PHASES: tuple[str, ...] = (
    "stage",
    "build",
    "bind",
    "materials",
    "kernels",
    "ready",
    "reset",
    "observe",
    "score",
    "close",
)
"""Every task-level phase, in the order a scene passes through them.

Load-bearing, and the compiler's own docstring says so: doing a step later or
earlier than this "fails, usually silently". Scenery stands before the fixtures
that sit on it, the robot before the objects it will hold, deformables after the
things they drape over -- and ``materials`` is the one window between a
finalized model and the solvers copying its arrays.
"""

PHASE_METHODS: dict[str, str] = {
    "build": "build_into",
    "bind": "bind",
    "materials": "apply_model_materials",
    "kernels": "contribute_hooks",
    "ready": "on_scene_ready",
    "reset": "reset",
    "observe": "observe",
    "close": "close",
}
"""Phase → the method name :func:`steps_of` looks for.

The names are the ones the entities already use, so this is a *reading* of the
existing contract rather than a second one beside it. ``stage`` and ``score``
are absent deliberately: staging happens before any part exists to have a
method, and scoring is the evaluator's whole job rather than one phase of it.
"""


@dataclass
class Step:
    """One participant in one phase.

    Attributes:
        name: For a manifest and for whoever reads a compile trace.
        fn: The work. Its signature is the phase's, not this class's -- see the
            module docstring.
        phase: One of :data:`TASK_PHASES`.
        cadence: Run every this many episodes (``reset``) or steps (``observe``,
            ``score``). 1 is every time. Meaningless for the once-per-compile
            phases, where it is left at 1.
    """

    name: str
    fn: Callable[..., Any]
    phase: str
    cadence: int = 1

    def __post_init__(self) -> None:
        if self.phase not in TASK_PHASES:
            raise ValueError(f"step {self.name!r}: phase must be one of {TASK_PHASES}, got {self.phase!r}")
        if int(self.cadence) < 1:
            raise ValueError(f"step {self.name!r}: cadence must be >= 1, got {self.cadence}")


class TaskSchedule:
    """Steps by phase, in contribution order within a phase."""

    def __init__(self) -> None:
        self.phases: dict[str, list[Step]] = {phase: [] for phase in TASK_PHASES}

    def add(self, step: Step) -> Step:
        self.phases[step.phase].append(step)
        return step

    def steps(self, phase: str) -> list[Step]:
        if phase not in self.phases:
            raise KeyError(f"no such phase {phase!r}; one of {TASK_PHASES}")
        return self.phases[phase]

    def extend(self, other: TaskSchedule) -> None:
        """Append every step of ``other``, preserving contribution order."""
        for phase in TASK_PHASES:
            self.phases[phase].extend(other.phases[phase])

    def names(self) -> tuple[tuple[str, str], ...]:
        """``(phase, name)`` for every step, in order."""
        return tuple((phase, step.name) for phase in TASK_PHASES for step in self.phases[phase])

    def __bool__(self) -> bool:
        return any(self.phases[phase] for phase in TASK_PHASES)

    def __len__(self) -> int:
        return sum(len(self.phases[phase]) for phase in TASK_PHASES)


def steps_of(part: Any, *, name: str = "") -> TaskSchedule:
    """Every phase ``part`` implements, discovered by method name.

    Args:
        part: Anything. An entity, an evaluator, a camera rig, a metric -- the
            discovery is by method name and nothing here checks a base class,
            which is the whole reason this exists.
        name: What to call it in a trace. Defaults to the part's ``name``
            attribute, then its type.

    Returns:
        A schedule holding one step per phase the part implements. A part
        implementing none returns an empty schedule, which is falsey.

    A method is taken as implementing its phase when the part *has* it and it is
    callable. Inheriting a base class's do-nothing default therefore counts --
    which is correct: a schedule that tried to tell an override from an
    inherited stub would be guessing, and calling a no-op costs nothing.
    """
    label = name or str(getattr(part, "name", "") or type(part).__name__)
    schedule = TaskSchedule()
    for phase, method in PHASE_METHODS.items():
        fn = getattr(part, method, None)
        if callable(fn):
            schedule.add(Step(name=f"{label}.{method}", fn=fn, phase=phase))
    return schedule
