# SPDX-License-Identifier: Apache-2.0
"""Which policies a task ships, by name, and how each is constructed.

One answer for the studio's policy panel, the record tool and the certifier,
which used to work it out three times: a scripted ``routine`` is offered on the
task's native pairing only (a routine plans against the arm it was written
for), and every graph the binding claims is offered by its label.

The scripted routine is one module, :data:`ROUTINES_MODULE`, and a release may
leave it out: there a task's policy is a graph, and :func:`routines_installed` is
what every place that would offer a routine asks first.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from robosimstudio.specs import Binding, TaskSpec

__all__ = [
    "ROUTINE",
    "ROUTINES_ABSENT",
    "ROUTINES_MODULE",
    "declared_solutions",
    "routines_installed",
    "solution_source",
]

ROUTINE = "routine"
"""The label of the scripted routine, where a graph's label is the graph's own."""

ROUTINES_MODULE = "robosimstudio_bench.solutions.routines"
"""The whole scripted-routine mechanism: the single-world wrapper and its batched twin."""

ROUTINES_ABSENT = (
    "Scripted routines are not in this release: a task's policy is a graph. "
    'Declare one with (:graph LABEL "DIR") and run it with robosimstudio_policy_agent.SolutionGraph.'
)
"""The one sentence every refusal of a routine ends with."""


def routines_installed() -> bool:
    """Whether :data:`ROUTINES_MODULE` is part of this install."""
    try:
        return importlib.util.find_spec(ROUTINES_MODULE) is not None
    except (ImportError, ValueError):
        return False


def declared_solutions(task: TaskSpec, binding: Binding | None = None) -> tuple[str, ...]:
    """Every policy this pairing supports, the routine first.

    Args:
        task: The task.
        binding: The pairing, defaulting to the task's native one.
    """
    native = task.native_binding
    binding = binding or native
    native_routine = task.routine is not None and binding.key(task) == native.key(task)
    routine = (ROUTINE,) if native_routine and routines_installed() else ()
    return routine + tuple(binding.graphs)


def solution_source(label: str) -> tuple[str, dict[str, Any]]:
    """The dotted path ``load_solution`` resolves and the kwargs it takes, for one label."""
    if label == ROUTINE:
        return f"{ROUTINES_MODULE}:SolutionRoutine", {}
    return "robosimstudio_policy_agent.graph_solution:SolutionGraph", {"graph": label}
