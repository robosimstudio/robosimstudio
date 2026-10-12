# SPDX-License-Identifier: Apache-2.0
"""Policies: what drives an environment.

Only the base is imported eagerly. The graph and remote policies pull in
optional dependencies, so they are reached by importing their modules -- or,
for a job file, by naming their dotted path. ``SolutionRoutine`` is reached by
name, and a release that ships graphs only says so in one sentence
(:data:`~robosimstudio_bench.solutions.declared.ROUTINES_ABSENT`).
"""

from .base import SolutionBase, SolutionIdle, load_solution

__all__ = ["SolutionBase", "SolutionIdle", "load_solution"]


def __getattr__(name: str):
    if name == "SolutionRoutine":
        from .declared import ROUTINES_ABSENT, ROUTINES_MODULE, routines_installed

        if not routines_installed():
            # An ImportError, so the sentence survives ``from ... import SolutionRoutine``.
            raise ModuleNotFoundError(ROUTINES_ABSENT, name=ROUTINES_MODULE)
        from .routines import SolutionRoutine

        return SolutionRoutine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
