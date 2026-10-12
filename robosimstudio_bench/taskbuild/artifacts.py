# SPDX-License-Identifier: Apache-2.0
"""Resolving what a document references: a routine, a tool registrar, a graph.

A task document names its solutions rather than containing them -- *the task is
data; solutions may be code* -- so something has to turn ``"routine:RoutineWeave"``
into a class and ``"graphs/gt"`` into a directory. That is this module, and the
rule it applies is what makes a task directory movable.

**The rule is the dot.** A module part with no dot in it is resolved against the
task's *own* package; one with dots is resolved by ``importlib`` as written::

    (:routine "routine:RoutineWeave")                          this task's routine.py
    (:tools   "tools:register_cable_tools")                    this task's tools.py
    (:tools   "robosimstudio_policy_agent.sim_tools:register_goal_cursor")
                                                                a library tool, verbatim

A document no longer names the tree it happens to sit in, so moving a task
between roots -- or handing it to somebody whose checkout is laid out
differently -- takes no edits. And a library reference stays exactly what it
was, because a library *is* an installed package and naming one by its dotted
path is correct.

**A graph is a directory, not an import.** ``(:graph gt "graphs/gt")`` names
workflow JSON and node scripts read by the graph runtime rather than by Python,
so it resolves relative to the task's own directory. Suites record that directory
when they register, which is why this needs no import at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["graph_dir", "resolve_dotted", "task_resolver"]


def resolve_dotted(path: str) -> Any:
    """The thing a dotted path names, without calling it.

    Args:
        path: ``"package.module:Name"`` or ``"package.module.Name"``.
    """
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"path {path!r} must be 'module:Name' or 'module.Name'")
    return getattr(importlib.import_module(module_name), attr)


def _holds(package: str, module_name: str) -> bool:
    """Whether ``package`` has a ``module_name`` of its own, without executing it."""
    try:
        return importlib.util.find_spec(f"{package}.{module_name}") is not None
    except ModuleNotFoundError:  # a parent on the dotted path is not the task's
        return False


def task_resolver(directory: Path, package: str = "") -> Callable[[str], Any]:
    """An artifact resolver for one task, applying the dot rule.

    Args:
        directory: The task's own directory, for error messages.
        package: The synthetic package the task was executed as. Empty means the
            task ships no Python, in which case a task-relative reference is a
            document naming a file it does not have -- and the error says so
            rather than raising ``ModuleNotFoundError`` about a name nobody
            wrote.

    Returns:
        A callable taking one reference string.
    """

    def resolve(path: str) -> Any:
        module_name, _, attr = path.partition(":")
        if not attr:
            module_name, _, attr = path.rpartition(".")
        # The task's own package first -- ``routine:X`` and ``prep.layout:X``
        # alike -- then the installed world for a dotted name it does not hold.
        if package and _holds(package, module_name):
            return getattr(importlib.import_module(f"{package}.{module_name}"), attr)
        if "." in module_name:
            return resolve_dotted(path)
        if not package:
            raise ValueError(
                f"{directory.name!r} references {path!r}, which names a module in the task's own "
                f"directory, but {directory} ships no Python to import it from"
            )
        return getattr(importlib.import_module(f"{package}.{module_name}"), attr)

    return resolve


def graph_dir(suite: str, relative: str) -> str:
    """A shipped graph directory, from the suite it belongs to.

    The suite records where it was loaded from, so this is a join rather than an
    import -- which is what lets a graph live in a directory that is not on
    ``sys.path`` and never was.
    """
    from .registry import get_suite

    home = getattr(get_suite(suite), "home", None)
    if home is None:
        raise ValueError(f"suite {suite!r} did not record where it was loaded from; cannot find {relative!r}")
    return str(Path(home) / relative)
