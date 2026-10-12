# SPDX-License-Identifier: Apache-2.0
"""Which task suites exist, and how a variant is named.

One registry, populated from the content packages a discovery pass imports. A
suite registers itself at module scope while being imported, and it *builds* its
variants there -- a registry that can hand out a task which then fails to build
is a registry nobody can trust.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from robosimstudio.registry import Registry
from robosimstudio.specs import TaskSuite


def _discover() -> None:
    """Execute every declared task root. Imported late to avoid a cycle.

    The discovery pass imports the asset library and the lowering machinery,
    both of which reach back here -- so it is named at call time rather than at
    module scope, exactly as the corpus package is.
    """
    from .discovery import load_roots

    load_roots()


__all__ = [
    "SUITES",
    "get_suite",
    "get_task",
    "list_tasks",
    "load_all",
    "register_suite",
]


SUITES: Registry[TaskSuite] = Registry("task suite")
SUITES.add_discover(_discover)
"""Task suites. **No package at all: directories on a root.**

This held two for the length of the migration -- a library ``tasks`` package for the
suites still defined in Python, the corpus for the ported ones -- and the
docstring said the first entry would drop away when the last suite was ported.
It has, and the library package went with it: the policy artifacts that kept it
alive after the ports (a routine, a tool registry, a directory of graph scripts)
are content by the same argument the documents are, and now sit beside them under
``tasks/<name>/``. A graph path resolves through
``policies.base.graph_dir``, which names this same package.

``iter_modules`` yields immediate children only, so a suite's ``routine.py`` and
``tools.py`` are *not* imported by the scan -- they are resolved by dotted path
when a document's artifact is asked for, which is what keeps a broken graph
script from breaking every task that shares the process.

Duplicate refusal stays too, and is still doing work: two documents claiming the
name ``tool_sort`` is an error at import rather than a coin toss at lookup."""


def register_suite(suite: TaskSuite) -> TaskSuite:
    """Register a task suite. Call at module scope in the suite's loader.

    Stamps :attr:`~robosimstudio.specs.TaskSuite.home` from the caller's own file
    when the suite did not set it, so a ``(:graph ...)`` reference resolves
    against the directory the suite was actually loaded from. Read off the frame
    rather than passed, because every loader would otherwise have to remember to
    say where it is -- and the one that forgot would fail only when something
    asked for its graph.
    """
    if suite.home is None:
        frame = inspect.stack()[1]
        object.__setattr__(suite, "home", Path(frame.filename).resolve().parent)
    return SUITES.register(suite.name, suite, origin=getattr(suite, "__module__", ""))


def get_suite(name: str) -> TaskSuite:
    return SUITES.get(name)


def get_task(full_name: str):
    """Resolve ``"<suite>/<variant>"`` to its :class:`~robosimstudio.specs.TaskSpec`.

    A bare suite name resolves to its first variant, which is the one a suite
    should put first.
    """
    suite_name, _, variant = full_name.partition("/")
    suite = get_suite(suite_name)
    return suite.variant(variant) if variant else suite.variants[0]


def list_tasks() -> list[str]:
    """Every registered ``"<suite>/<variant>"``, sorted.

    Deliberately unchanged by the scene and embodiment axes: this answers "what
    tasks exist", and a caller wanting the combinations wants :func:`list_pairs`.
    Folding them together would make every existing consumer of this list --
    menus, CLI defaults, test fixtures -- start reporting a matrix.
    """
    return sorted(f"{name}/{v}" for name, suite in SUITES for v in suite.variant_names)


def load_all() -> None:
    """Import every registering module in every registry.

    The one place that knows the full set. Ordered suites-last would be tidier
    to read and wrong to run: a suite registers entity classes and asset records
    while being imported, so anything importing it must already be able to take
    them.
    """
    from robosimstudio.robots.registry import EMBODIMENTS, GRIPPERS
    from robosimstudio.scene.registry import CELLS

    SUITES.load()
    CELLS.load()
    GRIPPERS.load()
    EMBODIMENTS.load()
