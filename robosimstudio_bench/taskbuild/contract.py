# SPDX-License-Identifier: Apache-2.0
"""What a task directory may hold and what it may name: the layout contract, as functions.

The rules live here, once, so a tool can certify a directory that is not in this
checkout -- somebody else's collection -- with the same checks applied to ours.
Every function returns plain problems as strings, empty when the directory is
clean; a caller asserts emptiness or turns them into findings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from robosimstudio.roots import SUITE_SUFFIX, TASK_SUFFIX, is_task_dir

__all__ = [
    "LIBRARY_PREFIXES",
    "POSES_DIR",
    "TASK_PYTHON",
    "artifact_problems",
    "layout_problems",
    "reaches_out",
    "seam_problems",
    "task_dirs",
]

TASK_PYTHON: tuple[str, ...] = (
    "difficulty.py",
    "staging.py",
    "scenes.py",
    "records.py",
    "repro_physics.py",
    "routine.py",
    "tools.py",
)
"""Python a task directory may hold, by name, and what each is for.

Three kinds. **Builders** -- ``staging.py`` (the profile and arrangements),
``scenes.py`` (the cell), ``records.py`` (the asset declarations and the
entity code they bind), ``repro_physics.py`` (an imported reproduction's
physics) -- build the scene a document cannot describe. **Solutions** --
``routine.py``, ``tools.py`` -- answer the question the document asks. The
**measure** -- ``difficulty.py`` -- orders the variants. Any other module at
the top of a task directory has to be imported by one of these, which is
what makes it that file's part rather than a task defined in Python; ``assets/``
holds data only, and ``graphs/`` the graphs.
"""

LOADER = "task.py"
"""The per-suite loader every suite once carried. None may, since 2026-09-14."""

POSES_DIR = "poses"
"""Where a task's recorded layouts go. ``start_poses/`` was the name one suite used and is refused."""

LIBRARY_PREFIXES = ("robosimstudio.", "robosimstudio_bench.", "robosimstudio_policy_agent.")
"""Dotted artifact references a document may make besides its own task's."""

SHARED = "_shared"
"""The one directory on a root that is not a task: what several tasks of one collection run."""

_VOUCHED = re.compile(r"^from \. import [\w, ]*\b{stem}\b")


def task_dirs(root: str | Path) -> list[Path]:
    """Every task directory on a root, in name order."""
    return sorted(p for p in Path(root).iterdir() if is_task_dir(p))


def layout_problems(task_dir: str | Path) -> list[str]:
    """What is wrong with the shape of one task directory, relative to its root.

    A task is a directory of documents. It holds at least one ``.task``; no
    ``task.py``; Python only from :data:`TASK_PYTHON`, or a module one of its
    solutions imports (``routine.py``/``tools.py`` vouch for a split-out part
    the way a loader used to); recorded layouts under ``poses/``.
    """
    task = Path(task_dir)
    root = task.parent
    problems: list[str] = []
    if not any(task.glob(f"*{TASK_SUFFIX}")):
        problems.append(f"{task.name}: states no variants; a directory with no document is not a task")
    if (task / LOADER).is_file():
        problems.append(f"{task.name}/{LOADER}: a loader; a task is built from its documents, its .suite and assets/")
    if (task / "start_poses").is_dir():
        problems.append(f"{task.name}/start_poses: recorded layouts go under {POSES_DIR}/")
    # Every module of the task but the one under question, sub-packages included:
    # a module is the task's part when something of the task imports it, and an
    # orphan is a task defined in Python.
    sources = {p: p.read_text() for p in task.rglob("*.py") if "graphs" not in p.relative_to(task).parts}
    # A module a document names by the artifact rule (``layout:LEVEL1``,
    # ``look:GEAR_MESH_LOOK``) is the task's part too: the document is what
    # imports it.
    named: set[str] = set()
    for doc in [*task.glob(f"*{TASK_SUFFIX}"), *task.glob(f"*{SUITE_SUFFIX}")]:
        named |= set(re.findall(r'[\s("]([a-z_]+)(?:\.[a-z_]+)*:[A-Za-z_]', doc.read_text()))
    for path in sorted(task.glob("*.py")):
        if path.name in TASK_PYTHON:
            continue
        reachable = "\n".join(text for other, text in sources.items() if other != path)
        vouched = (
            re.search(_VOUCHED.pattern.format(stem=path.stem), reachable, re.M)
            or f"from .{path.stem} import" in reachable
            or f"from ..{path.stem} import" in reachable
            or f"from ...{path.stem} import" in reachable
            or path.stem in named
        )
        if not vouched:
            problems.append(
                f"{path.relative_to(root)}: a task directory's own Python is one of {TASK_PYTHON} or a module one "
                "of those imports; anything else is a task defined in Python, which is what a document is for"
            )
    return problems


def reaches_out(here: tuple[str, ...], level: int, module: str | None) -> bool:
    """Whether one relative import leaves the task ``here`` sits in.

    ``here`` is the importing module's package as parts under the root
    (``("cable_route", "assets")``), ``level`` the import's dot count and
    ``module`` what follows the dots. ``_shared`` is the one escape hatch.
    """
    # Climbing past the root is outside whatever the depth: a negative bound
    # would wrap and keep a prefix, reading an over-deep import as inside.
    landing = here[: max(0, len(here) - (level - 1))]
    if landing:
        return False
    return (module or "").partition(".")[0] != SHARED


def seam_problems(task_dir: str | Path) -> list[str]:
    """Every relative import that lands outside the task, relative to its root.

    A task importing a sibling is a task nobody can hand to anybody. Graph
    scripts are skipped: they run inside the graph runtime, which resolves
    them by name rather than by import.
    """
    task = Path(task_dir)
    root = task.parent
    problems: list[str] = []
    for path in sorted(task.rglob("*.py")):
        if "graphs" in path.relative_to(root).parts:
            continue
        here = path.parent.relative_to(root).parts
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level and reaches_out(here, node.level, node.module):
                problems.append(
                    f"{path.relative_to(root)}: from {'.' * node.level}{node.module or ''} import ... leaves the task"
                )
    return problems


def artifact_problems(document: str | Path) -> list[str]:
    """Every artifact reference in one document that names neither its own task nor the library."""
    from robosimstudio_bench.taskdoc import read_problem

    path = Path(document)
    problems: list[str] = []
    for key, value in read_problem(path).artifacts.items():
        for reference in value if isinstance(value, list) else [value]:
            text = str(reference).strip('"')
            module = text.partition(":")[0]
            if "." in module and not module.startswith(LIBRARY_PREFIXES):
                problems.append(
                    f"{path.parent.name}/{path.stem}: {key} -> {text} names neither its task nor the library"
                )
    return problems
