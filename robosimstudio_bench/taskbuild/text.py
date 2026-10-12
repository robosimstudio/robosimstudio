# SPDX-License-Identifier: Apache-2.0
"""A task written as a string: the language a ``.task`` file holds, without the file.

A folder of documents is how a suite ships; a string is how a script or a test
states one task and runs it. Both are lowered by
:func:`~robosimstudio_bench.taskbuild.discovery.build_variant`, so a document
means the same thing in either place.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from robosimstudio.specs import TaskSpec, TaskSuite

__all__ = ["task"]

_FROM_TEXT: set[str] = set()
"""Suites this module registered. Stating the same task again replaces it, as
re-running a cell would; a name a folder registered is never replaced."""


def task(text: str, *, home: str | Path | None = None, suite: str = "") -> TaskSpec:
    """Parse, lower and register one task document written as a string.

    Args:
        text: One ``(define (problem NAME) ...)`` document.
        home: The folder its relative paths resolve against: ``(:graph LABEL
            "DIR")`` names a folder under it. Defaults to the folder of the file
            that called this, or the working directory from a notebook or a
            prompt. A suite name a task root also holds is refused when that root
            is scanned, as two folders sharing a name are.
        suite: The suite to register it as. Defaults to the problem's name, so
            ``(define (problem push) ...)`` is ``push/push``, and ``push`` alone
            finds it.

    Returns:
        The task. It is also what :func:`~robosimstudio_bench.taskbuild.get_task`,
        the run command and the studio find under its name.

    Raises:
        robosimstudio_bench.taskdoc.reader.ParseError: The text is not well formed.
        robosimstudio_bench.taskbuild.lower.LowerError: It is well formed and asks
            for something that does not exist.
    """
    from robosimstudio_bench.taskdoc.parse import parse_problem
    from robosimstudio_bench.taskdoc.reader import parse_sexp

    from .artifacts import resolve_dotted
    from .discovery import build_variant
    from .registry import SUITES, register_suite

    if home is None:
        caller = Path(inspect.stack()[1].filename)
        # A notebook cell's "file" is a name in a temp directory (ipykernel_<pid>/),
        # and an interactive line has none: both mean the working directory.
        home = caller.parent if caller.is_file() and not caller.parent.name.startswith("ipykernel_") else Path.cwd()
    home = Path(home).resolve()
    doc = parse_problem(parse_sexp(text, where="<string>"), where="<string>")
    suite = suite or doc.name
    spec = build_variant(doc, directory=home, suite=suite, name=doc.name, resolve=resolve_dotted)
    stated = TaskSuite(name=suite, variants=(spec,), description=doc.language, home=home)
    if suite in _FROM_TEXT:
        SUITES.replace(suite, stated, origin="a task written as a string")
    else:
        register_suite(stated)
        _FROM_TEXT.add(suite)
    return stated.variants[0]
