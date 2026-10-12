# SPDX-License-Identifier: Apache-2.0
"""From a registered task id to the document it was built from, and the profile it reads under."""

from __future__ import annotations

from pathlib import Path

from robosimstudio.roots import SUITE_SUFFIX, TASK_SUFFIX

__all__ = ["document_of", "documents_of", "profile_of"]


def documents_of(suite: str) -> dict[str, Path]:
    """Every document of a registered suite, by variant name, from the directory it was loaded from."""
    from .registry import get_suite

    home = getattr(get_suite(suite), "home", None)
    if home is None:
        raise LookupError(f"suite {suite!r} did not record where it was loaded from")
    return {path.stem: path for path in sorted(Path(home).glob(f"*{TASK_SUFFIX}"))}


def document_of(task: str) -> Path:
    """``suite/variant`` to the ``.task`` file it is."""
    suite, _, variant = task.partition("/")
    documents = documents_of(suite)
    if variant not in documents:
        raise LookupError(f"suite {suite!r} has no variant {variant!r}; it has {sorted(documents)}")
    return documents[variant]


def profile_of(document: str | Path) -> str:
    """The profile a document reads under: its own ``(:profile ...)``, merged over its suite's.

    The same rule the loader applies, and not cosmetic: checking ``tool_sort``
    unbound reports eight false problems on a clean document, because standard
    BDDL's own ``In`` is not an atom anybody catalogued.
    """
    from robosimstudio_bench.taskdoc import merge_suite, read_problem, read_suite

    path = Path(document)
    suites = sorted(path.parent.glob(f"*{SUITE_SUFFIX}"))
    staging = dict(merge_suite(read_problem(path), read_suite(suites[0]) if suites else None).staging)
    return str(staging.get("profile", ""))
