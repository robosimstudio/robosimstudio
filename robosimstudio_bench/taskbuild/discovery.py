# SPDX-License-Identifier: Apache-2.0
"""Finding task directories on disk, and executing them.

A task is a directory rather than an installed package, so it cannot be found by
``importlib`` and cannot be imported by name. This module is the bridge: it walks
the roots :mod:`robosimstudio.roots` declares, and for each task directory
executes its loader under a **synthetic package name** so that ordinary Python
imports work inside it.

**Why a synthetic package rather than ``sys.path``.** Putting a root on
``sys.path`` would make every task directory a top-level importable name, so two
roots holding a ``cable_route`` would shadow each other silently and a task
called ``json`` would be a very bad afternoon. Instead each root gets an opaque
parent -- ``rstask.<slug>_<hash>`` -- and each task becomes a subpackage of it.
The consequences are the ones a task author wants:

* ``from .rod import SpecCable`` works, because the task dir is a package
  with its directory on its own ``__path__``;
* nothing a task defines can collide with anything installed.

The hash is of the root's real path, so declaring the same root twice -- once
relative, once through a symlink -- reaches the same modules rather than
executing every task in it a second time under a different name.

**Loading is eager, and stays that way.** A suite builds all its variants while
being imported, so a registry that can hand out a task which then fails to build
is a registry nobody can trust. The cost is that listing tasks reads every
document; the benefit is that a broken task is a loud failure at load rather
than a mystery at build, and the error names the directory and the root it came
from so the remedy is obvious.

**A directory with documents and no loader is still a task.** The generic loader
below builds each document with no task-specific arithmetic, which is the
zero-Python path: a root of pure documents needs nothing else to run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

from robosimstudio import roots

__all__ = [
    "PARENT_PACKAGE",
    "load_roots",
    "package_for_root",
    "scan",
    "task_loader",
    "task_module",
]

PARENT_PACKAGE = "rstask"
"""The synthetic parent every root's package hangs off.

Never installed and never importable by a human: it exists so that task
directories can be packages without being on ``sys.path``.
"""

_LOADED: set[str] = set()
"""Roots already executed this process, by real path. Load is idempotent."""

_SCANNING = False
"""Whether a load_roots pass is on the stack — see its docstring."""


def _slug(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or "root"


def package_for_root(root: Path) -> str:
    """The synthetic package name a root's tasks live under.

    Keyed on the *real* path so the same directory reached two ways is one
    package, prefixed with the parent directory's name so a collection's root
    reads as its own, and suffixed with a hash so two roots that happen to both
    be called ``tasks`` do not collide.
    """
    real = str(root.resolve())
    digest = hashlib.sha1(real.encode()).hexdigest()[:8]
    # The parent's name too: three collections each hold a ``tasks/``, and a
    # traceback through ``…tasks_3f2a1b9c`` names nothing a reader can find.
    return f"{PARENT_PACKAGE}.{_slug(root.parent.name)}_{_slug(root.name)}_{digest}"


def _namespace(name: str, paths: list[str]) -> ModuleType:
    """Install (or fetch) a synthetic package with ``paths`` as its search path."""
    existing = sys.modules.get(name)
    if existing is not None:
        for path in paths:
            if path not in existing.__path__:
                existing.__path__.append(path)
        return existing
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = list(paths)
    sys.modules[name] = module
    return module


def _exec(name: str, path: Path, *, is_package: bool) -> ModuleType:
    """Import one file as ``name``, registered before execution.

    Registered *before* ``exec_module`` because a package that imports its own
    submodules -- which every task loader does -- would otherwise re-enter this
    function and execute itself twice. Popped again on failure, so a task that
    raises leaves nothing half-built behind for the next lookup to find.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if is_package else None
    )
    if spec is None or spec.loader is None:  # pragma: no cover -- unreadable file
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def scan(root: Path) -> list[str]:
    """Execute every task directory in one root. Returns the task names found.

    Order is not load-bearing, and that is the point of a task carrying its own
    declarations: two tasks that stage the same cell each register it, the second
    registration is recognised as an identical copy, and neither depends on
    having been imported first.
    """
    root = Path(root)
    package = package_for_root(root)
    _namespace(PARENT_PACKAGE, [])
    _namespace(package, [str(root)])

    found: list[str] = []
    for child in sorted(root.iterdir()):
        if not roots.is_task_dir(child):
            continue
        found.append(child.name)
        loader = child / "task.py"
        try:
            if loader.is_file():
                _exec(f"{package}.{child.name}", loader, is_package=True)
            else:
                _generic(child, f"{package}.{child.name}")
        except Exception as exc:
            raise RuntimeError(
                f"task {child.name!r} in root {root} failed to load: {exc}. "
                f"Drop that root from {roots.ROOTS_ENV} to run without it."
            ) from exc
    return found


def _generic(directory: Path, package: str = "") -> None:
    """Register the suite a directory of documents describes, with no loader.

    The zero-Python path, and -- now that a document can say everything a loader
    used to -- the *only* path for a suite that ships none. In order:

    1. The suite's own Python, if any, is executed under the synthetic package:
       ``staging.py`` (its arrangements and rigs), ``scenes.py`` (its cell) and
       ``assets/`` (its records, entities and the factories that place them).
       A ``task.py`` used to do this by importing them; a directory with no
       ``task.py`` still needs them registered before a document that names
       them is lowered.
    2. ``<name>.suite``, if present, is read and every variant merged over it;
       its ``(:variants ...)`` fixes the order, headline first.
    3. Each variant is lowered with the shipped evaluator and an artifact
       resolver rooted at the package, so ``routine:X`` reaches ``routine.py``,
       at the start poses its staging names -- a recorded file, or the export's
       sidecar.
    4. Every entity the document typed is built through its record
       (:func:`~.place.build_entities`), and what the export's sidecar knows is
       applied (:func:`~.place.from_export`).
    """
    from robosimstudio.specs import TaskSuite
    from robosimstudio_bench.taskdoc import merge_suite, read_problem, read_suite

    from .artifacts import task_resolver
    from .registry import register_suite

    package = package or f"{package_for_root(directory.parent)}.{directory.name}"
    suite_files = sorted(directory.glob(f"*{roots.SUITE_SUFFIX}"))
    if len(suite_files) > 1:
        raise RuntimeError(f"{directory}: one .suite file per suite, found {[p.name for p in suite_files]}")
    suite_doc = read_suite(suite_files[0]) if suite_files else None
    if suite_doc is not None and suite_doc.name != directory.name:
        raise RuntimeError(
            f"{suite_files[0]}: declares suite {suite_doc.name!r} in a directory called {directory.name!r}"
        )
    _namespace(package, [str(directory)])
    for module in ("staging", "scenes", "records"):
        candidate = directory / f"{module}.py"
        if candidate.is_file():
            _exec(f"{package}.{module}", candidate, is_package=False)

    documents = _documents(directory, suite_doc)

    variants = []
    for name, path in documents.items():
        doc = merge_suite(read_problem(path), suite_doc)
        resolve = task_resolver(directory, package)
        variants.append(build_variant(doc, directory=directory, suite=directory.name, name=name, resolve=resolve))
    # ``DOCUMENTS`` and ``VARIANTS`` on the suite's module: what
    # ``scripts/gauntlet.py`` reads to find the document a variant came from.
    module = sys.modules[package]
    module.DOCUMENTS = documents
    module.VARIANTS = tuple(documents)
    register_suite(
        TaskSuite(
            name=directory.name,
            variants=tuple(variants),
            home=directory,
            description=suite_doc.language if suite_doc else "",
            provenance=" ".join(_text(v) for v in suite_doc.provenance.values()) if suite_doc else "",
        )
    )


def build_variant(doc, *, directory: Path, suite: str, name: str, resolve):
    """One merged document to the spec its suite registers.

    The per-variant half of :func:`_generic`, shared with
    :func:`~robosimstudio_bench.taskbuild.text.task` so a document read from a
    folder and one written as a string are lowered by the same lines: staged,
    lowered with the shipped evaluator, its entities built through their records,
    and what an export's sidecar knows applied.

    Args:
        doc: The document, already merged over its suite's.
        directory: Where its relative paths (recorded start poses) resolve.
        suite: The suite it is registered under.
        name: The variant's name.
        resolve: The artifact resolver for its ``"module:Name"`` references.
    """
    from robosimstudio_bench.goals.evaluate import evaluator_factory

    from .lower import export_keeps_free_bodies, lower
    from .place import build_entities, from_export, sidecar_poses
    from .staging import bind_profile, stage

    staging = stage(doc.staging).staging
    poses = _poses(directory, staging, sidecar_poses, resolve)
    lowered = lower(
        bind_profile(doc),
        suite=suite,
        name=name,
        make_evaluator=lambda d, task=f"{suite}/{name}": evaluator_factory(d, task=task),
        resolve_artifact=resolve,
        poses=poses,
    )
    spec = build_entities(lowered.spec, poses=poses, resident=export_keeps_free_bodies(staging))
    return from_export(spec, staging)


def _documents(directory: Path, suite_doc) -> dict[str, Path]:
    """The documents a directory holds, in the order its ``.suite`` fixes when it fixes one."""
    documents = {path.stem: path for path in sorted(directory.glob(f"*{roots.TASK_SUFFIX}"))}
    if suite_doc is not None and suite_doc.variants:
        listed, present = set(suite_doc.variants), set(documents)
        if listed != present:
            raise RuntimeError(
                f"{directory}: (:variants ...) names {sorted(listed - present)} with no document "
                f"and omits {sorted(present - listed)}"
            )
        documents = {name: documents[name] for name in suite_doc.variants}
    return documents


def _loader_for_home(home: Path | None) -> ModuleType:
    """The synthetic module a suite directory was executed as, found by the directory.

    Not by root order: a base suite may have come from a root nothing declared
    (a test's scratch root scanned by hand), and the directory is the one fact a
    registered suite records about where it came from.
    """
    if home is not None:
        wanted = str(Path(home).resolve())
        for name, module in list(sys.modules.items()):
            if name.startswith(f"{PARENT_PACKAGE}.") and wanted in [
                str(Path(p).resolve()) for p in getattr(module, "__path__", [])
            ]:
                return module
    raise LookupError(f"no loaded suite was executed from {home}")


def _text(value) -> str:
    """A suite clause's value as prose: a quoted string is one token, an unquoted sentence is many."""
    return " ".join(_text(v) for v in value) if isinstance(value, list) else str(value)


def _poses(directory: Path, staging, sidecar_poses, resolve):
    """Where the free bodies start: a recorded file or artifact the staging names, else the export's sidecar, else nowhere.

    ``(:start-poses poses/bench.json)`` is a recorded table; ``(:start-poses
    assets.layout:POSES)`` is the same table stated as the suite's own
    arithmetic -- a dict, or a zero-argument callable returning one.
    """
    if staging.start_poses and ":" in staging.start_poses:
        table = resolve(staging.start_poses)
        table = table() if callable(table) else table
        return {name: dict(entry) for name, entry in dict(table.get("objects", table)).items()}
    if staging.start_poses:
        recorded = json.loads((directory / staging.start_poses).read_text())
        return {name: dict(entry) for name, entry in recorded["objects"].items()}
    if staging.export:
        return sidecar_poses(staging.export, collider=staging.collider)
    return None


def load_roots() -> None:
    """Execute every declared root once. Safe to call from a registry lookup.

    Re-entrant calls are no-ops, and that is load-bearing for *multi-root*
    setups: a task loader's own registry lookups fire this again mid-scan, and
    the outer pass has already marked the root it is inside as loaded — so a
    nested pass would skip the half-scanned builtin root and scan *later*
    roots early, handing an external root a registry its loader can watch
    being empty. With one root the nested call always was a no-op (the root
    was marked); this makes that the contract rather than an accident of
    counting.
    """
    global _SCANNING  # noqa: PLW0603 -- the one re-entrancy latch a scan holds, module-wide by design
    if _SCANNING:
        return
    _SCANNING = True
    try:
        for root in roots.task_roots():
            key = str(root.resolve())
            if key in _LOADED:
                continue
            _LOADED.add(key)
            try:
                scan(root)
            except BaseException:
                _LOADED.discard(key)
                raise
    finally:
        _SCANNING = False


def task_loader(suite: str) -> ModuleType:
    """The Python module a suite's variants were built by.

    A task is a directory executed under a synthetic package name, so there is no
    importable name to write -- that is what stops two roots each holding a
    ``cable_route`` from shadowing one another. Whoever wants a task's own
    measured constants asks for them by *suite* instead, and this resolves it.

    This is here rather than beside the tests that first needed it because
    ``scripts/`` needs it too and may not import from ``tests/``.

    Args:
        suite: The registry name, e.g. ``"tool_sort"``.

    Returns:
        The loader module -- a task directory's ``task.py``.

    Raises:
        LookupError: If the suite ships no loader module, which a documents-only
            task legitimately does not. Nothing to import is a different fact
            from failing to import, so it is a different exception.
    """
    load_roots()
    for root in roots.task_roots():
        name = f"{package_for_root(root)}.{suite}"
        if name in sys.modules:
            return sys.modules[name]
    raise LookupError(f"suite {suite!r} ships no loader module")


def task_module(suite: str, name: str) -> ModuleType:
    """One module inside a task directory, e.g. ``task_module("tool_sort", "records")``."""
    return importlib.import_module(f"{task_loader(suite).__name__}.{name}")
