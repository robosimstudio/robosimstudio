# SPDX-License-Identifier: Apache-2.0
"""The contract for the file under the page, and the one function that holds it.

When the file has run, ``scene`` is an ``rss.Scene``. That is all. The server
builds it with ``scene.build(control_mode="joint_position")`` -- or takes the
``sim`` the file bound, if it bound one -- and a ``BUILD = {...}`` dict is
passed to ``build`` as keyword arguments. ``rss.view``, ``rss.draw`` and
``rss.viewer`` are no-ops that return None while the file runs, so an example
whose ``main()`` body is moved to module level works with its ``rss.view`` or
``rss.draw`` calls left in (one that drives the viewer ``rss.viewer`` returns
does not: ``viewer.scene`` is an attribute of None): the page is this
process's, and a file that opened a second one would block on viser's tunnel.
The file runs as ``scene``, not ``__main__``, so a ``__main__`` guard does not
fire, and an example's ``main()`` is not called.

Why the doors can be swapped with ``setattr``: ``robosimstudio/__init__.py``
resolves its flat names through a :pep:`562` ``__getattr__`` that caches into
the module's globals, so an attribute set on the module shadows the export and
``from robosimstudio import view`` inside the file gets the no-op too. A file
that reaches past the front door -- ``from robosimstudio.sim.draw import
view`` -- gets the real thing, and that is its own choice.

``build`` is a parameter so the merge of ``BUILD`` over the default can be
tested with no GPU: the fast tier hands in a function that records what it was
called with, and a page hands in nothing.

What the file registers -- an entity class, a cell, a gripper, a scenery --
goes into process-wide registries that outlive the run, and a registry keeps
the first of two registrations with the same source and refuses two that
differ. So an edited class would be ignored on the next save and an edited
cell refused. :class:`SceneRegistrations` notes every registration the file or
a helper beside it makes, and takes them back before the next run.
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import sys
import sysconfig
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKED_HASH = py_compile.PycInvalidationMode.CHECKED_HASH

__all__ = [
    "DOORS",
    "WATCH_DEPTH",
    "WATCH_LIMIT",
    "WATCH_SKIP",
    "Loaded",
    "ScenePaths",
    "SceneRegistrations",
    "fallback_loaded",
    "fallback_scene",
    "hash_checked",
    "helper_files",
    "library_name",
    "run_scene_file",
    "stamp",
]

DOORS = ("view", "draw", "viewer")
"""The ``rss.`` names that would open a page of their own, held shut while the file runs."""
WATCH_SKIP = frozenset({"run", "ref", "assets", "twins", "__pycache__"})
"""Folders beside the scene file whose ``.py`` files are not the scene's: its readings and scratch scripts, its
references, its assets, the twins under it. Hidden folders and this library's packages are not the scene's either
(:func:`scene_folder`)."""
WATCH_DEPTH = 2
"""How many folders down from the scene file's a helper module is the scene's."""
WATCH_LIMIT = 500
"""The most ``.py`` files watched: a scene file at the top of a large tree watches the first this many."""


def library_name(name: str) -> bool:
    """Whether a folder, a module beside the scene file or a top-level package is named as this library's
    (``robosimstudio*``), and so is never the scene's: the one rule :class:`ScenePaths` and :func:`helper_files`
    both apply."""
    return name.startswith("robosimstudio")


def scene_folder(name: str) -> bool:
    """Whether a folder under the scene file's can hold the scene's own code: the one rule for which folders do,
    which :class:`ScenePaths` and :func:`helper_files` both apply."""
    return not name.startswith(".") and name not in WATCH_SKIP and not library_name(name)


@dataclass(frozen=True)
class Loaded:
    """What running the file produced: its namespace, the scene, the built sim, and how long it took."""

    ns: dict
    scene: Any
    sim: Any
    seconds: float


def _noop(*args, **kwargs) -> None:
    return None


def _build(scene, **kwargs):
    return scene.build(**kwargs)


def run_scene_file(path: Path | str, *, build: Callable[..., Any] | None = None) -> Loaded:
    """Execute the file and build what it declares. Raises on any failure, with the doors restored."""
    import robosimstudio as rss

    path = Path(path)
    started = time.perf_counter()
    code = compile(path.read_text(), str(path), "exec")
    ns: dict = {"__name__": "scene", "__file__": str(path)}
    doors = {name: getattr(rss, name) for name in DOORS}
    try:
        for name in doors:
            setattr(rss, name, _noop)
        exec(code, ns)
    finally:
        for name, door in doors.items():
            setattr(rss, name, door)
    scene = ns.get("scene")
    if not isinstance(scene, rss.Scene):
        raise TypeError(f"{path.name} must leave `scene` bound to an rss.Scene; it is {type(scene).__name__}")
    sim = ns.get("sim")
    if sim is None:
        kwargs: dict = {"control_mode": "joint_position"}
        kwargs.update(ns.get("BUILD") or {})
        sim = (build or _build)(scene, **kwargs)
    elif not isinstance(sim, rss.Sim):
        raise TypeError(
            f"{path.name} bound `sim` to {type(sim).__name__}; it must be an rss.Sim built from `scene`, or left unbound"
        )
    return Loaded(ns=ns, scene=scene, sim=sim, seconds=time.perf_counter() - started)


def stamp(path: Path | str) -> tuple[int, int] | None:
    """What the watcher compares: mtime and size, or None for a file that is not there."""
    try:
        st = Path(path).stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size)


def helper_files(path: Path | str) -> list[Path]:
    """The ``.py`` files that are the scene's besides the file itself, in a fixed order: every one under its folder,
    down :data:`WATCH_DEPTH` folders, but in :data:`WATCH_SKIP`, hidden or this library's; at most
    :data:`WATCH_LIMIT`. Every one, imported or not: one that failed to import, or that the file imports later, is
    the scene's all the same. A save to one rebuilds, and the log's scene hash covers them."""
    path = Path(path).resolve()
    found: list[Path] = []
    level = [path.parent]
    for _ in range(WATCH_DEPTH + 1):
        below = []
        for folder in level:
            try:
                entries = sorted(os.scandir(folder), key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries:
                name = entry.name
                if entry.is_dir(follow_symlinks=False):
                    if scene_folder(name):
                        below.append(Path(entry.path))
                elif (
                    name.endswith(".py")
                    and not name.startswith(".")
                    # beside the scene file, a module named so is the library's (ScenePaths: its top-level name)
                    and not (folder == path.parent and library_name(name[:-3]))
                    and Path(entry.path) != path
                ):
                    found.append(Path(entry.path))
                    if len(found) >= WATCH_LIMIT:
                        return found
        level = below
    return found


def hash_checked(path: Path | str) -> None:
    """Make a module's cached bytecode hash-checked, once: the import then checks it against the source's text, not
    its whole-second mtime and size, and writes it so again when it does not match. A file with no bytecode yet is
    left alone -- nothing stale can be read, and nothing is written into the folder the person zips; its first
    import writes one, made hash-checked before the import after. A file that will not compile, or a cache that
    cannot be written, is left to the import."""
    path = str(path)
    try:
        cached = importlib.util.cache_from_source(path)
        with open(cached, "rb") as f:
            head = f.read(8)
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        head = b""
    if len(head) == 8 and int.from_bytes(head[4:8], "little") & 0b11 == 0b11:
        return  # hash-based, and checked
    try:
        if py_compile.compile(path, doraise=False, quiet=2, invalidation_mode=CHECKED_HASH) is None:
            Path(importlib.util.cache_from_source(path)).unlink(missing_ok=True)  # will not compile: no old code stays
    except (OSError, ValueError):
        pass


def fallback_scene():
    """The bare scene that stands in for the file's: a floor and one robot. Served when the file does not build at
    start-up, and when a rebuilt scene would not go into the page."""
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="live-fallback")
    scene.add(rss.robot("yam_i2rt"))  # a scene needs something that moves before it will build
    return scene


def fallback_loaded() -> Loaded:
    """:func:`fallback_scene`, built, as a run of the file would have left it."""
    scene = fallback_scene()
    return Loaded(ns={}, scene=scene, sim=scene.build(control_mode="joint_position"), seconds=0.0)


class ScenePaths:
    """What is the scene's own code: one rule, for the modules a rebuild imports afresh and the registrations it
    takes back, so the two never disagree.

    A module is the scene's when its file is under the scene file's directory,
    at most :data:`WATCH_DEPTH` folders down and in no folder :func:`scene_folder`
    rules out -- the files :func:`helper_files` watches -- and is not an
    installed package's, and the module -- by name, its package by its
    top-level name, and its file -- was not imported before the scene file
    first ran; and its top-level package is not this library's
    (``robosimstudio_materials``, say, imported for the first time by a scene
    file at the repository root, stays the library's). Verdicts
    are kept per name and file: a registry's discovery asks about thousands of
    frames.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._installed = tuple(
            Path(p).resolve() for key in ("purelib", "platlib") if (p := sysconfig.get_paths().get(key))
        )
        self._baseline: set[Path] = set()
        self._baseline_names: set[str] = set()
        self._verdicts: dict[tuple[str, str], bool] = {}

    def set_baseline(self, module_names: set[str]) -> None:
        """The modules imported before the scene file first ran: never the scene's, wherever they are."""
        files = set()
        for name in module_names:
            file = getattr(sys.modules.get(name), "__file__", None)
            if file:
                files.add(Path(file).resolve())
        self._baseline, self._baseline_names = files, set(module_names)
        self._verdicts.clear()

    def is_scene(self, filename: str, module: str = "") -> bool:
        """Whether the module ``module`` (its ``__name__``), at ``filename``, is the scene's own code."""
        key = (module, filename)
        verdict = self._verdicts.get(key)
        if verdict is None:
            path = Path(filename).resolve()
            top = module.partition(".")[0]
            folders = path.relative_to(self.root).parts[:-1] if path.is_relative_to(self.root) else None
            verdict = (
                module not in self._baseline_names
                and top not in self._baseline_names
                and folders is not None
                and len(folders) <= WATCH_DEPTH
                and all(scene_folder(name) for name in folders)
                and path not in self._baseline
                and not any(part in ("site-packages", "dist-packages") for part in path.parts)
                and not any(path.is_relative_to(lib) for lib in self._installed)
                and not library_name(top)
            )
            self._verdicts[key] = verdict
        return verdict


@dataclass(eq=False)  # a watcher is found and removed by identity, never by its notes
class SceneRegistrations:
    """Every registry write the scene file, a helper beside it or an ``eval`` made, to take back before a rebuild.

    Installed with :func:`robosimstudio.registry.watch`. A write is the scene's
    when a frame above the registry is the scene's own code (:class:`ScenePaths`)
    or ``<eval>``, before the walk meets one of two stops; the writes behind
    those stay: a library module registering as it is first imported -- even
    when the scene's import set it off -- which the import machinery's frame
    stops the walk at; and a registry's discovery, run by a lookup the scene
    made (``rss.robot`` loading the embodiments, the task roots), which
    :meth:`Registry.load <robosimstudio.registry.Registry.load>`'s frame stops
    it at. Either, taken back, would never be registered again: the module is
    imported and the registry loaded.
    """

    paths: ScenePaths
    writes: list[tuple[Any, ...]] = field(default_factory=list)
    """``(registry, name, value, origin, previous, previous_origin)`` per write, oldest first."""
    _load_code: Any = None

    def __post_init__(self) -> None:
        from robosimstudio.registry import Registry

        if not isinstance(self.paths, ScenePaths):
            self.paths = ScenePaths(self.paths)  # a directory, as a test or a caller may give
        self._load_code = Registry.load.__code__

    def __call__(self, registry: Any, name: str, value: Any, origin: str, previous: Any, previous_origin: str) -> None:
        if self._made_by_scene(sys._getframe(1)):
            self.writes.append((registry, name, value, origin, previous, previous_origin))

    def _made_by_scene(self, frame: Any) -> bool:
        while frame is not None:
            if frame.f_code is self._load_code:
                return False  # a registry's discovery, whoever's lookup set it off
            filename = frame.f_code.co_filename
            if filename == "<eval>":
                return True
            if filename.startswith("<"):
                return False  # the import machinery: a module registering as it loads
            if self.paths.is_scene(filename, frame.f_globals.get("__name__", "")):
                return True
            frame = frame.f_back  # library code between the registry and whoever called it
        return False

    def undo(self) -> list[tuple[Any, ...]]:
        """Take back every noted write, newest first, and forget them; returns them, for :meth:`redo`."""
        from robosimstudio.registry import MISSING

        taken = list(self.writes)
        for registry, name, _value, _origin, previous, previous_origin in reversed(taken):
            if previous is MISSING:
                registry.forget(name)
            else:
                registry.restore(name, previous, previous_origin)
        self.writes.clear()
        return taken

    def redo(self, taken: list[tuple[Any, ...]]) -> None:
        """Put back writes :meth:`undo` took, oldest first, and note them again: a run that failed gives way."""
        for registry, name, value, origin, _previous, _previous_origin in taken:
            registry.restore(name, value, origin)
        self.writes[:0] = taken
