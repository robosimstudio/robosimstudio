# SPDX-License-Identifier: Apache-2.0
"""The contract for the file under the page, and the one function that holds it.

When the file has run, ``scene`` is an ``rss.Scene``. That is all. The server
builds it with ``scene.build(control_mode="joint_position")`` -- or takes the
``sim`` the file bound, if it bound one -- and a ``BUILD = {...}`` dict is
passed to ``build`` as keyword arguments. ``rss.view``, ``rss.draw`` and
``rss.viewer`` are no-ops that return None while the file runs, so a script
pasted from ``examples/`` that calls ``rss.view`` or ``rss.draw`` works
unchanged (one that drives the viewer ``rss.viewer`` returns does not:
``viewer.scene`` is an attribute of None): the page is this process's, and a file
that opened a second one would block on viser's tunnel. The file runs as
``scene``, not ``__main__``, so a ``__main__`` guard does not fire.

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
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DOORS", "LIBRARY", "Loaded", "SceneRegistrations", "fallback_scene", "run_scene_file", "stamp"]

LIBRARY = (
    "robosimstudio",
    "robosimstudio_materials",
    "robosimstudio_studio",
    "robosimstudio_bench",
    "robosimstudio_policy_agent",
    "robosimstudio_real2sim_agent",
)
"""The packages whose code is never the scene's own, wherever the scene file sits."""

DOORS = ("view", "draw", "viewer")
"""The ``rss.`` names that would open a page of their own, held shut while the file runs."""


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


def fallback_scene():
    """The scene served when the file does not build at start-up: a bare floor and one robot."""
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="live-fallback")
    scene.add(rss.robot("yam_i2rt"))  # a scene needs something that moves before it will build
    return scene


def _package_dirs(names: tuple[str, ...]) -> tuple[Path, ...]:
    dirs = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        for location in (spec.submodule_search_locations or []) if spec else []:
            dirs.append(Path(location).resolve())
    return tuple(dirs)


@dataclass
class SceneRegistrations:
    """Every registry write the scene file, a helper beside it or an ``eval`` made, to take back before a rebuild.

    Installed with :func:`robosimstudio.registry.watch`. A write is the scene's
    when the first frame above the registry that is not library code is the
    scene file, a module under its directory, or ``<eval>``. Two kinds of
    write stay: a library module registering as it is first imported -- even
    when the scene's import set it off -- which the import machinery's frame
    stops the walk at; and a registry's discovery, run by a lookup the scene
    made (``rss.robot`` loading the embodiments, the task roots), which
    :meth:`Registry.load <robosimstudio.registry.Registry.load>`'s frame stops
    it at. Either, taken back, would never be registered again: the module is
    imported and the registry loaded.
    """

    root: Path
    writes: list[tuple[Any, ...]] = field(default_factory=list)
    """``(registry, name, value, origin, previous, previous_origin)`` per write, oldest first."""
    _library: tuple[Path, ...] = ()
    _load_code: Any = None

    def __post_init__(self) -> None:
        from robosimstudio.registry import Registry

        self.root = Path(self.root).resolve()
        self._library = _package_dirs(LIBRARY)
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
            path = Path(filename).resolve()
            if not any(path.is_relative_to(lib) for lib in self._library):
                return path.is_relative_to(self.root) and "site-packages" not in path.parts
            frame = frame.f_back
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
