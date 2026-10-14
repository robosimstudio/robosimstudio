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
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DOORS", "Loaded", "fallback_scene", "run_scene_file", "stamp"]

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
