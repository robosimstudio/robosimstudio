# SPDX-License-Identifier: Apache-2.0
"""The seam a coding agent drives: the studio held open over a file, and a command line onto it.

A page (:mod:`robosimstudio_studio`) that stays open while the scene under it
is replaced -- ``scene.py`` is executed on every save, the ``rss.Scene`` it
binds is built and swapped in, the browser never reloads -- and a Unix socket
onto that process with a standard-library client: status, a snapshot beside a
reference picture, stepping, python on the simulator's thread. It is what an
agent asked to rebuild a real bench from a photograph works through: it edits
the file, and the page and its snapshots show what it built.

The *coding* agent, outside the process, editing a file -- not the policy
agent inside an episode, which asks ``sim.query``. Those are different programs with different questions, and
this package has nothing to say to the second.

Beside the library and above the page: it names ``robosimstudio_studio`` and
``robosimstudio`` at module scope and nothing names it. The client
(:mod:`.ctl`, :mod:`.wire`) imports neither, so ``python -m
robosimstudio_real2sim_agent status`` answers in milliseconds; the server side
(:mod:`.server`, :mod:`.page`, :mod:`.serve`) is reached lazily from here for
the same reason. ``examples/live/REFERENCE.md`` is the contract and
the procedure; ``examples/live/`` is the working directory it was set up for.
"""

from __future__ import annotations

import importlib

from .wire import READY_PREFIX, socket_path

__all__ = ["READY_PREFIX", "LiveServer", "LiveSession", "Loaded", "run_scene_file", "socket_path"]

_LAZY = {
    "LiveServer": ".server",
    "LiveSession": ".page",
    "Loaded": ".scenefile",
    "run_scene_file": ".scenefile",
}
"""The server side, resolved on first use so importing the package costs nothing the client does not need."""


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
