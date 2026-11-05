# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the live page's server, for the client's half and the server's bookkeeping.

``ctl`` talks to a page over a Unix socket and waits on what ``status`` says.
:class:`FakePage` is a real :class:`~robosimstudio_real2sim_agent.wire.ControlSocket`
answering ``status`` from a dict the test changes; :class:`FakeStatus` answers
from a script of replies, the last one forever. Neither builds a scene: the
client is standard library only, and what it is tested on is the words.

:func:`bare_live_server` is the server's half: a real
:class:`~robosimstudio_real2sim_agent.server.LiveServer`, built by its own
constructor over a scene folder whose file is never run, with a stand-in sim
and session. Five test modules each set twenty-odd private fields on a
``LiveServer.__new__`` by hand, and each copy had drifted from the constructor
it stood in for; the constructor itself costs a ``mkdir`` and cannot drift.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from robosimstudio_real2sim_agent import wire

__all__ = ["FakePage", "FakeStatus", "bare_live_server"]


def bare_live_server(
    root: Path | str | None = None,
    *,
    sim: Any = None,
    session: Any = None,
    run_dir: Path | str | None = None,
    watch: bool = False,
    **attrs: Any,
):
    """A ``LiveServer`` over ``root/scene.py``, never loaded, nothing bound, with a stand-in sim and session.

    ``root`` defaults to a fresh temp directory, removed when the process
    exits; the run directory is ``root/run`` unless ``run_dir`` names another.
    The page does not watch its file unless asked (``watch=True``), and its
    log path is empty, whatever the test runner did with stdout. ``sim``
    defaults to a namespace at step 0 and ``session`` to one with no arms;
    every other keyword is set on the server as it stands.
    """
    from robosimstudio_real2sim_agent.server import LiveServer
    from robosimstudio_studio.session import StudioConfig

    if root is None:
        root = tempfile.mkdtemp(prefix="live-")
        atexit.register(shutil.rmtree, root, True)
    root = Path(root)
    live = LiveServer(
        scene_path=root / "scene.py",
        sock_path=root / "ctl.sock",
        run_dir=root / "run" if run_dir is None else run_dir,
        config=StudioConfig(),
        watch=watch,
    )
    live.log_path = ""
    live.sim = SimpleNamespace(step_count=0) if sim is None else sim
    live.session = SimpleNamespace(arms={}) if session is None else session
    for name, value in attrs.items():
        setattr(live, name, value)
    return live


class FakePage:
    """A socket answering ``status`` from a dict the test changes, for the client's half."""

    def __init__(self, path: Path, status: dict) -> None:
        self.status = dict(status)
        self.socket = wire.ControlSocket(path, lambda request: {"ok": True, **self.status})
        self.socket.start()

    def close(self) -> None:
        self.socket.close()


class FakeStatus:
    """A socket that answers ``status`` from a script of replies, the last one forever."""

    def __init__(self, path: Path, replies: list[dict]) -> None:
        self.replies = list(replies)
        self._lock = threading.Lock()
        """The socket answers each connection on a thread of its own: two polls must not take one reply."""
        self.socket = wire.ControlSocket(path, self._answer)
        self.socket.start()

    def _answer(self, request: dict) -> dict:
        with self._lock:
            return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def close(self) -> None:
        self.socket.close()
