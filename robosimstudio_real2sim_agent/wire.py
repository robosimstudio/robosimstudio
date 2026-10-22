# SPDX-License-Identifier: Apache-2.0
"""The line the live page and its command line speak, and nothing that needs the library.

One JSON object in, one JSON object out, over a Unix socket -- and the one
line the server prints when the page is up, which the client that started it
detached (``ctl up``) and the test that drives it both read back. Both ends
import this module and the client imports nothing else of the package, so
``python -m robosimstudio_real2sim_agent status`` answers in milliseconds without
loading numpy, viser or the library. That is why :func:`encode` spells
numpy's ``tolist`` by attribute rather than by type.

Threads are the server's problem, not the protocol's: :class:`ControlSocket`
accepts connections on a thread of its own and hands each request to a
handler, and the server's handler marshals the call onto the simulator's
thread through ``SimBus``. Nothing here has seen a simulator.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "DEFAULT_RUN_DIR",
    "READY_PREFIX",
    "SOCKET_NAME",
    "ControlSocket",
    "NoReply",
    "NoServer",
    "encode",
    "parse_ready",
    "ready_line",
    "request",
    "socket_path",
]

DEFAULT_RUN_DIR = "run"
"""Where a page keeps its socket, its log and its snapshots: under the directory it was started from."""
SOCKET_NAME = "ctl.sock"
READY_PREFIX = "[live] page "
"""The start of the one line the server prints once the page is being served."""


class NoServer(OSError):
    """No page is listening on the socket named."""


class NoReply(TimeoutError):
    """A page is listening but did not answer within the timeout."""


SUN_PATH_MAX = 100
"""Bytes a Unix socket path may have on Linux (108, less the terminator and a margin)."""


def socket_path(run_dir: str | os.PathLike | None, sock: str | os.PathLike | None = None) -> Path:
    """Where the socket is: an explicit ``sock`` wins, else ``<run_dir>/ctl.sock``.

    A run directory deep enough that the socket's path would not fit a
    ``sockaddr_un`` -- a worktree under a session's scratch directory did it --
    is given a short path in the temp directory instead, named by a hash of
    the run directory so the server and every client resolve the same one.
    """
    if sock:
        return Path(sock)
    path = Path(run_dir or DEFAULT_RUN_DIR) / SOCKET_NAME
    full = path.resolve()
    if len(os.fsencode(full)) <= SUN_PATH_MAX:  # resolved, so every spelling of one run dir chooses alike
        return path if len(os.fsencode(path)) <= SUN_PATH_MAX else full  # the path returned is one that fits
    import hashlib
    import tempfile

    digest = hashlib.sha1(os.fsencode(full)).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"rss-live-{digest}.sock"


def ready_line(url: str, sock: Path | str, scene: Path | str, share: str | None = None) -> str:
    """The line the server prints once the page is up; :func:`parse_ready` reads it back.

    Fields are two spaces apart, ``key value``, so a path with one space in it
    survives the trip.
    """
    line = f"{READY_PREFIX}{url}  socket {sock}  scene {scene}"
    if share:
        line += f"  share {share}"
    return line


def parse_ready(text: str) -> dict | None:
    """The last ready line in ``text`` as ``{"url", "sock", "scene", "share"}``, or None."""
    found = None
    for line in text.splitlines():
        if not line.startswith(READY_PREFIX):
            continue
        head, *pairs = line[len(READY_PREFIX) :].split("  ")
        fields: dict = {"url": head.strip(), "sock": "", "scene": "", "share": None}
        for pair in pairs:
            key, _, value = pair.strip().partition(" ")
            fields["sock" if key == "socket" else key] = value
        found = fields
    return found


def encode(reply: dict) -> bytes:
    """One JSON line. Arrays and numpy scalars go through ``tolist``, anything else through ``str``."""

    def default(value):
        tolist = getattr(value, "tolist", None)
        return tolist() if callable(tolist) else str(value)

    return (json.dumps(reply, default=default) + "\n").encode()


def request(sock: Path | str, req: dict, timeout: float) -> dict:
    """Send one request and return the reply. Raises :class:`NoServer` or :class:`NoReply`."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(sock))
    except (BlockingIOError, TimeoutError) as exc:  # a full backlog: a server frozen, not gone
        client.close()
        raise NoReply(f"the server at {sock} is not accepting connections") from exc
    except OSError as exc:
        client.close()
        raise NoServer(f"no server at {sock} ({exc.strerror or exc})") from exc
    with client:
        try:
            stream = client.makefile("rwb")
            stream.write(encode(req))
            stream.flush()
            line = stream.readline()
        except TimeoutError as exc:
            raise NoReply(f"no reply within {timeout:.0f} s") from exc
        except OSError as exc:
            raise NoReply(f"the server dropped the connection ({exc.strerror or exc})") from exc
    if not line:
        raise NoReply("the server closed the connection without answering")
    return json.loads(line)


class ControlSocket:
    """A Unix socket that answers one JSON line per connection with what ``handler`` returns.

    :meth:`serve` blocks; :meth:`start` binds on the caller's thread -- so a
    socket that cannot be had is the caller's exception, raised before it says
    the page is up -- and accepts on a daemon thread. A socket file some live
    server is answering on is refused rather than taken over; one nobody
    answers on is a stale leftover and is replaced. Accepts time out every
    half second so :meth:`close` is honoured promptly. A line that is not a
    JSON object is answered with an error rather than dropped, and a handler
    that raises is reported the same way: the client always gets a line back.
    """

    def __init__(self, path: Path | str, handler: Callable[[dict], dict]) -> None:
        self.path = Path(path)
        self.handler = handler
        self._stop = threading.Event()
        self._bound = False
        """Whether the file at :attr:`path` is ours to unlink."""
        self.thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        server = self.bind()
        self.thread = threading.Thread(target=self._accept, args=(server,), name="live-socket", daemon=True)
        self.thread.start()
        return self.thread

    def serve(self) -> None:
        self._accept(self.bind())

    def bind(self) -> socket.socket:
        """Take the socket path and listen on it. Raises if a live server holds it, or on any bind failure."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.path.is_socket():
            raise FileExistsError(f"{self.path} exists and is not a socket; name another --sock")
        if self.path.is_socket():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(1.0)
            try:
                probe.connect(str(self.path))
            except OSError:
                self._unlink()  # nobody behind it: a server that died without closing
            else:
                raise FileExistsError(f"a server is already listening at {self.path}; `quit` it first")
            finally:
                probe.close()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.path))
            server.listen(8)
        except OSError:
            server.close()
            raise
        self._bound = True
        server.settimeout(0.5)
        return server

    def _accept(self, server: socket.socket) -> None:
        with server:
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            stream = conn.makefile("rwb")
            line = stream.readline()
            if not line:
                return
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("a request is a JSON object")
            except ValueError as exc:
                reply = {"ok": False, "error": f"bad request: {exc}"}
            else:
                try:
                    reply = self.handler(request)
                except Exception as exc:
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                stream.write(encode(reply))
                stream.flush()
            except OSError:
                pass  # the client stopped waiting before a command it started ended: nobody to tell

    def close(self) -> None:
        self._stop.set()
        if self._bound:  # never another server's: a socket this one was refused is left alone
            self._bound = False
            self._unlink()

    def _unlink(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
