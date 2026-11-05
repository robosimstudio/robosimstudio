# SPDX-License-Identifier: Apache-2.0
"""The live page's socket: where it is, what it binds, and one line in for one line out.

A run directory names one socket path however it is spelled, and a deep one
still gets a path that fits ``sockaddr_un``. A live socket is never taken over
and a stale one is; a regular file at the path is never deleted. The client
says NoServer or NoReply, never a traceback, whatever the other end does.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio_real2sim_agent import wire
from tests.fixtures.sockets import chdir, short_dir

TIER = "fast"
"""No GPU and no page: sockets on temp paths, answered by plain functions."""


class TestTheWire(unittest.TestCase):
    def test_where_the_socket_is(self) -> None:
        with short_dir("w-") as here, chdir(here):  # `run/ctl.sock` is resolved against the cwd, so the cwd is short
            self.assertEqual(wire.socket_path(None), Path("run") / "ctl.sock")
            self.assertEqual(wire.socket_path("d"), Path("d") / "ctl.sock")
        deep = Path(tempfile.gettempdir()) / ("d" * 60) / ("e" * 60) / "run"
        short = wire.socket_path(deep)
        self.assertLessEqual(
            len(str(short)),
            wire.SUN_PATH_MAX,
            "a run directory too deep for sockaddr_un gets a short path",
        )
        self.assertEqual(
            short,
            wire.socket_path(deep),
            "the same one every time, so the server and its clients agree",
        )
        self.assertNotEqual(
            short,
            wire.socket_path(deep.with_name("run2")),
            "and a different one per run directory",
        )
        self.assertEqual(wire.socket_path("d", "s.sock"), Path("s.sock"), "an explicit socket wins")

    def test_the_ready_line_reads_back(self) -> None:
        line = wire.ready_line("http://localhost:8080", "/r/ctl.sock", "/r/scene.py")
        self.assertTrue(line.startswith(wire.READY_PREFIX))
        parsed = wire.parse_ready("noise\n" + line + "\nmore noise\n")
        self.assertEqual(
            parsed,
            {
                "url": "http://localhost:8080",
                "sock": "/r/ctl.sock",
                "scene": "/r/scene.py",
                "share": None,
            },
        )
        shared = wire.parse_ready(wire.ready_line("http://localhost:1", "/s", "/f", "https://x.y/z"))
        self.assertEqual(shared["share"], "https://x.y/z")
        self.assertIsNone(wire.parse_ready("nothing here"))

    def test_encode_spells_tolist_by_attribute(self) -> None:
        reply = json.loads(wire.encode({"a": np.arange(3), "b": np.float32(1.5), "c": Path("p")}))
        self.assertEqual(reply, {"a": [0, 1, 2], "b": 1.5, "c": "p"})

    def test_one_line_in_one_line_out(self) -> None:
        tmp = self.enterContext(short_dir("w-"))
        path = tmp / "ctl.sock"

        def handler(request: dict) -> dict:
            if request.get("cmd") == "boom":
                raise ValueError("no")
            return {"ok": True, "echo": request}

        server = wire.ControlSocket(path, handler)
        server.start()
        deadline = time.time() + 5.0
        while not path.exists() and time.time() < deadline:
            time.sleep(0.05)
        try:
            self.assertEqual(
                wire.request(path, {"cmd": "status", "n": 1}, 5.0),
                {"ok": True, "echo": {"cmd": "status", "n": 1}},
            )
            failed = wire.request(path, {"cmd": "boom"}, 5.0)
            self.assertFalse(failed["ok"])
            self.assertIn("ValueError: no", failed["error"])
            for raw in (b"not json\n", b"[1, 2]\n"):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(5.0)
                    client.connect(str(path))
                    client.sendall(raw)
                    reply = json.loads(client.makefile("rb").readline())
                self.assertFalse(reply["ok"])
                self.assertIn("bad request", reply["error"])
        finally:
            server.close()
        self.assertFalse(path.exists(), "close unlinks the socket")
        with self.assertRaises(wire.NoServer):
            wire.request(path, {"cmd": "status"}, 1.0)

    def test_a_live_socket_is_refused_and_a_stale_one_replaced(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        first = wire.ControlSocket(path, lambda request: {"ok": True, "who": "first"})
        first.start()
        try:
            second = wire.ControlSocket(path, lambda request: {"ok": True, "who": "second"})
            with self.assertRaises(FileExistsError, msg="a running server's socket is not taken over"):
                second.start()
            second.close()
            self.assertEqual(
                wire.request(path, {"cmd": "status"}, 5.0)["who"],
                "first",
                "and the refused one did not unlink it on the way out",
            )
        finally:
            first.close()
        # A socket file nobody answers on: what a server killed with -9 leaves.
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()
        self.assertTrue(path.exists())
        third = wire.ControlSocket(path, lambda request: {"ok": True, "who": "third"})
        third.start()
        try:
            self.assertEqual(wire.request(path, {"cmd": "status"}, 5.0)["who"], "third")
        finally:
            third.close()

    def test_a_bind_failure_is_the_callers(self) -> None:
        """Raised by ``start`` on the caller's thread, so the server never prints a ready line over it."""
        path = self.enterContext(short_dir("w-")) / ("s" * 200 + ".sock")
        with self.assertRaises(OSError):
            wire.ControlSocket(path, lambda request: {"ok": True}).start()


class TestTheSocketPath(unittest.TestCase):
    def test_every_spelling_of_one_run_dir_agrees(self) -> None:
        deep = self.enterContext(short_dir("w-")) / ("d" * 60) / ("e" * 40)
        deep.mkdir(parents=True)
        with chdir(deep):
            self.assertEqual(wire.socket_path("run"), wire.socket_path(deep / "run"))
            self.assertEqual(wire.socket_path("run"), wire.socket_path(os.path.join(".", "run")))

    def test_a_long_spelling_of_a_short_run_dir_comes_back_short(self) -> None:
        short = self.enterContext(short_dir("w-")).resolve()
        padded = os.path.join(*([".."] * 40), *short.parts[1:], "run")  # from the cwd, past / and back down
        self.assertGreater(len(os.fsencode(padded)), wire.SUN_PATH_MAX)
        path = wire.socket_path(padded)
        self.assertLessEqual(len(os.fsencode(path)), wire.SUN_PATH_MAX, "the path returned can be bound")
        self.assertEqual(path, wire.socket_path(short / "run"))


DEEP_TEMP = "/tmp/" + "t" * 150
"""A session's temp dir as deep as an agent's scratch directory is: a socket's name alone does not fit under it."""


class TestASocketThatFitsWhateverTheTempDir(unittest.TestCase):
    """A run directory too deep for its own socket gets a short one, chosen by length alone.

    The temp directory when the path fits there -- private where ``TMPDIR`` is --
    else ``/tmp``, else a refusal naming ``--sock``. Never by whether a directory
    can be written: a client only connects, and must resolve what the server bound.
    """

    def deep_run_dir(self) -> Path:
        return self.enterContext(short_dir("w-")) / ("d" * 60) / ("e" * 60) / "run"

    def test_a_deep_run_dir_under_a_deep_temp_dir_still_gets_a_path_that_fits(self) -> None:
        with mock.patch.object(tempfile, "tempdir", DEEP_TEMP):
            path = wire.socket_path(self.deep_run_dir())
        self.assertLessEqual(len(os.fsencode(path)), wire.SUN_PATH_MAX, f"{path} cannot be bound")
        self.assertEqual(Path("/tmp"), path.parent, "a temp dir too deep falls to /tmp")

    def test_a_temp_dir_that_fits_keeps_the_socket(self) -> None:
        """Where it always was, and private where ``TMPDIR`` is, though ``/tmp`` would fit too."""
        private = self.enterContext(short_dir("own-"))
        with mock.patch.object(tempfile, "tempdir", str(private)):
            path = wire.socket_path(self.deep_run_dir())
        self.assertEqual(private, path.parent)

    def test_the_server_and_its_clients_in_one_environment_agree(self) -> None:
        deep = self.deep_run_dir()
        deep.parent.mkdir(parents=True)
        for temp in (DEEP_TEMP, str(self.enterContext(short_dir("own-")))):
            with self.subTest(temp=temp[:20]), mock.patch.object(tempfile, "tempdir", temp):
                server = wire.socket_path(deep)
                with chdir(deep.parent):
                    client = wire.socket_path("run")  # the same run dir, spelled from where the client stands
                self.assertEqual(server, client)

    def test_where_the_socket_goes_does_not_ask_what_can_be_written(self) -> None:
        """A client only connects: one that cannot write ``/tmp`` must still find the server's socket there."""
        with (
            mock.patch.object(tempfile, "tempdir", DEEP_TEMP),
            mock.patch.object(os, "access", lambda *a, **k: False),
        ):
            path = wire.socket_path(self.deep_run_dir())
        self.assertEqual(Path("/tmp"), path.parent)

    def test_nowhere_short_enough_is_refused_saying_how_to_name_one(self) -> None:
        with (
            mock.patch.object(tempfile, "tempdir", DEEP_TEMP),
            mock.patch.object(wire, "SOCKET_DIRS", (DEEP_TEMP,), create=True),
            self.assertRaisesRegex(OSError, "--sock"),
        ):
            wire.socket_path(self.deep_run_dir())

    def test_ctl_says_so_in_one_line(self) -> None:
        from robosimstudio_real2sim_agent import ctl

        err = io.StringIO()
        with (
            mock.patch.object(tempfile, "tempdir", DEEP_TEMP),
            mock.patch.object(wire, "SOCKET_DIRS", (DEEP_TEMP,), create=True),
            contextlib.redirect_stderr(err),
        ):
            code = ctl.main(["--run-dir", str(self.deep_run_dir()), "status"])
        self.assertEqual(2, code)
        self.assertTrue(err.getvalue().startswith("ctl: no directory short enough"), err.getvalue())

    def _nowhere(self) -> contextlib.ExitStack:
        """No root short enough for a socket: the temp dir and every one of :data:`wire.SOCKET_DIRS` too deep.

        The deep temp dir is made for the block and removed after it: code under
        test still makes temporary files there (cutting stills does), and a path
        that exists only because an earlier run left it would pass here and fail
        on a clean machine.
        """
        stack = contextlib.ExitStack()
        deep = Path(DEEP_TEMP)
        if not deep.exists():
            deep.mkdir(parents=True)
            stack.callback(shutil.rmtree, deep, ignore_errors=True)
        stack.enter_context(mock.patch.object(tempfile, "tempdir", DEEP_TEMP))
        stack.enter_context(mock.patch.object(wire, "SOCKET_DIRS", (DEEP_TEMP,)))
        return stack

    def test_a_twin_is_made_with_no_socket_to_be_had(self) -> None:
        from robosimstudio_real2sim_agent import ctl

        twin = self.enterContext(short_dir("new-")) / "lid"
        with self._nowhere(), contextlib.redirect_stdout(io.StringIO()):
            code = ctl.main(["--run-dir", str(self.deep_run_dir()), "new", "lid", "--dir", str(twin)])
        self.assertEqual(0, code)
        self.assertTrue((twin / "scene.py").exists())

    def test_a_camera_is_fitted_with_no_socket_to_be_had(self) -> None:
        from robosimstudio.core.cameras import CameraSpec
        from robosimstudio_real2sim_agent import ctl

        cam = CameraSpec(name="c", width=640, height=480, fov_y=48, eye=(1.9, -0.7, 1.6), look_at=(0.6, 0.1, 0.75))
        world = np.array([[0.2, -0.5, 0.75], [1.0, -0.4, 0.8], [0.3, 0.5, 0.9], [0.9, 0.6, 1.1], [0.6, 0.0, 1.0]])
        points = [
            f"--point={x},{y},{z}={u},{v}" for (x, y, z), (u, v) in zip(world, cam.project(world)[0], strict=True)
        ]
        with self._nowhere(), contextlib.redirect_stdout(io.StringIO()):
            code = ctl.main(
                ["--run-dir", str(self.deep_run_dir()), "camera", "fit", *points, "--size", "640x480", "--fov", "48"]
            )
        self.assertEqual(0, code)

    def test_stills_are_cut_with_no_socket_to_be_had(self) -> None:
        from robosimstudio_real2sim_agent import ctl
        from tests.fixtures.real2sim_clips import flat_clip

        here = self.enterContext(short_dir("cut-"))
        clip = flat_clip(here / "c.mp4", 1, "red")
        argv = ["--run-dir", str(self.deep_run_dir()), "ref", "frames", str(clip), "--out", str(here / "ref")]
        with self._nowhere(), contextlib.redirect_stdout(io.StringIO()):
            code = ctl.main(argv)
        self.assertEqual(0, code)
        self.assertTrue((here / "ref" / "timeline.json").exists())

    def test_a_socket_the_server_cannot_bind_is_one_line_naming_sock(self) -> None:
        """Refused by the directory, or held by another page: the same one line, and no traceback."""
        from robosimstudio_real2sim_agent import serve, server

        for refusal in (PermissionError(13, "Permission denied"), FileExistsError("a server is already listening")):

            class Page:
                def __init__(self, **kwargs) -> None:
                    self.path = Path(kwargs["scene_path"]).resolve()
                    self.sock_path = kwargs["sock_path"]

                def serve(self, refusal=refusal) -> None:
                    raise wire.SocketRefused(self.sock_path, refusal)

            err = io.StringIO()
            here = self.enterContext(short_dir("bind-"))
            with (
                self.subTest(refusal=type(refusal).__name__),
                mock.patch.object(server, "LiveServer", Page),
                chdir(here),
                contextlib.redirect_stderr(err),
            ):
                code = serve.main(["--scene", str(here / "scene.py"), "--run-dir", str(here / "run")])
                self.assertEqual(2, code)
                self.assertIn("--sock", err.getvalue())
                self.assertEqual(1, len(err.getvalue().strip().splitlines()), err.getvalue())

    def test_the_page_turns_a_bind_it_was_refused_into_that_refusal(self) -> None:
        from unittest import mock as _mock

        from tests.fixtures.live_server import bare_live_server

        if os.geteuid() == 0:
            self.skipTest("root binds in any directory")
        locked = self.enterContext(short_dir("ro-"))
        os.chmod(locked, 0o500)
        self.addCleanup(os.chmod, locked, 0o700)
        live = bare_live_server(session=_mock.MagicMock())
        live.socket = wire.ControlSocket(locked / "ctl.sock", lambda request: {"ok": True})
        live.sock_path = locked / "ctl.sock"
        live.open_page = lambda: None  # type: ignore[method-assign]
        with (
            _mock.patch("signal.signal"),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(wire.SocketRefused, "--sock"),
        ):
            live.serve()

    def test_the_server_says_so_in_one_line_too(self) -> None:
        from robosimstudio_real2sim_agent import serve

        err = io.StringIO()
        with self._nowhere(), contextlib.redirect_stderr(err):
            code = serve.main(["--scene", "scene.py", "--run-dir", str(self.deep_run_dir())])
        self.assertEqual(2, code)
        self.assertIn("no directory short enough", err.getvalue())
        self.assertIn("--sock", err.getvalue())
        self.assertEqual(1, len(err.getvalue().strip().splitlines()), err.getvalue())


class TestTheBind(unittest.TestCase):
    def test_a_regular_file_is_not_deleted(self) -> None:
        path = self.enterContext(short_dir("w-")) / "f.txt"
        path.write_text("precious")
        with self.assertRaises(FileExistsError):
            wire.ControlSocket(path, lambda request: {"ok": True}).bind()
        self.assertEqual(path.read_text(), "precious")


class TestTheClient(unittest.TestCase):
    def test_a_server_that_drops_the_request_is_no_reply(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)

        def drop() -> None:
            conn, _ = server.accept()
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
            conn.close()

        thread = threading.Thread(target=drop, daemon=True)
        thread.start()
        try:
            with self.assertRaises((wire.NoReply, wire.NoServer)):
                wire.request(path, {"cmd": "status", "pad": "x" * 200_000}, 2.0)
        finally:
            thread.join(2.0)
            server.close()

    def test_a_frozen_server_is_no_reply_not_a_traceback(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(0)
        try:
            for _ in range(4):  # fills the backlog: later connects fail with EAGAIN
                with self.assertRaises(wire.NoReply):
                    wire.request(path, {"cmd": "status"}, 0.2)
        finally:
            server.close()

    def test_an_unreadable_socket_is_no_server(self) -> None:
        path = self.enterContext(short_dir("w-")) / "ctl.sock"
        path.write_text("not a socket")
        with self.assertRaises(wire.NoServer):
            wire.request(path, {"cmd": "status"}, 0.5)


if __name__ == "__main__":
    unittest.main()
