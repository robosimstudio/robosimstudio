"""The socket's path, its bind and its client, and the stills' camera and browser frame, held to what they say."""

from __future__ import annotations

import os
import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import stills, wire

TIER = "fast"


class TestTheSocketPath(unittest.TestCase):
    def test_every_spelling_of_one_run_dir_agrees(self) -> None:
        deep = Path(tempfile.mkdtemp(prefix="w-")) / ("d" * 60) / ("e" * 40)
        deep.mkdir(parents=True)
        here = os.getcwd()
        os.chdir(deep)
        try:
            self.assertEqual(wire.socket_path("run"), wire.socket_path(deep / "run"))
            self.assertEqual(wire.socket_path("run"), wire.socket_path(os.path.join(".", "run")))
        finally:
            os.chdir(here)

    def test_a_long_spelling_of_a_short_run_dir_comes_back_short(self) -> None:
        short = Path(tempfile.mkdtemp(prefix="w-")).resolve()
        padded = os.path.join(*([".."] * 40), *short.parts[1:], "run")  # from the cwd, past / and back down
        self.assertGreater(len(os.fsencode(padded)), wire.SUN_PATH_MAX)
        path = wire.socket_path(padded)
        self.assertLessEqual(len(os.fsencode(path)), wire.SUN_PATH_MAX, "the path returned can be bound")
        self.assertEqual(path, wire.socket_path(short / "run"))


class TestUpOnAFile(unittest.TestCase):
    def test_up_on_a_regular_file_leaves_it(self) -> None:
        import argparse
        import contextlib
        import io

        from robosimstudio_real2sim_agent import ctl

        tmp = Path(tempfile.mkdtemp(prefix="w-"))
        path, scene = tmp / "f.txt", tmp / "scene.py"
        path.write_text("precious")
        scene.write_text("")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = ctl.cmd_up(argparse.Namespace(scene=str(scene), share=False), path)
        self.assertEqual(code, 2)
        self.assertIn("is not a socket", err.getvalue())
        self.assertEqual(path.read_text(), "precious")


class TestTheBind(unittest.TestCase):
    def test_a_regular_file_is_not_deleted(self) -> None:
        path = Path(tempfile.mkdtemp(prefix="w-")) / "f.txt"
        path.write_text("precious")
        with self.assertRaises(FileExistsError):
            wire.ControlSocket(path, lambda request: {"ok": True}).bind()
        self.assertEqual(path.read_text(), "precious")


class TestTheClient(unittest.TestCase):
    def test_a_server_that_drops_the_request_is_no_reply(self) -> None:
        path = Path(tempfile.mkdtemp(prefix="w-")) / "ctl.sock"
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
        path = Path(tempfile.mkdtemp(prefix="w-")) / "ctl.sock"
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
        path = Path(tempfile.mkdtemp(prefix="w-")) / "ctl.sock"
        path.write_text("not a socket")
        with self.assertRaises(wire.NoServer):
            wire.request(path, {"cmd": "status"}, 0.5)


class TestTheCamera(unittest.TestCase):
    def test_eye_without_look_at_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            stills.camera_for(None, None, eye="1,0,1")
        with self.assertRaises(ValueError):
            stills.camera_for(None, None, look_at="0,0,0")
        with self.assertRaises(ValueError):
            stills.camera_for(None, None, name="top", eye="1,0,1", look_at="0,0,0")


class TestTheBrowserStill(unittest.TestCase):
    def test_the_frame_is_asked_of_a_client_as_opaque_rgb(self) -> None:
        asked = {}

        def get_render(height, width, transport_format="jpeg", timeout=None):
            asked.update(format=transport_format, timeout=timeout)
            return np.full((height, width, 3), 200, np.uint8)

        client = types.SimpleNamespace(get_render=get_render)
        server = types.SimpleNamespace(get_clients=lambda: {0: client})
        rgb = stills.browser_still(server, "64x32")
        self.assertEqual(rgb.shape, (32, 64, 3))
        self.assertEqual(asked["format"], "jpeg")
        self.assertIsNotNone(asked["timeout"])
        self.assertIsNone(stills.browser_still(types.SimpleNamespace(get_clients=dict)))

    def test_a_browser_that_never_answers_times_out(self) -> None:
        def get_render(height, width, transport_format="jpeg", timeout=None):
            raise TimeoutError

        server = types.SimpleNamespace(get_clients=lambda: {0: types.SimpleNamespace(get_render=get_render)})
        with self.assertRaisesRegex(TimeoutError, "no frame"):
            stills.browser_still(server)


if __name__ == "__main__":
    unittest.main()
