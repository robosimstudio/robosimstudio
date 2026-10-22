# SPDX-License-Identifier: Apache-2.0
"""The server's teardown, the board's bound, check under a routine, and routine names: no GPU."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import ScenePaths, SceneRegistrations
from robosimstudio_real2sim_agent.server import LiveServer

TIER = "fast"


def _bare_server() -> LiveServer:
    live = LiveServer.__new__(LiveServer)
    live.path = Path("/nonexistent/scene.py")
    live.head_stepped, live._head_ready = False, None
    live.ended, live.takes_written, live.serving = {}, [], "scene"
    live.runs, live.run_target, live.scene_id, live.watch_enabled = 0, 0, "", False
    live.runs_ended, live._walked = {}, (-1.0, [])
    live._arms_before, live.resets = {}, 0
    live.log_path, live._motions = "", {}
    live.scene_paths = ScenePaths(Path("/nonexistent"))
    live.build_error, live._tick_error = "", ""
    live.busy_state = ("", 0.0, False)
    live.script_meta = {}
    live.scripts_started = 0
    live.last_script = {}
    live.registrations = SceneRegistrations(Path("/nonexistent"))
    live.sim = mock.MagicMock(step_count=0)
    live.script = []
    live.recorder = None
    live.run_budget = 0
    live.last_error = ""
    live.board = {}
    live.ns = {}
    live.session = mock.MagicMock()
    live.session.paused = False
    return live


class ServeFinishesTheTake(unittest.TestCase):
    def test_quit_mid_take_finishes_the_recorder(self) -> None:
        live = _bare_server()
        live.open_page = lambda: None
        live.socket = mock.MagicMock()
        live.bus = mock.MagicMock()
        live.config = mock.MagicMock(share=False)
        live.url, live.sock_path, live.share_url = "http://x", Path("/tmp/x.sock"), None
        live.session.viewer.is_running.return_value = False
        live.script = [iter(())]
        live.recorder = object()
        finished = []
        live._finish_recorder = finished.append
        with mock.patch.object(server_mod.signal, "signal"):
            live.serve()
        self.assertEqual(finished, ["the server stopped"])
        self.assertEqual(live.script, [])
        live.session.close.assert_called_once()

    def test_a_failing_finish_does_not_skip_the_teardown(self) -> None:
        live = _bare_server()
        live.open_page = lambda: None
        live.socket = mock.MagicMock()
        live.bus = mock.MagicMock()
        live.config = mock.MagicMock(share=False)
        live.url, live.sock_path, live.share_url = "http://x", Path("/tmp/x.sock"), None
        live.session.viewer.is_running.return_value = False
        live.stop_scripts = mock.MagicMock(side_effect=RuntimeError("boom"))
        with mock.patch.object(server_mod.signal, "signal"):
            live.serve()
        live.socket.close.assert_called_once()
        live.session.close.assert_called_once()


class BoardIsBounded(unittest.TestCase):
    def test_board_keeps_the_newest(self) -> None:
        live = _bare_server()
        image = np.zeros((2, 2, 3), np.uint8)
        with mock.patch.object(server_mod.stills, "load_reference", return_value=(image, "x")):
            for i in range(server_mod.BOARD_LIMIT + 5):
                live.cmd_show("p.png", label=f"compare {i}")
        self.assertEqual(len(live.board), server_mod.BOARD_LIMIT)
        self.assertEqual(next(iter(live.board)), "compare 5")
        live.session.clear.assert_any_call("compare 0")

    def test_a_label_shown_again_is_the_newest(self) -> None:
        live = _bare_server()
        image = np.zeros((2, 2, 3), np.uint8)
        with mock.patch.object(server_mod.stills, "load_reference", return_value=(image, "x")):
            live.cmd_show("p.png", label="a")
            live.cmd_show("p.png", label="b")
            live.cmd_show("p.png", label="a")
        self.assertEqual(list(live.board), ["b", "a"])


class CheckUnderARoutine(unittest.TestCase):
    def test_check_is_refused_while_a_routine_plays(self) -> None:
        live = _bare_server()
        live.script = [iter(())]
        with mock.patch.object(server_mod.checks, "gate") as gate:
            reply = live.cmd_check()
        self.assertFalse(reply["ok"])
        self.assertIn("routine is playing", reply["error"])
        gate.assert_not_called()
        self.assertEqual(len(live.script), 1)


class RoutineNames(unittest.TestCase):
    def test_a_non_str_key_plays(self) -> None:
        live = _bare_server()
        live.sim = object()

        def fn(sim):
            yield

        live.ns = {"routines": {1: fn}}
        live.start_script = mock.MagicMock()
        self.assertEqual(live.routine_names(), ["1"])
        reply = live.cmd_play("1")
        self.assertTrue(reply["ok"], reply)
        live.start_script.assert_called_once()

    def test_a_routines_key_named_routine_plays(self) -> None:
        live = _bare_server()
        live.sim = object()

        def fn(sim):
            yield

        live.ns = {"routines": {"routine": fn}}
        live.start_script = mock.MagicMock()
        reply = live.cmd_play("routine")
        self.assertTrue(reply["ok"], reply)


if __name__ == "__main__":
    unittest.main()
