# SPDX-License-Identifier: Apache-2.0
"""``ctl``, the command line onto the live page: what each line parses to, what it waits on, and ``up`` and ``new``.

Every command line parses and every ``--help`` exits 0; the client imports
nothing of the library; ``run``, ``play`` and ``status --wait`` end on the
page's own word for how a run or routine ended; ``up`` recognises a server
that came up, one that died and one still building; ``new`` makes a twin's
folder and refuses one inside another scene.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio_real2sim_agent import ctl, wire
from tests.fixtures.live_server import FakePage, FakeStatus
from tests.fixtures.real2sim_clips import flat_clip
from tests.fixtures.sockets import short_dir

TIER = "fast"
"""No GPU and no page: a socket answering from a dict stands in for the page."""


ROOT = Path(__file__).resolve().parents[1]


SCENE = 'import robosimstudio as rss\nscene = rss.Scene(cell="void", name="t")\n'


def _write(directory: Path, text: str, name: str = "scene.py") -> Path:
    path = directory / name
    path.write_text(text)
    return path


def _subcommands(parser) -> list[str]:
    """The subcommands a parser knows, read off its subparsers action."""
    (action,) = (a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return list(action.choices)


class TestTheCommandLine(unittest.TestCase):
    LINES = (
        ["up"],
        ["up", "--port", "8081", "--scene", "s.py", "--share"],
        ["status"],
        ["reload"],
        ["snapshot", "--beside", "ref.png", "--size", "320x180"],
        ["run", "20"],
        ["pause"],
        ["resume"],
        ["reset"],
        ["eval", "1 + 1"],
        ["catalog"],
        ["catalog", "robots", "cells"],
        ["ref", "show", "p.png"],
        ["ref", "hide"],
        [
            "ref",
            "frames",
            "clip.mov",
            "--at",
            "0:closed,3.7:lid_up,end",
            "--every",
            "1.5",
        ],
        ["snapshot", "--note", "after moving the case"],
        [
            "compare",
            "--ref",
            "ref.jpg",
            "--key",
            "case=red",
            "--key",
            "socket=@1240,410",
            "--note",
            "first look",
        ],
        ["compare", "--take", "tidy", "--key", "case=red"],
        [
            "check",
            "--key",
            "case=red",
            "--object",
            "socket",
            "--steps",
            "30",
            "--settle-mm",
            "1",
        ],
        ["log", "--last", "5"],
        ["play"],
        ["play", "tidy", "--record", "take1", "--camera", "phone"],
        ["show", "run/cmp_001_beside.png", "--label", "the pair"],
        ["show", "--clear"],
        ["eval", "t.play()", "--record", "tidy", "--every", "4", "--camera", "phone"],
        ["quit"],
        ["new", "lid", "--ref", "clip.mov", "--dir", "elsewhere/lid"],
        ["camera", "fit", "--point", "-0.3,0.2,0.75=812,455", "--size", "640x480", "--fov", "48"],
        ["physics", "lift", "--only", "frozen,open", "--settle-mm", "1.5"],
    )

    def test_every_command_parses(self) -> None:
        parser = ctl.build_parser()
        for line in self.LINES:
            with self.subTest(line=line):
                self.assertEqual(parser.parse_args(ctl._join_points(line)).cmd, line[0])
        args = parser.parse_args(["--run-dir", "d", "--json", "status"])
        self.assertEqual((args.run_dir, args.json), ("d", True), "--json before the command")
        self.assertTrue(parser.parse_args(["status", "--json"]).json, "--json after the command")
        self.assertFalse(parser.parse_args(["status"]).json)

    def test_every_command_has_a_line_here(self) -> None:
        """A subcommand added without one would be parsed by nothing in this class."""
        self.assertEqual(set(_subcommands(ctl.build_parser())), {line[0] for line in self.LINES})

    def test_a_path_is_resolved_where_the_caller_sits(self) -> None:
        parser = ctl.build_parser()
        req, timeout = ctl.build_request(parser.parse_args(["snapshot", "--beside", "ref.png"]), parser)
        self.assertEqual(req["cmd"], "snapshot")
        self.assertTrue(
            Path(req["beside"]).is_absolute(),
            "paths are resolved where the caller sits",
        )
        self.assertEqual(timeout, ctl.TIMEOUTS["snapshot"])

    def test_catalog_names_its_kinds_and_refuses_one_it_does_not_know(self) -> None:
        parser = ctl.build_parser()
        req, _ = ctl.build_request(parser.parse_args(["catalog", "robots", "cells"]), parser)
        self.assertEqual(req, {"cmd": "catalog", "kinds": ["robots", "cells"]})
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ctl.build_request(parser.parse_args(["catalog", "planets"]), parser)

    def test_run_is_its_steps_at_the_default_timeout(self) -> None:
        parser = ctl.build_parser()
        req, timeout = ctl.build_request(parser.parse_args(["run", "7"]), parser)
        self.assertEqual((req, timeout), ({"cmd": "run", "steps": 7}, ctl.DEFAULT_TIMEOUT))

    def test_compare_carries_its_keys_its_note_and_its_reference(self) -> None:
        parser = ctl.build_parser()
        line = ["compare", "--ref", "r.jpg", "--key", "case=red", "--key", "socket=@1,2", "--note", "n"]
        req, timeout = ctl.build_request(parser.parse_args(line), parser)
        self.assertEqual(
            (req["cmd"], req["keys"], req["note"]),
            ("compare", ["case=red", "socket=@1,2"], "n"),
        )
        self.assertTrue(Path(req["ref"]).is_absolute())
        self.assertEqual(timeout, ctl.TIMEOUTS["compare"])

    def test_check_asks_nothing_of_a_photograph(self) -> None:
        parser = ctl.build_parser()
        req, _ = ctl.build_request(
            parser.parse_args(["check", "--object", "a", "--object", "b", "--steps", "9"]),
            parser,
        )
        self.assertEqual((req["objects"], req["steps"]), (["a", "b"], 9))
        self.assertNotIn("ref", req, "the gate asks nothing of a photograph")

    def test_ref_frames_leaves_the_stills_to_the_page_unless_told_where(self) -> None:
        parser = ctl.build_parser()
        req, _ = ctl.build_request(parser.parse_args(["ref", "frames", "c.mov", "--at", "0,end"]), parser)
        self.assertEqual((req["op"], req["at"]), ("frames", "0,end"))
        # no --out: the server puts the stills in ref/ beside scene.py, wherever the shell stands
        self.assertTrue(Path(req["path"]).is_absolute())
        self.assertEqual(req["out"], "")
        req, _ = ctl.build_request(parser.parse_args(["ref", "frames", "c.mov", "--out", "elsewhere"]), parser)
        self.assertTrue(Path(req["out"]).is_absolute())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ctl.build_request(parser.parse_args(["ref", "frames"]), parser)

    def test_eval_and_play_record_only_when_asked(self) -> None:
        parser = ctl.build_parser()
        req, _ = ctl.build_request(
            parser.parse_args(["eval", "go()", "--record", "take1", "--every", "2"]),
            parser,
        )
        self.assertEqual((req["code"], req["record"], req["every"]), ("go()", "take1", 2))
        req, _ = ctl.build_request(parser.parse_args(["eval", "go()"]), parser)
        self.assertNotIn("record", req, "no recording unless asked")
        req, _ = ctl.build_request(parser.parse_args(["play", "tidy", "--record", "t1"]), parser)
        self.assertEqual((req["name"], req["record"], req["every"]), ("tidy", "t1", 4))
        req, _ = ctl.build_request(
            parser.parse_args(["play", "tidy", "--record", "t1", "--also", "gl,blender"]), parser
        )
        self.assertEqual(req["also"], "gl,blender")
        with self.assertRaises(SystemExit):
            ctl.build_request(parser.parse_args(["play", "tidy", "--also", "gl"]), parser)  # nothing filmed to draw

    def test_log_and_show(self) -> None:
        parser = ctl.build_parser()
        req, _ = ctl.build_request(parser.parse_args(["log", "--last", "3"]), parser)
        self.assertEqual(req, {"cmd": "log", "last": 3})
        req, _ = ctl.build_request(parser.parse_args(["show", "p.png", "--label", "pair"]), parser)
        self.assertTrue(Path(req["path"]).is_absolute() and req["label"] == "pair" and not req["clear"])
        self.assertEqual(ctl.build_request(parser.parse_args(["show", "--clear"]), parser)[0]["clear"], True)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ctl.build_request(parser.parse_args(["show"]), parser)

    def test_every_help_exits_zero(self) -> None:
        """Every subcommand's ``--help`` in this process; the three entry points each in a fresh one, at once."""
        parser = ctl.build_parser()
        for name in _subcommands(parser):
            with self.subTest(command=name), contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as caught:
                    parser.parse_args([name, "--help"])
                self.assertIn(caught.exception.code, (0, None))
                self.assertIn("usage:", out.getvalue())
        entry_points = {
            "the client": [sys.executable, "-m", "robosimstudio_real2sim_agent", "--help"],
            "the server": [sys.executable, "-m", "robosimstudio_real2sim_agent.serve", "--help"],
            "scripts/live.py": [sys.executable, str(ROOT / "scripts" / "live.py"), "--help"],
        }
        running = {
            what: subprocess.Popen(line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
            for what, line in entry_points.items()
        }
        for what, proc in running.items():
            stdout, stderr = proc.communicate(timeout=300)
            with self.subTest(entry_point=what):
                self.assertEqual(proc.returncode, 0, stderr[-2000:])
                if what != "the client":
                    self.assertIn("--camera-view", stdout)
                    self.assertIn("--share", stdout)
                    self.assertNotIn(
                        "--live",
                        stdout,
                        "the studio's word for its camera view is not this flag",
                    )

    def test_the_client_is_standard_library_only(self) -> None:
        probe = (
            "import sys, robosimstudio_real2sim_agent.ctl; "
            "print(sorted(m for m in sys.modules if m.startswith(('robosimstudio', 'numpy', 'viser', 'warp'))))"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        self.assertEqual(
            json.loads(done.stdout.replace("'", '"')),
            [
                "robosimstudio_real2sim_agent",
                "robosimstudio_real2sim_agent.ctl",
                "robosimstudio_real2sim_agent.wire",
            ],
            "the client answers in milliseconds because it loads nothing else",
        )

    def test_status_with_no_server_exits_2_and_says_how_to_start_one(self) -> None:
        tmp = self.enterContext(short_dir("nosrv-"))
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "robosimstudio_real2sim_agent",
                "--run-dir",
                str(tmp),
                "status",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("no server", done.stderr)
        self.assertIn(" up", done.stderr)


class TestTheRunWait(unittest.TestCase):
    """``run N`` waits on the page's word for how that run ended; whatever ends it, the wait ends saying so."""

    def setUp(self) -> None:
        self.path = self.enterContext(short_dir("run-")) / "ctl.sock"
        self.page = FakePage(self.path, {"step": 0, "time": 0.0, "runs_ended": {}})
        self.addCleanup(self.page.close)
        self.run = {"ok": True, "run": 2, "step": 40, "target": 1040}

    def _ended(self, how: str, step: int, finite=None) -> None:
        self.page.status.update(
            step=step, runs_ended={"2": {"serial": 2, "how": how, "step": step, "target": 1040, "finite": finite}}
        )

    def test_a_rebuild_under_the_run_ends_it(self) -> None:
        self._ended("the scene was rebuilt", 3)
        started = time.time()
        reply = ctl.wait_for_steps(self.path, self.run, 60.0)
        self.assertFalse(reply["ok"])
        self.assertIn("rebuilt", reply["error"])
        self.assertLess(time.time() - started, 5.0, "said at once")

    def test_a_pause_under_the_run_is_no_success(self) -> None:
        self._ended("paused", 240, finite=True)
        reply = ctl.wait_for_steps(self.path, self.run, 60.0)
        self.assertFalse(reply["ok"])
        self.assertIn("ended at step 240 of 1040: paused", reply["error"])

    def test_steps_taken_end_it(self) -> None:
        self._ended("done", 1040, finite=True)
        reply = ctl.wait_for_steps(self.path, self.run, 60.0)
        self.assertEqual(reply, {"ok": True, "step": 1040, "time": 0.0, "finite": True})

    def test_an_earlier_runs_ending_is_not_this_ones(self) -> None:
        self.page.status.update(step=41, runs_ended={"1": {"serial": 1, "how": "done", "step": 40, "target": 40}})
        reply = ctl.wait_for_steps(self.path, self.run, 1.0)
        self.assertFalse(reply["ok"])
        self.assertIn("still running, step 41 of 1040", reply["error"])

    def test_play_waits_as_long_as_eval(self) -> None:
        self.assertEqual(ctl.TIMEOUTS["play"], ctl.TIMEOUTS["eval"])


class TestUp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = self.enterContext(short_dir("up-"))
        self.log = self.tmp / "serve.log"
        self.procs: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(10)

    def spawn(self, code: str) -> subprocess.Popen:
        proc = ctl.spawn_server([sys.executable, "-c", code], self.log)
        self.procs.append(proc)
        return proc

    def test_the_server_command_line(self) -> None:
        args = argparse.Namespace(port=8081, scene="s.py", run_dir="r", share=True, hold=False)
        argv = ctl.serve_argv(args, Path("r/ctl.sock"))
        self.assertEqual(argv[:3], [sys.executable, "-m", "robosimstudio_real2sim_agent.serve"])
        self.assertIn("--share", argv)
        self.assertNotIn("--hold", argv)
        self.assertEqual(
            argv[argv.index("--sock") + 1],
            "r/ctl.sock",
            "the same socket the client will use",
        )

    def test_a_server_that_came_up(self) -> None:
        line = wire.ready_line("http://localhost:1", "/s", "/f")
        proc = self.spawn(f"print({line!r}, flush=True)\nimport time\ntime.sleep(60)\n")
        ready = ctl.wait_ready(self.log, lambda: proc.poll() is None, time.monotonic() + 30.0, poll=0.1)
        self.assertEqual(ready["url"], "http://localhost:1")
        self.assertIsNone(proc.poll(), "and it is still serving")

    def test_a_server_that_died(self) -> None:
        proc = self.spawn("print('boom', flush=True)\nraise SystemExit(3)\n")
        with self.assertRaises(ctl.ServerDied) as caught:
            ctl.wait_ready(self.log, lambda: proc.poll() is None, time.monotonic() + 30.0, poll=0.1)
        self.assertIn("boom", str(caught.exception), "the log's tail is the message")

    def test_a_server_that_is_still_building(self) -> None:
        proc = self.spawn("import time\ntime.sleep(60)\n")
        with self.assertRaises(ctl.NotReady):
            ctl.wait_ready(self.log, lambda: proc.poll() is None, time.monotonic() + 0.6, poll=0.1)
        self.assertIsNone(proc.poll(), "left running: a cold first build is worth waiting for")

    def test_every_flag_up_forwards_is_the_servers(self) -> None:
        from robosimstudio_real2sim_agent.serve import create_parser

        known = {option for action in create_parser()._actions for option in action.option_strings}
        for flag in ctl.SERVE_FLAGS:
            self.assertIn(
                "--" + flag.replace("_", "-"),
                known,
                "a switch `up` forwards that the server does not take",
            )

    def test_up_makes_the_run_dir_before_writing_into_it(self) -> None:
        from unittest import mock

        scene = _write(self.tmp, SCENE)
        run_dir = self.tmp / "new" / "run"
        line = wire.ready_line("http://localhost:1", run_dir / "ctl.sock", scene)
        code = f"print({line!r}, flush=True)\nimport time\ntime.sleep(60)\n"
        real = ctl.spawn_server

        def spawn(argv, log_path, renderer=""):
            proc = real([sys.executable, "-c", code], log_path, renderer)
            self.procs.append(proc)
            return proc

        with mock.patch.object(ctl, "spawn_server", spawn):
            rc = ctl.main(["--run-dir", str(run_dir), "up", "--scene", str(scene), "--timeout", "30"])
        self.assertEqual(rc, 0)
        self.assertEqual(int((run_dir / "serve.pid").read_text()), self.procs[0].pid)

    def test_up_on_a_running_page_says_when_it_is_not_the_one_asked_for(self) -> None:
        scene = _write(self.tmp, SCENE)
        page = FakePage(self.tmp / "ctl.sock", {"file": str(scene.resolve()), "share_url": None, "url": "u", "pid": 1})
        self.addCleanup(page.close)
        argv = ["--run-dir", str(self.tmp), "up", "--scene"]
        self.assertEqual(ctl.main([*argv, str(scene)]), 0, "the same file: a no-op naming the page")
        other = _write(self.tmp, SCENE, name="other.py")
        self.assertEqual(ctl.main([*argv, str(other)]), 1, "another file is not silently ignored")
        self.assertEqual(ctl.main([*argv, str(scene), "--share"]), 1, "nor is --share on a page without one")

    def test_the_server_draws_with_the_renderer_up_names(self) -> None:
        proc = ctl.spawn_server(
            [sys.executable, "-c", f"import os; print(os.environ.get({ctl.RENDERER_ENV!r}), flush=True)"],
            self.log,
            "gl",
        )
        self.procs.append(proc)
        proc.wait(30)
        self.assertEqual(self.log.read_text().strip(), "gl")

    def test_up_refuses_a_renderer_nobody_registered(self) -> None:
        scene = _write(self.tmp, SCENE)
        rc = ctl.main(["--run-dir", str(self.tmp), "up", "--scene", str(scene), "--renderer", "raytracer"])
        self.assertEqual(rc, 2)
        self.assertFalse(self.log.exists(), "nothing was started")

    def test_the_renderers_up_names_are_the_librarys(self) -> None:
        from robosimstudio.solvers.render import AUTO, RENDERERS

        self.assertEqual(set(ctl.RENDERERS), {*RENDERERS, AUTO})

    def test_up_on_a_running_page_says_when_it_draws_with_another_renderer(self) -> None:
        scene = _write(self.tmp, SCENE)
        status = {
            "file": str(scene.resolve()),
            "share_url": None,
            "url": "u",
            "pid": 1,
            "renderer_asked": "",
            "renderer_drawing": "rtx",
        }
        page = FakePage(self.tmp / "ctl.sock", status)
        self.addCleanup(page.close)
        argv = ["--run-dir", str(self.tmp), "up", "--scene", str(scene)]
        self.assertEqual(ctl.main([*argv, "--renderer", "auto"]), 0, "the machine's pick is what it draws with")
        self.assertEqual(ctl.main([*argv, "--renderer", "rtx"]), 0, "and the renderer that pick is")
        page.status.update(renderer_asked="rtx", renderer_drawing="rtx", renderer_machine="rtx")
        self.assertEqual(ctl.main([*argv, "--renderer", "auto"]), 0, "rtx named is what auto picks here")
        page.status.update(renderer_asked="gl", renderer_drawing="gl", renderer_machine="rtx")
        self.assertEqual(ctl.main([*argv, "--renderer", "auto"]), 1, "gl named is not what auto picks here")
        self.assertEqual(ctl.main([*argv, "--renderer", "gl"]), 0, "the one it was started under")
        self.assertEqual(ctl.main([*argv, "--renderer", "tiled"]), 1, "another renderer is not silently ignored")
        self.assertEqual(ctl.main([*argv, "--renderer", "blendr"]), 2, "a misspelling is said before the page is")

    def test_up_refuses_a_missing_scene_file(self) -> None:
        rc = ctl.main(["--run-dir", str(self.tmp), "up", "--scene", str(self.tmp / "absent.py")])
        self.assertEqual(rc, 2)
        self.assertFalse(self.log.exists(), "nothing was started")


class TestUpOnAFile(unittest.TestCase):
    def test_up_on_a_regular_file_leaves_it(self) -> None:
        import argparse
        import contextlib
        import io

        from robosimstudio_real2sim_agent import ctl

        tmp = self.enterContext(short_dir("w-"))
        path, scene = tmp / "f.txt", tmp / "scene.py"
        path.write_text("precious")
        scene.write_text("")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = ctl.cmd_up(argparse.Namespace(scene=str(scene), share=False), path)
        self.assertEqual(code, 2)
        self.assertIn("is not a socket", err.getvalue())
        self.assertEqual(path.read_text(), "precious")


class CtlWaits(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = self.enterContext(short_dir())
        self.sock = self.tmp / "s.sock"
        self.log = self.tmp / "serve.log"
        self.log.write_text("")

    def test_a_routine_is_waited_for_with_its_output_streamed(self) -> None:
        playing = {"ok": True, "step": 10, "time": 0.2, "last_script": {}, "busy": ""}
        done = {"ok": True, "step": 30, "time": 0.5, "busy": "", "speed": 0.4, "takes_written": ["take1"]}
        done["ended"] = {"7": {"serial": 7, "name": "tidy", "how": "finished", "steps": 20, "wall_seconds": 1.2}}
        fake = FakeStatus(self.sock, [playing, done])
        self.addCleanup(fake.close)
        with open(self.log, "a") as f:
            f.write("[tidy] lid open\n")
        printed: list[str] = []
        with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            reply = ctl.wait_for_script(self.sock, {"serial": 7, "recording": "/r/take1.mp4"}, self.log, 0, 30.0)
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["take"], "take1")
        self.assertIn("[tidy] lid open", printed)

    def test_a_routine_is_told_its_own_ending_not_the_newest(self) -> None:
        mine = {"serial": 1, "name": "a", "how": "failed", "error": "RoutineError: a broke"}
        newer = {"serial": 2, "name": "b", "how": "finished"}
        fake = FakeStatus(self.sock, [{"ok": True, "busy": "", "last_script": newer, "ended": {"1": mine, "2": newer}}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 1}, self.log, 0, 30.0)
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["routine"], "a")

    def test_a_take_written_and_then_another_started_counts_as_written(self) -> None:
        done = {"ok": True, "busy": "", "ended": {"3": {"serial": 3, "name": "a", "how": "finished"}}}
        fake = FakeStatus(self.sock, [{**done, "recording": "take2", "last_take": "", "takes_written": ["take1"]}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 3, "take": "take1"}, self.log, 0, 30.0)
        self.assertEqual(reply["take"], "take1")

    def test_status_wait_does_not_wait_out_a_playing_routine(self) -> None:
        stepping = {"ok": True, "busy": "playing one step of a routine", "busy_routine": True}
        fake = FakeStatus(self.sock, [stepping])
        self.addCleanup(fake.close)
        self.assertTrue(ctl.wait_until_free(self.sock, 5.0)["busy_routine"])

    def test_a_failed_routine_is_an_error_with_its_traceback(self) -> None:
        ended = {"serial": 2, "name": "eval", "how": "failed", "error": "RoutineError: stopped 364 mm short"}
        fake = FakeStatus(self.sock, [{"ok": True, "busy": "", "ended": {"2": ended}}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 2}, self.log, 0, 30.0)
        self.assertFalse(reply["ok"])
        self.assertIn("364 mm short", reply["error"])

    def test_status_wait_returns_once_the_page_is_free(self) -> None:
        busy = {"ok": True, "busy": "rendering a still", "busy_seconds": 3.0}
        fake = FakeStatus(self.sock, [busy, {"ok": True, "busy": "", "pending_save": True}, {"ok": True, "busy": ""}])
        self.addCleanup(fake.close)
        self.assertEqual(ctl.wait_until_free(self.sock, 30.0), {"ok": True, "busy": ""})

    def test_a_run_is_waited_for_through_a_busy_page_until_its_ending(self) -> None:
        busy = {"ok": True, "busy": "rebuilding the scene file", "step": 90, "runs_ended": {}}
        ended = {"serial": 3, "how": "done", "step": 160, "target": 160, "finite": True}
        fake = FakeStatus(
            self.sock, [busy, busy, {"ok": True, "busy": "", "step": 160, "time": 2.7, "runs_ended": {"3": ended}}]
        )
        self.addCleanup(fake.close)
        reply = ctl.wait_for_steps(self.sock, {"run": 3, "step": 40, "target": 160}, 60.0)
        self.assertEqual((reply["ok"], reply["step"], reply["finite"]), (True, 160, True))

    def test_a_routine_that_finished_without_its_take_is_no_success(self) -> None:
        ended = {"serial": 5, "name": "lift", "how": "finished"}
        status = {"ok": True, "busy": "", "ended": {"5": ended}, "takes_written": [], "last_error": "ffmpeg died"}
        fake = FakeStatus(self.sock, [status])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 5, "take": "t1"}, self.log, 0, 5.0)
        self.assertFalse(reply["ok"])
        self.assertIn("its take t1 was not written: ffmpeg died", reply["error"])

    def test_a_failure_on_a_scene_that_was_not_the_files_says_so(self) -> None:
        printed: list[str] = []
        with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = ctl.show({"ok": False, "error": "RoutineError", "stale": "the last save did not build"}, False)
        self.assertEqual(code, 1)
        self.assertIn("stale: the last save did not build", printed)

    def test_a_run_keeps_the_notes_its_own_reply_carried(self) -> None:
        ended = {"serial": 1, "how": "done", "step": 5, "target": 5, "finite": True}

        def answer(request: dict) -> dict:
            if request["cmd"] == "run":
                return {"ok": True, "run": 1, "step": 0, "target": 5, "stale": "the last save did not build"}
            return {"ok": True, "busy": "", "step": 5, "time": 0.1, "runs_ended": {"1": ended}}

        sock = self.tmp / "run.sock"
        server = wire.ControlSocket(sock, answer)
        server.start()
        self.addCleanup(server.close)
        printed: list[str] = []
        with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = ctl.main(["--run-dir", str(self.tmp / "run"), "--sock", str(sock), "--json", "run", "5"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(printed[-1])["stale"], "the last save did not build")

    def test_a_long_run_says_where_it_is_and_gives_up_at_its_limit(self) -> None:
        fake = FakeStatus(self.sock, [{"ok": True, "busy": "", "step": 45, "runs_ended": {}}])
        self.addCleanup(fake.close)
        said: list[str] = []
        clock = iter(range(0, 10_000, 4))
        fake_time = types.SimpleNamespace(monotonic=lambda: float(next(clock)), sleep=lambda s: None)
        with mock.patch.object(ctl, "time", fake_time), mock.patch.object(ctl, "_say", said.append):
            reply = ctl.wait_for_steps(self.sock, {"run": 3, "step": 40, "target": 1040}, 30.0)
        self.assertFalse(reply["ok"])
        self.assertIn("still running, step 45 of 1040 after 30 s", reply["error"])
        self.assertTrue(said and "step 45 of 1040" in said[0], said)

    def test_a_take_is_not_held_up_by_another_routines_step(self) -> None:
        done = {"serial": 4, "name": "lift", "how": "finished"}
        stepping = {"busy": "playing one step of a routine", "busy_routine": True, "recording": ""}
        fake = FakeStatus(self.sock, [{"ok": True, **stepping, "ended": {"4": done}, "takes_written": ["t1"]}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 4, "take": "t1"}, self.log, 0, 5.0)
        self.assertEqual(reply["take"], "t1")

    def test_up_asks_a_running_page_for_the_light_status(self) -> None:
        asked: list[dict] = []
        scene = self.tmp / "scene.py"
        scene.write_text("")

        def answer(request: dict) -> dict:
            asked.append(request)
            return {"ok": True, "file": str(scene.resolve()), "url": "http://x", "pid": 1, "share_url": None}

        sock = self.tmp / "up.sock"
        server = wire.ControlSocket(sock, answer)
        server.start()
        self.addCleanup(server.close)
        with mock.patch("builtins.print"):
            ctl.main(["--run-dir", str(self.tmp / "run"), "--sock", str(sock), "up", "--scene", str(scene)])
        self.assertTrue(asked and asked[0].get("light"), asked)

    def test_a_pending_save_is_counted_in_the_waits_own_seconds(self) -> None:
        fake = FakeStatus(self.sock, [{"ok": True, "busy": "", "busy_seconds": 0.0, "pending_save": True}])
        self.addCleanup(fake.close)
        said: list[str] = []
        clock = iter(range(0, 10_000, 4))
        fake_time = types.SimpleNamespace(monotonic=lambda: float(next(clock)), sleep=lambda s: None)
        with mock.patch.object(ctl, "time", fake_time), mock.patch.object(ctl, "_say", said.append):
            ctl.wait_until_free(self.sock, 30.0)
        self.assertTrue(said and not said[-1].endswith("(0 s)"), said)

    def test_ref_frames_without_a_page_says_what_was_wrong_in_one_line(self) -> None:
        clip = flat_clip(self.tmp / "c.mp4", 2, "red")
        (self.tmp / "scene.py").write_text("")
        printed: list[str] = []
        with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = ctl.main(["--run-dir", str(self.tmp / "run"), "ref", "frames", str(clip), "--at", "0:a,x:b"])
        self.assertEqual(code, 1)
        self.assertTrue(printed and printed[-1].startswith("ctl: "), printed)

    def test_up_prints_its_log_and_status_does_not(self) -> None:
        printed: list[str] = []
        with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            ctl.show({"ok": True, "url": "http://x", "log": "/x/serve.log"}, False)
            ctl.show({"ok": True, "busy": "", "server_log": "/x/serve.log"}, False)
        self.assertEqual(sum("/x/serve.log" in line for line in printed), 1)

    def test_ref_frames_needs_no_page(self) -> None:
        clip = flat_clip(self.tmp / "c.mp4", 2, "red")
        (self.tmp / "scene.py").write_text("")  # the page would put ref/ beside it
        run_dir = self.tmp / "run"
        code = ctl.main(["--run-dir", str(run_dir), "--json", "ref", "frames", str(clip), "--at", "0:start,end"])
        self.assertEqual(code, 0)
        timeline = json.loads((self.tmp / "ref" / "timeline.json").read_text())
        self.assertEqual([a.get("label", "") for a in timeline["anchors"]], ["start", ""])


class TwinInsideAScene(unittest.TestCase):
    def test_a_twin_is_not_made_where_another_scenes_page_would_take_it_for_its_own(self) -> None:
        live = self.enterContext(short_dir())
        (live / "ctl").write_text("")
        outer = live / "twins" / "a"
        outer.mkdir(parents=True)
        (outer / "scene.py").write_text("")
        with mock.patch("builtins.print"):
            code = ctl.main(["--run-dir", str(live / "run"), "new", "b", "--dir", str(outer / "b")])
        self.assertEqual(code, 1)
        self.assertFalse((outer / "b").exists())
        with mock.patch("builtins.print"):
            self.assertEqual(ctl.main(["--run-dir", str(live / "run"), "new", "c"]), 0, "twins/ is no scene's code")


class TwinInsideAServedFolder(unittest.TestCase):
    def test_a_folder_a_page_served_from_is_a_scene_whatever_its_file_is_called(self) -> None:
        live = self.enterContext(short_dir())
        (live / "ctl").write_text("")
        served = live / "bench"
        (served / "run").mkdir(parents=True)
        (served / "run" / "serve.log").write_text("")
        with mock.patch("builtins.print"):
            code = ctl.main(["--run-dir", str(live / "run"), "new", "t", "--dir", str(served / "t")])
        self.assertEqual(code, 1)


class TwinFolder(unittest.TestCase):
    def setUp(self) -> None:
        self.live = self.enterContext(short_dir())
        self.run_dir = self.live / "run"
        self.clip = self.live / "clip.mov"
        self.clip.write_bytes(b"not really a clip")

    def test_new_makes_the_folder_with_its_own_ctl(self) -> None:
        self.assertEqual(ctl.main(["--run-dir", str(self.run_dir), "new", "lid", "--ref", str(self.clip)]), 0)
        twin = self.live / "twins" / "lid"
        self.assertEqual(sorted(p.name for p in twin.iterdir()), ["REPORT.md", "ctl", "ref", "scene.py"])
        self.assertEqual((twin / "ref" / "clip.mov").read_bytes(), b"not really a clip")
        self.assertTrue(os.access(twin / "ctl", os.X_OK))
        self.assertIn("## Shortcuts and departures from the reference", (twin / "REPORT.md").read_text())
        compile((twin / "scene.py").read_text(), "scene.py", "exec")
        self.assertEqual(
            ctl.main(["--run-dir", str(self.run_dir), "new", "lid"]), 1, "an existing twin is not overwritten"
        )
        self.assertEqual(ctl.main(["--run-dir", str(self.run_dir), "new", "../escape"]), 2)

    def test_new_from_inside_a_twin_made_anywhere_is_refused(self) -> None:
        elsewhere = self.live / "work" / "lid"
        ctl.main(["--run-dir", str(self.run_dir), "new", "lid", "--dir", str(elsewhere)])
        self.assertEqual(ctl.main(["--run-dir", str(elsewhere / "run"), "new", "nested"]), 1)

    def test_two_references_of_one_name_are_refused(self) -> None:
        (self.live / "a").mkdir()
        (self.live / "b").mkdir()
        for d in ("a", "b"):
            (self.live / d / "IMG_0001.MOV").write_bytes(b"x")
        argv = ["--run-dir", str(self.run_dir), "new", "lid"]
        argv += ["--ref", str(self.live / "a" / "IMG_0001.MOV"), "--ref", str(self.live / "b" / "IMG_0001.MOV")]
        self.assertEqual(ctl.main(argv), 2)
        self.assertFalse((self.live / "twins" / "lid").exists())

    def test_a_twin_unzipped_away_from_a_checkout_runs_on_python(self) -> None:
        import subprocess

        ctl.main(["--run-dir", str(self.run_dir), "new", "lid"])
        moved = self.enterContext(short_dir()) / "lid"
        shutil.copytree(self.live / "twins" / "lid", moved)
        env = {**os.environ, "PYTHON": sys.executable}
        done = subprocess.run(
            [str(moved / "ctl"), "status"], capture_output=True, text=True, timeout=60, check=False, env=env
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn(str(moved / "run" / "ctl.sock"), done.stderr)

    def test_new_from_inside_a_twin_is_refused(self) -> None:
        ctl.main(["--run-dir", str(self.run_dir), "new", "lid"])
        twin = self.live / "twins" / "lid"
        self.assertEqual(ctl.main(["--run-dir", str(twin / "run"), "new", "nested"]), 1)
        self.assertFalse((twin / "twins").exists())

    def test_its_ctl_pins_run_in_the_folder(self) -> None:
        import subprocess

        twin = self.live / "elsewhere"
        ctl.main(["--run-dir", str(self.run_dir), "new", "lid", "--dir", str(twin)])
        done = subprocess.run([str(twin / "ctl"), "status"], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(done.returncode, 2)
        self.assertIn(str(twin / "run" / "ctl.sock"), done.stderr)

    def test_up_serves_the_scene_beside_its_run_directory(self) -> None:
        printed: list[str] = []
        with mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            code = ctl.main(["--run-dir", str(self.run_dir), "up", "--timeout", "1"])
        self.assertEqual(code, 2)
        self.assertIn(str(self.live / "scene.py"), printed[0])


def _ctl(argv: list[str]) -> tuple[int, str, str]:
    """``ctl.main(argv)`` with what it printed on stdout and on stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ctl.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestTheBusyNote(unittest.TestCase):
    """The clause a timeout is followed by: what the page is busy with, if it is, and what to do about it."""

    def setUp(self) -> None:
        self.sock = self.enterContext(short_dir("busy-")) / "ctl.sock"

    def _page(self, **status) -> None:
        self.addCleanup(FakePage(self.sock, status).close)

    def test_no_page_is_no_note(self) -> None:
        self.assertEqual("", ctl.busy_note(self.sock))

    def test_a_free_page_is_no_note(self) -> None:
        self._page(busy="", busy_seconds=0.0)
        self.assertEqual("", ctl.busy_note(self.sock))

    def test_a_busy_page_says_with_what_for_how_long_and_how_to_wait_it_out(self) -> None:
        self._page(busy="rendering a still", busy_seconds=12.4)
        self.assertEqual(
            "; the page is busy rendering a still (12 s so far) and carries on -- `status --wait` returns when it is done",
            ctl.busy_note(self.sock),
        )

    def test_a_routines_step_is_the_page_at_work_and_status_wait_would_not_wait_it_out(self) -> None:
        self._page(busy="playing one step of a routine", busy_routine=True, busy_seconds=3.0)
        note = ctl.busy_note(self.sock)
        self.assertIn("a routine is playing, one of its steps 3 s so far", note)
        self.assertNotIn("status --wait", note)


class TestTheServersLog(unittest.TestCase):
    """Where the page's output goes: as it says, else the note ``up`` left for that page's pid, else serve.log."""

    def setUp(self) -> None:
        self.run_dir = self.enterContext(short_dir("log-"))
        self.sock = self.run_dir / "ctl.sock"

    def _page(self, **status) -> None:
        self.addCleanup(FakePage(self.sock, status).close)

    def test_the_page_names_its_log(self) -> None:
        self._page(server_log="/x/elsewhere.log", pid=7)
        (self.run_dir / "serve.log.path").write_text("7 /x/noted.log\n")
        self.assertEqual(Path("/x/elsewhere.log"), ctl.server_log(self.sock, self.run_dir))

    def test_the_note_is_read_for_the_page_its_pid_still_is(self) -> None:
        self._page(pid=7)
        (self.run_dir / "serve.log.path").write_text("7 /x/noted.log\n")
        self.assertEqual(Path("/x/noted.log"), ctl.server_log(self.sock, self.run_dir))

    def test_a_note_left_by_another_page_is_not_trusted(self) -> None:
        self._page(pid=8)
        (self.run_dir / "serve.log.path").write_text("7 /x/noted.log\n")
        self.assertEqual(self.run_dir / "serve.log", ctl.server_log(self.sock, self.run_dir))

    def test_with_no_page_and_no_note_it_is_serve_log_in_the_run_dir(self) -> None:
        self.assertEqual(self.run_dir / "serve.log", ctl.server_log(self.sock, self.run_dir))


class TestCameraFitOnTheCommandLine(unittest.TestCase):
    """``camera fit`` prints a CameraSpec that is that camera, and refuses in one line what it cannot fit."""

    def setUp(self) -> None:
        from robosimstudio.core.cameras import CameraSpec

        self.CameraSpec = CameraSpec
        self.cam = CameraSpec(
            name="c", width=1280, height=720, fov_y=48, eye=(1.9, -0.7, 1.6), look_at=(0.6, 0.1, 0.75)
        )
        rng = np.random.default_rng(3)
        self.world = np.c_[rng.uniform(0.2, 1.0, 10), rng.uniform(-0.6, 0.6, 10), rng.uniform(0.75, 1.1, 10)]
        self.pixels = self.cam.project(self.world)[0]
        self.run_dir = str(self.enterContext(short_dir("cam-")))

    def _points(self, pixels=None) -> list[str]:
        pixels = self.pixels if pixels is None else pixels
        out = []
        for (x, y, z), (u, v) in zip(self.world, pixels, strict=True):
            out += ["--point", f"{x},{y},{z}={u},{v}"]
        return out

    def test_the_printed_camera_puts_the_points_back_on_their_pixels(self) -> None:
        argv = ["--run-dir", self.run_dir, "--json", "camera", "fit", *self._points(), "--size", "1280X720"]
        code, out, err = _ctl([*argv, "--fov", "48", "--name", "phone"])
        self.assertEqual(0, code, err)
        reply = json.loads(out)
        self.assertNotIn("warning", reply)
        spec = eval(reply["camera"], {"rss": types.SimpleNamespace(CameraSpec=self.CameraSpec)})
        self.assertEqual(("phone", 1280, 720), (spec.name, spec.width, spec.height))
        self.assertLess(np.abs(spec.project(self.world)[0] - self.pixels).max(), 0.5)

    def test_a_point_read_off_wrong_draws_the_warning(self) -> None:
        pixels = self.pixels.copy()
        pixels[4] += (300.0, 0.0)
        argv = ["--run-dir", self.run_dir, "--json", "camera", "fit", *self._points(pixels), "--size", "1280x720"]
        code, out, _ = _ctl([*argv, "--fov", "48"])
        self.assertEqual(0, code)
        self.assertIn("is no camera", json.loads(out)["warning"])

    def test_no_size_and_no_still_is_refused(self) -> None:
        code, _, err = _ctl(["--run-dir", self.run_dir, "camera", "fit", *self._points(), "--fov", "48"])
        self.assertEqual(1, code)
        self.assertEqual("ctl: give --ref (the still the pixels came from) or --size WxH\n", err)

    def test_a_point_with_no_pixel_is_refused_in_one_line(self) -> None:
        argv = ["--run-dir", self.run_dir, "camera", "fit", "--point", "0.6,0.3,0.75", "--size", "640x480"]
        code, _, err = _ctl(argv)
        self.assertEqual(1, code)
        self.assertTrue(err.startswith("ctl: a point is x,y,z=u,v"), err)
        self.assertEqual(1, err.count("\n"))

    def test_too_few_points_are_refused_saying_how_many_it_needs(self) -> None:
        argv = ["--run-dir", self.run_dir, "camera", "fit", *self._points()[:6], "--size", "1280x720", "--fov", "48"]
        code, _, err = _ctl(argv)
        self.assertEqual(1, code)
        self.assertIn("3 points; this fit needs at least 4", err)


class TestRefPlateWithoutAPage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = self.enterContext(short_dir("plate-"))
        self.enterContext(contextlib.chdir(self.tmp))  # no scene.py here either

    def test_no_clip_is_refused(self) -> None:
        code, _, err = _ctl(["--run-dir", str(self.tmp / "run"), "ref", "plate"])
        self.assertEqual((2, "ctl: ref plate needs the clip\n"), (code, err))

    def test_nowhere_to_put_the_plate_is_refused(self) -> None:
        code, _, err = _ctl(["--run-dir", str(self.tmp / "run"), "ref", "plate", "clip.mov"])
        self.assertEqual(2, code)
        self.assertIn("give --out", err)

    def test_a_clip_that_will_not_open_is_one_line_and_exit_1(self) -> None:
        clip = self.tmp / "clip.mov"
        clip.write_bytes(b"not a clip")
        code, _, err = _ctl(["--run-dir", str(self.tmp / "run"), "ref", "plate", str(clip), "--out", "plates"])
        self.assertEqual(1, code)
        self.assertTrue(err.startswith("ctl: "), err)
        self.assertFalse((self.tmp / "plates" / "plate_clip.png").exists())


class TestPhysicsExitCodes(unittest.TestCase):
    def test_no_page_exits_2_and_says_how_to_start_one(self) -> None:
        with short_dir("phys-") as tmp:
            code, _, err = _ctl(["--run-dir", str(tmp), "physics"])
        self.assertEqual(2, code)
        self.assertIn("no server at", err)
        self.assertIn(ctl.START_HINT, err)

    def test_a_page_that_stops_answering_exits_1_saying_what_it_is_busy_with(self) -> None:
        def page(sock, req, timeout):
            if req["cmd"] == "status" and timeout == 5.0:  # the busy note's own light status
                return {"ok": True, "busy": "rebuilding the scene file", "busy_seconds": 40.0}
            raise wire.NoReply("the page did not answer within 30 s")

        with mock.patch.object(ctl, "request", side_effect=page):
            code, _, err = _ctl(["physics"])
        self.assertEqual(1, code)
        self.assertIn("ctl: the page did not answer within 30 s; the page is busy rebuilding the scene file", err)

    def test_a_page_that_refuses_a_replay_exits_1_and_is_resumed(self) -> None:
        asked = []

        def page(sock, req, timeout):
            asked.append(req["cmd"] + (":" + req["op"] if req.get("op") else ""))
            if req["cmd"] == "physics":
                return {"ok": False, "error": "the scene binds no `routine`"}
            return {"ok": True, "playing": 0, "paused": False, "resets": len([a for a in asked if a == "reset"])}

        with (
            mock.patch.object(ctl, "request", side_effect=page),
            mock.patch.object(ctl, "poll", return_value={"ok": True}),
            mock.patch.object(ctl, "wait_until_free", return_value={"ok": True}),
        ):
            code, _, err = _ctl(["physics"])
        self.assertEqual(1, code)
        self.assertIn("binds no `routine`", err)
        self.assertEqual("resume", asked[-1], "a page that was running is left running")


if __name__ == "__main__":
    unittest.main()
