# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_real2sim_agent`` with no page: the file's contract, the wire, the command line, and ``up``.

:mod:`tests.test_agent_server` starts a page on a GPU and drives it the way
an agent does. This is the half that needs neither a GPU nor a browser, and it
is the half that gates every commit: the scene file's contract holds on files
that never build, the socket answers one line for one line with a handler
that is not a simulator, every command line parses and every ``--help`` exits
0, the client imports nothing of the library, and ``up`` recognises a server
that came up, one that died, and one that is still building.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio_real2sim_agent import ctl, scenefile, stills, wire

TIER = "fast"
"""No GPU, no page: files that are compiled, a socket on a temp path, subprocesses that only parse."""

ROOT = Path(__file__).resolve().parents[1]
SCENE = 'import robosimstudio as rss\nscene = rss.Scene(cell="void", name="t")\n'


def _write(directory: Path, text: str, name: str = "scene.py") -> Path:
    path = directory / name
    path.write_text(text)
    return path


class TestTheSceneFileContract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agent-"))
        self.calls: list[dict] = []

    def build(self, scene, **kwargs):
        self.calls.append(kwargs)
        return ("built", scene)

    def test_a_file_that_binds_no_scene_is_refused(self) -> None:
        path = _write(self.tmp, "x = 1\n")
        with self.assertRaises(TypeError) as caught:
            scenefile.run_scene_file(path, build=self.build)
        self.assertIn("scene", str(caught.exception))
        self.assertEqual(self.calls, [], "nothing is built for a file with no scene")

    def test_a_bound_sim_wins_over_building(self) -> None:
        loaded = scenefile.run_scene_file(
            _write(self.tmp, SCENE + "sim = rss.Sim.__new__(rss.Sim)\n"), build=self.build
        )
        self.assertIsInstance(loaded.sim, rss.Sim)
        self.assertEqual(self.calls, [])
        self.assertIsInstance(loaded.scene, rss.Scene)
        self.assertGreaterEqual(loaded.seconds, 0.0)

    def test_build_is_the_default_control_mode_with_BUILD_over_it(self) -> None:
        scenefile.run_scene_file(_write(self.tmp, SCENE), build=self.build)
        self.assertEqual(self.calls, [{"control_mode": "joint_position"}])
        scenefile.run_scene_file(
            _write(self.tmp, SCENE + 'BUILD = {"control_mode": "osc", "x": 1}\n'),
            build=self.build,
        )
        self.assertEqual(
            self.calls[-1],
            {"control_mode": "osc", "x": 1},
            "BUILD merges over the default",
        )

    def test_the_doors_are_shut_while_the_file_runs_and_open_after(self) -> None:
        doors = {name: getattr(rss, name) for name in scenefile.DOORS}
        text = (
            SCENE
            + "opened = [rss.view(None), rss.draw(None), rss.viewer(None)]\nfrom robosimstudio import view\nvia = view(1)\n"
        )
        loaded = scenefile.run_scene_file(_write(self.tmp, text), build=self.build)
        self.assertEqual(
            loaded.ns["opened"],
            [None, None, None],
            "a page-opening call is a no-op while the file runs",
        )
        self.assertIsNone(
            loaded.ns["via"],
            "reached through `from robosimstudio import`, still the no-op",
        )
        for name, door in doors.items():
            self.assertIs(getattr(rss, name), door, f"rss.{name} is the real one again after")
        with self.assertRaises(ZeroDivisionError):
            scenefile.run_scene_file(_write(self.tmp, "1 / 0\n"), build=self.build)
        for name, door in doors.items():
            self.assertIs(
                getattr(rss, name),
                door,
                f"rss.{name} is restored even when the file raises",
            )

    def test_the_file_runs_as_scene_not_main(self) -> None:
        text = SCENE + 'if __name__ == "__main__":\n    raise RuntimeError("a main guard fired")\n'
        loaded = scenefile.run_scene_file(_write(self.tmp, text), build=self.build)
        self.assertEqual(loaded.ns["__name__"], "scene")

    def test_stamp_and_the_fallback(self) -> None:
        self.assertIsNone(scenefile.stamp(self.tmp / "absent.py"))
        path = _write(self.tmp, "a = 1\n")
        first = scenefile.stamp(path)
        self.assertIsInstance(first, tuple)
        path.write_text("a = 12\n")
        self.assertNotEqual(scenefile.stamp(path), first, "a longer file is a different stamp")
        self.assertIsInstance(scenefile.fallback_scene(), rss.Scene)


class TestTheWire(unittest.TestCase):
    def test_where_the_socket_is(self) -> None:
        self.assertEqual(wire.socket_path(None), Path("run") / "ctl.sock")
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
        self.assertEqual(wire.socket_path("d"), Path("d") / "ctl.sock")
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
        tmp = Path(tempfile.mkdtemp(prefix="w-"))
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
        path = Path(tempfile.mkdtemp(prefix="w-")) / "ctl.sock"
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
        path = Path(tempfile.mkdtemp(prefix="w-")) / ("s" * 200 + ".sock")
        with self.assertRaises(OSError):
            wire.ControlSocket(path, lambda request: {"ok": True}).start()


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
    )

    def test_every_command_parses(self) -> None:
        parser = ctl.build_parser()
        for line in self.LINES:
            with self.subTest(line=line):
                self.assertEqual(parser.parse_args(line).cmd, line[0])
        args = parser.parse_args(["--run-dir", "d", "--json", "status"])
        self.assertEqual((args.run_dir, args.json), ("d", True), "--json before the command")
        self.assertTrue(parser.parse_args(["status", "--json"]).json, "--json after the command")
        self.assertFalse(parser.parse_args(["status"]).json)

    def test_the_request_a_line_becomes(self) -> None:
        parser = ctl.build_parser()
        req, timeout = ctl.build_request(parser.parse_args(["snapshot", "--beside", "ref.png"]), parser)
        self.assertEqual(req["cmd"], "snapshot")
        self.assertTrue(
            Path(req["beside"]).is_absolute(),
            "paths are resolved where the caller sits",
        )
        self.assertEqual(timeout, ctl.TIMEOUTS["snapshot"])
        req, _ = ctl.build_request(parser.parse_args(["catalog", "robots", "cells"]), parser)
        self.assertEqual(req, {"cmd": "catalog", "kinds": ["robots", "cells"]})
        with self.assertRaises(SystemExit):
            ctl.build_request(parser.parse_args(["catalog", "planets"]), parser)
        req, timeout = ctl.build_request(parser.parse_args(["run", "7"]), parser)
        self.assertEqual((req, timeout), ({"cmd": "run", "steps": 7}, ctl.DEFAULT_TIMEOUT))
        req, timeout = ctl.build_request(
            parser.parse_args(
                [
                    "compare",
                    "--ref",
                    "r.jpg",
                    "--key",
                    "case=red",
                    "--key",
                    "socket=@1,2",
                    "--note",
                    "n",
                ]
            ),
            parser,
        )
        self.assertEqual(
            (req["cmd"], req["keys"], req["note"]),
            ("compare", ["case=red", "socket=@1,2"], "n"),
        )
        self.assertTrue(Path(req["ref"]).is_absolute())
        self.assertEqual(timeout, ctl.TIMEOUTS["compare"])
        req, _ = ctl.build_request(
            parser.parse_args(["check", "--object", "a", "--object", "b", "--steps", "9"]),
            parser,
        )
        self.assertEqual((req["objects"], req["steps"]), (["a", "b"], 9))
        self.assertNotIn("ref", req, "the gate asks nothing of a photograph")
        req, _ = ctl.build_request(parser.parse_args(["ref", "frames", "c.mov", "--at", "0,end"]), parser)
        self.assertEqual((req["op"], req["at"]), ("frames", "0,end"))
        # no --out: the server puts the stills in ref/ beside scene.py, wherever the shell stands
        self.assertTrue(Path(req["path"]).is_absolute())
        self.assertEqual(req["out"], "")
        req, _ = ctl.build_request(parser.parse_args(["ref", "frames", "c.mov", "--out", "elsewhere"]), parser)
        self.assertTrue(Path(req["out"]).is_absolute())
        with self.assertRaises(SystemExit):
            ctl.build_request(parser.parse_args(["ref", "frames"]), parser)
        req, _ = ctl.build_request(
            parser.parse_args(["eval", "go()", "--record", "take1", "--every", "2"]),
            parser,
        )
        self.assertEqual((req["code"], req["record"], req["every"]), ("go()", "take1", 2))
        req, _ = ctl.build_request(parser.parse_args(["eval", "go()"]), parser)
        self.assertNotIn("record", req, "no recording unless asked")
        req, _ = ctl.build_request(parser.parse_args(["log", "--last", "3"]), parser)
        self.assertEqual(req, {"cmd": "log", "last": 3})
        req, _ = ctl.build_request(parser.parse_args(["play", "tidy", "--record", "t1"]), parser)
        self.assertEqual((req["name"], req["record"], req["every"]), ("tidy", "t1", 4))
        req, _ = ctl.build_request(parser.parse_args(["show", "p.png", "--label", "pair"]), parser)
        self.assertTrue(Path(req["path"]).is_absolute() and req["label"] == "pair" and not req["clear"])
        self.assertEqual(ctl.build_request(parser.parse_args(["show", "--clear"]), parser)[0]["clear"], True)
        with self.assertRaises(SystemExit):
            ctl.build_request(parser.parse_args(["show"]), parser)

    def test_every_help_exits_zero(self) -> None:
        lines = [[sys.executable, "-m", "robosimstudio_real2sim_agent", "--help"]]
        lines += [[sys.executable, "-m", "robosimstudio_real2sim_agent", line[0], "--help"] for line in self.LINES]
        lines += [
            [sys.executable, "-m", "robosimstudio_real2sim_agent.serve", "--help"],
            [sys.executable, str(ROOT / "scripts" / "live.py"), "--help"],
        ]
        for line in lines:
            with self.subTest(line=" ".join(line[1:])):
                done = subprocess.run(
                    line,
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    timeout=300,
                    check=False,
                )
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                if line[-2].endswith(("serve", "live.py")):
                    self.assertIn("--camera-view", done.stdout)
                    self.assertIn("--share", done.stdout)
                    self.assertNotIn(
                        "--live",
                        done.stdout,
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
        tmp = Path(tempfile.mkdtemp(prefix="nosrv-"))
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


class _FakePage:
    """A socket answering ``status`` from a dict the test changes, for the client's half."""

    def __init__(self, path: Path, status: dict) -> None:
        self.status = dict(status)
        self.socket = wire.ControlSocket(path, lambda request: {"ok": True, **self.status})
        self.socket.start()

    def close(self) -> None:
        self.socket.close()


class TestTheRunWait(unittest.TestCase):
    """``run N`` polls until N steps are taken; what ends the run otherwise must end the wait too."""

    def setUp(self) -> None:
        self.path = Path(tempfile.mkdtemp(prefix="run-")) / "ctl.sock"
        base = {"step": 0, "time": 0.0, "finite": True, "paused": False, "playing": 0, "generation": 1}
        self.page = _FakePage(self.path, base)
        self.addCleanup(self.page.close)

    def test_a_rebuild_under_the_run_ends_it(self) -> None:
        self.page.status.update(step=3, generation=2)
        started = time.time()
        reply = ctl.wait_for_steps(self.path, {"ok": True, "step": 40, "generation": 1}, 1000)
        self.assertFalse(reply["ok"])
        self.assertIn("rebuilt", reply["error"])
        self.assertLess(time.time() - started, 5.0, "said at once, not at the deadline")

    def test_a_reset_under_the_run_ends_it(self) -> None:
        self.page.status.update(step=2)
        reply = ctl.wait_for_steps(self.path, {"ok": True, "step": 40, "generation": 1}, 1000)
        self.assertFalse(reply["ok"])
        self.assertIn("reset", reply["error"])

    def test_steps_taken_end_it(self) -> None:
        self.page.status.update(step=50)
        reply = ctl.wait_for_steps(self.path, {"ok": True, "step": 40, "generation": 1}, 10)
        self.assertEqual(reply, {"ok": True, "step": 50, "time": 0.0, "finite": True})

    def test_play_waits_as_long_as_eval(self) -> None:
        self.assertEqual(ctl.TIMEOUTS["play"], ctl.TIMEOUTS["eval"])


class TestUp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="up-"))
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

        def spawn(argv, log_path):
            proc = real([sys.executable, "-c", code], log_path)
            self.procs.append(proc)
            return proc

        with mock.patch.object(ctl, "spawn_server", spawn):
            rc = ctl.main(["--run-dir", str(run_dir), "up", "--scene", str(scene), "--timeout", "30"])
        self.assertEqual(rc, 0)
        self.assertEqual(int((run_dir / "serve.pid").read_text()), self.procs[0].pid)

    def test_up_on_a_running_page_says_when_it_is_not_the_one_asked_for(self) -> None:
        scene = _write(self.tmp, SCENE)
        page = _FakePage(self.tmp / "ctl.sock", {"file": str(scene.resolve()), "share_url": None, "url": "u", "pid": 1})
        self.addCleanup(page.close)
        argv = ["--run-dir", str(self.tmp), "up", "--scene"]
        self.assertEqual(ctl.main([*argv, str(scene)]), 0, "the same file: a no-op naming the page")
        other = _write(self.tmp, SCENE, name="other.py")
        self.assertEqual(ctl.main([*argv, str(other)]), 1, "another file is not silently ignored")
        self.assertEqual(ctl.main([*argv, str(scene), "--share"]), 1, "nor is --share on a page without one")

    def test_up_refuses_a_missing_scene_file(self) -> None:
        rc = ctl.main(["--run-dir", str(self.tmp), "up", "--scene", str(self.tmp / "absent.py")])
        self.assertEqual(rc, 2)
        self.assertFalse(self.log.exists(), "nothing was started")


class TestStills(unittest.TestCase):
    def test_parsing(self) -> None:
        self.assertEqual(stills.parse_vec("1, 2.5,-3"), (1.0, 2.5, -3.0))
        with self.assertRaises(ValueError):
            stills.parse_vec("1,2")
        self.assertEqual(stills.parse_size("", (4, 3)), (4, 3))
        self.assertEqual(stills.parse_size("320X180", (4, 3)), (320, 180))

    def test_to_uint8(self) -> None:
        self.assertEqual(stills.to_uint8(np.full((2, 2, 3), 0.5)).tolist()[0][0], [127, 127, 127])
        kept = np.full((2, 2, 3), 200, dtype=np.uint8)
        self.assertIs(stills.to_uint8(kept).dtype.type, np.uint8)
        self.assertEqual(
            int(stills.to_uint8(np.full((1, 1, 3), 300.0))[0, 0, 0]),
            255,
            "clipped, not wrapped",
        )

    def test_the_reference_stands_beside_the_render_at_its_height(self) -> None:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="pair-"))
        ref = tmp / "ref.png"
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(ref)
        render = np.full((8, 8, 3), 255, dtype=np.uint8)
        pair = stills.side_by_side(render, str(ref))
        self.assertEqual(pair.shape, (8, 8 + 12, 3), "the reference is scaled to the render's height")
        self.assertEqual(int(pair[0, 0, 0]), 255)
        image, label = stills.load_reference(str(ref), width=42)
        self.assertEqual((image.shape[1], label), (42, "ref.png"))
        out = stills.save_png(pair, tmp / "deep" / "pair.png")
        self.assertTrue(out.exists(), "save_png makes the directory")


if __name__ == "__main__":
    unittest.main()
