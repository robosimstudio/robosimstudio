# SPDX-License-Identifier: Apache-2.0
"""The live page's server, with no page: runs, routines, takes, the board, status and rebuilds.

A run ends saying how; a routine that takes the session is stepped by the
page through the arm panels, and one that does not has them held off and
given back; a take ends with its own routine; ``status`` is answered at once
while the main thread builds; a scene that will not go in is let go before
the stand-in is built. All of it is bookkeeping on a ``LiveServer`` whose
session and sim are stand-ins.
"""

from __future__ import annotations

import gc
import inspect
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio import registry
from robosimstudio_real2sim_agent import checks, ctl
from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import (
    Loaded,
    helper_files,
)
from robosimstudio_real2sim_agent.server import LiveServer
from tests.fixtures.live_server import FakePage, bare_live_server
from tests.fixtures.sockets import short_dir
from tests.fixtures.studio_fakes import FakeArmPanel

TIER = "fast"
"""No GPU and no page: the server's state machine over stand-in sessions and sims."""


def _server() -> LiveServer:
    """A server whose session holds two arm panels and whose takes nobody writes."""
    session = types.SimpleNamespace(arms={"left": FakeArmPanel("left"), "right": FakeArmPanel("right")})
    live = bare_live_server(session=session)
    live._finish_recorder = lambda why: None  # type: ignore[method-assign]
    return live


def _pass(live: LiveServer, page_steps: int = 1) -> None:
    """One loop pass with the page's own step stood in: it steps when the page would."""

    def tick() -> bool:
        if not live.script or live.page_steps_script():
            live.sim.step_count += page_steps
        return False

    live.bus = types.SimpleNamespace(pump=lambda budget: 0)
    live.session.tick = tick
    live._take_clicks = lambda: None  # type: ignore[method-assign]
    live.watch_enabled, live._tick_error, live._refreshed = False, "", time.perf_counter()
    live.tick_once()


class RunEndings(unittest.TestCase):
    def _page(self) -> LiveServer:
        live = _server()
        live.session = types.SimpleNamespace(arms={}, paused=True)
        live.session.set_paused = lambda paused: setattr(live.session, "paused", paused)
        live.sim = types.SimpleNamespace(step_count=10, state_is_finite=lambda: True)
        return live

    def test_a_run_ends_done_and_held_at_its_last_step(self) -> None:
        live = self._page()
        reply = live.cmd_run(2)
        self.assertEqual((reply["run"], reply["target"]), (1, 12))

        def tick() -> bool:
            live.sim.step_count += 1
            return True

        live.bus = types.SimpleNamespace(pump=lambda budget: 0)
        live.session.tick = tick
        live._take_clicks = lambda: None  # type: ignore[method-assign]
        live._refreshed = time.perf_counter()
        live.tick_once()
        self.assertEqual(live.last_run, {})
        live.tick_once()
        self.assertEqual((live.last_run["how"], live.last_run["step"], live.last_run["finite"]), ("done", 12, True))
        self.assertTrue(live.session.paused)

    def test_a_pause_a_routine_or_another_run_ends_it_saying_which(self) -> None:
        live = self._page()
        live.cmd_run(100)
        live.cmd_pause()
        self.assertEqual(live.last_run["how"], "paused")
        live.cmd_run(100)
        live.start_script(iter(()), name="lift")
        self.assertEqual(live.last_run["how"], "cut short: a routine started")
        self.assertTrue(live.session.paused, "held, not running on unbounded once the routine ends")
        live.script.clear()
        live.cmd_run(100)
        live.cmd_run(5)
        self.assertEqual((live.last_run["serial"], live.last_run["how"]), (3, "another run started"))


class RunEndingsKept(unittest.TestCase):
    def test_an_earlier_runs_ending_is_kept_for_its_waiting_client(self) -> None:
        live = RunEndings._page(self)  # type: ignore[arg-type]
        live.cmd_run(600)
        live.cmd_run(5)  # another client's run, ended before the first client polls again
        live._end_run("done")
        self.assertEqual(live.runs_ended[1]["how"], "another run started")
        self.assertEqual(live.runs_ended[2]["how"], "done")

    def test_a_failing_loop_pass_ends_the_run_saying_so(self) -> None:
        live = RunEndings._page(self)  # type: ignore[arg-type]
        live.cmd_run(120)

        def tick() -> bool:
            raise RuntimeError("the drain broke")

        live.bus = types.SimpleNamespace(pump=lambda budget: 0)
        live.session.tick, live.session.status = tick, ""
        live._take_clicks = lambda: None  # type: ignore[method-assign]
        live.refresh_live = lambda force=False: None  # type: ignore[method-assign]
        live._refreshed = time.perf_counter()
        with mock.patch("builtins.print"):
            live.tick_once()
        self.assertIn("the loop failed: RuntimeError: the drain broke", live.last_run["how"])

    def test_a_run_ended_by_a_failing_pass_whose_checkbox_is_gone_still_ends(self) -> None:
        live = RunEndings._page(self)  # type: ignore[arg-type]
        live.cmd_run(120)

        def broken(paused: bool) -> None:
            live.session.paused = paused
            raise RuntimeError("the checkbox is mid-teardown")

        live.session.set_paused = broken
        live._end_run("the loop failed")  # inside the loop's own failure handler: it must not raise
        self.assertEqual((live.last_run["how"], live.session.paused), ("the loop failed", True))

    def test_eval_that_calls_exit_is_an_error_not_the_pages_end(self) -> None:
        live = _server()
        live.ns, live.scene, live.rendered = {}, None, True
        live.session = types.SimpleNamespace(arms={}, server=None)
        reply = live.dispatch({"cmd": "eval", "code": "import sys; sys.exit(3)"})
        self.assertFalse(reply["ok"])
        self.assertIn("SystemExit", reply["error"])

    def test_a_run_of_no_steps_is_refused(self) -> None:
        live = RunEndings._page(self)  # type: ignore[arg-type]
        self.assertFalse(live.cmd_run(0)["ok"])
        self.assertEqual(live.runs, 0)

    def test_a_command_the_client_gave_up_on_is_not_run(self) -> None:
        live = _server()

        def call_starting_within(timeout, fn, request):
            self.assertLessEqual(timeout, 28.0, "the page gives up just before the client does")
            raise server_mod.BusTimeout("dispatch was not serviced")

        live.bus = types.SimpleNamespace(call_starting_within=call_starting_within, timeout=900.0)
        for cmd in ("run", "quit"):  # one rule for every command
            reply = live._on_request({"cmd": cmd, "steps": 120, "timeout": 30.0})
            self.assertFalse(reply["ok"])
            self.assertIn(f"`{cmd}` was not run", reply["error"])

    def test_a_client_that_says_no_wait_gets_the_buses_and_the_same_rule(self) -> None:
        live = _server()
        waited: list[float] = []
        live.bus = types.SimpleNamespace(
            call_starting_within=lambda timeout, fn, request: waited.append(timeout) or {"ok": True}, timeout=900.0
        )
        live._on_request({"cmd": "eval", "code": "1"})
        self.assertEqual(waited, [900.0])


class StartUpStandIn(unittest.TestCase):
    def test_a_scene_that_builds_but_will_not_go_in_at_start_up_is_stood_in_for(self) -> None:
        from robosimstudio_studio.session import StudioConfig

        folder = Path(tempfile.mkdtemp())
        (folder / "scene.py").write_text("")
        live = LiveServer(
            scene_path=folder / "scene.py", sock_path=folder / "s.sock", run_dir=folder / "run", config=StudioConfig()
        )
        made: list[str] = []

        class Session:
            def __init__(self, sim, *, config, host) -> None:
                if sim == "the twin":
                    raise RuntimeError("adopt failed")
                made.append(sim)
                self.server = types.SimpleNamespace(get_port=lambda: 8080)

            def begin_run(self) -> None:
                pass

        bare = types.SimpleNamespace(ns={}, scene="bare", sim="the bare scene", seconds=0.0)
        live.load = lambda files=None: Loaded(ns={}, scene="twin", sim="the twin", seconds=1.0)  # type: ignore[method-assign]
        live.refresh_live = lambda force=False: None  # type: ignore[method-assign]
        self.addCleanup(registry.unwatch, live.registrations)
        self.addCleanup(lambda: sys.path.remove(str(folder.resolve())) if str(folder.resolve()) in sys.path else None)
        with (
            mock.patch.object(server_mod, "LiveSession", Session),
            mock.patch.object(server_mod, "fallback_loaded", lambda: bare),
            mock.patch("builtins.print"),
        ):
            live.open_page()
        self.assertEqual((made, live.serving, live.scene_id), (["the bare scene"], "stand-in", "bare"))
        self.assertIn("adopt failed", live.last_error)


class RoutineStepping(unittest.TestCase):
    def test_a_routine_that_takes_the_session_is_stepped_by_the_page_through_live_arms(self) -> None:
        live = _server()

        def glide():
            for _ in range(3):
                yield

        serial = live.start_script(glide(), name="glide", drives_page=True)
        _pass(live)
        self.assertTrue(live.page_steps_script())
        self.assertFalse(live.session.arms["left"].suspended)
        for _ in range(4):
            if live.script:
                _pass(live)
        self.assertEqual(live.last_script["serial"], serial)
        self.assertEqual(live.last_script["how"], "finished")
        self.assertEqual(live.last_script["steps"], 3, "the page's steps for it are its steps")

    def test_one_that_does_not_has_the_arms_held_off_and_given_back(self) -> None:
        live = _server()
        live.session.arms["right"].suspended = True  # held off already: not ours to give back

        def drive():
            for _ in range(2):
                live.sim.step_count += 1
                yield

        live.start_script(drive(), name="drive", drives_page=False)
        _pass(live)
        self.assertTrue(live.head_stepped)
        self.assertFalse(live.page_steps_script())
        self.assertEqual(live.session.arms["left"].calls, ["suspend"])
        _pass(live)
        _pass(live)
        self.assertEqual(live.session.arms["left"].calls, ["suspend", "resume"])
        self.assertEqual(live.session.arms["right"].calls, [])
        self.assertEqual(live.last_script["steps"], 2)

    def test_one_that_drives_the_scene_and_yields_an_extra_time_is_not_stepped_twice(self) -> None:
        live = _server()

        def drive():
            for _ in range(3):
                live.sim.step_count += 1
                yield
                yield  # to let the page redraw

        live.start_script(drive(), name="drive", drives_page=False)
        while live.script:
            _pass(live)
        self.assertEqual(live.last_script["steps"], 3)

    def test_one_that_drives_the_scene_and_never_steps_it_is_stopped_saying_why(self) -> None:
        live = _server()

        def waiting():
            while True:
                yield  # waits on a page that does not step for it

        live.start_script(waiting(), name="lid")
        for _ in range(server_mod.IDLE_LIMIT + 1):
            if live.script:
                _pass(live)
        self.assertEqual(live.script, [])
        self.assertEqual(live.last_script["how"], "failed")
        self.assertIn("give the routine a `session` parameter", live.last_script["error"])

    def test_one_stopped_idle_whose_finally_raises_still_ends_and_gives_the_arms_back(self) -> None:
        live = _server()

        def stubborn():
            try:
                while True:
                    yield
            finally:
                raise RuntimeError("cleanup failed")

        serial = live.start_script(stubborn(), name="lid")
        live.start_script(iter(range(3)), name="next")
        for _ in range(server_mod.IDLE_LIMIT):
            _pass(live)
        self.assertEqual(live.ended[serial]["how"], "failed")
        self.assertIn("without stepping", live.ended[serial]["error"])
        self.assertEqual(live.session.arms["left"].calls, ["suspend", "resume"], "given back, then the next takes them")

    def test_an_arm_a_session_routine_held_off_itself_is_given_back_when_it_ends(self) -> None:
        live = _server()

        def holds_right():
            live.session.arms["right"].suspend()
            yield
            raise RuntimeError("broke with the arm held off")

        live.start_script(holds_right(), name="hold", drives_page=True)
        _pass(live)
        _pass(live)
        self.assertEqual(live.session.arms["right"].calls, ["suspend", "resume"])
        self.assertFalse(live.session.arms["right"].suspended)

    def test_an_arm_held_off_before_a_routine_that_took_it_back_is_held_off_again(self) -> None:
        live = _server()
        live.session.arms["left"].suspended = True

        def takes_left():
            live.session.arms["left"].resume()
            yield

        live.start_script(takes_left(), name="take", drives_page=True)
        _pass(live)
        _pass(live)
        self.assertTrue(live.session.arms["left"].suspended)

    def test_a_recording_under_an_old_take_name_does_not_count_as_written(self) -> None:
        live = _server()
        live.takes_written = ["take1", "take0"]
        fake = types.SimpleNamespace(
            start=lambda fresh=False: None, probe=lambda: None, close=lambda: None, name="take1"
        )
        with mock.patch.object(server_mod.checks, "Recorder", lambda *a, **k: fake):
            live.start_script(iter(range(3)), name="again", record="take1")
        self.assertEqual(live.takes_written, ["take0"])

    def test_the_page_steps_a_session_routine_while_paused(self) -> None:
        from robosimstudio_real2sim_agent.page import LiveSession

        stepped = {"now": True}
        page = types.SimpleNamespace(host=types.SimpleNamespace(page_steps_script=lambda: stepped["now"]))
        self.assertTrue(LiveSession.steps_forced(page))
        stepped["now"] = False
        self.assertFalse(LiveSession.steps_forced(page))

    def test_a_routine_queued_behind_another_is_not_stepped_for_before_it_plays(self) -> None:
        live = _server()
        live.start_script(iter(range(1)), name="a", drives_page=False)
        live.start_script(iter(range(5)), name="b", drives_page=True)
        live._play()  # a: one step
        live._play()  # a ends; b is at the head, not played yet
        self.assertFalse(live.page_steps_script())

    def test_a_session_routine_that_reaches_settles_and_reaches_again_is_driven_throughout(self) -> None:
        live = _server()

        def both():
            yield  # reach: the page steps
            live.sim.step_count += 30  # sim.run(30): it steps itself
            yield
            yield  # reach again: the page steps, through the arms, which never left

        live.start_script(both(), name="both", drives_page=True)
        _pass(live)
        self.assertTrue(live.page_steps_script())
        _pass(live)
        self.assertFalse(live.page_steps_script(), "a pass it stepped itself is not stepped twice")
        _pass(live)
        self.assertTrue(live.page_steps_script())
        self.assertEqual(live.session.arms["left"].calls, [], "its arms were never taken from it")
        _pass(live)
        self.assertEqual(live.last_script["steps"], 1 + 30 + 1)

    def test_a_routine_whose_finally_raises_still_ends_and_the_queue_empties(self) -> None:
        live = _server()

        def stubborn():
            try:
                while True:
                    yield
            finally:
                raise RuntimeError("cleanup failed")

        gen = stubborn()
        live.start_script(gen, name="stubborn")
        live.start_script(iter(range(3)), name="next")
        live._play()
        live._end_scripts("stopped: the scene was rebuilt")
        self.assertEqual(live.script, [])
        self.assertIn("cleanup failed", live.last_error)

    def test_an_error_raised_ending_the_routines_before_a_new_one_stays(self) -> None:
        live = _server()
        live.build_error, live._tick_error, live.last_error = "", "", "an old routine's failure"

        def stubborn():
            try:
                while True:
                    yield
            finally:
                raise RuntimeError("cleanup failed")

        live.start_script(stubborn(), name="old")
        live._play()
        fake = types.SimpleNamespace(start=lambda fresh=False: None, probe=lambda: None, close=lambda: None)
        with mock.patch.object(server_mod.checks, "Recorder", lambda *a, **k: fake):
            live.start_script(iter(range(3)), name="new", record="take2")  # ends the old one first
        self.assertIn("cleanup failed", live.last_error)

    def test_a_failure_is_reported_with_its_traceback(self) -> None:
        live = _server()

        def broken():
            yield
            raise RuntimeError("the right hand stopped 278 mm short")

        live.start_script(broken(), name="broken")
        live._play()
        live._play()
        self.assertEqual(live.last_script["how"], "failed")
        self.assertIn("278 mm short", live.last_script["error"])


class BusyStatus(unittest.TestCase):
    def _busy_server(self) -> LiveServer:
        live = _server()
        live.url, live.share_url, live.run_dir, live.generation = "http://x", None, Path("/tmp"), 3
        live._loaded_stamp, live.watch_enabled = None, True
        release = threading.Event()
        self.queued = 0

        def call(fn, req):
            self.queued += 1
            return release.wait(5) or {"ok": True, "late": True}

        live.bus = types.SimpleNamespace(call=call)
        self.addCleanup(release.set)
        return live

    def test_a_client_polling_through_a_long_render_queues_one_status(self) -> None:
        live = self._busy_server()
        live.busy_state = ("rendering a still", time.perf_counter(), False)
        for _ in range(5):
            self.assertEqual(live._on_request({"cmd": "status"})["busy"], "rendering a still")
        self.assertEqual(self.queued, 1)

    def test_a_save_to_a_helper_beside_the_scene_is_pending_too(self) -> None:
        live = self._busy_server()
        folder = Path(tempfile.mkdtemp())
        live.path, live.watch_enabled = folder / "scene.py", True
        live.path.write_text("x = 1\n")
        helper = folder / "lidlib.py"
        helper.write_text("A = 1\n")
        live._loaded_stamp = live._stamps()
        self.assertFalse(live._pending_save())
        helper.write_text("A = 22\n")
        self.assertTrue(live._pending_save())

    def _reloading(self, build) -> tuple[LiveServer, Path]:
        """A server over a folder whose rebuild runs ``build(helper)`` in place of the scene file."""
        live = self._busy_server()
        folder = Path(tempfile.mkdtemp())
        live.path, live.watch_enabled = folder / "scene.py", True
        live.path.write_text("import lidlib\n")
        helper = folder / "lidlib.py"
        helper.write_text("A = 1\n")
        live.session.status = ""
        live.refresh_live = lambda force=False: None  # type: ignore[method-assign]
        live.swap = lambda loaded: True  # type: ignore[method-assign]
        live.load = lambda files=None: build(helper)  # type: ignore[method-assign]
        return live, helper

    def test_a_helper_saved_while_the_scene_builds_is_built_next(self) -> None:
        def build(helper: Path):
            helper.write_text("A = 2222\n")  # saved ten seconds into a minute-long build
            return types.SimpleNamespace(seconds=60.0, sim=types.SimpleNamespace(names=lambda: []))

        live, _ = self._reloading(build)
        self.assertTrue(live.reload("saved")["ok"])
        self.assertTrue(live._pending_save(), "the helper changed after the build read it")

    def test_fixing_a_helper_that_did_not_import_rebuilds(self) -> None:
        def build(helper: Path):
            raise SyntaxError("lidlib.py: invalid syntax")  # and Python drops lidlib from sys.modules

        live, helper = self._reloading(build)
        live.build_error = live.last_error = ""
        self.assertFalse(live.reload("saved")["ok"])
        self.assertFalse(live._pending_save())
        helper.write_text("A = 33\n")  # a length of its own: two writes in one clock tick share an mtime
        self.assertTrue(live._pending_save(), "a helper is watched whether its import worked or not")

    def test_the_files_watched_are_the_scenes_folder_but_its_readings_and_references(self) -> None:
        live = self._busy_server()
        folder = Path(tempfile.mkdtemp())
        live.path = folder / "scene.py"
        for rel in (
            "scene.py",
            "lidlib.py",
            "parts/__init__.py",
            "parts/hinge/latch.py",
            "parts/hinge/deep/too_deep.py",
            "run/probe.py",
            "ref/fit.py",
            "assets/gen.py",
            "twins/other/scene.py",
            ".hidden/x.py",
            "robosimstudio_materials/cloth.py",
            "notes.txt",
            "robosimstudio_tools.py",
            "parts/robosimstudio_x.py",
        ):
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_text("")
        watched = sorted(str(p.relative_to(folder)) for p in helper_files(live.path))
        self.assertEqual(
            watched, ["lidlib.py", "parts/__init__.py", "parts/hinge/latch.py", "parts/robosimstudio_x.py"]
        )

    def test_no_save_is_pending_on_a_page_that_does_not_watch(self) -> None:
        live = self._busy_server()
        live.path = Path(tempfile.mkdtemp()) / "scene.py"
        live.path.write_text("x = 1\n")
        live.watch_enabled = False
        self.assertFalse(live._pending_save())
        live.watch_enabled = True
        self.assertTrue(live._pending_save())
        live.path.unlink()
        self.assertFalse(live._pending_save(), "a file that is gone is no save to wait for")

    def test_status_is_answered_at_once_while_the_main_thread_builds(self) -> None:
        live = self._busy_server()
        live.busy_state = ("rebuilding the scene file", time.perf_counter() - 41, False)
        started = time.perf_counter()
        reply = live._on_request({"cmd": "status"})
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(reply["busy"], "rebuilding the scene file")
        self.assertGreaterEqual(reply["busy_seconds"], 41)

    def test_a_waiting_poll_is_answered_off_the_main_thread_reading_nothing_off_the_scene(self) -> None:
        live = self._busy_server()
        live.sim = types.SimpleNamespace(step_count=42, time=0.7)
        live.session = types.SimpleNamespace(paused=False, arms={})
        live._speed = __import__("collections").deque()
        reply = live._on_request({"cmd": "status", "light": True})
        self.assertEqual((reply["step"], reply["time"], reply["busy"]), (42, 0.7, ""))
        self.assertEqual(self.queued, 0, "nothing was put on the main thread's bus")

    def test_only_a_path_traced_picture_counts_as_the_first_render(self) -> None:
        live = _server()
        live.rendered, live.build_error = False, ""
        live.cmd_compare = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        live.cmd_snapshot = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        live.dispatch({"cmd": "compare", "take": "take1"})
        live.dispatch({"cmd": "snapshot", "browser": True})
        self.assertFalse(live.rendered)
        live.dispatch({"cmd": "snapshot", "camera": "phone"})
        self.assertTrue(live.rendered)

    def test_a_status_waits_out_a_routine_step_only_so_long(self) -> None:
        live = self._busy_server()
        live.busy_state = ("playing one step of a routine", time.perf_counter(), True)
        started = time.perf_counter()
        with mock.patch.object(server_mod, "ROUTINE_WAIT", 0.5):
            reply = live._on_request({"cmd": "status"})
        self.assertLess(time.perf_counter() - started, 3.0)
        self.assertTrue(reply["busy_routine"])

    def test_a_routine_started_after_a_failed_save_keeps_its_traceback(self) -> None:
        live = _server()
        live.build_error, live.last_error = "Traceback ...\nNameError: lid", "RoutineError: a broke"
        live.start_script(iter(()))
        self.assertEqual(live.last_error, live.build_error)

    def test_an_eval_that_defines_a_routine_and_ends_calling_it_plays_it(self) -> None:
        live = _server()
        live.ns, live.scene = {}, None
        live.session = types.SimpleNamespace(arms={}, server=None)
        code = "def lift():\n    yield\n\nlift()\n"
        reply = live.cmd_eval(code=code)
        self.assertTrue(reply["playing"])
        self.assertEqual(reply["serial"], 1)
        self.assertEqual(live.cmd_eval(code="x = 2\nx * 21")["value"], "42")
        self.assertIsNone(live.cmd_eval(code="y = 1")["value"])

    def test_a_command_while_a_save_is_not_built_says_so(self) -> None:
        live = _server()
        live.rendered, live.watch_enabled = True, True
        live._pending_save = lambda stamps=None: True  # type: ignore[method-assign]
        live.cmd_eval = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        self.assertIn("not built yet", live.dispatch({"cmd": "eval", "code": "1"})["pending"])

    def test_a_run_button_on_an_empty_page_is_refused_as_ctl_play_is(self) -> None:
        live = _server()
        live.rendered, live.serving = True, "nothing"
        live.build_error = "Traceback ...\nRuntimeError: CUDA out of memory"
        live.clicks = __import__("queue").Queue()
        live.clicks.put(("play", "lift"))
        live.refresh_live = lambda force=False: None  # type: ignore[method-assign]
        live._take_clicks()
        self.assertIn("no scene is in the page", live.last_error)

    def test_a_reload_keeps_the_walk_it_stamped_with(self) -> None:
        def build(helper: Path):
            return types.SimpleNamespace(seconds=1.0, sim=types.SimpleNamespace(names=lambda: []))

        live, helper = self._reloading(build)
        live._walked = (time.perf_counter(), [])  # a walk from before the helper was made
        self.assertTrue(live.reload("saved")["ok"])
        self.assertEqual(live._walked[1], [helper.resolve()])
        self.assertFalse(live._pending_save(), "status reads the walk the build was stamped with")

    def test_a_reload_ties_the_log_to_the_files_its_build_read(self) -> None:
        def build(helper: Path):
            helper.write_text("A = 2222\n")  # saved during the build: not what it read
            return types.SimpleNamespace(seconds=1.0, sim=types.SimpleNamespace(names=lambda: []))

        live, _helper = self._reloading(build)
        before = checks.scene_hash(live.path)
        live.scene_id = "old"
        self.assertTrue(live.reload("saved")["ok"])
        self.assertEqual(live.scene_id, before)
        live.load = lambda files=None: (_ for _ in ()).throw(SyntaxError("x"))  # type: ignore[method-assign]
        live.reload("saved")
        self.assertEqual(live.scene_id, before, "a save that did not build leaves the scene, and its hash, up")

    def test_a_command_on_the_stand_in_scene_says_it_is_not_the_users(self) -> None:
        live = _server()
        live.rendered, live.serving = True, "stand-in"
        live.build_error = "Traceback ...\nRuntimeError: adopt failed"
        live.cmd_eval = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        reply = live.dispatch({"cmd": "eval", "code": "1"})
        self.assertIn("a bare scene stands in for it", reply["stale"])
        self.assertNotIn("the scene before it", reply["stale"])

    def test_with_no_scene_in_the_page_its_commands_are_refused_and_status_says_why(self) -> None:
        live = _server()
        live.rendered, live.serving = True, "nothing"
        live.build_error = live.last_error = "Traceback ...\nRuntimeError: CUDA out of memory"
        live.url, live.share_url, live.run_dir, live.generation = "http://x", None, Path("/tmp"), 3
        live._loaded_stamp, live.watch_enabled = None, False
        live.session.paused = False
        live._speed = __import__("collections").deque()
        live.cmd_eval = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        reply = live.dispatch({"cmd": "eval", "code": "1"})
        self.assertFalse(reply["ok"])
        self.assertIn("no scene is in the page", reply["error"])
        status = live.dispatch({"cmd": "status"})
        self.assertEqual((status["serving"], status["last_error"]), ("nothing", live.build_error))

    def test_a_status_whose_request_fails_oddly_is_still_a_reply(self) -> None:
        live = self._busy_server()

        def call(fn, req):
            raise RuntimeError("the bus broke")

        live.bus = types.SimpleNamespace(call=call)
        reply = live._on_request({"cmd": "status"})
        self.assertFalse(reply["ok"])
        self.assertIn("the bus broke", reply["error"])

    def test_the_page_step_after_a_reset_is_not_credited_to_the_routine_it_ended(self) -> None:
        live = _server()

        def glide():
            for _ in range(5):
                yield

        serial = live.start_script(glide(), name="glide", drives_page=True)
        _pass(live)  # one yield, one page step

        def reset_then_step() -> bool:
            live._end_scripts("stopped: reset")  # what a Reset pressed on the page does in this tick
            live.sim.step_count += 1
            return False

        live._play = lambda: 0  # type: ignore[method-assign]
        live.session.tick = reset_then_step
        live.tick_once()
        self.assertEqual(live.ended[serial]["steps"], 1)

    def test_a_command_after_a_failed_save_says_the_scene_is_the_one_before(self) -> None:
        live = _server()
        live.rendered = True
        live.build_error = "Traceback ...\nNameError: name 'lid' is not defined"
        live.cmd_status = lambda: {"ok": True}  # type: ignore[method-assign]
        live.cmd_eval = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        reply = live.dispatch({"cmd": "eval", "code": "1"})
        self.assertIn("did not build (NameError: name 'lid' is not defined)", reply["stale"])


class TheSceneThatWouldNotGoIn(unittest.TestCase):
    """A scene that builds but will not go into the page is given back before the bare one is built in its place."""

    class Twin:
        def __init__(self, made: list, closed: list) -> None:
            made.append(weakref.ref(self))
            self.closed = closed

        def close(self) -> None:
            self.closed.append("sim")

    def _rebuild(self, bare_builds: bool = True) -> tuple:
        """A rebuild whose scene will not go in; what the stand-in's build saw, the reply and the server."""
        live = _server()
        made: list = []
        closed: list[str] = []
        seen: list = []

        def adopt(sim, *, label: str = "") -> None:
            if isinstance(sim, self.Twin):
                raise RuntimeError("adopt failed")

        def bare() -> types.SimpleNamespace:
            gc.collect()
            seen.append((made[0]() is None, list(closed), live.sim))
            if not bare_builds:
                raise RuntimeError("the bare scene failed")
            return types.SimpleNamespace(ns={}, scene="bare", sim="the bare scene", seconds=0.0)

        live.session = types.SimpleNamespace(
            adopt=adopt,
            begin_run=lambda: None,
            status="",
            teardown=lambda: closed.append("teardown"),
            viewer=types.SimpleNamespace(set_model=lambda model: closed.append("the viewer let go: the camera resets")),
        )
        live.generation, live.ns, live.scene = 0, {"routine": None}, None
        live._end_scripts = lambda why: None  # type: ignore[method-assign]
        live._end_run = lambda why, hold=False: None  # type: ignore[method-assign]
        live.refresh_live = lambda force=False: None  # type: ignore[method-assign]
        live.load = lambda files=None: Loaded(ns={}, scene="twin", sim=self.Twin(made, closed), seconds=1.0)  # type: ignore[method-assign]
        with (
            mock.patch.object(server_mod, "helper_files", lambda path: []),
            mock.patch.object(server_mod.checks, "scene_hash", lambda path, files: "h"),
            mock.patch.object(server_mod, "fallback_loaded", bare),
            mock.patch("builtins.print"),
        ):
            reply = live.reload("saved")
        return seen, reply, live

    def test_a_rebuild_lets_it_go_before_the_stand_in_is_built(self) -> None:
        seen, reply, live = self._rebuild()
        self.assertEqual(seen, [(True, ["teardown", "sim"], None)], "nothing of it held, the camera where it was")
        self.assertEqual(
            (reply["ok"], live.serving, live.scene_id, live.sim), (False, "stand-in", "bare", "the bare scene")
        )
        self.assertIn("adopt failed", live.last_error)

    def test_when_the_bare_scene_fails_too_nothing_is_served_and_the_names_stay_usable(self) -> None:
        _seen, reply, live = self._rebuild(bare_builds=False)
        self.assertEqual((reply["ok"], live.serving, live.scene_id, live.ns), (False, "nothing", "", {}))
        self.assertIn("the bare scene that should have replaced it failed too", live.last_error)

    def test_at_start_up_it_is_let_go_before_the_stand_in_is_built(self) -> None:
        from robosimstudio_studio.session import StudioConfig

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        (folder / "scene.py").write_text("")
        live = LiveServer(
            scene_path=folder / "scene.py", sock_path=folder / "s.sock", run_dir=folder / "run", config=StudioConfig()
        )
        made: list = []
        closed: list[str] = []
        seen: list = []
        twin = self.Twin

        class Session:
            def __init__(self, sim, *, config, host) -> None:
                if isinstance(sim, twin):
                    raise RuntimeError("adopt failed")
                self.server = types.SimpleNamespace(get_port=lambda: 8080)

            def begin_run(self) -> None:
                pass

        def bare() -> types.SimpleNamespace:
            gc.collect()
            seen.append((made[0]() is None, list(closed), live.sim))
            return types.SimpleNamespace(ns={}, scene="bare", sim="the bare scene", seconds=0.0)

        live.load = lambda files=None: Loaded(ns={}, scene="twin", sim=twin(made, closed), seconds=1.0)  # type: ignore[method-assign]
        live.refresh_live = lambda force=False: None  # type: ignore[method-assign]
        self.addCleanup(registry.unwatch, live.registrations)
        self.addCleanup(lambda: sys.path.remove(str(folder.resolve())) if str(folder.resolve()) in sys.path else None)
        with (
            mock.patch.object(server_mod, "LiveSession", Session),
            mock.patch.object(server_mod, "fallback_loaded", bare),
            mock.patch("builtins.print"),
        ):
            live.open_page()
        self.assertEqual(seen, [(True, ["sim"], None)], "nothing of it held while building")
        self.assertEqual((live.serving, live.scene_id), ("stand-in", "bare"))


class HeldUpByARoutine(unittest.TestCase):
    def test_a_command_a_routines_step_held_up_says_it_goes_in_between_steps(self) -> None:
        live = _server()
        live.busy_state = ("playing one step of a routine", time.perf_counter(), True)

        def call_starting_within(timeout, fn, request):
            raise server_mod.BusTimeout("dispatch was not serviced")

        live.bus = types.SimpleNamespace(call_starting_within=call_starting_within, timeout=900.0)
        reply = live._on_request({"cmd": "quit", "timeout": ctl.TIMEOUTS["quit"]})
        self.assertIn("between two of the routine's steps", reply["error"])
        self.assertNotIn("status --wait", reply["error"], "which does not wait out a routine")
        self.assertIn("serve.pid", reply["error"], "the way out of a step that never ends")
        for cmd in ("quit", "reset", "pause", "resume"):
            self.assertGreaterEqual(ctl.TIMEOUTS[cmd], ctl.TIMEOUTS["reload"], f"{cmd} waits out a long step")


def _lifecycle_server() -> LiveServer:
    """A server over a sim and a session that answer anything, for what serve and show do around them."""
    return bare_live_server(sim=mock.MagicMock(step_count=0), session=mock.MagicMock(paused=False))


class ServeFinishesTheTake(unittest.TestCase):
    def test_quit_mid_take_finishes_the_recorder(self) -> None:
        live = _lifecycle_server()
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
        live = _lifecycle_server()
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
        live = _lifecycle_server()
        image = np.zeros((2, 2, 3), np.uint8)
        with mock.patch.object(server_mod.stills, "load_reference", return_value=(image, "x")):
            for i in range(server_mod.BOARD_LIMIT + 5):
                live.cmd_show("p.png", label=f"compare {i}")
        self.assertEqual(len(live.board), server_mod.BOARD_LIMIT)
        self.assertEqual(next(iter(live.board)), "compare 5")
        live.session.clear.assert_any_call("compare 0")

    def test_a_label_shown_again_is_the_newest(self) -> None:
        live = _lifecycle_server()
        image = np.zeros((2, 2, 3), np.uint8)
        with mock.patch.object(server_mod.stills, "load_reference", return_value=(image, "x")):
            live.cmd_show("p.png", label="a")
            live.cmd_show("p.png", label="b")
            live.cmd_show("p.png", label="a")
        self.assertEqual(list(live.board), ["b", "a"])


class CheckUnderARoutine(unittest.TestCase):
    def test_check_is_refused_while_a_routine_plays(self) -> None:
        live = _lifecycle_server()
        live.script = [iter(())]
        with mock.patch.object(server_mod.checks, "gate") as gate:
            reply = live.cmd_check()
        self.assertFalse(reply["ok"])
        self.assertIn("routine is playing", reply["error"])
        gate.assert_not_called()
        self.assertEqual(len(live.script), 1)


class RoutineNames(unittest.TestCase):
    def test_a_non_str_key_plays(self) -> None:
        live = _lifecycle_server()
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
        live = _lifecycle_server()
        live.sim = object()

        def fn(sim):
            yield

        live.ns = {"routines": {"routine": fn}}
        live.start_script = mock.MagicMock()
        reply = live.cmd_play("routine")
        self.assertTrue(reply["ok"], reply)


def _script_server(root: Path | None = None) -> LiveServer:
    """A server over ``root`` (a fresh folder when not given) whose sim answers anything."""
    return bare_live_server(root, sim=mock.MagicMock(step_count=0))


class FakeRecorder:
    def __init__(self, server, name, *, camera="", every=4, also=()) -> None:
        if camera == "phnoe":
            raise KeyError("no camera named phnoe")
        self.name = name
        self.frames: list[int] = []
        self.closed = False
        self.fail = False

    def capture(self) -> None:
        self.frames.append(-1)

    def start(self, fresh: bool = False) -> None:
        self.capture()

    def probe(self) -> None:
        if self.name == "dark":
            raise RuntimeError("the path tracer failed")

    def maybe(self) -> None:
        if self.fail:
            raise RuntimeError("broken pipe")
        self.frames.append(0)

    def close(self) -> None:
        self.closed = True


class ScriptQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.live = _script_server()
        self.finished: list[str] = []
        self.live._finish_recorder = self.finished.append  # type: ignore[method-assign]
        patcher = mock.patch.object(server_mod.checks, "Recorder", FakeRecorder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_new_recording_replaces_what_was_queued(self) -> None:
        old = iter(range(100))
        new = iter(range(3))
        self.live.start_script(old, record="t1")
        self.live.start_script(new, record="t2")
        self.assertEqual(self.live.script, [new])
        self.assertEqual(self.finished, ["another recording started", "another recording started"])
        self.live._play()
        self.assertEqual(next(new), 1)  # the new routine stepped, not the old one

    def test_a_recording_whose_first_frame_fails_leaves_what_plays_and_its_take(self) -> None:
        playing = iter(range(5))
        self.live.start_script(playing, record="t1")
        filming = self.live.recorder
        self.finished.clear()
        with self.assertRaises(RuntimeError):
            self.live.start_script((s for s in range(3)), record="dark")
        self.assertEqual((self.live.script, self.finished), ([playing], []))
        self.assertIs(self.live.recorder, filming)

    def test_a_recording_that_cannot_start_leaves_what_plays_and_its_take(self) -> None:
        playing, new = iter(range(5)), (step for step in range(3))
        self.live.start_script(playing, record="t1")
        filming = self.live.recorder
        self.finished.clear()
        with self.assertRaises(KeyError):
            self.live.start_script(new, record="t2", camera="phnoe")
        self.assertEqual(self.live.script, [playing])
        self.assertIs(self.live.recorder, filming)
        self.assertEqual(self.finished, [], "t1 is not cut short")
        self.assertEqual(inspect.getgeneratorstate(new), "GEN_CLOSED", "the routine that will never play is closed")

    def test_a_take_ends_with_its_own_routine_not_with_the_queue(self) -> None:
        filmed, queued = iter(range(1)), iter(range(5))
        self.live.start_script(filmed, record="t1")
        self.live.start_script(queued)  # a Run click behind it
        self.finished.clear()
        self.live._play()  # its one yield
        self.live._play()  # and its end
        self.assertEqual(self.finished, ["the script ended"])
        self.assertEqual(self.live.script, [queued], "the routine behind it plays on, not filmed into the take")

    def test_a_new_recording_writes_the_take_before_the_routines_it_drops_clean_up(self) -> None:
        self.live.start_script(iter(range(5)), record="t1")
        order: list[str] = []
        self.live._finish_recorder = lambda why: order.append("take")  # type: ignore[method-assign]
        self.live._end_scripts = lambda how: order.append("scripts")  # type: ignore[method-assign]
        started = FakeRecorder.start
        with mock.patch.object(
            FakeRecorder, "start", lambda rec, fresh=False: (order.append(f"start fresh={fresh}"), started(rec, fresh))
        ):
            self.live.start_script(iter(range(3)), record="t2")
        # after their cleanup, and filmed afresh: the cleanup may have moved things without a step
        self.assertEqual(order, ["take", "scripts", "start fresh=True"])

    def test_a_routine_whose_cleanup_calls_exit_is_stopped_and_the_page_goes_on(self) -> None:
        def leave():
            try:
                yield
            finally:
                raise SystemExit(0)

        gen = leave()
        self.live.start_script(gen)
        self.live._play()
        with mock.patch("builtins.print"):
            self.live.stop_scripts("reset")
        self.assertEqual(self.live.script, [])
        self.assertIn("SystemExit", self.live.last_error)

    def test_a_routine_that_calls_exit_fails_and_the_page_goes_on(self) -> None:
        def leave():
            yield
            raise SystemExit(0)

        gen = leave()
        self.live.start_script(gen)
        self.live._play()
        with mock.patch("builtins.print"):
            self.live._play()
        self.assertEqual(self.live.script, [])
        self.assertEqual(self.live.last_script["how"], "failed")

    def test_a_take_is_filmed_to_the_reset_before_its_routine_cleans_up(self) -> None:
        order: list[str] = []
        self.live._finish_recorder = lambda why: order.append("take")  # type: ignore[method-assign]
        self.live._end_scripts = lambda how: order.append("scripts")  # type: ignore[method-assign]
        self.live.stop_scripts("reset")
        self.assertEqual(order, ["take", "scripts"])

    def test_a_take_written_whose_strip_fails_is_still_written(self) -> None:
        live = self.live
        live.recorder = types.SimpleNamespace(
            name="t1", frames=[], finish=lambda: {"take": "t1", "frames": 3, "seconds": 0.1}
        )
        live.run_dir = Path(tempfile.mkdtemp())

        def show(*args):
            raise ValueError("timeline.json: Expecting property name")

        live._show_take = show  # type: ignore[method-assign]
        with (
            mock.patch.object(server_mod.checks, "take_motion", lambda frames: {}),
            mock.patch.object(server_mod.checks, "motion_line", lambda motion: ""),
            mock.patch("builtins.print"),
        ):
            live._write_take("the script ended")
        self.assertEqual(live.takes_written, ["t1"])
        self.assertIn("was written, but putting it beside the clip failed", live.last_error)
        self.assertIsNone(live.recorder)

    def test_a_failing_recording_leaves_the_routine_playing(self) -> None:
        gen = iter(range(5))
        self.live.start_script(gen, record="t1")
        recorder = self.live.recorder
        recorder.fail = True
        self.live._play()
        self.assertEqual(self.live.script, [gen])
        self.assertIsNone(self.live.recorder)
        self.assertTrue(recorder.closed)
        self.assertIn("the recording of 't1' failed", self.live.last_error, "status says why the take is missing")
        self.live._play()
        self.assertEqual(next(gen), 2)


class _Session:
    def __init__(self) -> None:
        self.paused = False

    def set_paused(self, paused: bool) -> None:
        self.paused = paused


def _running(budget: int) -> LiveServer:
    """A server half way through ``run 100`` (step 50), ``budget`` steps of it left."""
    host = bare_live_server(
        sim=types.SimpleNamespace(step_count=50, state_is_finite=lambda: True),
        session=_Session(),
        run_budget=budget,
        runs=1,
        run_target=100,
    )
    host._finish_recorder = lambda why: None  # type: ignore[method-assign]
    host._end_scripts = lambda how: None  # type: ignore[method-assign]
    return host


class TestResetUnderARun(unittest.TestCase):
    def test_a_reset_ends_a_run_held(self) -> None:
        host = _running(50)
        host.stop_scripts("reset")
        self.assertEqual(host.run_budget, 0)
        self.assertTrue(host.session.paused, "the run ends paused, as it would have at its last step")
        self.assertEqual((host.last_run["how"], host.last_run["step"]), ("cut short: reset", 50))

    def test_a_free_running_page_is_left_running(self) -> None:
        host = _running(-1)
        host.stop_scripts("reset")
        self.assertEqual(host.run_budget, -1)
        self.assertFalse(host.session.paused)
        self.assertEqual(host.last_run, {})

    def test_the_wait_says_the_run_was_reset(self) -> None:
        path = self.enterContext(short_dir("run-")) / "ctl.sock"
        ended = {"serial": 1, "how": "cut short: reset", "step": 0, "target": 10, "finite": None}
        page = FakePage(path, {"step": 0, "time": 0.0, "runs_ended": {"1": ended}})
        self.addCleanup(page.close)
        reply = ctl.wait_for_steps(path, {"ok": True, "run": 1, "step": 0, "target": 10}, 30.0)
        self.assertFalse(reply["ok"])
        self.assertIn("reset", reply["error"])


if __name__ == "__main__":
    unittest.main()
