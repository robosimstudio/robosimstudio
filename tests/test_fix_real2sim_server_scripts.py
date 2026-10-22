# SPDX-License-Identifier: Apache-2.0
"""The scene file's `sim`, helper eviction, and the script queue under a recording: no GPU."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import ScenePaths, run_scene_file
from robosimstudio_real2sim_agent.server import LiveServer

TIER = "fast"

SCENE = """
import robosimstudio as rss
scene = rss.Scene(cell="void", name="t")
{sim}
"""


class SceneFileSim(unittest.TestCase):
    def _run(self, sim_line: str):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.py"
            path.write_text(SCENE.format(sim=sim_line))
            return run_scene_file(path, build=lambda scene, **kw: calls.append(kw) or "built"), calls

    def test_unbound_sim_is_built(self) -> None:
        loaded, calls = self._run("")
        self.assertEqual(loaded.sim, "built")
        self.assertEqual(len(calls), 1)

    def test_a_bound_sim_wins_over_building(self) -> None:
        loaded, calls = self._run("from unittest import mock\nsim = mock.MagicMock(spec=rss.Sim)")
        self.assertIsInstance(loaded.sim, mock.MagicMock)
        self.assertEqual(calls, [])

    def test_a_module_named_sim_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "bound `sim` to module"):
            self._run("import robosimstudio.sim as sim")

    def test_a_dict_named_sim_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "bound `sim` to dict"):
            self._run("sim = {'dt': 0.01}")


def _bare_server(root: Path) -> LiveServer:
    live = LiveServer.__new__(LiveServer)
    live.path = root / "scene.py"
    live.head_stepped, live._head_ready = False, None
    live.ended, live.takes_written, live.serving = {}, [], "scene"
    live.runs, live.run_target, live.scene_id, live.watch_enabled = 0, 0, "", False
    live.runs_ended, live._walked = {}, (-1.0, [])
    live._arms_before, live.resets = {}, 0
    live.log_path, live._motions = "", {}
    live.scene_paths = ScenePaths(root)
    live.build_error, live._tick_error = "", ""
    live.busy_state = ("", 0.0, False)
    live.script_meta = {}
    live.scripts_started = 0
    live.last_script = {}
    live.sim = mock.MagicMock(step_count=0)
    live.session = types.SimpleNamespace(arms={})
    live.script = []
    live.recorder = None
    live.run_budget = 0
    live.last_error = ""
    return live


class EvictHelpers(unittest.TestCase):
    def test_installed_packages_under_the_scene_dir_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = root / "helper.py"
            lib = root / ".venv" / "lib" / "python3.12" / "site-packages" / "pkgx" / "__init__.py"
            for f in (helper, lib):
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("")
            mods = {
                "fix_r35_helper": types.ModuleType("fix_r35_helper"),
                "fix_r35_pkgx": types.ModuleType("fix_r35_pkgx"),
            }
            mods["fix_r35_helper"].__file__ = str(helper)
            mods["fix_r35_pkgx"].__file__ = str(lib)
            with mock.patch.dict(sys.modules, mods):
                gone = _bare_server(root).evict_helpers()
                self.assertIn("fix_r35_pkgx", sys.modules)
            self.assertEqual(list(gone), ["fix_r35_helper"])

    def test_an_evicted_helper_is_read_from_its_source_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = root / "lidlib.py"
            helper.write_text("K = 0.1\n")
            cached = Path(importlib.util.cache_from_source(str(helper)))
            cached.parent.mkdir()
            cached.write_bytes(b"stale")  # trusted over a same-second, same-length edit
            os.utime(cached, ns=(helper.stat().st_mtime_ns - 10**8,) * 2)  # compiled before the source's last save
            module = types.ModuleType("fix_r4_lidlib")
            module.__file__ = str(helper)
            with mock.patch.dict(sys.modules, {"fix_r4_lidlib": module}):
                _bare_server(root).evict_helpers()
            self.assertTrue(_hash_checked(cached), "checked against the text at import, not the mtime")

    def test_a_build_that_fails_keeps_the_helpers_the_scene_up_was_built_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "scene.py").write_text("")
            (root / "lidlib.py").write_text("A = 1\n")
            old = types.ModuleType("fix_r7_lidlib")
            old.__file__ = str(root / "lidlib.py")
            live = _bare_server(root)
            live.registrations = types.SimpleNamespace(undo=lambda: [], redo=lambda taken: None)

            def run(path):
                new = types.ModuleType("fix_r7_lidlib")
                new.__file__ = str(root / "lidlib.py")
                sys.modules["fix_r7_lidlib"] = new  # the edited helper imported, then the scene fails
                raise NameError("lid")

            with (
                mock.patch.dict(sys.modules, {"fix_r7_lidlib": old}),
                mock.patch.object(server_mod, "run_scene_file", run),
            ):
                with self.assertRaises(NameError):
                    live.load()
                self.assertIs(sys.modules["fix_r7_lidlib"], old)

    def test_a_helpers_bytecode_goes_even_when_newer_than_its_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = root / "lidlib.py"
            helper.write_text("K = 0.1\n")
            cached = Path(importlib.util.cache_from_source(str(helper)))
            cached.parent.mkdir()
            cached.write_bytes(b"compiled from the text before a same-second save")
            os.utime(cached, ns=(helper.stat().st_mtime_ns + 10**8,) * 2)  # written after, read before
            _bare_server(root).evict_helpers()
            self.assertTrue(_hash_checked(cached), "nothing in a timestamp says which text it was compiled from")

    def test_a_helper_whose_import_failed_is_read_from_its_source_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = root / "parts" / "hinge.py"
            helper.parent.mkdir()
            helper.write_text("raise RuntimeError\n")
            cached = Path(importlib.util.cache_from_source(str(helper)))
            cached.parent.mkdir()
            cached.write_bytes(b"compiled before it raised")  # and no module left in sys.modules to evict
            os.utime(cached, ns=(helper.stat().st_mtime_ns - 10**8,) * 2)
            _bare_server(root).evict_helpers()
            self.assertTrue(_hash_checked(cached), "checked against the text at import, not the mtime")


def _hash_checked(cached: Path) -> bool:
    head = cached.read_bytes()[:8]
    return int.from_bytes(head[4:8], "little") & 0b11 == 0b11


class FakeRecorder:
    def __init__(self, server, name, *, camera="", every=4) -> None:
        if camera == "phnoe":
            raise KeyError("no camera named phnoe")
        self.name = name
        self.frames: list[int] = []
        self.closed = False
        self.fail = False

    def capture(self) -> None:
        self.frames.append(-1)

    def start(self) -> None:
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
        self.live = _bare_server(Path(tempfile.gettempdir()))
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
        with mock.patch.object(FakeRecorder, "start", lambda rec: (order.append("start"), started(rec))):
            self.live.start_script(iter(range(3)), record="t2")
        self.assertEqual(order, ["take", "scripts", "start"], "the new take starts after their cleanup")

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


if __name__ == "__main__":
    unittest.main()
