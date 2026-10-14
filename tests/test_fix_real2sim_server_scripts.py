# SPDX-License-Identifier: Apache-2.0
"""The scene file's `sim`, helper eviction, and the script queue under a recording: no GPU."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import run_scene_file
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
    live._baseline_modules = set()
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
            self.assertEqual(gone, ["fix_r35_helper"])


class FakeRecorder:
    def __init__(self, server, name, *, camera="", every=4) -> None:
        self.name = name
        self.frames: list[int] = []
        self.closed = False
        self.fail = False

    def capture(self) -> None:
        self.frames.append(-1)

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

    def test_a_failing_recording_leaves_the_routine_playing(self) -> None:
        gen = iter(range(5))
        self.live.start_script(gen, record="t1")
        recorder = self.live.recorder
        recorder.fail = True
        self.live._play()
        self.assertEqual(self.live.script, [gen])
        self.assertIsNone(self.live.recorder)
        self.assertTrue(recorder.closed)
        self.assertEqual(self.live.last_error, "")
        self.live._play()
        self.assertEqual(next(gen), 2)


if __name__ == "__main__":
    unittest.main()
