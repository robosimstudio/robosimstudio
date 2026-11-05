# SPDX-License-Identifier: Apache-2.0
"""The scene file's contract: what it must bind, what runs it, and which files are the scene's own.

A file that binds no scene is refused, a bound ``sim`` wins over building,
the doors are shut while it runs, and the helpers beside it are watched,
hashed, evicted and read afresh -- while the library's own modules, and what
a registry discovered for itself, are left alone.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import robosimstudio as rss
from robosimstudio import registry
from robosimstudio.registry import MISSING, Registry
from robosimstudio_real2sim_agent import checks, scenefile
from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import (
    ScenePaths,
    SceneRegistrations,
    hash_checked,
    helper_files,
    run_scene_file,
)
from robosimstudio_real2sim_agent.server import LiveServer
from tests.fixtures.live_server import bare_live_server

TIER = "fast"
"""No GPU: scene files that are compiled and run against a ``build`` that only records."""


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


SCENE_WITH_SIM = """
import robosimstudio as rss
scene = rss.Scene(cell="void", name="t")
{sim}
"""


class SceneFileSim(unittest.TestCase):
    def _run(self, sim_line: str):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.py"
            path.write_text(SCENE_WITH_SIM.format(sim=sim_line))
            return run_scene_file(path, build=lambda scene, **kw: calls.append(kw) or "built"), calls

    def test_unbound_sim_is_built(self) -> None:
        loaded, calls = self._run("")
        self.assertEqual(loaded.sim, "built")
        self.assertEqual(len(calls), 1)

    def test_a_module_named_sim_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "bound `sim` to module"):
            self._run("import robosimstudio.sim as sim")

    def test_a_dict_named_sim_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "bound `sim` to dict"):
            self._run("sim = {'dt': 0.01}")


def _script_server(root: Path) -> LiveServer:
    """A server over the scene folder ``root`` whose sim answers anything."""
    return bare_live_server(root, sim=mock.MagicMock(step_count=0))


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
                gone = _script_server(root).evict_helpers()
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
                _script_server(root).evict_helpers()
            self.assertTrue(_hash_checked(cached), "checked against the text at import, not the mtime")

    def test_a_build_that_fails_keeps_the_helpers_the_scene_up_was_built_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "scene.py").write_text("")
            (root / "lidlib.py").write_text("A = 1\n")
            old = types.ModuleType("fix_r7_lidlib")
            old.__file__ = str(root / "lidlib.py")
            live = _script_server(root)
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
            _script_server(root).evict_helpers()
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
            _script_server(root).evict_helpers()
            self.assertTrue(_hash_checked(cached), "checked against the text at import, not the mtime")


def _hash_checked(cached: Path) -> bool:
    head = cached.read_bytes()[:8]
    return int.from_bytes(head[4:8], "little") & 0b11 == 0b11


class SceneCode(unittest.TestCase):
    def test_one_rule_for_what_is_the_scenes_own_code(self) -> None:
        from robosimstudio_real2sim_agent.scenefile import ScenePaths

        root = Path(tempfile.mkdtemp())
        (root / "helper.py").write_text("")
        (root / "lib.py").write_text("")
        sys.modules["_r2s_baseline_lib"] = types.SimpleNamespace(__file__=str(root / "lib.py"))  # type: ignore[assignment]
        self.addCleanup(sys.modules.pop, "_r2s_baseline_lib", None)
        paths = ScenePaths(root)
        paths.set_baseline({"_r2s_baseline_lib"})
        self.assertTrue(paths.is_scene(str(root / "helper.py")))
        self.assertFalse(paths.is_scene(str(root / "lib.py")), "imported before the scene ran: the library's")
        self.assertFalse(paths.is_scene("/elsewhere/x.py"))
        (root / "robosimstudio_materials").mkdir()
        (root / "robosimstudio_materials" / "bag.py").write_text("")
        self.assertFalse(
            paths.is_scene(str(root / "robosimstudio_materials" / "bag.py"), "robosimstudio_materials.bag"),
            "an installed distribution's module is never the scene's, wherever the scene file sits",
        )

    def test_the_modules_imported_afresh_are_the_files_watched(self) -> None:
        from robosimstudio_real2sim_agent.scenefile import ScenePaths

        root = Path(tempfile.mkdtemp()).resolve()
        for rel in (
            "scene.py",
            "rig.py",
            "robosimstudio_tools.py",
            "parts/robosimstudio_x.py",
            "parts/hinge/latch.py",
            "parts/a/b/deep.py",
            "assets/gen.py",
            "run/p.py",
        ):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("")
        paths = ScenePaths(root)
        paths.set_baseline(set())
        watched = set(helper_files(root / "scene.py"))
        for file in sorted(root.rglob("*.py")):
            if file.name != "scene.py":
                module = ".".join(file.relative_to(root).with_suffix("").parts)  # as `import` names it
                self.assertEqual(paths.is_scene(str(file), module), file in watched, module)

    def test_a_watcher_is_unwatched_by_identity(self) -> None:
        from robosimstudio_real2sim_agent.scenefile import SceneRegistrations

        root = Path(tempfile.mkdtemp())
        a, b = SceneRegistrations(root), SceneRegistrations(root)
        self.assertNotEqual(a, b)
        registry.watch(a)
        registry.watch(b)
        registry.unwatch(b)
        self.assertIn(a, registry._WATCHERS)
        registry.unwatch(a)


class RegistryWrites(unittest.TestCase):
    def test_forget_restore_and_watch(self) -> None:
        reg: Registry[int] = Registry("thing")
        seen = []
        fn = registry.watch(lambda r, name, value, origin, prev, prev_origin: seen.append((name, value, prev)))
        try:
            reg.register("a", 1)
            reg.replace("a", 2)
        finally:
            registry.unwatch(fn)
        self.assertEqual(seen, [("a", 1, MISSING), ("a", 2, 1)])
        reg.forget("a")
        self.assertNotIn("a", reg._items)
        reg.restore("a", 1)
        self.assertEqual(reg._items["a"], 1)


class SceneRegistrationsTakeBack(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.reg: Registry[type] = Registry("entity")
        self.reg.register("library_thing", int)
        self.notes = SceneRegistrations(self.tmp)
        registry.watch(self.notes)
        self.addCleanup(registry.unwatch, self.notes)
        sys.modules["_r2s_test_registry"] = types.SimpleNamespace(REG=self.reg)  # type: ignore[assignment]
        self.addCleanup(sys.modules.pop, "_r2s_test_registry", None)

    def test_what_a_helper_and_the_file_registered_is_taken_back_and_a_replace_undone(self) -> None:
        helper = self.tmp / "r2s_helper_mod.py"
        helper.write_text("import _r2s_test_registry as t\nclass Lid: pass\nt.REG.register('lid', Lid)\n")
        scene = self.tmp / "scene.py"
        scene.write_text("import _r2s_test_registry as t\nimport r2s_helper_mod\nt.REG.replace('library_thing', str)\n")
        sys.path.insert(0, str(self.tmp))
        self.addCleanup(sys.path.remove, str(self.tmp))
        self.addCleanup(sys.modules.pop, "r2s_helper_mod", None)
        exec(compile(scene.read_text(), str(scene), "exec"), {"__name__": "scene"})
        self.assertIn("lid", self.reg._items)
        self.assertEqual(self.reg._items["library_thing"], str)
        taken = self.notes.undo()
        self.assertEqual(sorted(name for _, name, *_ in taken), ["library_thing", "lid"])
        self.assertNotIn("lid", self.reg._items)
        self.assertEqual(self.reg._items["library_thing"], int)
        self.notes.redo(taken)  # the run after it failed: the scene still up gets them back
        self.assertEqual(self.reg._items["library_thing"], str)
        self.assertIn("lid", self.reg._items)
        self.assertEqual(len(self.notes.writes), 2, "and they are noted again, for the next save to take back")

    def test_a_registry_discovering_on_a_lookup_from_the_scene_keeps_what_it_found(self) -> None:
        reg: Registry[type] = Registry("suite")
        reg.add_discover(lambda: reg.register("task_from_a_root", dict))
        sys.modules["_r2s_test_registry"].REG = reg  # type: ignore[attr-defined]
        scene = self.tmp / "scene.py"
        scene.write_text("import _r2s_test_registry as t\nt.REG.get('task_from_a_root')\nt.REG.register('mine', set)\n")
        exec(compile(scene.read_text(), str(scene), "exec"), {"__name__": "scene"})
        self.assertEqual([name for _, name, *_ in self.notes.undo()], ["mine"], "the discovery is the library's")
        self.assertIn("task_from_a_root", reg._items)

    def test_a_module_registering_as_it_is_imported_from_outside_the_scene_dir_stays(self) -> None:
        elsewhere = Path(tempfile.mkdtemp())
        (elsewhere / "r2s_lib_mod.py").write_text("import _r2s_test_registry as t\nt.REG.register('bag', float)\n")
        sys.path.insert(0, str(elsewhere))
        self.addCleanup(sys.path.remove, str(elsewhere))
        self.addCleanup(sys.modules.pop, "r2s_lib_mod", None)
        scene = self.tmp / "scene.py"
        scene.write_text("import r2s_lib_mod\n")
        exec(compile(scene.read_text(), str(scene), "exec"), {"__name__": "scene"})
        self.assertEqual(self.notes.undo(), [])
        self.assertIn("bag", self.reg._items)


class SceneHash(unittest.TestCase):
    def test_the_log_hash_covers_the_files_a_save_to_which_rebuilds(self) -> None:
        folder = Path(tempfile.mkdtemp())
        (folder / "scene.py").write_text("import helpers.rig\n")
        rig = folder / "helpers" / "rig.py"
        rig.parent.mkdir()
        rig.write_text("FOV = 40\n")
        (folder / "run").mkdir()
        before = checks.scene_hash(folder / "scene.py")
        (folder / "run" / "probe.py").write_text("x = 1\n")
        self.assertEqual(checks.scene_hash(folder / "scene.py"), before, "a scratch script is not the scene")
        rig.write_text("FOV = 46\n")
        self.assertNotEqual(checks.scene_hash(folder / "scene.py"), before)


class LibraryNamedHelpers(unittest.TestCase):
    def test_a_module_named_as_the_librarys_beside_the_scene_is_neither_watched_nor_the_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name in ("scene", "robosimstudio2", "robosimstudio_lid", "lidlib"):
                (root / f"{name}.py").write_text("")
            self.assertEqual([p.name for p in helper_files(root / "scene.py")], ["lidlib.py"])
            paths = ScenePaths(root)
            self.assertEqual(
                {
                    n: paths.is_scene(str(root / f"{n}.py"), n)
                    for n in ("robosimstudio2", "robosimstudio_lid", "lidlib")
                },
                {"robosimstudio2": False, "robosimstudio_lid": False, "lidlib": True},
            )

    def test_a_helper_never_imported_gets_no_bytecode_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / "notes.py"
            helper.write_text("X = 1\n")
            hash_checked(helper)
            self.assertFalse((Path(tmp) / "__pycache__").exists(), "nothing written into the folder the person zips")


if __name__ == "__main__":
    unittest.main()
