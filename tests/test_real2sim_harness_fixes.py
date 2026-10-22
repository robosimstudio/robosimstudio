# SPDX-License-Identifier: Apache-2.0
"""The live page's fixes from the agents' runs, held to what they say they do: no GPU.

Each one cost an agent minutes to hours on a real twin: the segmentation that
raised on any deformable in view, the routine helpers that never moved an arm
on the page, the entity class an edit did not reach, the rest check that could
not see a rope sink, and the command that timed out with no word of why.
"""

from __future__ import annotations

import gc
import json
import os
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
from robosimstudio.core.cameras import CameraSpec
from robosimstudio.registry import MISSING, Registry
from robosimstudio.solvers.render.tiled import SEG_PARTICLES, SEG_SOFT_MESH, _labels
from robosimstudio_real2sim_agent import checks, ctl, wire
from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import Loaded, ScenePaths, SceneRegistrations, hash_checked, helper_files
from robosimstudio_real2sim_agent.server import LiveServer
from tests.test_fix_real2sim_frames_reset import _clip

TIER = "fast"


class TiledLabels(unittest.TestCase):
    def test_reserved_ids_are_their_own_labels_and_a_miss_is_zero(self) -> None:
        lut = np.array([0, 5, 7], np.int32)
        idx = np.array([1, 2, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD], np.int64)
        np.testing.assert_array_equal(_labels(idx, lut), [5, 7, 0, SEG_PARTICLES, SEG_SOFT_MESH])

    def test_without_a_table_shapes_are_their_own_index(self) -> None:
        idx = np.array([3, 0xFFFFFFFF, 0xFFFFFFFE], np.int64)
        np.testing.assert_array_equal(_labels(idx, None), [3, 0, SEG_PARTICLES])

    def test_a_shape_past_the_table_is_still_an_error(self) -> None:
        with self.assertRaisesRegex(IndexError, "not this model's"):
            _labels(np.array([1, 9], np.int64), np.array([0, 5, 7], np.int32))


def _camera() -> CameraSpec:
    return CameraSpec(name="c", width=100, height=80, fov_y=60.0, eye=(0.0, 0.0, 1.0), look_at=(0.0, 0.0, 0.0))


class CameraProject(unittest.TestCase):
    def test_the_point_looked_at_lands_on_the_principal_point(self) -> None:
        uv, depth = _camera().project(np.array([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(uv[0], [50.0, 40.0], atol=1e-9)
        self.assertAlmostEqual(float(depth[0]), 1.0)

    def test_a_point_behind_the_camera_has_no_depth(self) -> None:
        _, depth = _camera().project(np.array([[0.0, 0.0, 2.0]]))
        self.assertLess(float(depth[0]), 0.0)


def _deformable_sim(points: np.ndarray, handles: dict) -> types.SimpleNamespace:
    state = types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=lambda: points), body_q=None)
    scene = types.SimpleNamespace(
        handles=types.SimpleNamespace(segmentation_names=lambda: {1: "table", 2: "__robot__"}, deformables=handles)
    )
    return types.SimpleNamespace(scene=scene, state=state)


def _handle(particles, triangles=()) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        particles=np.asarray(particles), triangles=np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    )


class NameDeformables(unittest.TestCase):
    def test_labels_follow_the_rigid_ones_in_name_order(self) -> None:
        sim = _deformable_sim(np.zeros((0, 3)), {"towel": _handle([]), "rice": _handle([])})
        names, labels = checks.seg_names(sim)
        self.assertEqual(labels, {"rice": 3, "towel": 4})
        self.assertEqual(names[3], "rice")

    def test_one_of_each_kind_takes_every_hit_of_its_kind(self) -> None:
        sim = _deformable_sim(np.zeros((3, 3)), {"towel": _handle([0, 1, 2], [[0, 1, 2]]), "rice": _handle([0])})
        _, labels = checks.seg_names(sim)
        seg = np.array([[1, SEG_SOFT_MESH], [SEG_PARTICLES, 0]], np.int32)
        out = checks.name_deformables(sim, _camera(), seg, labels)
        np.testing.assert_array_equal(out, [[1, labels["towel"]], [labels["rice"], 0]])

    def test_two_heaps_split_by_where_their_particles_are_drawn(self) -> None:
        left = np.array([[-0.3, 0.0, 0.0], [-0.3, 0.05, 0.0]])
        right = np.array([[0.3, 0.0, 0.0], [0.3, 0.05, 0.0]])
        sim = _deformable_sim(np.concatenate([left, right]), {"beans": _handle([0, 1]), "rice": _handle([2, 3])})
        _, labels = checks.seg_names(sim)
        seg = np.full((80, 100), SEG_PARTICLES, np.int32)
        cam = _camera()
        out = checks.name_deformables(sim, cam, seg, labels)
        for name, points in (("beans", left), ("rice", right)):
            uv, _ = cam.project(points[:1])
            self.assertEqual(out[int(uv[0, 1]), int(uv[0, 0])], labels[name], name)
        self.assertEqual(set(np.unique(out)), {labels["beans"], labels["rice"]}, "every hit goes to one of them")


class GateSeesDeformables(unittest.TestCase):
    def test_a_rope_that_sinks_fails_rest_and_a_thing_with_no_position_is_named(self) -> None:
        points = np.zeros((4, 3))

        class Sim:
            cameras = ()
            scene = _deformable_sim(points, {"rope": _handle([0, 1, 2, 3])}).scene
            state = types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=points.copy))

            def state_is_finite(self) -> bool:
                return True

            def names(self) -> list[str]:
                return ["rope", "ghost"]

            def run(self, steps: int) -> None:
                points[2, 2] -= 0.0079  # one node sinks 7.9 mm; the centre moves 2 mm

            def sync(self) -> None:
                pass

            def __getitem__(self, name):
                if name == "ghost":
                    raise KeyError(name)
                return types.SimpleNamespace(position=points.mean(axis=0))

        server = types.SimpleNamespace(sim=Sim(), build_error="")
        with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
            result = checks.gate(server, steps=1)
        rest = next(c for c in result["checks"] if c["name"] == "rest")
        self.assertFalse(rest["ok"])
        self.assertIn("rope drifted 7.90 mm in 1 steps left alone (bar 2 mm)", rest["detail"])
        self.assertIn("not measured (no position to read): ghost", rest["detail"])
        self.assertIn("SKIP  objects", result["report"], "no names given: a line that looked at nothing")


class RestOfAHeap(unittest.TestCase):
    def test_one_jittering_grain_does_not_fail_a_heap(self) -> None:
        points = np.zeros((100, 3))
        heap = types.SimpleNamespace(
            particles=np.arange(100), triangles=np.zeros((0, 3)), tube_radius=0.0, point_radius=0.002
        )

        class Sim:
            cameras = ()
            scene = _deformable_sim(points, {"rice": heap}).scene
            state = types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=points.copy))

            def state_is_finite(self) -> bool:
                return True

            def names(self) -> list[str]:
                return ["rice"]

            def run(self, steps: int) -> None:
                points[7, 2] += 0.01  # one grain bounces 10 mm

            def sync(self) -> None:
                pass

            def __getitem__(self, name):
                raise KeyError(name)

        with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
            rest = next(
                c
                for c in checks.gate(types.SimpleNamespace(sim=Sim(), build_error=""), steps=1)["checks"]
                if c["name"] == "rest"
            )
        self.assertTrue(rest["ok"], rest["detail"])


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


def _case_rest(lid_after: tuple) -> dict:
    """The gate's rest line for a case whose base stays and whose lid ends at the quaternion ``lid_after``."""
    body_q = np.zeros((3, 7))
    body_q[:, 6] = 1.0
    com = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    handles = types.SimpleNamespace(
        objects={"case": None}, statics={}, deformables={}, entity=lambda name: types.SimpleNamespace(bodies=(1, 2))
    )

    class Sim:
        cameras = ()
        scene = types.SimpleNamespace(handles=handles)
        state = types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=body_q.copy))
        model = types.SimpleNamespace(body_com=types.SimpleNamespace(numpy=lambda: com))

        def state_is_finite(self) -> bool:
            return bool(np.isfinite(body_q).all())

        def names(self) -> list[str]:
            return ["case"]

        def run(self, steps: int) -> None:
            body_q[2, 3:7] = lid_after

        def sync(self) -> None:
            pass

        def __getitem__(self, name):
            return types.SimpleNamespace(position=body_q[1, :3].copy())  # the base, which stays

    with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
        result = checks.gate(types.SimpleNamespace(sim=Sim(), build_error=""), steps=1)
    return next(c for c in result["checks"] if c["name"] == "rest")


class GateSeesJointedThings(unittest.TestCase):
    def test_a_lid_that_swings_while_its_base_stays_fails_rest(self) -> None:
        rest = _case_rest((0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)))  # the lid swings 45 degrees
        self.assertFalse(rest["ok"], rest["detail"])
        self.assertIn("case drifted", rest["detail"])

    def test_a_lid_that_blows_up_while_its_base_stays_fails_rest(self) -> None:
        rest = _case_rest((np.nan,) * 4)
        self.assertFalse(rest["ok"], rest["detail"])
        self.assertIn("case drifted nan", rest["detail"])


class TakeAccounting(unittest.TestCase):
    def test_a_reset_under_the_take_does_not_stop_the_count(self) -> None:
        rec = checks.Recorder.__new__(checks.Recorder)
        rec.sim = types.SimpleNamespace(step_count=100)
        rec.steps, rec._raw, rec.last_capture, rec.every = 0, 100, -1, 4
        rec._mark = checks.reset_mark(rec.sim)
        captured: list[int] = []
        rec.capture = lambda: (rec._advance(), captured.append(rec.steps), setattr(rec, "last_capture", rec.steps))
        for step in (100, 104, 110, 2, 6, 9):  # the routine resets the scene after step 110
            rec.sim.step_count = step
            rec.maybe()
        self.assertEqual(captured, [0, 4, 10, 16], "a frame every four steps, the reset at 12 included")
        self.assertEqual(rec.steps, 19)

    def test_a_reset_then_more_steps_than_before_in_one_go_is_still_a_reset(self) -> None:
        sim = types.SimpleNamespace(step_count=100, reset_count=4)
        mark = checks.reset_mark(sim)
        sim.reset_count, sim.step_count = 5, 500  # sim.reset(); sim.run(500)
        self.assertEqual(checks.steps_since(sim, 100, mark), 500)

    def test_a_long_chain_keeps_a_sample_of_its_links(self) -> None:
        body_q = np.zeros((130, 7))
        handles = types.SimpleNamespace(
            objects={"cable": None}, statics={}, entity=lambda name: types.SimpleNamespace(bodies=tuple(range(120)))
        )
        body_q[:, 6] = 1.0
        sim = types.SimpleNamespace(
            scene=types.SimpleNamespace(handles=handles),
            state=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: body_q)),
            model=types.SimpleNamespace(body_com=types.SimpleNamespace(numpy=lambda: np.zeros((130, 3)))),
        )
        self.assertEqual(len(checks._part_positions(sim)["cable"]), checks.PARTS_KEPT)
        sim.state.body_q = None
        self.assertEqual(checks._part_positions(sim), {}, "a take is never lost for want of these")

    def test_a_jointed_thing_records_every_body(self) -> None:
        body_q = np.zeros((4, 7))
        body_q[2, :3] = (0.1, 0.2, 0.3)
        handles = types.SimpleNamespace(
            objects={"case": None, "cube": None},
            statics={},
            entity=lambda name: types.SimpleNamespace(bodies=(1, 2) if name == "case" else (), body=3),
        )
        body_q[:, 6] = 1.0  # unrotated
        com = np.zeros((4, 3))
        sim = types.SimpleNamespace(
            scene=types.SimpleNamespace(handles=handles),
            state=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: body_q)),
            model=types.SimpleNamespace(body_com=types.SimpleNamespace(numpy=lambda: com)),
        )
        self.assertEqual(
            checks._part_positions(sim), {"case": [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], "cube": [[0.0, 0.0, 0.0]]}
        )
        com[2] = (0.1, 0.0, 0.0)  # the lid's mass 100 mm along its own x from the hinge at its origin
        body_q[2, 3:7] = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # swung 90 degrees about z
        np.testing.assert_allclose(checks._part_positions(sim)["case"][1], [0.1, 0.3, 0.3], atol=1e-4)


class MaskKeys(unittest.TestCase):
    def test_any_pixel_not_black_is_the_thing_and_an_empty_mask_says_so(self) -> None:
        from PIL import Image

        from robosimstudio_real2sim_agent import compare

        tmp = Path(tempfile.mkdtemp())
        mask = np.zeros((4, 4, 3), np.uint8)
        mask[1, 1] = (0, 0, 1)  # nearly black, and not black
        Image.fromarray(mask).save(tmp / "m.png")
        key = compare.parse_key(f"lid=mask:{tmp / 'm.png'}")
        self.assertEqual(int(compare.photo_mask(np.zeros((4, 4, 3), np.uint8), key).sum()), 1)
        reading = compare.compare_masks(None, np.ones((4, 4), bool))
        self.assertIn("the mask holds nothing at this size", compare.advice("lid", reading, 4, key))

    def test_a_deformable_is_sampled_for_the_take(self) -> None:
        points = np.arange(300, dtype=float).reshape(100, 3)
        sim = _deformable_sim(points, {"rope": _handle(list(range(100)))})
        sim.scene.handles.objects, sim.scene.handles.statics = {}, {}
        self.assertEqual(len(checks._part_positions(sim)["rope"]), checks.PARTS_KEPT)


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


class _Arm:
    def __init__(self) -> None:
        self.suspended = False
        self.calls: list[str] = []

    def suspend(self) -> None:
        self.suspended = True
        self.calls.append("suspend")

    def resume(self) -> None:
        self.suspended = False
        self.calls.append("resume")


def _server() -> LiveServer:
    live = LiveServer.__new__(LiveServer)
    live.path = Path("/nonexistent/scene.py")
    live.script, live.recorder, live.run_budget, live.last_error = [], None, 0, ""
    live.head_stepped, live._head_ready, live.last_take = False, None, {}
    live.ended, live.takes_written, live.serving = {}, [], "scene"
    live.runs, live.run_target, live.scene_id, live.watch_enabled = 0, 0, "", False
    live.runs_ended, live._walked = {}, (-1.0, [])
    live._arms_before, live.resets = {}, 0
    live.log_path, live._motions = "", {}
    live.build_error, live._tick_error = "", ""
    live.busy_state = ("", 0.0, False)
    live._status_lock, live._status_pending = threading.Lock(), None
    live.script_meta, live.scripts_started, live.last_script = {}, 0, {}
    live.sim = types.SimpleNamespace(step_count=0)
    live.session = types.SimpleNamespace(arms={"left": _Arm(), "right": _Arm()})
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
        live.load = lambda files=None: types.SimpleNamespace(ns={}, scene="twin", sim="the twin", seconds=1.0)  # type: ignore[method-assign]
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
        fake = types.SimpleNamespace(start=lambda: None, probe=lambda: None, close=lambda: None, name="take1")
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
        fake = types.SimpleNamespace(start=lambda: None, probe=lambda: None, close=lambda: None)
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


class _FakeStatus:
    """A socket that answers ``status`` from a script of replies, the last one forever."""

    def __init__(self, path: Path, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.socket = wire.ControlSocket(path, self._answer)
        self.socket.start()

    def _answer(self, request: dict) -> dict:
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def close(self) -> None:
        self.socket.close()


class CtlWaits(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(dir="/tmp"))
        self.sock = self.tmp / "s.sock"
        self.log = self.tmp / "serve.log"
        self.log.write_text("")

    def test_a_routine_is_waited_for_with_its_output_streamed(self) -> None:
        playing = {"ok": True, "step": 10, "time": 0.2, "last_script": {}, "busy": ""}
        done = {"ok": True, "step": 30, "time": 0.5, "busy": "", "speed": 0.4, "takes_written": ["take1"]}
        done["ended"] = {"7": {"serial": 7, "name": "tidy", "how": "finished", "steps": 20, "wall_seconds": 1.2}}
        fake = _FakeStatus(self.sock, [playing, done])
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
        fake = _FakeStatus(
            self.sock, [{"ok": True, "busy": "", "last_script": newer, "ended": {"1": mine, "2": newer}}]
        )
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 1}, self.log, 0, 30.0)
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["routine"], "a")

    def test_a_take_written_and_then_another_started_counts_as_written(self) -> None:
        done = {"ok": True, "busy": "", "ended": {"3": {"serial": 3, "name": "a", "how": "finished"}}}
        fake = _FakeStatus(self.sock, [{**done, "recording": "take2", "last_take": "", "takes_written": ["take1"]}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 3, "take": "take1"}, self.log, 0, 30.0)
        self.assertEqual(reply["take"], "take1")

    def test_status_wait_does_not_wait_out_a_playing_routine(self) -> None:
        stepping = {"ok": True, "busy": "playing one step of a routine", "busy_routine": True}
        fake = _FakeStatus(self.sock, [stepping])
        self.addCleanup(fake.close)
        self.assertTrue(ctl.wait_until_free(self.sock, 5.0)["busy_routine"])

    def test_a_failed_routine_is_an_error_with_its_traceback(self) -> None:
        ended = {"serial": 2, "name": "eval", "how": "failed", "error": "RoutineError: stopped 364 mm short"}
        fake = _FakeStatus(self.sock, [{"ok": True, "busy": "", "ended": {"2": ended}}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 2}, self.log, 0, 30.0)
        self.assertFalse(reply["ok"])
        self.assertIn("364 mm short", reply["error"])

    def test_status_wait_returns_once_the_page_is_free(self) -> None:
        busy = {"ok": True, "busy": "rendering a still", "busy_seconds": 3.0}
        fake = _FakeStatus(self.sock, [busy, {"ok": True, "busy": "", "pending_save": True}, {"ok": True, "busy": ""}])
        self.addCleanup(fake.close)
        self.assertEqual(ctl.wait_until_free(self.sock, 30.0), {"ok": True, "busy": ""})

    def test_a_run_is_waited_for_through_a_busy_page_until_its_ending(self) -> None:
        busy = {"ok": True, "busy": "rebuilding the scene file", "step": 90, "runs_ended": {}}
        ended = {"serial": 3, "how": "done", "step": 160, "target": 160, "finite": True}
        fake = _FakeStatus(
            self.sock, [busy, busy, {"ok": True, "busy": "", "step": 160, "time": 2.7, "runs_ended": {"3": ended}}]
        )
        self.addCleanup(fake.close)
        reply = ctl.wait_for_steps(self.sock, {"run": 3, "step": 40, "target": 160}, 60.0)
        self.assertEqual((reply["ok"], reply["step"], reply["finite"]), (True, 160, True))

    def test_a_routine_that_finished_without_its_take_is_no_success(self) -> None:
        ended = {"serial": 5, "name": "lift", "how": "finished"}
        status = {"ok": True, "busy": "", "ended": {"5": ended}, "takes_written": [], "last_error": "ffmpeg died"}
        fake = _FakeStatus(self.sock, [status])
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
        fake = _FakeStatus(self.sock, [{"ok": True, "busy": "", "step": 45, "runs_ended": {}}])
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
        fake = _FakeStatus(self.sock, [{"ok": True, **stepping, "ended": {"4": done}, "takes_written": ["t1"]}])
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
        fake = _FakeStatus(self.sock, [{"ok": True, "busy": "", "busy_seconds": 0.0, "pending_save": True}])
        self.addCleanup(fake.close)
        said: list[str] = []
        clock = iter(range(0, 10_000, 4))
        fake_time = types.SimpleNamespace(monotonic=lambda: float(next(clock)), sleep=lambda s: None)
        with mock.patch.object(ctl, "time", fake_time), mock.patch.object(ctl, "_say", said.append):
            ctl.wait_until_free(self.sock, 30.0)
        self.assertTrue(said and not said[-1].endswith("(0 s)"), said)

    def test_ref_frames_without_a_page_says_what_was_wrong_in_one_line(self) -> None:
        clip = _clip(self.tmp / "c.mp4", 2, "red")
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
        clip = _clip(self.tmp / "c.mp4", 2, "red")
        (self.tmp / "scene.py").write_text("")  # the page would put ref/ beside it
        run_dir = self.tmp / "run"
        code = ctl.main(["--run-dir", str(run_dir), "--json", "ref", "frames", str(clip), "--at", "0:start,end"])
        self.assertEqual(code, 0)
        timeline = json.loads((self.tmp / "ref" / "timeline.json").read_text())
        self.assertEqual([a.get("label", "") for a in timeline["anchors"]], ["start", ""])


class TwinInsideAScene(unittest.TestCase):
    def test_a_twin_is_not_made_where_another_scenes_page_would_take_it_for_its_own(self) -> None:
        live = Path(tempfile.mkdtemp())
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
        live = Path(tempfile.mkdtemp())
        (live / "ctl").write_text("")
        served = live / "bench"
        (served / "run").mkdir(parents=True)
        (served / "run" / "serve.log").write_text("")
        with mock.patch("builtins.print"):
            code = ctl.main(["--run-dir", str(live / "run"), "new", "t", "--dir", str(served / "t")])
        self.assertEqual(code, 1)


class TwinFolder(unittest.TestCase):
    def setUp(self) -> None:
        self.live = Path(tempfile.mkdtemp())
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
        moved = Path(tempfile.mkdtemp()) / "lid"
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


if __name__ == "__main__":
    unittest.main()


class TheSceneThatWouldNotGoIn(unittest.TestCase):
    """A scene that builds but will not go into the page is given back before the bare one is built in its place."""

    class Twin:
        def __init__(self, made: list, closed: list) -> None:
            made.append(weakref.ref(self))
            self.closed = closed

        def close(self) -> None:
            self.closed.append("sim")

    def test_a_rebuild_lets_it_go_before_the_stand_in_is_built(self) -> None:
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
            return types.SimpleNamespace(ns={}, scene="bare", sim="the bare scene", seconds=0.0)

        live.session = types.SimpleNamespace(
            adopt=adopt,
            begin_run=lambda: None,
            status="",
            teardown=lambda: closed.append("teardown"),
            viewer=types.SimpleNamespace(set_model=lambda model: closed.append(f"viewer {model}")),
        )
        live.generation, live.ns, live.scene = 0, {}, None
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
        self.assertEqual(seen, [(True, ["teardown", "viewer None", "sim"], None)], "nothing of it held while building")
        self.assertEqual(
            (reply["ok"], live.serving, live.scene_id, live.sim), (False, "stand-in", "bare", "the bare scene")
        )
        self.assertIn("adopt failed", live.last_error)

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


class TakeProbe(unittest.TestCase):
    def _recorder(self, height: int = 540) -> checks.Recorder:
        rec = checks.Recorder.__new__(checks.Recorder)
        rec.sim = types.SimpleNamespace(step_count=7, reset_count=0)
        rec.cam = types.SimpleNamespace(name="side")
        rec._probed = None
        rec.looks = []
        rec._look = lambda: (rec.looks.append(1), (np.zeros((height, 960, 3), np.uint8), "seg", None))[1]
        rec.captured = []
        rec.capture = lambda look=None: rec.captured.append(look)
        return rec

    def test_a_video_the_encoder_cannot_make_fails_the_probe(self) -> None:
        with mock.patch("robosimstudio.core.render.video.ffmpeg_exe", return_value="/bin/true"):
            with self.assertRaisesRegex(ValueError, "even width and height"):
                self._recorder(height=541).probe()
        with mock.patch("robosimstudio.core.render.video.ffmpeg_exe", return_value="/nonexistent/ffmpeg"):
            with self.assertRaisesRegex(FileNotFoundError, "no ffmpeg"):
                self._recorder().probe()

    def test_frame_0_is_the_probes_unless_the_scene_stepped_since(self) -> None:
        rec = self._recorder()
        with mock.patch("robosimstudio.core.render.video.ffmpeg_exe", return_value="/bin/true"):
            rec.probe()
            rec.start()
            self.assertEqual((len(rec.looks), rec.captured[-1][1]), (1, "seg"), "rendered once")
            rec.probe()
            rec.sim.step_count += 1  # what was dropped stepped the scene as it cleaned up
            rec.start()
        self.assertIsNone(rec.captured[-1], "filmed afresh")


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
        for cmd in ("quit", "reset", "pause", "resume"):
            self.assertGreaterEqual(ctl.TIMEOUTS[cmd], ctl.TIMEOUTS["reload"], f"{cmd} waits out a long step")
