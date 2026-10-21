# SPDX-License-Identifier: Apache-2.0
"""The live page's fixes from the agents' runs, held to what they say they do: no GPU.

Each one cost an agent minutes to hours on a real twin: the segmentation that
raised on any deformable in view, the routine helpers that never moved an arm
on the page, the entity class an edit did not reach, the rest check that could
not see a rope sink, and the command that timed out with no word of why.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio import registry
from robosimstudio.core.cameras import CameraSpec
from robosimstudio.registry import MISSING, Registry
from robosimstudio.solvers.render.tiled import SEG_PARTICLES, SEG_SOFT_MESH, _labels
from robosimstudio_real2sim_agent import checks, ctl, wire
from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.scenefile import SceneRegistrations
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
    live.head_stepped, live._head_ready, live._suspended_arms, live.last_take = False, None, [], {}
    live.build_error, live._tick_error = "", ""
    live.busy, live.busy_since = "", 0.0
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
        fake = types.SimpleNamespace(capture=lambda: None, close=lambda: None)
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
        live.busy, live.busy_since = "rendering a still", time.perf_counter()
        for _ in range(5):
            self.assertEqual(live._on_request({"cmd": "status"})["busy"], "rendering a still")
        self.assertEqual(self.queued, 1)

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
        live.busy, live.busy_since = "rebuilding the scene file", time.perf_counter() - 41
        started = time.perf_counter()
        reply = live._on_request({"cmd": "status"})
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(reply["busy"], "rebuilding the scene file")
        self.assertGreaterEqual(reply["busy_seconds"], 41)

    def test_a_command_after_a_failed_save_says_the_scene_is_the_one_before(self) -> None:
        live = _server()
        live.busy, live.busy_since, live.rendered = "", 0.0, True
        live.build_error = "Traceback ...\nNameError: name 'lid' is not defined"
        live.cmd_status = lambda: {"ok": True}  # type: ignore[method-assign]
        live.cmd_eval = lambda **kw: {"ok": True}  # type: ignore[method-assign]
        reply = live.dispatch({"cmd": "eval", "code": "1"})
        self.assertIn("did not build (NameError: name 'lid' is not defined)", reply["stale"])


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
        done = {"ok": True, "step": 30, "time": 0.5, "busy": "", "speed": 0.4, "last_take": "take1"}
        done["last_script"] = {"serial": 7, "name": "tidy", "how": "finished", "steps": 20, "wall_seconds": 1.2}
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

    def test_a_failed_routine_is_an_error_with_its_traceback(self) -> None:
        ended = {"serial": 2, "name": "eval", "how": "failed", "error": "RoutineError: stopped 364 mm short"}
        fake = _FakeStatus(self.sock, [{"ok": True, "busy": "", "last_script": ended}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_script(self.sock, {"serial": 2}, self.log, 0, 30.0)
        self.assertFalse(reply["ok"])
        self.assertIn("364 mm short", reply["error"])

    def test_status_wait_returns_once_the_page_is_free(self) -> None:
        busy = {"ok": True, "busy": "rendering a still", "busy_seconds": 3.0}
        fake = _FakeStatus(self.sock, [busy, {"ok": True, "busy": "", "pending_save": True}, {"ok": True, "busy": ""}])
        self.addCleanup(fake.close)
        self.assertEqual(ctl.wait_until_free(self.sock, 30.0), {"ok": True, "busy": ""})

    def test_a_run_waits_through_a_busy_page(self) -> None:
        busy = {"ok": True, "busy": "rebuilding the scene file", "busy_seconds": 2.0}
        done = {"ok": True, "busy": "", "step": 160, "time": 2.7, "paused": True, "playing": 0, "finite": True}
        fake = _FakeStatus(self.sock, [busy, busy, {**done, "generation": 1, "resets": 0}])
        self.addCleanup(fake.close)
        reply = ctl.wait_for_steps(self.sock, {"step": 40, "generation": 1, "resets": 0}, 120)
        self.assertEqual(reply["step"], 160)

    def test_ref_frames_needs_no_page(self) -> None:
        clip = _clip(self.tmp / "c.mp4", 2, "red")
        (self.tmp / "scene.py").write_text("")  # the page would put ref/ beside it
        run_dir = self.tmp / "run"
        code = ctl.main(["--run-dir", str(run_dir), "--json", "ref", "frames", str(clip), "--at", "0:start,end"])
        self.assertEqual(code, 0)
        timeline = json.loads((self.tmp / "ref" / "timeline.json").read_text())
        self.assertEqual([a.get("label", "") for a in timeline["anchors"]], ["start", ""])


if __name__ == "__main__":
    unittest.main()
