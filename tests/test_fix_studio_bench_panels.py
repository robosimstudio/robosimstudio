# SPDX-License-Identifier: Apache-2.0
"""The task page's policy runner and task menu fail without taking the page down.

Each case is one a page reached without anything being wrong with the page:
a policy that raises mid-episode, a Run that needs a recompile that fails, a
leader-held arm handed back to its gizmo by a policy that never held it, a
switch to a task with fewer episodes than the one being shown, and a load that
was refused but kept the world count it asked for. All of it is bookkeeping,
so none of it needs a scene: the panels are built over stand-ins.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

TIER = "fast"


class _Arm:
    def __init__(self, suspended: bool = False) -> None:
        self.suspended = suspended

    def suspend(self) -> None:
        self.suspended = True

    def resume(self) -> None:
        self.suspended = False


class _Policy:
    control_mode = "joint"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False

    def reset(self, obs, info) -> None:
        del obs, info

    def get_action(self, obs):
        del obs
        if self.fail:
            raise RuntimeError("the policy broke")
        return [0.0]

    def on_step(self, obs, reward, info) -> None:
        del obs, reward, info

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, arms: dict, reset_fails: bool = False) -> None:
        self.env = SimpleNamespace(control_mode="joint", step=None)
        self.server = None
        self.run_dir = None
        self.arms = arms
        self.paused = False
        self.status = ""
        self.obs = {}
        self.episode_id = 0
        self.reset_fails = reset_fails

    def reset_env(self):
        if self.reset_fails:
            raise RuntimeError("the episode did not recompile")
        return {}, {}

    def on_reset(self, obs, info) -> None:
        del obs, info

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)


def _driver(session, policy):
    from pathlib import Path

    from robosimstudio_studio.bench_panels import PanelSolution

    session.run_dir = Path("/nonexistent")
    driver = PanelSolution(session)
    driver._state = SimpleNamespace(value="")
    driver._build = lambda selected: policy
    return driver


def _quiet():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())


class PolicyRunner(unittest.TestCase):
    def test_a_failed_reset_pauses_and_releases_the_policy(self) -> None:
        session = _Session({"arm": _Arm()}, reset_fails=True)
        policy = _Policy()
        driver = _driver(session, policy)
        out, err = _quiet()
        with out, err:
            driver.start("graph")
        self.assertFalse(driver.running)
        self.assertIsNone(driver.policy)
        self.assertTrue(policy.closed)
        self.assertTrue(session.paused)
        self.assertIn("did not recompile", driver._state.value)
        self.assertFalse(session.arms["arm"].suspended)

    def test_a_raising_policy_is_stopped_and_the_arm_given_back(self) -> None:
        session = _Session({"arm": _Arm()})
        policy = _Policy(fail=True)
        driver = _driver(session, policy)
        driver.start("graph")
        self.assertTrue(driver.running)
        self.assertTrue(session.arms["arm"].suspended)
        out, err = _quiet()
        with out, err:
            action, mode = driver.step()
        self.assertIsNone(action)
        self.assertEqual(mode, "joint")
        self.assertFalse(driver.running)
        self.assertTrue(policy.closed)
        self.assertFalse(session.arms["arm"].suspended)
        self.assertIn("the policy broke", driver._state.value)

    def test_stop_gives_back_only_the_arms_start_took(self) -> None:
        held = _Arm(suspended=True)  # a leader arm holds this one
        free = _Arm()
        session = _Session({"held": held, "free": free})
        driver = _driver(session, _Policy())
        driver.start("graph")
        self.assertTrue(held.suspended and free.suspended)
        driver.stop("stopped")
        self.assertTrue(held.suspended)
        self.assertFalse(free.suspended)


class _Prof:
    def phase(self, name):
        del name
        return contextlib.nullcontext()


def _recording_panel(session):
    """A PanelRecord with a take open, over a recorder that counts what it is handed."""
    from robosimstudio_studio.controls import PanelRecord

    panel = PanelRecord.__new__(PanelRecord)
    panel.session, panel.env, panel.server = session, session.env, None
    panel.recorder = mock.Mock()
    panel.steps = 0
    panel.mode = "joint"
    return panel


class TheLoopCapturesOnlyWhatStepped(unittest.TestCase):
    def _loop(self, policy):
        from robosimstudio_studio.session import StudioSession

        session = _Session({"arm": _Arm()})
        session.env.step_count = 0

        def step(action, mode=""):
            del action, mode
            session.env.step_count += 1
            return {}, 0.0, False, False, {}

        session.env.step = step
        driver = _driver(session, policy)
        driver.start("graph")
        loop = SimpleNamespace(
            env=session.env,
            driver=driver,
            video=mock.Mock(),
            record=_recording_panel(session),
            _captured_step=0,
            _device_span=contextlib.nullcontext,
        )
        return StudioSession._step_and_capture, loop

    def test_a_raising_policy_puts_no_step_in_the_take(self) -> None:
        step_and_capture, loop = self._loop(_Policy(fail=True))
        out, err = _quiet()
        with out, err:
            step_and_capture(loop, _Prof())
        self.assertFalse(loop.driver.running)
        self.assertEqual(loop.record.steps, 0, "a step that never happened was recorded")
        loop.record.recorder.step.assert_not_called()
        loop.video.capture.assert_not_called()

    def test_a_policy_that_steps_is_captured(self) -> None:
        step_and_capture, loop = self._loop(_Policy())
        loop.record._result = lambda: SimpleNamespace(success=False, reward=0.0, metrics={})
        step_and_capture(loop, _Prof())
        self.assertEqual(loop.record.steps, 1)
        loop.video.capture.assert_called_once_with()
        self.assertEqual(loop._captured_step, 1)


class ARunThatRecompilesRebindsThePage(unittest.TestCase):
    def test_the_policy_is_not_started_on_a_scene_it_was_not_built_for(self) -> None:
        session = _Session({"arm": _Arm()})
        session.env.model = object()

        def reset_env():
            session.env.model = object()
            return {}, {}

        session.reset_env = reset_env
        session._drop_panels = mock.Mock()
        session._bind = mock.Mock()
        policy = _Policy()
        driver = _driver(session, policy)
        driver.start("graph")
        self.assertFalse(driver.running)
        self.assertTrue(policy.closed)
        self.assertFalse(session.arms["arm"].suspended)
        session._drop_panels.assert_called_once_with()
        session._bind.assert_called_once_with(session.env)
        self.assertIn("press Run again", session.status)


class ARebuiltRecordPanelKeepsTheRun(unittest.TestCase):
    def test_the_run_start_and_the_kept_count_come_off_the_directory(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from robosimstudio_studio.controls import PanelRecord

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_meta.json").write_text(json.dumps({"started": "2026-01-01T00:00:00+00:00"}))
            (root / "episode_results.jsonl").write_text('{"episode_id": 0}\n{"episode_id": 1}\n')
            session = SimpleNamespace(env=None, server=None, run_dir=root)
            panel = PanelRecord(session, root=root)
            self.assertEqual(panel.run_started, "2026-01-01T00:00:00+00:00")
            self.assertEqual(panel.kept, 2)

    def test_a_fresh_directory_starts_from_nothing(self) -> None:
        import tempfile
        from pathlib import Path

        from robosimstudio_studio.controls import PanelRecord

        with tempfile.TemporaryDirectory() as tmp:
            session = SimpleNamespace(env=None, server=None, run_dir=Path(tmp))
            panel = PanelRecord(session)
            self.assertEqual((panel.run_started, panel.kept, panel.take), ("", 0, 0))


class TaskMenu(unittest.TestCase):
    def _panel(self, selected_family: str, episode_id: int):
        from robosimstudio_studio.bench_panels import PanelTask

        panel = PanelTask.__new__(PanelTask)
        task = SimpleNamespace(full_name="fam/a", scene="", embodiment="", gripper="")
        panel.session = SimpleNamespace(current_task=task, episode_id=episode_id, status="", load_request=None)
        for flag in (
            "episode_changed",
            "run_requested",
            "snapshot_requested",
            "repro_requested",
            "family_changed",
            "embodiment_changed",
            "reload_requested",
        ):
            setattr(panel, flag, False)
        panel.select_requested = True
        panel._family = SimpleNamespace(value=selected_family)
        panel._variant = SimpleNamespace(value="b" if selected_family == "other" else "a")
        panel._scene = panel._embodiment = panel._gripper = SimpleNamespace(value="")
        panel._scenes = panel._embodiments = panel._grippers = {}
        panel._readout = SimpleNamespace(value="")
        panel._worlds = SimpleNamespace(value=1)
        return panel

    def test_a_task_switch_starts_on_episode_zero(self) -> None:
        panel = self._panel("other", episode_id=5)
        panel.drain()
        self.assertEqual(panel.session.load_request[0], "other/b")
        self.assertEqual(panel.session.load_request[1], 0)

    def test_a_reload_keeps_the_episode(self) -> None:
        panel = self._panel("fam", episode_id=5)
        panel.reload_requested = True
        panel.drain()
        self.assertEqual(panel.session.load_request[1], 5)


class LoadTask(unittest.TestCase):
    def _session(self):
        return SimpleNamespace(
            config=SimpleNamespace(
                obs_renderer=None, obs_render_every=1, device=None, max_steps=0, control_mode="joint"
            ),
            num_worlds=1,
            task_panel=None,
            status="",
            _narrowed="",
        )

    def test_a_refused_load_keeps_the_world_count(self) -> None:
        from robosimstudio_studio import bench_session

        session = self._session()
        out, err = _quiet()
        with mock.patch.object(bench_session, "unbuildable", return_value="cannot build"), out, err:
            self.assertFalse(bench_session.load_task(session, "fam/a", 0, num_worlds=64))
        self.assertEqual(session.num_worlds, 1)

    def test_a_failed_fleet_keeps_the_world_count(self) -> None:
        from robosimstudio_studio import bench_session

        session = self._session()
        with (
            mock.patch.object(bench_session, "unbuildable", return_value=""),
            mock.patch.object(bench_session, "load_fleet", return_value=False),
        ):
            self.assertFalse(bench_session.load_task(session, "fam/a", 0, num_worlds=64))
        self.assertEqual(session.num_worlds, 1)

    def test_a_failed_compile_after_narrowing_keeps_the_fleet_width(self) -> None:
        from robosimstudio.scene.compiler import CannotReplicate
        from robosimstudio_studio import bench_session

        session = self._session()
        session.num_worlds = 8
        out, err = _quiet()
        with (
            mock.patch.object(bench_session, "unbuildable", return_value=""),
            mock.patch.object(bench_session, "load_fleet", side_effect=CannotReplicate("no")),
            mock.patch.object(bench_session, "Env", side_effect=RuntimeError("no compile")),
            out,
            err,
        ):
            self.assertFalse(bench_session.load_task(session, "fam/a", 0, num_worlds=16))
        self.assertEqual(session.num_worlds, 8)
        self.assertEqual(session._narrowed, "")


if __name__ == "__main__":
    unittest.main()
