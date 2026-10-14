# SPDX-License-Identifier: Apache-2.0
"""The fleet grid's spacing, labels and viewer, and the studio's recorders.

Each case is bookkeeping over stand-ins, so none of it needs a scene: a
re-spaced grid moves its labels and is applied on the drawing thread, a view
that opened its own viewer closes it, a scene with no bodies can be filmed,
and a run's meta keeps the run's start across takes.
"""

from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np

TIER = "fast"


class _Handle:
    def __init__(self, text: str, position) -> None:
        self.text = text
        self.position = position
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _Scene:
    def __init__(self) -> None:
        self.added: dict[str, _Handle] = {}

    def add_label(self, name: str, *, text: str, position) -> _Handle:
        handle = _Handle(text, position)
        self.added[name] = handle
        return handle


class _Viewer:
    def __init__(self) -> None:
        self.server = SimpleNamespace(scene=_Scene())
        self.model = SimpleNamespace(up_axis=2)
        self.world_offsets = np.zeros((2, 3), np.float32)
        self.closed = False
        self.calls: list[str] = []

    def set_world_offsets(self, grid) -> None:
        self.calls.append("set_world_offsets")
        self.world_offsets = np.array([[0.0, 0.0, 0.0], list(grid)], np.float32)

    def begin_frame(self, t) -> None:
        self.calls.append("begin_frame")

    def log_state(self, state) -> None:
        self.calls.append("log_state")

    def end_frame(self) -> None:
        self.calls.append("end_frame")

    def close(self) -> None:
        self.closed = True


def _view(owns: bool = False):
    from robosimstudio_studio.fleet_view import FleetView

    view = object.__new__(FleetView)
    view.viewer = _Viewer()
    view.owns_viewer = owns
    view.shown = [0, 1]
    view.interval = 0.0
    view._last = 0.0
    view._labels = {}
    view._captions = {}
    view.pending_spacing = None
    return view


_SLOTS = [SimpleNamespace(parked=True), SimpleNamespace(parked=True)]
_FLEET = SimpleNamespace(control_dt=0.01, pipeline=SimpleNamespace(state_0=None))


class FleetViewSpacing(unittest.TestCase):
    def test_respacing_moves_existing_labels(self) -> None:
        view = _view()
        view._update_labels(_SLOTS)
        handle = view._labels[1]
        self.assertEqual(handle.position, (0.0, 0.0, 1.4))
        view.set_spacing(3.0)
        view._update_labels(_SLOTS)
        self.assertIs(view._labels[1], handle)
        self.assertEqual(handle.position, (3.0, 3.0, 1.4))

    def test_pending_spacing_is_applied_by_the_draw(self) -> None:
        view = _view()
        view.pending_spacing = 2.5
        view.on_tick(_FLEET, _SLOTS, 1)
        self.assertIsNone(view.pending_spacing)
        calls = view.viewer.calls
        self.assertIn("set_world_offsets", calls)
        self.assertLess(calls.index("set_world_offsets"), calls.index("log_state"))
        np.testing.assert_allclose(view.viewer.world_offsets[1], [2.5, 2.5, 0.0])

    def test_slider_callback_only_assigns(self) -> None:
        from robosimstudio_studio import bench_session

        callbacks = []
        slider = SimpleNamespace(value=1.75, on_update=callbacks.append)
        gui = mock.MagicMock()
        gui.add_slider.return_value = slider
        view = _view()
        session = SimpleNamespace(
            server=SimpleNamespace(gui=gui),
            config=SimpleNamespace(dark_mode=False),
            fleet=SimpleNamespace(view=view, build_gui=lambda gui: None),
            lighting=mock.MagicMock(),
        )
        with mock.patch.object(bench_session, "PanelTask", mock.MagicMock()):
            bench_session.build_fleet_gui(session)
        self.assertEqual(len(callbacks), 1)
        callbacks[0](None)
        self.assertEqual(view.pending_spacing, 1.75)
        self.assertNotIn("set_world_offsets", view.viewer.calls)


class FleetViewClose(unittest.TestCase):
    def test_owned_viewer_is_closed(self) -> None:
        view = _view(owns=True)
        view.close()
        self.assertTrue(view.viewer.closed)

    def test_borrowed_viewer_stays_open(self) -> None:
        view = _view(owns=False)
        view.close()
        self.assertFalse(view.viewer.closed)


class PanelVideoCapture(unittest.TestCase):
    def test_a_scene_without_bodies_is_captured(self) -> None:
        from robosimstudio_studio.controls import PanelVideo

        panel = object.__new__(PanelVideo)
        particles = np.ones((4, 3), np.float32)
        panel.env = SimpleNamespace(
            step_count=2,
            state=SimpleNamespace(body_q=None, particle_q=SimpleNamespace(numpy=lambda: particles)),
            model=SimpleNamespace(particle_count=4),
        )
        panel.every = 2
        panel.buffer = deque(maxlen=30)
        panel.dropped = 0
        panel.capture()
        self.assertEqual(len(panel.buffer), 1)
        self.assertNotIn("body_q", panel.buffer[0])
        np.testing.assert_array_equal(panel.buffer[0]["particle_q"], particles)


class PanelRecordRunMeta(unittest.TestCase):
    def test_run_meta_keeps_the_first_take_start(self) -> None:
        from robosimstudio_studio import controls

        panel = object.__new__(controls.PanelRecord)
        panel.root = mock.MagicMock()
        panel.root.name = "demos"
        panel.spec = None
        panel.take = 0
        panel.recorder = None
        panel.started = ""
        panel.run_started = ""
        panel._state = SimpleNamespace(value="")
        panel.session = SimpleNamespace(status="")
        written = []
        panel.store = SimpleNamespace(results=lambda: [], write_run_meta=written.append)
        panel.env = SimpleNamespace(
            episode=SimpleNamespace(episode_id=0),
            request=SimpleNamespace(physics=SimpleNamespace(control_hz=50), full_name="", scene="s", embodiment=""),
            renderer_name="r",
            model=SimpleNamespace(device="cpu"),
        )
        stamps = iter(["t0", "t1", "t2", "t3"])
        fake_datetime = SimpleNamespace(now=lambda tz: SimpleNamespace(isoformat=lambda: next(stamps)))
        with (
            mock.patch.object(controls, "EpisodeRecorder", mock.MagicMock()),
            mock.patch.object(controls, "replace", lambda episode, **kw: episode),
            mock.patch.object(controls, "datetime", fake_datetime),
        ):
            panel.start()
            panel.recorder = None
            panel.start()
            self.assertEqual(panel.started, "t1")
            panel._write_run_meta()
        self.assertEqual(written[-1].started, "t0")


if __name__ == "__main__":
    unittest.main()
