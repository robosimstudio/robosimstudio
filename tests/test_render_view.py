# SPDX-License-Identifier: Apache-2.0
"""The path-traced viewport's decisions, without a GPU or a browser.

What is worth pinning here is not that OVRTX renders -- that needs a card and is
the medium tier's business -- but the four decisions the panel makes before it
ever asks for a frame, each of which has already been wrong once:

* which mode it resolves to, and that a page with no tracer still opens;
* that it is **inert without a browser**, so a headless session costs what it
  cost before this panel existed;
* what it hides and what it keeps, because hiding everything takes the gizmos
  with it and hiding nothing leaves the stand-in stage over the frame;
* that it puts back exactly what it hid.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

import numpy as np

from robosimstudio_studio.render_view import PanelRenderView
from tests.fixtures.viser_stubs import _StubGui, _StubSession

TIER = "fast"


class _Env:
    """The little of an environment the panel touches before it renders."""

    model = object()
    cameras = ()
    state = object()


def _panel(**kwargs) -> PanelRenderView:
    session = _StubSession(_Env(), _StubGui())
    panel = PanelRenderView(session, **kwargs)
    panel.build(session.server.gui)
    return panel


def _rtx_panel(why: str = "NVIDIA RTX PRO 6000 (sm_120)", **kwargs) -> PanelRenderView:
    """A page whose world model is on *and* resolved to the tracer.

    Which renderer a page gets depends on the machine (a CI runner has no ovrtx
    and no RT cores), so a test about what the RTX viewport does to the scene
    pins the choice; nothing below renders, so the pin costs nothing. ``why`` is
    the reason the rule gave -- a GPU for the machine's pick, ``asked for`` or
    ``ROBOSIMSTUDIO_RENDERER=rtx`` for a name somebody chose.
    """
    with unittest.mock.patch("robosimstudio_studio.render_view.choose_renderer", return_value=("rtx", why)):
        return _panel(world_model=True, **kwargs)


def _tiled_panel(**kwargs) -> PanelRenderView:
    """A page on a machine where the rule picks ``tiled``: no ovrtx, or no RT cores."""
    with unittest.mock.patch(
        "robosimstudio_studio.render_view.choose_renderer",
        return_value=("tiled", "NVIDIA A100 (sm_80) has no RT cores"),
    ):
        return _panel(**kwargs)


class TestTheClickListener(unittest.TestCase):
    """The server outlives every panel built on it, so a callback added per build must come off on close."""

    def test_one_listener_however_often_it_is_built_and_none_after_close(self) -> None:
        panel = _panel()
        scene = panel.server.scene
        panel.build(panel.server.gui)  # a page rebuilt over the same panel
        self.assertEqual(scene.click_callbacks, [panel._on_click])
        panel.close()
        self.assertEqual(scene.click_callbacks, [], "a closed panel left on the click list is kept alive by it")
        panel.close()  # idempotent


class TestTheModeItResolvesTo(unittest.TestCase):
    def test_off_is_raster_whatever_is_installed(self) -> None:
        self.assertEqual(_panel(world_model=False).mode, "raster")

    def test_on_follows_the_tracer(self) -> None:
        """On asks for the path tracer, and gets raster where there is none to ask."""
        self.assertIn(_panel(world_model=True).mode, ("rtx", "raster"))


class TestItIsInertWithoutABrowser(unittest.TestCase):
    """No client, no renderer -- what keeps a headless session as cheap as before."""

    def test_draining_with_nobody_looking_builds_nothing(self) -> None:
        panel = _panel(world_model=True)
        for _ in range(5):
            panel.drain()
        self.assertIsNone(panel._renderer, "a page nobody opened must not build a path tracer")
        self.assertEqual(panel._hidden, [], "and must not hide the scene it is not replacing")

    def test_raster_mode_drains_without_touching_the_scene(self) -> None:
        panel = _panel(world_model=False)
        panel.drain()
        self.assertIsNone(panel._renderer)
        self.assertEqual(panel._hidden, [])


class TestWhatItHidesAndKeeps(unittest.TestCase):
    """The rule the first prototype got wrong in both directions."""

    def _scene_with(self, panel):
        scene = panel.server.scene
        for name in (
            "/model/shapes/shape_1",
            "/model/deformables/sponge",
            "/studio/right/cursor",
            "/studio/object_handle_0",
            "/studio/stage",
            "/studio/scenery/floor",
            "/studio/scenery/wall_north",
            "/grid",
        ):
            scene._node(name, visible=True)
        return scene

    def test_the_model_goes_and_the_furniture_stays(self) -> None:
        panel = _panel(world_model=True)
        scene = self._scene_with(panel)
        panel._hide_model()
        visible = {name for name, node in scene._handle_from_node_name.items() if node.visible}
        self.assertEqual(
            visible,
            {"/studio/right/cursor", "/studio/object_handle_0"},
            "the gizmos are what you aim with, so they stay; everything else is in the frame already",
        )

    def test_the_stand_in_stage_goes_with_the_model(self) -> None:
        """It stands in for the hidden bench, and the real bench is in the frame."""
        panel = _panel(world_model=True)
        scene = self._scene_with(panel)
        panel._hide_model()
        self.assertFalse(scene._handle_from_node_name["/studio/stage"].visible)

    def test_the_drawn_room_goes_with_it(self) -> None:
        """The room arrived after this panel did, under the ``/studio`` prefix
        that otherwise means "furniture, keep it" -- so it was kept, and a
        floor-to-wall raster room drew over the path-traced one. There is no
        depth channel, so whatever is kept is simply on top.
        """
        panel = _panel(world_model=True)
        scene = self._scene_with(panel)
        panel._hide_model()
        room = [scene._handle_from_node_name[n] for n in ("/studio/scenery/floor", "/studio/scenery/wall_north")]
        self.assertTrue(all(not node.visible for node in room))

    def test_restoring_puts_back_exactly_what_it_took(self) -> None:
        panel = _panel(world_model=True)
        scene = self._scene_with(panel)
        already_hidden = scene._node("/studio/object_handle_1", visible=False)
        panel._hide_model()
        panel._restore()
        self.assertTrue(
            all(node.visible for name, node in scene._handle_from_node_name.items() if node is not already_hidden),
            "a mode switch that leaves the scene hidden is a blank page",
        )
        self.assertFalse(already_hidden.visible, "and one that shows a handle nobody placed is a lie")

    def test_restoring_twice_is_harmless(self) -> None:
        panel = _panel(world_model=True)
        self._scene_with(panel)
        panel._hide_model()
        panel._restore()
        panel._restore()
        self.assertEqual(panel._hidden, [])


class TestTheWorldModelSwitch(unittest.TestCase):
    """One checkbox: on is the path-traced world, off is plain viser."""

    def _built(self, world_model: bool):
        session = _StubSession(_Env(), _StubGui())
        panel = PanelRenderView(session, world_model=world_model)
        panel.build(session.server.gui)
        return panel, session.server.gui.find("World model")

    def test_it_starts_where_the_mode_resolved(self) -> None:
        """`auto` decides at startup; the switch shows what it decided."""
        self.assertFalse(self._built(False)[1].value)

    def test_it_is_off_unless_asked_for(self) -> None:
        """A page opens at the speed it can always sustain. Starting it on costs
        a browser about a minute of looking unloaded while its renderer builds,
        so `auto` is an opt-in rather than the default."""
        session = _StubSession(_Env(), _StubGui())
        panel = PanelRenderView(session)
        panel.build(session.server.gui)
        self.assertEqual(panel.mode, "raster")
        self.assertFalse(session.server.gui.find("World model").value)

    def test_the_session_default_agrees_with_the_panel(self) -> None:
        """Two defaults for one thing is how they drift apart."""
        from robosimstudio_studio import StudioConfig

        self.assertFalse(StudioConfig().world_model)

    def test_switching_it_off_gives_the_raster_scene_back(self) -> None:
        panel = _rtx_panel()
        scene = panel.server.scene
        for name in ("/model/shapes/shape_1", "/studio/stage"):
            scene._node(name, visible=True)
        panel._hide_model()
        self.assertTrue(panel._hidden, "nothing was hidden, so the test proves nothing")
        panel.server.gui.find("World model").fire(False)
        panel._service_requests()
        self.assertEqual(panel.mode, "raster")
        self.assertEqual(panel._hidden, [], "a switched-off viewport that leaves the scene hidden is a blank page")
        self.assertTrue(all(node.visible for node in scene._handle_from_node_name.values()))

    def test_switching_it_on_asks_for_the_frame_again(self) -> None:
        panel = _panel(world_model=False)
        panel.server.gui.find("World model").fire(True)
        panel._service_requests()
        self.assertEqual(panel.mode, "rtx")

    def test_without_the_tracer_it_renders_with_tiled(self) -> None:
        """No ovrtx or no RT cores is a colliders-only frame, not a dead switch."""
        panel = _tiled_panel(world_model=True)
        self.assertEqual((panel.mode, panel.backend), ("rtx", "tiled"))
        self.assertFalse(panel.server.gui.find("World model").disabled)

    def test_tiled_has_no_passes_to_set(self) -> None:
        """Passes are the path tracer's convergence; a dial that does nothing is greyed out."""
        panel = _tiled_panel()
        self.assertTrue(panel.server.gui.find("Passes moving").disabled)
        self.assertTrue(panel.server.gui.find("Passes still").disabled)
        self.assertFalse(_rtx_panel().server.gui.find("Passes still").disabled)

    def test_a_misspelled_renderer_disables_it_and_says_so(self) -> None:
        """A bad ``ROBOSIMSTUDIO_RENDERER`` is reported on the page, not raised through it."""
        with unittest.mock.patch.dict("os.environ", {"ROBOSIMSTUDIO_RENDERER": "rtxx"}):
            panel = _panel(world_model=True)
        self.assertEqual(panel.mode, "raster")
        self.assertTrue(panel.server.gui.find("World model").disabled)
        panel.refresh()
        self.assertIn("rtxx", panel._readout.value)


class TestWhenTheRendererFails(unittest.TestCase):
    """The rule picks ``rtx`` by a table of GPUs; a failure is the table being wrong."""

    def test_the_machines_pick_falls_back_to_tiled(self) -> None:
        panel = _rtx_panel()
        panel._failed("rtx would not build -- RuntimeError: no ray tracing pipeline")
        self.assertEqual((panel.mode, panel.backend), ("rtx", "tiled"))
        self.assertTrue(panel.server.gui.find("Passes still").disabled)

    def test_a_renderer_somebody_named_is_not_swapped(self) -> None:
        """``rtx`` asked for is tried and, failing, switched off -- the same rule ``choose_renderer`` keeps."""
        for why in ("asked for", "ROBOSIMSTUDIO_RENDERER=rtx"):
            panel = _rtx_panel(why=why)
            panel._failed("rtx would not build -- RuntimeError: no ray tracing pipeline")
            self.assertEqual((panel.mode, panel.backend), ("raster", "rtx"), why)
            self.assertFalse(panel.server.gui.find("World model").value, "a switch left on over no frame is a lie")


class TestCleanView(unittest.TestCase):
    """With the frame alone, the furniture goes too -- and comes back."""

    def _scene(self, panel):
        for name in ("/model/shapes/shape_1", "/studio/right/cursor", "/studio/stage"):
            panel.server.scene._node(name, visible=True)
        return panel.server.scene

    def test_clean_hides_the_furniture_as_well(self) -> None:
        panel = _rtx_panel()
        scene = self._scene(panel)
        panel._clean_request = True
        panel._service_requests()
        self.assertTrue(panel.clean)
        self.assertEqual({n for n, node in scene._handle_from_node_name.items() if node.visible}, set())

    def test_turning_clean_off_gives_the_furniture_back(self) -> None:
        panel = _rtx_panel()
        scene = self._scene(panel)
        panel._clean_request = True
        panel._service_requests()
        panel._clean_request = False
        panel._service_requests()
        self.assertFalse(panel.clean)
        visible = {n for n, node in scene._handle_from_node_name.items() if node.visible}
        self.assertEqual(visible, {"/studio/right/cursor"}, "the model stays hidden; the gizmo comes back")


class TestLockingToARigCamera(unittest.TestCase):
    def test_locking_drops_the_renderer(self) -> None:
        """A rig camera has its own resolution and film back."""
        panel = _panel(world_model=True)
        panel._built = (100, 100, object(), "")
        panel._locked_request = "overhead"
        panel._service_requests()
        self.assertEqual(panel.locked, "overhead")
        self.assertIsNone(panel._built)

    def test_free_is_spelled_as_no_lock(self) -> None:
        panel = _panel(world_model=True)
        panel._locked_request = ""
        panel._service_requests()
        self.assertEqual(panel.locked, "")
        self.assertIsNone(panel._locked_camera())

    def test_a_frame_of_another_shape_is_padded_not_stretched(self) -> None:
        """viser draws a background across the whole frustum, so an unpadded
        frame of another aspect would quietly lie about what the camera sees."""
        panel = _panel(world_model=True)
        client = types.SimpleNamespace(camera=types.SimpleNamespace(image_width=1600, image_height=900))
        square = np.zeros((100, 100, 3), dtype=np.uint8)
        square[:] = 255
        padded = panel._letterbox(square, client)
        self.assertAlmostEqual(padded.shape[1] / padded.shape[0], 1600 / 900, places=2)
        self.assertEqual(padded.shape[0], 100, "the picture keeps its own height")
        self.assertTrue((padded[:, 0] == 0).all(), "and is padded rather than scaled")

    def test_a_frame_that_already_fits_is_left_alone(self) -> None:
        panel = _panel(world_model=True)
        client = types.SimpleNamespace(camera=types.SimpleNamespace(image_width=1600, image_height=900))
        fits = np.zeros((90, 160, 3), dtype=np.uint8)
        self.assertIs(panel._letterbox(fits, client), fits)


class TestTheQualityDial(unittest.TestCase):
    def test_the_passes_pair_is_taken_as_given(self) -> None:
        self.assertEqual(_panel(world_model=True, passes=(2, 9)).passes, (2, 9))

    def test_zero_passes_is_not_a_thing_a_tracer_can_do(self) -> None:
        self.assertEqual(_panel(world_model=True, passes=(0, 0)).passes, (1, 1))

    def test_a_scale_change_drops_the_renderer(self) -> None:
        """The film back is authored once, so a new scale is a new renderer."""
        panel = _panel(world_model=True)
        panel._built = (100, 100, object())
        panel._scale_request = 0.25
        panel._service_requests()
        self.assertIsNone(panel._built, "a scale the renderer was not built for would render the wrong frame")

    def test_switching_to_raster_gives_the_scene_back(self) -> None:
        panel = _rtx_panel()
        node = panel.server.scene._node("/model/shapes/shape_1", visible=True)
        panel._hide_model()
        panel._mode_request = "raster"
        panel._service_requests()
        self.assertEqual(panel.mode, "raster")
        self.assertTrue(node.visible)


if __name__ == "__main__":
    unittest.main()
