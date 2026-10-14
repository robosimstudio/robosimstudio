# SPDX-License-Identifier: Apache-2.0
"""Four studio panels that went stale when something other than the obvious path changed.

* The cloth fold's Set dropdown kept the previous cloth's strips when the Cloth
  dropdown was changed by hand, so Grab set grabbed nothing.
* Unticking Names left every region label frozen where it last was.
* Unticking Tint when satisfied left a satisfied region green.
* The path-traced viewport sized its one renderer from the first browser and
  stretched that frame over a second browser of another shape.
* A reset that recompiled the model left the page bound to the old one.
"""

from __future__ import annotations

import dataclasses
import types
import unittest
import unittest.mock

import numpy as np

from robosimstudio_studio.panels import Panel, PanelCloth
from robosimstudio_studio.regions import HELD_COLOR, IDLE_COLOR, PanelRegions
from robosimstudio_studio.render_view import PanelRenderView
from tests.fixtures.viser_stubs import _StubGui, _StubScene, _StubSession

TIER = "fast"


class TestTheSetListFollowsTheClothDropdown(unittest.TestCase):
    def _cloth_panel(self) -> tuple[PanelCloth, _StubGui]:
        deformables = {
            "towel": types.SimpleNamespace(sets={"towel/corner": [0, 1]}),
            "shirt": types.SimpleNamespace(sets={"shirt/hem": [2, 3], "shirt/collar": [4]}),
        }
        env = types.SimpleNamespace(handles=types.SimpleNamespace(deformables=deformables))
        gui = _StubGui()
        session = _StubSession(env, gui)
        # Built without its constructor, which wants a GPU model for its draggers.
        panel = PanelCloth.__new__(PanelCloth)
        Panel.__init__(panel, session)
        panel.names = ["towel", "shirt"]
        panel.selected = "towel"
        panel.draggers = [object()]
        panel.chrome = True
        panel.grab_on_click = True
        panel.follow_arm = [""]
        panel._dropdown = None
        panel.build(gui)
        return panel, gui

    def test_changing_the_cloth_by_hand_moves_the_set_list(self) -> None:
        panel, gui = self._cloth_panel()
        self.assertEqual(panel.selected_set, "corner")
        gui.find("Cloth").fire("shirt")
        self.assertEqual(panel.selected, "shirt")
        self.assertEqual(tuple(gui.find("Set").options), ("collar", "hem"))
        self.assertIn(panel.selected_set, ("collar", "hem"), "the towel's strip is not on the shirt")


class _RegionScene(_StubScene):
    def add_box(self, name, **kwargs):
        node = self._node(name, True)
        node.color = kwargs.get("color")
        return node

    def add_label(self, name, text, **kwargs):
        node = self._node(name, True)
        node.position = kwargs.get("position")
        return node


def _regions_panel():
    region = types.SimpleNamespace()
    env = types.SimpleNamespace(
        handles=types.SimpleNamespace(regions={"bin": region}),
        goal=None,
        step_count=0,
        model=types.SimpleNamespace(body_count=0),
    )
    session = _StubSession(env, _StubGui())
    session.server.scene = _RegionScene()
    panel = PanelRegions(session)
    panel.bounds = lambda name, body_q: (np.zeros(3), np.ones(3))
    panel.build(session.server.gui)
    return panel, env


class TestRegionToggles(unittest.TestCase):
    def test_unticking_names_takes_the_labels_down(self) -> None:
        panel, _env = _regions_panel()
        panel.drain()
        label = panel._nodes["bin/label"]
        panel.server.gui.find("Names").fire(False)
        panel.drain()  # same step: a paused page
        self.assertTrue(label.removed, "a label left standing, frozen where the region was")
        self.assertNotIn("bin/label", panel._nodes)
        panel.server.gui.find("Names").fire(True)
        panel.drain()
        self.assertIn("bin/label", panel._nodes)

    def test_unticking_tint_repaints_a_green_box(self) -> None:
        panel, env = _regions_panel()
        panel._clause = {"bin": "in bin"}
        env.last_result = types.SimpleNamespace(metrics={"in bin": True})
        panel.drain()
        box = panel._nodes["bin/box"]
        self.assertEqual(box.color, HELD_COLOR)
        panel.server.gui.find("Tint when satisfied").fire(False)
        panel.drain()
        self.assertEqual(box.color, IDLE_COLOR)


@dataclasses.dataclass
class _Spec:
    name: str = "overhead"
    width: int = 0
    height: int = 0
    fx: object = None
    fy: object = None
    cx: object = None
    cy: object = None
    fov_y: float = 60.0
    xform: object = None
    parent_body: int = -1
    parent_frame: str = ""
    eye: tuple = ()
    look_at: tuple = ()
    up: tuple = ()
    channels: tuple = ()


class _Renderer:
    def __init__(self, cameras) -> None:
        self.cameras = list(cameras)
        self.closed = False
        self.accumulate = 1

    def render(self, state):
        spec = self.cameras[0]
        return {spec.name: {"rgb": np.zeros((spec.height, spec.width, 3), dtype=np.uint8)}}

    def close(self) -> None:
        self.closed = True


def _client(cid: int, width: int, height: int):
    camera = types.SimpleNamespace(
        image_width=width,
        image_height=height,
        position=(1.0, 0.0, 1.0),
        look_at=(0.0, 0.0, 0.0),
        up_direction=(0.0, 0.0, 1.0),
        fov=1.0,
    )
    scene = types.SimpleNamespace(set_background_image=lambda image, **_: setattr(scene, "background", image))
    return types.SimpleNamespace(client_id=cid, camera=camera, scene=scene)


class TestEachBrowserGetsItsOwnShape(unittest.TestCase):
    def test_two_browsers_of_different_aspect_get_frames_of_their_own_aspect(self) -> None:
        built: list[_Renderer] = []

        def make_renderer(backend, cameras, channels, **_):
            built.append(_Renderer(cameras))
            return built[-1]

        env = types.SimpleNamespace(
            model=object(), cameras=(_Spec(),), state=object(), make_renderer=make_renderer, handles=None
        )
        session = _StubSession(env, _StubGui())
        clients = {1: _client(1, 1600, 900), 2: _client(2, 800, 800)}
        session.server.get_clients = lambda: clients
        with unittest.mock.patch("robosimstudio_studio.render_view.choose_renderer", return_value=("tiled", "x")):
            panel = PanelRenderView(session, world_model=True, scale=0.5)
        panel.drain()
        wide, square = clients[1].scene.background, clients[2].scene.background
        self.assertEqual(wide.shape[:2], (450, 800))
        self.assertEqual(square.shape[:2], (400, 400), "the second browser got the first one's frame")
        self.assertEqual(len(built), 2)

        # The second browser goes away: its renderer is given back.
        del clients[2]
        panel.budget.reset()
        panel.drain()
        self.assertEqual([r.closed for r in built], [False, True])
        panel.close()
        self.assertTrue(all(r.closed for r in built))

    def test_a_resize_gives_the_old_renderer_back_before_building_the_new_one(self) -> None:
        built: list[_Renderer] = []
        alive_at_build: list[int] = []

        def make_renderer(backend, cameras, channels, **_):
            alive_at_build.append(sum(not r.closed for r in built))
            built.append(_Renderer(cameras))
            return built[-1]

        env = types.SimpleNamespace(
            model=object(), cameras=(_Spec(),), state=object(), make_renderer=make_renderer, handles=None
        )
        session = _StubSession(env, _StubGui())
        clients = {1: _client(1, 1600, 900)}
        session.server.get_clients = lambda: clients
        with unittest.mock.patch("robosimstudio_studio.render_view.choose_renderer", return_value=("tiled", "x")):
            panel = PanelRenderView(session, world_model=True, scale=0.5)
        panel.drain()
        clients[1] = _client(1, 1200, 900)
        panel.budget.reset()
        panel.drain()
        self.assertEqual(len(built), 2)
        self.assertEqual(alive_at_build, [0, 0], "the retired shape was still alive while its successor built")
        self.assertEqual(clients[1].scene.background.shape[:2], (450, 600))
        panel.close()


class TestAResetThatRecompilesRebindsThePage(unittest.TestCase):
    def _session(self, swap: bool):
        from robosimstudio_studio.session import StudioSession

        session = StudioSession.__new__(StudioSession)
        env = types.SimpleNamespace(model=object())

        def reset():
            if swap:
                env.model = object()
            return {}, {}

        env.reset = reset
        session.env = env
        session.panels = []
        session._drop_panels = unittest.mock.Mock()
        session._bind = unittest.mock.Mock()
        session.on_reset = unittest.mock.Mock()
        return session

    def test_a_new_model_rebuilds_the_page_around_it(self) -> None:
        session = self._session(swap=True)
        self.assertTrue(session.reset_now())
        session._drop_panels.assert_called_once_with()
        session._bind.assert_called_once_with(session.env)
        session.on_reset.assert_called_once()

    def test_the_same_model_leaves_the_page_alone(self) -> None:
        session = self._session(swap=False)
        self.assertTrue(session.reset_now())
        session._bind.assert_not_called()
        session._drop_panels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
