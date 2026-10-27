# SPDX-License-Identifier: Apache-2.0
"""A page's view starts on the middle of the work surface, and orbits it.

Newton 1.7's ``ViewerViser`` opens on a camera of its own: 10 m out along ``+x``, level,
orbiting a point 5 m in front of it, which is 4.4 m past a bench and in empty air. A drag
swung the bench round that point instead of turning it, a scroll zoomed towards it, and
Newton sends that camera to each browser as it connects, so a page that set
``server.initial_camera`` was overridden. The bare Newton viewer is built beside RSS's, so
the day Newton stops opening there this test says so rather than passing on nothing.
"""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

TIER = "fast"

NEWTON_PORT = 9821
PAGE_PORT = 9822
STUDIO_PORT = 9823


def _model():
    import newton

    return newton.ModelBuilder().finalize(device="cpu")


def _browser():
    """A browser as Newton's connect handler meets it: an id, a camera, and a batch of writes."""
    return SimpleNamespace(client_id=1, camera=SimpleNamespace(), atomic=nullcontext)


def _connects(viewer) -> SimpleNamespace:
    """What a browser that connects now is shown, through Newton's own connect handler."""
    browser = _browser()
    viewer._handle_client_connect(browser)
    return browser.camera


def _bench(name: str = "bench_075") -> SimpleNamespace:
    """A built scene as the page reads it: a model, and the cell its compiled scene was staged in."""
    return SimpleNamespace(model=_model(), name="page", scene=SimpleNamespace(handles=SimpleNamespace(scene_name=name)))


class TestNewtonsOwnCameraOrbitsEmptyAir(unittest.TestCase):
    def test_a_bare_newton_viewer_sends_each_browser_its_own_camera_over_initial_camera(self) -> None:
        from newton.viewer import ViewerViser

        bare = ViewerViser(port=NEWTON_PORT)
        try:
            bare.set_model(_model())
            bare._server.initial_camera.look_at = (0.6, 0.0, 0.75)  # what two demos wrote
            camera = _connects(bare)
        finally:
            bare.close()
        np.testing.assert_allclose(camera.position, (10.0, 0.0, 2.0), atol=1e-9)
        np.testing.assert_allclose(camera.look_at, (5.0, 0.0, 2.0), atol=1e-9)


class TestAPageOpensOnTheWorkSurface(unittest.TestCase):
    def test_rss_viewer_opens_on_the_middle_of_the_bench_and_aim_camera_moves_an_open_browser(self) -> None:
        from robosimstudio.scene.registry import get_cell
        from robosimstudio.sim.draw import HOME_EYE, Viewer

        cell = get_cell("bench_075")
        middle = np.array((cell.center[0], cell.center[1], cell.surface_z))
        page = Viewer(_bench(), port=PAGE_PORT)
        try:
            camera = _connects(page._viewer)
            np.testing.assert_allclose(camera.look_at, middle, atol=1e-9)
            np.testing.assert_allclose(camera.position, middle + HOME_EYE, atol=1e-9)
            np.testing.assert_allclose(camera.up_direction, (0.0, 0.0, 1.0))
            np.testing.assert_allclose(page.server.initial_camera.look_at, middle, atol=1e-9)

            open_browser = _browser()
            page.server.get_clients = lambda: {1: open_browser}
            page.aim_camera((1.0, 0.2, 1.1), (0.5, 0.1, 0.8))
            np.testing.assert_allclose(open_browser.camera.look_at, (0.5, 0.1, 0.8))
            np.testing.assert_allclose(_connects(page._viewer).position, (1.0, 0.2, 1.1))
        finally:
            page.close()

    def test_a_scene_in_no_registered_cell_looks_at_the_origin(self) -> None:
        from robosimstudio.sim.draw import HOME_EYE, cell_of, home_view

        self.assertIsNone(cell_of(SimpleNamespace(model=None)))
        self.assertIsNone(cell_of(_bench("")))
        self.assertIsNone(
            cell_of(_bench("a_cell_its_task_directory_declares")), "not registered here: no cell, no raise"
        )
        eye, look_at = home_view(None)
        self.assertEqual((0.0, 0.0, 0.0), look_at)
        self.assertEqual(HOME_EYE, eye)


class TestTheStudioAimsOnlyWhenTheSceneMoves(unittest.TestCase):
    """The live page swaps scenes under an open browser, and its camera stays where it was left."""

    def test_the_first_scene_turns_an_open_browser_a_rebuild_keeps_it_and_another_cell_rehomes_new_ones(self) -> None:
        from robosimstudio.scene.registry import get_cell
        from robosimstudio.sim.draw import aim_camera
        from robosimstudio_studio.garments import viewer_studio
        from robosimstudio_studio.session import StudioSession

        viewer = viewer_studio(port=STUDIO_PORT, share=False)
        studio = SimpleNamespace(viewer=viewer, _aimed_at=None)
        try:
            viewer.set_model(_model())
            open_browser = _browser()  # connected while the first scene compiled, so on Newton's camera
            viewer._server.get_clients = lambda: {1: open_browser}
            StudioSession._aim_home(studio, _bench("bench_075"))
            self.assertEqual("bench_075", studio._aimed_at)
            first = get_cell("bench_075")
            np.testing.assert_allclose(
                open_browser.camera.look_at, (first.center[0], first.center[1], first.surface_z), atol=1e-9
            )

            aim_camera(viewer, (0.95, -0.1, 1.1), (0.53, -0.12, 0.76))  # a page's own view, as page(studio) aims it
            StudioSession._aim_home(studio, _bench("bench_075"))  # a save that rebuilt the same place
            np.testing.assert_allclose(_connects(viewer).look_at, (0.53, -0.12, 0.76))

            open_browser.camera = SimpleNamespace(look_at="where the person left it")
            StudioSession._aim_home(studio, _bench("bench_0728"))
            self.assertEqual("where the person left it", open_browser.camera.look_at)
            other = get_cell("bench_0728")
            np.testing.assert_allclose(
                _connects(viewer).look_at, (other.center[0], other.center[1], other.surface_z), atol=1e-9
            )
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
