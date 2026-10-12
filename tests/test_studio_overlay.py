# SPDX-License-Identifier: Apache-2.0
"""Drawing declared geometry without a browser.

The **overlay** must not compute a transform. Every payload it draws goes through
the same `bind` reader a *measure* resolves its operands through, so the picture
and the measurement cannot disagree -- and a disagreeing picture is worse than no
picture, because it is believed. What is testable here without a scene is the
geometry it builds from an already-placed payload, and the fallbacks: a feature
that cannot be placed is not drawn, rather than drawn wrong or raised over.
"""

from __future__ import annotations

import itertools
import types
import unittest

import numpy as np

from robosimstudio_studio.overlay import COLOR, HELD_COLOR, PanelOverlay, _basis, _unit, _vec, _wxyz
from robosimstudio_studio.regions import HELD_COLOR as REGION_GREEN

TIER = "fast"
"""No GPU, no browser. Gates every commit."""


class _Scene:
    """Records what would have been drawn, as the handles viser returns."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def _make(self, kind, name, **kwargs):
        self.calls.append((kind, name, kwargs))
        return types.SimpleNamespace(name=name, remove=lambda: None)

    def add_icosphere(self, name, **kw):
        return self._make("icosphere", name, **kw)

    def add_line_segments(self, name, **kw):
        return self._make("line", name, **kw)

    def add_spline_catmull_rom(self, name, **kw):
        return self._make("spline", name, **kw)

    def add_box(self, name, **kw):
        return self._make("box", name, **kw)

    def add_point_cloud(self, name, **kw):
        return self._make("cloud", name, **kw)


def _overlay() -> PanelOverlay:
    panel = PanelOverlay.__new__(PanelOverlay)
    panel.showing = {}
    panel.server = types.SimpleNamespace(scene=_Scene())
    return panel


class TestEachKindDrawsItsOwnShape(unittest.TestCase):
    """The kind decides the shape, because the kind is what says which payload
    fields are geometry at all."""

    def setUp(self) -> None:
        self.panel = _overlay()
        self.scene = self.panel.server.scene

    def test_a_bore_draws_its_mouth_and_a_line_down_it(self) -> None:
        payload = {"mouth": [0.5, 0.0, 0.8], "axis": [0.0, 0.0, -1.0], "depth": 0.013}
        nodes = self.panel._draw_cavity(self.scene, payload, "/f", COLOR)
        self.assertEqual([kind for kind, _, _ in self.scene.calls], ["icosphere", "line"])
        self.assertEqual(len(nodes), 2)
        # The line runs from the mouth *along* the axis, by the declared depth.
        points = self.scene.calls[1][2]["points"]
        np.testing.assert_allclose(points[0][0], [0.5, 0.0, 0.8], atol=1e-9)
        np.testing.assert_allclose(points[0][1], [0.5, 0.0, 0.787], atol=1e-6)

    def test_a_rod_draws_between_its_two_endpoints(self) -> None:
        """Both ends travel with the body, which is why `rod.axis` is declared as
        a pair of points rather than an origin and a direction."""
        payload = {"axis": [[0.44, -0.06, 0.75], [0.44, -0.06, 0.81]], "radius": 0.013}
        self.panel._draw_rod(self.scene, payload, "/f", COLOR)
        points = self.scene.calls[0][2]["points"]
        np.testing.assert_allclose(points[0][0], [0.44, -0.06, 0.75], atol=1e-6)
        np.testing.assert_allclose(points[0][1], [0.44, -0.06, 0.81], atol=1e-6)

    def test_a_loop_draws_a_ring_in_the_plane_its_normal_defines(self) -> None:
        payload = {"center": [0.0, 0.0, 1.0], "normal": [0.0, 0.0, 1.0], "inner_r": 0.05}
        self.panel._draw_loop(self.scene, payload, "/f", COLOR)
        ring = np.asarray(self.scene.calls[0][2]["positions"], dtype=np.float64)
        # Every point is one radius from the centre, and all of them are in the
        # z = 1 plane -- which is what "in the plane the normal defines" means.
        np.testing.assert_allclose(np.linalg.norm(ring[:, :2], axis=1), 0.05, atol=1e-6)
        np.testing.assert_allclose(ring[:, 2], 1.0, atol=1e-9)

    def test_an_axis_draws_through_its_origin_both_ways(self) -> None:
        """An `axis` declares no extent -- it is a directed line, not a segment --
        so the drawn length is presentation and the origin is the fixed point."""
        payload = {"origin": [1.0, 2.0, 3.0], "direction": [0.0, 0.0, 1.0]}
        self.panel._draw_axis(self.scene, payload, "/f", COLOR)
        points = np.asarray(self.scene.calls[0][2]["points"][0], dtype=np.float64)
        np.testing.assert_allclose(points.mean(axis=0), [1.0, 2.0, 3.0], atol=1e-9)

    def test_a_grasp_box_takes_its_orientation_from_the_body(self) -> None:
        """`half` is extents, not geometry, so the feature reader passes it
        through untouched -- the box's rotation is the one thing not in the
        payload and has to come from the body it hangs off."""
        panel = self.panel
        subject = types.SimpleNamespace(
            handle=types.SimpleNamespace(body=-1, xform=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        )
        panel.env = types.SimpleNamespace(state=None)
        payload = {"half": [0.01, 0.005, 0.05], "center_local": [0.0, 0.0, 1.0]}
        panel._draw_grasp_box(self.scene, payload, subject, "/f", COLOR)
        kwargs = self.scene.calls[0][2]
        self.assertEqual(kwargs["dimensions"], (0.02, 0.01, 0.1))
        self.assertEqual(kwargs["wxyz"], (1.0, 0.0, 0.0, 0.0))

    def test_a_kind_with_no_geometry_draws_nothing(self) -> None:
        self.assertEqual(self.panel._draw_kind("template", {"steps": []}, None, "/f", COLOR), [])


class TestAnUndrawablePayloadIsNotDrawn(unittest.TestCase):
    """Not drawn wrong, and not raised over. A drawing that cannot be made is a
    drawing that is not made -- this is a page somebody is looking at."""

    def setUp(self) -> None:
        self.panel = _overlay()
        self.scene = self.panel.server.scene

    def test_a_bore_missing_its_axis(self) -> None:
        self.assertEqual(self.panel._draw_cavity(self.scene, {"mouth": [0, 0, 0]}, "/f", COLOR), [])

    def test_a_loop_with_no_radius(self) -> None:
        payload = {"center": [0, 0, 0], "normal": [0, 0, 1], "inner_r": 0.0}
        self.assertEqual(self.panel._draw_loop(self.scene, payload, "/f", COLOR), [])

    def test_a_rod_with_one_endpoint(self) -> None:
        self.assertEqual(self.panel._draw_rod(self.scene, {"axis": [[0, 0, 0]]}, "/f", COLOR), [])

    def test_a_degenerate_direction_is_not_normalised_into_nonsense(self) -> None:
        self.assertIsNone(_unit([0.0, 0.0, 0.0]))


class TestTheRingBasisIsNeverDegenerate(unittest.TestCase):
    def test_it_holds_for_a_normal_along_every_axis(self) -> None:
        """A seed parallel to the normal would cross to zero, and the loop whose
        normal is +x is the one that would find it."""
        for normal in (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2], _unit([1.0, 1.0, 1.0])):
            with self.subTest(tuple(np.round(normal, 3))):
                u, v = _basis(np.asarray(normal, dtype=np.float64))
                self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=6)
                self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
                self.assertAlmostEqual(float(np.dot(u, normal)), 0.0, places=6)
                self.assertAlmostEqual(float(np.dot(v, normal)), 0.0, places=6)


class TestQuaternionsCrossTheBoundary(unittest.TestCase):
    def test_xyzw_becomes_wxyz(self) -> None:
        """Newton stores xyzw and viser wants wxyz, and a page that got this
        wrong would draw every box a quarter turn out."""
        self.assertEqual(_wxyz([0.0, 0.0, 0.7071, 0.7071]), (0.7071, 0.0, 0.0, 0.7071))

    def test_a_short_vector_is_refused_rather_than_padded(self) -> None:
        self.assertIsNone(_vec([1.0, 2.0]))


class TestRedrawingIsCheap(unittest.TestCase):
    """Two properties that regress silently, because nothing fails when they do
    -- the page keeps working and just costs more.

    `drain` runs once per control step. Redrawing there at step rate sends frames
    the browser is never asked to render, and doing it by remove-then-add sends
    two messages where viser's name-addressed graph needs one. Neither shows up
    as a bug; both show up as a page that stutters on a scene with a cloud drawn.
    """

    def _panel(self, *, render_every: int, step: int) -> PanelOverlay:
        panel = _overlay()
        panel.showing = {}
        panel._synced = -1
        panel.env = types.SimpleNamespace(step_count=step)
        panel.session = types.SimpleNamespace(overlay_request=[], render_every=render_every, index=None)
        return panel

    def test_a_checkbox_is_serviced_whatever_the_gate_says(self) -> None:
        """That one is a person waiting for something to appear."""
        panel = self._panel(render_every=4, step=1)
        panel.session.overlay_request.append(("object:a", "bore", True))
        panel.drain()
        self.assertEqual(panel.session.overlay_request, [], "the request must be taken, not left to repeat")

    def test_it_does_not_redraw_faster_than_the_page_refreshes(self) -> None:
        """The *interval* is the contract, not the phase.

        When the first update lands depends on when the feature was switched on,
        and it does not matter -- switching one on draws it immediately through
        `_show`. What matters is that updates afterwards come no faster than the
        browser is refreshed.
        """
        panel = self._panel(render_every=4, step=0)
        redraws = []
        panel._replace = lambda: redraws.append(panel.env.step_count)
        panel.showing = {("object:a", "bore"): []}
        for step in range(1, 21):
            panel.env.step_count = step
            panel.drain()
        self.assertEqual(len(redraws), 5, f"20 steps at one frame in 4 is 5 redraws, got {redraws}")
        gaps = {b - a for a, b in itertools.pairwise(redraws)}
        self.assertEqual(gaps, {4})

    def test_a_slower_page_redraws_less(self) -> None:
        panel = self._panel(render_every=10, step=0)
        redraws = []
        panel._replace = lambda: redraws.append(panel.env.step_count)
        panel.showing = {("object:a", "bore"): []}
        for step in range(1, 21):
            panel.env.step_count = step
            panel.drain()
        self.assertEqual(len(redraws), 2)

    def test_nothing_showing_costs_nothing(self) -> None:
        panel = self._panel(render_every=1, step=10)
        panel._replace = lambda: self.fail("redrew with nothing to draw")
        panel.drain()

    def test_a_redraw_updates_in_place_rather_than_deleting_first(self) -> None:
        """viser addresses its scene graph by name, so re-adding is an update.
        Removing first makes it a delete *and* a create -- for a drawn particle
        set, the whole cloud twice."""
        panel = self._panel(render_every=1, step=1)
        removed = []
        node = types.SimpleNamespace(name="/f/bore", remove=lambda: removed.append("/f/bore"))
        panel.showing = {("object:a", "bore"): [node]}
        panel._draw = lambda _s, _f: [types.SimpleNamespace(name="/f/bore", remove=lambda: None)]
        panel.session.index = types.SimpleNamespace(by_key=lambda _k: object())
        panel._replace()
        self.assertEqual(removed, [], "a node that is being redrawn must not be removed first")

    def test_a_node_that_stops_being_drawn_is_removed(self) -> None:
        """A payload that stops being placeable would otherwise leave its old
        geometry behind, showing a bore that is no longer being measured."""
        panel = self._panel(render_every=1, step=1)
        removed = []
        stale = types.SimpleNamespace(name="/f/gone", remove=lambda: removed.append("/f/gone"))
        panel.showing = {("object:a", "bore"): [stale]}
        panel._draw = lambda _s, _f: [types.SimpleNamespace(name="/f/kept", remove=lambda: None)]
        panel.session.index = types.SimpleNamespace(by_key=lambda _k: object())
        panel._replace()
        self.assertEqual(removed, ["/f/gone"])


class TestColoursMeanWhatTheRegionBoxesMean(unittest.TestCase):
    def test_amber_and_green_are_distinct(self) -> None:
        self.assertNotEqual(COLOR, HELD_COLOR)

    def test_the_held_colour_matches_the_region_panels(self) -> None:
        """Same green, because it means the same thing -- a clause that holds."""
        self.assertEqual(HELD_COLOR, REGION_GREEN)


if __name__ == "__main__":
    unittest.main()
