# SPDX-License-Identifier: Apache-2.0
"""The page's per-frame costs, its reset, and its Attach button.

A deforming PBR garment must not leave a GLB behind every frame; a static batch
drawn through the PBR path must still be skipped once pushed, and the shape
colours read back once per repack rather than once per batch; the overlay must
redraw straight after a reset; and an arm chosen in the Object dropdown must be
taken hold of by Attach rather than ignored.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from robosimstudio.core.render.finish import GLOSS
from robosimstudio_studio import finish_gltf
from robosimstudio_studio.fleet_view import _place
from robosimstudio_studio.garments import ViewerStudio
from robosimstudio_studio.overlay import PanelOverlay
from robosimstudio_studio.panels import PanelObject

TIER = "fast"
"""No GPU, no browser."""


class TestDeformingGlbIsNotKept(unittest.TestCase):
    def test_an_uncached_encode_leaves_the_cache_as_it_was(self) -> None:
        finish_gltf.clear_cache()
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        for frame in range(3):
            moved = points + 0.01 * frame
            blob = finish_gltf.glb_for(moved, faces, (0.5, 0.5, 0.5), GLOSS, cache=False)
            if blob is None:
                self.skipTest("no trimesh: nothing is encoded")
        self.assertEqual(len(finish_gltf._CACHE), 0)


class TestStaticPbrBatch(unittest.TestCase):
    def _viewer(self) -> ViewerStudio:
        viewer = ViewerStudio.__new__(ViewerStudio)
        viewer._qualify = lambda name: name
        viewer._static_skip_ok = True
        viewer._static_batches = {"/b"}
        viewer._scene_handles = {}
        viewer._pbr_handles = {"/b": {"/b/m0": object()}}
        viewer._finishes = {0: GLOSS}
        viewer._host_shape_color = None
        return viewer

    def test_a_pushed_pbr_batch_is_skipped(self) -> None:
        viewer = self._viewer()
        drawn = []
        viewer._log_pbr_instances = lambda *a: drawn.append(a) or True
        viewer.log_instances("/b", "/mesh", None, None, None, None)
        self.assertEqual(drawn, [])

    def test_colours_are_read_back_once_per_repack(self) -> None:
        viewer = self._viewer()
        reads = []

        class _Colors:
            def numpy(self):
                reads.append(1)
                return np.ones((2, 3), dtype=np.float32)

        viewer.model = types.SimpleNamespace(shape_color=_Colors())
        viewer._meshes = {"/mesh": object()}
        viewer._plane_meshes = set()
        viewer._live_shapes = frozenset()
        viewer._batch_shapes = lambda q: [0, 1]
        viewer._pbr_groups = {}
        viewer._glb = lambda data, color, finish: None  # stop after the colours
        viewer._retire_pbr_nodes = lambda *a: None
        xforms = np.zeros((2, 7), dtype=np.float32)
        xforms[:, 6] = 1.0
        for _ in range(3):
            viewer._log_pbr_instances("/b", "/mesh", xforms, None)
        self.assertEqual(len(reads), 1)
        viewer.mark_shape_colors_dirty()
        viewer._log_pbr_instances("/b", "/mesh", xforms, None)
        self.assertEqual(len(reads), 2)


class TestOverlayAfterReset(unittest.TestCase):
    def test_the_next_drain_redraws(self) -> None:
        panel = PanelOverlay.__new__(PanelOverlay)
        panel.showing = {("k", "f"): []}
        panel.session = types.SimpleNamespace(overlay_request=[], render_every=4)
        panel.env = types.SimpleNamespace(step_count=100)
        panel._synced = -1
        redrawn = []
        panel._replace = lambda: redrawn.append(panel.env.step_count)
        panel.drain()
        panel.env.step_count = 0
        panel.on_reset({}, {})
        panel.drain()
        self.assertEqual(redrawn, [100, 0])


class _NoHandles(list):
    def drain(self) -> None:
        pass


class TestAttachAnArm(unittest.TestCase):
    def test_attach_takes_the_selected_arm(self) -> None:
        panel = PanelObject.__new__(PanelObject)
        panel.pending_body = None
        panel.attach_requested = True
        panel.release_requested = panel.release_all_requested = False
        panel.selected = "left"
        panel.chains = {}
        panel.arm_bodies = {"left": (7, 5, 3)}
        panel.env = types.SimpleNamespace(handles=types.SimpleNamespace(objects={}))
        panel.handles = _NoHandles()
        panel.draggers = []
        attached = []
        panel._attach = attached.append
        panel.drain()
        self.assertEqual(attached, [7])


class TestFleetViewPlace(unittest.TestCase):
    def test_a_drifted_quaternion_does_not_scale_the_box(self) -> None:
        # A quarter turn about z, its quaternion grown about 2%: not the
        # identity, which an unnormalised matrix also gets right.
        pose = np.array([0, 0, 0, 0, 0, 0.7212, 0.7212])
        _, half = _place(pose, np.zeros(3), np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(half, [0.2, 0.1, 0.3], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
