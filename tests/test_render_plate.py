# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio.core.render.plate``: the composite over a clean plate, its shadow pass, and what is static.

The render's moving pixels over the clip's own background: a static row is
the plate's, a broad thing's outer ring fades into it, a shadow darkens only
the plate, and only what fixed joints hold to the world counts as static.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robosimstudio.core.render import plate

TIER = "fast"
"""No GPU: segmentations drawn by hand and a CPU Newton model."""


class TestComposite(unittest.TestCase):
    def test_moving_pixels_are_the_render_and_the_rest_the_plate(self) -> None:
        rgb = np.full((6, 6, 3), 10, np.uint8)
        backdrop = np.full((6, 6, 3), 200, np.uint8)
        seg = np.zeros((6, 6), np.int32)  # a miss
        seg[0, :] = plate.LABEL_STATIC
        seg[1, :] = plate.LABEL_MOVING
        seg[2, :] = -2  # a particle
        seg[3, :] = -3  # a deformable's mesh
        out = plate.composite(rgb, seg, backdrop)
        self.assertTrue((out[0] == 200).all(), "a static row is the plate's")
        self.assertTrue((out[2, 1:5] == 10).all(), "moving, particle and mesh pixels inside are the render's")
        thin = np.zeros((6, 6), np.int32)
        thin[2, :] = -2  # a rope one pixel across is all edge, and keeps its own pixels
        self.assertTrue((plate.composite(rgb, thin, backdrop)[2] == 10).all())
        rgb, backdrop = np.full((9, 9, 3), 10, np.uint8), np.full((9, 9, 3), 200, np.uint8)
        seg = np.zeros((9, 9), np.int32)
        seg[2:7, 2:7] = plate.LABEL_MOVING
        out = plate.composite(rgb, seg, backdrop)
        self.assertEqual(int(out[4, 4, 0]), 10, "a broad thing's inside is the render's")
        self.assertGreater(int(out[2, 4, 0]), 10, "its outer ring fades into the plate")
        self.assertLess(int(out[2, 4, 0]), 200)
        self.assertEqual(int(out[0, 0, 0]), 200)

    def test_a_shadow_darkens_only_the_plate_and_leaves_no_halo(self) -> None:
        bare = np.full((20, 20, 3), 200, np.uint8)
        full = bare.copy()
        full[5:10, 5:15] = 100  # the moving thing's shadow on the table
        full[12:16, 5:15] = 30  # the moving thing itself, dark
        seg = np.zeros((20, 20), np.int32)
        seg[12:16, 5:15] = plate.LABEL_MOVING
        shadow = plate.shade(full, bare, seg)
        self.assertAlmostEqual(float(shadow[7, 10]), 101 / 201, places=3)
        self.assertEqual(float(shadow[14, 10]), 1.0, "the thing's own pixels are not a shadow")
        soft = plate.shade(full, bare, seg, soften=2)
        self.assertGreater(float(soft[17, 10]), 0.99, "no halo of the dark thing under the blur")

    def test_a_thin_thing_is_the_render_only_where_the_render_drew_it(self) -> None:
        seg = np.zeros((9, 9), np.int32)
        seg[2:7, 2:7] = -2  # a rope's collider spheres, wider than the rope drawn
        static_depth = np.full((9, 9), 1.0, np.float32)  # the bench, a metre away
        depth = static_depth.copy()
        depth[4, 2:7] = 0.996  # the rope drawn: 4 mm proud of the bench
        depth[0, 0] = 0.9993  # the path tracer's own mat, 0.7 mm proud, outside the colliders
        out = plate.nearer(seg, depth, static_depth, grow=1)
        moving = (out == plate.LABEL_MOVING) | (out < 0)
        self.assertEqual(np.argwhere(moving)[:, 0].tolist(), [4] * 5, "the rope's row alone")
        self.assertFalse(moving[0, 0], "the mat is the plate's")

    def test_a_rope_in_the_air_keeps_only_its_own_pixels(self) -> None:
        seg = np.zeros((9, 9), np.int32)
        seg[2:7, 2:7] = -2  # the rope's collider spheres, half a metre from the camera
        static_depth = np.zeros((9, 9), np.float32)  # nothing static behind it: the colliders hit nothing
        own = np.where(seg < 0, 0.5, 0.0).astype(np.float32)
        depth = np.full((9, 9), 3.0, np.float32)  # the render's own room, three metres off
        depth[4, 2:7] = 0.502  # the rope's tube
        out = plate.nearer(seg, depth, static_depth, bounds=plate.collider_bounds(seg, own, grow=1), grow=1)
        moving = (out == plate.LABEL_MOVING) | (out < 0)
        self.assertEqual(np.argwhere(moving)[:, 0].tolist(), [4] * 5, "the room behind the spheres is the plate's")

    def test_a_moving_thing_keeps_its_recesses_and_what_stands_proud_of_it(self) -> None:
        # a lockbox face (1 m), a key standing 5 cm proud of it, a seam 3 cm deep across it, nothing static behind
        seg = np.full((12, 12), -2, np.int32)
        own = np.full((12, 12), 1.0, np.float32)
        own[4:8, 5:7] = 0.95  # the key's collider
        depth = own.copy()
        depth[10, :] = 1.03  # the seam: the render looks into it, past the colliders' face
        out = plate.nearer(
            seg, depth, np.zeros((12, 12), np.float32), bounds=plate.collider_bounds(seg, own, grow=4), grow=4
        )
        self.assertTrue(
            ((out == plate.LABEL_MOVING) | (out < 0)).all(), "the face round the key and the seam are drawn"
        )

    def test_a_bulging_collider_over_another_moving_thing_shows_that_thing(self) -> None:
        # a gripper (0.7 m) in front of the box face (1 m); its collider is a pixel wider than its drawn mesh
        seg = np.full((9, 9), -2, np.int32)
        own = np.full((9, 9), 1.0, np.float32)
        own[2:7, 2:7] = 0.7
        depth = own.copy()
        depth[2, 2:7] = 1.0  # the drawn gripper stops a pixel short: the render shows the box there
        out = plate.nearer(
            seg, depth, np.zeros((9, 9), np.float32), bounds=plate.collider_bounds(seg, own, grow=2), grow=2
        )
        self.assertTrue(((out == plate.LABEL_MOVING) | (out < 0)).all(), "the box behind the bulge is the render's")

    def test_a_plate_takes_any_size_asked(self) -> None:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="plate-"))
        Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(tmp / "p.png")
        self.assertEqual(plate.load_plate(tmp / "p.png", 80, 60).shape, (60, 80, 3))
        self.assertEqual(plate.load_plate(tmp / "p.png", 64, 64).shape, (64, 64, 3), "a camera scaled to a square")
        with self.assertRaises(FileNotFoundError):
            plate.load_plate(tmp / "nothing.png", 80, 60)


class TestStatic(unittest.TestCase):
    def test_only_what_is_held_by_fixed_joints_is_static(self) -> None:
        import newton

        b = newton.ModelBuilder()
        fixed = b.add_link()
        b.add_shape_box(fixed, hx=0.1, hy=0.1, hz=0.1)
        j0 = b.add_joint_fixed(-1, fixed)
        hinged = b.add_link()
        b.add_shape_box(hinged, hx=0.1, hy=0.1, hz=0.1)
        j1 = b.add_joint_revolute(fixed, hinged, axis=(0, 0, 1))
        b.add_articulation([j0, j1])
        free = b.add_body()
        b.add_shape_sphere(free, radius=0.05)
        b.add_shape_box(-1, hx=1.0, hy=1.0, hz=0.01)  # on the world
        model = b.finalize(device="cpu")
        self.assertEqual(plate.static_bodies(model).tolist(), [True, False, False])
        self.assertEqual(plate.static_shapes(model).tolist(), [True, False, False, True])


if __name__ == "__main__":
    unittest.main()
