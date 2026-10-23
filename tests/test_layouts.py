# SPDX-License-Identifier: Apache-2.0
"""Seeded spatial variants of a scene, without building one.

:func:`robosimstudio.layouts` is a declaration: it clones a scene and jitters
named things in the work-surface frame. The compiler honouring those poses is
a different claim, tested on a CPU in :mod:`tests.test_layouts_cpu`. This file
is the arithmetic -- independence, seeds, clipping, refusals -- so it gates
every commit.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "fast"
"""No GPU. Gates every commit."""


def _scene():
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075", name="layout")
    scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.1, 0.03)))
    scene.add(rss.sphere(name="ball", radius=0.03, at=(0.05, -0.05, 0.03)))
    scene.add(rss.slab(name="wall", half=(0.01, 0.10, 0.08), at=(0.20, 0.0, 0.08)))
    scene.add(rss.robot("panda"))
    return scene


def _pose(scene, name: str):
    return next(spec.pose for spec in scene.request.entities if spec.name == name)


def _quat(scene, name: str):
    return next(spec.quat for spec in scene.request.entities if spec.name == name)


class TestCloneIsIndependent(unittest.TestCase):
    def test_a_clone_keeps_the_original_poses(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        copy = rss.clone(scene)
        copy.place("cube", at=(0.10, 0.0, 0.03))
        np.testing.assert_allclose(_pose(scene, "cube"), (0.0, 0.1, 0.03))
        np.testing.assert_allclose(_pose(copy, "cube"), (0.10, 0.0, 0.03))
        self.assertIsNot(scene, copy)
        self.assertEqual(sorted(scene._refs), sorted(copy._refs))

    def test_building_the_original_is_not_what_clone_is_for_but_the_refs_differ(self) -> None:
        scene = _scene()
        copy = scene.clone()
        self.assertIsNot(scene._refs["cube"], copy._refs["cube"])


class TestPlace(unittest.TestCase):
    def test_at_moves_a_body_in_the_work_surface_frame(self) -> None:
        scene = _scene()
        scene.place("cube", at=(-0.05, 0.12, 0.04))
        np.testing.assert_allclose(_pose(scene, "cube"), (-0.05, 0.12, 0.04))

    def test_yaw_turns_the_body_about_the_surface_normal(self) -> None:
        scene = _scene()
        scene.place("cube", yaw=math.pi / 2)
        q = np.asarray(_quat(scene, "cube"))
        # A quarter turn about +z from identity is (0, 0, sin(pi/4), cos(pi/4)).
        np.testing.assert_allclose(q, (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)), atol=1e-9)

    def test_a_seven_number_at_and_a_yaw_together_are_refused(self) -> None:
        scene = _scene()
        with self.assertRaisesRegex(ValueError, "already has a rotation"):
            scene.place("cube", at=(0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0), yaw=0.1)

    def test_a_welded_static_has_no_pose_to_amend(self) -> None:
        scene = _scene()
        with self.assertRaisesRegex(ValueError, "cannot be moved"):
            scene.place("wall", at=(0.0, 0.0, 0.08))

    def test_a_missing_name_is_a_keyerror(self) -> None:
        scene = _scene()
        with self.assertRaisesRegex(KeyError, "nothing called 'lid'"):
            scene.place("lid", at=(0.0, 0.0, 0.03))

    def test_place_needs_at_or_yaw(self) -> None:
        scene = _scene()
        with self.assertRaisesRegex(TypeError, "needs at= or yaw="):
            scene.place("cube")


class TestLayouts(unittest.TestCase):
    def test_the_same_seed_draws_the_same_layouts(self) -> None:
        import robosimstudio as rss

        a = rss.layouts(_scene(), 4, seed=7, xy=0.04, yaw=0.3)
        b = rss.layouts(_scene(), 4, seed=7, xy=0.04, yaw=0.3)
        self.assertEqual(len(a), 4)
        for left, right in zip(a, b, strict=True):
            np.testing.assert_allclose(_pose(left, "cube"), _pose(right, "cube"))
            np.testing.assert_allclose(_quat(left, "cube"), _quat(right, "cube"))

    def test_a_different_seed_draws_a_different_layout(self) -> None:
        import robosimstudio as rss

        a = rss.layouts(_scene(), 1, seed=0, xy=0.05)[0]
        b = rss.layouts(_scene(), 1, seed=1, xy=0.05)[0]
        self.assertFalse(np.allclose(_pose(a, "cube"), _pose(b, "cube")))

    def test_the_original_scene_is_not_modified(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        rss.layouts(scene, 3, seed=0, xy=0.08)
        np.testing.assert_allclose(_pose(scene, "cube"), (0.0, 0.1, 0.03))

    def test_a_scalar_xy_moves_every_free_body_and_not_the_wall(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        wall = next(s.xform for s in scene.request.entities if getattr(s, "name", "") == "wall")
        variant = rss.layouts(scene, 1, seed=0, xy=0.05)[0]
        self.assertFalse(np.allclose(_pose(variant, "cube"), _pose(scene, "cube")))
        self.assertFalse(np.allclose(_pose(variant, "ball"), _pose(scene, "ball")))
        got = next(s.xform for s in variant.request.entities if getattr(s, "name", "") == "wall")
        np.testing.assert_allclose(got, wall)

    def test_a_mapping_moves_only_the_named_body(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        variant = rss.layouts(scene, 1, seed=0, xy={"cube": 0.05})[0]
        self.assertFalse(np.allclose(_pose(variant, "cube"), _pose(scene, "cube")))
        np.testing.assert_allclose(_pose(variant, "ball"), _pose(scene, "ball"))

    def test_names_restricts_which_bodies_move(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        variant = rss.layouts(scene, 1, seed=0, xy=0.05, names=("ball",))[0]
        np.testing.assert_allclose(_pose(variant, "cube"), _pose(scene, "cube"))
        self.assertFalse(np.allclose(_pose(variant, "ball"), _pose(scene, "ball")))

    def test_the_robot_is_left_at_its_mount_unless_named(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        self.assertIsNone(scene._robot.at)
        variant = rss.layouts(scene, 1, seed=0, xy=0.05)[0]
        self.assertIsNone(variant._robot.at)

    def test_a_robot_with_at_is_jittered_with_the_free_bodies(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="mounted")
        scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.03)))
        scene.add(rss.robot("panda", at=(-0.5, 0.0, 0.0)))
        variant = rss.layouts(scene, 1, seed=0, xy=0.04)[0]
        self.assertIsNotNone(variant._robot.at)
        self.assertFalse(np.allclose(variant._robot.at[:3], (-0.5, 0.0, 0.0)))

    def test_a_wide_draw_stays_on_the_bench(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        hx, hy = rss.get_cell("bench_075").half[:2]
        for variant in rss.layouts(scene, 16, seed=0, xy=1.0):
            x, y, _ = _pose(variant, "cube")
            self.assertLessEqual(abs(x), hx + 1e-9)
            self.assertLessEqual(abs(y), hy + 1e-9)

    def test_z_stays_put_unless_asked(self) -> None:
        import robosimstudio as rss

        scene = _scene()
        variant = rss.layouts(scene, 1, seed=0, xy=0.05)[0]
        self.assertAlmostEqual(_pose(variant, "cube")[2], 0.03)
        raised = rss.layouts(scene, 1, seed=0, z=0.02)[0]
        self.assertNotAlmostEqual(_pose(raised, "cube")[2], 0.03)

    def test_n_less_than_one_is_refused(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(ValueError, "n >= 1"):
            rss.layouts(_scene(), 0, xy=0.05)

    def test_a_mapping_that_names_a_missing_thing_is_refused(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(ValueError, "lid"):
            rss.layouts(_scene(), 1, xy={"lid": 0.05})

    def test_a_mapping_that_names_the_wall_is_refused(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(ValueError, "cannot be moved"):
            rss.layouts(_scene(), 1, xy={"wall": 0.05})

    def test_layouts_is_on_the_flat_namespace(self) -> None:
        import robosimstudio as rss

        self.assertIn("layouts", rss._EXPORTS)
        self.assertIn("clone", rss._EXPORTS)
        self.assertIn("place", rss._EXPORTS)
        self.assertTrue(callable(rss.layouts))
        self.assertTrue(callable(rss.clone))
        self.assertTrue(callable(rss.place))


if __name__ == "__main__":
    unittest.main()
