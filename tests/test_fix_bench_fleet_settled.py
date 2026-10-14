# SPDX-License-Identifier: Apache-2.0
"""A fleet world's start pose is the post-reset pose, however late it is asked for."""

import unittest

import numpy as np

TIER = "medium"

TASK = """
(define (problem fix_bench_settled)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 50)
            (:episodes (:count 2)))
  (:objects cube - fix_bench_cube_50mm)
  (:goal (<= (speed cube) 0.02)))
"""


class TestTheSettledPoseIsTakenAtReset(unittest.TestCase):
    def test_asked_after_the_world_moved(self):
        import robosimstudio as rss
        import robosimstudio_bench as bench
        from robosimstudio.assetlib import register_asset
        from robosimstudio.domains.rigid import Body
        from robosimstudio_bench.envs.fleet import FleetEnv
        from robosimstudio_bench.run.fleet import plan_waves

        # Dropped from 100 mm, so the first steps move it.
        cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.10))
        register_asset(Body.from_spec(cube, name="fix_bench_cube_50mm"))
        task = bench.task(TASK)
        fleet = FleetEnv(task, num_worlds=2)
        fleet.reset_wave(plan_waves(task, [0, 1], 2)[0])
        start = fleet.snapshot()
        for _ in range(20):
            fleet.step()
        moved = fleet.snapshot()
        for world in range(2):
            self.assertGreater(np.abs(moved.body(world) - start.body(world)).max(), 1e-3)
            np.testing.assert_array_equal(fleet.settled_body_q(world), start.body(world))


if __name__ == "__main__":
    unittest.main()
