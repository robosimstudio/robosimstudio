# SPDX-License-Identifier: Apache-2.0
"""Without privilege

A graph that finds the cube and the target through a camera, as a real robot would.

With ``privileged=False`` no ``sim.*`` tool is registered: the graph has what a
real rig has, camera frames and the arm's own state. ``pick_place_camera/``
reads the overhead depth image, takes what stands 50 mm above the bench as the
cube and what stands 4 mm above it as the pad, and runs the same pick, check
and place as the retry example.

**Key features:**

* ``SolutionGraph(..., privileged=False)``: no ``sim.*`` tool, only what a rig would have
* a camera rig the task names, rendered by ``bench.Env(..., renderer="rtx")``
* a script node that back-projects a depth image into world points
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.core.cameras import CameraSpec
from robosimstudio.domains.rigid import Body
from robosimstudio_policy_agent import SolutionGraph

TASK = """
(define (problem pick_by_camera)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 2500)
            (:cameras overhead@v1))
  (:objects cube - cube_50mm)
  (:fixtures pad - pad_120mm)
  (:regions (on_pad (:target pad) (:ranges ((-0.05 -0.05 0.05 0.05)))))
  (:goal (once (hold-for 0.5 (and (>= (containment cube on_pad) 1.0) (<= (speed cube) 0.02)
                                  (> (hands-clear cube) 0.08)))))
  (:artifacts (:graph look "pick_place_camera")))
"""

OVERHEAD = CameraSpec(name="overhead", width=320, height=240, eye=(0.62, 0.0, 1.55), look_at=(0.62, 0.0, 0.75),
                      up=(1.0, 0.0, 0.0), fov_y=50.0, channels=("rgb", "depth"))  # fmt: skip


def main() -> None:
    cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03))
    register_asset(Body.from_spec(cube, name="cube_50mm"))
    pad = rss.slab(name="pad", half=(0.06, 0.06, 0.002), at=(0.1, -0.15, 0.002))
    register_asset(Body.from_spec(pad, name="pad_120mm"))
    bench.register_camera_rig("overhead@v1", (OVERHEAD,))

    env = bench.Env(bench.task(TASK), renderer="rtx")
    result = bench.rollout(env, SolutionGraph(env, graph="look", privileged=False, roll="held"))
    print(f"success={result.success} after {result.steps} steps")

    rss.view(env)


if __name__ == "__main__":
    main()
