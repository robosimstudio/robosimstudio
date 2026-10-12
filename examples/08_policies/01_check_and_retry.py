# SPDX-License-Identifier: Apache-2.0
"""Check and retry

A graph can check its own work and branch: here, a missed grasp is tried once more.

``pick_place_retry/`` splits the job into two subgraphs. ``pick`` turns the jaws
to close across the cube's faces, lifts, and raises if the hand shut on air; its
``on_error`` names that outcome, and the top level routes it to a second copy of
``pick``. The cube starts somewhere new, turned, in each of four episodes.

**Key features:**

* subgraphs, ``on_error`` outcomes and ``conditional_edges`` that route them
* a script node that calls a tool itself: ``ctx.tool("robot.grasp_frame", ...)``
* ``roll="held"``, so the wrist turns to the heading the graph asked for
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_policy_agent import SolutionGraph

TASK = """
(define (problem pick_turned_cube)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 2500)
            (:episodes (:count 4) (:sampler cube_turned)))
  (:objects cube - cube_50mm)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (once (hold-for 0.5 (and (>= (containment cube square) 1.0) (<= (speed cube) 0.02)
                                  (> (hands-clear cube) 0.08)))))
  (:artifacts (:graph pick "pick_place_retry")))
"""


@bench.register_sampler("cube_turned")
def cube_turned(task, episode: int, rng) -> dict:
    return {"object_poses": {"cube": (rng.uniform(-0.10, 0.05), rng.uniform(0.0, 0.15), 0.03, rng.uniform(-0.8, 0.8))}}


def main() -> None:
    cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03))
    register_asset(Body.from_spec(cube, name="cube_50mm"))

    env = bench.Env(bench.task(TASK))
    policy = SolutionGraph(env, graph="pick", roll="held")
    results = [bench.rollout(env, policy, episode_id=k) for k in range(4)]
    print(f"{sum(r.success for r in results)} of {len(results)} episodes succeeded")

    rss.view(env)


if __name__ == "__main__":
    main()
