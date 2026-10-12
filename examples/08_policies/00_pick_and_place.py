# SPDX-License-Identifier: Apache-2.0
"""Pick and place

A policy is a graph: a folder holding a ``workflow.json`` and the scripts it names.

``pick_place/`` reads where the cube and the square are, works out four hand
poses in one script node, and moves the arm with ``robot.*`` tools. The task
names that folder in ``(:graph pick "pick_place")``, and ``SolutionGraph`` runs
it for one episode, scored the way the benchmark scores every episode.

**Key features:**

* a graph folder: tool nodes, a script node, and ``$ref`` from one node's output to another's input
* ``(:artifacts (:graph LABEL "DIR"))``, resolved beside this script
* :class:`robosimstudio_policy_agent.SolutionGraph` and :func:`robosimstudio_bench.rollout`
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_policy_agent import SolutionGraph

TASK = """
(define (problem pick_and_place)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 1500))
  (:objects cube - cube_50mm)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (once (hold-for 0.5 (and (>= (containment cube square) 1.0) (<= (speed cube) 0.02)
                                  (> (hands-clear cube) 0.08)))))
  (:artifacts (:graph pick "pick_place")))
"""


def main() -> None:
    cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03))
    register_asset(Body.from_spec(cube, name="cube_50mm"))

    env = bench.Env(bench.task(TASK))
    result = bench.rollout(env, SolutionGraph(env, graph="pick"))
    print(f"success={result.success} after {result.steps} steps")

    rss.view(env)


if __name__ == "__main__":
    main()
