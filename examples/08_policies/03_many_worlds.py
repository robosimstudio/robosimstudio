# SPDX-License-Identifier: Apache-2.0
"""Many worlds

A benchmark number is a success rate: run the same graph in many worlds at once.

A job names the task, the policy and how many episodes to run. With
``num_worlds=16`` the fleet runner builds sixteen copies of the scene in one
model and steps them together, each world running its own episode of the
sampler and its own copy of the graph. It writes one results row per episode
and the rate with its 95% interval.

**Key features:**

* :class:`robosimstudio.specs.JobSpec`: task, policy, episodes, ``num_worlds``
* the fleet runner, which batches the graph's motions across the worlds
* :func:`robosimstudio_bench.wilson_interval` for the rate's uncertainty
"""

import tempfile

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio.specs import JobSpec
from robosimstudio_bench.run.fleet import RunnerFleet

TASK = """
(define (problem pick_sixteen)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1) (:max-steps 2500)
            (:episodes (:count 16) (:sampler cube_spread)))
  (:objects cube - cube_50mm)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (once (hold-for 0.5 (and (>= (containment cube square) 1.0) (<= (speed cube) 0.02)
                                  (> (hands-clear cube) 0.08)))))
  (:artifacts (:graph pick "pick_place_retry")))
"""


@bench.register_sampler("cube_spread")
def cube_spread(task, episode: int, rng) -> dict:
    return {"object_poses": {"cube": (rng.uniform(-0.10, 0.05), rng.uniform(0.0, 0.15), 0.03, rng.uniform(-0.8, 0.8))}}


def main() -> None:
    cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03))
    register_asset(Body.from_spec(cube, name="cube_50mm"))
    task = bench.task(TASK)

    job = JobSpec(
        name="pick_sixteen",
        task=task.full_name,
        solution="robosimstudio_policy_agent.graph_solution:SolutionGraph",
        solution_kwargs=(("graph", "pick"), ("roll", "held")),
        episodes=(0, 16),
        num_worlds=16,
    )
    rows = RunnerFleet(job, root=tempfile.mkdtemp()).run()
    wins = sum(bool(row["success"]) for row in rows)
    low, high = bench.wilson_interval(wins, len(rows))
    print(f"{wins} of {len(rows)} succeeded; 95% interval {low:.0%} to {high:.0%}")


if __name__ == "__main__":
    main()
