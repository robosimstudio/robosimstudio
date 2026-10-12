# SPDX-License-Identifier: Apache-2.0
"""Episodes and samplers

A task has numbered episodes, and a sampler decides how each one starts.

``register_sampler`` names a function a document can cite. It returns where the
cube starts in episode *k*, drawn from a generator seeded for that episode, so
``env.reset(k)`` builds the same start every time it is asked for.

**Key features:**

* :func:`robosimstudio_bench.register_sampler` and ``(:episodes (:count N) (:sampler NAME))``
* ``env.reset(k)``: the same start for the same ``k``
* ``object_poses``: x, y, z and yaw, measured from the middle of the bench top
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body

TASK = """
(define (problem cube_anywhere)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1)
            (:episodes (:count 8) (:sampler cube_anywhere)))
  (:objects cube - cube_50mm)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (>= (containment cube square) 1.0)))
"""


@bench.register_sampler("cube_anywhere")
def cube_anywhere(task, episode: int, rng) -> dict:
    x, y, yaw = rng.uniform(-0.10, 0.05), rng.uniform(0.0, 0.15), rng.uniform(-0.5, 0.5)
    return {"object_poses": {"cube": (x, y, 0.03, yaw)}}


def main() -> None:
    cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03))
    register_asset(Body.from_spec(cube, name="cube_50mm"))

    env = bench.Env(bench.task(TASK))
    for episode in (0, 1, 0):
        env.reset(episode)
        print(f"episode {episode}: the cube starts at {env['cube'].position.round(3)}")

    rss.view(env)


if __name__ == "__main__":
    main()
