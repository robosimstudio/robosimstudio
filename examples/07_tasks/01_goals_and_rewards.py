# SPDX-License-Identifier: Apache-2.0
"""Goals and rewards

A goal is one clause built from measures; a reward says how close an episode came.

This goal asks for more than "in the square": the cube must also be still and
clear of the hand for a whole second. ``once`` latches success the first time
that holds, and ``(fraction)`` pays for each part that holds now. The loop
reads both for one second, then the studio shows each clause as you drag.

**Key features:**

* ``once``, ``hold-for`` and ``and`` over the ``containment``, ``speed`` and ``hands-clear`` measures
* ``(:reward (fraction))``: partial credit, one share per clause
* :meth:`robosimstudio_bench.Env.step`, which returns the reward and ``info["success"]``
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body

TASK = """
(define (problem settle_in_square)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1))
  (:objects cube - cube_50mm)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (once (hold-for 1.0 (and (>= (containment cube square) 1.0)
                                  (<= (speed cube) 0.02)
                                  (> (hands-clear cube) 0.10)))))
  (:reward (fraction)))
"""


def main() -> None:
    cube = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03))
    register_asset(Body.from_spec(cube, name="cube_50mm"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    for _ in range(50):  # one second at 50 Hz, the arm holding still
        _, reward, _, _, info = env.step()
    print(f"success={info['success']} reward={reward:.2f}")  # the cube is still and clear, but not in the square

    rss.view(env)


if __name__ == "__main__":
    main()
