# SPDX-License-Identifier: Apache-2.0
"""Open a bag, scored

The same bag and hand as a task document, with a goal on how far the mouth stands open: turn the right hand to point down, open the bag as in ``03_materials/06_bag``, and the goal turns true at 30 mm.

The bag is a ``Bag.from_spec`` record the document places under its own name, the physics is
``rss.BAG_60HZ`` under a task-language name, and the staging states the page's own finger servo
the pinch needs. The hands start at the YAM's home, pointing ahead, so turn the right gizmo down
first. ``tests/test_open_a_bag_scored`` drives the right hand through the page and ``evaluate``
says success, the mouth 75 mm open; 12 ms per control step at rest through the page on an
RTX PRO 6000, GPU 3. Needs the ``[superdex]`` extra (Python 3.12).
"""

import numpy as np

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio_materials.bag import Bag

TASK = """
(define (problem open_a_bag)
  (:language open the bag)
  (:staging (:scene bench_0728) (:embodiment yam_bimanual) (:physics bag_60hz@v1)
            (:gripper-actuation (:ke 1500) (:kd 60)))
  (:objects bag - canvas_bag)
  (:goal (>= (mouth_opening bag) 0.03)))
"""


def main() -> None:
    turn = (0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5))  # the mouth toward -y, the right hand's side, as in 00

    bench.register_arrangement("bag_60hz@v1", rss.BAG_60HZ)
    register_asset(Bag.from_spec(rss.bag(name="bag", quat=turn), name="canvas_bag"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()
