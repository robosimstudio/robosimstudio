# SPDX-License-Identifier: Apache-2.0
"""Peel a label, scored

The same box and label as a task document, with a goal on the share of the label's bonds that have let go: turn the right hand to point down, take the tab and peel as in ``03_materials/08_tape``, and the goal turns true at 80%.

The label is a ``Tape.from_spec`` record and the box a ``Body.from_spec`` one, placed by the
document under the names the label's ``on`` and the goal use, and the physics is
``rss.TAPE_60HZ`` under a task-language name. The staging states the finger servo of
``03_materials/08_tape``: gentler than the YAM's own 50 N clamp, damped, and capped at 12 N a finger. The cap
is what Close's first push meets: 1500 times the 47.5 mm to go is about 71 N, held to 12 N as on the
material page; the pads then stall on the tab 3.8 mm short of shut and hold it with about 5.8 N
through the peel (``g2_close.py``, ``g2_grip.py``). The hands start at the YAM's home, pointing
ahead, so turn the right gizmo down first. ``tests/test_peel_a_label_scored`` drives the right
hand through the page and ``evaluate`` says success.
"""

import robosimstudio as rss
import robosimstudio_bench as bench
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body
from robosimstudio_materials.tape import Tape

TASK = """
(define (problem peel_a_label)
  (:language peel the label off the box)
  (:staging (:scene bench_0728) (:embodiment yam_bimanual) (:physics tape_60hz@v1)
            (:gripper-actuation (:ke 1500) (:kd 240) (:effort-limit 12)))
  (:objects box - parcel label - paper_label)
  (:goal (>= (peeled_fraction label) 0.8)))
"""


def main() -> None:
    box_half = (0.05, 0.06, 0.025)  # [m]
    box_at = (0.0, -0.08, box_half[2])  # its centre, in the bench's work-surface frame [m], as in 00
    corner = (0.02, -0.025, 2 * box_half[2] + 0.0005)  # the label's tab end, a particle radius over the top
    half_turn = (0.0, 0.0, 1.0, 0.0)  # the label runs toward -y, back toward the right arm

    bench.register_arrangement("tape_60hz@v1", rss.TAPE_60HZ)
    box = rss.box(name="box", half=box_half, at=box_at, mass=0.5, color=(0.72, 0.56, 0.38))
    register_asset(Body.from_spec(box, name="parcel"))
    label = rss.tape(name="label", on="box", at=corner, quat=half_turn, material=rss.tapes.PAPER_LABEL)
    register_asset(Tape.from_spec(label, name="paper_label"))

    env = bench.Env(bench.task(TASK))
    env.reset()
    rss.view(env)


if __name__ == "__main__":
    main()
