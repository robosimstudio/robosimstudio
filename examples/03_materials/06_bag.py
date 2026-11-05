# SPDX-License-Identifier: Apache-2.0
"""Bag: open its mouth

A canvas bag lies with its mouth toward the right hand and its top layer's lip turned up: drag the right hand down until its open jaws straddle the lip, press **Close**, drag it up 80 mm and back over the bag, and the mouth opens.

Nothing is pinned: the lip comes up because the pads squeeze it, about 0.3 N a side, and
nothing holds the bag down; its closed end moved 5 mm. Closed 25 mm over the bench and drawn
60 mm back, the mouth read 75-77 mm in seven drives; closed 12-30 mm over it, 66-90 mm; closed
over the lip's top, 45 mm up, the hand lifted nothing and the bag read shut (``f_bag.py``, GPUs
1 and 3). The bag is one SuperDex shell that collides with itself: its layers rest 3.9 mm apart,
a contact radius, and stayed 3.5 mm apart while the hand opened it. Set ``self_contact`` to
False and the top layer lies in the bottom one instead: -0.24 mm at rest, -0.29 mm held open.
The grip is the page's own servo, 8 N a finger: on the YAM's old kp 200 spring the pads squeezed
the lip with 0.02-0.05 N, it slipped, and the mouth opened 64-67 mm (its hand is a 50 N clamp now). The **Bag** folder reads
``mouth_opening``, the widest span across the rim and the area it encloses, and ``layer_gap``.
Through the page, 11 ms per control step at rest and 41 ms while the pads hold the cloth (RTX
PRO 6000, GPU 1). Needs the ``[superdex]`` extra (Python 3.12).
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.bag import layer_gap, mouth_opening


def main():
    cell = "bench_0728"
    self_contact = True  # False: the same bag, its top layer lying in the bottom one
    size = (0.24, 0.20)  # closed end to mouth, and across [m]; 10 mm cells
    turn = (0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5))  # the mouth toward -y, the right hand's side
    # The right arm's joints [rad]: pointing down, jaws closing along y, the tool centre 100 mm over the
    # bench above the lip (``sim.arm("right").solve_q_full``).
    over_lip = (0.437, 2.222, 1.994, -1.344, 0.0, 0.437)
    grip = rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=8.0)  # [N] per finger

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_lip)))

    scene = rss.Scene(cell=cell, physics=rss.BAG_60HZ, name="open_a_bag")
    scene.add(rss.robot(yam, gripper=grip))
    scene.add(rss.bag(name="bag", size=size, quat=turn, self_contact=self_contact))
    sim = scene.build(control_mode="joint_position")
    pouch = sim["bag"]

    def page(studio):
        with studio.gui.add_folder("Bag"):
            height = studio.gui.add_number("mouth height (mm)", 0.0, step=0.1, disabled=True)
            area = studio.gui.add_number("mouth area (cm²)", 0.0, step=0.1, disabled=True)
            gap = studio.gui.add_number("least layer gap (mm)", 0.0, step=0.01, disabled=True)

        def each_step() -> None:
            points = pouch.particle_q
            mouth = mouth_opening(points, pouch.entity)
            height.value = round(1000.0 * mouth["height"], 1)
            area.value = round(1.0e4 * mouth["area"], 1)
            gap.value = round(1000.0 * min(layer_gap(points, pouch.entity), 1.0), 2)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
