# SPDX-License-Identifier: Apache-2.0
"""Drag rigid bodies

A cube under an open hand, a ball and a capsule on a bench: drag the hand down onto the cube, close it and lift.

The right hand starts pointing down over the cube. Drag its gizmo straight down,
press **Close**, and drag it up: the fingers hold the cube by friction alone, so
a grip that slips is a grip that slips. That needs ``rss.MUJOCO_500HZ``, which
puts the arm and every free body in one MuJoCo island; the default physics lets
a hand push a body but not pinch one. A click on any body puts a handle on it
instead, and the body stays fully dynamic under it.

**Key features:**

* :class:`robosimstudio.Scene` and :meth:`robosimstudio.Scene.add` to put things in a place
* :func:`robosimstudio.box`, :func:`robosimstudio.sphere` and :func:`robosimstudio.capsule` for shapes with no asset file
* :data:`robosimstudio.MUJOCO_500HZ` for physics a hand can pick a rigid body up in
* :class:`robosimstudio.ArmSpec` ``home_q`` for a hand that starts over the cube
* :func:`robosimstudio.view` to open the studio page on it
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    cell = "bench_0728"
    cube = (0.0, -0.20)  # where the cube sits, in the bench's work-surface frame [m]
    # The right arm's joints [rad] with its tool centre about 95 mm over the cube's top
    # (145 mm over the bench), pointing down, jaws closing across the bench. Solved once with
    # ``sim.arm("right").solve_q_full``; move the cube, solve again.
    over_cube = (0.274, 2.114, 2.003, -1.459, 0.0, 0.274)

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_cube)))

    scene = rss.Scene(cell=cell, physics=rss.MUJOCO_500HZ, name="rigid")
    scene.add(rss.robot(yam))
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(*cube, 0.025)))
    scene.add(rss.sphere(name="ball", radius=0.03, at=(0.02, 0.0, 0.10)))
    scene.add(rss.capsule(name="pin", radius=0.012, half_height=0.05, at=(0.10, 0.06, 0.12)))
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
