# SPDX-License-Identifier: Apache-2.0
"""Pinch a towel

A terry towel on a bench, its edge under an open hand: drag the hand down onto the edge, close it and lift.

The right hand starts pointing down over the towel's near edge, its jaws across it. Drag
its gizmo down until the fingertips reach the bench, press **Close**, and drag it up: the
pads gather the edge and hold it by friction alone, and the towel hangs from the hand. That
needs one-way coupling and full-surface contact; the default physics lets a hand push a
sheet but not pinch one. A click on the cloth pins a patch to a handle instead, and two
handles with ``Move together`` carry the sheet by two corners.

**Key features:**

* :func:`robosimstudio.sheet` for a rectangular cloth, placed by its corner
* ``rss.materials.TERRY_TOWELLING``, one of the measured fabrics
* :func:`robosimstudio.replace` on ``rss.GENERIC`` with :func:`robosimstudio.coupler` and :attr:`robosimstudio.CollisionSpec.full_surface_contact`, for a pinch that holds
* :class:`robosimstudio.ArmSpec` ``home_q`` for a hand that starts over the edge
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    cell = "bench_0728"
    size = 0.35
    cells = 30  # quads per axis; the cost is quadratic
    edge = -0.20  # where the towel's near edge lies, in the bench's work-surface frame [m]
    # The right arm's joints [rad] with its tool centre 145 mm over the near edge, pointing
    # down, jaws closing across it. Solved once with ``sim.arm("right").solve_q_full``; move
    # the edge, solve again.
    over_edge = (0.274, 2.114, 2.003, -1.459, 0.0, 0.274)

    physics = rss.replace(
        rss.GENERIC,
        coupling=rss.coupler("one_way"),
        collision=rss.CollisionSpec(full_surface_contact=True),
    )

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_edge)))

    scene = rss.Scene(cell=cell, physics=physics, name="cloth")
    scene.add(rss.robot(yam))
    # ``at`` places the sheet's origin corner, not its middle. Built 20 mm up, it lands flat.
    scene.add(
        rss.sheet(
            name="towel",
            size=(size, size),
            cells=(cells, cells),
            at=(-size / 2, edge, 0.02),
            material=rss.materials.TERRY_TOWELLING,
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
