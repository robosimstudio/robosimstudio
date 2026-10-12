# SPDX-License-Identifier: Apache-2.0
"""Drag a cable

A cable lying on the bench between two arms: click any segment and drag it, or close a gripper on it and lift.

A rope is a chain of short capsules on cable joints, and every capsule answers
a click on its own, so pulling it by an end and by its middle are two different
pulls. Take a handle on each end and ``Move together`` to carry it by both.
``at`` places the cable's first end, not its middle.

**Key features:**

* :func:`robosimstudio.rod` for a cable, a rope or a cord
* :class:`robosimstudio.RodSpec`, what :func:`robosimstudio.rod` returns, for its stiffness and friction
* :func:`robosimstudio.robot` with ``"yam_bimanual"`` for two arms to pick it up with
* :func:`robosimstudio.view` to drag it and drive the arms from the page
"""

import robosimstudio as rss


def main():
    length = 0.6  # end to end [m]
    radius = 0.011  # of each capsule [m]
    segments = 28  # capsules; their spacing must stay above twice the radius

    scene = rss.Scene(cell="bench_075", name="cable")
    scene.add(rss.robot("yam_bimanual"))
    # ``at`` is the first end, so half the length back centres the cable on the bench.
    scene.add(
        rss.rod(
            name="cable",
            length=length,
            radius=radius,
            segments=segments,
            at=(0.0, -length / 2, radius + 0.01),
            along=(0.0, 1.0, 0.0),
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
