# SPDX-License-Identifier: Apache-2.0
"""Drag a cable

A cable lying on the bench under an open hand: drag the hand down onto it, close it and lift, or click any segment and drag it.

The right hand starts pointing down over the capsule nearest the cable's middle.
Drag its gizmo straight down until the fingertips are just over the bench, press
**Close**, and drag it up slowly: the pads hold the cable by friction alone, so a
grip that slips is a grip that slips. That takes static friction, which the
default contact law does not have: ``contact_matching="sticky"`` keeps each
contact's anchor from step to step. A click on any segment puts a handle on it
instead, and ``at`` places the cable's first end, not its middle.

**Key features:**

* :func:`robosimstudio.rod` for a cable, a rope or a cord, with ``scene_friction`` for the pads' grip
* :class:`robosimstudio.CollisionSpec` ``contact_matching="sticky"`` for friction that holds still
* :class:`robosimstudio.ArmSpec` ``home_q`` for a hand that starts over the cable
* :func:`robosimstudio.view` to drag it and drive the arms from the page
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main():
    length = 0.62  # end to end [m]
    radius = 0.011  # of each capsule [m]
    segments = 28  # capsules; their spacing (22.1 mm) must stay above twice the radius
    x = -0.05  # the line the cable lies on, in the bench's frame [m]: 50 mm toward the arms, inside their reach
    # The right arm's joints [rad]: tool centre 100 mm over the bench above capsule 13, by
    # the middle, pointing down, jaws across the cable. Solved with ``sim.arm("right").solve_q_full``.
    over_cable = (0.857, 2.337, 2.189, -1.422, 0.0, -0.714)

    # Under the default contact law, which has no static friction, a pinched capsule crept
    # through the pads at 15 mm/s and fell out of a 0.15 m/s lift. Anchored contacts (both
    # settings below) held it within 1 mm of where it was pinched through a 0.4 m/s lift.
    physics = rss.replace(
        rss.GENERIC.with_backend(
            "*", rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)
        ),
        collision=rss.CollisionSpec(contact_matching="sticky"),
    )

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    yam = rss.replace(yam, arms=(left, rss.replace(right, home_q=over_cable)))

    scene = rss.Scene(cell="bench_075", physics=physics, name="cable")
    scene.add(rss.robot(yam))
    # ``at`` is the first end, so half the length back centres the cable on the bench.
    scene.add(
        rss.rod(
            name="cable",
            length=length,
            radius=radius,
            segments=segments,
            at=(x, -length / 2, radius + 0.01),
            along=(0.0, 1.0, 0.0),
            allow_grasp_latch=False,  # nothing holds it but the pads
            scene_friction=True,  # the pads' friction is the rod's mu_gripper (60), not the bench's 2
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
