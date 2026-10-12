# SPDX-License-Identifier: Apache-2.0
"""Cable between two posts

A cable clamped by both ends on top of two posts: hook it with a gripper or drag its middle, and it springs back.

``anchors`` clamps a cable's ends to the world where they were built: the end's
position, its direction and its twist are held, so a cable clamped by one end
is a cantilever and by both a span that sags under its own weight. The posts
are scenery to rest the ends on, and the cable is built just clear of their
tops so that nothing pushes against a clamp.

**Key features:**

* :func:`robosimstudio.rod` with ``anchors=("start", "end")`` to clamp both ends
* :func:`robosimstudio.post` for something welded in place to string it between
* :func:`robosimstudio.robot` with ``"yam_bimanual"`` for two arms to hook it with
* :func:`robosimstudio.view` to pull the span and watch it recover
"""

import robosimstudio as rss


def main():
    span = 0.5  # from post to post [m]
    radius = 0.008  # of the cable [m]
    segments = 24
    post_height = 0.16  # [m]

    scene = rss.Scene(cell="bench_075", name="cable_between_posts")
    scene.add(rss.robot("yam_bimanual"))
    for name, y in (("post_start", -span / 2), ("post_end", span / 2)):
        scene.add(rss.post(name=name, radius=0.02, half_height=post_height / 2, at=(0.0, y, post_height / 2)))
    scene.add(
        rss.rod(
            name="cable",
            length=span,
            radius=radius,
            segments=segments,
            at=(0.0, -span / 2, post_height + radius + 0.002),  # 2 mm clear of the post's top
            along=(0.0, 1.0, 0.0),
            anchors=("start", "end"),
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
