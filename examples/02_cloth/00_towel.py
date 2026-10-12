# SPDX-License-Identifier: Apache-2.0
"""Pinch a towel

A terry towel between two arms: click the cloth to pin a patch and drag it, or close a gripper on an edge.

A click on cloth pins the particles around it; two handles and ``Move together``
carry the sheet by two corners. The default physics lets a hand push a sheet but
not pinch one, so this scene asks for one-way coupling and full-surface contact.
``at`` places the sheet's origin corner, not its middle.

**Key features:**

* :func:`robosimstudio.sheet` for a rectangular cloth
* ``rss.materials.TERRY_TOWELLING``, one of the measured fabrics
* :func:`robosimstudio.replace` on ``rss.GENERIC`` to change the physics arrangement
* :func:`robosimstudio.coupler` and :attr:`robosimstudio.CollisionSpec.full_surface_contact` for a pinch that holds
"""

import robosimstudio as rss


def main():
    cell = "void"
    embodiment = "yam_bimanual"
    size = 0.35
    cells = 30  # quads per axis; the cost is quadratic

    physics = rss.replace(
        rss.GENERIC,
        coupling=rss.coupler("one_way"),
        collision=rss.CollisionSpec(full_surface_contact=True),
    )

    scene = rss.Scene(cell=cell, physics=physics, name="cloth")
    scene.add(rss.robot(embodiment))
    # ``at`` places the sheet's origin corner, not its middle -- so half the edge
    # length in each axis is what centres it under the hands.
    scene.add(
        rss.sheet(
            name="towel",
            size=(size, size),
            cells=(cells, cells),
            at=(-size / 2, -size / 2, 0.25),
            material=rss.materials.TERRY_TOWELLING,
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
