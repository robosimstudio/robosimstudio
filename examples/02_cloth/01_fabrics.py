# SPDX-License-Identifier: Apache-2.0
"""Fabrics

The same square of cloth in four shipped fabrics, each draped over its own post.

What a sheet is made of is a :class:`robosimstudio.ClothMaterialSpec`: stretch
and bending stiffness, weight per area, thickness. ``rss.materials`` holds the
fabrics that ship. Along the bench (+y) the posts carry a jersey tee, packing
tape, sweatshirt fleece and terry towelling, from limp to stiff.

In the studio open the **Material** folder: ``Cloth`` picks one of the four and
the sliders under it change that sheet while it hangs.

**Key features:**

* :func:`robosimstudio.sheet` with ``material=`` to say what a sheet is made of
* ``rss.materials``, the shipped fabrics, each a :class:`robosimstudio.ClothMaterialSpec`
* :func:`robosimstudio.post` for something welded in place to drape over
* :class:`robosimstudio.CollisionSpec` with ``full_surface_contact`` for contact over faces, not only vertices
* ``panels=`` on :func:`robosimstudio.view` to put the live material sliders on the page
"""

import robosimstudio as rss


def main() -> None:
    size = 0.24  # edge of each square sheet [m]
    cells = 24  # quads along an edge; the cost grows with its square
    post_radius = 0.03
    post_height = 0.16
    spacing = 0.30  # from one post to the next [m]
    fabrics = {
        "jersey": rss.materials.SHIRT_JERSEY,
        "tape": rss.materials.TAPE_MATERIAL,
        "fleece": rss.materials.SHIRT_FLEECE,
        # Last on purpose. Contact stiffness and friction against a sheet are one
        # number for the whole scene, and the last sheet added sets them.
        "towelling": rss.materials.TERRY_TOWELLING,
    }

    # Contact over each sheet's faces and edges as well as its vertices. With
    # vertices alone a post's rim slips between them and shows through.
    physics = rss.replace(rss.GENERIC, collision=rss.CollisionSpec(full_surface_contact=True))

    scene = rss.Scene(cell="bench_075", physics=physics, name="fabrics")
    for index, (name, fabric) in enumerate(fabrics.items()):
        y = (index - 1.5) * spacing
        scene.add(
            rss.post(name=f"post_{name}", radius=post_radius, half_height=post_height / 2, at=(0.0, y, post_height / 2))
        )
        # A sheet's ``at`` is its corner, so half an edge back centres it over the post.
        corner = (-size / 2, y - size / 2, post_height + 0.02)
        scene.add(rss.sheet(name=name, size=(size, size), cells=(cells, cells), at=corner, material=fabric))
    sim = scene.build()

    # "tune" adds the Material folder to the panels a scene script gets by default.
    rss.view(sim, panels=("inspector", "cloth", "tune"))


if __name__ == "__main__":
    main()
