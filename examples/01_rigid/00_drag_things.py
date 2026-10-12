# SPDX-License-Identifier: Apache-2.0
"""Drag rigid bodies

A box, a ball, a capsule and two arms on the floor: click any of them and drag it.

A click puts a handle on a body, and the body stays fully dynamic under it, so
you can push one shape with another or tug on something a hand is holding. Drag
an arm's gizmo to move it through IK. Poses are in the work-surface frame; in
the ``void`` cell that surface is the floor, so ``z`` is a height above ground.

**Key features:**

* :class:`robosimstudio.Scene` and :meth:`robosimstudio.Scene.add` to put things in a place
* :func:`robosimstudio.box`, :func:`robosimstudio.sphere` and :func:`robosimstudio.capsule` for shapes with no asset file
* :func:`robosimstudio.robot` for the arms
* :meth:`robosimstudio.Scene.build` for a running sim
* :func:`robosimstudio.view` to open the studio page on it
"""

import robosimstudio as rss


def main():
    cell = "void"
    embodiment = "yam_bimanual"

    scene = rss.Scene(cell=cell, name="rigid")
    scene.add(rss.robot(embodiment))
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(-0.08, 0.0, 0.20)))
    scene.add(rss.sphere(name="ball", radius=0.03, at=(0.0, 0.0, 0.26)))
    scene.add(rss.capsule(name="pin", radius=0.012, half_height=0.05, at=(0.09, 0.0, 0.22)))
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
