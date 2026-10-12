# SPDX-License-Identifier: Apache-2.0
"""Fold the towel

A YAM arm with one edge of a terry towel in its hand: drag the wrist gizmo up and over and the towel folds in half.

The towel is a 30 cm slab of 5 mm, 500 g/m^2 terry on a bench, with a softer core
than the measured fabric so that it drapes in folds from the hand rather than
lifting like a board. The scene starts with the hand already on the far edge: a
70 mm patch under the fingers is the sheet's ``held`` set, and ``held_by`` names
the wrist that carries it, weightless and pinned, from the build onward. Lift the
gizmo to raise the edge, swing it across to the other edge and lower it to lay the
fold. The other arm is free, and a click on the towel puts a handle on that patch.

**Key features:**

* :func:`robosimstudio.sheet` with ``held_by=`` and a ``held`` particle set, for an edge already in the hand
* ``rss.materials.TERRY_TOWELLING`` restated with :func:`robosimstudio.replace` for a limper towel
* :class:`robosimstudio.ArmSpec` ``home_q`` for an arm that starts with its hand on the towel
* :func:`robosimstudio.backend` and :func:`robosimstudio.coupler` for the cloth solver a fold needs
* ``page=`` on :func:`robosimstudio.view` for where the browser's view opens
"""

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment


def main() -> None:
    side = 0.30  # the towel's edge [m]: what the left arm can reach, and carry back over
    cells = 30  # quads along an edge: a 10 mm cell, 961 particles on each face
    middle = (-0.05, 0.06)  # where the towel lies, in the bench's work-surface frame [m]
    grip = (0.07, 0.012)  # the patch in the hand: along the edge, and in from it [m]
    # The left arm's joints [rad] with its tool centre 25 mm over the middle of the
    # far edge, pointing down, jaws along the edge, fingertips just clear of the
    # towel. Solved once with ``sim.arm("left").solve_q_full``; move the towel, solve again.
    hold = (-0.301, 1.989, 1.329, -0.910, 0.0, -0.301)

    terry = rss.replace(
        rss.materials.TERRY_TOWELLING,
        core_shear_modulus=3.0e3,  # the measured 1e5 ties the faces into a plate, lifted as a flat wall
        edge_ke=1.0,  # the faces carry the drape: rounded tents and folds from the grip
        edge_kd=1.0e-2,  # bending damping, so a fast lift does not whip the towel into knots
    )
    physics = rss.replace(
        rss.GENERIC,
        substeps=10,  # 60 Hz x 10: at a coarser step the sheet tunnels through the bench
        backends=(
            ("articulated", rss.backend("mujoco", nconmax=2048, njmax=2048)),
            (
                "cloth",
                rss.backend(
                    "vbd",
                    iterations=10,
                    particle_self_contact_radius=0.0025,  # a fold is two layers pressed together
                    particle_self_contact_margin=0.004,
                    particle_rest_shape_contact_exclusion_radius=0.0075,  # 1.5 thicknesses: the faces never collide
                    particle_topological_contact_filter_threshold=1,  # keep the contacts a tight crease makes
                ),
            ),
        ),
        coupling=rss.coupler("one_way"),
        collision=rss.CollisionSpec(soft_contact_margin=0.01, full_surface_contact=True),
    )

    yam = get_embodiment("yam_bimanual")
    left, right = yam.arms
    holding = rss.replace(yam, arms=(rss.replace(left, home_q=hold), right))

    edge = middle[1] + side / 2  # the far edge, nearest the left arm

    def patch(rest):
        """The particles under the fingers, on the sheet as built (work-surface frame)."""
        return (abs(rest[:, 0] - middle[0]) <= grip[0] / 2) & (rest[:, 1] >= edge - grip[1])

    scene = rss.Scene(cell="bench_0728", physics=physics, name="fold_the_towel")
    scene.add(rss.robot(holding))
    towel = scene.add(
        rss.sheet(
            name="towel",
            size=(side, side),
            cells=(cells, cells),
            at=(middle[0] - side / 2, middle[1] - side / 2, 0.004),  # the corner; z is the slab's middle
            material=terry,
            particle_sets=(("held", patch),),
            held_by="left_link_6",  # the left wrist: the patch rides it from the start
            sim=rss.SimRequirements(domain="cloth", self_contact=True),  # a fold must not pass through itself
            color=(0.88, 0.85, 0.77),
            decor="towel",
        )
    )
    sim = scene.build(control_mode="joint_position")
    x, y, z = (float(v) for v in towel.particle_q.mean(axis=0))  # the towel's middle, in the world

    def page(studio):
        camera = studio.server.initial_camera  # where a browser's view starts, rather than outside the room
        camera.up, camera.look_at = (0.0, 0.0, 1.0), (x, y, z)
        camera.position = (x + 0.55, y, z + 0.4)  # from the bench's open side, across from the arms

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
