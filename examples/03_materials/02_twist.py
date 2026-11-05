# SPDX-License-Identifier: Apache-2.0
"""Twist: turn two cables

Two cables hang from a clamp, one twisting ten times more easily than it bends and one ten times less: drag each hand onto its cable's tip, press **Close**, turn each gizmo's ring a third of a turn about the cable and press **Open**, and the stiff one springs the turn back out while the soft one keeps it.

``twist_ke`` splits Newton's rod joint from its bend. The pads alone hold each
tip, turned about the vertical. Turned 120 degrees, the soft cable takes 0.32
turns and the stiff 0.32 too, held by the hand's 50 N clamp; opened, the stiff
reads 0.001 in 30 steps, the soft 0.106 in 60 (``tests/test_cable_robot``, GPU 7).
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.rod.measures import twist_turns


def main():
    length, radius, segments, bend_ke = 0.4, 0.006, 24, 0.04  # [m], and the shipped cable's bend [N*m/rad]
    top, x, y = 0.55, 0.05, 0.18  # the clamps [m]: tips 150 mm over the bench, 360 mm apart, so the arms clear
    right_q = (0.8756, 1.7574, 0.7021, 1.0552, 1.3992, 0.0)  # [rad]: level, 30 degrees out, 60 mm back from the tip
    left_q = (-0.8756, 1.7574, 0.7021, 1.0552, -1.3992, 0.0)  # its mirror; both by ``sim.arm(...).solve_q_full``
    vbd = rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)
    physics = rss.replace(rss.GENERIC.with_backend("*", vbd), collision=rss.CollisionSpec(contact_matching="sticky"))
    yam = get_embodiment("yam_bimanual")
    yam = rss.replace(yam, arms=(rss.replace(yam.arms[0], home_q=left_q), rss.replace(yam.arms[1], home_q=right_q)))
    scene = rss.Scene(cell="bench_075", physics=physics, name="cable_twist")
    scene.add(rss.robot(yam))
    cables = {
        name: scene.add(
            rss.rod(
                name=name,
                length=length,
                radius=radius,
                segments=segments,
                at=(x, side, top),
                along=(0.0, 0.0, -1.0),
                anchors=("start",),  # clamped: position, tangent and twist held at the top
                bend_ke=bend_ke,
                twist_ke=twist_ke,
                allow_grasp_latch=False,  # nothing holds a tip but the pads
                scene_friction=True,  # the pads at mu 60
            )
        )
        for name, side, twist_ke in (("soft_twist", y, bend_ke / 10), ("stiff_twist", -y, bend_ke * 10))
    }
    sim = scene.build(control_mode="joint_position", device="cuda")  # VBD: a CPU's rope has no twist joint

    def page(studio):
        roll = studio.gui.add_slider("tip roll (turns)", min=0.0, max=2.0, step=0.05, initial_value=0.0)
        held = studio.gui.add_checkbox("hold the tips", initial_value=False)
        out = {n: studio.gui.add_number(f"{n}: mid-cable twist (turns)", 0.0, step=1e-3, disabled=True) for n in cables}

        def drive() -> None:
            body_q, middle = np.asarray(sim.pipeline.snapshot.body_q), slice(segments // 3, 2 * segments // 3 + 1)
            for name, cable in cables.items():
                if held.value:  # the page's kinematic hold, not a hand: roll the tip towards the slider
                    cable.entity.roll(sim.pipeline, 2 * np.pi * roll.value)
                else:  # the tips are the hands' (a release of a free tip does nothing)
                    cable.entity.pin(sim.pipeline, on=False)
                out[name].value = round(twist_turns(body_q[cable.bodies[middle], 3:7]), 3)  # the middle third

        studio.each_step(drive)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
