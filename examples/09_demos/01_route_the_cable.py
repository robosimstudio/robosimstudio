# SPDX-License-Identifier: Apache-2.0
"""Route the cable

A Panda holds the end of a cable over three clips: drag the wrist gizmo to lift it, carry it along and lay it in.

The cable is a Cosserat rod that SuperDex steps beside the Newton arm, clamped
in its socket at one end and pinned to the tool centre at the other. The bench,
the clips' prongs and the fingers are mirrored into SuperDex as colliders and
padded so the cable lies on them rather than half inside them. Lift the end
until the cable hangs off the bench, carry it past the far clip, then lower it
straight down onto the line: the horns funnel it into the channels. Needs the
``[superdex]`` extra (Python 3.12).

**Key features:**

* :class:`robosimstudio.RodSpec` with ``nodes`` and ``anchors`` for a cable laid along a path and clamped in its socket
* :func:`robosimstudio.backend` ``("superdex", mirror_padding=...)`` for contact on the cable's surface, not its axis
* :class:`robosimstudio.StaticSpec` capsules for the clips' prongs
* :func:`robosimstudio.view` with ``page=`` to put the cable's end in the hand
"""

import math

import numpy as np

import robosimstudio as rss

RADIUS, SEGMENTS = 0.0025, 45  # a 5 mm cable, 45 cm long in 1 cm elements
LINE_X, CLIP_Y = -0.06, (-0.10, -0.01, 0.08)  # the clips stand on the line x = LINE_X, in the bench's frame
CHANNEL, PRONG_R, POST_HALF, HORN_HALF, HORN_LEAN = 0.0084, 0.003, 0.006, 0.008, math.radians(26.0)
SOCKET, HEADING = (LINE_X, -0.21, RADIUS), math.radians(12.0)  # the socket faces 12 degrees off the line
HAND = (-0.143, -0.115, 0.231)  # where the Panda's tool centre starts
WHITE, BLUE, NAVY = (0.93, 0.93, 0.91), (0.13, 0.42, 0.85), (0.10, 0.30, 0.62)


def clip(number, y):
    """A base plate and two Y-shaped prongs either side of the line: an upright post, a horn leaning out."""
    plate = rss.ShapeSpec(kind="box", half_extents=(0.014, 0.008, 0.0006))
    parts = [
        rss.StaticSpec(
            name=f"clip{number}", shape=plate, xform=(LINE_X, y, 0.0002, 0, 0, 0, 1), collide=False, color=NAVY
        )
    ]
    for side in (1.0, -1.0):
        x, knee = LINE_X + side * (CHANNEL / 2 + PRONG_R), PRONG_R + 2 * POST_HALF
        horn = (x + side * HORN_HALF * math.sin(HORN_LEAN), y, knee + HORN_HALF * math.cos(HORN_LEAN))
        lean = (0.0, math.sin(side * HORN_LEAN / 2), 0.0, math.cos(side * HORN_LEAN / 2))
        post = (x, y, knee - POST_HALF)
        for part, centre, half, quat in (("post", post, POST_HALF, (0, 0, 0, 1)), ("horn", horn, HORN_HALF, lean)):
            capsule = rss.ShapeSpec(kind="capsule", radius=PRONG_R, half_height=half)
            name = f"clip{number}_{part}_{'ab'[side < 0]}"
            parts.append(rss.StaticSpec(name=name, shape=capsule, xform=(*centre, *quat), friction=0.25, color=BLUE))
    return parts


def socket():
    """A box the cable leaves along the heading, its face 3.5 mm behind the cable's clamped end."""
    centre = np.subtract(SOCKET, 0.0195 * np.array([math.sin(HEADING), math.cos(HEADING), 0.0]))
    yaw = (math.pi / 2 - HEADING) / 2
    box = rss.ShapeSpec(kind="box", half_extents=(0.016, 0.012, 0.009))
    xform = (centre[0], centre[1], 0.009, 0.0, 0.0, math.sin(yaw), math.cos(yaw))
    return rss.StaticSpec(name="socket", shape=box, xform=xform, friction=0.5, color=(0.16, 0.16, 0.18))


def cable_nodes():
    """Out of the socket along the bench for 18 cm, then up into the hand, in equal legs."""
    path = np.array([SOCKET, np.add(SOCKET, 0.18 * np.array([math.sin(HEADING), math.cos(HEADING), 0.0])), HAND])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    at = np.linspace(0.0, arc[-1], SEGMENTS + 1)
    return tuple(zip(*(np.interp(at, arc, path[:, k]).tolist() for k in range(3)), strict=True))


def in_hand(sim, cable):
    """The page: the hand takes the cable's end on the first step, and again after every Reset, which lets go."""
    end = len(cable.particles) - 1

    def page(studio):
        studio.server.initial_camera.position = (1.02, 0.03, 1.12)  # from the bench's open side
        studio.server.initial_camera.look_at = (0.55, 0.03, 0.84)

        def take_the_end():
            if sim.step_count == 0:
                cable.entity.hold(True, sim.state, at=end)  # the end, at the tool centre
                cable.entity.hold(True, sim.state, at=end - 1, slot=1)  # and its neighbour: a clamp, not a hinge

        studio.each_step(take_the_end)

    return page


def main():
    physics = rss.ArrangementSpec(
        control_hz=60.0,
        substeps=4,
        backends=(
            ("*", rss.backend("mujoco")),
            # The engine stands a collider off by 1 mm, so 1.5 mm more is one radius: the cable's skin on the mirror.
            ("rod", rss.backend("superdex", substeps=2, mirror_padding=RADIUS - 0.001)),
        ),
    )
    scene = rss.Scene(cell="bench_075", physics=physics, name="route_the_cable")
    scene.add(rss.robot("panda"))
    for number, y in enumerate(CLIP_Y, start=1):
        for part in clip(number, y):
            scene.add(part)
    scene.add(socket())
    cable = scene.add(
        rss.RodSpec(
            name="cable",
            nodes=cable_nodes(),
            radius=RADIUS,
            density=1500.0,  # copper in a PVC jacket: 29 g/m
            young_modulus=1.5e7,  # EI = 4.6e-4 N m^2: it springs back from a bend, as a cable does and a rope does not
            mu_cable=0.0625,  # 0.25 on the engine's side: it grips the bench and slides down the prongs
            anchors=("start",),
            stress_free="straight",
            skin=False,  # contact on the axis, lifted to the surface by the padding: a third of a skin's cost
            color=WHITE,
        )
    )
    sim = scene.build(control_mode="joint_position")

    rss.view(sim, page=in_hand(sim, cable))


if __name__ == "__main__":
    main()
