# SPDX-License-Identifier: Apache-2.0
"""Fold the towel

Two YAM arms fold a terry towel in half from their home pose, holding it by friction alone: press Run.

The towel, bench and physics are the cloth_fold suite's ``towel_half`` task. The
right hand presses the near half down; the left pinches the far edge on the bench
and carries it over. Nothing is pinned: a one-way coupler with full-surface
contact lets the pads hold the edge by friction.
"""

import math

import numpy as np

import robosimstudio as rss
from robosimstudio_materials.cloth import selector_beyond, selector_extreme
from robosimstudio_studio.routine import RoutineError, grip, hand, reach, wait

# -- the towel: the cloth_fold suite's, in the bench's work-surface frame ----------------------
CELL = "bench_0728"
SIDE = 0.30
"""The towel's edge [m]: the band the left arm can both pick from and put down in."""
CELLS = 30
"""Quads along an edge: a 10 mm cell, 961 particles on each face."""
CENTER = (-0.05, 0.06)
"""Where the towel lies [m]: on the left arm's reach band, y in [-0.09, +0.21]."""
CREASE_Y = CENTER[1]
"""The fold line: the towel's own centreline across y."""

# -- the routine: every number measured on this scene, the others held at their values here ----
UP = np.array([0.0, 0.0, 1.0])
DOWN = (0.0, 1.0, 0.0, 0.0)
"""The tool pointing down with the jaws closing along y, across the far edge (xyzw)."""
PINCH_X = -0.08
"""Where along the far edge the left hand takes it [m]. At the middle (-0.05) the corner
nearer the robot folds back on itself and the fold scores 0.82; at -0.07, 0.85. At -0.09 it
scores 0.87-0.90 but that corner can overshoot the near edge by 19 mm, a span of 0.61."""
EDGE_IN = 0.006
"""Tool centre in from the far edge [m]: the outer pad lands on bare bench, the inner on the towel."""
ON_BENCH = 0.0155
"""Tool centre over the bench [m] that puts the fingertips on it: the pads end 16.7 mm below
the tool centre, so the bench stops them about a millimetre short of this."""
SQUEEZE = 0.95
"""The grip [0-1]: pads 3.5 mm apart. Shut (1.0) they close through each other and the
towel, -1.3 mm, and carry the edge away with them when they open; at 0.90 (8.4 mm) the
edge slips out on the lift."""
SQUEEZE_RISE = 0.003
"""How far the hand rises while it squeezes [m]. On the bench, the pads' friction with it
holds them 42-45 mm open after a second; in free air 1.5 mm over it, they pinched 5-7
particles and lifted nothing in 3 of 4 runs. Down, then up while squeezing, pinches 31-37."""
APEX = 0.06
"""How high the carry arcs over the place [m]: the suite's low, page-turn arc."""
CARRY = 5.0
"""Seconds over the arc. At 4 s the fold scored 0.85-0.89, at 5 s 0.87-0.92."""
SLACK = 0.0125
"""How far short of its mirror the edge is laid [m]: 2.5 thicknesses, the suite's, because a
5 mm slab's crease loop takes up cloth."""
PLACE_Z = 0.03
"""Tool centre over the bench at the place [m]: the flap's own thickness over the near half."""
PRESS = (0.0, -0.04)
"""Where the right hand presses the near half [m]. Without it the near half was dragged
27-35 mm and the fold scored 0.18-0.45. At (-0.13, -0.015), the suite's spot on the other side
of the carry, the flap's near corner caught on it: 0.79."""
PRESS_UNTIL = 0.9
"""The fraction of the carry the press stays down for. Left down through the landing the flap
lands on the fingers; lifted at 0.4-0.6 the near half drifts 13-14 mm. Over three runs each,
lifted at 0.8 the fold scored 0.844-0.878, and at 0.9, 0.851-0.885."""


def together(*moves):
    """Advance several routines one control step at a time, until every one is done."""
    live = list(moves)
    while live:
        for move in list(live):
            try:
                next(move)
            except StopIteration:
                live.remove(move)
        yield


def glide(session, arm, path, seconds, quat, *, tolerance=0.006):
    """Move the arm's gizmo along ``path(f)``, f from 0 to 1, eased; then wait for the hand.

    :func:`~robosimstudio_studio.routine.reach` goes in a straight line, and a fold is an arc.
    """
    panel = session.arms[arm]
    hz = float(session.env.request.physics.control_hz)
    steps = max(1, math.ceil(seconds * hz))
    for k in range(1, steps + 1):
        s = k / steps
        panel.aim(path(s * s * (3.0 - 2.0 * s)), quat, moving=True)
        yield
    end = path(1.0)
    panel.aim(end, quat, moving=False)
    for _ in range(round(2.0 * hz)):
        if float(np.linalg.norm(hand(session, arm)[0] - end)) < tolerance:
            return
        yield
    raise RoutineError(f"the {arm} hand stopped short of the end of its arc")


def routine(sim, session):
    """From the home pose to a towel folded in half and both hands home again."""
    origin = np.array(rss.get_cell(CELL).surface_frame[:3])
    down = np.array(DOWN)
    edge = float(sim["towel"].particle_q[:, 1].max())
    grasp = origin + np.array([PINCH_X, edge - EDGE_IN, ON_BENCH])
    press = origin + np.array([*PRESS, ON_BENCH])

    def press_down():
        yield from grip(session, "right", 1.0, 0.1)  # shut, so the closed jaws press flat
        yield from reach(session, "right", press + UP * 0.08, down, speed=0.25)
        yield from reach(session, "right", press, tolerance=0.008)

    # 1. Both hands to their marks, the left fingertips on the bench at the far edge.
    yield from together(reach(session, "left", grasp + UP * 0.10, down, speed=0.25), press_down())
    yield from reach(session, "left", grasp, tolerance=0.008)

    # 2. Squeeze and rise.
    left = session.arms["left"]
    left.grip_request = SQUEEZE
    yield from wait(session, 0.2)
    start = np.array(left.cursor_pos)
    yield from glide(session, "left", lambda f: start + UP * SQUEEZE_RISE * f, 0.3, down, tolerance=0.008)
    yield from wait(session, 1.0)

    # 3. Carry over the crease in a low arc, short of the edge's mirror; the press lifts before it lands.
    held = hand(session, "left")[0]
    place = np.array([held[0], 2.0 * (origin[1] + CREASE_Y) - held[1] + SLACK, origin[2] + PLACE_Z])
    apex = (held + place) / 2.0
    apex[2] = max(held[2], place[2]) + APEX
    bend = 2.0 * apex - (held + place) / 2.0  # the Bezier control point that puts the arc's top at apex

    def arc(f):
        return (1 - f) ** 2 * held + 2 * f * (1 - f) * bend + f**2 * place

    def lift_press():
        yield from wait(session, PRESS_UNTIL * CARRY)
        yield from reach(
            session, "right", hand(session, "right")[0] + np.array([0.0, -0.10, 0.10]), speed=0.3, tolerance=0.02
        )

    yield from together(glide(session, "left", arc, CARRY, down), lift_press())

    # 4. Let go and go home.
    yield from grip(session, "left", 0.0, 1.0)
    yield from reach(session, "left", place + np.array([0.0, 0.08, 0.10]), speed=0.1, tolerance=0.02)
    yield from together(
        *(reach(session, arm, *session.arms[arm].ik.home_pose, speed=0.25) for arm in ("left", "right"))
    )
    yield from wait(session, 5.0)  # the suite calls a fold done once it has lain still for five seconds
    print(fold_report(sim), flush=True)


def fold_measures(sim) -> dict:
    """The cloth_fold suite's verdict on this towel now: extent, score and settle speed."""
    from robosimstudio.assetlib.vocabulary.measures import cloth_fold, cloth_fold_error, mirror_pairs
    from robosimstudio_materials.cloth import contact_thickness

    entity = sim["towel"].entity
    probe = entity.probe
    probe.update(sim.state)
    ctx = type("Ctx", (), {"probe": staticmethod(lambda name: probe)})
    crease = float(rss.get_cell(CELL).surface_frame[1] + CREASE_Y)
    pairs = mirror_pairs(probe, "towel/moving", "towel/base", axis=1, origin=crease)
    score, group, fixation = cloth_fold_error(
        ctx, "towel", pairs, layer_thickness=contact_thickness(entity.spec.material)
    )
    ratio, _ = cloth_fold(ctx, "towel", axis=1)
    return {
        "fold_extent": ratio,
        "fold_score": score,
        "group_dist": group,
        "fixation_dist": fixation,
        "settle_speed": float(probe.settle_speed("towel")),
    }


def fold_report(sim) -> str:
    m = fold_measures(sim)
    return (
        f"[fold] extent {m['fold_extent']:.3f} (<= 0.6), score {m['fold_score']:.3f} (>= 0.85), "
        f"near half moved {1000 * m['fixation_dist']:.1f} mm, settling at {m['settle_speed']:.4f} m/s"
    )


def build():
    """The bench, the arms at home with their hands open, and the towel lying flat."""
    terry = rss.replace(
        rss.materials.TERRY_TOWELLING,
        # The measured core (1e5 Pa) lifts as a plate: the pinch lost 44% of the edge by the
        # top of the arc and all of it by the place, and the fold scored 0.52.
        core_shear_modulus=3.0e3,
        edge_ke=1.0,  # the faces carry the drape
        edge_kd=1.0e-2,  # so a carried towel does not whip itself into knots
    )
    physics = rss.replace(
        rss.GENERIC,
        control_hz=60.0,
        substeps=10,  # a 1/600 s step: at 1/240 s the sheet tunnels through the bench
        backends=(
            ("articulated", rss.backend("mujoco", nconmax=2048, njmax=2048)),
            (
                "cloth",
                rss.backend(
                    "vbd",
                    iterations=10,
                    particle_self_contact_radius=0.002,
                    particle_self_contact_margin=0.003,
                    particle_topological_contact_filter_threshold=1,  # keep the contacts a tight crease makes
                    particle_rest_shape_contact_exclusion_radius=0.0075,  # 1.5 thicknesses: the faces never collide
                    particle_vertex_contact_buffer_size=32,
                    particle_edge_contact_buffer_size=64,
                ),
            ),
        ),
        coupling=rss.coupler("one_way"),
        collision=rss.CollisionSpec(soft_contact_margin=0.010, full_surface_contact=True),
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="fold_the_towel")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(
        rss.sheet(
            name="towel",
            size=(SIDE, SIDE),
            cells=(CELLS, CELLS),
            at=(CENTER[0] - SIDE / 2, CENTER[1] - SIDE / 2, 0.001),  # the corner, 1 mm over the bench
            material=terry,
            particle_sets=(
                ("moving", selector_beyond(1, CREASE_Y, above=True)),  # the far half, carried over
                ("base", selector_beyond(1, CREASE_Y, above=False)),  # the near half, which must stay
                ("grasp", selector_extreme(1, band=0.025, high=True)),  # the far edge strip
            ),
            sim=rss.SimRequirements(domain="cloth", self_contact=True),  # a fold must not pass through itself
            color=(0.88, 0.85, 0.77),
            decor="towel",
        )
    )
    return scene.build(control_mode="joint_position")


def main() -> None:
    sim = build()
    x, y, z = (float(v) for v in sim["towel"].particle_q.mean(axis=0))  # the towel's middle, in the world

    def page(studio):
        studio.aim_camera((x + 0.45, y, z + 0.35), (x, y, z))  # from the bench's open side, across from the arms

    rss.view(sim, page=page, routines={"Fold the towel": routine})


if __name__ == "__main__":
    main()
