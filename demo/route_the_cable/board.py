# SPDX-License-Identifier: Apache-2.0
"""The board the cable is routed on, and what counts as routed.

VABAR's easy cable-routing tier (``cable_route/port_easy``), at the reference's
measured numbers: a 600 mm cable of 50 capsules, 7 mm across, welded under an
anchor bracket at one end; two ManipulationNet F1 spools 240 mm apart on the
line x = -75 mm; a port 100 mm past the second. The cable must pass the first
spool on the robot's side and the second on the far side, and end in the port.
Positions are in the bench's work-surface frame, converted from the
reference's world coordinates (bench_075, whose surface centre is world
``(0.6, 0, 0.75)``).

Two things differ from the reference, and both are written where they are:
the spools are drawn from three cylinders at the F1's measured radii rather
than imported from its mesh, and the port faces the run (its channel along y)
rather than the robot.
"""

from __future__ import annotations

import math

import numpy as np

import robosimstudio as rss

CELL = "bench_075"
ORANGE, METAL = (0.85, 0.35, 0.12), (0.38, 0.40, 0.43)
"""The fixtures' colour (the library's ``ACCENT_PROP``) and the anchor's rails."""

LENGTH, RADIUS, SEGMENTS = 0.60, 0.0035, 50
"""The reference's terminated-tier rod: 600 mm, 7 mm across, 84 capsules a metre rounded to even."""
START = (-0.18, -0.36)
"""The cable's welded end: the reference's spawn line, 100 mm nearer the robot than the spools."""
PEGS = ((-0.075, -0.26), (-0.075, -0.02))
"""The two spools: the outer stations of a 120 mm-pitch line, 8 mm toward the robot."""
SIDES = (-1, +1)
"""Which side of each spool the cable must pass, in x: -1 is the robot's side."""
ANCHOR = (START[0], START[1] + 0.008, math.pi / 2)
"""The bracket over the welded end, its centre 8 mm past the tip so the end is under its hood."""
PORT = (-0.014, 0.08, -math.pi / 2)
"""The port: 61 mm out from the second spool and 100 mm past it, the reference's place.

Turned to face the run, where the reference faces it at the robot. A YAM wrist rolls
120 degrees either way, and an end held across the jaws has to be turned 90 degrees
to go into a channel square to the run: from the grasps this board leaves, the roll
ran out (measured, both arms, both rolls)."""

SHAFT_R, FLANGE_R, GROOVE, HEIGHT = 0.00475, 0.0125, (0.0025, 0.019679), 0.0235
"""ManipulationNet's F1, measured off its mesh: a 4.75 mm shaft between 12.5 mm flanges, 23.5 mm tall."""
TERMINAL = 0.016
"""Half the terminals' 32 x 32 mm footprint; a channel 10 mm across between 11 mm rails."""


def spool(name: str, xy) -> list:
    """The F1 as three cylinders: a foot flange, the shaft the cable seats against, a rim."""
    x, y = xy
    parts = (
        ("foot", FLANGE_R, 0.0, GROOVE[0]),
        ("shaft", SHAFT_R, GROOVE[0], GROOVE[1]),
        ("rim", FLANGE_R, GROOVE[1], HEIGHT),
    )
    return [
        rss.post(name=f"{name}_{part}", radius=r, half_height=(z1 - z0) / 2, at=(x, y, (z0 + z1) / 2), color=ORANGE)
        for part, r, z0, z1 in parts
    ]


def _box(name: str, frame, centre, half, color):
    """A box given in a terminal's own frame (local +x is the way the run leaves)."""
    x, y, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    at = (x + c * centre[0] - s * centre[1], y + s * centre[0] + c * centre[1], centre[2])
    return rss.StaticSpec(
        name=name,
        shape=rss.ShapeSpec(kind="box", half_extents=half),
        xform=(*at, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
        color=color,
    )


def anchor() -> list:
    """Two rails either side of the welded end, a hood over it and a saddle across it: nothing under the cable."""
    rail = (TERMINAL, 0.0055, 0.0025)
    hood = 0.5 * (0.0125 - 0.008)  # the hood's underside 8 mm up, just over the cable's 7 mm crown
    return [
        _box("anchor_rail_a", ANCHOR, (0.0, 0.0105, 0.0025), rail, METAL),
        _box("anchor_rail_b", ANCHOR, (0.0, -0.0105, 0.0025), rail, METAL),
        _box("anchor_hood", ANCHOR, (-0.008, 0.0, 0.008 + hood), (0.008, TERMINAL, hood), ORANGE),
        _box("anchor_saddle", ANCHOR, (0.008, 0.0, 0.008 + hood), (0.0032, TERMINAL, hood), ORANGE),
    ]


def port() -> list:
    """Two walls 12.5 mm tall either side of a 10 mm channel, open on top, and a stop at the back."""
    wall = (TERMINAL, 0.0055, 0.00625)
    return [
        _box("port_wall_a", PORT, (0.0, 0.0105, 0.00625), wall, ORANGE),
        _box("port_wall_b", PORT, (0.0, -0.0105, 0.00625), wall, ORANGE),
        _box("port_back", PORT, (-0.0128, 0.0, 0.00625), (0.0032, 0.005, 0.00625), ORANGE),
    ]


def cable() -> rss.RodSpec:
    """The rod, lying straight along +y from its welded end, held by nothing but what touches it."""
    # The reference rod's bend (0.04 N m at 11 mm and a 32.7 mm pitch) at the same material:
    # EI goes as r^4 and a joint spans one pitch, so (3.5/11)^4 * (32.7/12) of it.
    scale = (RADIUS / 0.011) ** 4 * ((0.9144 / 28) / (LENGTH / SEGMENTS))
    return rss.RodSpec(
        name="cable",
        start=(START[0], START[1], RADIUS),
        direction=(0.0, 1.0, 0.0),
        length=LENGTH,
        radius=RADIUS,
        segments=SEGMENTS,
        bend_ke=0.04 * scale,
        bend_kd=0.06 * scale,
        anchors=("start",),
        allow_grasp_latch=False,  # no hand may latch it: what holds it is the pads' friction
        filter_fingertip_scenery=False,  # the fingertips touch the bench, the spools and the port
        scene_friction=True,  # pads at the rod's mu_gripper (60), the bench at mu_surface (0.02)
    )


def scenery() -> list:
    return [*spool("peg0", PEGS[0]), *spool("peg1", PEGS[1]), *anchor(), *port()]


# ----------------------------------------------------------------------------- routed?
SEAT = (0.003, 0.045, 0.035)
"""A seat box: from 3 to 45 mm off the spool's axis on its side, 35 mm either way along the line."""
PORT_BOX = 0.045
"""The port's box: 45 mm every way from its centre."""
HEIGHT_BOX = 0.15
"""Every box runs from the bench to 150 mm over it, the reference documents' default."""


def origin() -> np.ndarray:
    """The work-surface frame's origin in the world [m]."""
    cell = rss.get_cell(CELL)
    return np.array([cell.center[0], cell.center[1], cell.surface_z])


def capsules(sim) -> np.ndarray:
    """The cable's capsule centres now, in the work-surface frame, from the welded end."""
    bodies = np.asarray(sim["cable"].bodies)
    return sim.state.body_q.numpy()[bodies, :3].astype(np.float64) - origin()


def _share(points: np.ndarray, lo, hi) -> float:
    inside = np.all((points >= np.asarray(lo)) & (points <= np.asarray(hi)), axis=1)
    return float(inside.mean())


def routed(sim) -> dict:
    """The reference goal's terms: the share of capsules in each seat and in the port, and the end in the port.

    Passed at ``seat0, seat1 >= 0.04`` (two capsules of fifty), ``port >= 0.06``
    (three) and ``end`` 1.0 -- the reference's thresholds -- with the hands
    gone and the cable at rest, which :func:`done` adds.
    """
    c = capsules(sim)
    out = {}
    for i, ((x, y), side) in enumerate(zip(PEGS, SIDES, strict=True)):
        near, far = sorted((x + side * SEAT[0], x + side * SEAT[1]))
        out[f"seat{i}"] = _share(c, (near, y - SEAT[2], 0.0), (far, y + SEAT[2], HEIGHT_BOX))
    lo = (PORT[0] - PORT_BOX, PORT[1] - PORT_BOX, 0.0)
    hi = (PORT[0] + PORT_BOX, PORT[1] + PORT_BOX, HEIGHT_BOX)
    out["port"] = _share(c, lo, hi)
    # The free end is the far cap of the last capsule, half a pitch past its centre.
    end = c[-1] + 0.5 * (c[-1] - c[-2])
    out["end"] = _share(end[None, :], lo, hi)
    return out


def done(sim, hands) -> dict:
    """:func:`routed`, the hands' distance from the cable [m] and its fastest capsule [m/s]: the whole goal."""
    out = routed(sim)
    c = capsules(sim) + origin()
    out["hands"] = float(min(np.linalg.norm(c - np.asarray(h), axis=1).min() for h in hands))
    bodies = np.asarray(sim["cable"].bodies)
    out["speed"] = float(np.linalg.norm(sim.state.body_qd.numpy()[bodies, :3], axis=1).max())
    out["routed"] = (
        (out["seat0"] >= 0.04 and out["seat1"] >= 0.04 and out["port"] >= 0.06 and out["end"] >= 1.0)
        and out["hands"] > 0.05
        and out["speed"] <= 0.05
    )
    return out
