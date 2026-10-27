# SPDX-License-Identifier: Apache-2.0
"""Route the cable

Two arms route a cable from its anchor, past two spools and into a port, holding it by friction alone: press Run.

Board and goal are in ``board.py`` (VABAR ``cable_route/port_easy``): a 7 mm
cable welded at one end, two spools to pass on opposite sides, and a port. The
right hand makes three placements and the left plugs the end, lining it up from
the cable's own shape; nothing latches the cable.
"""

import math

import board
import numpy as np

import robosimstudio as rss
from robosimstudio_studio.routine import grip, hand, reach, wait

LOW = 0.0115
"""Where the tool centre is sent to pinch [m over the bench]. The i2rt fingertips reach
10.5 mm past it and a YAM at rest sits ~2.5 mm over its target, so the tips stop on the
bench with the whole 7 mm cable between the pads."""
CARRY, SPEED = 0.03, 0.05
"""Carry height [m] and speed [m/s]. With the tool centre 30 mm up the pinched capsule rides
17 mm over the bench; at 20 mm it slid out of the pads in a sideways drag (77 mm, measured)."""
LOADED = 0.02
"""How far a hand dragging the cable may stop from its target [m]: the cable pulls it back."""


def world(point) -> np.ndarray:
    """A point on the bench (work-surface frame) in the world, where the gizmos live [m]."""
    return board.origin() + np.asarray(point, dtype=np.float64)


def yaw_quat(yaw: float) -> np.ndarray:
    """Tool pointing down, its x axis at ``yaw`` about world z: the jaws close across that direction (xyzw)."""
    return np.array([math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0])


def _tangent(c: np.ndarray, k: int) -> np.ndarray:
    t = c[min(k + 1, len(c) - 1), :2] - c[max(k - 1, 0), :2]
    return t / np.linalg.norm(t)


def _solve(sim, arm, position, yaw, seed=None):
    """The arm's joints for a tool-down pose, and how far off they leave it (position [m] + angle [rad])."""
    ik = sim.arm(arm)
    seed = sim.state.joint_q.numpy().astype(np.float64) if seed is None else seed
    q = ik.solve_q_full(seed, world(position), yaw_quat(yaw))
    return q, sum(ik.pose_errors(q, world(position), yaw_quat(yaw)))


def _offset(sim, session, arm, k) -> np.ndarray:
    """Where capsule ``k`` sits from the tool centre [m]."""
    return board.capsules(sim)[k] - (hand(session, arm)[0] - board.origin())


def _held(sim, session, arm, k, steps, log):
    """Step a move while measuring how far the pinched capsule has slid in the fingers since they closed."""
    for _ in steps:
        log["slip"] = max(log["slip"], float(np.linalg.norm(_offset(sim, session, arm, k) - log["closed"])))
        yield


def _take(sim, session, arm, k, yaw, log):
    """Over capsule ``k``, straight down onto it, and Close."""
    p = board.capsules(sim)[k]
    yield from reach(session, arm, world((p[0], p[1], 0.06)), yaw_quat(yaw), speed=0.12)
    p = board.capsules(sim)[k]
    yield from reach(session, arm, world((p[0], p[1], LOW)), speed=0.05)
    yield from grip(session, arm, 1.0, 0.8)
    log.update(arm=arm, capsule=k, slip=0.0, closed=_offset(sim, session, arm, k))


def place(sim, session, arm, s, to, *, carry=CARRY, log=None):
    """Pinch the cable ``s`` metres from its welded end, carry that point to ``to`` (bench x, y), set it down, let go."""
    c = board.capsules(sim)
    k = round(s / (board.LENGTH / board.SEGMENTS) - 0.5)
    yaw = math.atan2(*_tangent(c, k)[::-1])
    # The jaws close across the cable either way round: take the roll this arm reaches better.
    above = np.array([c[k][0], c[k][1], 0.06])
    yaw = min((yaw, yaw + math.pi), key=lambda y: _solve(sim, arm, above, y)[1])
    entry = {"step": f"place {arm} s={s:.3f}"}
    yield from _take(sim, session, arm, k, yaw, entry)
    here = hand(session, arm)[0] - board.origin()
    for target in ((here[0], here[1], carry), (*to, carry), (*to, LOW + 0.003)):
        yield from _held(sim, session, arm, k, reach(session, arm, world(target), speed=SPEED, tolerance=LOADED), entry)
    yield from grip(session, arm, 0.0, 0.6)
    yield from reach(session, arm, world((*to, 0.07)), speed=0.08, tolerance=LOADED)
    if log is not None:
        log.append(entry)


def plug(sim, session, arm, s, *, log=None):
    """Take the cable ``s`` metres from its welded end, line its end up with the port's channel, and drop it in.

    The port's channel runs along +y from its mouth. The jaws' blades are wider than the
    channel, so the hand stays outside the port and the last 30 mm of cable go in ahead of
    it; whichever roll the hand takes the cable in, it must be able to turn that end to +y.
    """
    c = board.capsules(sim)
    k = round(s / (board.LENGTH / board.SEGMENTS) - 0.5)
    channel_x, mouth = board.PORT[0], board.PORT[1] - board.TERMINAL
    stage = np.array([channel_x, mouth - 0.044, 0.04])  # over the bench, 44 mm short of the mouth
    yaw = math.atan2(*_tangent(c, k)[::-1])
    above = np.array([c[k][0], c[k][1], 0.06])
    options = []
    for grasp, into in ((yaw, math.pi / 2), (yaw + math.pi, -math.pi / 2)):  # +s along tool +x, or along -x
        q, err = _solve(sim, arm, above, grasp)
        options.append((max(err, _solve(sim, arm, stage, into, seed=q)[1]), grasp, into))
    _, grasp, into = min(options)
    entry = {"step": f"plug {arm} s={s:.3f}"}
    yield from _take(sim, session, arm, k, grasp, entry)
    here = hand(session, arm)[0] - board.origin()
    for target, quat in (((here[0], here[1], CARRY), None), (stage, yaw_quat(into))):
        yield from _held(
            sim, session, arm, k, reach(session, arm, world(target), quat, speed=SPEED, tolerance=LOADED), entry
        )
    # Twice: yaw the hand until the cable ahead points down the channel, then slide so its line hits the mouth.
    for _ in range(2):
        c = board.capsules(sim)
        ahead = c[-1, :2] - c[k, :2]
        err = math.atan2(ahead[0], ahead[1])
        quat = np.asarray(session.arms[arm].cursor_quat, dtype=np.float64)
        turned = _qmul((0.0, 0.0, math.sin(err / 2), math.cos(err / 2)), quat)
        here = hand(session, arm)[0] - board.origin()
        yield from _held(
            sim,
            session,
            arm,
            k,
            reach(session, arm, world(here), turned, speed=SPEED, turn=0.5, tolerance=LOADED),
            entry,
        )
        c = board.capsules(sim)
        ahead = c[-1, :2] - c[k, :2]
        meets = c[k, 0] + ahead[0] / max(ahead[1], 1e-6) * (mouth - c[k, 1])
        here = hand(session, arm)[0] - board.origin()
        aim = (here[0] + channel_x - meets, stage[1], stage[2])
        yield from _held(sim, session, arm, k, reach(session, arm, world(aim), speed=0.03, tolerance=LOADED), entry)
    entry["aim"] = float(meets - channel_x)
    # Over the channel with the end, then down: the end drops between the walls.
    here = hand(session, arm)[0] - board.origin()
    for target in ((here[0], mouth - 0.014, stage[2]), (here[0], mouth - 0.014, LOW + 0.003)):
        yield from _held(sim, session, arm, k, reach(session, arm, world(target), speed=0.025, tolerance=LOADED), entry)
    yield from grip(session, arm, 0.0, 0.6)
    yield from reach(session, arm, world((here[0], mouth - 0.034, 0.07)), speed=0.08, tolerance=LOADED)
    if log is not None:
        log.append(entry)


def _qmul(a, b) -> np.ndarray:
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def routine(sim, session, log=None):
    """The job, from the robot's home pose to home again: three placements and the plug.

    ``log`` collects one entry per grasp: which capsule, and how far it slid in the
    fingers between Close and Open [m].
    """
    (x0, y0), (x1, y1) = board.PEGS
    yield from place(sim, session, "right", 0.165, (x0 - 0.025, y0 + 0.005), log=log)  # beside the first spool
    yield from place(sim, session, "right", 0.28, (x0 + 0.003, (y0 + y1) / 2), log=log)  # across the line
    # Round the far side of the second, lifted clear of its 23.5 mm rim on the way over.
    yield from place(sim, session, "right", 0.44, (x1 + 0.040, y1 + 0.028), carry=0.065, log=log)
    yield from reach(session, "right", *session.arms["right"].ik.home_pose, speed=0.15, tolerance=LOADED)
    yield from plug(sim, session, "left", 0.565, log=log)
    yield from reach(session, "left", *session.arms["left"].ik.home_pose, speed=0.15, tolerance=LOADED)
    yield from wait(session, 2.0)


def main():
    # Under the default contact law, which has no static friction, a pinched capsule crept
    # through the pads at 15 mm/s. Anchored contacts with their multipliers carried over
    # hold it: 1-9 mm of slip over each of the four carries, in 16 runs.
    physics = rss.replace(
        rss.GENERIC.with_backend(
            "*", rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)
        ),
        collision=rss.CollisionSpec(contact_matching="sticky"),
    )
    # A cable dragged across the jaws pries them open. At the i2rt fingers' own 600 N/m it
    # slid out of 3 drags of 7; at 2000 N/m the carry round the second spool still pried
    # them 5 mm apart and lost the cable in 3 runs of 26. At 6000 N/m: 16 of 16, 1-9 mm.
    fingers = rss.ActuationSpec(target_ke=6000.0, target_kd=120.0)

    scene = rss.Scene(cell=board.CELL, physics=physics, name="route_the_cable")
    scene.add(rss.robot("yam_i2rt", gripper=fingers))
    for part in board.scenery():
        scene.add(part)
    scene.add(board.cable())
    sim = scene.build(control_mode="joint_position")

    def page(studio):
        studio.aim_camera((0.95, -0.10, 1.10), (0.53, -0.12, 0.76))  # from the bench's open side

    rss.view(sim, page=page, routines={"Route the cable": routine})


if __name__ == "__main__":
    main()
