# SPDX-License-Identifier: Apache-2.0
"""Hang the labelled Clawhauser volume on a horizontal rod by its loop.

The toy stands on the bench by its tail, loop up and the hole's axis horizontal
along the rod. A two-finger gripper pinches the loop's outer sides at the hole's
own centre height -- narrow pads, no ledge, the hole left free -- lifts the toy
until the rod's free end is inside the hole, slides it through, sets the toy down
on the rod and lets go. The default mover is the AIRBOT Play arm with its own
parallel hand (`arm_gripper`); `--claw claw` runs the scripted claw this scene
was built on, which stays as the reference the arm is measured against. What holds it then is the rod under the loop's inner surface:
one rigid-soft contact patch carrying 1.227 N of toy.

The arm's 40 mm pads squeeze the ring's upper half shut once the toy hangs, so
the open part of the hole is below them. After the hang the arm measures that
part, aligns it with the rod, slides it on, lowers the pads to just over the rod
and lets go.

That height is not a convenience. It is the loop's widest point, and the only
place on a closed loop where the outer surface faces the closing direction: a
pad pushed at the arch above it meets a nearly horizontal surface, so the contact
normals point up, the pads ride over the cord and the grasp is gone before the
jaws lift -- measured, in the first build, as 224 finger contacts through the
close then zero from the first lifted step with the toy's minimum z unmoved. And
the pads aim a fixed interference past the measured width, so the squeeze is the
drive's own clamp rather than whatever closing gap the loop's compliance leaves.

Two things this scene is built to show. The rod and its post are rigid and never
move, so nothing is scripted once the jaws open. And every jaw number is bounded:
the claw's two padded fingers slide under a 20 N budget; the arm's hand is an
actuator whose force range is the same 20 N, so its ``peak_force_n`` is the force
the pads could feel rather than a command. ``peak_torque_nm`` is the claw's, and
absent on the arm.

Geometry is derived from the mesh rather than hard-coded: the loop's hole centre
and inner radius by an inscribed-circle fixed point over the loop's vertices, the
loop's width in the closing direction from that same plane, and the rod's height
from the hanging body's own reach. The printed ``geometry`` line is what the
motion was built on.

Controls, for contrast rather than for the result:

    --coarse           run the common-translation solve under load

Knobs, each reported with its measured effect:

    --rod-radius   radius of the rod the loop is threaded onto [m]
    --mu           the pipeline's single global soft-contact friction coefficient

Run on a CUDA node with Newton 1.8 development and RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
from clawhauser_scene import (
    COARSE_BUDGET,
    CONTACT_GAP,
    SELF_CONTACT_RADIUS,
    add_scene_arguments,
    arrangement,
    check_build,
    load_toy,
    scene_kwargs,
    toy_spec,
)
from clawhauser_scene import REFERENCE as PHYSICS
from loop_geometry import ring_geometry, smooth
from mesh_quality import MeshQuality, penetration
from parallel_gripper import ParallelGripperSpec

import robosimstudio as rss

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "clawhauser_on_rod"
CONTROL_HZ = PHYSICS.control_hz
SUBSTEPS = PHYSICS.substeps
DURATION = 10.0
"""Long enough for the guarded seat to stop early or late and still leave the last
half second still, so that the window the hang is judged on is a quiet one.
"""
SEAT_TRIGGER = 0.95
"""Fraction of the toy's weight the rod must carry before the descent stops.

Nearly all of it, not half: half is reached by a loop lying across the rod's
free end with the pinch still carrying the rest, and releasing there drops the
toy -- measured at t = 5.33 s of the run that first stopped at half. The descent
keeps going until the pinch is empty, which is the state in which opening the
jaws does nothing at all.

"

The seat cannot be open-loop, and this is the measurement that says so. The
loop sags towards the rod as soon as the claw's lift takes the toy's weight, so
the hole reaches the rod *before* any nominal seat height does; a prescribed
descent then keeps pushing the jaws down past a toy the rod is holding, and the
loop slides up out of the 12 mm pad faces and the toy is dropped (measured: the
loss at t = 5.2 s of a run whose lift and thread were clean). Descending until
the rod reports support and stopping there is the move a real guarded move is.

The signal is the *upward* part of the rod's contact force, not its magnitude. A
loop grazing the rod's side reports a large penalty force -- 18.4 N was measured
at 0.11 mm of penetration -- with no vertical component at all, so a magnitude
would stop the descent on a graze.

It also has to be sustained, not a single frame. A loop dragged across the rod's
free end during the thread reports support for one or two frames -- measured 2.18 N
at t = 4.4 s of the run that first used this guard -- and a guard that fired there
would hold the toy while it was still only half threaded on.
"""
SEAT_FRAMES = round(0.1 * CONTROL_HZ)
"""Consecutive frames (0.1 s) the rod must carry that fraction before the seat stops."""
SEAT_FLOOR = 0.006
"""How far past the nominal seat height the planned descent is allowed to go.

A guard that never fires -- a failed grasp -- must not drive the jaws through
the toy, so the planned path stops 6 mm below where the hole touches the rod and
holds there. Six and not twelve now that the thread lifts the toy 5 mm rather
than 30: the whole planned descent is then 11 mm, and its over-travel past a
contact the guard somehow misses is the 6.
"""
SEAT_RATE = 0.025
"""How fast the planned descent goes [m/s]: a ramp, not a smoothstep.

The catch is an impact whose severity is its speed. A smoothstep over the whole
phase peaks at several times its mean, and the same catch at 100 mm/s was
measured to push the loop back out of the pinch, while at 44 mm/s it survived.
A constant 25 mm/s is under that everywhere, and the guard stops the ramp a few
hundred milliseconds in anyway -- 11 mm of planned descent in total here.
"""
OPENED_HALF_GAP = 0.045
"""How far apart the jaws open once the toy is on the rod."""
SEAT_HOLD = 1.0
SEAT_OPEN = 0.6
SEAT_WITHDRAW = 0.5
"""After the guarded stop: hold the toy still, open, and rise clear."""
TABLE_Z = 0.75
REST_Z = TABLE_Z + PHYSICS.particle_radius
"""Where the toy's lowest node rests [m]: one particle radius above the bench.
Clearances are measured from here, so they read the same at any radius."""
SOURCE_X = -0.13
ROD_RADIUS = 0.006
POST_RADIUS = 0.010
ROD_FREE_Y = 0.030
"""The rod's free end, bench-relative: 30 mm ahead of the loop's plane."""
POST_Y = 0.160
CLEARANCE = 0.030
"""Space the hanging toy keeps under its lowest point [m]; the gate asks 20 mm."""
ARM_CLEARANCE = 0.060
"""The arm's rod height margin: its pinched loop hangs ~30 mm lower than the claw's."""
ARM_LIFT = (1.1, 3.1)
ARM_HANG_END = 3.9
"""The arm's lift window and the end of its hang [s]: pick-and-drop's, which holds the loop."""
ARM_THREAD = {
    "align": (0.0, 0.8),
    "thread": (1.0, 2.6),
    "lower": (3.0, 3.5),
    "open": (3.8, 4.3),
    "withdraw": (4.5, 5.1),
}
"""Arm phase windows after the hang [s]: align the hole, slide on, lower, open, rise."""
LIFT_ABOVE = 0.030
"""The lift leaves the pads' bottoms this far over the rod's top [m]."""
THREAD_ENGAGE = 0.035
"""How far past the rod's free end the ring is slid [m]."""
PAD_ROD_GAP = 0.002
"""Minimum pad-bottom clearance over the rod's top [m]; the arm does not collide with it."""
THREAD_LIFT = 0.005
RIM_HANG = 0.020
"""How far the loop's centre sits below the pads when the toy dangles from a rim
pinch [m], so the thread height is measured from the hole and not the pads.
"""
"""How far the hole's centre rides above its seated height while threading.

The thread has to happen with the rod *inside* the hole. Raising the toy until
the hole clears the rod's top by 20-30 mm does the opposite: the rod's free end
then meets the loop's lower material instead of its hole, and the slide drags the
loop along the rod's end until it is pulled out of the pinch -- measured 1.0-2.3 N
on the rod through the thread, `finger_contacts` falling from 4 to 0 at t = 4.0 s,
and the loop left draped across the free end with 12 mm of engagement where the
path asks 42. At 5 mm the tip sits in the hole with 12.6 mm of room under it and
5 mm over it, so the slide touches nothing and the guarded seat has 5 mm to fall.
"""
WITHDRAW_Z = 0.060
"""How far the opened jaws rise to clear the toy.

Straight up, and it matters that it is not also sideways: a first build moved
them 20 mm across while withdrawing, and because the toy is not yet on the rod
at that moment the open jaws swept it with them -- measured as the loop's
centroid travelling 75 mm in x and 34 mm up with the fingers, with nothing
touching it but the pads (`finger_contacts` 3, the rod's 0.000 N) at the end of
a run whose earlier frames had it threaded cleanly.
"""
PAD_HALF_THICKNESS = 0.003
"""The pad box's half-extent along the closing axis, as ``gripper.py`` builds it."""
ARM_BASE = arm_gripper.MOUNT
CELL = arm_gripper.CELL
PAD_CLEARANCE = 0.001
"""How far the pads' bottom faces start above the bench [m]. The close starts at
the ground, so a loop lying there is straddled instead of one assumed to sit at
the body's mid-plane.
"""
GRIP_TO_BODY = 0.006
"""How far inboard of the loop's widest band the pads close [m].

The widest band is the ring's equator; six millimetres toward the body the band
is still 22.96 mm of its 23.13 mm maximum, so the pinch stays on the wide part
while its faces get cord behind them. Measured on the rod: the forces are the
same 11.8 / 10.5 N and the ring still stretches to 1.96x and leaves the pads at
frame 61, so where along the band the pads land is not what the grip is short of.
"""
GRIP_SQUEEZE = 0.016
"""How far past the loop's measured width the closed pads aim.

Both ends of this were measured. At 1 mm the pads press about 1 N and the cord,
which is round where they meet it, slides straight back out of the pinch on the
first lifted step: five finger contacts through the close, none from the lift
on, and the toy never left the bench, in three identical runs. At 6 mm the drive
sits at its clamp and the grasp holds through lift, thread and seat -- the
squeeze is a bounded 20 N, which is what that clamp is for.
"""
# The seat is the slow one on purpose. Its descent is what brings the loop's
# hole down onto the rod, and the catch is an impact: the guided ramp runs at a
# constant 25 mm/s, whose momentum over one substep is about 1.9 N -- well under
# the 9.3 N the pinch could hold at the 5 N clamp. A smoothstep reaching 100 mm/s
# instead was measured to push the loop back out of the jaws (three runs).
PHASES = {
    # The same windows the other two scenes use, moved as early as the control loop
    # allows: lying flat, the loop is a limp hoop and the toy slides across the
    # bench within half a second of the scene starting.
    "close": (0.0, 0.6),
    # The pads hold still once they are closed: the toy settles onto the pinch
    # before the lift loads it.
    "lift": (1.1, 2.7),
    # The pad is still while the toy swings upright under the pinch -- the same
    # beat the cup and pick-and-drop scenes take between lift and carry.
    "thread": (3.5, 4.7),
    "seat": (4.7, 7.7),
    "open": (8.3, 8.8),
    "withdraw": (8.8, 9.3),
}
"""Planned phase windows. The seat's own end is not one of them: the descent
stops when the rod reports support, and ``SEAT_FLOOR`` bounds it if it does not.
The open and withdraw windows are only the fallback's, for a run whose guard
never fires; a working run gets its own ``after_seat`` path instead.
"""


def build_scene(
    *,
    claw: str = "arm",
    coarse: bool = False,
    solver_name: str = "chebyshev",
    iterations: int = 160,
    mu: float | None = None,
    grip_mu: float | None = None,
    force_limit: float = arm_gripper.GRASP_LIMIT_N,
    squeeze: float = GRIP_SQUEEZE,
    self_contact: bool = True,
    self_contact_radius: float = SELF_CONTACT_RADIUS,
    feedback_smoothing: float = 1.0,
    feedback_clamp: float = 0.0,
    surface_contact: bool = False,
    rigid_loop: bool = False,
    rod_radius: float = ROD_RADIUS,
):
    vertices, tets, labels = load_toy()
    ring = ring_geometry(vertices, tets, labels)

    # Lying flat in its own rest pose, grasped on the loop's rim: the same pose and
    # grasp the other two scenes use. The toy's lowest vertex rests it on the bench,
    # the loop lies flat at its end, and the toy is laid so the hole's centre sits on
    # the bench origin in y -- that origin is what the rod is aimed at.
    quat = (0.0, 0.0, 0.0, 1.0)
    hole = ring["hole_centre_local"]
    base = ring["base_local_y"]
    pose = (SOURCE_X, float(-hole[1]), float(-vertices[:, 2].min()) + 0.001)
    grip_z = float(ParallelGripperSpec.pad_half_height + PAD_CLEARANCE)
    hole_height = float(hole[1] - base)
    seat_drop = ring["hole_inradius"] - rod_radius
    rod_z = seat_drop + hole_height + (CLEARANCE if claw == "claw" else ARM_CLEARANCE)
    ring["inner_ids"] = hole_surface_ids(vertices, ring)
    # Where the hole's centre sits when the toy hangs on the rod, and how far
    # above that the slide runs. Both are about the hole's centre, so the lift
    # the pads make is their difference and nothing else.
    seat_height = rod_z - seat_drop
    thread_height = seat_height + THREAD_LIFT
    ring_plane_y = 0.0
    geometry = {
        "gripper": "claw" if claw == "claw" else "arm",
        "hole_centre_local": np.round(hole, 4).tolist(),
        "hole_inradius_m": ring["hole_inradius"],
        "hole_inradius_median_m": ring["hole_inradius_median"],
        "reach_m": ring["side_reach"],
        "pad_half_thickness_m": PAD_HALF_THICKNESS,
        "grip_squeeze_m": squeeze,
        "force_cap_n": force_limit,
        "toy_height_m": float(vertices[:, 1].max() - base),
        "hole_height_m": hole_height,
        # The pads start at the bench and the hole starts where the mesh puts it,
        # so the descent is measured from the hole, not from the pads.
        "grip_z_m": grip_z,
        "loop_z_m": float(pose[2] + hole[2]),
        "seat_drop_m": seat_drop,
        "rod_z_m": rod_z,
        "rod_free_y_m": ROD_FREE_Y,
        "lift_m": thread_height - grip_z + RIM_HANG,
        # Threaded well inboard of the free end, not merely past it: a build that
        # left 21 mm of engagement had the loop walk along the rod and off the
        # tip during the seat -- 20 mm of it, measured -- and lose the rod.
        "slide_m": ROD_FREE_Y + rod_radius + ring["loop_half_thickness"] + 0.030 - ring_plane_y,
        "engagement_m": rod_radius + ring["loop_half_thickness"] + 0.030,
        "rod_radius_m": rod_radius,
        "seat_z_m": seat_height,
        "thread_z_m": thread_height,
        "thread_lift_m": THREAD_LIFT,
        "withdraw_z_m": WITHDRAW_Z,
        "grip_centre_m": [SOURCE_X + float(hole[0]), -GRIP_TO_BODY if claw == "claw" else 0.0, float(grip_z)],
    }
    print(json.dumps({"geometry": geometry}), flush=True)

    split = claw != "claw"
    physics = arrangement(
        PHYSICS,
        claw=claw,
        coarse=coarse,
        solver_name=solver_name,
        iterations=iterations,
        self_contact=self_contact,
        self_contact_radius=self_contact_radius,
        feedback_smoothing=feedback_smoothing,
        feedback_clamp=feedback_clamp,
        surface_contact=surface_contact,
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="clawhauser_on_rod")
    rod_x = SOURCE_X + float(hole[0])
    scene.add(rss.slab(name="rod_base", half=(0.030, 0.030, 0.006), at=(rod_x, POST_Y, 0.006), collide=True))
    scene.add(rss.post(name="rod_post", radius=POST_RADIUS, half_height=rod_z / 2.0, at=(rod_x, POST_Y, rod_z / 2.0)))
    span = POST_Y - POST_RADIUS - ROD_FREE_Y
    rod = scene.add(
        rss.post(
            name="rod",
            radius=rod_radius,
            half_height=span / 2.0,
            # A quarter turn about x lays the cylinder's own z axis along y.
            at=(rod_x, ROD_FREE_Y + span / 2.0, rod_z, -np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0)),
        )
    )
    obj = scene.add(toy_spec(PHYSICS, pose=pose, quat=quat))
    grip = None
    if split:
        arm_gripper.add_arm(scene, base_offset=ARM_BASE, force_limit=force_limit)
    else:
        grip = scene.add(
            ParallelGripperSpec(
                name="rod_loop_gripper",
                # The loop's widest point, in the work-surface frame: the faces rest
                # on that width, and the squeeze is the aim past it the drive clamps.
                center=(SOURCE_X + float(hole[0]), -GRIP_TO_BODY, grip_z),
                half_gap=ring["side_reach"] + PAD_HALF_THICKNESS + 0.012,
                closed_half_gap=ring["side_reach"] + PAD_HALF_THICKNESS - squeeze,
                force_limit=force_limit,
                pad_half_thickness=PAD_HALF_THICKNESS,
                pad_half_width=0.006,
                duration=DURATION,
                control_hz=CONTROL_HZ,
                substeps=SUBSTEPS,
            )
        )
    if split:
        # Across the rod from the bench's far side.
        eye, look_at, fov_y = (1.02, -0.36, 1.14), (0.46, 0.06, 0.93), 45.0
    else:
        # Pulled back from the toy-and-rod framing so both ends of the scene stay
        # in frame for the whole carry, as in the cup and pick-and-drop scenes.
        eye, look_at, fov_y = (0.76, -0.55, 1.163), (0.46, 0.0, 0.95), 45.0
    camera = scene.add(rss.CameraSpec(name="overview", width=960, height=540, eye=eye, look_at=look_at, fov_y=fov_y))
    sim = scene.build(device="cuda:0", warmup=False, use_cuda_graph=False)
    entity = obj.entity
    geometry.update(
        check_build(
            sim,
            entity,
            tets,
            PHYSICS,
            claw=claw,
            force_limit=force_limit,
            mu=mu,
            grip_mu=grip_mu,
            rigid_loop=rigid_loop,
        )
    )
    if grip is not None and abs(float(grip.entity.center[2]) - (TABLE_Z + geometry["grip_z_m"])) > 1.0e-6:
        raise RuntimeError(
            f"the claw sits at z={grip.entity.center[2]:.4f}, not the bench-referenced "
            f"z={TABLE_Z + geometry['grip_z_m']:.4f} the path is built from"
        )
    return sim, obj, grip, camera, rod, ring, geometry


def waypoints(grip, capacity: int, dt: float, geometry: dict) -> np.ndarray:
    """The claw's own path: settle, close, lift, thread, seat, hold, open, withdraw.

    The descent runs to ``SEAT_FLOOR`` below the height where the hole meets the
    rod, not to that height: a run whose guard never fires -- a failed grasp --
    holds there instead of driving the jaws through the toy. ``after_seat`` is
    what every working run follows instead.
    """
    centre = np.asarray(grip.entity.center, dtype=np.float64)
    opened = float(grip.spec.half_gap)
    closed = float(grip.spec.closed_half_gap)
    lift = geometry["lift_m"]
    slide = geometry["slide_m"]
    seat = (TABLE_Z + geometry["seat_z_m"] - SEAT_FLOOR) - (TABLE_Z + geometry["loop_z_m"])
    # A ramp at SEAT_RATE, floored at the seat: a smoothstep would reach the rod at
    # twice that rate.
    floor = min(0.0, seat - lift)
    poses = np.zeros((capacity, 2, 7), dtype=np.float32)
    poses[:, :, 6] = 1.0
    for index in range(capacity):
        time = index * dt
        gap = opened + (closed - opened) * smooth(time, *PHASES["close"])
        gap += (OPENED_HALF_GAP - gap) * smooth(time, *PHASES["open"])
        away = smooth(time, *PHASES["withdraw"])
        # The pads come down to the height where the hole meets the rod: the
        # thread runs 5 mm above that, so what the last millimetres press with is
        # the guarded descent and not the clamp. The toy is on the rod before the
        # jaws open, and they leave upwards.
        place = centre + np.array(
            [
                0.0,
                slide * smooth(time, *PHASES["thread"]),
                lift * smooth(time, *PHASES["lift"])
                # max(0, ...) is what keeps this zero before the seat's window
                # opens: a ramp read at a negative time is a *rise*, which is how
                # the pads once flew 120 mm up at t = 0 and never touched the toy.
                + max(-SEAT_RATE * max(0.0, time - PHASES["seat"][0]), floor)
                + WITHDRAW_Z * away,
            ]
        )
        poses[index, 0, :3] = place + np.array([-gap, 0.0, 0.0])
        poses[index, 1, :3] = place + np.array([gap, 0.0, 0.0])
    return poses


def after_seat(positions: np.ndarray, dt: float, closed_half_gap: float) -> np.ndarray:
    """The path the guarded stop hands over to: hold, open, and rise clear.

    Built from where the fingers actually are when the rod reports the toy, so
    the hold is a hold and not a resumed descent -- but the *gap* it holds is the
    one the trajectory was aiming at, not the gap the fingers are at. Commanding
    the measured gap lets the squeeze go: the pads are pushed out of the loop's
    own reaction, so aiming at where they already are asks for no force at all,
    and the released loop slides straight out of the pinch. Measured: a run whose
    loop was on the rod and carrying 0.87 N lost all finger contact in the frame
    after that hand-over and dropped the toy at t = 5.2 s.
    """
    positions = np.asarray(positions, dtype=np.float64)
    count = int(np.ceil((SEAT_HOLD + SEAT_OPEN + SEAT_WITHDRAW) / dt)) + 2
    centre = 0.5 * (positions[0] + positions[1])
    half = float(closed_half_gap)
    poses = np.zeros((count, 2, 7), dtype=np.float32)
    poses[:, :, 3:] = (0.0, 0.0, 0.0, 1.0)
    for index in range(count):
        time = index * dt
        gap = half + (OPENED_HALF_GAP - half) * smooth(time, SEAT_HOLD, SEAT_HOLD + SEAT_OPEN)
        rise = WITHDRAW_Z * smooth(time, SEAT_HOLD + SEAT_OPEN, SEAT_HOLD + SEAT_OPEN + SEAT_WITHDRAW)
        poses[index, 0, :3] = centre + np.array([-gap, 0.0, rise])
        poses[index, 1, :3] = centre + np.array([gap, 0.0, rise])
    return poses


def hole_surface_ids(vertices: np.ndarray, ring: dict) -> np.ndarray:
    """Vertex ids of the hole's inner surface: within 1.5 mm of each angular sector's nearest."""
    loop_ids = np.asarray(ring["loop_ids"])
    hole = np.asarray(ring["hole_centre_local"], dtype=np.float64)
    offset = vertices[loop_ids][:, :2] - hole[:2]
    radius = np.linalg.norm(offset, axis=1)
    sectors = 72
    bins = np.floor((np.arctan2(offset[:, 1], offset[:, 0]) + np.pi) / (2.0 * np.pi) * sectors).astype(int) % sectors
    nearest = np.full(sectors, np.inf)
    np.minimum.at(nearest, bins, radius)
    return loop_ids[radius <= nearest[bins] + 0.0015]


def arm_route(sim, geometry: dict, capacity: int, dt: float):
    """Close on the loop's sides and lift the pads' bottoms ``LIFT_ABOVE`` over the rod's top."""
    control = arm_gripper.ArmControl(sim)
    centre = arm_gripper.tcp_for_pads(
        np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
        + np.asarray(geometry["grip_centre_m"], dtype=np.float64)
    )
    reach = float(geometry["reach_m"])
    squeeze = float(geometry["grip_squeeze_m"])
    lift = geometry["rod_z_m"] + geometry["rod_radius_m"] + LIFT_ABOVE - PAD_CLEARANCE

    def tcp_at(time: float) -> np.ndarray:
        return centre + np.array((0.0, 0.0, lift * smooth(time, *ARM_LIFT)))

    def half_gap_at(time: float) -> float:
        return reach + squeeze - 2.0 * squeeze * smooth(time, *PHASES["close"])

    route = control.plan(quat=arm_gripper.GRASP_QUAT, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
    return control, route


def arm_thread_route(control, sim, owned: np.ndarray, ring: dict, geometry: dict, rod_world: np.ndarray, dt: float):
    """Align the open part of the hole with the rod, slide on, lower, open and withdraw.

    The open part is the hole's inner surface below the pads' bottoms; the rod's
    axis is put midway between that surface's lowest point and the pads, and the
    lowering stops with the rod's top ``PAD_ROD_GAP`` under the pads.
    ``rod_world`` is the rod axis in world coordinates (x, free-end y, z).
    """
    tcp = np.asarray(control.tcp(), dtype=np.float64)
    low, _ = arm_gripper.pad_bounds(sim)
    pads_low = float(low[2])
    points = owned[ring["inner_ids"]]
    free = points[points[:, 2] < pads_low - 0.001]
    if len(free) < 3:
        free = points
    radius = float(geometry["rod_radius_m"])
    axis = 0.5 * (pads_low + float(free[:, 2].min()))
    dx = float(rod_world[0]) - float(free[:, 0].mean())
    dy = float(rod_world[1]) + THREAD_ENGAGE - float(free[:, 1].mean())
    dz = float(rod_world[2]) - axis
    lower = max(0.0, pads_low - PAD_ROD_GAP - radius - axis)
    capacity = int(np.ceil((ARM_THREAD["withdraw"][1] + 0.5) / dt)) + 2
    held = float(geometry["reach_m"]) - float(geometry["grip_squeeze_m"])

    def tcp_at(time: float) -> np.ndarray:
        return tcp + np.array(
            (
                dx * smooth(time, *ARM_THREAD["align"]),
                dy * smooth(time, *ARM_THREAD["thread"]),
                dz * smooth(time, *ARM_THREAD["align"])
                - lower * smooth(time, *ARM_THREAD["lower"])
                + WITHDRAW_Z * smooth(time, *ARM_THREAD["withdraw"]),
            )
        )

    def half_gap_at(time: float) -> float:
        # The commanded gap, not the measured one: aiming at where the fingers
        # already are asks for no force and lets the loop go.
        return held + (OPENED_HALF_GAP - held) * smooth(time, *ARM_THREAD["open"])

    route = control.plan(quat=arm_gripper.GRASP_QUAT, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
    info = {
        "free_points": len(free),
        "free_height_mm": 1000.0 * (pads_low - float(free[:, 2].min())),
        "align_mm": np.round(1000.0 * np.array((dx, dy, dz)), 2).tolist(),
        "lower_mm": 1000.0 * lower,
        "route_error_m": route.worst_position_error_m,
    }
    return route, info


def loop_stretch(sim, entity) -> dict:
    """Principal stretch of the loop, frame-only: how far the grasp pulled it."""
    q = sim.pipeline.state_0.particle_q.numpy()[entity.particles]
    tets = entity.source_tets[entity.tet_part_labels == 2]
    rest = entity.source_vertices * entity.spec.scale
    dm = rest[tets[:, 1:]] - rest[tets[:, :1]]
    ds = q[tets[:, 1:]] - q[tets[:, :1]]
    singular = np.linalg.svd(np.linalg.solve(dm, ds), compute_uv=False)
    return {"p95": float(np.quantile(singular[:, 0], 0.95)), "max": float(singular[:, 0].max())}


def newton_commit() -> str:
    """The Newton revision this run used; a staged source tree has no .git."""
    root = Path(newton.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f"unknown ({root} is not a git checkout)"


def run(
    out: Path,
    *,
    steps: int,
    render_every: int = 0,
    claw: str = "arm",
    coarse: bool = False,
    solver_name: str = "chebyshev",
    iterations: int = 160,
    mu: float | None = None,
    grip_mu: float | None = None,
    force_limit: float = arm_gripper.GRASP_LIMIT_N,
    squeeze: float = GRIP_SQUEEZE,
    enable_self_contact: bool = True,
    self_contact_radius: float = SELF_CONTACT_RADIUS,
    feedback_smoothing: float = 1.0,
    feedback_clamp: float = 0.0,
    surface_contact: bool = False,
    rigid_loop: bool = False,
    rod_radius: float = ROD_RADIUS,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, grip, camera, rod, ring, geometry = build_scene(
        claw=claw,
        coarse=coarse,
        solver_name=solver_name,
        iterations=iterations,
        mu=mu,
        grip_mu=grip_mu,
        force_limit=force_limit,
        squeeze=squeeze,
        self_contact=enable_self_contact,
        self_contact_radius=self_contact_radius,
        feedback_smoothing=feedback_smoothing,
        feedback_clamp=feedback_clamp,
        surface_contact=surface_contact,
        rigid_loop=rigid_loop,
        rod_radius=rod_radius,
    )
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    rod_world = surface + np.array((SOURCE_X + float(ring["hole_centre_local"][0]), ROD_FREE_Y, geometry["rod_z_m"]))
    thread_frame = round(ARM_HANG_END * CONTROL_HZ)
    entity = obj.entity
    particles = entity.particles
    tets = entity.source_tets
    loop_ids = particles[ring["loop_ids"]]
    # The toy's own solver: the coupling wrapper when the arm is in the scene, and
    # every VBD attribute below has to be read from it, not from the wrapper.
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    self_contact = {
        "enabled": bool(getattr(solver, "particle_enable_self_contact", False)),
        "margin_m": float(getattr(solver, "particle_self_contact_margin", 0.0)),
        "rest_exclusion_m": float(getattr(solver, "particle_rest_shape_contact_exclusion_radius", 0.0)),
    }
    if self_contact["enabled"] != enable_self_contact:
        raise RuntimeError("the solver's self-contact setting does not match the scene's")
    # The solver owns the self-contact detector, so its pairs live in the solver's
    # buffer, not ``sim.pipeline.contacts``. Read whichever one is populated.
    pair_info, pair_buffer = None, "absent"
    for name, info in (
        ("solver.detector", getattr(solver, "trimesh_collision_info", None)),
        ("solver.contacts", getattr(getattr(solver, "contacts", None), "soft_self_contact_data", None)),
        ("pipeline.contacts", getattr(sim.pipeline.contacts, "soft_self_contact_data", None)),
    ):
        if info is not None:
            pair_info, pair_buffer = info, name
            break
    self_contact["pair_buffer"] = pair_buffer
    if self_contact["enabled"] and pair_info is None:
        raise RuntimeError("self-contact is on but no pair buffer was found; the counts would read zero")
    print(json.dumps({"self_contact": self_contact}), flush=True)
    quality = MeshQuality(sim, entity)
    dt = 1.0 / (CONTROL_HZ * SUBSTEPS)
    control_dt = 1.0 / CONTROL_HZ
    arm, route, route_start = None, None, 0
    if grip is None:
        arm, route = arm_route(sim, geometry, thread_frame + 2, control_dt)
        if route.worst_position_error_m > 0.025:
            raise RuntimeError(f"the arm misses the rod path by {route.worst_position_error_m:.3f} m")
        # The claw starts at the grasp; the arm starts above it, so it is placed
        # where its route starts before the first step.
        arm.start(route)
        arms = sim.handles.arms[arm_gripper.ARM]
        finger_sides = [
            np.array([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32)
            for side in arms.finger_groups
        ]
        finger_shapes = np.unique(np.concatenate(finger_sides))
        shape_forces = getattr(sim.pipeline.program.solver, "shape_contact_forces", None)
        print(json.dumps({"arm_path_error_m": route.worst_position_error_m}), flush=True)
    else:
        finger_sides = [np.asarray(grip.entity.finger_shapes, dtype=np.int32)]
        finger_shapes = finger_sides[0]
        shape_forces = None
        grip.entity.set_trajectory(waypoints(grip, grip.entity.capacity, dt, geometry))
    q0 = sim.pipeline.state_0.particle_q.numpy()[particles]
    rest_det = np.linalg.det(q0[tets[:, 1:]] - q0[tets[:, :1]])
    if np.any(rest_det <= 0):
        raise RuntimeError("source tetrahedra were not positively oriented after build")
    rod_shapes = {int(v) for v in rod.shapes}
    mass = float(entity.mass.sum())
    weight = mass * 9.81
    print(json.dumps({"mass_kg": mass, "weight_n": weight}), flush=True)

    videos, video_error = None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        # Written at 60 fps: each rendered control frame fills a 60th times the
        # render interval, so the video runs at the scene's own speed.
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    diagnostics = []
    seat_frame = None
    seat_streak = 0
    started = time.perf_counter()
    try:
        for frame in range(steps):
            if arm is not None:
                arm.follow(route, frame - route_start)
            sim.run(1).sync()
            q = sim.pipeline.state_0.particle_q.numpy()
            owned = q[particles]
            body_q = sim.pipeline.state_0.body_q.numpy()
            contacts = sim.pipeline.contacts
            count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
            shapes = contacts.soft_contact_shape.numpy()[:count]
            touching = contacts.soft_contact_particle.numpy()[:count]
            depth = penetration(q, body_q, sim.model, contacts)
            # The pipeline stores a contact for every particle inside the 20 mm
            # detection margin, so a *record* is not a touch: only the ones with
            # real depth are. ``max_penetration_m`` and the loads below are read
            # from those depths already.
            touching_shape = depth > 0.0
            finger = np.isin(shapes, finger_shapes)
            on_rod = np.isin(shapes, list(rod_shapes))
            # Newton mixes a particle contact's stiffness as the mean of the two
            # sides: the object's own soft_contact_ke and the shape's. The rod is
            # built from the post fixture cfg (ke = 1e4), so a read at the object's
            # 1e5 would be 1.8x the truth. Of that, ``support`` keeps the upward
            # part, which is what carries weight; ``bench_support_n`` is the same
            # sum over every contact bar the pads', and it reads the toy's 1.227 N
            # while the toy is standing -- the calibration for ``rod_support_n``
            # once the rod is the only thing under it.
            load = 0.5 * (float(sim.model.soft_contact_ke) + sim.model.shape_material_ke.numpy()[shapes]) * depth
            support = load * contacts.soft_contact_normal.numpy()[:count, 2]
            det = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]])
            pairs, nearest = 0, None
            if pair_info is not None:
                # ``min_dist`` only means something on rows that found a pair.
                for pair_counts, distances in (
                    (
                        pair_info.vertex_colliding_triangles_count.numpy(),
                        pair_info.vertex_colliding_triangles_min_dist.numpy(),
                    ),
                    (
                        pair_info.edge_colliding_edges_count.numpy(),
                        pair_info.edge_colliding_edges_min_dist.numpy(),
                    ),
                ):
                    pairs += int(pair_counts.sum())
                    hit = pair_counts > 0
                    if hit.any():
                        found = float(distances[hit].min())
                        nearest = found if nearest is None else min(nearest, found)
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "ring_centre_m": owned[ring["loop_ids"]].mean(axis=0).tolist(),
                "minimum_z_m": float(owned[:, 2].min()),
                "minimum_tet_volume_ratio": float((det / rest_det).min()),
                "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
                "soft_contacts": count,
                "self_contact_pairs": pairs,
                "self_contact_min_dist_m": nearest,
                "finger_contacts": int(np.count_nonzero(finger & touching_shape)),
                "rod_ring_contacts": int(np.count_nonzero(on_rod & np.isin(touching, loop_ids) & touching_shape)),
                "rod_load_n": float(load[on_rod].sum()),
                "rod_support_n": float(support[on_rod].sum()),
                "bench_support_n": float(support[~finger].sum()),
                "max_penetration_m": float(depth.max()) if depth.size else 0.0,
                "finger_positions": [[float(v) for v in body_q[int(side[0][0])][:3]] for side in arms.finger_groups]
                if arm is not None
                else [[float(v) for v in body_q[b][:3]] for b in grip.entity.bodies],
            }
            diagnostics.append(row)
            if arm is not None and frame + 1 == thread_frame:
                route, info = arm_thread_route(arm, sim, owned, ring, geometry, rod_world, control_dt)
                route_start = frame + 1
                seat_frame = frame + 1 + round(ARM_THREAD["lower"][1] * CONTROL_HZ)
                print(json.dumps({"thread": {"frame": frame + 1, **info}}), flush=True)
            if (
                arm is None
                and frame + 1 >= round(PHASES["seat"][0] * CONTROL_HZ)
                and row["rod_support_n"] >= SEAT_TRIGGER * weight
            ):
                seat_streak += 1
            else:
                seat_streak = 0
            if arm is None and seat_frame is None and seat_streak >= SEAT_FRAMES:
                # The rod has the toy, and has had it for a few frames: stop
                # descending and hand over to hold, open, withdraw, built from
                # where the fingers actually are.
                seat_frame = frame + 1
                fingers = np.array([[body_q[b][0], body_q[b][1], body_q[b][2]] for b in grip.entity.bodies])
                grip.entity.set_trajectory(after_seat(fingers, dt, grip.spec.closed_half_gap))
                print(json.dumps({"seat_frame": seat_frame, "rod_support_n": row["rod_support_n"]}), flush=True)
            if arm is None:
                row["applied"] = grip.entity.applied_wrenches.numpy().tolist()
                row["peak_force_n"] = grip.entity.peak_force.numpy().tolist()
                row["peak_torque_nm"] = grip.entity.peak_torque.numpy().tolist()
            elif shape_forces is not None:
                # What each pad would feel: the readback carries friction, so the
                # grip is its closing-axis component and the total is the pad's load.
                load, squeeze = arm_gripper.pad_forces(sim, shape_forces(), finger_sides)
                row["peak_force_n"] = squeeze
                row["pad_force_n"] = load
            if frame % 15 == 0 or frame == steps - 1:
                print(json.dumps(row), flush=True)
            if videos is not None and frame % render_every == 0:
                try:
                    videos.append_frames({camera.name: sim.render(camera)})
                except Exception as exc:  # recorded, not swallowed
                    video_error = repr(exc)
                    videos = None
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:  # recorded, not swallowed
                video_error = repr(exc)

    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "scene": "clawhauser_on_rod",
                "asset": "clawhauser",
                "newton_version": newton.__version__,
                "newton_source_commit": newton_commit(),
                "solver": type(sim.pipeline.program.solver).__name__,
                "coarse": bool(coarse),
                "coarse_contact_gap_m": CONTACT_GAP if coarse else None,
                "coarse_budget_m": COARSE_BUDGET if coarse else None,
                "claw": geometry["gripper"],
                "self_contact": self_contact,
                "geometry": geometry,
                "mass_kg": mass,
                "seated_frame": seat_frame,
                "attachment_particles": int(sim.model.attachment_particle_body_count),
                "completed_steps": len(diagnostics),
                # Physics plus this scene's own per-frame diagnostics, i.e. what a
                # run costs with rendering off. The corrector's share on its own is
                # measured by the W1 probe, not here.
                "mean_frame_ms": 1000.0 * (time.perf_counter() - started) / max(1, len(diagnostics)),
                "mean_step_ms": 1000.0 * (time.perf_counter() - started) / max(1, len(diagnostics) * SUBSTEPS),
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )
    return judge(
        out, diagnostics, quality, geometry, sim, entity, steps, video_error=video_error, seat_frame=seat_frame
    )


def peak_over(diagnostics, key: str):
    """The run's largest value per pad for ``key``, not the last frame's.

    A grip's peak is what it held with, and the last frame is after the release:
    reading it there reported 0 N for a grasp that closed at the cap.
    """
    rows = [row[key] for row in diagnostics if row.get(key)]
    return [max(values) for values in zip(*rows, strict=True)] if rows else None


def judge(out: Path, diagnostics, quality, geometry, sim, entity, steps, *, video_error, seat_frame) -> dict:
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    # The hold is the window after the guarded stop: the toy is on the rod and the
    # jaws have not opened yet. Where it falls depends on when the rod reported
    # the toy, so it is read off that frame and not off a clock.
    hold = []
    if seat_frame is not None:
        hold = [
            row
            for row in diagnostics
            if seat_frame + max(1, CONTROL_HZ // 3) <= row["frame"] <= seat_frame + round(1.2 * CONTROL_HZ)
        ]
    settled = diagnostics[-max(2, int(0.5 * CONTROL_HZ)) :]
    weight = float(entity.mass.sum() * 9.81)
    # The claw's own support, while it is still holding and the toy must be up.
    rod_load = float(np.mean([row["rod_support_n"] for row in hold])) if hold else 0.0
    # What holds the toy once the jaws have opened. The loop is the part that is
    # still there to measure: it hangs from the hole, so its rise is the hole
    # centre's, and the depth by which that sits below the rod's axis is the
    # seating the geometry asked for (hole_inradius - rod_radius).
    standing = diagnostics[0]["ring_centre_m"]
    loop_rise = float(np.mean([row["ring_centre_m"][2] for row in settled])) - float(standing[2])
    seated_depth = geometry["rod_z_m"] - ((float(standing[2]) - TABLE_Z) + loop_rise)
    settled_load = float(np.mean([row["rod_support_n"] for row in settled]))
    settled_clearance = min(row["minimum_z_m"] for row in settled) - REST_Z
    # The rod's own support is the direct evidence of what holds the toy once the
    # jaws are gone. ``loop_rise`` and ``seated_depth`` are reported beside it but
    # not gated on: the loop deforms under the pinch, so its centroid is only a
    # proxy for where the hole sits, and a proxy is not what holds 1.227 N up.
    threaded = bool(
        seat_frame is not None
        and max(row["finger_contacts"] for row in settled) == 0
        and settled_clearance >= 0.020
        and settled_load >= 0.9 * weight
        and abs(float(np.mean([row["ring_centre_m"][0] for row in settled])) - float(standing[0])) < 0.010
    )
    stretch = loop_stretch(sim, entity)
    result = {
        "geometry": geometry,
        "self_contact": {
            "enabled": bool(getattr(solver, "particle_enable_self_contact", False)),
            "margin_m": float(getattr(solver, "particle_self_contact_margin", 0.0)),
            "rest_exclusion_m": float(getattr(solver, "particle_rest_shape_contact_exclusion_radius", 0.0)),
        },
        "weight_n": weight,
        "rod_load_n": rod_load,
        "rod_load_over_weight": rod_load / weight if weight else None,
        # The weight check for the load metric: half a second after the 1 mm
        # placement drop, with the toy standing on the bench.
        "standing_load_n": float(
            np.mean([row["bench_support_n"] for row in diagnostics[CONTROL_HZ // 2 : CONTROL_HZ]])
        ),
        "seated_frame": seat_frame,
        "peak_force_n": peak_over(diagnostics, "peak_force_n"),
        "peak_torque_nm": peak_over(diagnostics, "peak_torque_nm"),
        "settled_load_n": settled_load,
        "settled_clearance_m": settled_clearance,
        "hold_clearance_m": (min(row["minimum_z_m"] for row in hold) - REST_Z) if hold else None,
        "hold_finger_contacts": max((row["finger_contacts"] for row in hold), default=None),
        "released": max(row["finger_contacts"] for row in settled) == 0,
        "rod_threaded": threaded,
        "loop_rise_m": loop_rise,
        "seated_depth_m": seated_depth,
        "loop_stretch_p95": stretch["p95"],
        "loop_stretch_max": stretch["max"],
        "max_penetration_m": max(row["max_penetration_m"] for row in diagnostics),
        "max_inverted_tetrahedra": max(row["inverted_tetrahedra"] for row in diagnostics),
        "settled_inverted_tetrahedra": max(row["inverted_tetrahedra"] for row in settled),
        "quality": quality.read(),
        "steps": len(diagnostics),
        "video_error": video_error,
    }
    peak_force = max(result["peak_force_n"] or [0.0])
    peak_torque = max(result["peak_torque_nm"] or [0.0])
    # The claw's peak is its drive's wrench and exact; the arm's is a readback
    # through the coupling, which can sit a percent over the clamp for a substep.
    tolerance = 1.0e-6 if geometry["gripper"] == "claw" else arm_gripper.GRIP_TOLERANCE
    result["success"] = bool(
        steps >= round(DURATION * CONTROL_HZ)
        and result["seated_frame"] is not None
        and result["rod_threaded"]
        and result["hold_clearance_m"] is not None
        and result["hold_clearance_m"] >= 0.020
        and result["released"]
        and result["settled_inverted_tetrahedra"] == 0
        and result["quality"]["nonfinite_tet_substeps_by_part"] == [0, 0, 0]
        and min(result["quality"]["minimum_J_by_part"]) > 0.0
        and result["max_penetration_m"] <= 0.0005
        and peak_force <= float(geometry["force_cap_n"]) * (1.0 + tolerance)
        and peak_torque <= 0.03
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_arguments(parser, out=OUTPUT / "run", steps=round(DURATION * CONTROL_HZ), squeeze=GRIP_SQUEEZE)
    parser.add_argument("--rod-radius", type=float, default=ROD_RADIUS, help="rod radius [m]")
    args = parser.parse_args()
    run(args.out, **scene_kwargs(args), rod_radius=args.rod_radius)


if __name__ == "__main__":
    main()
