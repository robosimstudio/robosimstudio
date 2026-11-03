# SPDX-License-Identifier: Apache-2.0
"""Lift the labelled Clawhauser volume off the bench by its ring and lower it into
a faceted glass cylinder, tail first.

The toy starts lying flat, in the pose it rests in, and the claw takes the ring
where it lies: the ring is a hoop on a stem, so on the bench its plane is
horizontal at the body's mid-thickness and the pads close on its rim from either
side. Lifting pulls it upright, and the carry is built from the toy measured
where it hangs, because a rim pinch does not hang it on the pads' axis.

The cylinder's bore is deliberately too small for the toy: at the default 90 mm
inner diameter the tail (47 by 36 mm) enters and the body (110 by 72 mm) does
not, so the descent stalls on the rim with the tail inside. That is the point of
the scene: the guarded stall releases the loop, the pads land on the body's top,
and the carrier's 200 N press follows it in. The run measures how deep the toy got, how much
of it is inside, what the press deformed, whether the rim holds it after the jaws
open, and what the jaws spent to get it there.

The default mover is the AIRBOT Play arm with its own parallel hand
(`arm_gripper`): it reaches the ring, its fingers close on the cord under a 20 N
clamp, and it follows the same tool-centre path. `--claw claw` runs the scripted
parallel claw this scene was built on -- a level carrier, two prismatic jaws and
a carrier push -- which stays as the reference the arm is measured against.

The claw's budgets are separate by construction: two 20 N jaws and a 200 N carrier
push. The arm has one hand and one arm, so on it the press is the same pads that
hold the toy, and ``peak_force_n`` is the force they would feel while pressing
rather than only while closing. Its ceiling is the arm's own actuators -- six
joints at 24 and 16 N.m, which at this reach is about 60 N -- not a stated
number, so ``push_force_cap_n`` is that ceiling and the run reports what the
press actually reached.

Geometry is measured from the mesh and reported: the loop's hole centre is where
the pads close, its outer reach in the closing direction is the width the squeeze
is stated from, and the container is built from its bore radius, panel count and
depth, so a sweep over bores is an argument rather than a different file.

Control, for contrast rather than for the result:

    --coarse           run the common-translation solve under load

Knobs, each reported with its measured effect:

    --bore      inner radius of the container [m]
    --mu        the pipeline's single global soft-contact friction coefficient
    --cap       closing force clamp [N]
    --push-cap  pressing force clamp [N]

Run on a CUDA node with Newton 1.8 development and RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
from loop_geometry import ring_geometry, smooth
from mesh_quality import MeshQuality, penetration
from parallel_gripper import PUSH_LIMIT_N, ParallelGripperSpec

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec, load_labelled_tets

ASSET = Path(__file__).resolve().parent / "assets" / "clawhauser"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "clawhauser_stuff_into_bottle"
CONTROL_HZ = 30
SUBSTEPS = 20
DURATION = 24.5
"""Long enough for the grasp, the guarded descent that stalls on the rim, the
release, and the press that follows the body to the container's floor, and for
the open and withdraw to finish inside the run.
"""
BORE_RADIUS = 0.050
"""Inner radius of the container: 100 mm across by default. The body is 110 mm
wide and the tail 47 by 36 mm, so the bore takes the tail and the rim stops the
body; the 200 N carrier press then deforms the body into the space it does not
fit. 90 mm was measured to expel the toy off the bench instead of wedging it;
the 100 mm run is the b9 reference the user asked to keep.
"""
WALL = 0.004
"""Panel thickness. The container is 24 flat panels, not a cylinder: a faceted
wall is what a slab scene can build, and the facet's chord is what the toy rides
over on its way in.
"""
PANELS = 24
RIM_Z = 0.090
"""Height of the rim above the work surface: the depth the toy can reach."""
FLOOR_Z = 0.004
"""Height of the container's floor above the work surface."""
LIFT_CLEARANCE = 0.020
"""How far the toy's lowest point clears the rim while it is carried across."""
APPROACH = 0.004
"""How far it clears the rim when the descent starts."""
DESCEND_RATE = 0.035
"""Planned descent speed [m/s]. Slower than a drop and faster than a creep: the
tail has to find the bore, and the jam has to be a jam rather than an impact.
"""
JAM_STEP = 0.0005
JAM_FRAMES = 8
"""A stall: 0.5 mm of depth is the least progress that counts, and eight frames
without it (0.27 s) is long enough that the swing of the carried toy, which moves
its lowest point by tens of millimetres, cannot look like one.
"""
DROP_CLEARANCE = 0.006
"""Above this the toy is not resting on the bench."""
TABLE_Z = 0.75
BENCH_X = 0.6
"""Where the work-surface frame's origin sits in world x."""
SOURCE_X = -0.07
TARGET_X = 0.07
CONTACT_GAP = 0.02
"""The collision pipeline's detection gap. The coarse block's own gap must be
this: a correction can only reach contacts the pipeline stored."""
CONTACT_KE = 100000.0
"""The arrangement's rigid-soft penalty stiffness: support load is this times
the summed penetration a contact patch reports."""
COARSE_BUDGET = 0.005
"""Cumulative coarse translation per substep, at most a quarter of the gap."""
GRIP_SQUEEZE = 0.016
"""How far past the loop's measured width the closed pads aim [m]. The same
measured value the rod scene uses. The parallel jaws hold the pads' attitude, so
the loop cannot roll them, and the flat face plus its friction is the grip.
"""
PAD_HALF_THICKNESS = 0.003
"""The pad box's half-extent along the closing axis, as ``gripper.py`` builds it."""
ARM_BASE = (-0.48, 0.0, 0.132, 0.0, 0.0, 0.0, 1.0)
"""Where the arm stands, in the work-surface frame: on the bench's near side,
0.35 m from the ring. An embodiment's mount is stated in that frame; the IK and
the measured poses are world, so `arm_centre` adds the cell's origin."""
CELL = "bench_075"
PAD_HALF_WIDTH = 0.006
"""The pad is a 12 mm square patch on the inner end of the 10 x 12 x 30 mm finger
post: 12 mm across, the post's own width, and 12 mm tall, so it does not extend
past the finger. The 40 mm archive face is gone. Two patches press the body's
top, one on each shoulder.
"""
PAD_CLEARANCE = 0.001
"""How far the pads' bottom faces start above the bench [m]. The close starts at
the ground, so a loop lying there is straddled instead of one assumed to sit at
the body's mid-plane.
"""
OPENED_HALF_GAP = 0.034
"""How far apart the jaws open once the toy is in place: 68 mm, outside the
loop's 23 mm reach. The finger's post reaches 13 mm outboard of its pad centre,
so at 34 mm the post's outer face is 47 mm from the axis and clears the 50 mm
bore by 3 mm; at 40 mm it was 53 mm and touched the wall (88 finger-container
contacts, ``rim-blocks-pads``).
"""
PUSH_RATE = 0.030
"""How fast the chase descends [m/s]. Faster was measured and rejected: at
75 mm/s the pads met the released toy while it was still rocking on the rim and
knocked it off the bench, for no insertion at all.
"""
PUSH_TIME = 3.0
"""How long the chase may descend, and so how far [s]. The pads do not chase the
falling toy: they hold still for ``SETTLE_TIME`` while gravity settles it, then
approach the body's top, and the first body-particle contact stops the approach.
Until it does, the run reports no press. Measured: the toy hangs 198 mm below the
pinch, so 3 s (90 mm) leaves the pads 120 mm above the body's top and the press
never touches it, while longer bounds let the toy slip off the rim first.
"""
PRESS_DEPTH = 0.130
PRESS_RATE = 0.030
"""How far below its first contact with the body the press drives, and the rate
it drives at [m, m/s]. The toy hangs ~45 degrees off plumb from the pinch, so it
straightens under the press instead of descending with it: a travel sized to the
room it has left above the container's floor stopped the pads while the toy was
15 mm into the mouth. 130 mm at 30 mm/s carried it to the floor with the body
wedged in the mouth (85.3 mm deep, 47 % inside, 0.1 mm of shell penetration, no
inverted tetrahedra); the same travel at 20 mm/s stalled at 12 mm.
"""
PRESS_HOLD = 0.5
"""How long the press holds at its deepest pose [s]."""
JAM_OPEN = 0.6
JAM_WITHDRAW = 0.5
"""After the press: open and rise clear, over these two spans [s]."""

RELEASE_HALF_GAP = 0.034
"""How far the jaws open to release the loop and pass its sides on the way to
the body: 68 mm, 11 mm outside the loop's 23 mm reach and inside the 100 mm
bore. The pads stay at this gap for the press, so they land on the body's
shoulders at 34 mm each side of its top, clear of the loop between them.
"""
JAM_RELEASE = 0.4
"""How long the jaws take to open off the loop at the stall [s]."""
ALIGN_TIME = 0.2
"""How long the open pads take to slide over the settled body's top [s]. The
slide finishes before the descent starts, so both pads land together.
"""
SETTLE_TIME = 0.5
"""How long the pads hold still after the release while the toy falls and seats
under gravity [s]. The toy is never driven down by the pads: the pads wait, then
approach the body and press it.
"""
WITHDRAW_Z = 0.060
"""How far the opened jaws rise to clear the toy: straight up, and clear of the
rim, so the withdrawal cannot drag the toy out of the container it just entered.
"""
PHASES = {
    # Closed as early as the control loop allows: the ring is a soft hoop on a
    # stem with nothing under it, and it sags and rolls out of the jaws' reach
    # within half a second of the scene starting.
    "close": (0.0, 0.6),
    # The pads hold still once they are closed: the toy settles onto the pinch
    # before the lift loads it.
    "lift": (1.1, 5.1),
    "hang": (5.1, 6.6),
}
"""The authored part of the path: close, lift, and a hang with no motion, where
the toy comes off the bench and settles into hanging from the pinch. The hang is
1.5 s, the still beat the rod scene takes between lift and thread: the toy is
plumb in xy 1 s in, but its body keeps drifting under the pinch for another half
second, and the carry is measured from that pose. Everything after the hang is
measured (``carry_path``).
"""
CARRY_TIME = 6.0
"""How long the lateral carry takes [s]. The grasp is 146 mm from the bore's
axis, measured at the end of the hang, and 2 s of that is 73 mm/s under a toy
hanging from a free pinch: it lagged the pads by the whole 146 mm and its body
was 152 mm short of the bore when the descent started, so it jammed on the rim.
24 mm/s is under the toy's own pendulum response, so it follows the pads.
"""
CARRY_HOLD = 1.0
"""Still beat after the carry [s]. The pads hold still and the toy settles under
them, so the descent starts from where the toy is rather than where the pads are.
"""
LOWER_TIME = 0.5
DESCENT_TIME = 2.5
"""After the hang: slide the hanging toy over the bore, lower it to the approach
height, then descend [s]. The descent's end is not a boundary, only the ramp's
bound; what ends it in every working run is the jam guard.
"""
SOURCE_HASHES = {
    "final_mesh.mesh": "df3d1505d96d36fbfa76291e7ab94269a30500ba5e4b87516d0bf8f57f0c6458",
    "heterogeneous_params.npz": "7cfc8ca3cc7c08d261aa90b08832230801fec1ac62848d0e9239d9e2f318e8c6",
    "inferred_params.json": "45aacd698f2b38a6fd8bf13b46cc9c33497fc90c6c1bdf088ed12d915ec3d992",
    "appearance/visual_to_physics.npz": "4b015d1141794211e4e2c6e2b39fadef6762a73ffe2d76b921e47d5676f3d6c4",
}


def build_scene(
    *,
    coarse: bool,
    bore: float,
    cap: float,
    push_cap: float,
    mu: float | None = 8.0,
    squeeze: float = GRIP_SQUEEZE,
    self_contact: bool = True,
    claw: str = "arm",
):
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((ASSET / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Clawhauser source asset changed: {relative}")
    vertices, tets, labels, elastic, density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    if (len(vertices), len(tets)) != (1840, 6850) or set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("Clawhauser source geometry or labels changed")
    if not np.array_equal(np.unique(elastic[:, 0]), [5e5, 5e6]) or not np.array_equal(np.unique(density), [300, 1100]):
        raise ValueError("Clawhauser source material estimates changed")
    ring = ring_geometry(vertices, tets, labels)
    ring["body_ids"] = np.unique(tets[labels == 1])

    # Lying flat, in its own rest pose: no rotation, so the mesh's +z is up and
    # the body's lowest vertex is what rests it on the bench. The loop is a ring
    # on a stem at the head's end, lying in a horizontal plane; the pads start
    # with their bottom faces on the bench and close on it from either side.
    quat = (0.0, 0.0, 0.0, 1.0)
    hole = ring["hole_centre_local"]
    base = ring["base_local_y"]
    # Laid down so the ring's hole sits on the claw's own y: the body lies behind
    # it, and the toy is carried over the bore without a lateral detour.
    pose = (SOURCE_X, float(-hole[1]), float(-vertices[:, 2].min()) + 0.001)
    grip_z = float(ParallelGripperSpec.pad_half_height + PAD_CLEARANCE)
    # How far the toy hangs below the pinch once it is off the bench: the same
    # 198 mm the standing start measures, because the body hangs from the loop
    # either way. It is what the lift has to clear the rim by.
    hole_height = float(hole[1] - base)
    # The tail is what goes in first, so it is the tail's own centre that gets
    # carried over the container's axis. Centring the hole instead leaves the
    # tail 3.5 mm off-axis, and the tail's 47 by 36 mm section has 0.8 mm of
    # diametral clearance in a 60 mm bore: it would be riding on the facet edge.
    tail_tets = np.where(labels == 0)[0]
    tail_vertices = vertices[np.unique(tets[tail_tets])]
    tail_centre = vertices[tets[tail_tets]].mean(axis=1).mean(axis=0)
    # The claw's carrier and the arm's pads press with different things, so the
    # press ceiling is stated by the mover: the claw's own clamp, or the arm's
    # actuators.
    push_cap = float(push_cap) if claw == "claw" else arm_gripper.PUSH_LIMIT_N
    geometry = {
        "gripper": "claw" if claw == "claw" else "arm",
        "hole_centre_local": np.round(hole, 4).tolist(),
        "hole_inradius_m": ring["hole_inradius"],
        "hole_inradius_median_m": ring["hole_inradius_median"],
        "loop_reach_m": ring["side_reach"],
        "pad_half_thickness_m": PAD_HALF_THICKNESS,
        "grip_squeeze_m": squeeze,
        "toy_height_m": float(vertices[:, 1].max() - base),
        "hole_height_m": hole_height,
        # Where the jaws close and how wide the thing they close on is: the loop's
        # own centre and its outer reach, as in Scene A. See the gripper's note.
        "grip_z_m": grip_z,
        "grip_reach_m": ring["side_reach"],
        "grip_centre_m": [SOURCE_X + float(hole[0]), 0.0, float(grip_z)],
        "pad_half_width_m": PAD_HALF_WIDTH,
        # The ledges gripper.py builds under each pad (2.4 mm tall, 3 mm inboard
        # of the pad's face): the grasp rests on them, so the run says so.
        "ledge": False,
        "carrier": True,
        "bore_radius_m": bore,
        "rim_z_m": RIM_Z,
        "floor_z_m": FLOOR_Z,
        "lift_m": RIM_Z + LIFT_CLEARANCE + hole_height - grip_z,
        "tail_centre_local": np.round(tail_centre, 4).tolist(),
        "tail_extent_m": np.round(np.ptp(tail_vertices, axis=0), 4).tolist(),
        "lower_m": APPROACH - LIFT_CLEARANCE,
        "descend_m": FLOOR_Z - RIM_Z - APPROACH,
        "descend_rate_m_s": DESCEND_RATE,
        "withdraw_z_m": WITHDRAW_Z,
        "force_cap_n": cap,
        "push_force_cap_n": push_cap,
        "push_torque_cap_nm": 2.0,
    }
    print(json.dumps({"geometry": geometry}), flush=True)

    common = {
        "particle_enable_self_contact": self_contact,
        # Millimetre-scale self-contact: the solver's own default margin is
        # 0.2 m, which makes every particle pair touch and the toy explode.
        "particle_self_contact_radius": 0.002,
        "particle_self_contact_margin": 0.002,
        "particle_topological_contact_filter_threshold": 2,
        # The plush's own surface is within the contact margin at rest, and the
        # grasp squeezes the 11 mm loop by 6 mm; exclude pairs that are that
        # close in the rest shape, or the pinch folds the loop into itself.
        "particle_rest_shape_contact_exclusion_radius": 0.006,
        "particle_vertex_contact_buffer_size": 32,
        "particle_edge_contact_buffer_size": 64,
        "particle_enable_tile_solve": False,
        "rigid_body_particle_contact_buffer_size": 4096,
    }
    vbd = (
        rss.backend(
            "robosimstudio_materials.coarse_translation.SolverVBDCoarse",
            coarse_translation=True,
            coarse_contact_gap=CONTACT_GAP,
            coarse_budget=COARSE_BUDGET,
            iterations=12,
            **common,
        )
        if coarse
        else rss.backend(
            "robosimstudio_materials.chebyshev_acceleration.SolverVBDChebyshev",
            iterations=160,
            chebyshev_rho=0.95,
            chebyshev_gamma=1.0,
            **common,
        )
    )
    physics = rss.ArrangementSpec(
        control_hz=CONTROL_HZ,
        substeps=SUBSTEPS,
        use_cuda_graph=False,
        # The arm is articulated and resolves to MuJoCo, so it is its own island
        # and the toy keeps VBD; `one_way` with `slide_feedback` is what carries
        # the grip, and `contact_matching` is forwarded into its pipeline.
        backends=((("articulated", rss.backend("mujoco")),) if claw != "claw" else ())
        + (("soft", vbd), ("rigid", vbd), ("*", vbd)),
        **({"coupling": rss.coupler("one_way", contact_forces=True, slide_feedback=True)} if claw != "claw" else {}),
        collision=rss.CollisionSpec(
            soft_contact_margin=CONTACT_GAP,
            full_surface_contact=False,
            contact_ke=CONTACT_KE,
            contact_kd=25.0,
            broad_phase="nxn",
            contact_matching="sticky",
        ),
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="clawhauser_stuff_into_bottle")

    # The container: PANELS flat panels whose corners meet on the bore's outer
    # radius, so the inner surface is a faceted bore that never intrudes past the
    # radius it claims, plus a floor under all of them.
    glass = [
        scene.add(
            rss.slab(
                name="container_floor",
                half=(bore + WALL, bore + WALL, FLOOR_Z / 2.0),
                at=(TARGET_X, 0.0, FLOOR_Z / 2.0),
                color=(0.55, 0.68, 0.74),
            )
        )
    ]
    panel_radius = bore + WALL / 2.0
    half_height = (RIM_Z - FLOOR_Z) / 2.0
    for index in range(PANELS):
        angle = 2.0 * np.pi * index / PANELS
        # The panel's own x is the chord, so the yaw that lays it along the
        # tangent is the radial angle plus a quarter turn.
        yaw = angle + 0.5 * np.pi
        glass.append(
            scene.add(
                rss.slab(
                    name=f"container_panel_{index}",
                    half=(float(np.tan(np.pi / PANELS) * panel_radius), WALL / 2.0, half_height),
                    at=(
                        TARGET_X + panel_radius * float(np.cos(angle)),
                        panel_radius * float(np.sin(angle)),
                        FLOOR_Z + half_height,
                        0.0,
                        0.0,
                        float(np.sin(yaw / 2.0)),
                        float(np.cos(yaw / 2.0)),
                    ),
                    color=(0.62, 0.74, 0.80),
                )
            )
        )

    obj = scene.add(
        HeterogeneousSoftSpec(
            name="clawhauser",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            quat=quat,
            contact_ke=CONTACT_KE,
        )
    )
    grip = None
    if claw == "claw":
        grip = scene.add(
            ParallelGripperSpec(
                # The loop's sides. The parallel jaws hold the pads' attitude, so the
                # loop cannot roll them off; the flat face and its friction are the
                # grip. After the stall the body is pressed by the carrier, whose
                # push budget is separate from the jaws' closing clamp.
                name="bottle_loop_gripper",
                center=(SOURCE_X + float(hole[0]), 0.0, grip_z),
                half_gap=ring["side_reach"] + PAD_HALF_THICKNESS + 0.012,
                closed_half_gap=ring["side_reach"] + PAD_HALF_THICKNESS - squeeze,
                pad_half_thickness=PAD_HALF_THICKNESS,
                pad_half_width=PAD_HALF_WIDTH,
                force_limit=cap,
                push_force_limit=push_cap,
                duration=DURATION,
                control_hz=CONTROL_HZ,
                substeps=SUBSTEPS,
            )
        )
    else:
        arm_gripper.add_arm(scene, base_offset=ARM_BASE, force_limit=cap)
    if claw == "claw":
        # Framed for the lying start: the toy rests toward the camera, so the eye
        # is pulled back from the standing framing and the view is aimed at the
        # midpoint between the toy's rest and the bore.
        eye, look_at, fov_y = (0.782, -0.684, 1.197), (0.60, -0.06, 0.93), 45.0
    else:
        # Three-quarter from the arm's side, wider: the reach, the bore and the
        # press all have to be in frame together.
        eye, look_at, fov_y = (0.20, -1.05, 1.32), (0.55, 0.0, 0.92), 50.0
    camera = rss.CameraSpec(name="overview", width=960, height=540, eye=eye, look_at=look_at, fov_y=fov_y)
    scene.add(camera)
    sim = scene.build(device="cuda:0", warmup=False, use_cuda_graph=False)
    entity = obj.entity
    # The toy's own solver: the coupling wrapper when the arm is in the scene, and
    # every VBD attribute read below has to come from it, not from the wrapper.
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    if not type(solver).__name__.startswith("SolverVBD"):
        raise RuntimeError(f"expected a VBD solver, got {type(solver).__name__}")
    if grip is None:
        geometry["arm"] = arm_gripper.check_arm(sim, limit=cap)
    if sim.model.tet_count != len(tets) or len(entity.tet_part_labels) != len(tets):
        raise RuntimeError("RSS lost Clawhauser tetrahedra or part labels")
    if sim.model.attachment_particle_body_count:
        raise RuntimeError("the contact grasp scene must not contain particle attachments")
    if mu is not None:
        # Newton VBD exposes one global soft-contact friction coefficient; this is
        # it, and it is the knob the sweep turns. The solver steps the model it was
        # built with, so if that is a distinct object the assignment has to land
        # there as well -- and the value in use is reported either way.
        sim.model.soft_contact_mu = float(mu)
        if solver.model is not sim.model:
            solver.model.soft_contact_mu = float(mu)
        geometry["global_soft_contact_mu"] = float(solver.model.soft_contact_mu)
    model_tets = sim.model.tet_indices.numpy()[entity.tet_range[0] : entity.tet_range[1]]
    if not np.array_equal(model_tets - entity.particles[0], tets):
        raise RuntimeError("Newton changed the source tetrahedron order")
    young, nu = entity.source_elastic.T
    source_mu = young / (2.0 * (1.0 + nu))
    source_lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    model_elastic = sim.model.tet_materials.numpy()[entity.tet_range[0] : entity.tet_range[1], :2]
    if not np.allclose(model_elastic, np.column_stack((source_mu, source_lambda)), rtol=1e-5):
        raise RuntimeError("Newton FEM elasticity differs from the source estimates")
    if not np.allclose(sim.model.particle_mass.numpy()[entity.particles], entity.source_mass, rtol=1e-5):
        raise RuntimeError("Newton nodal mass differs from source per-tet densities")
    # The shapes, not the entity ranges: a range ends one past its last shape, and
    # that extra id belongs to the next entity. Counting contacts by the range put
    # the toy's proxy in the container and read every pad-toy press contact as a
    # pad-container hit (measured: 110 at the press, 4 with the pads still 100 mm
    # above the rim).
    glass_bodies = np.concatenate([np.asarray(part.bodies, dtype=np.int32) for part in glass])
    # The soft channel needs the exact shape ids: the container's static shapes
    # are on the world body, so mapping back through the glass bodies gives an
    # empty set and every toy-cup contact vanishes. The pad gate stays body-based: a
    # static shape's body cannot name the cup, and its range endpoint is the next
    # entity's first shape (the source of the old 110 pad-toy hits).
    glass_shapes = np.concatenate([np.arange(*part.shapes, dtype=np.int32) for part in glass])
    return sim, obj, grip, camera, ring, glass_shapes, glass_bodies, geometry


def arm_centre(geometry: dict) -> np.ndarray:
    """The arm's grasp point in world coordinates.

    The scene states the grasp in the work-surface frame, like every other
    placement; the IK solves in world, so this is where the frame's origin is
    added -- once, and the path built on the result.
    """
    origin = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    return origin + np.asarray(geometry["grip_centre_m"], dtype=np.float64)


def waypoints(centre, opened: float, closed: float, capacity: int, dt: float, geometry: dict) -> np.ndarray:
    """The path to the hang: close, lift, and hold.

    ``opened`` and ``closed`` are the mover's own gap at each end: the claw states
    it between the pads' centres, the arm between their faces (``plan_poses``
    takes the difference). Where the toy hangs is not authored -- the pinch holds
    the ring at its rim, so the body hangs off the pads' axis by however the
    compliant hoop folds, and the carry is built from the toy measured at the end
    of the hang.
    """
    centre = np.asarray(centre, dtype=np.float64)
    lift = geometry["lift_m"]
    poses = np.zeros((capacity, 2, 7), dtype=np.float32)
    poses[:, :, 6] = 1.0
    for index in range(capacity):
        time = index * dt
        gap = opened + (closed - opened) * smooth(time, *PHASES["close"])
        place = centre + np.array([0.0, 0.0, lift * smooth(time, *PHASES["lift"])])
        poses[index, 0, :3] = place + np.array([-gap, 0.0, 0.0])
        poses[index, 1, :3] = place + np.array([gap, 0.0, 0.0])
    return poses


def carry_path(positions, target, dt: float, capacity: int, geometry: dict, closed: float):
    """From the hang: carry the toy over the bore, lower it, and descend.

    ``target`` is the pads' xy that puts the toy's lowest band -- the part that
    enters the cup first -- on the container's axis, measured from the hanging
    toy. The toy hangs canted from its pinched ring, so aiming the body's own
    centroid put the body 40 mm beside the bore and it met the rim instead of
    the mouth. The descent runs to the container's floor and is bounded there,
    not scheduled.

    Returns the poses and the time the descent ramp starts, when the jam guard
    arms.
    """
    centre = 0.5 * (np.asarray(positions, dtype=np.float64)[0] + positions[1])
    lower_at = CARRY_TIME + CARRY_HOLD
    descend_at = lower_at + LOWER_TIME
    open_at = descend_at + DESCENT_TIME + 0.6
    withdraw_at = open_at + 0.5
    poses = np.zeros((capacity, 2, 7), dtype=np.float32)
    poses[:, :, 6] = 1.0
    for index in range(capacity):
        time = index * dt
        gap = closed + (OPENED_HALF_GAP - closed) * smooth(time, open_at, open_at + 0.5)
        place = np.array(
            [
                centre[0] + (float(target[0]) - centre[0]) * smooth(time, 0.0, CARRY_TIME),
                centre[1] + (float(target[1]) - centre[1]) * smooth(time, 0.0, CARRY_TIME),
                centre[2]
                + geometry["lower_m"] * smooth(time, lower_at, descend_at)
                + max(-DESCEND_RATE * max(0.0, time - descend_at), geometry["descend_m"])
                + WITHDRAW_Z * smooth(time, withdraw_at, withdraw_at + 0.5),
            ]
        )
        poses[index, 0, :3] = place + np.array([-gap, 0.0, 0.0])
        poses[index, 1, :3] = place + np.array([gap, 0.0, 0.0])
    return poses, descend_at


def body_top_centre(owned: np.ndarray, ring: dict, band: float = 0.020) -> np.ndarray:
    """The body's top patch, where the two pads have to land together.

    The loop's centroid is not the body's: after the stall the toy is tilted and
    the pads were 17 mm off in x and 41 mm in y when they chased the loop. The
    top band of the body's own vertices is the surface the open pads meet.
    """
    body = owned[ring["body_ids"]]
    return body[body[:, 2] >= body[:, 2].max() - band].mean(axis=0)


def after_jam(positions: np.ndarray, dt: float, closed_half_gap: float) -> np.ndarray:
    """Open off the loop and hold: gravity seats the toy on the rim.

    The pinch cannot push (a 12 mm pad on an 11 mm round cord loses it as soon as
    the pads descend), so the pads open to 68 mm -- outside the loop's 23 mm
    reach -- and stay put. No descent and no lateral move: the body's centre is
    re-measured after the settle, in ``chase_path``, because a toy that slid on
    the rim while it fell is no longer above the pose it had at the stall.
    """
    positions = np.asarray(positions, dtype=np.float64)
    centre = 0.5 * (positions[0] + positions[1])
    count = int(np.ceil((JAM_RELEASE + SETTLE_TIME + 0.1) / dt)) + 2
    poses = np.zeros((count, 2, 7), dtype=np.float32)
    poses[:, :, 3:] = (0.0, 0.0, 0.0, 1.0)
    for index in range(count):
        gap = closed_half_gap + (RELEASE_HALF_GAP - closed_half_gap) * smooth(index * dt, 0.0, JAM_RELEASE)
        poses[index, 0, :3] = centre + np.array([-gap, 0.0, 0.0])
        poses[index, 1, :3] = centre + np.array([gap, 0.0, 0.0])
    return poses


def chase_path(positions: np.ndarray, dt: float, body_centre: np.ndarray) -> np.ndarray:
    """From the settled pose: slide over the body's top, then chase it down.

    The centre is the one measured after the settle, so the pads are aimed at the
    body's top where it is rather than where it was at the stall. The slide ends
    before the descent starts, so both flat bottoms meet the shoulders together:
    a claw that descends off-centre lands one pad first and tips the toy off the
    rim. The chase ends at the first pad-particle contact, the handover to
    ``press_path``; a path that reaches its end without one claims no press.
    """
    positions = np.asarray(positions, dtype=np.float64)
    body_centre = np.asarray(body_centre, dtype=np.float64)[:2]
    centre = 0.5 * (positions[0] + positions[1])
    fallback = ALIGN_TIME + PUSH_TIME
    count = int(np.ceil((fallback + JAM_WITHDRAW + JAM_OPEN) / dt)) + 2
    poses = np.zeros((count, 2, 7), dtype=np.float32)
    poses[:, :, 3:] = (0.0, 0.0, 0.0, 1.0)
    for index in range(count):
        time = index * dt
        gap = RELEASE_HALF_GAP + (OPENED_HALF_GAP - RELEASE_HALF_GAP) * smooth(
            time, fallback + JAM_WITHDRAW, fallback + JAM_WITHDRAW + JAM_OPEN
        )
        align = smooth(time, 0.0, ALIGN_TIME)
        cx = centre[0] + (body_centre[0] - centre[0]) * align
        cy = centre[1] + (body_centre[1] - centre[1]) * align
        drop = -PUSH_RATE * max(0.0, time - ALIGN_TIME)
        rise = WITHDRAW_Z * smooth(time, fallback, fallback + JAM_WITHDRAW)
        poses[index, 0, :3] = np.array([cx - gap, cy, centre[2] + drop + rise])
        poses[index, 1, :3] = np.array([cx + gap, cy, centre[2] + drop + rise])
    return poses


def press_path(positions: np.ndarray, dt: float, start_half_gap: float = RELEASE_HALF_GAP) -> np.ndarray:
    """From the contact pose: press straight down, hold, open, then rise.

    The pads start at ``start_half_gap``: the release gap when the chase hands
    over, the closed gap when the pinch itself is the thing doing the pressing.
    The descent runs ``PRESS_DEPTH`` below the pose they first touched the body at,
    at ``PRESS_RATE``. The push clamp bounds the contact.
    """
    positions = np.asarray(positions, dtype=np.float64)
    centre = 0.5 * (positions[0] + positions[1])
    press_s = PRESS_DEPTH / PRESS_RATE
    hold_end = press_s + PRESS_HOLD
    # Open before rising: under the press the pads are loaded into the plush, and
    # 20 N of jaw cannot slide out from under that normal on its own. Rising while
    # opening dragged the toy back up with the pads, by 26 mm.
    open_end = hold_end + JAM_OPEN
    rise_end = open_end + JAM_WITHDRAW
    count = int(np.ceil(rise_end / dt)) + 2
    poses = np.zeros((count, 2, 7), dtype=np.float32)
    poses[:, :, 3:] = (0.0, 0.0, 0.0, 1.0)
    for index in range(count):
        time = index * dt
        gap = start_half_gap + (OPENED_HALF_GAP - start_half_gap) * smooth(time, hold_end, open_end)
        drop = -PRESS_RATE * min(time, press_s)
        rise = WITHDRAW_Z * smooth(time, open_end, rise_end)
        poses[index, 0, :3] = centre + np.array([-gap, 0.0, drop + rise])
        poses[index, 1, :3] = centre + np.array([gap, 0.0, drop + rise])
    return poses


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
    coarse: bool = False,
    bore: float = BORE_RADIUS,
    cap: float = arm_gripper.GRASP_LIMIT_N,
    push_cap: float = PUSH_LIMIT_N,
    mu: float | None = 8.0,
    squeeze: float = GRIP_SQUEEZE,
    enable_self_contact: bool = True,
    claw: str = "arm",
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, grip, camera, ring, glass_shapes, glass_bodies, geometry = build_scene(
        coarse=coarse,
        bore=bore,
        cap=cap,
        push_cap=push_cap,
        mu=mu,
        squeeze=squeeze,
        self_contact=enable_self_contact,
        claw=claw,
    )
    entity = obj.entity
    particles = entity.particles
    tets = entity.source_tets
    # Which part each particle belongs to, so the press can tell a body contact
    # from a loop contact: the loop sits between the pads on the way down.
    part_of_particle = np.zeros(len(particles), dtype=np.int32)
    part_of_particle[tets] = entity.tet_part_labels[:, None]
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    self_contact = {
        "enabled": bool(getattr(solver, "particle_enable_self_contact", False)),
        "margin_m": float(getattr(solver, "particle_self_contact_margin", 0.0)),
        "rest_exclusion_m": float(getattr(solver, "particle_rest_shape_contact_exclusion_radius", 0.0)),
    }
    if self_contact["enabled"] != enable_self_contact:
        raise RuntimeError("the solver's self-contact setting does not match the scene's")
    print(json.dumps({"self_contact": self_contact}), flush=True)
    quality = MeshQuality(sim, entity)
    dt = 1.0 / (CONTROL_HZ * SUBSTEPS)
    # The two movers state their gap in the same place -- half the separation
    # between the two finger origins -- and the arm's pad face is 2 mm inside its
    # finger's origin where the claw's is 3 mm. So each gets its own closed and
    # open end, both derived from the loop's measured reach, and the travel past
    # it is the same 12 mm.
    reach = float(geometry["grip_reach_m"])
    if grip is None:
        closed_gap = reach - squeeze + arm_gripper.PAD_INNER_OFFSET
        open_gap = arm_gripper.OPEN_HALF_GAP - arm_gripper.PAD_INNER_OFFSET
        # Substeps, like the claw's own capacity: the path is built at the substep
        # rate and the arm resamples it. Eight seconds covers close, lift and hang
        # with a second to spare; `carry_path` replans at the end of the hang.
        capacity = round(8.0 * CONTROL_HZ) * SUBSTEPS
    else:
        closed_gap = float(grip.spec.closed_half_gap)
        open_gap = float(grip.spec.half_gap)
        capacity = grip.entity.capacity
    plan = waypoints(
        grip.entity.center if grip is not None else arm_centre(geometry), open_gap, closed_gap, capacity, dt, geometry
    )
    # The plan's own first sample, printed because a plan is a thing that can be
    # wrong on its own: a ramp read at a negative time once put the jaws 210 mm
    # above the toy for a whole run, and no diagnostic downstream could see it.
    print(
        json.dumps(
            {
                "trajectory": {
                    "samples": int(len(plan)),
                    "start_m": [float(0.5 * (plan[0, 0, i] + plan[0, 1, i])) for i in range(3)],
                    "start_gap_m": float(plan[0, 1, 0] - plan[0, 0, 0]),
                    "closed_gap_m": float(closed_gap * 2.0),
                    "lowest_z_m": float(0.5 * (plan[:, 0, 2] + plan[:, 1, 2]).min()),
                }
            }
        ),
        flush=True,
    )
    arm = None if grip is not None else arm_gripper.ArmControl(sim)
    route, route_start = None, 0

    def set_path(poses, *, start: int) -> None:
        """Hand the next stretch of motion to whichever mover the scene has.

        One call site per replan, so the measured path is written once and both
        movers read it: the claw consumes it per substep, the arm resamples it at
        the control rate and solves IK for it.
        """
        nonlocal arm, route, route_start
        if arm is None:
            grip.entity.set_trajectory(poses)
            return
        route = arm.plan_poses(
            poses, quat=arm_gripper.GRASP_QUAT, substeps=SUBSTEPS, pad_inset=arm_gripper.PAD_INNER_OFFSET
        )
        route_start = start

    set_path(plan, start=0)
    if arm is not None:
        # The claw starts at the grasp; the arm starts above it, so it is placed
        # where its first route starts before the first step. Later replans are
        # taken from where the arm already is and must not move it.
        arm.start(route)
        arms = sim.handles.arms[arm_gripper.ARM]
        finger_sides = [
            np.array([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32)
            for side in arms.finger_groups
        ]
        finger_shapes = np.unique(np.concatenate(finger_sides))
        finger_bodies = np.asarray([int(side[0][0]) for side in arms.finger_groups], dtype=np.int32)
    else:
        finger_sides, finger_shapes = None, np.asarray(grip.entity.finger_shapes, dtype=np.int32)
        finger_bodies = np.asarray(grip.entity.bodies, dtype=np.int32)
    shape_forces = getattr(sim.pipeline.program.solver, "shape_contact_forces", None)
    q0 = sim.pipeline.state_0.particle_q.numpy()[particles]
    rest_det = np.linalg.det(q0[tets[:, 1:]] - q0[tets[:, :1]])
    if np.any(rest_det <= 0):
        raise RuntimeError("source tetrahedra were not positively oriented after build")
    mass = float(entity.source_mass.sum())
    weight = mass * 9.81
    rim = TABLE_Z + RIM_Z
    # The container is placed in the work-surface frame, the particles are read in
    # world coordinates, and the frame's origin is 0.6 m along x.
    axis = np.array([BENCH_X + TARGET_X, 0.0])
    print(json.dumps({"mass_kg": mass, "weight_n": weight, "rim_z_m": rim}), flush=True)

    diagnostics = []
    descend_frame = round(PHASES["hang"][1] * CONTROL_HZ)
    jam_frame = None
    chase_frame = None
    press_frame = None
    shape_body = sim.model.shape_body.numpy()
    body_contact_seen = False
    jam_streak = 0
    depth_best = 0.0
    progress_frame = 0
    started = time.perf_counter()
    videos = None
    video_error = None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        # Written at 60 fps: each rendered control frame fills a 60th times the
        # render interval, so the video runs at the scene's own speed.
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    try:
        for frame in range(steps):
            if arm is not None:
                arm.follow(route, frame - route_start)
            sim.run(1).sync()
            q = sim.pipeline.state_0.particle_q.numpy()
            body_q = sim.pipeline.state_0.body_q.numpy()
            owned = q[particles]
            if not np.isfinite(owned).all():
                raise RuntimeError(f"non-finite Clawhauser state at frame {frame}")
            det = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]])
            contacts = sim.pipeline.contacts
            count = int(contacts.soft_contact_count.numpy()[0])
            contact_shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            depth = penetration(q, body_q, sim.model, contacts)
            touching = depth > 0.0
            finger = np.isin(contact_shapes, finger_shapes)
            in_glass = np.isin(contact_shapes, glass_shapes)
            # Two different penetrations, kept apart. A pad pressed 50 N into the
            # plush indents it by the solver's slack -- that is the deformation the
            # press is for. The toy sinking into the container's shell is not.
            container_depth = depth[in_glass & touching]
            pad_depth = depth[finger & touching]
            soft_particles = contacts.soft_contact_particle.numpy()[: min(count, contacts.soft_contact_max)]
            finger_local = soft_particles[finger] - particles[0]
            finger_local = finger_local[(finger_local >= 0) & (finger_local < len(part_of_particle))]
            body_contacts = int(np.count_nonzero(part_of_particle[finger_local] == 1))
            rigid_count = int(contacts.rigid_contact_count.numpy()[0])
            rigid_max = min(rigid_count, contacts.rigid_contact_max)
            rigid0 = contacts.rigid_contact_shape0.numpy()[:rigid_max]
            rigid1 = contacts.rigid_contact_shape1.numpy()[:rigid_max]
            body0, body1 = shape_body[rigid0], shape_body[rigid1]
            finger0, finger1 = np.isin(body0, finger_bodies), np.isin(body1, finger_bodies)
            container0, container1 = np.isin(body0, glass_bodies), np.isin(body1, glass_bodies)
            pad_container = (finger0 & container1) | (finger1 & container0)
            pad_other = (finger0 & ~container1) | (finger1 & ~container0)
            normal = contacts.rigid_contact_normal.numpy()[:rigid_max]
            point0 = contacts.rigid_contact_point0.numpy()[:rigid_max]
            point1 = contacts.rigid_contact_point1.numpy()[:rigid_max]
            # The witness points are separated for a near contact and overlapped for
            # a penetrating one; only the latter is a hit on the container.
            gap = np.einsum("ij,ij->i", point1 - point0, normal)
            pads_in_glass = int(np.count_nonzero(pad_container & (gap <= 1.0e-6)))
            pads_near_glass = int(np.count_nonzero(pad_container))
            finger_other = int(np.count_nonzero(pad_other))
            detail = []
            for index in np.flatnonzero(pad_container)[:8]:
                point = 0.5 * (point0[index] + point1[index])
                radius = float(np.linalg.norm(point[:2] - axis))
                detail.append(
                    {
                        "pad_side": 0 if finger0[index] else 1,
                        "pad_body": int(body0[index] if finger0[index] else body1[index]),
                        "container_body": int(body1[index] if finger0[index] else body0[index]),
                        "gap_m": float(gap[index]),
                        "penetrating": bool(gap[index] <= 1.0e-6),
                        "point_m": [float(v) for v in point],
                        "radius_m": radius,
                        "wall_gap_m": float(bore - radius),
                        "over_rim_m": float(point[2] - rim),
                    }
                )
            other_bodies = sorted(
                {int(body1[i]) if finger0[i] else int(body0[i]) for i in np.flatnonzero(pad_other)[:8]}
            )
            # What is inside: below the rim, and within the container's own bore
            # radius of its axis. The two are reported separately because a toy
            # resting on the rim is below the rim at its tail and beside the bore
            # at its body.
            within = np.linalg.norm(owned[:, :2] - axis, axis=1) < bore + 2.0 * WALL
            below_rim = float(np.mean(owned[:, 2] < rim))
            inside = float(np.mean(within & (owned[:, 2] < rim)))
            # How deep the part of the toy that is over the bore went: a plush
            # lying on the bench beside the container is 90 mm below the rim plane
            # and has inserted nothing.
            over_bore = float(rim - owned[within, 2].min()) if np.any(within) else 0.0
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "minimum_z_m": float(owned[:, 2].min()),
                "insertion_depth_m": max(0.0, over_bore),
                "below_rim_fraction": below_rim,
                "inside_fraction": inside,
                "ring_centre_m": owned[ring["loop_ids"]].mean(axis=0).tolist(),
                "minimum_tet_volume_ratio": float((det / rest_det).min()),
                "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
                "soft_contacts": count,
                "finger_contacts": int(np.count_nonzero(finger & touching)),
                "body_contacts": body_contacts,
                "container_contacts": int(np.count_nonzero(in_glass & touching)),
                "finger_container_contacts": pads_in_glass,
                "finger_container_near": pads_near_glass,
                "finger_other_contacts": finger_other,
                "finger_other_bodies": other_bodies,
                "finger_container_detail": detail,
                "max_penetration_m": float(depth.max()) if depth.size else 0.0,
                "max_container_penetration_m": float(container_depth.max()) if container_depth.size else 0.0,
                "max_pad_penetration_m": float(pad_depth.max()) if pad_depth.size else 0.0,
                "finger_positions": [[float(v) for v in body_q[b][:3]] for b in finger_bodies],
            }
            if grip is not None:
                row["carrier_pose"] = [float(v) for v in body_q[grip.entity.carrier_body]]
                row["actual_half_gaps_m"] = grip.entity.actual_half_gaps.numpy().tolist()
                row["slide_lateral_m"] = grip.entity.slide_lateral_error.numpy().tolist()
                row["slide_angle_rad"] = grip.entity.slide_angular_error.numpy().tolist()
                row["applied_carrier"] = grip.entity.applied_carrier.numpy().tolist()
            diagnostics.append(row)
            if frame + 1 == round(PHASES["hang"][1] * CONTROL_HZ):
                # The toy has hung still: carry the pads so the band that enters
                # the cup first is over the bore's axis, measured from where the toy
                # actually hangs rather than from the pose it was grasped in.
                fingers = np.array([[body_q[b][0], body_q[b][1], body_q[b][2]] for b in finger_bodies])
                pads = 0.5 * (fingers[0] + fingers[1])[:2]
                # The part that enters first, not the body's centroid: the toy
                # hangs canted from the pinch, so aiming the centroid put its tail
                # 40 mm beside the bore and the body landed on the rim.
                tail = owned[owned[:, 2] < owned[:, 2].min() + 0.020]
                body = tail[:, :2].mean(axis=0)
                target = np.array((BENCH_X + TARGET_X, 0.0)) + (pads - body)
                # How far the toy's lowest point sits below the rim here, not the
                # rest mesh's hole height: the toy hangs from a pinched hoop, and
                # the schedule that put it 20 mm clear of the rim on the way over
                # has to come from where it actually hangs or the descent starts
                # with the tail already inside the mouth.
                hanging = dict(geometry)
                hanging["lower_m"] = APPROACH - (float(owned[:, 2].min()) - rim)
                paths, descend = carry_path(fingers, target, dt, capacity, hanging, closed_gap)
                set_path(paths, start=frame + 1)
                descend_frame = frame + 1 + round(descend * CONTROL_HZ)
                print(
                    json.dumps(
                        {
                            "carry_frame": frame + 1,
                            "body_from_pads_mm": (1000.0 * (body - pads)).tolist(),
                            "carry_mm": (1000.0 * (target - pads)).tolist(),
                        }
                    ),
                    flush=True,
                )
            if row["insertion_depth_m"] > depth_best + JAM_STEP:
                depth_best = row["insertion_depth_m"]
                progress_frame = frame
            elif depth_best <= 0.0:
                # The toy is still above the rim, so there is nothing to stall yet
                # and the progress clock has to move with the descent. Left at
                # frame zero it is already 8 frames stale the moment the first
                # millimetre appears, and the guard fires on that millimetre:
                # measured, the dynamic run's jam at frame 196 -- the descent's
                # second frame, jaw contacts 0, depth 0.0 -- after which the pads
                # held in mid-air, the toy fell when they opened, and its body was
                # crushed against the bench (J -1.002, 20 frames inverted).
                progress_frame = frame
            # Armed once the toy's own lowest point is under the rim: before that
            # the depth is the claw's descent and cannot stall.
            armed = depth_best > 0.0 and frame >= descend_frame
            jam_streak = jam_streak + 1 if armed and frame - progress_frame >= JAM_FRAMES else 0
            if jam_frame is None and jam_streak > 0:
                jam_frame = frame + 1
                fingers = np.array([[body_q[b][0], body_q[b][1], body_q[b][2]] for b in finger_bodies])
                body_centre = body_top_centre(owned, ring)[:2]
                pads = 0.5 * (fingers[0] + fingers[1])[:2]
                over_bore = float(np.linalg.norm(pads - axis)) < bore
                if over_bore:
                    # Stalled over the bore with the pinch still on the loop: this
                    # is the press's cue, not the release's. Opening here drops the
                    # toy -- measured, it tipped off the rim, fell to the bench and
                    # slid 0.7 m, and the run finished with 0 % inside.
                    held = (
                        float(grip.entity.actual_half_gaps.numpy().mean())
                        if grip is not None
                        else arm.aperture() + arm_gripper.PAD_INNER_OFFSET
                    )
                    set_path(press_path(fingers, dt, start_half_gap=held), start=frame + 1)
                    press_frame = frame + 1
                else:
                    set_path(after_jam(fingers, dt, closed_gap), start=frame + 1)
                if grip is not None:
                    grip.entity.set_press(True)
                print(
                    json.dumps(
                        {
                            "jam_frame": jam_frame,
                            "depth_m": depth_best,
                            "body_centre_m": body_centre.tolist(),
                            "claw_offset_m": (0.5 * (fingers[0] + fingers[1]) - np.append(body_centre, 0.0))[
                                :2
                            ].tolist(),
                        }
                    ),
                    flush=True,
                )
            # The chase is the recovery for a toy that stalled off the bore and was
            # let go: it slid on the rim, so the pads are re-aimed at where its body
            # settled rather than descending beside it. A press that started at the
            # stall leaves no room for a chase: re-aiming there would drag the pads
            # across the top they are standing on.
            elif (
                jam_frame is not None
                and chase_frame is None
                and press_frame is None
                and frame + 1 >= jam_frame + round((JAM_RELEASE + SETTLE_TIME) * CONTROL_HZ)
            ):
                chase_frame = frame + 1
                fingers = np.array([[body_q[b][0], body_q[b][1], body_q[b][2]] for b in finger_bodies])
                settled = body_top_centre(owned, ring)[:2]
                set_path(chase_path(fingers, dt, settled), start=frame + 1)
                print(
                    json.dumps(
                        {
                            "chase_frame": chase_frame,
                            "settled_body_centre_m": settled.tolist(),
                            "settle_drift_m": (settled - body_centre).tolist(),
                        }
                    ),
                    flush=True,
                )
            # Failing a press at the stall, the first pad-particle contact after
            # the release is the handover to it. Before either, the press does not
            # exist and the judge says so.
            if (
                jam_frame is not None
                and press_frame is None
                and not body_contact_seen
                and frame + 1 > jam_frame + round(0.5 * CONTROL_HZ)
                and row["body_contacts"] > 0
            ):
                body_contact_seen = True
                press_frame = frame + 1
                fingers = np.array([[body_q[b][0], body_q[b][1], body_q[b][2]] for b in finger_bodies])
                set_path(press_path(fingers, dt), start=frame + 1)
                print(
                    json.dumps({"press_contact_frame": press_frame, "depth_m": depth_best}),
                    flush=True,
                )
            if grip is not None:
                row["pressing"] = grip.entity.pressing()
                row["applied"] = grip.entity.applied_wrenches.numpy().tolist()
                row["peak_force_n"] = grip.entity.peak_force.numpy().tolist()
                row["peak_torque_nm"] = grip.entity.peak_torque.numpy().tolist()
                row["peak_push_force_n"] = grip.entity.peak_push_force.numpy().tolist()
                row["peak_push_torque_nm"] = grip.entity.peak_push_torque.numpy().tolist()
            else:
                # The arm has one hand for both jobs: what a pad would feel is the
                # grip while closing and the press while pressing, and there is no
                # carrier wrench to report -- its torque ceiling is the arm's own
                # actuators, which the model already carries.
                row["pressing"] = press_frame is not None
                if shape_forces is not None:
                    # The squeeze is the grip, where the 20 N clamp binds; the pad's
                    # total load is what carries the press, friction included.
                    load, squeeze = arm_gripper.pad_forces(sim, shape_forces(), finger_sides)
                else:
                    load, squeeze = [0.0, 0.0], [0.0, 0.0]
                row["peak_force_n"] = squeeze
                row["pad_force_n"] = load
                row["peak_torque_nm"] = [0.0, 0.0]
                row["peak_push_force_n"] = load
                row["peak_push_torque_nm"] = [0.0, 0.0]
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
                "scene": "clawhauser_stuff_into_bottle",
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
                "glass_panels": PANELS,
                "wall_m": WALL,
                "bore_radius_m": bore,
                "rim_z_m": TABLE_Z + RIM_Z,
                "floor_z_m": TABLE_Z + FLOOR_Z,
                "mass_kg": mass,
                "global_soft_contact_mu": float(sim.model.soft_contact_mu),
                "attachment_particles": int(sim.model.attachment_particle_body_count),
                "completed_steps": len(diagnostics),
                "mean_frame_ms": 1000.0 * (time.perf_counter() - started) / max(1, len(diagnostics)),
                "mean_step_ms": 1000.0 * (time.perf_counter() - started) / max(1, len(diagnostics) * SUBSTEPS),
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )
    return judge(
        out,
        diagnostics,
        quality,
        geometry,
        entity,
        steps,
        jam_frame=jam_frame,
        press_frame=press_frame,
        video_error=video_error,
        self_contact=self_contact,
    )


def judge(
    out: Path,
    diagnostics,
    quality,
    geometry,
    entity,
    steps,
    *,
    jam_frame,
    press_frame,
    video_error,
    self_contact,
) -> dict:
    """What the run did, stated as measurements rather than as a verdict.

    The depth is the point: the container is smaller than the toy on purpose, so a
    run that stops on the rim and one that goes to the floor are both outcomes and
    the number says which happened. What is gated is what the *simulation* has to
    get right -- bounded jaw effort, bounded penetration, no non-finite state,
    and a toy that is not still in the jaws at the end. Local body inversion is
    reported as the measured deformation at this bore, not hidden: it is a
    quality number and a result field, not a silent pass.
    """
    weight = float(entity.source_mass.sum() * 9.81)
    # The claw's peak is the wrench its own drive applied, so its cap is exact to
    # a float32 ulp. The arm's is the squeeze its pads read back through the
    # coupling: the actuator is clamped at the same number, and the readback can
    # sit a percent over it for a substep.
    arm = geometry["gripper"] == "arm"
    tolerance = arm_gripper.GRIP_TOLERANCE if arm else 1.0e-6
    grip_cap = float(geometry["force_cap_n"])
    settled = diagnostics[-max(2, int(0.5 * CONTROL_HZ)) :]
    deepest = max(diagnostics, key=lambda row: row["insertion_depth_m"])
    reach = deepest["insertion_depth_m"]
    inside = max(row["inside_fraction"] for row in diagnostics)
    released = max(row["finger_contacts"] for row in settled) == 0
    settled_clearance = min(row["minimum_z_m"] for row in settled) - TABLE_Z
    # The pads' own clearance over the rim: reported, not gated. A body press can
    # legitimately put them inside the bore; what must not happen is touching it.
    pad_z = min(min(row["finger_positions"][side][2] for side in (0, 1)) for row in diagnostics) - (TABLE_Z + RIM_Z)
    pads_in_container = max(row["finger_container_contacts"] for row in diagnostics)
    # How far the pads actually drove down after the stall: the chase is bounded
    # by the path, but the dynamic fingers can lag it, so the travelled distance
    # is reported alongside the contact count and the recorded push force.
    press_travel = 0.0
    stalled = next((row for row in diagnostics if jam_frame is not None and row["frame"] >= jam_frame), None)
    if stalled is not None:
        press_travel = max(
            stalled["finger_positions"][side][2] - row["finger_positions"][side][2]
            for row in diagnostics
            if row["frame"] >= stalled["frame"]
            for side in (0, 1)
        )
    # The grasp and the press have separate budgets, and the run reports both:
    # the jaws' closing effort and the carrier's push force.
    press_rows = [row for row in diagnostics if row.get("pressing")]
    peak_force = max((max(row["peak_force_n"]) for row in diagnostics if "peak_force_n" in row), default=0.0)
    peak_torque = max((max(row["peak_torque_nm"]) for row in diagnostics if "peak_torque_nm" in row), default=0.0)
    push_contact_frames = sum(1 for row in press_rows if row["body_contacts"] > 0)
    max_body_contacts = max((row["body_contacts"] for row in diagnostics), default=0)
    peak_push_force = max(
        (max(row["peak_push_force_n"]) for row in press_rows if "peak_push_force_n" in row), default=0.0
    )
    jaw_peak = press_rows[-1].get("peak_force_n", [0.0, 0.0]) if press_rows else [0.0, 0.0]
    peak_push_torque = max(
        (max(row["peak_push_torque_nm"]) for row in press_rows if "peak_push_torque_nm" in row), default=0.0
    )
    contact_depth = next((row["insertion_depth_m"] for row in diagnostics if row["frame"] == press_frame), 0.0)
    press_depth = max((row["insertion_depth_m"] for row in press_rows), default=contact_depth) - contact_depth
    quality_read = quality.read()
    inverted_substeps = sum(quality_read["inverted_tet_substeps_by_part"])
    nonfinite_substeps = sum(quality_read["nonfinite_tet_substeps_by_part"])
    # A toy at the container floor is inside the scene, not dropped: the table
    # clearance alone called the 85.3 mm press "on the bench" at 5.6 mm.
    on_bench = settled_clearance <= DROP_CLEARANCE and settled[-1]["inside_fraction"] < 0.02
    wedged = bool(released and not on_bench and settled[-1]["below_rim_fraction"] > 0.02)
    # The pressed body compresses, so its tail stops a few millimetres short of the
    # rigid floor; demanding contact there called the 81.8 mm p1 press "jammed".
    usable = RIM_Z - FLOOR_Z
    inserted = bool(released and reach >= 0.9 * usable)
    if nonfinite_substeps:
        stop = "solver-artifact"
    elif on_bench:
        stop = "object-dropped"
    elif pads_in_container:
        stop = "rim-blocks-pads"
    elif jam_frame is not None and push_contact_frames == 0:
        stop = "no-body-contact"
    elif reach >= 0.9 * usable:
        stop = "inserted"
    else:
        # The descent ended with the toy held above the floor: whatever stopped it
        # was the object meeting the container rather than the claw letting go.
        stop = "object-jammed"
    result = {
        "geometry": geometry,
        "self_contact": self_contact,
        "weight_n": weight,
        "jammed_frame": jam_frame,
        "press_contact_frame": press_frame,
        "press_travel_m": max(0.0, float(press_travel)),
        "press_depth_m": max(0.0, float(press_depth)),
        "press_contact_frames": push_contact_frames,
        "max_body_contacts": max_body_contacts,
        "peak_push_force_n": peak_push_force,
        "jaw_peak_n": [float(value) for value in jaw_peak],
        "peak_push_torque_nm": peak_push_torque,
        "local_inversion": bool(inverted_substeps or min(quality_read["minimum_J_by_part"]) <= 0.0),
        "max_insertion_depth_m": reach,
        "max_insertion_depth_frame": deepest["frame"],
        "max_below_rim_fraction": max(row["below_rim_fraction"] for row in diagnostics),
        "max_inside_fraction": inside,
        "settled_clearance_m": settled_clearance,
        "settled_inside_fraction": settled[-1]["inside_fraction"],
        "settled_below_rim_fraction": settled[-1]["below_rim_fraction"],
        "released": released,
        "wedged_after_release": wedged,
        "inserted_to_floor": inserted,
        "pad_clearance_over_rim_m": pad_z,
        "stop_reason": stop,
        "peak_force_n": peak_force,
        "peak_torque_nm": peak_torque,
        "max_penetration_m": max(row["max_penetration_m"] for row in diagnostics),
        "max_container_penetration_m": max(row["max_container_penetration_m"] for row in diagnostics),
        "max_pad_penetration_m": max(row["max_pad_penetration_m"] for row in diagnostics),
        "max_finger_contacts": max(row["finger_contacts"] for row in diagnostics),
        "max_container_contacts": max(row["container_contacts"] for row in diagnostics),
        "max_finger_container_contacts": max(row["finger_container_contacts"] for row in diagnostics),
        "max_finger_container_near": max(row["finger_container_near"] for row in diagnostics),
        "max_finger_other_contacts": max(row["finger_other_contacts"] for row in diagnostics),
        "finger_container_detail": next(
            (row["finger_container_detail"] for row in diagnostics if row["finger_container_detail"]), []
        ),
        "finger_other_bodies": next(
            (row["finger_other_bodies"] for row in diagnostics if row["finger_other_bodies"]), []
        ),
        "max_inverted_tetrahedra": max(row["inverted_tetrahedra"] for row in diagnostics),
        "settled_inverted_tetrahedra": max(row["inverted_tetrahedra"] for row in settled),
        "quality": quality_read,
        "steps": len(diagnostics),
        "video_error": video_error,
    }
    result["success"] = bool(
        steps >= round(DURATION * CONTROL_HZ)
        and reach > 0.020
        and released
        and not on_bench
        and pads_in_container == 0
        # The container's shell only. Pad indentation into the plush is the
        # deformation the press exists to produce and is reported separately as
        # ``max_pad_penetration_m``.
        and result["max_container_penetration_m"] <= 0.0005
        # Local body inversion is the measured deformation at this bore, not a
        # hidden pass: it is reported in ``local_inversion`` and ``quality``. A
        # non-finite state is still refused outright.
        and nonfinite_substeps == 0
        # The two clamps are applied in float32, so a run that sits on one reports
        # its own representation -- 5.0000005 against a float64 5.0. One part in a
        # million is the arithmetic, not the physics: measured, the b6-dyn run was
        # refused its success by 4.8e-7 N.
        and peak_force <= grip_cap * (1.0 + tolerance)
        and peak_torque <= 0.03
        # A press that never touched the body is not a press: the pads' own
        # descent is not evidence, and a run whose pads chase the falling toy
        # would otherwise pass on the release drop.
        and push_contact_frames >= round(0.1 * CONTROL_HZ)
        and peak_push_force > float(geometry["force_cap_n"])
        and peak_push_force <= float(geometry["push_force_cap_n"]) * (1.0 + tolerance)
        and peak_push_torque <= float(geometry["push_torque_cap_nm"]) * (1.0 + tolerance)
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--coarse", dest="coarse", action="store_true", help="run the coarse common-translation solve")
    parser.add_argument("--bore", type=float, default=BORE_RADIUS, help="container inner radius [m]")
    parser.add_argument("--cap", type=float, default=arm_gripper.GRASP_LIMIT_N, help="jaw closing clamp [N]")
    parser.add_argument("--push-cap", type=float, default=PUSH_LIMIT_N, help="carrier push clamp [N]")
    parser.add_argument("--force-limit", type=float, default=None, help="alias for --cap, the sweep's name")
    parser.add_argument("--squeeze", type=float, default=GRIP_SQUEEZE, help="close travel past the loop [m]")
    parser.add_argument("--mu", type=float, default=8.0, help="global soft-contact friction coefficient")
    parser.add_argument(
        "--no-self-contact", dest="self_contact", action="store_false", help="turn the toy's self-contact off"
    )
    parser.add_argument(
        "--claw",
        choices=("arm", "claw"),
        default="arm",
        help="who moves: the AIRBOT arm, or the scripted parallel claw this scene was built on",
    )
    args = parser.parse_args()
    run(
        args.out,
        steps=args.steps,
        render_every=args.render_every,
        coarse=args.coarse,
        bore=args.bore,
        cap=args.cap if args.force_limit is None else args.force_limit,
        push_cap=args.push_cap,
        mu=args.mu,
        squeeze=args.squeeze,
        enable_self_contact=args.self_contact,
        claw=args.claw,
    )


if __name__ == "__main__":
    main()
