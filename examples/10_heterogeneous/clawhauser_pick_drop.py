# SPDX-License-Identifier: Apache-2.0
"""Pick the labelled Clawhauser volume up by its loop, carry it, and drop it.

The toy starts lying on its lowest face, loop flat, and the parallel claw closes
on the loop's rim, lifts it clear, carries it 80 mm, opens and lets it fall. The run measures whether the grip held, how far
the toy moved, whether it came to rest on the bench, and what the mesh did.

The claw is the same mechanism as the two longer scenes: a level carrier on a
prismatic guide, two jaws on native prismatic slides, a 5 N closing clamp. The
carry is the carrier following its path; there is no press in this task.

Geometry is measured from the mesh and reported, so the grasp point and the
lift are stated rather than tuned.

Run on a CUDA node with Newton 1.8 development and RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from loop_geometry import ring_geometry, smooth
from mesh_quality import MeshQuality, penetration
from parallel_gripper import GRASP_LIMIT_N, ParallelGripperSpec

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec, load_labelled_tets

ASSET = Path(__file__).resolve().parent / "assets" / "clawhauser"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "clawhauser_pick_drop"
CONTROL_HZ = 30
SUBSTEPS = 20
# 20 s: the phases below still end at 5.2 s, so the drop is unchanged and the
# remaining 11 s is the toy at rest. That is the window a drift is measured in.
DURATION = 20.0
TABLE_Z = 0.75
# The move: 80 mm along the bench, and a lift that takes the toy's own length
# clear of the surface it is resting on. The toy hangs from the pinch with its
# tail about the loop's height (197 mm) below the pads, so a 210 mm lift is what
# leaves the tail 40 mm off the bench before the carry starts.
SOURCE_X = -0.13
TARGET_X = -0.05
LIFT_M = 0.210
CONTACT_GAP = 0.02
CONTACT_KE = 100000.0
COARSE_BUDGET = 0.005
GRIP_SQUEEZE = 0.006
"""How far past the loop's measured width the closed pads aim [m]. The same value
the rod and cup scenes use: the pads close on the cord's rim from either side and
a deep overshoot is what holds it.
"""
PAD_HALF_THICKNESS = 0.003
PAD_HALF_WIDTH = 0.006
PAD_CLEARANCE = 0.001
"""How far the pads' bottom faces start above the bench [m]. The close starts at
the ground, so a loop lying there is straddled instead of one assumed to sit at
the body's mid-plane."""
OPENED_HALF_GAP = 0.034
"""Where the jaws open to release: 68 mm, well outside the loop's 23 mm reach."""
PHASES = {
    # Closed as early as the control loop allows: lying flat, the loop is a limp
    # hoop and the toy slides across the bench within half a second of the scene
    # starting (measured, p3: the pads closed 20 mm behind the hoop and lifted
    # nothing).
    "close": (0.0, 0.6),
    # The pads hold still once they are closed: the toy settles onto the pinch
    # before the lift loads it.
    "lift": (1.1, 2.7),
    "hang": (2.7, 3.5),
    "carry": (3.5, 4.7),
    "open": (5.3, 5.7),
}
"""Planned phase windows. ``hang`` is a phase with no motion: the toy comes off
the bench at an angle and swings to hang from the pinch before it is carried.
"""
SOURCE_HASHES = {
    "final_mesh.mesh": "df3d1505d96d36fbfa76291e7ab94269a30500ba5e4b87516d0bf8f57f0c6458",
    "heterogeneous_params.npz": "7cfc8ca3cc7c08d261aa90b08832230801fec1ac62848d0e9239d9e2f318e8c6",
    "inferred_params.json": "45aacd698f2b38a6fd8bf13b46cc9c33497fc90c6c1bdf088ed12d915ec3d992",
    "appearance/visual_to_physics.npz": "4b015d1141794211e4e2c6e2b39fadef6762a73ffe2d76b921e47d5676f3d6c4",
}


def build_scene(*, coarse: bool, self_contact: bool = True):
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

    # Lying flat in its own rest pose, grasped on the loop's rim: the same pose and
    # grasp the cup scene uses. The toy's lowest vertex is what rests it on the
    # bench, and the loop is a ring at its end lying in a horizontal plane.
    quat = (0.0, 0.0, 0.0, 1.0)
    hole = ring["hole_centre_local"]
    base = ring["base_local_y"]
    pose = (SOURCE_X, float(-hole[1]), float(-vertices[:, 2].min()) + 0.001)
    grip_z = float(ParallelGripperSpec.pad_half_height + PAD_CLEARANCE)
    grip = (SOURCE_X + float(hole[0]), 0.0, grip_z)
    # The pads start at the bench, not at the loop's modelled height, so the lift
    # grows by that drop and the tail still leaves the bench by the same amount.
    lift = float(LIFT_M + (pose[2] + hole[2]) - grip_z)
    hole_height = float(hole[1] - base)
    geometry = {
        "hole_centre_local": np.round(hole, 4).tolist(),
        "loop_reach_m": ring["side_reach"],
        "pad_half_thickness_m": PAD_HALF_THICKNESS,
        "grip_squeeze_m": GRIP_SQUEEZE,
        "grip_z_m": grip_z,
        "hole_height_m": hole_height,
        "lift_m": lift,
        "carry_m": TARGET_X - SOURCE_X,
        "force_cap_n": GRASP_LIMIT_N,
        "resting_pose": "lying flat, no rotation",
    }
    print(json.dumps({"geometry": geometry}), flush=True)

    common = {
        "iterations": 12,
        "particle_enable_self_contact": self_contact,
        "particle_self_contact_radius": 0.002,
        "particle_self_contact_margin": 0.002,
        "particle_topological_contact_filter_threshold": 2,
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
            **common,
        )
        if coarse
        else rss.backend("vbd", **common)
    )
    physics = rss.ArrangementSpec(
        control_hz=CONTROL_HZ,
        substeps=SUBSTEPS,
        use_cuda_graph=False,
        backends=(("soft", vbd), ("rigid", vbd), ("*", vbd)),
        collision=rss.CollisionSpec(
            soft_contact_margin=CONTACT_GAP,
            full_surface_contact=False,
            contact_ke=CONTACT_KE,
            contact_kd=25.0,
            broad_phase="nxn",
        ),
    )
    scene = rss.Scene(cell="bench_075", physics=physics, name="clawhauser_pick_drop")
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
    jaw = scene.add(
        ParallelGripperSpec(
            name="pick_drop_gripper",
            center=grip,
            half_gap=ring["side_reach"] + PAD_HALF_THICKNESS + 0.012,
            closed_half_gap=ring["side_reach"] + PAD_HALF_THICKNESS - GRIP_SQUEEZE,
            pad_half_thickness=PAD_HALF_THICKNESS,
            pad_half_width=PAD_HALF_WIDTH,
            duration=DURATION,
            control_hz=CONTROL_HZ,
            substeps=SUBSTEPS,
        )
    )
    camera_spec = rss.CameraSpec(
        name="overview",
        width=960,
        height=540,
        # Framed for the lying start: the toy rests toward the camera, so the eye
        # is pulled back and the view is aimed between the toy's rest and the place
        # it is carried to.
        eye=(0.835, -0.89, 1.476),
        look_at=(0.51, -0.05, 0.95),
        fov_y=45.0,
    )
    camera = scene.add(camera_spec)
    sim = scene.build(device="cuda:0", warmup=False, use_cuda_graph=False)
    entity = obj.entity
    if not type(sim.pipeline.program.solver).__name__.startswith("SolverVBD"):
        raise RuntimeError(f"expected a VBD solver, got {type(sim.pipeline.program.solver).__name__}")
    if sim.model.tet_count != len(tets) or len(entity.tet_part_labels) != len(tets):
        raise RuntimeError("RSS lost Clawhauser tetrahedra or part labels")
    if sim.model.attachment_particle_body_count:
        raise RuntimeError("the contact grasp scene must not contain particle attachments")
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
        raise RuntimeError("Newton nodal mass differs from the source per-tet densities")
    return sim, obj, jaw, camera, geometry


def waypoints(grip, capacity: int, dt: float, lift: float) -> np.ndarray:
    """The claw's own path: close, lift, hang, carry, open.

    The pads close on the loop where it lies, the lift takes the toy off the bench,
    the hang lets it swing to hang from the pinch, the carry moves the claw, and the
    open is what lets it fall.
    """
    centre = np.asarray(grip.entity.center, dtype=np.float64)
    opened = float(grip.spec.half_gap)
    closed = float(grip.spec.closed_half_gap)
    poses = np.zeros((capacity, 2, 7), dtype=np.float32)
    poses[:, :, 3:] = (0.0, 0.0, 0.0, 1.0)
    for index in range(capacity):
        time = index * dt
        gap = opened + (closed - opened) * smooth(time, *PHASES["close"])
        gap += (OPENED_HALF_GAP - gap) * smooth(time, *PHASES["open"])
        place = centre + np.array(
            [
                (TARGET_X - SOURCE_X) * smooth(time, *PHASES["carry"]),
                0.0,
                lift * smooth(time, *PHASES["lift"]),
            ]
        )
        poses[index, 0, :3] = place + np.array([-gap, 0.0, 0.0])
        poses[index, 1, :3] = place + np.array([gap, 0.0, 0.0])
    return poses


def judge(geometry: dict, diagnostics, quality, entity, steps: int) -> dict:
    final = diagnostics[-1]
    start_x = diagnostics[0]["body_centre_m"][0]
    # Where the toy was carried to, at the last frame the pads held it: after the
    # release it rolls, and where it rolls to is not what the carry did.
    held = max((row["frame"] for row in diagnostics if row["finger_contacts"]), default=1)
    moved = float(diagnostics[held - 1]["body_centre_m"][0] - start_x)
    rolled = float(final["body_centre_m"][0] - diagnostics[held - 1]["body_centre_m"][0])
    # How much clearance the toy gained at its lowest point: the toy hangs from
    # the pinch, so its centre drops as it swings, but its tail leaving the bench
    # is what says it was picked up.
    lifted = float(max(row["minimum_z_m"] for row in diagnostics) - diagnostics[0]["minimum_z_m"])
    tail = diagnostics[-round(0.5 * CONTROL_HZ) :]
    released = max(row["finger_contacts"] for row in tail) == 0
    clearance = float(min(row["minimum_z_m"] for row in tail) - TABLE_Z)
    # Touch-down: the first frame after the pads let go whose lowest point is on
    # the bench. Drift is measured from there, not from the release, because the
    # fall itself is not drift.
    landing = next(
        (row for row in diagnostics if row["frame"] > held and row["minimum_z_m"] - TABLE_Z <= 0.002),
        None,
    )
    landed_frame = None if landing is None else landing["frame"]
    drift = None
    if landing is not None:
        drift = float(
            np.linalg.norm(np.asarray(final["body_centre_m"][:2]) - np.asarray(landing["body_centre_m"][:2]))
        )
    # Mean speed of the centre over the last half second of the run.
    span = min(len(diagnostics) - 1, round(0.5 * CONTROL_HZ))
    rest_speed = 0.0
    if span >= 1:
        rest_speed = float(
            np.linalg.norm(np.asarray(final["body_centre_m"]) - np.asarray(diagnostics[-1 - span]["body_centre_m"]))
            / (final["time_s"] - diagnostics[-1 - span]["time_s"])
        )
    max_finger_contacts = max(row["finger_contacts"] for row in diagnostics)
    quality_read = quality.read()
    inverted_substeps = sum(quality_read["inverted_tet_substeps_by_part"])
    max_inverted = max(row["inverted_tetrahedra"] for row in diagnostics)
    max_penetration = float(np.max(np.concatenate([row["penetration_m"] for row in diagnostics] + [np.zeros(1)])))
    if not released:
        stop = "held"
    elif lifted < 0.5 * geometry["lift_m"]:
        stop = "not-lifted"
    else:
        stop = "dropped"
    result = {
        "geometry": geometry,
        "weight_n": float(entity.source_mass.sum() * 9.81),
        "stop_reason": stop,
        "released": bool(released),
        "lifted_m": lifted,
        "moved_x_m": moved,
        "rolled_after_release_m": rolled,
        "landed_frame": landed_frame,
        "drift_after_landing_m": drift,
        "rest_speed_m_s": rest_speed,
        "settled_clearance_m": clearance,
        "max_finger_contacts": max_finger_contacts,
        "max_inverted_tetrahedra": max_inverted,
        "local_inversion": bool(inverted_substeps or min(quality_read["minimum_J_by_part"]) <= 0.0),
        "max_penetration_m": max_penetration,
        "quality": quality_read,
        "steps": steps,
        "success": bool(
            released
            and abs(moved - (TARGET_X - SOURCE_X)) <= 0.020
            and -0.0005 <= clearance <= 0.002
            and max_inverted == 0
            and max_penetration <= 0.0005
        ),
    }
    return result


def run(
    out: Path, *, steps: int, render_every: int = 0, coarse: bool = True, enable_self_contact: bool = True
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, jaw, camera, geometry = build_scene(coarse=coarse, self_contact=enable_self_contact)
    entity = obj.entity
    particles = entity.particles
    tets = entity.source_tets
    solver = sim.pipeline.program.solver
    contacts = sim.pipeline.contacts
    # Self-contact pairs are written by whichever detector the solver built. This
    # scene drives detection from RSS, so the solver owns a tri-mesh detector and
    # its pairs are NOT in ``contacts.soft_self_contact_data`` (allocated before
    # the solver existed, and therefore without the struct). Read the buffer that
    # is actually populated, and record which one it was.
    pair_info, pair_buffer = None, "absent"
    for name, info in (
        ("solver.detector", getattr(solver, "trimesh_collision_info", None)),
        ("solver.contacts", getattr(getattr(solver, "contacts", None), "soft_self_contact_data", None)),
        ("pipeline.contacts", getattr(contacts, "soft_self_contact_data", None)),
    ):
        if info is not None:
            pair_info, pair_buffer = info, name
            break
    self_contact = {
        "enabled": bool(getattr(solver, "particle_enable_self_contact", False)),
        "margin_m": float(getattr(solver, "particle_self_contact_margin", 0.0)),
        "rest_exclusion_m": float(getattr(solver, "particle_rest_shape_contact_exclusion_radius", 0.0)),
        "pair_buffer": pair_buffer,
    }
    if self_contact["enabled"] != enable_self_contact:
        raise RuntimeError("the solver's self-contact setting does not match the scene's")
    if self_contact["enabled"] and pair_info is None:
        raise RuntimeError("self-contact is on but no pair buffer was found; the counts would read zero")
    # The block's own motion, cumulative since the last read: ``path_length`` is
    # cleared with the substep budget, so reading it here would only see the
    # last of the step's 20 substeps.
    block = getattr(solver, "coarse_block", None)
    quality = MeshQuality(sim, entity)
    dt = 1.0 / (CONTROL_HZ * SUBSTEPS)
    q0 = sim.pipeline.state_0.particle_q.numpy()[particles]
    jaw.entity.set_trajectory(waypoints(jaw, jaw.entity.capacity, dt, geometry["lift_m"]))
    rest_det = np.linalg.det(q0[tets[:, 1:]] - q0[tets[:, :1]])
    if np.any(rest_det <= 0):
        raise RuntimeError("source tetrahedra were not positively oriented after build")
    mass = float(entity.source_mass.sum())
    start = q0.mean(axis=0)
    print(json.dumps({"mass_kg": mass, "weight_n": mass * 9.81, "start_centre_m": start.tolist()}), flush=True)

    diagnostics = []
    videos, video_error = None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        # Written at 60 fps: each rendered control frame fills a 60th times the
        # render interval, so the video runs at the scene's own speed.
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q = state.particle_q.numpy()
            body_q = state.body_q.numpy()
            owned = q[particles]
            det = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]])
            count = int(contacts.soft_contact_count.numpy()[0])
            contact_shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            depth = penetration(q, body_q, sim.model, contacts)
            touching = depth > 0.0
            finger = np.isin(contact_shapes, jaw.entity.finger_shapes)
            pairs, nearest = 0, None
            if pair_info is not None:
                # Counts record every detected pair, including a row that
                # overflowed its capacity; ``min_dist`` only means something on
                # rows that found one, and starts at zero everywhere else.
                for counts, distances in (
                    (
                        pair_info.vertex_colliding_triangles_count.numpy(),
                        pair_info.vertex_colliding_triangles_min_dist.numpy(),
                    ),
                    (
                        pair_info.edge_colliding_edges_count.numpy(),
                        pair_info.edge_colliding_edges_min_dist.numpy(),
                    ),
                ):
                    pairs += int(counts.sum())
                    hit = counts > 0
                    if hit.any():
                        found = float(distances[hit].min())
                        nearest = found if nearest is None else min(nearest, found)
            coarse_path = None
            if block is not None:
                coarse_path = float(block.step_path.numpy()[0])
                block.step_path.zero_()
            row = {
                "self_contact_pairs": pairs,
                "self_contact_min_dist_m": nearest,
                "coarse_path_m": coarse_path,
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "body_centre_m": owned.mean(axis=0).tolist(),
                "minimum_z_m": float(owned[:, 2].min()),
                "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
                "finger_contacts": int(np.count_nonzero(finger & touching)),
                "penetration_m": depth,
            }
            diagnostics.append(row)
            if videos is not None and frame % render_every == 0:
                videos.append_frames({camera.name: sim.render(camera)})
            if frame % 15 == 0 or frame == steps - 1:
                print(json.dumps({k: v for k, v in row.items() if k != "penetration_m"}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)

    (out / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    # The per-frame penetration arrays stay in memory for the judge; the file
    # keeps the scalars, since a list of arrays is not JSON.
    (out / "diagnostics.json").write_text(
        json.dumps([{k: v for k, v in row.items() if k != "penetration_m"} for row in diagnostics], indent=2)
        + "\n"
    )
    result = judge(geometry, diagnostics, quality, entity, steps)
    result["self_contact"] = self_contact
    result["video_error"] = video_error
    result["mean_frame_ms"] = 1000.0 * (time.perf_counter() - started) / max(1, len(diagnostics))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "quality")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--no-coarse", action="store_true", help="stock VBD, no common-translation solve")
    parser.add_argument(
        "--no-self-contact", dest="self_contact", action="store_false", help="turn the toy's self-contact off"
    )
    args = parser.parse_args()
    run(
        args.out,
        steps=args.steps,
        render_every=args.render_every,
        coarse=not args.no_coarse,
        enable_self_contact=args.self_contact,
    )


if __name__ == "__main__":
    main()
