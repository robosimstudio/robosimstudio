# SPDX-License-Identifier: Apache-2.0
"""Pick up the existing labelled Clawhauser volume by its loop and place it on a table.

Run on a CUDA node with Newton 1.8 development and RoboSimStudio on PYTHONPATH.
The two kinematic fingers use only collision contact; no particle attachment or
direct object motion is added. Outputs are written under ``outputs/clawhauser``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import newton
import numpy as np
from gripper import ScriptedGripperSpec

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec, load_labelled_tets

ASSET = Path(__file__).resolve().parent / "assets" / "clawhauser"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "clawhauser"
CONTROL_HZ = 30
SUBSTEPS = 20
DURATION = 13.0
TABLE_Z = 0.75
# RSS positions are relative to the bench's work-surface frame, whose origin
# is world x=0.6 m. These values place the toy at world x=0.47 and 0.65 m.
SOURCE_X = -0.13
TARGET_X = 0.05
SOURCE_HASHES = {
    "final_mesh.mesh": "df3d1505d96d36fbfa76291e7ab94269a30500ba5e4b87516d0bf8f57f0c6458",
    "heterogeneous_params.npz": "7cfc8ca3cc7c08d261aa90b08832230801fec1ac62848d0e9239d9e2f318e8c6",
    "inferred_params.json": "45aacd698f2b38a6fd8bf13b46cc9c33497fc90c6c1bdf088ed12d915ec3d992",
    "appearance/visual_to_physics.npz": "4b015d1141794211e4e2c6e2b39fadef6762a73ffe2d76b921e47d5676f3d6c4",
}


def build_scene():
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

    # A flat pose makes the loop accessible and the plush body stable on the
    # table. The asset's local +y points toward its loop; rotate 180° about x.
    pose = (SOURCE_X, 0.0, float(vertices[:, 2].max() + 0.002))
    rotation = (1.0, 0.0, 0.0, 0.0)
    with np.load(ASSET / "appearance" / "visual_to_physics.npz") as binding:
        faces = binding["visual_faces"][binding["boundary_face_part_ids"] == 2]
        ring_surface = np.unique(binding["physics_vertex_indices"][faces])
    grasp_ids = ring_surface[vertices[ring_surface, 1] > 0.095]
    if len(grasp_ids) < 8:
        raise ValueError("Clawhauser loop grasp patch is missing")
    local_center = vertices[grasp_ids].mean(axis=0)
    grasp_center = (SOURCE_X + local_center[0], -local_center[1], pose[2] - local_center[2])
    half_span = np.max(np.abs(vertices[grasp_ids, 0] - local_center[0]))

    vbd = rss.backend(
        "vbd",
        iterations=12,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
        rigid_body_particle_contact_buffer_size=4096,
    )
    physics = rss.ArrangementSpec(
        control_hz=CONTROL_HZ,
        substeps=SUBSTEPS,
        use_cuda_graph=False,
        backends=(("soft", vbd), ("rigid", vbd), ("*", vbd)),
        collision=rss.CollisionSpec(
            soft_contact_margin=0.002,
            full_surface_contact=False,
            contact_ke=100000.0,
            contact_kd=25.0,
            broad_phase="nxn",
        ),
    )
    scene = rss.Scene(cell="bench_075", physics=physics, name="clawhauser_loop_pick_place")
    scene.add(
        rss.slab(
            name="source_zone",
            half=(0.06, 0.06, 0.00025),
            at=(SOURCE_X, 0.0, 0.001),
            collide=False,
            color=(0.13, 0.28, 0.43),
        )
    )
    scene.add(
        rss.slab(
            name="target_zone",
            half=(0.06, 0.06, 0.00025),
            at=(TARGET_X, -0.10, 0.001),
            collide=False,
            color=(0.10, 0.38, 0.28),
        )
    )
    obj = scene.add(
        HeterogeneousSoftSpec(
            name="clawhauser",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            quat=rotation,
            contact_ke=100000.0,
        )
    )
    grip = scene.add(
        ScriptedGripperSpec(
            center=grasp_center,
            half_gap=float(half_span + 0.012),
            closed_half_gap=0.010,
            travel_x=TARGET_X - SOURCE_X,
            control_hz=CONTROL_HZ,
            substeps=SUBSTEPS,
            duration=DURATION,
        )
    )
    camera = rss.CameraSpec(
        name="overview",
        width=640,
        height=480,
        eye=(0.90, -0.50, 1.30),
        look_at=(0.57, -0.04, 0.95),
        fov_y=50.0,
    )
    scene.add(camera)
    sim = scene.build(device="cuda:0", warmup=False, use_cuda_graph=False)
    entity = obj.entity
    if type(sim.pipeline.program.solver).__name__ != "SolverVBD":
        raise RuntimeError(f"expected Newton VBD, got {type(sim.pipeline.program.solver).__name__}")
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
        raise RuntimeError("Newton nodal mass differs from source per-tet densities")
    return sim, obj, grip, camera, grasp_ids, labels


def run(out: Path, *, steps: int, render_every: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, grip, camera, ring_ids, labels = build_scene()
    entity = obj.entity
    particle_ids = entity.particles
    ring_ids = particle_ids[ring_ids]
    body_ids = particle_ids[np.unique(entity.source_tets[labels == 1])]
    q0 = sim.pipeline.state_0.particle_q.numpy()[particle_ids]
    rest_det = np.linalg.det(q0[entity.source_tets[:, 1:]] - q0[entity.source_tets[:, :1]])
    if np.any(rest_det <= 0):
        raise RuntimeError("source tetrahedra were not positively oriented after build")
    initial_body = q0[np.unique(entity.source_tets[labels == 1])].mean(axis=0)
    positions, body_poses, diagnostics = [], [], []

    videos = None
    video_error = None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=max(1, CONTROL_HZ // render_every))
        videos.__enter__()
    try:
        for frame in range(steps):
            sim.run(1).sync()
            q = sim.pipeline.state_0.particle_q.numpy()
            body_q = sim.pipeline.state_0.body_q.numpy()
            owned = q[particle_ids]
            if not np.isfinite(owned).all():
                raise RuntimeError(f"non-finite Clawhauser state at frame {frame}")
            det = np.linalg.det(owned[entity.source_tets[:, 1:]] - owned[entity.source_tets[:, :1]])
            contacts = sim.pipeline.contacts
            count = int(contacts.soft_contact_count.numpy()[0])
            contact_shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            finger_mask = np.isin(contact_shapes, grip.entity.finger_shapes)
            finger_contacts = int(np.count_nonzero(finger_mask))
            contact_particles = contacts.soft_contact_particle.numpy()[: len(contact_shapes)]
            ring_mask = np.isin(contact_particles, ring_ids)
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "ring_center_m": q[ring_ids].mean(axis=0).tolist(),
                "body_center_m": q[body_ids].mean(axis=0).tolist(),
                "body_translation_x_m": float(q[body_ids, 0].mean() - initial_body[0]),
                "minimum_z_m": float(owned[:, 2].min()),
                "minimum_tet_volume_ratio": float((det / rest_det).min()),
                "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
                "soft_contacts": count,
                "finger_contacts": finger_contacts,
                "finger_contacts_by_side": [
                    int(np.count_nonzero(np.isin(contact_shapes, shapes))) for shapes in grip.entity.shapes_by_side
                ],
                "ring_contacts_by_side": [
                    int(np.count_nonzero(np.isin(contact_shapes, shapes) & ring_mask))
                    for shapes in grip.entity.shapes_by_side
                ],
                "grasp_patch_contacts": int(np.count_nonzero(finger_mask & ring_mask)),
                "gripper_body_x_m": body_q[grip.entity.bodies, 0].tolist(),
                "gripper_body_y_m": body_q[grip.entity.bodies, 1].tolist(),
                "gripper_body_z_m": body_q[grip.entity.bodies, 2].tolist(),
            }
            diagnostics.append(row)
            positions.append(owned.astype(np.float32))
            body_poses.append(body_q.astype(np.float32))
            if videos is not None and frame % render_every == 0:
                try:
                    videos.append_frames({camera.name: sim.render(camera)})
                except Exception as exc:
                    video_error = repr(exc)
                    try:
                        videos.__exit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass
                    videos = None
            if frame % 15 == 0 or frame == steps - 1:
                print(json.dumps(row), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)

    np.savez_compressed(out / "states.npz", particle_q=np.asarray(positions), body_q=np.asarray(body_poses))
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    # A staged Newton -- an rsync'd source tree, which is how the cluster runs
    # it -- is not a git checkout, and an absent commit is not a failed run.
    newton_root = Path(newton.__file__).resolve().parents[1]
    try:
        newton_commit = subprocess.check_output(["git", "-C", str(newton_root), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        newton_commit = f"unknown ({newton_root} is not a git checkout)"
    manifest = {
        "asset": "clawhauser",
        "newton_version": newton.__version__,
        "newton_source_commit": newton_commit,
        "solver": type(sim.pipeline.program.solver).__name__,
        "tetrahedra": int(sim.model.tet_count),
        "part_tet_counts": {str(i): int(np.count_nonzero(labels == i)) for i in np.unique(labels)},
        "source_young_pa": sorted(float(v) for v in np.unique(entity.source_elastic[:, 0])),
        "source_poisson": sorted(float(v) for v in np.unique(entity.source_elastic[:, 1])),
        "source_density_kg_m3": sorted(float(v) for v in np.unique(entity.source_density)),
        "global_soft_contact_mu": float(sim.model.soft_contact_mu),
        "grasp_attachments": int(sim.model.attachment_particle_body_count),
        "completed_steps": len(diagnostics),
        "video_error": video_error,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if diagnostics:
        final = diagnostics[-1]
        transit = [row for row in diagnostics if 3.5 <= row["time_s"] <= 5.5]
        settled = [row for row in diagnostics if row["time_s"] >= 12.1]
        lifted = float(max((row["body_center_m"][2] - initial_body[2] for row in transit), default=0.0))
        clearance = float(max((row["minimum_z_m"] - TABLE_Z for row in transit), default=0.0))
        peak_finger_contacts = max(row["finger_contacts"] for row in diagnostics)
        target_error_x = float(abs(final["body_center_m"][0] - (initial_body[0] + TARGET_X - SOURCE_X)))
        body_drift = (
            float(np.linalg.norm(np.asarray(final["body_center_m"]) - settled[0]["body_center_m"])) if settled else None
        )
        finger_distance = min(
            float(
                np.linalg.norm(
                    np.asarray(final["ring_center_m"])
                    - np.asarray(
                        [final["gripper_body_x_m"][i], final["gripper_body_y_m"][i], final["gripper_body_z_m"][i]]
                    )
                )
            )
            for i in range(2)
        )
        bilateral_ring_grasp = bool(transit and all(min(row["ring_contacts_by_side"]) > 0 for row in transit))
        released = bool(settled and max(row["finger_contacts"] for row in settled) == 0)
        table_supported = bool(
            TABLE_Z - 0.002 <= final["minimum_z_m"] <= TABLE_Z + 0.004
            and final["soft_contacts"] > final["finger_contacts"]
        )
        max_inverted = max(row["inverted_tetrahedra"] for row in diagnostics)
        settled_inverted = max((row["inverted_tetrahedra"] for row in settled), default=None)
        max_inverted_fraction = max_inverted / len(labels)
        success = (
            steps >= round(DURATION * CONTROL_HZ)
            and lifted > 0.08
            and clearance > 0.05
            and bilateral_ring_grasp
            and target_error_x < 0.04
            and released
            and table_supported
            and body_drift is not None
            and body_drift < 0.02
            and finger_distance > 0.08
            and settled_inverted == 0
            and max_inverted_fraction < 0.002
        )
        result = {
            "transit_body_lift_m": lifted,
            "transit_clearance_m": clearance,
            "body_translation_x_m": final["body_translation_x_m"],
            "target_error_x_m": target_error_x,
            "peak_finger_contacts": peak_finger_contacts,
            "bilateral_ring_grasp": bilateral_ring_grasp,
            "released": released,
            "table_supported": table_supported,
            "post_release_body_drift_m": body_drift,
            "final_finger_ring_distance_m": finger_distance,
            "max_inverted_tetrahedra": max_inverted,
            "max_inverted_fraction": max_inverted_fraction,
            "settled_inverted_tetrahedra": settled_inverted,
            "success": bool(success),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    args = parser.parse_args()
    run(args.out, steps=args.steps, render_every=args.render_every)


if __name__ == "__main__":
    main()
