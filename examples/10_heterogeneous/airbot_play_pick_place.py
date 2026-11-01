# SPDX-License-Identifier: Apache-2.0
"""AIRBOT Play contact manipulation of the original labelled Clawhauser volume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
from clawhauser_pick_place import ASSET, CONTROL_HZ, SOURCE_HASHES, SOURCE_X, SUBSTEPS, TABLE_Z, TARGET_X
from gripper import smooth

import robosimstudio as rss
from robosimstudio.robots.registry import register_gripper
from robosimstudio.specs.robot import GripperSpec
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec, load_labelled_tets

AIRBOT_ASSET = Path(__file__).resolve().parent / "assets" / "airbot_play" / "airbot_play.xml"
AIRBOT_HOME = (0.0, -1.0, 1.2, 1.5708, -1.2, -1.5708)

register_gripper(
    GripperSpec(
        name="airbot_play_parallel_clawhauser",
        finger_names=("endleft", "endright"),
        finger_open_q=(0.04, 0.04),
        finger_closed_q=(0.005, 0.005),
        open_width=0.04,
        span=0.08,
        tcp_body="link6",
        tcp_offset=(0.0, 0.0, 0.005),
        approach_axis=(0.0, 0.0, 1.0),
        close_axis=(0.0, 1.0, 0.0),
        finger_bodies=(("left",), ("right",)),
    )
)


def build_scene():
    for relative, expected in SOURCE_HASHES.items():
        if hashlib.sha256((ASSET / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Clawhauser source asset changed: {relative}")
    vertices, tets, _labels, elastic, density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    pose = (SOURCE_X, 0.0, float(vertices[:, 2].max() + 0.002))
    with np.load(ASSET / "appearance" / "visual_to_physics.npz") as binding:
        faces = binding["visual_faces"][binding["boundary_face_part_ids"] == 2]
        ring_surface = np.unique(binding["physics_vertex_indices"][faces])
    grasp_ids = ring_surface[vertices[ring_surface, 1] > 0.095]
    local_center = vertices[grasp_ids].mean(axis=0)
    grasp_center = np.array((0.6 + SOURCE_X + local_center[0], -local_center[1], TABLE_Z + pose[2] - local_center[2]))
    print(
        json.dumps({"ring_patch_half_span_x_m": float(np.max(np.abs(vertices[grasp_ids, 0] - local_center[0])))}),
        flush=True,
    )

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
        backends=(("articulated", rss.backend("kinematic")), ("soft", vbd), ("rigid", vbd), ("*", vbd)),
        coupling=rss.coupler("one_way", contact_forces=True),
        collision=rss.CollisionSpec(
            soft_contact_margin=0.002,
            full_surface_contact=False,
            contact_ke=100000.0,
            contact_kd=25.0,
            broad_phase="nxn",
        ),
    )
    scene = rss.Scene(cell="bench_075", physics=physics, name="airbot_play_clawhauser_pick_place")
    for name, x, y, color in (
        ("source_zone", SOURCE_X, 0.0, (0.13, 0.28, 0.43)),
        ("target_zone", TARGET_X, -0.10, (0.10, 0.38, 0.28)),
    ):
        scene.add(rss.slab(name=name, half=(0.06, 0.06, 0.00025), at=(x, y, 0.001), collide=False, color=color))
    obj = scene.add(
        HeterogeneousSoftSpec(
            name="clawhauser",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            quat=(1.0, 0.0, 0.0, 0.0),
            contact_ke=100000.0,
        )
    )
    robot = rss.EmbodimentSpec(
        name="airbot_play_clawhauser",
        asset=rss.AssetSpec(path=str(AIRBOT_ASSET)),
        base_offset=(-0.48, 0.0, 0.132, 0.0, 0.0, 0.0, 1.0),
        arms=(
            rss.ArmSpec(
                name="right",
                joint_names=tuple(f"joint{i}" for i in range(1, 7)),
                ee_body_name="link6",
                gripper="airbot_play_parallel_clawhauser",
                home_q=AIRBOT_HOME,
            ),
        ),
    )
    scene.add(rss.robot(robot))
    camera = rss.CameraSpec(
        name="overview",
        width=640,
        height=480,
        eye=(0.92, -0.63, 1.45),
        look_at=(0.42, -0.04, 0.96),
        fov_y=50.0,
    )
    scene.add(camera)
    sim = scene.build(device="cuda:0", control_mode="joint_position", warmup=False, use_cuda_graph=False)
    if sim.model.tet_count != len(tets) or len(obj.entity.tet_part_labels) != len(tets):
        raise RuntimeError("AIRBOT scene changed the labelled Clawhauser tetrahedra")
    if sim.model.attachment_particle_body_count:
        raise RuntimeError("AIRBOT contact scene unexpectedly has grasp attachments")
    if not np.array_equal(np.unique(obj.entity.source_elastic[:, 0]), np.unique(elastic[:, 0])):
        raise RuntimeError("AIRBOT scene changed source elastic estimates")
    if not np.array_equal(np.unique(obj.entity.source_density), np.unique(density)):
        raise RuntimeError("AIRBOT scene changed source density estimates")
    entity = obj.entity
    model_tets = sim.model.tet_indices.numpy()[entity.tet_range[0] : entity.tet_range[1]]
    if not np.array_equal(model_tets - entity.particles[0], tets):
        raise RuntimeError("AIRBOT scene changed source tetrahedron order")
    young, nu = entity.source_elastic.T
    expected_mu = young / (2.0 * (1.0 + nu))
    expected_lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    model_elastic = sim.model.tet_materials.numpy()[entity.tet_range[0] : entity.tet_range[1], :2]
    if not np.allclose(model_elastic, np.column_stack((expected_mu, expected_lambda)), rtol=1e-5):
        raise RuntimeError("AIRBOT scene changed source FEM elasticity")
    if not np.allclose(sim.model.particle_mass.numpy()[entity.particles], entity.source_mass, rtol=1e-5):
        raise RuntimeError("AIRBOT scene changed source nodal masses")
    return sim, obj, camera, grasp_ids, grasp_center


def run(out: Path, *, steps: int, render_every: int):
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, camera, grasp_ids, grasp_center = build_scene()
    arm = sim.arm("right")
    hand = sim.gripper("right")
    q = sim.pipeline.state_0.joint_q.numpy()
    # The tool frame faces down and its local jaw axis (+y) spans world x.
    down_across_x = np.array((0.70710678, 0.70710678, 0.0, 0.0))
    at_ring = grasp_center.copy()
    # The loop settles about 30 mm below its source pose by grasp time.
    at_ring[2] -= 0.030
    targets = (
        ("hover", at_ring + np.array((0.0, 0.0, 0.13))),
        ("grasp", at_ring),
        ("lift", at_ring + np.array((0.0, 0.0, 0.20))),
        ("transfer", at_ring + np.array((TARGET_X - SOURCE_X, -0.10, 0.20))),
        ("place", at_ring + np.array((TARGET_X - SOURCE_X, -0.10, 0.015))),
        ("retreat", at_ring + np.array((TARGET_X - SOURCE_X, -0.10, 0.16))),
    )
    planned = []
    with arm.host_fk():
        seed = q.copy()
        for name, xyz in targets:
            solved = arm.solve_q_full(seed, xyz, down_across_x)
            pos_error, rot_error = arm.pose_errors(solved, xyz, down_across_x)
            planned.append((name, xyz, solved, pos_error, rot_error))
            seed = solved
    print(
        json.dumps(
            {
                "waypoints": [
                    {"name": name, "position": xyz.tolist(), "error_m": pe, "error_rad": re}
                    for name, xyz, _, pe, re in planned
                ]
            }
        ),
        flush=True,
    )
    if max(row[3] for row in planned) > 0.025:
        raise RuntimeError("AIRBOT cannot reach the Clawhauser pickup or place waypoints")
    # A straight Cartesian lift matters here: linear interpolation between
    # only the endpoint joint vectors swept the pads about 8 mm sideways and
    # pulled them free of the small loop before it rose.
    lift_qs = [planned[1][2]]
    with arm.host_fk():
        for fraction in np.linspace(0.1, 1.0, 10):
            lift_pos = planned[1][1] + (planned[2][1] - planned[1][1]) * fraction
            solved = arm.solve_q_full(lift_qs[-1], lift_pos, down_across_x)
            pos_error, _ = arm.pose_errors(solved, lift_pos, down_across_x)
            if pos_error > 0.01:
                raise RuntimeError(f"AIRBOT lift path misses by {pos_error:.3f} m")
            lift_qs.append(solved)
    print(
        json.dumps(
            {
                "arm_home_q": q[arm.arm_coords].tolist(),
                "arm_ee_world": arm.ee_pose(sim.pipeline.state_0)[0].tolist(),
                "grasp_center_world": grasp_center.tolist(),
                "joint_count": int(sim.model.joint_count),
                "solver": type(sim.pipeline.program.solver).__name__,
            }
        ),
        flush=True,
    )
    hand.command = 0.0
    hand.apply(sim.pipeline.control.joint_target_q)
    videos = None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=CONTROL_HZ // render_every)
        videos.__enter__()
    rows = []
    finger_coords = list(sim.handles.arms["right"].gripper_coords)
    finger_shapes = [np.concatenate([shapes for _, shapes in side]) for side in sim.handles.arms["right"].finger_groups]
    ring_particles = obj.entity.particles[grasp_ids]
    body_ids = obj.entity.particles[np.unique(obj.entity.source_tets[obj.entity.tet_part_labels == 1])]
    initial_body = sim.pipeline.state_0.particle_q.numpy()[body_ids].mean(axis=0)
    rest_particles = sim.pipeline.state_0.particle_q.numpy()[obj.entity.particles]
    source_tets = obj.entity.source_tets
    rest_det = np.linalg.det(rest_particles[source_tets[:, 1:]] - rest_particles[source_tets[:, :1]])
    try:
        for frame in range(steps):
            time_s = frame / CONTROL_HZ
            # Joint-space interpolation between IK waypoints keeps every move
            # arm-driven; the source object is never given a prescribed pose.
            if time_s < 1.0:
                qa, qb, s = q, planned[0][2], smooth(time_s, 0.0, 1.0)
            elif time_s < 2.2:
                qa, qb, s = planned[0][2], planned[1][2], smooth(time_s, 1.0, 2.2)
            elif time_s < 4.6:
                phase = min(9.999999, 10.0 * smooth(time_s, 3.0, 4.6))
                segment = int(phase)
                qa, qb, s = lift_qs[segment], lift_qs[segment + 1], phase - segment
            elif time_s < 6.6:
                qa, qb, s = planned[2][2], planned[3][2], smooth(time_s, 4.6, 6.6)
            elif time_s < 8.4:
                qa, qb, s = planned[3][2], planned[4][2], smooth(time_s, 6.6, 8.4)
            else:
                qa, qb, s = planned[4][2], planned[5][2], smooth(time_s, 10.0, 11.3)
            arm.set_joint_targets((qa + (qb - qa) * s)[arm.arm_coords], sim.pipeline.control.joint_target_q)
            close = smooth(time_s, 2.2, 3.0) * (1.0 - smooth(time_s, 8.7, 9.6))
            hand.command = close
            hand.apply(sim.pipeline.control.joint_target_q)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            particles = state.particle_q.numpy()[obj.entity.particles]
            if not np.isfinite(particles).all():
                raise RuntimeError(f"non-finite Clawhauser state at {frame}")
            contacts = sim.pipeline.contacts
            count = int(contacts.soft_contact_count.numpy()[0])
            contact_shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            contact_particles = contacts.soft_contact_particle.numpy()[: len(contact_shapes)]
            det = np.linalg.det(particles[source_tets[:, 1:]] - particles[source_tets[:, :1]])
            shape_forces = sim.pipeline.program.solver.shape_contact_forces()
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "ee_world": arm.ee_pose(state)[0].tolist(),
                "ring_center_world": state.particle_q.numpy()[ring_particles].mean(axis=0).tolist(),
                "finger_body_world": [
                    state.body_q.numpy()[side[0][0], :3].tolist() for side in sim.handles.arms["right"].finger_groups
                ],
                "finger_q": state.joint_q.numpy()[finger_coords].tolist(),
                "body_center_world": state.particle_q.numpy()[body_ids].mean(axis=0).tolist(),
                "body_translation_x_m": float(state.particle_q.numpy()[body_ids, 0].mean() - initial_body[0]),
                "minimum_z_m": float(particles[:, 2].min()),
                "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
                "minimum_tet_volume_ratio": float(np.min(det / rest_det)),
                "soft_contacts": count,
                "ring_contacts_by_side": [
                    int(np.count_nonzero(np.isin(contact_shapes, side) & np.isin(contact_particles, ring_particles)))
                    for side in finger_shapes
                ],
                "finger_contacts_by_side": [
                    int(np.count_nonzero(np.isin(contact_shapes, side))) for side in finger_shapes
                ],
                "finger_force_by_side_N": [shape_forces[side].sum(axis=0).tolist() for side in finger_shapes],
            }
            rows.append(row)
            if videos is not None and frame % render_every == 0:
                videos.append_frames({camera.name: sim.render(camera)})
            if frame % 15 == 0 or frame == steps - 1:
                print(json.dumps(row), flush=True)
    finally:
        if videos is not None:
            videos.__exit__(None, None, None)
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        transit = [row for row in rows if 4.6 <= row["time_s"] <= 6.6]
        settled = [row for row in rows if row["time_s"] >= 12.0]
        final_body = np.asarray(rows[-1]["body_center_world"])
        target_body = initial_body + np.array((TARGET_X - SOURCE_X, -0.10, 0.0))
        peak_lift = float(max(row["body_center_world"][2] - initial_body[2] for row in rows))
        target_error = float(np.linalg.norm(final_body[:2] - target_body[:2]))
        bilateral_transit = bool(transit and all(min(row["ring_contacts_by_side"]) > 0 for row in transit))
        released = bool(settled and all(sum(row["finger_contacts_by_side"]) == 0 for row in settled))
        table_supported = bool(
            settled and all(TABLE_Z - 0.002 <= row["minimum_z_m"] <= TABLE_Z + 0.005 for row in settled)
        )
        stable = bool(settled and np.linalg.norm(final_body - np.asarray(settled[0]["body_center_world"])) < 0.015)
        settled_inversions = max((row["inverted_tetrahedra"] for row in settled), default=None)
        result = {
            "frames": len(rows),
            "render_fps": CONTROL_HZ // render_every if render_every else 0,
            "newton_version": newton.__version__,
            "robot_model_source": "DISCOVERSE d67f47c084aba0e0cf422a8725235f8b9238655a",
            "solver": type(sim.pipeline.program.solver).__name__,
            "arm_joint_count": len(arm.arm_coords),
            "gripper_joint_count": len(finger_coords),
            "tetrahedra": int(sim.model.tet_count),
            "grasp_attachments": int(sim.model.attachment_particle_body_count),
            "final_translation_x_m": rows[-1]["body_translation_x_m"],
            "peak_lift_m": peak_lift,
            "target_xy_error_m": target_error,
            "bilateral_ring_contacts_in_transit": bilateral_transit,
            "released": released,
            "table_supported": table_supported,
            "stable_after_release": stable,
            "settled_inversions": settled_inversions,
            "success": bool(
                bilateral_transit
                and peak_lift >= 0.05
                and target_error <= 0.025
                and released
                and table_supported
                and stable
                and settled_inversions == 0
            ),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "clawhauser" / "airbot_play"
    )
    parser.add_argument("--steps", type=int, default=390)
    parser.add_argument("--render-every", type=int, default=1)
    args = parser.parse_args()
    run(args.out, steps=args.steps, render_every=args.render_every)
