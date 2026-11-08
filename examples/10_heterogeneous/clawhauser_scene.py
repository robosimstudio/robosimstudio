# SPDX-License-Identifier: Apache-2.0
"""What the three Clawhauser scenes -- pick-drop, rod and bottle -- share.

Each scene follows the same shape, and this module holds every step of it that
is not the scene's own task:

1. :func:`load_toy` reads the source asset and checks it has not changed.
2. :func:`arrangement` is the physics: rate, substeps, solver, coupling, contact.
3. :func:`toy_spec` places the toy, with the settings' material overrides.
4. :func:`check_build` checks what Newton built and applies the contact settings.
   ``build_scene(*, claw, coarse, solver_name, iterations, mu, grip_mu,
   force_limit, squeeze, self_contact, self_contact_radius, rigid_loop,
   <scene's own>)`` and ``run(out, *, steps, render_every, <the same>)`` take the
   common arguments in this order.
5. :func:`add_scene_arguments` / :func:`scene_kwargs` are the common CLI.

:class:`PhysicsSettings` holds what a scene takes from a reference setup;
:data:`REFERENCE` is the Newton VBD handoff's Franka pick
(``newton_vbd_code_slim/clawhauser_franka/sim_franka.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import arm_gripper
import numpy as np
import rigid_projection

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec, load_labelled_tets

__all__ = [
    "ASSET",
    "COARSE_BUDGET",
    "CONTACT_GAP",
    "REFERENCE",
    "SELF_CONTACT_RADIUS",
    "SOLVERS",
    "PhysicsSettings",
    "add_scene_arguments",
    "arrangement",
    "check_build",
    "load_toy",
    "scene_kwargs",
    "set_contact",
    "solver_backend",
    "toy_spec",
]

ASSET = Path(__file__).resolve().parent / "assets" / "clawhauser"
SOURCE_HASHES = {
    "final_mesh.mesh": "df3d1505d96d36fbfa76291e7ab94269a30500ba5e4b87516d0bf8f57f0c6458",
    "heterogeneous_params.npz": "7cfc8ca3cc7c08d261aa90b08832230801fec1ac62848d0e9239d9e2f318e8c6",
    "inferred_params.json": "45aacd698f2b38a6fd8bf13b46cc9c33497fc90c6c1bdf088ed12d915ec3d992",
    "appearance/visual_to_physics.npz": "4b015d1141794211e4e2c6e2b39fadef6762a73ffe2d76b921e47d5676f3d6c4",
}

CONTACT_GAP = 0.006
"""The collision pipeline's detection gap [m], the handoff's. The coarse block's own
gap must be this: a correction can only reach contacts the pipeline stored."""
COARSE_BUDGET = CONTACT_GAP / 4.0
"""Cumulative coarse translation per substep [m], at most a quarter of the gap."""
SELF_CONTACT_RADIUS = 0.002
"""The toy's self-contact radius and detection margin [m].

VBD truncates each particle's motion per substep to about 0.43 x this, so it also
caps the toy's speed: 0.002 m allows about 1 m/s at 1200 substeps per second."""

SOLVERS = {
    "chebyshev": (
        "robosimstudio_materials.chebyshev_acceleration.SolverVBDChebyshev",
        {"chebyshev_rho": 0.95, "chebyshev_gamma": 1.0},
    ),
    "vbd": ("vbd", {}),
}
"""Particle solvers a scene can run: the backend kind and its own parameters."""


@dataclass(frozen=True)
class PhysicsSettings:
    """Rate, contact and material settings a scene builds with."""

    control_hz: int
    """Frames per second."""
    substeps: int
    """Newton substeps per frame."""
    contact_ke: float
    """Particle-shape penalty stiffness [N/m], for the toy and every shape."""
    contact_kd: float
    """Particle-shape contact damping [N·s/m], for the toy and every shape."""
    soft_mu: float
    """The toy's soft-contact friction."""
    particle_radius: float
    """The toy's particle radius [m]: the distance at which it touches a shape."""
    shape_mu: float
    """Every collision shape's friction."""
    damping_ratio: float = 0.0
    """Stiffness-proportional tet damping [s]; 0 keeps the spec's uniform ``damping``."""
    part_overrides: tuple = ()
    """``(part_id, field, value)`` rows for ``HeterogeneousSoftSpec``."""

    @property
    def grip_mu(self) -> float:
        """Pad-toy friction as Newton mixes it: ``sqrt(soft_mu * shape_mu)``."""
        return math.sqrt(self.soft_mu * self.shape_mu)


DAMP_BETA = 5.0e-4
"""The handoff's damping ratio [s]: each tet's ``k_damp`` is this times its shear modulus."""
BODY_SHEAR_PA = 5.0e5 / (2.0 * (1.0 + 0.45))
"""The body's source shear modulus [Pa] (E 0.5 MPa, Poisson ratio 0.45)."""

REFERENCE = PhysicsSettings(
    control_hz=60,
    substeps=20,
    contact_ke=1.0e6,
    contact_kd=10.0,
    soft_mu=1.0,
    particle_radius=0.003,
    shape_mu=1.5,
    damping_ratio=DAMP_BETA,
    part_overrides=(
        (0, "youngs_modulus_pa", 1.0e5),  # tail: the source's 0.5 MPa x TAIL_SCALE 0.2
        (2, "density_kg_m3", 300.0),  # loop: LOOP_RHO, not the source's 1100
        (2, "damping_pa_s", DAMP_BETA * BODY_SHEAR_PA),  # loop: damped at the body's level, not its own
    ),
)
"""The handoff's values. Its body E, Poisson ratios and tail and body density
already match the source, so only the tail's E and the loop's density and damping
are overridden."""


def load_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The source ``(vertices, tets, labels)``, after checking the asset is unchanged."""
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
    return vertices, tets, labels


def solver_backend(name: str, iterations: int, **params):
    """The ``rss.backend`` for solver ``name`` at ``iterations`` sweeps per substep."""
    kind, own = SOLVERS[name]
    return rss.backend(kind, iterations=int(iterations), **own, **params)


def arrangement(
    settings: PhysicsSettings,
    *,
    claw: str,
    coarse: bool,
    solver_name: str,
    iterations: int,
    self_contact: bool,
    self_contact_radius: float = SELF_CONTACT_RADIUS,
) -> rss.ArrangementSpec:
    """The scenes' physics: the toy on VBD, and the arm on MuJoCo unless ``claw == "claw"``.

    The arm resolves to MuJoCo and is its own island; ``one_way`` with
    ``slide_feedback`` stops each finger on the loop's push, so the clamp is the
    grip. Sticky matching is the scene's and the coupler forwards it.
    """
    common = {
        "particle_enable_self_contact": self_contact,
        # Millimetre-scale self-contact: the solver's own default margin is
        # 0.2 m, which makes every particle pair touch and the toy explode.
        "particle_self_contact_radius": self_contact_radius,
        "particle_self_contact_margin": self_contact_radius,
        "particle_topological_contact_filter_threshold": 2,
        # The plush's own surface is within the contact margin at rest, and the
        # grasp squeezes the 11 mm loop by 6 mm; exclude pairs that are that
        # close in the rest shape, or the pinch folds the loop into itself.
        "particle_rest_shape_contact_exclusion_radius": 0.006,
        "particle_vertex_contact_buffer_size": 32,
        "particle_edge_contact_buffer_size": 64,
        # The handoff's, which are Newton's defaults.
        "particle_enable_tile_solve": True,
        "rigid_body_particle_contact_buffer_size": 256,
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
        else solver_backend(solver_name, iterations, **common)
    )
    split = claw != "claw"
    return rss.ArrangementSpec(
        control_hz=settings.control_hz,
        substeps=settings.substeps,
        use_cuda_graph=False,
        backends=((("articulated", rss.backend("mujoco")),) if split else ())
        + (("soft", vbd), ("rigid", vbd), ("*", vbd)),
        **({"coupling": rss.coupler("one_way", contact_forces=True, slide_feedback=True)} if split else {}),
        collision=rss.CollisionSpec(
            soft_contact_margin=CONTACT_GAP,
            full_surface_contact=False,
            contact_ke=settings.contact_ke,
            contact_kd=settings.contact_kd,
            broad_phase="nxn",
            # VBD measures body-particle friction by the motion inside each
            # collision pass, which creeps under any static tangential load: the
            # pad pinch slides the loop out however high the friction is. Sticky
            # matching keeps each pad contact's points across passes, and friction
            # then acts on the drift from them.
            contact_matching="sticky",
        ),
    )


def toy_spec(settings: PhysicsSettings, *, pose, quat) -> HeterogeneousSoftSpec:
    """The Clawhauser at ``pose``/``quat``, with the settings' contact and part overrides."""
    return HeterogeneousSoftSpec(
        name="clawhauser",
        mesh_path=str(ASSET / "final_mesh.mesh"),
        params_path=str(ASSET / "heterogeneous_params.npz"),
        materials_path=str(ASSET / "inferred_params.json"),
        pose=pose,
        quat=quat,
        contact_ke=settings.contact_ke,
        contact_kd=settings.contact_kd,
        contact_mu=settings.soft_mu,
        particle_radius=settings.particle_radius,
        damping_ratio=settings.damping_ratio,
        part_overrides=settings.part_overrides,
    )


def set_contact(
    sim, solver, settings: PhysicsSettings, *, mu: float | None = None, grip_mu: float | None = None
) -> dict:
    """Apply ``settings``' contact stiffness, damping and friction to the built model.

    The toy gets ``mu`` (``settings.soft_mu`` when None) and every shape gets
    ``settings.shape_mu``. With ``grip_mu`` (the arm), the hand's pads are then
    set so their contact with the toy mixes to it. The scene's model and the
    solver's are both written, in case they differ.
    """
    soft_mu = settings.soft_mu if mu is None else float(mu)
    for model in {id(sim.model): sim.model, id(solver.model): solver.model}.values():
        model.soft_contact_ke = settings.contact_ke
        model.soft_contact_kd = settings.contact_kd
        model.soft_contact_mu = soft_mu
        model.shape_material_ke.fill_(settings.contact_ke)
        model.shape_material_kd.fill_(settings.contact_kd)
        model.shape_material_mu.fill_(settings.shape_mu)
    report = {
        "soft_contact_ke": settings.contact_ke,
        "soft_contact_kd": settings.contact_kd,
        "global_soft_contact_mu": soft_mu,
        "shape_mu": settings.shape_mu,
    }
    if grip_mu is not None:
        report.update(arm_gripper.set_contact_friction(sim, global_mu=soft_mu, grip_mu=grip_mu))
    return report


def check_build(
    sim,
    entity,
    tets: np.ndarray,
    settings: PhysicsSettings,
    *,
    claw: str,
    force_limit: float,
    mu: float | None,
    grip_mu: float | None,
    rigid_loop: bool = False,
) -> dict:
    """Check what Newton built, apply the contact settings, and return the report.

    ``mu`` and ``grip_mu`` fall back to the settings' when None. ``rigid_loop``
    projects the loop onto a rigid pose after every solver step (:mod:`rigid_projection`). The VBD solver
    is read through the coupler, which every VBD attribute has to come from.
    """
    split = claw != "claw"
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    if not type(solver).__name__.startswith("SolverVBD"):
        raise RuntimeError(f"expected a VBD solver, got {type(solver).__name__}")
    report = {}
    if rigid_loop:
        nodes, free = rigid_projection.part_nodes(entity.source_tets, entity.tet_part_labels, 2, attached_to=1)
        rigid_projection.enable(solver, sim.model, np.asarray(entity.particles)[nodes], free)
        report["rigid_loop"] = {"nodes": int(len(nodes)), "projected": int(free.sum())}
    if split:
        report["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    if sim.model.tet_count != len(tets) or len(entity.tet_part_labels) != len(tets):
        raise RuntimeError("RSS lost Clawhauser tetrahedra or part labels")
    if sim.model.attachment_particle_body_count:
        raise RuntimeError("the contact grasp scene must not contain particle attachments")
    grip_mu = settings.grip_mu if grip_mu is None else grip_mu
    report["friction"] = set_contact(sim, solver, settings, mu=mu, grip_mu=grip_mu if split else None)
    report["global_soft_contact_mu"] = report["friction"]["global_soft_contact_mu"]
    span = slice(*entity.tet_range)
    if not np.array_equal(sim.model.tet_indices.numpy()[span] - entity.particles[0], tets):
        raise RuntimeError("Newton changed the source tetrahedron order")
    young, nu = entity.elastic.T
    lame = np.column_stack((young / (2.0 * (1.0 + nu)), young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
    if not np.allclose(sim.model.tet_materials.numpy()[span, :2], lame, rtol=1e-5):
        raise RuntimeError("Newton FEM elasticity differs from the toy's materials")
    if not np.allclose(sim.model.tet_materials.numpy()[span, 2], entity.damping, rtol=1e-5):
        raise RuntimeError("Newton FEM damping differs from the toy's materials")
    if not np.allclose(sim.model.particle_radius.numpy()[entity.particles], settings.particle_radius):
        raise RuntimeError("Newton particle radius differs from the settings'")
    if not np.allclose(sim.model.particle_mass.numpy()[entity.particles], entity.mass, rtol=1e-5):
        raise RuntimeError("Newton nodal mass differs from the toy's per-tet densities")
    return report


def add_scene_arguments(parser: argparse.ArgumentParser, *, out: Path, steps: int, squeeze: float) -> None:
    """The flags every Clawhauser scene takes; :func:`scene_kwargs` turns them into ``run`` arguments."""
    parser.add_argument("--out", type=Path, default=out)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument(
        "--claw",
        choices=("arm", "claw"),
        default="arm",
        help="who moves: the AIRBOT arm, or the scripted parallel claw the scene was built on",
    )
    parser.add_argument("--coarse", action="store_true", help="run the coarse common-translation solve")
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="chebyshev", help="particle solver")
    parser.add_argument("--iterations", type=int, default=160, help="solver sweeps per substep")
    parser.add_argument(
        "--mu", type=float, default=None, help="the toy's soft-contact friction; the settings' when unset"
    )
    parser.add_argument(
        "--grip-mu", type=float, default=None, help="the arm's pad-toy friction; the settings' when unset"
    )
    parser.add_argument(
        "--force-limit", type=float, default=arm_gripper.GRASP_LIMIT_N, help="the jaws' closing clamp [N]"
    )
    parser.add_argument("--squeeze", type=float, default=squeeze, help="close travel past the loop's width [m]")
    parser.add_argument(
        "--no-self-contact", dest="self_contact", action="store_false", help="turn the toy's self-contact off"
    )
    parser.add_argument(
        "--self-contact-radius", type=float, default=SELF_CONTACT_RADIUS, help="self-contact radius and margin [m]"
    )
    parser.add_argument(
        "--rigid-loop", action="store_true", help="project the loop onto a rigid pose after every substep"
    )


def scene_kwargs(args: argparse.Namespace) -> dict:
    """The common flags as ``run`` keyword arguments (``--out`` excepted)."""
    return {
        "steps": args.steps,
        "render_every": args.render_every,
        "claw": args.claw,
        "coarse": args.coarse,
        "solver_name": args.solver,
        "iterations": args.iterations,
        "mu": args.mu,
        "grip_mu": args.grip_mu,
        "force_limit": args.force_limit,
        "squeeze": args.squeeze,
        "enable_self_contact": args.self_contact,
        "self_contact_radius": args.self_contact_radius,
        "rigid_loop": args.rigid_loop,
    }
