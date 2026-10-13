# SPDX-License-Identifier: Apache-2.0
"""The Newton solvers as a bundle: ``mujoco``, ``vbd``, ``xpbd``, ``featherstone`` and ``kinematic``.

Importing this registers the five aliases and nothing else -- every ``cls`` is a
dotted path, so resolving an alias imports no solver, exactly as the two foreign
bundles beside this one work. Four are Newton's own; ``kinematic`` is ours, a
body that follows its joint targets exactly, and it integrates Newton's state on
the device like the rest, which is why it is here and not under a host engine.

Now every alias arrives the same way: a miss scans ``robosimstudio.engines``,
and this bundle is found as ``superdex`` and ``genesis`` are. An arm-only scene
still resolves ``mujoco`` with nothing imported by hand;
``tests/test_engine_roots.py`` proves it in a fresh subprocess with only
``robosimstudio.specs`` imported first.

Not a foreign engine, and it lives in ``engines/`` anyway because an alias has
one kind of home now. The schemas sit beside the registrations because they are
the registrations': ``MUJOCO_SCHEMA`` says what each of MuJoCo's constructor
kwargs is and what changing it costs, and it has no reader that does not go
through ``get_backend("mujoco").schema``.
"""

from __future__ import annotations

from robosimstudio.specs.params import RECAPTURE
from robosimstudio.specs.physics import ParamRule, register_backend, rule

__all__ = ["FEATHERSTONE", "KINEMATIC", "MUJOCO", "MUJOCO_SCHEMA", "VBD", "VBD_SCHEMA", "XPBD"]

MUJOCO_SCHEMA: dict[str, ParamRule] = {
    "nconmax": rule(group="buffers", doc="Contact ceiling, sized for the busiest frame."),
    "njmax": rule(group="buffers", doc="Constraint ceiling."),
    "jacobian": rule(choices=("sparse", "dense", "auto"), group="solver"),
    "iterations": rule(tier=RECAPTURE, apply="custom:mujoco_opt", range=(1.0, 200.0), group="solver"),
    "ls_iterations": rule(tier=RECAPTURE, apply="custom:mujoco_opt", range=(1.0, 50.0), group="solver"),
    "cone": rule(choices=("pyramidal", "elliptic"), group="solver"),
    "impratio": rule(range=(1.0, 100.0), group="solver"),
    "solver": rule(choices=("newton", "cg", "pgs"), group="solver"),
    "integrator": rule(choices=("euler", "implicitfast", "implicit"), group="solver"),
    "enable_multiccd": rule(group="contact", doc="Several contact points per pair; stops a resting mesh walking."),
    "use_mujoco_contacts": rule(group="contact", doc="Off routes collision through a Newton pipeline."),
}

VBD_SCHEMA: dict[str, ParamRule] = {
    "iterations": rule(tier=RECAPTURE, apply="attr:iterations", range=(1.0, 100.0), group="solver"),
    "friction_epsilon": rule(
        tier=RECAPTURE, apply="attr:friction_epsilon", log=True, range=(1e-4, 1e-1), group="solver"
    ),
    "particle_enable_self_contact": rule(group="self-contact"),
    "particle_self_contact_radius": rule(unit="m", range=(1.0e-4, 2.0e-2), group="self-contact"),
    "particle_self_contact_margin": rule(unit="m", range=(1.0e-4, 2.0e-2), group="self-contact"),
    "particle_topological_contact_filter_threshold": rule(range=(0.0, 5.0), step=1.0, group="self-contact"),
    "particle_rest_shape_contact_exclusion_radius": rule(unit="m", range=(0.0, 3.0e-2), group="self-contact"),
    "particle_vertex_contact_buffer_size": rule(range=(8.0, 512.0), group="buffers"),
    "particle_edge_contact_buffer_size": rule(range=(8.0, 512.0), group="buffers"),
    "rigid_avbd_beta": rule(log=True, range=(1.0, 1.0e4), group="rigid"),
    "rigid_contact_k_start": rule(log=True, range=(1.0, 1.0e6), group="rigid"),
    "rigid_contact_history": rule(group="rigid"),
    "rigid_body_contact_buffer_size": rule(range=(16.0, 1024.0), group="buffers"),
    "rigid_body_particle_contact_buffer_size": rule(range=(16.0, 2048.0), group="buffers"),
}

MUJOCO = register_backend(
    "mujoco",
    cls="newton.solvers.SolverMuJoCo",
    supports=("bodies", "joints", "shapes"),
    native_contacts=True,
    schema=MUJOCO_SCHEMA,
    summary="MuJoCo-Warp: articulated rigid bodies, the robot's solver.",
    # ``jacobian`` sparse rather than MuJoCo's "auto" is measured: five stacked
    # boxes stand every time under sparse and three times in eight under auto
    # (a stacking benchmark run during development); the ceilings are sized for the busiest frame,
    # because an overflow escalates into an illegal memory access.
    defaults={
        "nconmax": 2048,
        "njmax": 2048,
        "jacobian": "sparse",
        "iterations": 20,
        "ls_iterations": 20,
        "cone": "elliptic",
        "impratio": 10.0,
        "enable_multiccd": False,
        "use_mujoco_contacts": True,
    },
)
VBD = register_backend(
    "vbd",
    cls="newton.solvers.SolverVBD",
    supports=("bodies", "joints", "particles", "shapes"),
    maximal_coordinates=True,
    schema=VBD_SCHEMA,
    summary="Vertex Block Descent: cloth, soft bodies, and chains of capsules on cable joints.",
    defaults={"iterations": 5},
    # Millimetre-scale self-contact, on: Newton's own defaults are 0.2 m radii
    # sized for its centimetre-scale demos, and a sheet that does not collide
    # with itself is not a sheet that folds.
    particle_defaults={
        "particle_enable_self_contact": True,
        "particle_self_contact_radius": 0.002,
        "particle_self_contact_margin": 0.002,
        "particle_topological_contact_filter_threshold": 2,
        "particle_rest_shape_contact_exclusion_radius": 0.0,
        "particle_vertex_contact_buffer_size": 32,
        "particle_edge_contact_buffer_size": 64,
    },
)
XPBD = register_backend(
    "xpbd",
    cls="newton.solvers.SolverXPBD",
    supports=("bodies", "joints", "particles", "shapes"),
    maximal_coordinates=True,
    summary="Extended position-based dynamics.",
)
FEATHERSTONE = register_backend(
    "featherstone",
    cls="newton.solvers.SolverFeatherstone",
    supports=("bodies", "joints", "shapes"),
    summary="Featherstone articulated bodies.",
)
KINEMATIC = register_backend(
    "kinematic",
    cls="robosimstudio.scene.physics.kinematic.KinematicSolver",
    supports=("bodies", "joints", "shapes"),
    summary="Bodies that follow their joint targets exactly: a robot as a moving boundary.",
)
