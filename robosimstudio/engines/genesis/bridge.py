# SPDX-License-Identifier: Apache-2.0
"""Genesis's IPC coupler as a host engine beside the Newton islands.

**The idiom differs from SuperDex's, and it is worth saying how.** A SuperDex
mirror is a *static* actor moved by ``set_root_transform``. Genesis has no such
object, and it does not need one: its IPC coupler already implements the
penalty-mirror pattern internally. Each coupled rigid link enters the libuipc
solve as an affine body held by a soft transform constraint animated toward
whatever pose Genesis's *own* rigid solver reports. So a mirror here is a
genuine Genesis rigid entity whose pose we overwrite every step, and the
coupling reaction the constraint develops is exactly the wrench we want back.

Three measured facts shape the rest, each of which fails silently if ignored:

* **A ``fixed`` mirror is silent.** The reaction is
  ``(k/dt^2) * m * (pos_ipc - pos_aim)``, and a fixed link never deviates from
  its aim, so the reaction is identically zero while the cloth visibly rests on
  it. A two-way mirror must be a *free* base with ``gravity_compensation=1.0``;
  measured, that compensation is exact -- an identical scene with no cloth reads
  0.00000 N, so there is no phantom load to subtract.
* **The reaction needs no negation.** ``out_forces`` is the force *on the link*,
  unlike SuperDex's per-actor-pair query which reports the force on the
  deformable. It acts at the **link origin** and comes with a **pure couple**.
* **Genesis's ``FEM.Cloth`` simulates twice the mass** that
  ``rho * thickness * area`` predicts -- constant across density, thickness and
  mesh refinement. See :data:`CLOTH_MASS_FACTOR`.

Genesis's quaternions are ``wxyz``; Newton's, and therefore this package's, are
``xyzw``. Every conversion happens at this boundary and nowhere else, because a
swap that leaks past it produces a plausible-looking wrong rotation rather than
an error.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.engines.cosim import CosimRig, engine_dt
from robosimstudio.engines.feedback import WrenchAccumulator
from robosimstudio.engines.geometry import (
    ShapeGeometry,
    actor_builder,
    actor_geometry,
    builders_for,
    iter_mirrors,
    register_actor_builder,
)
from robosimstudio.engines.pose import compose, quat_rotate

from .runtime import ensure_initialized

__all__ = ["CLOTH_MASS_FACTOR", "GenesisBridge", "cloth_density", "mirror_role", "wxyz"]

log = logging.getLogger(__name__)

GROUND_EXTENT = 40.0
"""Side length [m] of the box a Newton ground plane becomes. See ``_add_mirror``."""

GROUND_THICKNESS = 0.02
"""Thickness [m] of that box, hung below the plane so its top face *is* the plane."""

CLOTH_MASS_FACTOR = 2.0
"""What Genesis's ``FEM.Cloth`` masses, over what ``rho * thickness * area`` says.

Measured as the reaction force holding a mirror still under a settled cloth, and
constant to three figures across a doubled density, a doubled thickness and a
halved mesh spacing -- so it is a convention (a shell giving each of its two
surfaces the full ``rho * thickness``) rather than an artefact.

It has to be divided out, and that is a correctness matter rather than a tuning
one: the Newton model and the Genesis model must agree on what the cloth weighs,
or the wrench fed back to the arm is twice what it should be and the cloth's own
dynamics are not the scene's. The division is checked at build --- see
:meth:`GenesisBridge._verify_mass` --- because a factor that changes upstream
would otherwise read as "the cloth feels heavy" for a week.
"""


def wxyz(q_xyzw) -> np.ndarray:
    """Newton's ``xyzw`` quaternion as Genesis's ``wxyz``."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]])


def cloth_density(areal_density: float, thickness: float) -> float:
    """The ``rho`` to hand ``FEM.Cloth`` for a given areal density [kg/m^2].

    Divides out :data:`CLOTH_MASS_FACTOR`, so the simulated sheet weighs what
    the Newton model says it weighs.
    """
    return float(areal_density) / (CLOTH_MASS_FACTOR * max(float(thickness), 1.0e-6))


def mirror_role(geom: ShapeGeometry) -> tuple[str, bool, float]:
    """``(coup_type, fixed, gravity_compensation)`` for one mirror.

    A two-way mirror must be free-based or its reaction is identically zero (see
    the module docstring); everything else is better off fixed, which makes it a
    perfect kinematic collider and costs Genesis no integration.
    """
    if geom.two_way and geom.moving:
        return "two_way_soft_constraint", False, 1.0
    return "ipc_only", True, 0.0


class GenesisBridge:
    """One Genesis scene, its actors and mirrors, and the per-control-step tick.

    Args:
        model: The finalized Newton model.
        recipe: The manifest's host-island record.
        substeps: Genesis steps per control step. ``SimOptions.substeps`` stays
            at 1 and this drives ``scene.step()`` that many times instead --
            libuipc takes its ``dt`` from the *step* dt while the coupler runs
            per substep, so a scene built with ``substeps > 1`` integrates the
            IPC world at the wrong dt.
    """

    def __init__(
        self,
        model: Any,
        recipe: dict,
        *,
        substeps: int = 1,
        precision: str = "64",
        feedback: str = "auto",
        pose_source: str = "fresh",
    ) -> None:
        gs = ensure_initialized(precision)
        self.gs = gs
        self.model = model
        self.recipe = dict(recipe)
        self.substeps = max(1, int(substeps))
        self.feedback = str(feedback)

        control_hz = float(recipe.get("control_hz", 60.0))
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=engine_dt(1.0 / control_hz, self.substeps),  # the co-sim contract, stated once
                substeps=1,
                gravity=tuple(float(v) for v in recipe.get("gravity", (0.0, 0.0, -9.81))),
            ),
            coupler_options=gs.options.IPCCouplerOptions(
                # Required even though Newton owns the mirrors: this flag is what
                # makes the coupler *write* the reaction arrays we read back.
                two_way_coupling=True,
                # Newton already resolved rigid-rigid. The mirrors are its bodies
                # seen twice, and letting them collide here double-counts.
                enable_rigid_rigid_contact=False,
                contact_d_hat=float(recipe.get("contact_d_hat", 1.0e-3)),
                constraint_strength_translation=float(recipe.get("constraint_strength_translation", 100.0)),
                constraint_strength_rotation=float(recipe.get("constraint_strength_rotation", 100.0)),
            ),
            show_viewer=False,
        )

        self.actors: list[dict] = [self._add_actor(r) for r in recipe.get("actors", ())]
        self.mirrors: list[dict] = []
        for geom in iter_mirrors(model, recipe.get("mirrors", ())):
            entry = self._add_mirror(geom)
            if entry is not None:
                self.mirrors.append(entry)

        self._check_initial_clearance()
        self.scene.build()
        self._verify_mass()
        self._link_of = {int(m["entity"].base_link.idx): m for m in self.mirrors}
        self._entity_by_shape = {m["geom"].shape: m["entity"] for m in self.mirrors}
        self._rest_frame: int | None = None

        self.rig = CosimRig(
            [m["geom"].body for m in self.mirrors],
            substeps=self.substeps,
            pose_source=str(pose_source),
        )

        self.feedback_enabled = (
            self.feedback != "off"
            and any(m["geom"].two_way and m["geom"].moving for m in self.mirrors)
            and int(getattr(model, "body_count", 0)) > 0
        )
        self.apply_wrenches = None
        """Set by the owning solver: ``(wrench[n_bodies, 6] float32) -> None``."""
        self._acc = (
            WrenchAccumulator(
                model,
                alpha=float(recipe.get("feedback_alpha", 0.35)),
                dv_max=float(recipe.get("feedback_dv_max", 0.5)),
            )
            if self.feedback_enabled
            else None
        )

    # ----------------------------------------------------------------- build
    def _add_actor(self, record: dict) -> dict:
        """One deformable, from the shared reader into a Genesis FEM entity.

        The builder is looked up by kind -- the same registry SuperDex reads --
        so a kind this engine has never been given is refused by name rather
        than falling into somebody else's branch.
        """
        gs = self.gs
        geom = actor_geometry(self.model, record)
        build = actor_builder("genesis", geom.kind)
        if build is None:
            raise ValueError(
                f"genesis cannot build a {geom.kind!r} actor ({geom.name!r}); it builds {builders_for('genesis')}. "
                "The door for a new kind on this engine is engines.geometry.register_actor_builder('genesis', kind, build)."
            )
        mesh, material = build(self, geom)
        # ``MeshSet`` is the in-memory path: a plain ``Mesh(file=<Trimesh>)``
        # falls through Genesis's morph dispatch and never reaches the loader.
        entity = self.scene.add_entity(
            morph=gs.morphs.MeshSet(files=(mesh,), scale=1.0, pos=(0.0, 0.0, 0.0)),
            material=material,
        )
        if geom.kind == "tet":
            # A non-cloth material otherwise runs tetgen and invents interior
            # vertices, which would break the 1:1 map to Newton's particles.
            # ``instantiate`` is the public seam that takes our own lattice.
            entity.instantiate(geom.rest.astype(np.float32), geom.tets.astype(np.int32))
        return {"name": geom.name, "entity": entity, "geom": geom, "prev": geom.rest.copy()}

    def _add_mirror(self, geom: ShapeGeometry) -> dict | None:
        """One Newton shape as a Genesis rigid entity, or ``None`` to skip."""
        gs = self.gs
        coup_type, fixed, gravcomp = mirror_role(geom)
        scale = geom.scale
        # **Every** mirror is built at its world pose, moving ones included, and
        # that is not the idiom SuperDex uses. A SuperDex mirror is a static
        # actor, so where it is built is irrelevant -- the first tick's
        # ``set_root_transform`` teleports it before anything can touch the
        # deformable. A Genesis mirror is a *dynamic* affine body held by a soft
        # constraint toward wherever the rigid solver says the link is, so
        # building it at the origin and then aiming it a metre away on the first
        # tick drags it across the scene at whatever speed the constraint can
        # manage. Measured: the cloth left at 2.8 m/s and never came back.
        #
        # A world-fixed mirror is placed here and never again -- Genesis refuses
        # ``set_pos`` on a fixed-based entity carrying a ``coup_type``, and its
        # shape transform is already world. A moving one is placed here *and*
        # driven each substep, so the first tick's teleport is a no-op.
        place = geom.local if not geom.moving else compose(self._body_pose(geom.body), geom.local)
        common = {"pos": tuple(float(v) for v in place[:3]), "quat": tuple(float(v) for v in wxyz(place[3:7]))}
        if geom.kind == "plane":
            # NOT gs.morphs.Plane, and this is an upstream constraint rather
            # than a preference: a scene holding a Plane *and* any non-plane
            # rigid fails in the IPC coupler's build --
            # ``_abd_state_feature.body_count()`` on ``None``. Isolated here:
            # box alone builds, plane alone builds, plane+plane builds, and
            # plane+box does not. A ground-sized flat box collides identically
            # for anything within reach of it and coexists with the rest.
            down = quat_rotate(place[3:7], np.array([0.0, 0.0, -0.5 * GROUND_THICKNESS]))
            morph = gs.morphs.Box(
                size=(GROUND_EXTENT, GROUND_EXTENT, GROUND_THICKNESS),
                pos=tuple(float(v) for v in place[:3] + down),
                quat=common["quat"],
                fixed=True,
            )
        elif geom.kind == "sphere":
            morph = gs.morphs.Sphere(radius=float(scale[0]), fixed=fixed, **common)
        elif geom.kind == "box":
            morph = gs.morphs.Box(size=tuple(2.0 * scale), fixed=fixed, **common)
        elif geom.kind == "cylinder":
            morph = gs.morphs.Cylinder(radius=float(scale[0]), height=2.0 * float(scale[1]), fixed=fixed, **common)
        else:
            mesh = geom.mesh()
            if mesh is None:
                return None
            path = self._cache_obj(geom, mesh)
            # Every flag here is load bearing. `align` reframes the root link
            # onto the centre of mass, which would silently offset every torque
            # arm; `convexify` would mirror a decomposed fingertip as a
            # different decomposition than the one Newton collides with.
            morph = gs.morphs.Mesh(
                file=path,
                fixed=fixed,
                align=False,
                convexify=False,
                decimate=False,
                watertighten=None,
                recompute_inertia=True,
                **common,
            )
        entity = self.scene.add_entity(
            morph=morph,
            material=gs.materials.Rigid(coup_type=coup_type, gravity_compensation=gravcomp, coup_friction=geom.mu),
        )
        return {"entity": entity, "geom": geom}

    def _body_pose(self, body: int) -> np.ndarray:
        """A body's world pose as the model was built. Where its mirror starts."""
        return np.asarray(self.model.body_q.numpy()[int(body)], dtype=np.float64)

    def _cache_obj(self, geom: ShapeGeometry, mesh) -> str:
        """A tessellated mirror as an ``.obj`` under this island's workspace.

        A file rather than a ``MeshSet`` because that in-memory path is verified
        only for the FEM entity; a rigid morph reading one is not. Mirrors are
        built once, so the write is not on any hot path. The name carries the
        shape index and nothing about the project, since a repro bundle's leak
        scan reads paths.
        """
        verts, faces = mesh
        root = pathlib.Path(tempfile.gettempdir()) / "rss-genesis" / str(self.recipe.get("island", "host"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"mirror-{geom.shape}.obj"
        with path.open("w") as handle:
            for v in verts:
                handle.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
            for f in faces:
                handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
        return str(path)

    def _check_initial_clearance(self) -> None:
        """Refuse a scene whose actors start inside a mirror, and say which.

        **IPC's barrier is undefined through an intersection**, so libuipc
        requires an intersection-free initial state -- that is a property of the
        method, not a tunable. A scene that violates it fails *inside*
        ``scene.build()`` with ``'NoneType' object has no attribute
        'body_count'``: the coupler never finishes wiring its accessors, and the
        error names neither the geometry nor the reason.

        The check is a conservative axis-aligned overlap of an actor's rest
        bounding box against each box-shaped mirror's (the world AABB of the
        rotated box) -- cheap, build-time only, and deliberately *not* an exact
        test: a false alarm names two pieces of
        geometry and asks a human to look, which is a far better failure than
        the one it replaces. Non-box mirrors are skipped rather than
        approximated, so this is a net that catches the common case and never
        pretends to catch every case.
        """
        for entry in self.actors:
            geom = entry["geom"]
            lo, hi = geom.rest.min(axis=0), geom.rest.max(axis=0)
            for mirror in self.mirrors:
                m = mirror["geom"]
                if m.kind != "box" or m.moving:
                    continue
                # The world AABB of the *rotated* box: half-extents |R| @ scale.
                # Ignoring the rotation under-reports a box turned about any
                # axis -- a 45-degree plate's footprint grows by sqrt(2).
                half = np.abs(quat_to_mat(m.local[3:7])) @ np.asarray(m.scale[:3], dtype=np.float64)
                m_lo = m.local[:3] - half
                m_hi = m.local[:3] + half
                if np.all(lo < m_hi) and np.all(hi > m_lo):
                    raise ValueError(
                        f"genesis island {self.recipe.get('island')!r}: actor {geom.name!r} starts "
                        f"inside mirror shape {m.shape} (actor bounds "
                        f"{np.round(lo, 4).tolist()}..{np.round(hi, 4).tolist()}, mirror "
                        f"{np.round(m_lo, 4).tolist()}..{np.round(m_hi, 4).tolist()}). IPC needs an "
                        "intersection-free initial state; move one clear of the other."
                    )

    def _verify_mass(self) -> None:
        """Genesis's simulated cloth mass against Newton's, at build.

        :data:`CLOTH_MASS_FACTOR` is divided out on the way in; this checks it
        landed. A silent factor here is the difference between an arm feeling a
        towel and an arm feeling two towels, and neither reads as an error.
        """
        newton_mass = self.model.particle_mass.numpy()
        for entry in self.actors:
            geom = entry["geom"]
            want = float(newton_mass[geom.start : geom.start + geom.count].sum())
            got = _entity_mass(entry["entity"])
            if got is None:
                # Say so rather than pass quietly. This Genesis build exposes no
                # simulated mass on an FEM entity -- the FEM solver's own
                # ``elements_v_info.mass`` is a placeholder (1.0 per vertex) and
                # the real figure lives inside libuipc, unreachable from here.
                # So CLOTH_MASS_FACTOR is applied and *not* confirmed against
                # this scene, and a silent pass would read exactly like a
                # confirmed one. The factor itself is measured and pinned by
                # tests/test_genesis_absent.py; what is unverified is whether
                # this version still agrees with the measurement.
                log.warning(
                    "genesis actor %r: this build exposes no simulated mass, so the x%.1f cloth-mass "
                    "convention is applied unverified. Newton says %.1f g. If the arm feels a cloth "
                    "of the wrong weight, re-measure CLOTH_MASS_FACTOR against this Genesis version.",
                    geom.name,
                    CLOTH_MASS_FACTOR,
                    want * 1e3,
                )
                continue
            if want <= 0.0:
                continue
            if abs(got - want) > 0.1 * want:
                raise ValueError(
                    f"genesis actor {geom.name!r} simulates {got * 1e3:.1f} g where the Newton model "
                    f"says {want * 1e3:.1f} g (ratio {got / want:.3f}). CLOTH_MASS_FACTOR is "
                    f"{CLOTH_MASS_FACTOR}; re-measure it against this Genesis version."
                )

    # ------------------------------------------------------------------ tick
    def attach_pipeline(self, pipeline: Any) -> None:
        self.rig.attach_pipeline(pipeline)

    def tick(self, state: Any, dt: float) -> None:
        """Advance Genesis by one control period and write the results back."""
        rig = self.rig
        targets = rig.body_poses(state)
        for _sub, posed in rig.substeps_of([m["geom"] for m in self.mirrors], targets):
            for geom, pose in posed:
                entity = self._entity_by_shape[geom.shape]
                # zero_velocity because Genesis's rigid solver integrates a free
                # body in substep_pre_coupling, *before* the coupler reads the
                # pose we just wrote. Friction still sees the right surface
                # velocity: IPC's is displacement-based, so it comes from the
                # sweep between aims, which is what the interpolation provides.
                entity.set_pos(pose[:3], zero_velocity=True)
                entity.set_quat(wxyz(pose[3:7]), zero_velocity=True)
            # Genesis takes its dt from SimOptions -- set at construction to
            # one control period over `substeps` -- rather than per call.
            self.scene.step()
        rig.commit(targets)

        if self.feedback_enabled and self.apply_wrenches is not None:
            self._harvest(targets, dt)

        for entry in self.actors:
            positions = np.asarray(entry["entity"].get_state().pos.cpu()).reshape(-1, 3)
            entry["prev"] = rig.write_particles(
                state, entry["geom"].start, entry["geom"].count, positions, dt, entry["prev"]
            )

    def _harvest(self, targets: dict[int, np.ndarray], dt: float) -> None:
        """The IPC constraint reaction, onto the Newton bodies.

        Read off the coupler's own arrays rather than intercepting the call it
        makes to its rigid solver: this is a *read* of a data holder the coupler
        fills once per ``couple()``, where a patch would be a behavioural change
        to a third-party object that fires for every caller. Private, and
        version-fragile -- pinned in ``[genesis]`` for exactly that reason.
        """
        acc = self._acc
        acc.begin()
        data = getattr(getattr(self.scene.sim, "_coupler", None), "_coupling_data", None)
        if data is not None:
            forces = np.asarray(data.out_forces)
            torques = np.asarray(data.out_torques)
            forces = forces[0] if forces.ndim == 3 else forces
            torques = torques[0] if torques.ndim == 3 else torques
            for row, link in enumerate(np.atleast_1d(np.asarray(data.links_idx))):
                entry = self._link_of.get(int(link))
                if entry is None or not entry["geom"].two_way:
                    continue
                body = entry["geom"].body
                if body not in targets:
                    continue
                # No negation: out_forces is already the force on the link. It
                # acts at the link origin, and out_torques is a pure couple.
                acc.add_at_shape(
                    body,
                    np.asarray(forces[row], dtype=np.float64),
                    targets[body],
                    entry["geom"].local,
                    torque=np.asarray(torques[row], dtype=np.float64),
                )
        self.apply_wrenches(acc.shape(dt))

    # ----------------------------------------------------------------- reset
    def capture_rest(self) -> None:
        """Record the settled state as what a reset returns to."""
        world = getattr(getattr(self.scene.sim, "_coupler", None), "_ipc_world", None)
        if world is None or not hasattr(world, "dump"):
            log.warning(
                "this genesis build exposes no uipc world dump; a reset will return to the "
                "as-built state rather than the settled one"
            )
            return
        world.dump()
        self._rest_frame = int(world.frame())

    def restore_rest(self) -> None:
        """Return Genesis to the captured rest; the next tick teleports mirrors."""
        world = getattr(getattr(self.scene.sim, "_coupler", None), "_ipc_world", None)
        if world is not None and self._rest_frame is not None:
            world.recover(self._rest_frame)
            world.retrieve()
        else:
            self.scene.reset()
        for entry in self.actors:
            entry["prev"] = np.asarray(entry["entity"].get_state().pos.cpu()).reshape(-1, 3).copy()
        self.rig.mark_teleport()
        if self.feedback_enabled and self.apply_wrenches is not None:
            self.apply_wrenches(self._acc.zero())


def _surface_of(tets: np.ndarray) -> np.ndarray:
    """The boundary triangles of a tetrahedral lattice: faces used exactly once."""
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    keyed = np.sort(faces, axis=1)
    _, index, counts = np.unique(keyed, axis=0, return_index=True, return_counts=True)
    return faces[index[counts == 1]]


def _entity_mass(entity) -> float | None:
    """An FEM entity's simulated mass, if this Genesis version exposes one."""
    for attr in ("mass", "total_mass", "_mass"):
        value = getattr(entity, attr, None)
        if value is not None:
            try:
                return float(np.asarray(value).sum())
            except (TypeError, ValueError):
                continue
    return None


def _build_shell(bridge: GenesisBridge, geom: Any) -> tuple[Any, Any]:
    """A membrane: a triangle mesh and Genesis's cloth material."""
    import trimesh

    gs, params = bridge.gs, geom.params
    mesh = trimesh.Trimesh(vertices=geom.rest, faces=geom.tris, process=False)
    thickness = float(params.get("thickness", 0.002))
    # This sheet's own moduli first, the island's only as the fallback -- the
    # same order ``SuperdexBridge`` reads them in. It was the recipe alone
    # here, so a fabric that stated ``shell_young`` got it on one engine and
    # silently did not on the other, and two sheets of different cloth in one
    # Genesis island were identically stiff.
    material = gs.materials.FEM.Cloth(
        E=float(params.get("young", bridge.recipe.get("shell_young", 1.0e5))),
        nu=float(params.get("poisson", bridge.recipe.get("shell_poisson", 0.3))),
        rho=cloth_density(float(params.get("areal_density", 0.3)), thickness),
        thickness=thickness,
        bending_stiffness=float(bridge.recipe.get("bending_stiffness", 10.0)),
    )
    return mesh, material


def _build_tet(bridge: GenesisBridge, geom: Any) -> tuple[Any, Any]:
    """A volume: its surface as the morph, its own lattice instantiated after."""
    import trimesh

    gs = bridge.gs
    mesh = trimesh.Trimesh(vertices=geom.rest, faces=_surface_of(geom.tets), process=False)
    material = gs.materials.FEM.Elastic(
        E=float(geom.young),
        nu=float(geom.poisson),
        rho=float(geom.params.get("density", 60.0)),
        model="stable_neohookean",
    )
    return mesh, material


register_actor_builder(
    "genesis",
    "shell",
    _build_shell,
    consumes=("areal_density", "poisson", "thickness", "young"),
    summary="Genesis builds a cloth (FEM.Cloth) from a shell actor.",
)
register_actor_builder(
    "genesis",
    "tet",
    _build_tet,
    consumes=("density", "poisson", "young"),
    summary="Genesis builds an elastic volume (FEM.Elastic) from a tet actor.",
)
