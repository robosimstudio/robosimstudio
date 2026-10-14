# SPDX-License-Identifier: Apache-2.0
"""Manipulable rigid bodies: the things a task is about moving.

Two ways one arrives:

* **Declared by the scene asset.** A benchmark's merged export already contains
  its objects with their free joints, so the spec only names them and this
  entity *finds* them -- resolving the body, its free-joint coordinates, and
  the pose the exporter recorded them settled at.
* **Imported per object.** A task assembling its own scene gives each object an
  asset, and this adds a body with a free joint and the geometry.

Either way the entity ends up holding the same three things: a body index, the
free joint's coordinate start (so a reset writes a pose without touching
anything else), and an OBB grasp box computed once at build time -- rigid, so
the box is fixed and only the pose changes.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from robosimstudio.paths import resolve_asset, resolve_path
from robosimstudio.scene.build.bodies import preserve_body_inertia, set_body_mass
from robosimstudio.scene.build.colliders import (
    build_hydroelastic_sdf,
    collider_vertices,
    copy_shape_material,
    slab_decomposition,
)
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.solvers import ObbGrasp, compute_obb
from robosimstudio.specs import ObjectSpec

__all__ = ["EntityRigid"]


class EntityRigid(Entity):
    """One manipulable rigid body.

    Args:
        spec: What object this is.
        scene_labels: Whether to find the body in an already-imported scene
            (``True``) or import it (``False``).
        gripper_span: Finger travel the OBB fit test uses [m].
        world: Which world.
    """

    spec: ObjectSpec

    def handle(self, model):
        """This body's place in the model: indices, grasp box, bounding box.

        It is a method now, so the contract is the method.
        """
        from robosimstudio.scene.handles import EntityHandle, split_shape_flags

        visual, collision = split_shape_flags(model, self.shapes)
        return EntityHandle(
            name=self.name,
            category=self.spec.category,
            body=self.body,
            coord_start=self.coord_start,
            shapes=self.shapes,
            visual_shapes=visual,
            collision_shapes=collision,
            obb=self.obb,
            bbox=self.bbox,
            static=self.spec.static,
        )

    def __init__(self, spec: ObjectSpec, *, gripper_span: float = 0.08, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.gripper_span = float(gripper_span)
        self.body: int = -1
        self.coord_start: int = -1
        self.dof_start: int = -1
        self.shapes: np.ndarray = np.zeros(0, dtype=np.int32)
        self.obb: ObbGrasp | None = None
        self.init_pose = np.asarray(spec.pose, dtype=np.float64)
        self.init_quat = np.asarray(spec.quat, dtype=np.float64)
        # What every episode's layout is measured from, set once. ``init_pose``
        # and ``init_quat`` follow the episode last placed; measuring from them
        # would carry that episode's pose into the next one that does not name
        # this object.
        self._declared = np.concatenate([self.init_pose, self.init_quat])
        self.bbox: np.ndarray | None = None if spec.bbox is None else np.asarray(spec.bbox, dtype=np.float64)

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        pose = self._placed(self._episode_pose(episode))
        if self.spec.asset is None and self.spec.shape is not None:
            self._add_primitive(builder, pose)
            return
        if self.spec.asset is None:
            # Declared by the scene asset; nothing to add, only to find.
            return
        path = resolve_asset(self.spec.asset)
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        if path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
            # Sets self.body and, for a free object, the joint coordinates --
            # which body it even is depends on what the stage declares.
            self._build_usd(builder, path, xform)
        elif path.suffix.lower() in (".obj", ".stl", ".ply", ".glb", ".gltf", ".dae"):
            if self.spec.static:
                body = builder.add_link(xform=xform, label=self.spec.name)
            else:
                # add_body creates the free joint AND the articulation. The
                # articulation half is load-bearing: without one SolverMuJoCo takes
                # the standalone-world-root fallback for the body and never
                # integrates it -- the object hangs at its spawn pose with velocity
                # exactly zero, which reads as "settled" until something expects it
                # to fall.
                body = builder.add_body(xform=xform, label=self.spec.name)
                # Record the free joint's coordinates now: only the scene-declared
                # path goes through find_in_scene, and without these a reset
                # returns early and the object keeps whatever pose the last episode
                # left it in -- measured as episode 1 scoring an insertion at step
                # 1 because episode 0's plug was still seated.
                joint = builder.joint_count - 1
                self.coord_start = int(builder.joint_q_start[joint])
                self.dof_start = int(builder.joint_qd_start[joint])
            self.body = body

            import trimesh

            mesh = trimesh.load(str(path), force="mesh")
            cfg = builder.default_shape_cfg.copy()
            cfg.density = float(self.spec.density)
            if self.spec.friction is not None:
                cfg.mu = float(self.spec.friction)
            if self.spec.contact_gap is not None:
                cfg.gap = float(self.spec.contact_gap)
            # The contact model an OBJ cannot carry and the MJCF beside it can.
            # See ObjectSpec.ke: without these the shape lands on the builder's
            # 2500 N/m and a driven gripper presses it through the table.
            if self.spec.ke is not None:
                cfg.ke = float(self.spec.ke)
            if self.spec.kd is not None:
                cfg.kd = float(self.spec.kd)
            shape_mesh = newton.Mesh(
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
                **self._texturing(mesh),
            )
            if self.spec.hydroelastic:
                # Named separately from the add_shape_mesh call because the SDF
                # is cooked into the mesh *before* it is handed over, and at the
                # same scale: a field cooked at x1 under a shape built at x2
                # reports half the distances it should. See ObjectSpec.
                cfg.is_hydroelastic = True
                cfg.kh = float(self.spec.kh)
                build_hydroelastic_sdf(
                    shape_mesh,
                    scale=(self.spec.asset.scale,) * 3,
                    target_voxel_size=self.spec.sdf_voxel,
                )
            visual_shape = builder.add_shape_mesh(
                body,
                mesh=shape_mesh,
                scale=(self.spec.asset.scale,) * 3,
                cfg=cfg,
                label=self.spec.name,
            )
            if self.spec.asset.material_colors:
                self._add_material_visuals(builder, body, path, cfg, visual_shape)
            marker_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
                has_particle_collision=False,
            )
            for index, (position, radius, color) in enumerate(self.spec.visual_markers):
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(tuple(float(v) for v in position), wp.quat(0.0, 0.0, 0.0, 1.0)),
                    radius=float(radius),
                    cfg=marker_cfg,
                    color=color,
                    label=f"{self.spec.name}_marker_{index:02d}",
                )
            proxy_paths = ((self.spec.asset.collision_path,) if self.spec.asset.collision_path else ()) + tuple(
                self.spec.asset.collision_paths
            )
            if proxy_paths:
                # Keep the authored mesh and its calibrated mass/inertia, but
                # route collision through lightweight task-owned proxies. The
                # stand-ins must neither render nor add another copy of mass.
                if self.spec.asset.approximate or self.spec.collider:
                    raise ValueError(
                        f"object {self.spec.name!r}: collision proxies cannot be combined with "
                        f"approximate={self.spec.asset.approximate!r} or collider={self.spec.collider!r} -- "
                        "the approximation would swallow the proxies into the very hull they exist "
                        "to avoid, and a built-in collider is a second answer to the same question"
                    )
                builder.shape_flags[visual_shape] = int(builder.shape_flags[visual_shape]) & ~int(
                    newton.ShapeFlags.COLLIDE_SHAPES
                )
                for index, proxy_asset_path in enumerate(proxy_paths):
                    # A collision proxy is already an authored simulation mesh.
                    # Preserve its explicit vertex/face indexing: trimesh's normal
                    # processing can merge coincident vertices after decimation and
                    # turn separate surface layers into non-manifold contact seams.
                    proxy_mesh = trimesh.load(str(resolve_path(proxy_asset_path)), force="mesh", process=False)
                    label = f"{self.spec.name}_collision"
                    if len(proxy_paths) > 1:
                        label = f"{label}_{index:02d}"
                    with preserve_body_inertia(builder, body):
                        proxy_shape = builder.add_shape_mesh(
                            body,
                            mesh=newton.Mesh(
                                np.asarray(proxy_mesh.vertices, dtype=np.float32),
                                np.asarray(proxy_mesh.faces, dtype=np.int32).reshape(-1),
                            ),
                            scale=(self.spec.asset.scale,) * 3,
                            cfg=cfg,
                            label=label,
                        )
                    builder.shape_flags[proxy_shape] = int(builder.shape_flags[proxy_shape]) & ~int(
                        newton.ShapeFlags.VISIBLE
                    )
                    copy_shape_material(builder, visual_shape, proxy_shape)
        else:
            raise ValueError(f"object {self.spec.name!r}: unsupported asset {path.suffix!r}")
        if self.spec.asset.approximate:
            targets = list(range(self.range.shapes[0], builder.shape_count))
            builder.approximate_meshes(
                method=self.spec.asset.approximate,
                shape_indices=targets,
                keep_visual_shapes=self.spec.asset.keep_visual,
            )
        # After the approximation, which is itself a mass event: swapping a mesh
        # collider for a box re-derives the body's mass from the box. A stated
        # mass is the source's, and has to be the last word on it.
        if self.spec.mass is not None and not self.spec.static:
            set_body_mass(builder, self.body, float(self.spec.mass))

    def _texturing(self, mesh) -> dict:
        """``uvs`` and ``texture`` for :class:`newton.Mesh`, or nothing.

        A vendored mesh carries its UVs and leaves its diffuse map in a sibling
        file, so the two are joined here: the coordinates come off the geometry
        and the image comes off :attr:`~robosimstudio.specs.AssetSpec.texture`.

        **A named texture with no UVs raises.** Passing an image through without
        coordinates to sample it at is the failure this exists to stop being
        silent -- the object arrives with a material, renders flat, and reads as
        a texture that did not load rather than as a mesh that cannot carry one.
        ``trimesh.load(force="mesh")`` drops UVs when it concatenates several
        materials, which is exactly when a vendored object needs re-exporting
        rather than shipping unnoticed.
        """
        if not self.spec.asset.texture:
            return {}
        uvs = getattr(getattr(mesh, "visual", None), "uv", None)
        if uvs is None or len(uvs) != len(mesh.vertices):
            raise ValueError(
                f"object {self.spec.name!r}: names texture {self.spec.asset.texture!r} but "
                f"{self.spec.asset.path!r} carries no per-vertex UVs to sample it at"
            )
        return {
            "uvs": np.asarray(uvs, dtype=np.float32),
            "texture": str(resolve_path(self.spec.asset.texture)),
        }

    def _add_material_visuals(self, builder, body: int, path, cfg, physics_shape: int) -> None:
        """Draw an OBJ's MTL sections without changing its collision or mass."""
        import trimesh

        scene = trimesh.load(str(path), force="scene", process=False)
        parts = []
        for node in scene.graph.nodes_geometry:
            if str(node).startswith("policy_"):
                continue
            transform, geometry = scene.graph.get(node)
            part = scene.geometry[geometry].copy()
            part.apply_transform(transform)
            diffuse = getattr(getattr(part.visual, "material", None), "diffuse", None)
            if diffuse is None:
                continue
            parts.append((str(geometry), part, np.asarray(diffuse, dtype=np.float64)[:3] / 255.0))
        if not parts:
            return

        builder.shape_flags[physics_shape] = int(builder.shape_flags[physics_shape]) & ~int(newton.ShapeFlags.VISIBLE)
        for index, (name, part, color) in enumerate(parts):
            with preserve_body_inertia(builder, body):
                shape = builder.add_shape_mesh(
                    body,
                    mesh=newton.Mesh(
                        np.asarray(part.vertices, dtype=np.float32),
                        np.asarray(part.faces, dtype=np.int32).reshape(-1),
                    ),
                    scale=(self.spec.asset.scale,) * 3,
                    cfg=cfg,
                    label=f"{self.spec.name}_material_{index:02d}_{name}",
                )
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            builder.shape_color[shape] = tuple(float(channel) for channel in color)

    def _build_usd(self, builder: newton.ModelBuilder, path, xform) -> None:
        """Import one object from a USD stage, either flavour.

        **The stage says which flavour it is.** A stage carrying
        ``PhysicsRigidBodyAPI`` is a rigid-body *description*: ``add_usd`` reads
        its mass, its collider approximation and its physics material and builds
        the body and free joint itself, so the stage decides what the object
        weighs and collides as. A stage with no physics schema anywhere is
        geometry, and is treated exactly as an ``.obj`` is -- this creates the
        body and the free joint, and the spec supplies density and collider.

        Asking the file rather than taking a flag is deliberate: the two are not
        interchangeable descriptions of one object, they are different claims,
        and a spec that disagreed with its own asset could then only be found by
        noticing that the physics was wrong. ``add_usd`` on a geometry-only
        stage does not fail -- it puts the mesh on the *world* body, where it is
        scenery that renders and never moves.

        Args:
            builder: The builder to import into.
            path: The resolved stage path.
            xform: Where this episode wants the object.
        """
        # pxr is the [usd] extra, so it is imported where it is used.
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError(f"object {self.spec.name!r}: could not open {path}")
        describes_body = any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())

        if describes_body:
            if self.spec.static:
                raise ValueError(
                    f"object {self.spec.name!r}: {path.name} declares a rigid body, "
                    f"so it cannot also be static; strip the physics schemas or drop static=True"
                )
            before = builder.body_count
            results = builder.add_usd(
                str(path),
                xform=xform,
                floating=True,
                # Left False so the stage's own ``physics:approximation`` is
                # honoured -- that approximation is the claim this flavour
                # exists to test. True means the opposite of what it sounds
                # like here: *keep the authored mesh* and ignore the collider
                # the stage asked for, which is why the grocery tote sets it.
                skip_mesh_approximation=self.spec.asset.skip_mesh_approximation,
            )
            bodies = sorted(int(b) for b in (results.get("path_body_map") or {}).values())
            if len(bodies) != 1:
                raise ValueError(
                    f"object {self.spec.name!r}: {path.name} declares {len(bodies)} rigid bodies; "
                    f"an object asset must declare exactly one"
                )
            self.body = bodies[0]
            if self.body < before:
                raise ValueError(f"object {self.spec.name!r}: {path.name} reused body {self.body}")
            joint = builder.joint_count - 1
            self.coord_start = int(builder.joint_q_start[joint])
            self.dof_start = int(builder.joint_qd_start[joint])
            return

        import newton.usd

        self.body = builder.add_body(xform=xform, label=self.spec.name)
        joint = builder.joint_count - 1
        self.coord_start = int(builder.joint_q_start[joint])
        self.dof_start = int(builder.joint_qd_start[joint])
        cfg = builder.default_shape_cfg.copy()
        cfg.density = float(self.spec.density)
        if self.spec.friction is not None:
            cfg.mu = float(self.spec.friction)
        if self.spec.contact_gap is not None:
            cfg.gap = float(self.spec.contact_gap)
        builder.add_shape_mesh(
            self.body,
            mesh=newton.usd.get_mesh(str(path), load_uvs=True),
            scale=(self.spec.asset.scale,) * 3,
            cfg=cfg,
            label=self.spec.name,
        )

    def _placed(self, pose: np.ndarray) -> np.ndarray:
        """One start pose in world coordinates, respecting who placed it.

        An object with its own asset is *this task's*, so its spec states the
        pose in the scene's work-surface frame and the frame composes on. An
        object with no asset was declared by the scene's own import and is
        already where that import put it -- composing the frame onto it moves
        it by the surface origin, which is how a tool-sorting scene's bins ended
        up (+0.100, +0.015) from their export's compartments.

        :attr:`~robosimstudio.specs.ObjectSpec.world_xform` is the third
        case: an object that carries its own asset but whose pose came out of an
        export placed at world, so it is positioned like a scene-declared one
        despite being imported like a task-supplied one.

        And a fourth, which is why "no asset" is not the test any more: an object
        drawn from a :class:`~robosimstudio.specs.asset.ShapeSpec` has no asset
        *and* is placed right here, so it takes the frame like any other
        task-supplied object. Keying on ``asset is None`` alone put a procedural
        cube at the surface frame's origin instead of on the bench -- inside the
        table, which it then fell through to the floor.
        """
        if self.spec.world_xform:
            return pose
        if self.spec.asset is None and self.spec.shape is None:
            # Declared by the scene's own import: already where that import put it.
            return pose
        return self.placed(pose)

    def episode_start_q(self, episode, *, base: np.ndarray | None = None) -> np.ndarray:
        """This object's world-coordinate start pose for one episode.

        What :meth:`reset` writes, without writing it or remembering it. A fleet
        reset needs the pose for several episodes at once -- one per world -- and
        cannot let each one overwrite the base the next is measured from, so it
        supplies that base explicitly and keeps its own.

        Args:
            episode: The episode whose layout to place this object for.
            base: The 7-vector to treat as the object's own declared pose. The
                entity's current one when omitted.
        """
        return self._placed(self._episode_pose(episode, base))

    @property
    def declared_q(self) -> np.ndarray:
        """The pose this object measures its episode layouts from: the spec's own."""
        return self._declared.copy()

    def _add_primitive(self, builder: newton.ModelBuilder, pose) -> None:
        """A body made from a primitive, with no file behind it.

        The same body a mesh object gets -- ``add_body`` creates the free joint
        *and* the articulation, and the joint's coordinates are recorded here for
        the same reason the mesh path records them: without them a reset returns
        early and the object keeps whatever pose the last episode left it in.
        """
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        if self.spec.static:
            body = builder.add_link(xform=xform, label=self.spec.name)
        else:
            body = builder.add_body(xform=xform, label=self.spec.name)
            joint = builder.joint_count - 1
            self.coord_start = int(builder.joint_q_start[joint])
            self.dof_start = int(builder.joint_qd_start[joint])
        self.body = body
        self.spec.shape.add_to(
            builder,
            body=body,
            xform=wp.transform_identity(),
            label=self.spec.name,
            cfg=self._contact_cfg(builder),
        )
        if self.spec.mass is not None:
            set_body_mass(builder, body, float(self.spec.mass))

    def _contact_cfg(self, builder: newton.ModelBuilder):
        """The builder default amended by whatever this object *declared*.

        Both mesh paths above have always done this; the primitive path did not,
        and passed no ``cfg`` at all -- so a box object's ``friction``, ``ke``,
        ``kd`` and ``contact_gap`` were read off the spec by nobody and the shape
        landed on the arrangement's defaults. Silent, and the kind of silent that
        reads as a physics mystery: polish/strip's document said ``friction=0.5``
        on its procedural box while every grip in the suite was measured against
        the builder's own 1.0, so its stated contact model and its numbers were
        about different scenes. The same hole swallowed
        ``rss.box(..., friction=0.3)`` on the public core API.
        """
        cfg = builder.default_shape_cfg.copy()
        if self.spec.friction is not None:
            cfg.mu = float(self.spec.friction)
        if self.spec.contact_gap is not None:
            cfg.gap = float(self.spec.contact_gap)
        if self.spec.ke is not None:
            cfg.ke = float(self.spec.ke)
        if self.spec.kd is not None:
            cfg.kd = float(self.spec.kd)
        return cfg

    def _episode_pose(self, episode, base: np.ndarray | None = None) -> np.ndarray:
        """This object's start pose for one episode, as a 7-vector.

        An episode sampler reports ``object_poses[name] = (x, y, z, yaw)``,
        which is the layout family every tabletop task varies over; anything
        richer overrides the full transform under ``object_xforms``.
        """
        pose = self.declared_q if base is None else np.asarray(base, dtype=np.float64).reshape(-1)[:7]
        if episode is None:
            return pose
        xforms = episode.get("object_xforms") or {}
        if self.spec.name in xforms:
            return np.asarray(xforms[self.spec.name], dtype=np.float64)
        poses = episode.get("object_poses") or {}
        if self.spec.name in poses:
            x, y, z, yaw = (float(v) for v in poses[self.spec.name])
            half = yaw * 0.5
            pose = np.array([x, y, z, 0.0, 0.0, np.sin(half), np.cos(half)])
        return pose

    def after_build(self, builder: newton.ModelBuilder, episode) -> None:
        """Find, collect, approximate, box, pose -- in that order."""
        # `asset is None` is two different objects, and treating them as one
        # quietly threw away the shapes a primitive had just built. `build`
        # already draws the line: no asset *and* a shape is a procedural box this
        # entity added itself; no asset and no shape is an object the scene's
        # arena declared and this only has to locate. `find_in_scene` is for the
        # second, and it ends by resetting `range.shapes` to (0, 0) -- so for the
        # first it erased a range that was already correct, and nothing refilled
        # it. Measured: polish's strip carries `color=STRIP_COLOR` and rendered
        # in the build-order palette's magenta, because `apply_display` iterates
        # that range and it was empty.
        if self.spec.asset is None and self.spec.shape is None:
            self.find_in_scene(builder)
        self.collect_shapes(builder)
        # Whether an object's collider is approximated is the *spec's* call, not
        # a side effect of wanting a grasp box: a bin of thin tools needs boxes
        # to settle at all, and a can boxified acquires a 45-degree yaw it does
        # not have. See ObjectSpec.collider.
        #
        # An *imported* object may say it here too, and has to be able to.
        # `AssetSpec.approximate` reaches only what `approximate_meshes` offers,
        # which is an oriented box or a hull -- and neither is what an upright
        # grocery wants, since the oriented fit inflates a round cross-section
        # and the hull does not stop the creep. `"aabb"` and `"slabs"` are built
        # in `approximate_collider` and work off the shapes either path produced.
        if self.spec.collider:
            if self.spec.asset is not None and self.spec.asset.approximate:
                raise ValueError(
                    f"object {self.spec.name!r}: states both asset.approximate="
                    f"{self.spec.asset.approximate!r} and collider={self.spec.collider!r}. "
                    "These are two opinions about one collider; state one."
                )
            self.approximate_collider(builder, self.spec.collider)
            self.collect_shapes(builder)
        self.compute_grasp_box(builder)
        self.write_initial_q(builder, episode)

    def find_in_scene(self, builder: newton.ModelBuilder) -> None:
        """Locate an object the scene asset already declared.

        Called by the compiler after the statics have built, since the object
        only exists once its arena has been imported.
        """
        body_name = self.spec.body_name or self.spec.name
        self.body = label_index(builder.body_label, body_name)
        if not self.spec.static:
            j = self._free_joint(builder)
            if newton.JointType(builder.joint_type[j]) is not newton.JointType.FREE:
                raise ValueError(f"object {self.spec.name!r}: expected a FREE joint, got {builder.joint_type[j]}")
            self.coord_start = int(builder.joint_q_start[j])
            self.dof_start = int(builder.joint_qd_start[j])
        self.range.bodies = (self.body, self.body + 1)
        self.range.shapes = (0, 0)  # filled by collect_shapes
        if self.spec.friction is not None:
            # Unlike EntityStatic.build, this cannot lean on a freshly-added
            # shape range: a resident object's shapes were already in the
            # builder before this method ran, imported whole with the arena.
            # A geom's own authored friction is not a reliable source to
            # leave this at -- the importer does not carry it onto
            # shape_material_mu the way EntityRigid.build's cfg.mu does for a
            # freshly loaded mesh -- so an override here is the only way a
            # scene-resident object's grip is ever anything but the
            # builder's bare default.
            for s in range(builder.shape_count):
                if int(builder.shape_body[s]) == self.body:
                    builder.shape_material_mu[s] = float(self.spec.friction)

    def _free_joint(self, builder: newton.ModelBuilder) -> int:
        """This object's free joint: the named one, else the one it is the child of.

        A joint is identified by its child body, and only secondarily by a name.
        That distinction is not academic -- MJCF's importer labels a free joint
        after the body it belongs to, so a name works there, while USD's invents
        ``joint_19`` for a body with no authored joint at all. Naming those in a
        suite file would be transcribing an importer's counter, which is not an
        interface: adding one object to the stage renumbers the rest.

        So an explicit :attr:`~robosimstudio.specs.ObjectSpec.joint_name`
        still wins -- a merged export that records its joint names is stating
        something worth honouring -- and otherwise the child body decides, with
        the ``<name>_joint0`` convention as the last resort.
        """
        if self.spec.joint_name:
            return label_index(builder.joint_label, self.spec.joint_name)
        for j in range(builder.joint_count):
            if int(builder.joint_child[j]) == self.body and newton.JointType(builder.joint_type[j]) is (
                newton.JointType.FREE
            ):
                return j
        return label_index(builder.joint_label, f"{self.spec.name}_joint0")

    def collect_shapes(self, builder: newton.ModelBuilder) -> None:
        """Record which shapes belong to this body, for contact queries."""
        self.shapes = np.asarray(
            [s for s in range(builder.shape_count) if builder.shape_body[s] == self.body], dtype=np.int32
        )

    def approximate_collider(self, builder: newton.ModelBuilder, method: str) -> int:
        """Swap this object's mesh colliders for an approximation of them.

        **What a flat-faced collider buys, measured.** A curved or many-vertex
        collider resting on a flat surface gives the solver a contact set whose
        chosen points flicker between steps, and the residual tangential impulse
        makes the object *creep*: on the grocery scene, five items nominally at
        rest slid 0.67-3.25 mm in xy over two seconds, with z steady -- sliding,
        not bouncing. Boxes cut that to 0.047-0.136 mm, about 25x. (A convex hull
        does not help: 0.6-4.0 mm, no better than the authored mesh.)

        Three methods, and the difference between the last two is the whole reason
        this is a spec field rather than a flag:

        ``"bounding_box"``
            :meth:`~newton.ModelBuilder.approximate_meshes`'s *oriented* box fit.
            Right for a thin tool at an angle in a bin, whose natural box is not
            aligned with its body frame. Wrong for anything round: the fit's yaw
            about a circular cross-section is arbitrary, and the body-frame bound
            of a box rotated 45 degrees is sqrt(2) larger than the mesh's -- which
            inflated a 71 mm can's grasp box to 96 mm, wider than the gripper's
            whole stroke, so it read as ungraspable.
        ``"aabb"``
            One box at the mesh's **body-frame** axis-aligned bound, built here
            rather than fitted. Flat faces, so it settles like a box, and extents
            identical to the mesh's own, so the grasp box is unchanged by the swap.
            What an upright grocery wants.
        ``"shell"``
            A floor and four walls around the object's declared
            :attr:`~robosimstudio.specs.ObjectSpec.cavity`, for a
            container that has an inside. The one method here that makes the
            collider a *different shape* rather than a coarser one, and the
            reason it has to: MuJoCo convexifies a mesh, so a basket imported as
            authored is a sealed block that nothing can be put into.
        ``"drawn_hull"``
            The convex hull of the mesh this object *draws*, replacing whatever
            it collides with. The one arm here that does not need a mesh
            collider to start from, and the only one that can reach an object
            whose collider was fitted by somebody else -- see
            :meth:`_add_drawn_collider`.
        ``"drawn_slabs"``
            The drawn mesh cut into a **stack of boxes** rather than wrapped in
            one hull. A hull is a single convex solid, so for anything that
            narrows in the middle it is a taper joining the two widest points --
            and a taper is the one thing a parallel jaw cannot hold, because
            sliding along it changes the width the fingers closed to. Measured on
            a tool-sorting scene's syringe, whose hull runs 0.4 mm at the needle to
            23.6 mm at the flange: the hull sheds **0.37 mm of width per mm of
            slide**, so 6.2 mm of travel takes the grip to zero. The slabs return
            a 94 mm barrel at a **constant** 10.7 mm with a 23.6 mm flange block
            above it, which is a stop rather than a slope. Boxes also earn a
            contact manifold where a hull does not -- mujoco_warp's multicontact
            gate takes box pairs -- so this does not depend on ``enable_multiccd``
            the way ``"drawn_hull"`` does.
        anything else
            Passed through to ``approximate_meshes``.

        The mesh shapes are kept in every case -- visible, and still counted by
        :meth:`compute_grasp_box` -- so the swap is invisible on camera. Because
        they are kept, the body's mass is already accounted for and the substitute
        must not add its own; see
        :func:`~robosimstudio.scene.build.bodies.preserve_body_inertia` for the 1.8-3.5x
        overweight that not holding it produced.

        Args:
            builder: The builder, pre-finalize.
            method: ``"aabb"``, or any method ``approximate_meshes`` takes.

        Returns:
            How many shapes were replaced.
        """
        if method == "drawn_hull":
            # Before the guard below, which asks for a mesh *collider*: these arms
            # exist precisely for the object that has none.
            return self._add_drawn_collider(builder)
        if method == "drawn_slabs":
            return self._add_drawn_slab_collider(builder)
        targets = [
            s
            for s in self.shapes
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES))
        ]
        if not targets:
            return 0
        if method == "slabs":
            return self._add_slab_collider(builder, targets)
        if method == "shell":
            return self._add_shell_collider(builder, targets)
        if method != "aabb":
            with preserve_body_inertia(builder, self.body):
                builder.approximate_meshes(method=method, shape_indices=targets, keep_visual_shapes=True)
            return len(targets)

        # The body-frame bound of what is there now, before anything is disabled.
        box = compute_obb(builder, self.body, builder.body_com[self.body], gripper_span=self.gripper_span)
        collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for s in targets:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collides
        with preserve_body_inertia(builder, self.body):
            added = builder.add_shape_box(
                body=self.body,
                xform=wp.transform(tuple(float(v) for v in box.center_local), wp.quat_identity()),
                hx=float(box.half[0]),
                hy=float(box.half[1]),
                hz=float(box.half[2]),
                label=f"{self.spec.name}_aabb",
            )
        # Collision geometry standing in for a mesh that is still drawn, so it
        # must not be drawn itself.
        builder.shape_flags[added] = int(builder.shape_flags[added]) & ~int(newton.ShapeFlags.VISIBLE)
        # And a substitute for its *material*, not just its shape: add_shape_box
        # takes the builder's default, which is mu 1.0 against the groceries'
        # authored 2.0. Losing that halved the friction at every contact the object
        # makes, and a grasped can slid 27 mm in the fingers on a lift.
        copy_shape_material(builder, targets[0], added)
        return len(targets)

    def _add_drawn_collider(self, builder: newton.ModelBuilder) -> int:
        """Collide on the mesh this object draws, whatever it collided on before.

        The ``"drawn_hull"`` arm of :meth:`approximate_collider`, and one of the
        two that start from the *visual* geometry -- see
        :meth:`_add_drawn_slab_collider` for the other. Every other arm replaces a mesh
        collider with something coarser; this one replaces a collider somebody
        else already fitted with something that matches the picture.

        **The case it is for is an export that shipped its own approximation.**
        VAB-industrial's scene authors each tool as one oriented box with
        ``contype=0 conaffinity=0`` on the mesh beside it, so the mesh is drawn
        and never touched and there is no mesh collider for the other arms to
        find -- ``approximate_collider`` returned 0 and the box stayed. Measured
        as the largest distance, over every heading a top-down jaw could close
        along, that the collider stands proud of the mesh it is drawn as::

            object                 authored box     drawn hull
            drill                     27.1 mm         0.00 mm
            clamp                     26.2            0.00
            adjustable_wrench         19.5            0.00
            pliers                    16.8            0.00
            utility_knife             15.9            0.00
            phillips_screwdriver       8.9            0.00

        Two centimetres of invisible shell is a gripper that stops in mid-air and
        a linkage that deflects against nothing, which is what this is for.

        **It departs from the export, deliberately, and only where the export was
        approximating.** The hammer and the round nut are untouched by it because
        they were already right -- upstream authored them as real primitives, a
        cylinder handle and a box head, and those measure 0.00 mm proud as they
        stand. Nothing here fits a shape; it takes the hull of the geometry the
        file itself ships as the object's appearance.

        Everything the ``"aabb"`` arm does applies unchanged, for the same
        reasons: the drawn mesh stays drawn, the substitute is invisible, it takes
        its material from the collider it replaces rather than the builder's
        default, and the body's inertia is held across the swap.

        A hull is a curved collider, and this class's own note records that one
        creeps. It did, on one contact point:
        the ``mujoco`` backend's ``enable_multiccd`` parameter is what
        makes it a manifold, and a suite stating this collider has to state that
        flag -- 44.8 mm of creep over 30 s of idle without it against 0.20 mm
        with it, measured on ``astro/heap_sort``.

        Returns:
            How many colliders were replaced; 0 when the object draws no mesh, so
            that an object this cannot help is left exactly as it was.
        """
        collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible = int(newton.ShapeFlags.VISIBLE)
        drawn = [
            s
            for s in self.shapes
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & visible)
            and not (builder.shape_flags[s] & collides)
        ]
        replaced = [s for s in self.shapes if builder.shape_flags[s] & collides]
        if not drawn or not replaced:
            return 0

        added: list[int] = []
        with preserve_body_inertia(builder, self.body):
            for i, source in enumerate(drawn):
                xf = builder.shape_transform[source]
                shape = builder.add_shape_mesh(
                    body=self.body,
                    xform=wp.transform(wp.vec3(*(float(v) for v in xf[:3])), wp.quat(*(float(v) for v in xf[3:7]))),
                    mesh=builder.shape_source[source],
                    scale=tuple(float(v) for v in builder.shape_scale[source][:3]),
                    label=f"{self.spec.name}_drawn_hull{i}",
                )
                added.append(shape)
            builder.approximate_meshes(method="convex_hull", shape_indices=added)
        for shape in added:
            builder.shape_flags[shape] = (int(builder.shape_flags[shape]) | collides) & ~visible
            copy_shape_material(builder, replaced[0], shape)
        for s in replaced:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collides
        return len(replaced)

    def _add_drawn_slab_collider(self, builder: newton.ModelBuilder) -> int:
        """Collide on a stack of boxes cut from the mesh this object draws.

        The ``"drawn_slabs"`` arm of :meth:`approximate_collider`:
        :meth:`_add_drawn_collider`'s reach -- an export that fitted its own
        collider and left the mesh untouchable -- with
        :meth:`_add_slab_collider`'s shape.

        **What it is for is the object a hull cannot describe.** A convex hull is
        one solid, so an object narrow in the middle and wide at one end comes
        back as the taper between them. A tool-sorting scene's syringe is the case that
        forced this: 0.4 mm at the needle, 10.7 mm along the barrel, 23.6 mm
        across the plunger flange, and a hull that interpolates straight between
        the last two because the drawn mesh carries no vertices in the 60 mm
        between. The jaws then close on a slope, and a slope moves under them --
        0.37 mm of width lost per mm of travel, so 6.2 mm of slide takes the grip
        to zero. Slabs give the barrel one width over its whole length and put
        the flange above it as a block, which is a stop and not a slope.

        Everything :meth:`_add_drawn_collider` does applies unchanged: the drawn
        mesh stays drawn, the substitute is invisible, it takes its material from
        the collider it replaces rather than the builder's default, and the body's
        inertia is held across the swap.

        Returns:
            How many colliders were replaced; 0 when the object draws no mesh, so
            that an object this cannot help is left exactly as it was.
        """
        collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible = int(newton.ShapeFlags.VISIBLE)
        drawn = [
            s
            for s in self.shapes
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & visible)
            and not (builder.shape_flags[s] & collides)
        ]
        replaced = [s for s in self.shapes if builder.shape_flags[s] & collides]
        if not drawn or not replaced:
            return 0

        boxes, _axis = slab_decomposition(collider_vertices(builder, drawn))
        if not boxes:
            return 0

        added: list[int] = []
        with preserve_body_inertia(builder, self.body):
            for i, (centre, half) in enumerate(boxes):
                added.append(
                    builder.add_shape_box(
                        body=self.body,
                        xform=wp.transform(tuple(float(v) for v in centre), wp.quat_identity()),
                        hx=float(half[0]),
                        hy=float(half[1]),
                        hz=float(half[2]),
                        label=f"{self.spec.name}_drawn_slab{i}",
                    )
                )
        for shape in added:
            builder.shape_flags[shape] = (int(builder.shape_flags[shape]) | collides) & ~visible
            copy_shape_material(builder, replaced[0], shape)
        for s in replaced:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collides
        return len(replaced)

    def _add_slab_collider(self, builder: newton.ModelBuilder, targets: list[int]) -> int:
        """Replace this object's mesh colliders with a stack of boxes.

        The ``"slabs"`` arm of :meth:`approximate_collider`. Everything the
        ``"aabb"`` arm does -- disable the mesh colliders, keep them visible, take
        the substitute's material from the mesh rather than the builder default,
        hold the body's inertia across the swap -- applies identically here and
        for the same reasons; the only difference is how many boxes come out.

        :func:`~robosimstudio.scene.build.colliders.slab_decomposition` merges bands whose
        cross-sections agree, so an upright can yields one box and this reduces to
        the AABB it already had. A mustard bottle or a gable-top carton yields
        several and stops being a full-width block up to its cap.
        """
        vertices = collider_vertices(builder, targets)
        boxes, _axis = slab_decomposition(vertices)
        if not boxes:
            # Nothing measurable to fit. Falling through to the AABB is better
            # than leaving the mesh colliders in place, which is what creeps.
            return self.approximate_collider(builder, "aabb")

        collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for s in targets:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collides

        added: list[int] = []
        with preserve_body_inertia(builder, self.body):
            for i, (centre, half) in enumerate(boxes):
                shape = builder.add_shape_box(
                    body=self.body,
                    xform=wp.transform(tuple(float(v) for v in centre), wp.quat_identity()),
                    hx=float(half[0]),
                    hy=float(half[1]),
                    hz=float(half[2]),
                    label=f"{self.spec.name}_slab{i}",
                )
                added.append(shape)
        for shape in added:
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.VISIBLE)
            copy_shape_material(builder, targets[0], shape)
        return len(targets)

    def _add_shell_collider(self, builder: newton.ModelBuilder, targets: list[int]) -> int:
        """Replace this object's mesh colliders with a floor and four walls.

        The ``"shell"`` arm of :meth:`approximate_collider`, and the one that
        exists because a container's collider has to be a *different shape* and
        not merely a coarser one. Every other method here is an outer bound; a
        basket's outer bound is exactly what makes it useless.

        The rectangle is the object's declared
        :attr:`~robosimstudio.specs.ObjectSpec.cavity`; the outer bound
        is the mesh's own, read off the shapes rather than declared, so a
        re-vendored mesh cannot leave the shell describing the old one. The
        generalisation of ``the retired grocery_stage module``'s container shelling,
        which does the same five boxes for one tote out of a sidecar.

        Everything the ``"aabb"`` arm does applies unchanged: the mesh shapes are
        kept and stay visible, the substitute takes the mesh's material rather
        than the builder's default, and the body's inertia is held across the
        swap -- five solid boxes standing in for a hollow shell have nothing like
        its volume, and not holding it took the grocery tote from 1.9 kg to
        6.5 kg.
        """
        cavity = self.spec.cavity
        if cavity is None:
            raise ValueError(
                f"object {self.spec.name!r}: collider 'shell' needs a cavity to leave open, and the spec "
                "states none. A shell without one is an outer box, which is what it exists not to be."
            )
        x0, y0, z0, x1, y1, z1 = (float(v) for v in cavity)
        vertices = collider_vertices(builder, targets)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)

        # Checked before anything is built, not after: a wall of negative
        # thickness is a box with a negative half-extent, which the builder
        # accepts and the solver reads inside-out. The container would then have
        # no walls at all, and the failure would look like a policy that cannot
        # aim.
        margins = (x0 - lo[0], hi[0] - x1, y0 - lo[1], hi[1] - y1, z0 - lo[2])
        if min(margins) <= 0.0:
            raise ValueError(
                f"object {self.spec.name!r}: the declared cavity {cavity} is not inside the mesh's own bound "
                f"{tuple(round(float(v), 5) for v in (*lo, *hi))}, so a wall would have negative thickness"
            )

        collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for s in targets:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collides

        added: list[int] = []

        def box(name: str, centre, half) -> None:
            added.append(
                builder.add_shape_box(
                    body=self.body,
                    xform=wp.transform(tuple(float(v) for v in centre), wp.quat_identity()),
                    hx=float(half[0]),
                    hy=float(half[1]),
                    hz=float(half[2]),
                    label=f"{self.spec.name}_{name}",
                )
            )

        # The walls run from the cavity floor to the cavity rim rather than to
        # the mesh's top: a basket's handles are above its lip, and a wall drawn
        # up to them would be a lid nothing can be dropped past.
        wall_centre_z = 0.5 * (z0 + z1)
        wall_half_z = 0.5 * (z1 - z0)
        with preserve_body_inertia(builder, self.body):
            box(
                "floor",
                (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + z0)),
                (0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), max(0.5 * (z0 - lo[2]), 1.0e-4)),
            )
            box(
                "wall_x_lo",
                (0.5 * (lo[0] + x0), 0.5 * (lo[1] + hi[1]), wall_centre_z),
                (0.5 * (x0 - lo[0]), 0.5 * (hi[1] - lo[1]), wall_half_z),
            )
            box(
                "wall_x_hi",
                (0.5 * (x1 + hi[0]), 0.5 * (lo[1] + hi[1]), wall_centre_z),
                (0.5 * (hi[0] - x1), 0.5 * (hi[1] - lo[1]), wall_half_z),
            )
            box(
                "wall_y_lo",
                (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + y0), wall_centre_z),
                (0.5 * (hi[0] - lo[0]), 0.5 * (y0 - lo[1]), wall_half_z),
            )
            box(
                "wall_y_hi",
                (0.5 * (lo[0] + hi[0]), 0.5 * (y1 + hi[1]), wall_centre_z),
                (0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - y1), wall_half_z),
            )

        visible = int(newton.ShapeFlags.VISIBLE)
        for shape in added:
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~visible
            copy_shape_material(builder, targets[0], shape)
        return len(targets)

    def compute_grasp_box(self, builder: newton.ModelBuilder) -> None:
        """Box this object for the grasp heuristic. Run *after* any collider
        approximation, so the box sees the shapes the gripper will actually hit."""
        if not self.spec.obb_grasp or self.body < 0:
            return
        com = np.asarray(builder.body_com[self.body], dtype=np.float64)
        self.obb = compute_obb(builder, self.body, com, gripper_span=self.gripper_span)
        if self.bbox is None:
            self.bbox = 2.0 * self.obb.half

    def write_initial_q(self, builder: newton.ModelBuilder, episode) -> None:
        """Seed ``builder.joint_q`` with this object's episode pose."""
        if self.coord_start < 0:
            return
        # The remembered start pose stays in the frame the *spec* stated it in,
        # the frame _episode_pose works in. It is a record of this episode's
        # start, not a base: the next episode is measured from ``declared_q``.
        raw = self._episode_pose(episode)
        self.init_pose, self.init_quat = raw[:3].copy(), raw[3:7].copy()
        pose = self._placed(raw)
        c = self.coord_start
        builder.joint_q[c : c + 3] = [float(v) for v in pose[:3]]
        builder.joint_q[c + 3 : c + 7] = [float(v) for v in pose[3:7]]

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        if self.body < 0:
            self.body = self.range.bodies[0]
        if not len(self.shapes):
            shape_body = model.shape_body.numpy()
            self.shapes = np.flatnonzero(shape_body == self.body).astype(np.int32)

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode) -> None:
        """Write this object's episode pose back into the state.

        Only its own coordinates are touched, so resetting one object never
        disturbs another or the arm.
        """
        if self.coord_start < 0:
            return
        # The remembered start pose stays in the frame the *spec* stated it in,
        # the frame _episode_pose works in. It is a record of this episode's
        # start, not a base: the next episode is measured from ``declared_q``.
        raw = self._episode_pose(episode)
        self.init_pose, self.init_quat = raw[:3].copy(), raw[3:7].copy()
        pose = np.asarray(self._placed(raw)[:7], dtype=np.float32)
        # Just the seven coordinates, uploaded: a read of the whole vector and a
        # write of it back, per object per state, is what a reset of forty
        # objects spent most of its time on.
        c = self.coord_start
        for state in (pipeline.state_0, pipeline.state_1):
            state.joint_q[c : c + 7].assign(pose)

    # ---------------------------------------------------------------- tuning
    def tunables(self, pipeline=None):
        """Contact material over this object's shapes, and its mass.

        Over ``self.shapes`` rather than ``range.shape_indices``: a
        scene-declared object does not *build* its shapes -- :meth:`find_in_scene`
        finds ones the arena import already made -- so its build range is empty
        and the indices it actually owns are the ones ``bind`` collected. Most
        objects in an imported arena are that case, which is to say: using the
        range here would have offered a working panel that wrote into no shapes
        at all, on the majority of objects in the benchmark's corpus.
        """
        from robosimstudio.core.tuning import BodyWriter, ShapeWriter, Tunable
        from robosimstudio.specs import BodySpec, ContactMaterialSpec

        if self.model is None:
            return ()
        out = []
        if len(self.shapes):
            shapes = ShapeWriter(self.model, self.shapes, pipeline)
            out.append(
                Tunable(
                    name="contact",
                    spec=ContactMaterialSpec(**shapes.read()),
                    write=shapes,
                    summary=f"What {self.name}'s {len(self.shapes)} shape(s) feel on contact.",
                )
            )
        if self.body >= 0:
            bodies = BodyWriter(self.model, [self.body], pipeline)
            out.append(
                Tunable(
                    name="mass",
                    spec=BodySpec(**bodies.read()),
                    write=bodies,
                    summary="Mass, with inertia scaled to match.",
                )
            )
        return tuple(out)

    # ---------------------------------------------------------------- runtime
    def pose(self, body_q: np.ndarray) -> np.ndarray:
        """This object's world pose from a read-back ``body_q``, as a 7-vector."""
        return np.asarray(body_q[self.body], dtype=np.float64)
