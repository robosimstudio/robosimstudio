# SPDX-License-Identifier: Apache-2.0
"""A jointed object imported from a USD stage: a cabinet, a door, a box with a lid.

The asset says what its parts and joints are; this entity places the whole
thing per episode, starts each joint where the spec or the episode says, and
names both so a goal can ask "is drawer_01 open".

**Two ways to stand.** A fixed base is the importer's own fixed joint from the
world to the root part (``add_usd(floating=False)``). MuJoCo makes the root of a
fixed-base articulation a mocap body, so the per-world placement channel a
welded fixture uses (:class:`~robosimstudio.domains.static.welded.WeldedFixture`)
moves this one too, with the joint the importer made as its weld. A free base
is a FREE joint on the root part, written like any object's: a fleet's wave
writes its seven coordinates from :meth:`episode_start_q`, and :meth:`reset`
writes them in a single-world run.

**Several roots.** An asset in parts no joint holds together (a jar and its lid)
imports one articulation per part (:func:`~robosimstudio.scene.build.imports.split_articulations`).
On a fixed base the heaviest is the base and the rest are free; each free one
is placed with the asset at reset, from where the asset puts it.

**No fixed-joint collapse.** On a fixed base the collapse folds the root part
into the world -- the cabinet body disappears and its drawers hang off body -1 --
and then nothing is left to place per world.
"""

from __future__ import annotations

import re
import warnings

import newton
import numpy as np
import warp as wp

from robosimstudio.core.math import compose_xforms, transform_matrix, xform_from_matrix
from robosimstudio.domains.rigid.entity import episode_pose
from robosimstudio.domains.static.welded import WeldedFixture
from robosimstudio.paths import resolve_asset
from robosimstudio.scene.entity import Entity
from robosimstudio.specs import ArticulatedSpec

__all__ = ["EntityArticulated", "joint_name", "part_name"]

_JOINT_KINDS = {int(newton.JointType.PRISMATIC): "prismatic", int(newton.JointType.REVOLUTE): "revolute"}


def part_name(label: str) -> str:
    """A body's name from its import label: SimReady's ``drawer_01_obj_00`` is ``drawer_01``."""
    leaf = label.rsplit("/", maxsplit=1)[-1]
    return re.sub(r"_(obj|mesh)_\d+$", "", leaf)


def joint_name(label: str) -> str:
    """A joint's name from its import label: SimReady's ``joint_drawer_01_joint_00`` is ``drawer_01``."""
    leaf = label.rsplit("/", maxsplit=1)[-1]
    found = re.fullmatch(r"joint_(.+)_joint_\d+", leaf)
    return found.group(1) if found else leaf


class EntityArticulated(WeldedFixture, Entity):
    """One articulated object from a USD stage.

    Args:
        spec: What object this is.
        world: Which world.
    """

    spec: ArticulatedSpec
    domain = "articulated"
    world_aware = True

    def __init__(self, spec: ArticulatedSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        # The articulated domain's role is the robot's; this is a thing the robot
        # works on, and the interaction graph and the display both read this.
        self.role = "objects"
        self.body: int = -1
        """The root part."""
        self.coord_start: int = -1
        """The root's free-joint coordinates, -1 on a fixed base."""
        self.articulation: int = -1
        self._root_joint: int = -1
        self._root_dof: int = -1
        self._coords: dict[str, int] = {}
        self._dofs: dict[str, int] = {}
        self._travel: dict[str, tuple[float, float]] = {}
        self._loose: list[dict] = []
        self._driven: list[str] = []
        self._world_base: dict[int, np.ndarray] = {}
        self._span: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._joint_handles: tuple[object, dict] | None = None
        self._articulation_of: tuple[object, np.ndarray] | None = None
        self.root_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        """The root part's pose in the frame :attr:`pose` places, read off the import."""
        self.parts: dict[str, int] = {}
        self.joints: dict[str, int] = {}
        """Joint name -> template joint index, the prismatic and revolute ones."""
        self.built_q: dict[str, float] = {}
        """Each joint's start coordinate as built: the spec's, else the asset's."""
        self._declared = np.concatenate(
            [np.asarray(spec.pose, dtype=np.float64), np.asarray(spec.quat, dtype=np.float64)]
        )

    # ------------------------------------------------------------ placement
    @property
    def declared_q(self) -> np.ndarray:
        """The pose this object measures its episode layouts from: the spec's own."""
        return self._declared.copy()

    def _placed(self, pose: np.ndarray) -> np.ndarray:
        return np.asarray(pose, dtype=np.float64) if self.spec.world_xform else self.placed(pose)

    def episode_start_q(self, episode, *, base: np.ndarray | None = None) -> np.ndarray:
        """The root part's world pose for one episode, xyz + xyzw.

        The root's, not the spec's origin's, because this is what a fleet's wave
        writes into the root's free-joint coordinates.
        """
        return compose_xforms(self._asset_pose(episode, base), self.root_offset)

    def _asset_pose(self, episode, base: np.ndarray | None = None) -> np.ndarray:
        """Where the asset's origin stands for one episode, in world coordinates."""
        return self._placed(episode_pose(self.spec.name, self.declared_q if base is None else base, episode))

    def move_refusal(self, episode, *, base: np.ndarray | None = None) -> str:
        """Nothing refuses a layout: either base is placed per world at reset."""
        del episode, base
        return ""

    move_remedy = ""

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        from robosimstudio.scene.build.imports import add_usd_stage

        spec = self.spec
        path = resolve_asset(spec.asset)
        if path.suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz"):
            raise ValueError(f"articulated {spec.name!r}: {path.name} is not a USD stage; its joints come from one")
        if spec.asset.scale != 1.0:
            raise ValueError(
                f"articulated {spec.name!r}: add_usd has no scale argument, so asset.scale="
                f"{spec.asset.scale} would be silently ignored; re-export the stage in metres"
            )
        placed = self._placed(episode_pose(spec.name, self.declared_q, episode))
        xform = wp.transform(tuple(float(v) for v in placed[:3]), wp.quat(*(float(v) for v in placed[3:7])))
        bodies_before, joints_before = builder.body_count, builder.joint_count
        add_usd_stage(
            builder,
            path,
            xform=xform,
            floating=not spec.fixed_base,
            enable_self_collisions=spec.self_collide,
            material_subsets=True,
            skip_mesh_approximation=spec.asset.skip_mesh_approximation,
        )
        if spec.asset.approximate:
            from robosimstudio.scene.build.bodies import approximate_imported_colliders

            approximate_imported_colliders(
                builder,
                range(self.range.shapes[0], builder.shape_count),
                spec.asset.approximate,
                keep_visual=spec.asset.keep_visual,
            )
        joints = range(joints_before, builder.joint_count)
        roots = [j for j in joints if int(builder.joint_parent[j]) == -1]
        if builder.body_count == bodies_before or not roots:
            raise ValueError(f"articulated {spec.name!r}: {path.name} declares no rigid body")
        # In several separate parts (a jar and its lid) on a fixed base, the import
        # welded the heaviest and left the rest free: that weld is the base. Only a
        # weld can be placed per world (MuJoCo's mocap) and only a free joint takes
        # seven coordinates, so a root that is neither -- a door hinged straight to
        # the world -- or a second weld is refused rather than placed wrong.
        free = [j for j in roots if int(builder.joint_type[j]) == int(newton.JointType.FREE)]
        welded = [j for j in roots if int(builder.joint_type[j]) == int(newton.JointType.FIXED)]
        other = [builder.joint_label[j] for j in roots if j not in free and j not in welded]
        wanted = 1 if spec.fixed_base else 0
        if other or len(welded) != wanted:
            raise ValueError(
                f"articulated {spec.name!r}: {path.name} imported {len(welded)} part(s) welded to the world and "
                f"{len(other)} jointed to it otherwise {other}; fixed_base={spec.fixed_base} places "
                f"{'one weld' if spec.fixed_base else 'free parts only'}, with every other part free"
            )
        root = welded[0] if spec.fixed_base else free[0]
        roots = [root, *(j for j in roots if j != root)]
        self.body = int(builder.joint_child[root])
        self.articulation = int(builder.joint_articulation[root])
        if spec.fixed_base:
            # The importer's fixed joint is this object's weld: WeldedFixture.place_welded
            # writes its joint_X_p per world, and MuJoCo reads it as the mocap pose.
            self.weld_joint = int(root)
        else:
            self.coord_start = int(builder.joint_q_start[root])
        self._root_joint = int(root)
        self._root_dof = int(builder.joint_qd_start[root])
        origin = np.linalg.inv(transform_matrix(placed))

        def offset(body: int) -> np.ndarray:
            pose = [float(v) for v in builder.body_q[body]]
            return np.asarray(xform_from_matrix(origin @ transform_matrix(pose)), dtype=np.float64)

        self.root_offset = offset(self.body)
        # Every other root is a part no joint holds to the first -- a jar's lid. Each is free,
        # and placed with the asset from where the asset puts it.
        self._loose = [
            {
                "coord": int(builder.joint_q_start[j]),
                "dof": int(builder.joint_qd_start[j]),
                "articulation": int(builder.joint_articulation[j]),
                "joint": int(j),
                "offset": offset(int(builder.joint_child[j])),
            }
            for j in roots[1:]
        ]

        self.parts = {}
        for body in range(bodies_before, builder.body_count):
            name = part_name(builder.body_label[body])
            self.parts[name if name not in self.parts else builder.body_label[body].rsplit("/", maxsplit=1)[-1]] = body
        self.joints = {}
        for j in joints:
            # A joint of no articulation closes a loop (the pliers' second jaw pivot):
            # its coordinate follows from the tree's, and nothing writes it.
            if int(builder.joint_type[j]) in _JOINT_KINDS and int(builder.joint_articulation[j]) >= 0:
                name = joint_name(builder.joint_label[j])
                self.joints[name if name not in self.joints else builder.joint_label[j].rsplit("/", maxsplit=1)[-1]] = j
        self._span = (
            int(builder.joint_q_start[joints_before]),
            int(builder.joint_coord_count),
            int(builder.joint_qd_start[joints_before]),
            int(builder.joint_dof_count),
        )
        self._apply_joint_specs(builder)
        self._write_joint_starts(builder.joint_q, episode, lambda q: q)
        for name in self._driven:
            builder.joint_target_q[self._coords[name]] = self.built_q[name]

    def _apply_joint_specs(self, builder: newton.ModelBuilder) -> None:
        """Write each :class:`~robosimstudio.specs.JointSpec`'s overrides onto the joint it names."""
        self._travel = {
            name: (float(builder.joint_limit_lower[d]), float(builder.joint_limit_upper[d]))
            for name, d in ((name, int(builder.joint_qd_start[j])) for name, j in self.joints.items())
        }
        for joint in self.spec.joints:
            if joint.name not in self.joints:
                raise ValueError(
                    f"articulated {self.spec.name!r} has no joint {joint.name!r}; it has {sorted(self.joints)}"
                )
            j = self.joints[joint.name]
            dof = int(builder.joint_qd_start[j])
            coord = int(builder.joint_q_start[j])
            if joint.q is not None:
                self._check_travel(joint.name, joint.q, "its JointSpec")
            for field, array in (
                ("damping", builder.joint_damping),
                ("friction", builder.joint_friction),
                ("armature", builder.joint_armature),
                ("target_ke", builder.joint_target_ke),
                ("target_kd", builder.joint_target_kd),
            ):
                value = getattr(joint, field)
                if value is not None:
                    array[dof] = float(value)
            if joint.target_ke is not None:
                builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
                if joint.q is not None:
                    builder.joint_target_q[coord] = float(joint.q)
        coords = {name: int(builder.joint_q_start[j]) for name, j in self.joints.items()}
        stated = {joint.name: joint.q for joint in self.spec.joints if joint.q is not None}
        # Every joint a position drive holds, the asset's own included: its target
        # follows the start, or a drawer started open is pulled shut by a drive
        # still aimed where the asset rested.
        self._driven = [
            name for name, j in self.joints.items() if builder.joint_target_ke[builder.joint_qd_start[j]] > 0.0
        ]
        # An asset can rest a part outside its own joint's travel (a sterilization
        # tray's locks rest at 0 between limits of 0.09 and 2.09 rad); started
        # there, the first step's limit force flings the part. It starts at the
        # nearer limit instead.
        self.built_q, outside = {}, []
        for name, c in coords.items():
            authored = float(stated.get(name, builder.joint_q[c]))
            self.built_q[name] = min(max(authored, self._travel[name][0]), self._travel[name][1])
            if self.built_q[name] != authored:
                outside.append(f"{name} at {authored:g} outside [{self._travel[name][0]:g}, {self._travel[name][1]:g}]")
        if outside:
            warnings.warn(
                f"articulated {self.spec.name!r}: the asset rests {', '.join(outside)}; "
                "started at the nearer limit instead",
                stacklevel=2,
            )
        self._coords = coords
        self._dofs = {name: int(builder.joint_qd_start[j]) for name, j in self.joints.items()}

    def joint_starts(self, episode) -> dict[str, float]:
        """Each joint's start coordinate for one episode: the episode's ``joint_q``, else the build's."""
        starts = dict(self.built_q)
        asked = ((episode or {}).get("joint_q") or {}).get(self.spec.name) or {}
        for name, value in asked.items():
            if name not in starts:
                raise ValueError(
                    f"episode joint_q names {self.spec.name!r}.{name}, which is not one of its joints {sorted(starts)}"
                )
            self._check_travel(name, value, "the episode's joint_q")
            starts[name] = float(value)
        return starts

    def _check_travel(self, name: str, value: float, where: str) -> None:
        low, high = self._travel[name]
        if not low <= float(value) <= high:
            raise ValueError(
                f"articulated {self.spec.name!r}: {where} starts {name} at {value}, outside its travel [{low}, {high}]"
            )

    def _write_joint_starts(self, joint_q, episode, coord) -> None:
        for name, value in self.joint_starts(episode).items():
            joint_q[coord(self._coords[name])] = value

    # -------------------------------------------------------------- phase 2
    def handle(self, model):
        from robosimstudio.scene.handles import ArticulatedHandle, split_shape_flags

        shapes = self.range.shape_indices
        visual, collision = split_shape_flags(model, shapes)
        joints = self._joints_of(model)
        return ArticulatedHandle(
            name=self.name,
            category=self.spec.category,
            body=self.body,
            bodies=tuple(int(b) for b in self.range.body_indices),
            coord_start=self.coord_start,
            shapes=shapes,
            visual_shapes=visual,
            collision_shapes=collision,
            static=self.spec.fixed_base,
            parts=dict(self.parts),
            joints=dict(joints),
            fixed_base=self.spec.fixed_base,
        )

    def published_states(self, snapshot) -> dict[str, float]:
        """Each joint's coordinate, ``<joint>.q``, read off where its two bodies are."""
        if snapshot is None or self.model is None:
            return {}
        body_q = np.asarray(snapshot.body_q)
        return {
            f"{name}.q": joint.coordinate_from_bodies(body_q) for name, joint in self._joints_of(self.model).items()
        }

    def _joints_of(self, model) -> dict:
        """Each joint's :class:`~robosimstudio.scene.handles.JointHandle`, read off ``model`` once."""
        if self._joint_handles is not None and self._joint_handles[0] is model:
            return self._joint_handles[1]
        from robosimstudio.scene.handles import JointHandle

        axis = model.joint_axis.numpy()
        X_p, X_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        kinds = model.joint_type.numpy()
        joints = {}
        for name, j in self.joints.items():
            dof = self._dofs[name]
            joints[name] = JointHandle(
                name=name,
                joint=j,
                kind=_JOINT_KINDS[int(kinds[j])],
                q_start=self._coords[name],
                qd_start=dof,
                parent=int(parent[j]),
                child=int(child[j]),
                axis=tuple(float(v) for v in axis[dof]),
                X_p=tuple(float(v) for v in X_p[j]),
                X_c=tuple(float(v) for v in X_c[j]),
                lower=float(lower[dof]),
                upper=float(upper[dof]),
            )
        self._joint_handles = (model, joints)
        return joints

    # ---------------------------------------------------------------- tuning
    def tunables(self, pipeline=None):
        """One drive group per joint: gains, armature, friction and travel."""
        from robosimstudio.core.tuning import JointWriter, Tunable
        from robosimstudio.specs import JointDriveSpec

        if self.model is None:
            return ()
        out = []
        for name in self.joints:
            writer = JointWriter(self.model, [self._dofs[name]], pipeline)
            out.append(
                Tunable(
                    name=f"joint {name}",
                    spec=JointDriveSpec(**writer.read()),
                    write=writer,
                    summary=f"How {self.name}'s {name} is driven, and how far it travels.",
                )
            )
        return tuple(out)

    # -------------------------------------------------------------- phase 5
    def place_from_document(self, pipeline, spec: ArticulatedSpec, *, world: int, document: str) -> None:
        """Take world ``world``'s base pose from the document it runs, for :meth:`reset`.

        A fleet calls this before the wave's resets, only for a world whose
        document is not the one built; its wave has already written a free root
        from that document's pose, and the reset that follows places the weld,
        the loose parts and the kinematics from it too.
        """
        del pipeline
        if spec.asset != self.spec.asset:
            raise ValueError(
                f"articulated {self.spec.name!r}: document {document!r} makes it from {spec.asset}, "
                f"not the {self.spec.asset} this model was built from"
            )
        self._world_base[int(world)] = np.concatenate(
            [np.asarray(spec.pose, dtype=np.float64), np.asarray(spec.quat, dtype=np.float64)]
        )

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Place this world's copy for ``episode`` and start every joint where it says.

        The root first -- through the weld on a fixed base, the free-joint
        coordinates on a free one -- then each joint's coordinate with its rate
        zeroed, then this articulation's forward kinematics, so ``body_q`` agrees
        with both before anything reads it. A fleet's wave has already run its
        own kinematics by the time it calls this, so the last step is not one a
        caller can be relied on for.

        Only this object's own block of coordinates and rates is read and
        written, once per state, as :meth:`EntityRigid.reset
        <robosimstudio.domains.rigid.entity.EntityRigid.reset>` does: a fleet
        resets every world.
        """
        worlds = self.worlds
        model = self.stepping_model or self.model
        asset = self._asset_pose(episode, self._world_base.pop(int(world), None))
        pose = compose_xforms(asset, self.root_offset)
        if self.spec.fixed_base:
            self.place_welded(pipeline, pose, world=world)
        q0, q1, d0, d1 = self._span
        shift_q = worlds.coord(q0, world) - q0 if worlds is not None else 0
        shift_d = worlds.dof(d0, world) - d0 if worlds is not None else 0
        starts = self.joint_starts(episode)
        for state in (pipeline.state_0, pipeline.state_1):
            if state is None:
                continue
            block = state.joint_q[q0 + shift_q : q1 + shift_q]
            joint_q = block.numpy()
            if not self.spec.fixed_base:
                joint_q[self.coord_start - q0 : self.coord_start - q0 + 7] = pose[:7]
            for part in self._loose:
                c = part["coord"] - q0
                joint_q[c : c + 7] = compose_xforms(asset, part["offset"])[:7]
            for name, value in starts.items():
                joint_q[self._coords[name] - q0] = value
            block.assign(joint_q)
            state.joint_qd[d0 + shift_d : d1 + shift_d].zero_()
        targets = getattr(pipeline.control, "joint_target_q", None) if getattr(pipeline, "control", None) else None
        if targets is not None and self._driven:
            block = targets[q0 + shift_q : q1 + shift_q]
            values = block.numpy()
            for name in self._driven:
                values[self._coords[name] - q0] = starts[name]
            block.assign(values)
        self._forward(model, pipeline, world)

    def _forward(self, model, pipeline, world: int) -> None:
        """This world's articulation's forward kinematics, into both states."""
        state_0, state_1 = pipeline.state_0, pipeline.state_1
        if self._articulation_of is None or self._articulation_of[0] is not model:
            # Read once per model: which articulation each joint is in never changes.
            self._articulation_of = (model, model.joint_articulation.numpy())
        articulation = self._articulation_of[1]
        roots = [self._root_joint] + [part["joint"] for part in self._loose]
        if self.worlds is not None:
            roots = [self.worlds.joint(j, world) for j in roots]
        indices = wp.array([int(articulation[j]) for j in roots], dtype=int, device=model.device)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, indices=indices)
        if state_1 is None:
            return
        b0, b1 = self.range.bodies
        first = self.worlds.body(b0, world) if self.worlds is not None else b0
        for name in ("body_q", "body_qd"):
            wp.copy(getattr(state_1, name), getattr(state_0, name), first, first, b1 - b0)
