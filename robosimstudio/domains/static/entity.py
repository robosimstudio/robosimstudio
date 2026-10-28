# SPDX-License-Identifier: Apache-2.0
"""Scenery: arenas, tables, bins, and everything welded to the world.

Three shapes of static, all through one entity:

* **A procedural box**, from ``half_extents``. Tables and boxes-to-stick-tape-to
  are boxes, and importing an asset for one is a lot of file for a cuboid.
* **An MJCF or USD arena**, imported wholesale. When a benchmark ships a merged
  scene its own compiler produced, importing it is the only way the bins, the
  workbench and the tool meshes stay the benchmark's rather than a re-authoring
  of them. A USD stage exported from Isaac Lab needs two fixups on the way in --
  its geared gripper's couplings, which ``add_usd`` drops, and its doubled
  visual/collision geometry -- and both are ``StaticSpec`` flags over
  :mod:`robosimstudio.scene.build.imports` helpers rather than anything task-specific.
* **The same, with free bodies stripped.** A robot arena that also declares
  loose props needs them gone when the task supplies its own -- and the
  keyframe with them, since a keyframe indexes the qpos of the bodies just
  removed.

A static may carry a JSON sidecar -- ``StaticSpec.meta`` is its path and
``EntityStatic.sidecar`` its contents -- holding what MJCF has no
place for: which joints drive the arm, where a pile settled, where compartment
centres are. The compiler reads it and hands it to whoever asks.

**Scenery is welded, not fixed.** A static has no ``body_q`` row and cannot move
during an episode -- but *which* episode decides where it was welded. See
:func:`episode_xform`: an episode may place any static, which is what turns a
fixture layout from a setting into a variable, and is how a cable-routing task
draws a fresh weave board every episode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.paths import resolve_asset, resolve_path
from robosimstudio.scene.build.imports import (
    add_mjcf_collapsed,
    add_usd_stage,
    strip_free_bodies,
    texture_box_geoms,
)
from robosimstudio.scene.entity import Entity
from robosimstudio.specs import StaticSpec

from .welded import WeldedFixture

__all__ = ["MESH_SUFFIXES", "EntityStatic", "PlacedAtBuild", "episode_xform", "placeable_static"]

MESH_SUFFIXES = (".obj", ".stl", ".ply", ".glb", ".gltf")
"""Asset kinds a static can carry on a body of its own: a bare mesh, and the
procedural shapes (no asset at all). An MJCF or USD arena imports its own
bodies and stays world-hung."""


def placeable_static(spec: StaticSpec) -> bool:
    """Whether a replicated build can give this static a per-world pose.

    True for a procedural box or shape and for a bare mesh; false for an
    imported arena, whose geometry arrives as its own bodies and whose pose is
    baked at conversion. The fleet's wave planner asks this for every static an
    episode moves, so a board of pegs batches and a drawn arena does not.
    """
    asset = getattr(spec, "asset", None)
    if asset is None:
        return True
    return Path(str(asset.path)).suffix.lower() in MESH_SUFFIXES


def episode_xform(spec: StaticSpec, episode) -> tuple[float, ...]:
    """Where this static stands for one episode, **in the spec's own frame**.

    Scenery has been episode-invariant, and for most tasks that is right: a bench
    that moved between episodes would be a different bench. What it refuses is a
    *board* -- a fixture layout that is the task's variable rather than its
    setting, which is what a cable-routing task's weave boards are once the
    stations stop standing on a fixed pitch.

    The override is read the way :class:`~robosimstudio.domains.rigid.entity.EntityRigid`
    reads an object's: ``static_poses[name] = (x, y, z, yaw)`` for the layout
    family anything on a table varies over, and ``static_xforms[name]`` for a
    full 7-vector when a pose needs more than a yaw.

    **In the spec's own frame, not in world**, and that is the whole reason this
    is a function rather than two similar lines in two files. A static's xform is
    stated in the work surface's frame unless ``world_xform`` says otherwise, and
    it is lifted twice on the way to a compiled scene: this function runs once, in
    :meth:`EntityStatic.build`, which lifts its result and builds a shape, and the
    pose it returned is lifted again by :meth:`EntityStatic.handle`, off the cached
    :attr:`EntityStatic.xform_now`, for the handle the region resolver reads, so a
    seat declared ``(:target peg0)`` lands on peg0. Substituting in world coordinates
    would need both lifts skipped, in both places, and a fixture whose *shape* moved
    while its *seat* did not is a scene that scores a weave nobody wove.
    """
    if episode is None:
        return tuple(float(v) for v in spec.xform)
    xforms = episode.get("static_xforms") or {}
    if spec.name in xforms:
        return tuple(float(v) for v in xforms[spec.name])[:7]
    poses = episode.get("static_poses") or {}
    if spec.name in poses:
        x, y, z, yaw = (float(v) for v in poses[spec.name])
        half = 0.5 * float(yaw)
        return (x, y, z, 0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
    return tuple(float(v) for v in spec.xform)


class EntityStatic(WeldedFixture, Entity):
    """One piece of scenery.

    World-hung in a single-world build: shapes on ``body=-1`` at the pose the
    episode welded them, which is what scenery has always been. In a
    **replicated** build (:attr:`replicated`, set by the compiler for every
    static the task declares) a :func:`placeable_static` is built instead as
    one kinematic body bolted to the world, so each world's copy can be moved
    to *its* episode's pose at reset through the mocap channel
    (:mod:`.welded`) -- which is what lets a wave hold twelve boards drawn
    differently. Everything that reads a fixture's pose already prefers a body
    row when there is one (``handle.body >= 0``: goal features, regions framed
    on it, the renderer), so nothing downstream has to know which build it is.

    Args:
        spec: What to build.
        world: Which world.
    """

    domain = "static"
    world_aware = True
    """Every world's copy stands where its own episode welded it.

    Claimed because :meth:`reset` writes one world's weld row rather than the
    whole array. Contributes no substep hooks, so the replication refusal has
    nothing to ask of it; the claim is about reset."""

    spec: StaticSpec
    body: int = -1
    welds_itself = False
    """A subclass that builds its own body and welds it in every build (the
    surface-preparation suite's blocks) says so, and the replicated-build body
    path below stands aside for it."""

    def handle(self, model):
        """Where this fixture sits, for anything measuring against a feature on it.

        A welded static has no ``body_q`` row, so its pose has nowhere else to
        come from -- which is why the handle carries an ``xform`` at all. The
        compiler used to resolve it by searching the *request's* statics by
        name, and returned identity for anything it could not find there: the
        cell's own furniture, which is declared by the cell rather than the task
        and is therefore never in that list. So every bench in the corpus
        reported itself at the world origin.

        It comes off the entity now, which knows: :attr:`xform_now` -- the pose
        this episode welded it at, which is the declared one unless an episode
        placed it (:func:`episode_xform`) -- lifted out of the work-surface frame
        by :meth:`placed` unless the spec says it is already world.
        """
        from robosimstudio.scene.handles import EntityHandle, split_shape_flags

        shapes = self.range.shape_indices
        visual, collision = split_shape_flags(model, shapes)
        xform = self.xform_now if getattr(self.spec, "world_xform", False) else self.placed(self.xform_now)
        bodies = self.range.bodies
        return EntityHandle(
            name=self.name,
            category="static",
            body=bodies[0] if bodies[1] > bodies[0] else -1,
            shapes=shapes,
            visual_shapes=visual,
            collision_shapes=collision,
            static=True,
            xform=tuple(float(v) for v in xform),
        )

    def __init__(self, spec: StaticSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.sidecar: dict[str, Any] = {}
        """The JSON sidecar's *contents*, or ``{}``.

        Named apart from ``StaticSpec.meta``, which is the sidecar's **path**.
        One word for a filename and its payload is survivable while they sit on
        two objects and stops being so the moment a spec's fields and its
        entity's attributes share a namespace -- which is the whole direction of
        this refactor, so the collision is resolved before it can bite rather
        than after."""
        self.xform_now: tuple[float, ...] = tuple(float(v) for v in spec.xform)
        """This static's xform for the episode being built, in the spec's frame.

        The declared one until :meth:`build` runs, so an entity inspected before
        it is built reads as declared rather than as empty."""
        self.replicated = False
        """Set by the compiler before :meth:`build` in a multi-world build.

        What decides whether a placeable static goes on a kinematic body of its
        own. False in every single-world build, where the static stays the
        world-hung scenery it always was and a single-world number is measured
        on exactly the model it always was."""
        self.body = -1

    @property
    def on_body(self) -> bool:
        """Whether this static was built on a body of its own (see the class docstring)."""
        return self.body >= 0

    def _shape_frame(self) -> tuple[int, Any]:
        """Which body a shape goes on, and at what xform: world-hung or on this static's body."""
        if self.on_body:
            return self.body, wp.transform_identity()
        placed = self._placed(self.xform_now)
        return -1, wp.transform(
            tuple(float(v) for v in placed[:3]),
            wp.quat(*[float(v) for v in placed[3:7]]),
        )

    claims_body = False
    """Set True by a subclass that overrides :meth:`build` AND calls :meth:`_claim_body` there.

    The base build claims its body itself, so a subclass that keeps it needs
    nothing. One that draws its own shapes must say so, or the claim below
    answers no for it: six cable-routing fixtures and the USB-C port all override
    ``build`` and drew world-hung, and a claim read off the spec alone said
    every one of them could be placed per world."""

    @classmethod
    def places_per_world(cls, spec: StaticSpec) -> bool:
        """Whether a replicated build of ``spec`` by THIS class stands on a per-world body.

        The claim the wave planner keys on and the compiler orders by, asked of
        the class that will build the static rather than of the spec alone. No
        for a class that welds itself (:attr:`welds_itself`), for a spec no
        replicated build can place (:func:`placeable_static`), and for a
        subclass that overrides :meth:`build` without declaring
        :attr:`claims_body`. A class whose build did not honour its own answer
        here would let a wave share a model between episodes whose statics it
        cannot place, and every world would silently keep the build episode's
        pose -- the usbc port did exactly that before it claimed its body.
        """
        if cls.welds_itself or not placeable_static(spec):
            return False
        return cls.build is EntityStatic.build or bool(cls.claims_body)

    def _claim_body(self, builder: newton.ModelBuilder, episode) -> None:
        """Take this episode's pose, and in a replicated build a body to carry it per world.

        Every :meth:`build`, the base one and a subclass's own, calls this first
        and then puts its shapes at :meth:`_shape_frame`: one body bolted to the
        world where this episode put the static, the shapes on it at identity,
        so the world's copy can be re-placed at reset.
        """
        # Scenery an episode may move; see :func:`episode_xform` for why the
        # substitution happens in the spec's frame and not in world.
        self.xform_now = episode_xform(self.spec, episode)
        spec = self.spec
        if self.replicated and type(self).places_per_world(spec):
            placed = self._placed(self.xform_now)
            xform = wp.transform(tuple(float(v) for v in placed[:3]), wp.quat(*[float(v) for v in placed[3:7]]))
            self.body = self.weld_in_place(builder, xform, label=spec.name)

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        self._claim_body(builder, episode)
        spec = self.spec
        if spec.asset is None:
            self._add_box(builder)
        else:
            self._add_asset(builder)
        if spec.friction is not None:
            # After the import rather than through a config, so it reaches an
            # asset's own shapes too -- an arena states friction per geom and a
            # scene overriding it has to mean all of them.
            for s in range(self.range.shapes[0], builder.shape_count):
                builder.shape_material_mu[s] = float(spec.friction)
        # Display is applied by Entity.build_into, once the range has closed.

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Bolt this world's copy where its episode welded it.

        Only on a body of its own (a replicated build): world-hung scenery has
        no per-world pose and a single-world scene rebuilds for a moved static
        exactly as before (``model_key``). The transform written is the one
        :meth:`build` places the shapes with, asked again for this episode --
        one arithmetic, two moments.
        """
        if not self.on_body or self.weld_joint < 0:
            return
        self.place_welded(pipeline, self._placed(episode_xform(self.spec, episode)), world=world)

    def tunables(self, pipeline=None):
        """Contact material over the shapes this fixture built.

        No mass: a static is welded to the world and its inverse mass is what
        marks it fixed. What a bench, a bin or a peel box *does* have is a
        surface, and its friction is half of whether anything set down on it
        stays put -- which is the tunable somebody actually reaches for.
        """
        from robosimstudio.core.tuning import ShapeWriter, Tunable
        from robosimstudio.specs import ContactMaterialSpec

        shapes = self.range.shape_indices
        if self.model is None or not len(shapes):
            return ()
        writer = ShapeWriter(self.model, shapes, pipeline)
        return (
            Tunable(
                name="contact",
                spec=ContactMaterialSpec(**writer.read()),
                write=writer,
                summary=f"What {self.name}'s {len(shapes)} shape(s) feel on contact.",
            ),
        )

    def _placed(self, xform):
        """This static's pose in world, unless its asset is the room itself."""
        return np.asarray(xform, dtype=np.float64) if self.spec.world_xform else self.placed(xform)

    def _add_box(self, builder: newton.ModelBuilder) -> None:
        body, xform = self._shape_frame()
        if self.spec.shape is not None:
            self.spec.shape.add_to(builder, body=body, xform=xform, label=self.spec.name)
            return
        if self.spec.half_extents is None:
            raise ValueError(f"static {self.spec.name!r} names no asset, no shape and no half_extents")
        hx, hy, hz = (float(v) for v in self.spec.half_extents)
        builder.add_shape_box(body=body, xform=xform, hx=hx, hy=hy, hz=hz, label=self.spec.name)

    def _add_asset(self, builder: newton.ModelBuilder) -> None:
        path = resolve_asset(self.spec.asset)
        if self.spec.meta:
            meta_path = path.parent / self.spec.meta if not os.path.isabs(self.spec.meta) else Path(self.spec.meta)
            with open(resolve_path(meta_path)) as f:
                self.sidecar = json.load(f)

        source = str(path)
        stripped = strip_free_bodies(source) if self.spec.strip_free_bodies else None
        try:
            # World-hung, or on this static's own body: an arena import is
            # never on a body (``placeable_static``), so the two import
            # branches below always see ``body == -1`` and the world xform.
            body, xform = self._shape_frame()
            if path.suffix.lower() in (".xml", ".mjcf"):
                # Collapsed through the helper, which folds only this import's
                # fixed joints; the importer's own flag folds the whole builder.
                if self.spec.collapse_fixed_joints:
                    add_mjcf_collapsed(builder, stripped or source, xform=xform, scale=self.spec.asset.scale)
                else:
                    builder.add_mjcf(stripped or source, xform=xform, scale=self.spec.asset.scale, verbose=False)
                # The MJCF sibling of the USD fixups below: the importer binds a
                # material's texture only to mesh geoms, so an arena that labels
                # anything with a textured primitive arrives unable to state it.
                # Read against the original, not the stripped copy -- the strip
                # removes free bodies, never assets, and the original is the file
                # whose relative texture paths still resolve.
                texture_box_geoms(builder, source)
            elif path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
                self._add_usd(builder, source, xform)
            elif path.suffix.lower() in (".obj", ".stl", ".ply", ".glb", ".gltf"):
                # A bare mesh as scenery -- a fixture, a jig, a receptacle. The
                # authored triangles are kept: a fixture whose geometry matters
                # (a socket with a bore) is precisely the case where an
                # approximation would erase the task.
                import trimesh

                mesh = trimesh.load(str(path), force="mesh")
                visual_shape = builder.add_shape_mesh(
                    body=body,
                    xform=xform,
                    mesh=newton.Mesh(
                        np.asarray(mesh.vertices, dtype=np.float32),
                        np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
                    ),
                    scale=(self.spec.asset.scale,) * 3,
                    label=self.spec.name,
                )
                proxy_paths = ((self.spec.asset.collision_path,) if self.spec.asset.collision_path else ()) + tuple(
                    self.spec.asset.collision_paths
                )
                if proxy_paths:
                    if self.spec.asset.approximate:
                        raise ValueError(
                            f"static {self.spec.name!r}: collision proxies cannot be combined with "
                            f"approximate={self.spec.asset.approximate!r}"
                        )
                    builder.shape_flags[visual_shape] = int(builder.shape_flags[visual_shape]) & ~int(
                        newton.ShapeFlags.COLLIDE_SHAPES
                    )
                    for index, proxy_asset_path in enumerate(proxy_paths):
                        proxy_mesh = trimesh.load(str(resolve_path(proxy_asset_path)), force="mesh", process=False)
                        proxy_shape = builder.add_shape_mesh(
                            body=body,
                            xform=xform,
                            mesh=newton.Mesh(
                                np.asarray(proxy_mesh.vertices, dtype=np.float32),
                                np.asarray(proxy_mesh.faces, dtype=np.int32).reshape(-1),
                            ),
                            scale=(self.spec.asset.scale,) * 3,
                            label=f"{self.spec.name}_collision_{index:02d}",
                        )
                        builder.shape_flags[proxy_shape] = int(builder.shape_flags[proxy_shape]) & ~int(
                            newton.ShapeFlags.VISIBLE
                        )
            else:
                raise ValueError(f"static {self.spec.name!r}: unsupported asset {path.suffix!r}")
        finally:
            if stripped is not None:
                os.unlink(stripped)

        if self.spec.asset.approximate:
            self._approximate(builder)

    def _add_usd(self, builder: newton.ModelBuilder, source: str, xform) -> None:
        """Import a USD stage, then apply the two fixups an Isaac export needs.

        ``scale`` is not forwarded: ``add_usd`` takes the stage's own
        ``metersPerUnit`` and has no uniform-scale argument, so a spec that asked
        for one would be silently ignored. An export in the wrong units is fixed
        by re-exporting, not by a multiplier here.
        """
        if self.spec.asset is not None and self.spec.asset.scale != 1.0:
            raise ValueError(
                f"static {self.spec.name!r}: add_usd has no scale argument, so asset.scale="
                f"{self.spec.asset.scale} would be silently ignored; re-export the stage in metres"
            )
        add_usd_stage(
            builder,
            source,
            xform=xform,
            collapse_fixed_joints=self.spec.collapse_fixed_joints,
            expose_declared_colliders=self.spec.expose_declared_colliders,
            enable_self_collisions=False,
            skip_mesh_approximation=bool(self.spec.asset and self.spec.asset.skip_mesh_approximation),
            verbose=False,
        )

    def _approximate(self, builder: newton.ModelBuilder) -> None:
        # The entity's own range is still open (close() runs after build), so
        # its shapes are everything from where it started to the current count.
        targets = [
            s
            for s in range(self.range.shapes[0], builder.shape_count)
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES))
        ]
        if targets:
            builder.approximate_meshes(
                method=self.spec.asset.approximate,
                shape_indices=targets,
                keep_visual_shapes=self.spec.asset.keep_visual,
            )


class PlacedAtBuild:
    """Mixed into a static whose per-world pose must be in the model *before* ``_assemble``.

    :meth:`EntityStatic.reset` already places each world's copy through the mocap
    channel, and for scenery that a foreign engine converts once and thereafter
    reads live, that is enough -- MuJoCo re-reads ``joint_X_p`` into this world's
    ``mocap_pos``, measured at 0.000 mm on a tool-hanging scene. It is not enough for
    a fixture something *deformable* bears against: the coupled solver mirrors a
    static's shapes into the deformable island as proxies when the island is
    built, which is inside ``_assemble``, so a pose written afterwards is a proxy
    made at world 0's layout and moved. Offering ``place`` puts the pose in front
    of that -- see
    :func:`~robosimstudio.scene.compiler._place_worlds` and
    :meth:`~.welded.WeldedFixture.place_in_model`.

    **Opt-in per class rather than on** :class:`EntityStatic`, because a scene
    holding any entity that offers ``place`` is built for ONE wave's benches and
    the fleet runner rebuilds it per wave. Every
    scene has a bench, so putting this on the base class would make every suite
    pay a build per wave for scenery that a reset already places correctly.

    Mixed in *before* the entity base, like :class:`~.welded.WeldedFixture`, so
    ``self.body`` and ``self.worlds`` are the entity's own.
    """

    def place(self, model: Any, episode, *, world: int = 0) -> None:
        """Bolt this world's copy where its episode welded it, in the model.

        The build-time half of :meth:`EntityStatic.reset`, over the same
        arithmetic: :func:`episode_xform` in the spec's frame, lifted once. A
        world-hung copy (a single-world build, or a static no replicated build
        can place) has no per-world pose and is skipped.
        """
        if not self.on_body or self.weld_joint < 0:
            return
        self.place_in_model(model, self._placed(episode_xform(self.spec, episode)), world=world)


class EntityArena(EntityStatic):
    """A robot arena imported with its fixed joints collapsed.

    Split from :class:`EntityStatic` because collapsing is what makes the
    custom-attribute re-key necessary, and that fixup should be attached to the
    thing that triggers it rather than switched on by a flag on everything.
    """

    def _add_asset(self, builder: newton.ModelBuilder) -> None:
        path = resolve_asset(self.spec.asset)
        source = str(path)
        stripped = strip_free_bodies(source) if self.spec.strip_free_bodies else None
        try:
            add_mjcf_collapsed(
                builder,
                stripped or source,
                scale=self.spec.asset.scale,
                floating=False,
                enable_self_collisions=False,
            )
            # The same gap EntityStatic._add_asset closes for the plain import:
            # the importer binds a material's texture only to mesh geoms, so a
            # textured *primitive* -- the tool-sort workbench top is a box
            # wearing steel plate -- arrives as a flat colour with no warning.
            # Collapsing changes nothing about that, so the collapsed arena was
            # the one import path still dropping them. Against the original
            # file, whose relative texture paths still resolve.
            texture_box_geoms(builder, source)
        finally:
            if stripped is not None:
                os.unlink(stripped)
