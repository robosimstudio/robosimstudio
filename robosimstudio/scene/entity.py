# SPDX-License-Identifier: Apache-2.0
"""The entity contract, and nothing that implements it.

Every concrete entity lives in the domain it builds -- ``domains/rigid``,
``domains/static``, ``domains/cloth``, ``domains/soft`` -- and the robot's is in
``robots``. What is here is the part the compiler actually depends on: the phase
order an entity is built through, and the index bookkeeping around it.

That is the whole point of the split. The compiler knows `build -> bind ->
materials -> kernels -> ready -> reset`; it does not know that a rigid body has a
grasp box or that a sheet is a grid of particles, and it cannot, because it no
longer imports anything that does.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import newton
import numpy as np

from robosimstudio.core.pipeline import ScenePipeline, SubstepHooks
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.tuning import Tunable
from robosimstudio.specs.physics import HostActor, SimRequirements

from .ref import Ref

__all__ = ["Entity", "EntityRange", "Tunable", "entity_layouts", "label_index", "label_indices"]


class EntityRange:
    """Where one entity landed in a model, as half-open index ranges.

    Recorded during :meth:`Entity.build`, when the builder's counts before and
    after the entity's calls bracket exactly what it added.
    """

    __slots__ = ("bodies", "joints", "particles", "shapes", "tris")

    def __init__(self) -> None:
        self.bodies: tuple[int, int] = (0, 0)
        self.shapes: tuple[int, int] = (0, 0)
        self.joints: tuple[int, int] = (0, 0)
        self.particles: tuple[int, int] = (0, 0)
        self.tris: tuple[int, int] = (0, 0)

    def open(self, builder: newton.ModelBuilder) -> None:
        """Record the builder's counts before the entity adds anything."""
        self.bodies = (builder.body_count, builder.body_count)
        self.shapes = (builder.shape_count, builder.shape_count)
        self.joints = (builder.joint_count, builder.joint_count)
        self.particles = (builder.particle_count, builder.particle_count)
        self.tris = (builder.tri_count, builder.tri_count)

    def close(self, builder: newton.ModelBuilder) -> None:
        """Close every range at the builder's counts afterwards."""
        self.bodies = (self.bodies[0], builder.body_count)
        self.shapes = (self.shapes[0], builder.shape_count)
        self.joints = (self.joints[0], builder.joint_count)
        self.particles = (self.particles[0], builder.particle_count)
        self.tris = (self.tris[0], builder.tri_count)

    @property
    def body_indices(self) -> np.ndarray:
        return np.arange(*self.bodies, dtype=np.int32)

    @property
    def shape_indices(self) -> np.ndarray:
        return np.arange(*self.shapes, dtype=np.int32)

    @property
    def particle_indices(self) -> np.ndarray:
        return np.arange(*self.particles, dtype=np.int32)

    def __repr__(self) -> str:
        return (
            f"EntityRange(bodies={self.bodies}, shapes={self.shapes}, joints={self.joints}, particles={self.particles})"
        )


class Entity:
    """Base class. Subclasses implement whichever phases they need.

    **There is deliberately no ``observe`` here.** One used to be declared, with
    the signature ``observe(state)`` and a ``None`` default, and it was a
    contract nothing honoured: the two callers that exist ask two different
    domains two different questions in two different shapes -- the robot's
    proprioception (``embodiment.observe(state)``, one argument) and a sheet's
    measurements (``cloth.observe(state, step, readback=...)``, three). A base
    method that no call site matches is worse than no base method: it invites an
    override that will silently never run.

    Args:
        spec: The frozen spec this entity realises.
        world: Which world it is being built into. Single-world today; the
            argument exists so handles and probes carry it from the start
            rather than growing it later.
    """

    role: str = ""
    """Override for :attr:`display_role`, or empty for the domain's own.

    An instance attribute rather than a ``ClassVar``: two entities of the same
    class play different parts depending on who declared them, which is exactly
    the table-versus-peg case."""

    domain: ClassVar[str] = "rigid"
    """The physics domain this kind of entity belongs to, unless its spec's
    ``sim`` declaration says otherwise. What the composer partitions on."""

    world_aware: ClassVar[bool] = False
    """Whether every phase of this entity carries a world axis.

    One flag rather than one per phase, because the claim is indivisible. A
    substep kernel with a template body index baked into it integrates world
    0's bodies whatever world it was launched for; a ``reset`` that writes a
    template coordinate range restores world 0 and leaves the rest of the wave
    where the last episode left it; a published state read off row 0 scores
    every world on one world's answer. An entity that has fixed one of those and
    not the others is not world aware, and a flag it could set to say it was
    would be worse than no flag.

    Read in two places, and both fail *safe* when it is unset. The compiler
    refuses to replicate a scene holding an entity that contributes substep
    hooks and has not claimed this -- silently integrating world 0 for every
    world is plausible motion in the wrong place, which is the failure worth
    refusing where it can still be named. And a wave's reset runs only the
    entities that claim it, because an unclaimed ``reset`` run once per world
    writes world 0's state ``W`` times, which is worse than not running it.

    :attr:`worlds` is what makes claiming it possible,
    :meth:`~robosimstudio.core.worlds.WorldMap.spread` turns the template
    indices resolved in :meth:`bind` into the per-world arrays a fleet-wide
    launch takes, and :attr:`stepping_model` is the model those launches run
    against.
    """

    ref: ClassVar[type] = Ref
    """What ``Scene.add`` hands back for this kind of thing.

    Declared on the entity because **the accessors a thing offers are a fact
    about the thing**. A rod can answer "where is arc length 0.6"; a cloth can
    answer "where is the corner I named"; a bracket can answer neither and needs
    no class of its own. The base :class:`~robosimstudio.scene.ref.Ref` is what
    the compiler knows about everything -- name, spec, the indices it owns, and
    where they are now -- so an entity that declares nothing here still gets a
    usable handle rather than a bare string.
    """

    def __init__(self, spec: Any, *, world: int = 0) -> None:
        self.spec = spec
        self.world = int(world)
        self.stepping_model: Any = None
        """The model that actually steps, when it is not the one :meth:`bind` got.

        ``bind`` is handed the **template** in a replicated build, deliberately:
        it is where names resolve, and a name resolved against a model holding
        sixteen copies of everything finds sixteen of each. Kernels are the other
        case -- a substep hook runs against the state of the model being stepped,
        so a hook holding the template integrates the wrong array, or the wrong
        length of one.

        So an entity that contributes hooks reads this rather than ``self.model``
        when what it needs is the thing under the solver. ``None`` at one world,
        where the two are the same model and there is nothing to tell apart.
        """

        self.worlds: Any = None
        """The build's :class:`~robosimstudio.core.worlds.WorldMap`, or ``None``.

        Set by the compiler between finalizing the models and :meth:`bind`, so an
        entity that has to write per-world state knows how many worlds there are
        and where each one's copy of a template index lives. ``None`` at one
        world, which is the case every entity already handles -- so reading it is
        an ``if``, not a rewrite.
        """
        self.range = EntityRange()
        self.finishes: dict[int, Finish] = {}
        """``shape index -> Finish`` for every shape :meth:`apply_display` painted.

        Filled beside the colour, from the spec's own ``finish`` or the one keyed
        to the colour the shape ended up with, and collected onto the scene
        handles by the compiler. Read by the path-traced backend; nothing else.
        """
        self.model: newton.Model | None = None
        self.embodiment: Any = None
        """The scene's robot, for entities whose physics couples to it. Set by
        the compiler on task-owned entities; ``None`` in a scene with no robot."""
        self.frame: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        """The scene's work-surface frame, set by the compiler before :meth:`build`.

        Task-owned entities state their poses *in this frame* -- rigid bodies,
        statics, rods and deformables alike, every one through :meth:`placed`. The
        scene's own furniture states them in world coordinates, because a place
        is not positioned relative to itself. Identity for a task compiled
        without a scene, so a coordinate "in the surface frame of no scene" is
        simply a world coordinate.

        The deformables were the last to move onto it: a rod's ``start`` and a
        cloth's ``pose`` went to the builder raw, and both shipped rod tasks
        wrote ``CELL.surface_z + ...`` by hand to compensate.
        """

    def placed(self, xform: Sequence[float]) -> np.ndarray:
        """One spec-declared pose, as world coordinates.

        Every task entity's pose goes through here rather than being used raw,
        so "which frame is this in?" has one answer per entity instead of one
        answer per call site.
        """
        from robosimstudio.core.math import transform_matrix, xform_from_matrix

        x = np.asarray(xform, dtype=np.float64).reshape(-1)[:7]
        if np.allclose(self.frame, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)):
            return x
        return np.asarray(
            xform_from_matrix(transform_matrix(np.asarray(self.frame)) @ transform_matrix(x)), dtype=np.float64
        )

    @property
    def name(self) -> str:
        return getattr(self.spec, "name", type(self).__name__)

    # ------------------------------------------------------------ declaration
    layouts: ClassVar[tuple[str, ...]] = ()
    """State layouts this entity may **additionally** be read as, beyond its domain's.

    A domain's ``reads`` chain says what every thing of that kind can be
    measured as. A capability says what *this class* can be measured as
    regardless of its domain -- a tetrahedral body that is also a curve is
    ``domain="soft"`` and reads as ``("volume", "cloth")`` by that chain, and
    it is still a curve to a measure over curves because its class declares
    the capability that says so
    (:mod:`robosimstudio.domains.capabilities`). Unioned across the MRO by
    :func:`entity_layouts`, so mixins compose.
    """

    def sim_requirements(self) -> SimRequirements:
        """What this entity needs of physics: the spec's ``sim`` when it carries
        one, else the domain default.

        The asset's declaration, not a solver choice: which backend serves the
        domain is the arrangement's, and which island this entity lands in is
        the composer's. An entity that needs more than its domain's defaults
        -- a cloth that must not touch the fixtures, a rod that hands may only
        push one way -- says so through its spec.
        """
        declared = getattr(self.spec, "sim", None)
        if isinstance(declared, SimRequirements):
            return declared
        return SimRequirements(domain=self.domain)

    def host_actor(self) -> HostActor | None:
        """What this entity is to a foreign engine, or ``None`` for "infer it".

        Read only when an arrangement routes this entity's domain to a host
        engine; an entity on a Newton solver never comes here. It says the two
        things the model cannot: **which kind of actor this is**, and the
        scalars the engine wants that no Newton array carries.

        **Declared, where it used to be duck-typed.** The composer reached this
        by ``getattr(entity, "host_actor_params", None)`` and, finding nothing,
        guessed the kind by looking for tetrahedra in the entity's particle
        range -- which can only ever answer "tet or shell", and is exactly why a
        rod could not be a host actor: a rod's centerline owns neither. An
        entity that returns ``None`` still gets that inference, so nothing that
        declared nothing has changed; what is new is that an entity *may* say.

        Returning ``None`` is therefore not "this is not a host actor" -- an
        entity that owns particles is one wherever the arrangement puts it. It
        is "the model already says what I am".
        """
        return None

    def owned_bodies(self) -> np.ndarray:
        """Global body indices this entity owns. The build range by default."""
        return self.range.body_indices

    @property
    def carries_particles(self) -> bool:
        """Whether this entity owns particles.

        The question twenty-odd call sites used to ask by looking in a
        ``deformables`` bin. Being made of particles is a fact about the entity,
        so it is answered here -- and a bin could not answer it anyway: a rope
        is particle-free as a capsule chain and particle-only as a centerline,
        which is why the sites that cared unioned two bins to find one thing.
        """
        return bool(len(self.owned_particles()))

    def owned_particles(self) -> np.ndarray:
        """Global particle indices this entity owns. The build range by default."""
        return self.range.particle_indices

    def owned_shapes(self) -> np.ndarray:
        """Shapes this entity owns that sit on no body it owns -- world-fixed
        scenery. The build range by default; the composer adds every shape on
        an owned body itself."""
        return self.range.shape_indices

    # -------------------------------------------------------------- phase 1
    def published_states(self, snapshot) -> dict[str, float]:
        """What this entity publishes about itself, by name.

        The reading half of a declared ``PublishedState``. An asset's record says
        *which* states exist and what each means; this is where the numbers come
        from, and until it existed every consumer reached into the entity with
        ``getattr`` and its own idea of the attribute's name.

        That mattered once there was a goal language. A document saying
        ``(<= (bond-fraction tape) 0.05)`` has to resolve ``bond-fraction``
        against something, and "whatever attribute the entity happens to expose"
        is not a contract -- it is a convention that holds until two suites spell
        one thing differently.

        Empty by default: most entities publish nothing, and an entity that
        computes a quantity nobody declared is not publishing it.

        Args:
            snapshot: The instant, for a state that has to be measured rather
                than read off a field. A composite assembly's mating gap is a
                function of where its bodies are right now, so it cannot be a
                cached attribute without being stale.
        """
        del snapshot
        return {}

    def handle(self, model: newton.Model) -> Any | None:
        """Where this entity landed, as the record everything downstream reads.

        ``SceneHandles`` is the one place anything asks "where is the hammer",
        "which shapes are the left gripper's", "which particles are the tab" --
        evaluators, renderers, recorders and teleop panels all read it and none
        of them walks the model. Building it was the compiler's job, in three
        literal loops that reached into whatever landed in each bin: ``.body``,
        ``.coord_start``, ``.obb`` and ``.bbox`` off an object, ``.particles``
        and ``.sets`` off a deformable. **None of those is on this class**, so
        the contract was real, unwritten, and enforced by an ``AttributeError``
        three hundred lines from the spec that got it wrong.

        The cost was not the coupling. It was that an ``extras`` entity -- the
        bin a kind the library has never heard of lands in -- got **no handle at
        all**, because the compiler had no loop for it. A novel kind was
        therefore second-class by construction: composable, steppable, and
        invisible to every consumer that reads handles.

        ``None`` means "I offer none", which is what most entities want and what
        every one of them did before this existed.

        Args:
            model: The finalized model. The *template* model when replicated --
                handles are label lookups and every world carries the same
                labels.
        """
        del model
        return None

    forms: ClassVar[tuple[str, ...]] = ()
    """Which :data:`~robosimstudio.assetlib.semantics.atoms.OPERAND_FORMS` this
    implementation answers for itself.

    Declared rather than discovered, so a substrate's claim is checkable without
    building a scene: a kind whose layout has handlers asking for ``points`` and
    that answers no form is a kind that will build, step, and bind to nothing.
    Empty means every form is inferred, which is what every entity got before
    any of them could say."""

    def read(self, form: str, snapshot) -> Any | None:
        """This entity's own state in one operand form, or ``None`` to be inferred.

        **The general case of** :meth:`substrate_points`, whose docstring makes
        the whole argument: a layout names a *state layout*, not a place to find
        one, and a reader that infers where the state lives is a reader that is
        right only while a kind has one implementation. There are eight operand
        forms and the points one was done first because a bodiless rod made it
        urgent; the rest were left inferred, which is the same guess waiting for
        the same second implementation.

        ``None`` keeps the reader's own inference, so an entity that says nothing
        reads exactly as it did.

        Args:
            form: One of ``OPERAND_FORMS``.
            snapshot: The instant. Read from it rather than from the model --
                the model's arrays are the *build-time* layout, and a thing that
                has moved reports itself where it was declared.
        """
        return self.substrate_points(snapshot) if form == "points" else None

    def substrate_points(self, snapshot) -> np.ndarray | None:
        """The ordered points this entity reads as, or ``None`` to be inferred.

        A :data:`~robosimstudio.specs.physics.SUBSTRATES` entry names a **state
        layout**, not a place to find one. ``"rod"`` means "an ordered sequence
        of positions along a length" -- and its reader assumed those positions
        were *body poses*, because for as long as a kind had one implementation
        that was the only way a rod could be built. A rope simulated as a
        centerline has no bodies at all, so it built, stepped, came to rest
        correctly, and could not be scored by a single rod goal:
        ``winding_turns``, ``straightness`` and ``cable.point(0.6)`` all failed
        to bind against it.

        So this is the (kind x engine) axis reaching **measurement**, which it
        has to: two implementations of one kind that agree about the physics and
        disagree about what may be asked of them are not two implementations of
        one kind. An entity says where its points are; the reader stops
        guessing.

        ``None`` keeps the reader's own inference, which is what every entity
        that never needed to say gets -- and is why nothing that worked before
        changed.

        Args:
            snapshot: The instant. Read from it rather than from the model: the
                model's arrays are the *build-time* layout, and a thing that has
                moved reports itself where it was declared, which is a plausible
                number and completely wrong.
        """
        del snapshot
        return None

    def tunables(self, pipeline: Any = None) -> tuple[Tunable, ...]:
        """What may be changed on this entity while it runs, and how.

        The writing half of what :meth:`published_states` reads, and it exists for
        the same reason. An asset's parameters are the asset's; a consumer that
        carries its own table of what may be tuned on what is carrying a copy
        that can disagree, and the material studio *was* that table -- it could
        tune a cloth and nothing else, because that is where somebody happened to
        write the mechanism.

        Each :class:`~robosimstudio.core.tuning.Tunable` pairs a described
        frozen dataclass with a writer that puts a field into the built model.
        The spec is **read back from the model**, never remembered from the
        build: after an import, a collider swap or a preset load, the model is
        the only thing that knows what it holds.

        Empty by default. An entity with nothing to tune returns nothing, which
        is different from an entity nobody has got to yet only in that the second
        one is a missing implementation and this is a statement.

        Args:
            pipeline: The scene's :class:`~robosimstudio.core.pipeline.ScenePipeline`,
                so a writer can tell the solvers which category of model data
                changed. ``None`` writes the arrays and notifies nobody, which is
                right for a test over a bare model and wrong for a running scene.
        """
        del pipeline
        return ()

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        """Add this entity to the builder. No device memory exists yet."""

    def build_into(self, builder: newton.ModelBuilder, episode) -> None:
        """Run :meth:`build` bracketed by the range bookkeeping."""
        self.range.open(builder)
        self.build(builder, episode)
        self.range.close(builder)
        added = self.range.shapes
        self.after_build(builder, episode)
        if added[1] > added[0] and self.range.shapes == added and builder.shape_count > added[1]:
            # ``after_build`` appended shapes, and nothing else was building: an
            # approximated collider keeps the mesh it replaced as a *new* visual
            # shape at the end of the builder (``approximate_meshes(...,
            # keep_visual_shapes=True)``). Those are this entity's shapes, and
            # the range has to say so -- or ``apply_display`` below paints the
            # invisible collider with the stated colour and leaves the one shape
            # a camera sees in the build-order palette. Measured on
            # a sharps-disposal scene's spare syringe, stated
            # (0.90, 0.91, 0.93) and rendered cyan. A range ``after_build`` reset
            # itself (a scene-resident object, found rather than added) is left
            # alone: it added nothing, so nothing appended is its.
            self.range.shapes = (added[0], builder.shape_count)
        self.apply_display(builder)

    def after_build(self, builder: newton.ModelBuilder, episode) -> None:
        """Work that needs this entity's own index ranges already closed.

        Where a rigid body collects its shapes, approximates its collider and
        computes its grasp box. It belongs to whoever owns the geometry.

        Default: nothing.
        """

    @property
    def display_role(self) -> str:
        """Which of :data:`~robosimstudio.specs.physics.ROLES` this entity looks like.

        The domain's own role, unless something set :attr:`role` -- which the
        compiler does for the cell's own furniture, because a table is
        ``static`` exactly like a peg is and is not a fixture: it is the surface
        both of them stand on. That distinction is one ``ROLES`` already draws
        and the build loop already knows, so it costs one assignment rather than
        a new spec field.
        """
        if self.role:
            return self.role
        from robosimstudio.specs.physics import DOMAINS, get_domain

        domain = self.sim_requirements().domain
        return get_domain(domain).role if domain in DOMAINS else "objects"

    @property
    def recolors_live(self) -> bool:
        """Whether this entity writes ``model.shape_color`` while the scene runs.

        False for everything that paints in the builder, which is almost
        everything: the palette, the static overrides, :meth:`apply_display`.
        True only for a field that fades its own shapes from a kernel -- a mark
        being polished off, a tape seam parting -- and it is the *live* viewers
        that need to be told. The tiled ray tracer and the GL viewer re-read
        ``shape_color`` every frame and so never ask; the studio's viser page
        repacks colours only when something says they moved, because the repack
        costs a device readback per shape batch per frame and in every other
        scene the answer is that they did not. So this is the declaration that
        buys that economy back for the one entity that needs it, rather than
        paying for it in all of them.
        """
        return False

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """Give this entity's own shapes the appearance its spec asked for.

        Here rather than per entity type because the alternative is a colour
        path on some specs and not others, and a shape nobody coloured does not
        come out untouched -- ``ModelBuilder`` falls back to a palette indexed by
        the shape's position in the build order (``_shape_palette_color``). So
        the colour of anything unstated is a function of how many shapes
        happened to be built before it, and inserting one object upstream
        repaints everything downstream.

        **So this no longer gives up when the spec says nothing.** A shape that
        still carries the colour the builder guessed for it -- exactly the one
        :func:`~robosimstudio.core.render.palette.is_build_order_color` tests for
        -- gets its role's default instead. Measured on the benchmark's corpus that
        is 30 of the 93 visible shapes in a tool-sorting scene, where a single
        hammer renders as five unrelated hues across its claw, face, handle, head
        and neck, and where those hues move whenever anything upstream changes.

        Only a *guessed* colour is replaced. An importer that stated one -- an
        MJCF ``rgba``, an MTL diffuse, a USD material -- is left alone, which is
        what makes this safe to switch on for every suite at once.

        Only shapes this entity added are touched: a scene-declared object's
        appearance belongs to the asset that declared it.
        """
        from robosimstudio.core.render.finish import finish_for
        from robosimstudio.core.render.palette import default_for, is_build_order_color
        from robosimstudio.scene.build.imports import is_textured

        color = getattr(self.spec, "color", None)
        finish = getattr(self.spec, "finish", None)
        visible = getattr(self.spec, "visible", True)
        collide = getattr(self.spec, "collide", True)
        stated = None if color is None else tuple(float(v) for v in color)
        role = self.display_role
        fallback = default_for(role)
        for s in range(*self.range.shapes):
            if not visible:
                builder.shape_flags[s] &= ~int(newton.ShapeFlags.VISIBLE)
            if not collide:
                builder.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            # A colour on a textured shape is what clear_texture_tints undoes,
            # scene-wide, once every import has run. Skipping here as well is not
            # redundant: this method is reachable from build_into without the
            # compiler, and a default that only survives because a later pass
            # happens to erase it is a rule stated in the wrong place.
            if is_textured(builder, s):
                # ...and the finish that rides on it CARRIES that texture, which
                # is the difference between a labelled jar and a white one. The
                # path tracer binds a material per drawn instance from this
                # finish, and an instance binding beats the textured material
                # the importer left on the prototype -- so handing it
                # ``finish_for(None, role)``, a finish with no texture, erased
                # the image and left the white ``clear_texture_tints`` had just
                # set. Every textured shape in the corpus rendered white under
                # RTX for that reason. A stated finish is still honoured; only
                # one that names no texture of its own inherits the shape's.
                base = finish if finish is not None else finish_for(None, role)
                if not base.texture:
                    source = getattr(builder.shape_source[s], "texture", None)
                    # A mesh may carry its texture as raw image data rather than
                    # a path; a Finish names a file, so that case keeps the
                    # importer's own material and is left alone.
                    if isinstance(source, str | os.PathLike):
                        base = replace(base, texture=str(source))
                self.finishes[s] = base
                continue
            if stated is not None:
                builder.shape_color[s] = stated
            elif is_build_order_color(builder.shape_color[s], s):
                builder.shape_color[s] = fallback
            # The finish rides on the colour the shape *ends up* with, so a
            # spec that states only a palette colour gets that colour's finish.
            painted = tuple(float(v) for v in builder.shape_color[s])
            self.finishes[s] = finish if finish is not None else finish_for(painted, role)

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        """The model is finalized. Resolve indices and allocate device arrays."""
        self.model = model

    def apply_model_materials(self, model: newton.Model) -> None:
        """Write whatever model-wide arrays this entity's materials need.

        After every entity is bound and before the solvers are composed -- the
        one window where ``shape_material_mu`` or ``soft_contact_ke`` can be
        written and still be seen, since a solver that keeps its own copy
        takes it at construction. A cloth sets the soft-contact scalars here; a
        rod carves the scene's friction into the three materials that matter.
        Two entities writing one model-wide scalar is a conflict the composer
        reports, so an entity that needs a *global* number states it once.
        """

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks: SubstepHooks) -> None:
        """Add this entity's substep kernels.

        Every hook must be a device kernel gated by a device-side flag and
        launched unconditionally, or the substep block stops being capturable.
        """

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """The pipeline can be stepped. Settle, snapshot, and anything else
        that needs physics to actually run."""

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline: ScenePipeline, episode, *, world: int = 0) -> None:
        """Restore this entity's start state for a new episode.

        Args:
            world: Which world of a replicated build is being reset. Defaulted,
                so an entity that owns no per-world state never mentions it and
                a single-world scene never notices it exists. An entity that
                declares :attr:`world_aware` is called once per world and owes
                this argument the same respect its kernels owe
                :attr:`worlds` -- a reset that ignores it restores world 0 as
                many times as there are worlds.
        """

    # ---------------------------------------------------------------- observe


def entity_layouts(entity: Any) -> tuple[str, ...]:
    """Every layout ``entity``'s class and its mixins declare, in MRO order, deduped."""
    if entity is None:
        return ()
    out: list[str] = []
    for cls in type(entity).__mro__:
        for layout in getattr(cls, "layouts", ()) if "layouts" in vars(cls) else ():
            if layout not in out:
                out.append(layout)
    return tuple(out)


def label_index(labels: Sequence[str], name: str) -> int:
    """Index of the one label ending in ``/name``.

    Importers carry the full asset path in a label, so a bare name is only ever
    a suffix of it. Ambiguity raises rather than silently taking the first
    match: a wrong index here surfaces much later as an arm that moves the
    wrong axis or a goal scored against the wrong body.
    """
    hits = [i for i, label in enumerate(labels) if label.split("/")[-1] == name]
    if len(hits) != 1:
        raise KeyError(f"{name!r} matched {len(hits)} labels, expected exactly 1")
    return hits[0]


def label_indices(labels: Sequence[str], predicate) -> list[int]:
    """Every index whose label's last path segment satisfies ``predicate``."""
    return [i for i, label in enumerate(labels) if predicate(label.split("/")[-1])]
