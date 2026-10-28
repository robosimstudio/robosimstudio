# SPDX-License-Identifier: Apache-2.0
"""What a thing in a scene is: geometry, material, and where it may sit.

The bottom of the IR. Every other spec module imports from this one and none of
it imports back -- an asset does not know what task uses it, which scene it
stands in, or which robot reaches for it, and that ignorance is what lets one
asset serve many tasks.

**Produced by** the asset library, lowering a declaration
(:mod:`robosimstudio.assetlib.lower`), and by task loaders that place what a
document names. **Consumed by** the scene compiler and its entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from robosimstudio.core.render.finish import Finish

from .base import IDENTITY, Quat, Vec3, Xform
from .params import RECAPTURE, ui
from .physics import SimRequirements

__all__ = [
    "AssetSpec",
    "BodySpec",
    "ContactMaterialSpec",
    "JointDriveSpec",
    "ObjectSpec",
    "RegionSpec",
    "ShapeSpec",
    "StaticSpec",
]


@dataclass(frozen=True, kw_only=True)
class AssetSpec:
    """A file on disk, resolved relative to the vendored asset root.

    Attributes:
        path: Path under ``assets/``, or an absolute path.
        collision_path: Optional separate mesh used only for collision. The
            primary ``path`` remains visible and supplies mass/inertia; this
            mesh is invisible, carries the same material, and is the only one
            with collision enabled. Intended for lightweight functional proxies
            that preserve holes or cavities -- a wrench ring or a scissor
            handle that a raw hull would seal shut. Refused in combination with
            ``approximate`` or with ``ObjectSpec.collider``: the first would
            swallow the proxies into the very hull they exist to avoid, the
            second is the built-in-primitive answer to the same problem and a
            body cannot take both.
        collision_paths: Additional collision-only meshes attached to the same
            rigid body. This is the compound-collider form of
            ``collision_path`` -- a CoACD decomposition arrives as one file per
            hull; the singular field remains supported for the common one-proxy
            case.
        scale: Uniform scale applied on import.
        approximate: Collider approximation to apply after import --
            ``"bounding_box"``, ``"convex_hull"``, or ``None`` to keep the
            authored geometry. Boxifying a thin tool is not cosmetic: a raw
            convex hull leaves it rocking in a limit cycle that creeps 15-25 mm
            while nominally settled, where a box settles to under a tenth of a
            millimetre.
        keep_visual: Keep the authored visual mesh when approximating the
            collider, so the swap is invisible on camera.
        skip_mesh_approximation: Keep a USD stage's authored meshes rather than
            the collision approximation it asks for. USD only: an Isaac-exported
            grocery is one mesh serving as both render geometry and collider,
            carrying ``physics:approximation = convexDecomposition``; Newton
            remeshes it and does not keep a visual copy, so the object renders as
            a hull and its UVs -- and therefore its printed label -- are lost.
            Skipping the approximation is what makes the groceries legible to a
            VLM, at the cost of collision against the full mesh.
        texture: Diffuse map for a **mesh** import, beside the geometry rather
            than inside it. Empty draws the mesh in its build-order palette
            colour.

            Stated rather than discovered, because a bare ``.obj`` says nothing
            a loader can trust: it may name an ``.mtl`` that names an image, or
            name nothing, or name one that is not shipped. All thirty textured
            tools under ``vab_tool_sort/assets/industrial/`` arrive the same way -- UVs on
            the mesh, the image in a sibling file, nothing connecting the two.

            **A grocery without its label is a different object.** The tasks
            that name these meshes say "put the milk in the basket", and a
            carton drawn flat grey is a question the picture no longer contains
            -- the same argument
            ``texture_box_geoms`` makes for the
            compartment plates a sorting instruction asks a policy to read.
        material_colors: Preserve the diffuse colours from an OBJ's MTL file.
            The physics mesh remains unchanged; one non-colliding visual shape
            is added for each authored material section. Intended for assets
            whose appearance is several flat materials rather than one texture.
    """

    path: str
    collision_path: str | None = None
    collision_paths: tuple[str, ...] = ()
    scale: float = 1.0
    approximate: str | None = None
    keep_visual: bool = True
    skip_mesh_approximation: bool = False
    texture: str = ""
    material_colors: bool = False


@dataclass(frozen=True, kw_only=True)
class ContactMaterialSpec:
    """What one entity's shapes feel when they touch something.

    A view of ``model.shape_material_*`` over the shapes one entity built, in the
    units Newton keeps them in. Read back rather than remembered: after an import,
    a collider swap or a preset load, the model is the only thing that knows.

    Attributes:
        mu: Sliding friction coefficient. The one that decides whether a grip
            holds -- ``copy_shape_material``'s docstring measures a
            can sliding 27 mm in the fingers and dropping out on the traverse,
            from a substitute collider that took the builder's default of 1.0
            where the mesh authored 2.0.
        mu_rolling: Rolling friction coefficient.
        mu_torsional: Torsional friction coefficient -- resistance to spinning
            about the contact normal, which is what stops a grasped cylinder
            twisting out of the jaws.
        restitution: Bounciness in [0, 1].
        ke: Contact normal stiffness [N/m].
        kd: Contact normal damping [N·s/m].
        kf: Contact friction stiffness [N/m].
        ka: Contact adhesion distance [m].
        thickness: Shape thickness [m], which offsets the contact surface.
    """

    mu: float = field(default=1.0, metadata=ui(tier=RECAPTURE, range=(0.0, 3.0), step=0.01, group="friction"))
    mu_rolling: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(0.0, 1.0), group="friction"))
    mu_torsional: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(0.0, 1.0), group="friction"))
    restitution: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(0.0, 1.0), group="response"))
    ke: float = field(
        default=1.0e5,
        metadata=ui(tier=RECAPTURE, unit="N/m", log=True, range=(1.0e2, 1.0e7), group="response"),
    )
    kd: float = field(
        default=1.0e3,
        metadata=ui(tier=RECAPTURE, unit="N*s/m", log=True, range=(1.0e-2, 1.0e5), group="response"),
    )
    kf: float = field(
        default=0.0,
        metadata=ui(tier=RECAPTURE, unit="N/m", log=False, range=(0.0, 1.0e5), group="response"),
    )
    ka: float = field(default=0.0, metadata=ui(tier=RECAPTURE, unit="m", range=(0.0, 0.01), group="response"))
    thickness: float = field(default=0.0, metadata=ui(tier=RECAPTURE, unit="m", range=(0.0, 0.02), group="geometry"))


@dataclass(frozen=True, kw_only=True)
class ShapeSpec:
    """A built-in primitive, for a thing that has no mesh.

    Attributes:
        kind: ``"box"``, ``"sphere"``, ``"capsule"``, ``"cylinder"`` or
            ``"cone"`` -- the primitives ``ModelBuilder`` draws.
        half_extents: Half-size for a box [m].
        radius: Radius for everything else [m].
        half_height: Half-length along the axis, for a capsule, cylinder or
            cone [m].
    """

    kind: str = "box"
    half_extents: Vec3 = (0.05, 0.05, 0.05)
    radius: float = 0.05
    half_height: float = 0.05

    KINDS: ClassVar[tuple[str, ...]] = ("box", "sphere", "capsule", "cylinder", "cone")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"shape kind must be one of {self.KINDS}, got {self.kind!r}")

    def add_to(self, builder, *, body: int, xform, label: str, cfg=None) -> int:
        """Draw this primitive onto ``body``, returning the shape index.

        Here rather than in an entity because both of the entities that draw one
        would otherwise carry the same five-branch dispatch, which is how the
        *asset* path came to be written twice.
        """
        common = {"body": body, "xform": xform, "label": label}
        if cfg is not None:
            common["cfg"] = cfg
        if self.kind == "box":
            hx, hy, hz = (float(v) for v in self.half_extents)
            return builder.add_shape_box(hx=hx, hy=hy, hz=hz, **common)
        if self.kind == "sphere":
            return builder.add_shape_sphere(radius=float(self.radius), **common)
        if self.kind == "capsule":
            return builder.add_shape_capsule(radius=float(self.radius), half_height=float(self.half_height), **common)
        if self.kind == "cylinder":
            return builder.add_shape_cylinder(radius=float(self.radius), half_height=float(self.half_height), **common)
        return builder.add_shape_cone(radius=float(self.radius), half_height=float(self.half_height), **common)


@dataclass(frozen=True, kw_only=True)
class BodySpec:
    """What one entity's bodies weigh.

    Attributes:
        mass: Mass of each body [kg]. Writing it scales the inertia tensor by the
            same factor -- the same shape, less dense. See
            :class:`~robosimstudio.core.tuning.BodyWriter` for why leaving the
            tensor alone is not the lighter object anyone means.
    """

    mass: float = field(
        default=1.0,
        metadata=ui(tier=RECAPTURE, unit="kg", log=True, range=(1.0e-3, 1.0e2), group="mass"),
    )


@dataclass(frozen=True, kw_only=True)
class JointDriveSpec:
    """How one entity's joints are driven, and how far they may travel.

    Attributes:
        target_ke: Drive stiffness towards the commanded coordinate.
        target_kd: Drive damping.
        armature: Added rotor inertia, which is what makes a stiff drive stable.
        friction: Static joint friction.
        limit_lower: Lower travel limit, in the joint's own units.
        limit_upper: Upper travel limit.
    """

    target_ke: float = field(default=1.0e2, metadata=ui(tier=RECAPTURE, log=True, range=(1.0e-2, 1.0e6), group="drive"))
    target_kd: float = field(default=1.0e1, metadata=ui(tier=RECAPTURE, log=True, range=(1.0e-3, 1.0e4), group="drive"))
    armature: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(0.0, 1.0), group="drive"))
    friction: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(0.0, 10.0), group="drive"))
    limit_lower: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(-6.284, 6.284), group="travel"))
    limit_upper: float = field(default=0.0, metadata=ui(tier=RECAPTURE, range=(-6.284, 6.284), group="travel"))


BUILD_STAGES: tuple[str, ...] = ("scenery", "objects", "deformables", "extras")
"""The order the compiler builds a scene in, and why it is declared rather than implied.

This order is load-bearing and always was -- ``compile_scene``'s docstring says
every step "is there because doing it later (or earlier) fails, usually
silently": a scene's floor must not land on top of a fixture standing on it, and
the robot must build before the loose bodies so that dropping them for an IK rig
leaves every earlier index untouched.

It used to be implied by *which of four tuples* a spec sat in, which is why a
new kind of thing had to pick a bucket or be an "extra". A spec says its stage
now, so the compiler runs one loop and a new kind of thing declares where it
belongs. The robot is not here: it is built from ``TaskSpec.embodiment`` between
``scenery`` and ``objects``, and is not a spec in the list.

Anything that declares no stage is an ``extras`` -- last, after everything it
might need to attach to.
"""


def build_stage(spec: Any) -> str:
    """Which stage builds this spec. Duck-typed, like ``entity``."""
    return str(getattr(spec, "stage", "") or "extras")


def _category_of(name: str) -> str:
    """An object's category from its name when none is declared: ``hammer_0`` is a ``hammer``."""
    head, _, tail = name.rpartition("_")
    return head if head and tail.isdigit() else name


@dataclass(frozen=True, kw_only=True)
class ObjectSpec:
    """One manipulable rigid body.

    Attributes:
        name: Key in handles, observations and goals.
        category: What kind of thing it is, e.g. ``"hammer"``. Goals and
            instructions are written against categories; instances add a suffix.
        asset: Its geometry, or ``None`` when it comes from the scene asset
            already (a merged MJCF export declares its own objects, and the
            spec only names them so the compiler can find and index them).
        body_name: Body label suffix in the imported scene. Empty uses
            :attr:`name`.
        joint_name: Free-joint label suffix, for writing an initial pose. Empty
            uses ``"<name>_joint0"``.
        pose: Initial position [m].
        quat: Initial orientation, xyzw.
        static: Whether it is welded to the world rather than free.
        obb_grasp: Compute an OBB grasp box for it at build time.
        collider: Replace this object's mesh colliders after import --
            ``"slabs"``, ``"aabb"``, ``"bounding_box"``, ``"convex_hull"``, or
            ``""`` to keep the authored geometry.

            Either way an object arrived -- declared by the scene asset or
            imported from its own file. An imported one may instead say it
            through :attr:`AssetSpec.approximate`, which is applied during the
            import rather than after it; stating both is refused, since they are
            two opinions about one collider. The reason to state it *here* for an
            imported object is that ``approximate`` reaches only what
            ``approximate_meshes`` offers, and the two below that matter most for
            a grocery are built rather than fitted.

            **A flat-faced collider is not cosmetic.** A curved or many-vertex
            one resting on a flat surface gives the solver a contact set whose
            points flicker between steps, and the residual tangential impulse
            makes the object creep: measured on the grocery scene, five items
            nominally at rest slid 0.67-3.25 mm in xy over two seconds, z steady.
            Boxes cut that to 0.047-0.136 mm. A convex hull does not help.

            Which box, though, differs per object, which is why this is stated
            rather than inferred. ``"bounding_box"`` fits an *oriented* box, right
            for a thin tool lying at an angle in a bin; on anything round its yaw
            is arbitrary and the body-frame bound of a box rotated 45 degrees is
            sqrt(2) too large -- it inflated a 71 mm can's grasp box to 96 mm,
            wider than the gripper's whole stroke, so the item read as
            ungraspable. ``"aabb"`` is one box at the mesh's own body-frame bound:
            flat faces *and* unchanged extents, which is what an upright grocery
            wants.

            ``"slabs"`` is ``"aabb"`` taken per band along the object's long axis,
            and it exists because one box cannot be narrower *anywhere*: it made
            the mustard bottle a full-width block up to its cap and the milk
            carton one up to its gable ridge, so both were grasped and scored
            against a shape they do not have. On the real meshes the bottle
            resolves into 7 boxes down to a 12.6 mm cap and the carton into a
            73.3 mm body plus a 27.9 mm ridge. Bands whose cross-sections agree
            are merged, so a prism yields the single box ``"aabb"`` already gave
            it -- the three cans are unchanged, and only tapered items spend more
            shapes. Measured worst creep 0.089 mm, and grasp boxes are unchanged
            because the outermost band still reaches the mesh's bound. See
            ``slab_decomposition``.
        cavity: The interior a ``"shell"`` collider must leave open, as
            ``(x0, y0, z0, x1, y1, z1)`` in the body frame [m]. ``None``
            everywhere else.

            **A hollow container imported as a mesh is not hollow.** MuJoCo
            convexifies a mesh collider, so a basket, a tote or a bowl arrives as
            a sealed block and nothing can be placed in it -- and the failure
            looks like a policy that cannot aim rather than like a scene that has
            no inside. The ``"shell"`` collider replaces the mesh with a floor and
            four walls fitted to this rectangle, which is the generalisation of
            what ``the retired grocery_stage module`` does for its one tote.

            Measured against the mesh's own bound, so a cavity that does not sit
            inside it is refused rather than producing a wall of negative
            thickness -- a box the builder accepts and the solver reads
            inside-out, leaving the container with no walls at all.
        bbox: Explicit bounding-box extents [m] for predicates that need one
            (a model database's numbers, frozen at t=0). Empty derives it.
        density: Material density for an *imported* mesh object [kg/m³]. A bare
            STL carries no mass; a scene-declared object ignores this because
            its exporter already said.
        mass: Total mass [kg] for an *imported* object, or ``None`` to let
            :attr:`density` and the geometry decide. Set this when the source
            states a mass, which a benchmark export generally does: density
            times the mesh's own volume is not that mass unless the mesh is
            solid, and the gap is the mesh's hollowness -- 1.7% fill on a pair
            of safety glasses, so the density that would be right for the box a
            collider approximation stands in for makes the object 47x light.
            Applied by rescaling, so the shape's inertia distribution survives;
            see ``set_body_mass``. Ignored when :attr:`static`: a welded body
            is kinematic, and no solver reads its mass.
        friction: Friction for the imported shapes, or ``None`` for Newton's
            default. Scene-declared objects state theirs in the scene.
        contact_gap: Contact activation gap for the imported shapes [m], or
            ``None`` for the builder's default. A tight-clearance part wants
            this at the scale of its fits.
        ke: Contact stiffness for the imported shapes [N/m], or ``None`` for the
            builder's default of 2500.
        kd: Contact damping for the imported shapes, or ``None`` for the
            builder's default of 100.

            **These exist because a mesh imported on its own arrives softer than
            the same mesh imported inside an MJCF, by a factor of four hundred.**
            A ``<geom>`` carries ``solref`` and ``solimp``, and Newton's MJCF
            importer turns them into a stiffness: measured on
            ``vab_tool_sort/assets/industrial/screwdriver/screwdriver.xml``,
            whose ``solref="0.001 1" solimp="0.998 0.998 0.001"`` becomes
            ``ke = 1e6, kd = 2000``. The same file's ``meshes/screwdriver.obj``,
            imported through :class:`AssetSpec`, gets 2500 and 100 -- because an
            OBJ carries no contact model and nothing was reading the MJCF beside
            it.

            2500 N/m is about four millimetres of give per ten newtons, which a
            resting object never notices and a position-controlled arm pressing
            down defeats completely: the object goes into the table, and the
            solver's attempt to push it back out is a buzz. Both are what a suite
            staging its objects per-object was seeing and a suite importing an
            arena was not.

            Stated per object rather than raised on
            :attr:`~robosimstudio.specs.physics.CollisionSpec.contact_ke`, which is scene-wide: the claim being
            made is "this part's own file says how hard it is", not "everything
            in this cell is stiffer". Carried across a collider approximation by
            ``copy_shape_material``.
        color: Display RGB in [0, 1] for the shapes this spec *imports*, or
            ``None`` to leave them as they arrived. A bare mesh (an STL, say)
            arrives with no appearance at all, so ``None`` there means the
            build-order palette picks -- see
            :meth:`~robosimstudio.scene.entity.Entity.apply_display`.
            A scene-declared object ignores this: its appearance is the scene's.
        finish: How the shapes reflect light -- roughness, metallicity,
            opacity, a diffuse map -- or ``None`` for the finish keyed to the
            colour they end up painted (:func:`~robosimstudio.core.render.finish.finish_for`).
            Display only, and honoured only by the path-traced backend; see
            :mod:`~robosimstudio.core.render.finish`.
        visual_markers: Non-colliding body-local point markers, each expressed
            as ``(position, radius, color)``. They move rigidly with the object
            and are intended for inspecting authored landmarks without changing
            its mass or contact geometry.
        hydroelastic: Collide this object's meshes as a **pressure field**
            rather than as a point set. Newton builds an iso-pressure patch
            where two overlapping bodies meet and emits contacts across it, so a
            tab in a slot is carried by a distributed surface instead of by
            whichever handful of points the narrow phase picked that step.

            It composes with any solver, MuJoCo included, because it changes
            what the narrow phase **emits** rather than who integrates it -- so
            an MJCF's own ``<position>`` actuators and ``gravcomp`` keep driving
            the arm. That is the property that makes it usable on a fit whose
            arm is MuJoCo's; swapping the *solver* to get a different contact
            model loses them.

            Requires a cooked SDF, so the entity builds one --
            ``build_hydroelastic_sdf``, slow
            once and cached on disk. **Both ends of a mate must declare it**: a
            hydroelastic shape against a point-contact shape falls back to
            points, silently, and the pair behaves as though neither had asked.
        kh: Hydroelastic contact stiffness [Pa/m] -- the slope of the default
            linear pressure law. Read only when :attr:`hydroelastic` is set, and
            it has to be stiff or the patch resolves too slowly to hold a fit:
            Newton's own nut-and-bolt example runs 1e11 against a 1e10 default
            and records the nut sitting visibly cocked at the default.
        sdf_voxel: Cooked-SDF voxel size [m] for this object's field, or
            ``None`` for the builder's global resolution cap. Read only when
            :attr:`hydroelastic` is set. The cooked field places a surface
            exactly only where the mesh is planar and *isolated at the voxel
            scale*; surfaces within a voxel of each other smear, and the field
            collides at whichever one the smear favours, give or take a cell.
            The bulkhead slot is the motivating case: a double-walled shell
            0.7 mm skin to skin, cooked at 128 voxels over a 66 mm part
            (0.52 mm cells), collided +-half a millimetre from the wall the
            mesh states -- which was the entire 0.2..0.45 mm design space of
            the fit the slot was supposed to carry. 0.15 mm cells put both
            skins and the slot liner exactly where the mesh says, for a
            few-second cook, cached content-addressed like every other.
        world_xform: This spec's :attr:`pose` is world coordinates rather than
            the scene's work-surface frame. The :class:`StaticSpec` field of the
            same name, for the same reason, one level down: an object imported
            *into* a world-placed arena is positioned in that arena's
            coordinates, not in the surface frame the scene axis would compose
            on. Needed by an object whose pose came out of the same export as
            the room it stands in -- a scene-declared object is already exempt
            because nothing places it at all, and this is how an object that
            carries its own asset says the same thing.

            Getting this wrong is not subtle but it is silent: the frame
            composes and the whole pile lands one surface origin away, which on
            a tool-sorting scene is (+0.100, +0.015, +0.780) and reads as every object
            spawning in the air and falling.
    """

    sim: SimRequirements | None = None
    """What this thing needs of physics -- its domain, what it touches, how it
    couples -- or ``None`` for the entity class's default. The asset record's
    declaration, carried onto the spec by the loader that built it, so the
    composer reads one thing."""

    stage: ClassVar[str] = "objects"
    shape: ShapeSpec | None = None
    """A built-in primitive to draw instead of importing a file.

    ``None`` keeps every existing path exactly as it was: geometry comes from
    :attr:`asset`, or (for a static) from :attr:`half_extents`, or the thing is
    declared inside the cell's own file and is only found here."""
    entity: str = "rigid"
    """Registry name of the entity class that builds this.

    Defaulted rather than derived, and that is the point: the compiler used to
    keep a table of one class per build stage and reach for it whenever a spec
    named none. A spec says what builds it now -- always -- so there is one
    rule, and a subclass that needs its own physics overrides one field."""
    name: str
    category: str = ""
    asset: AssetSpec | None = None
    body_name: str = ""
    joint_name: str = ""
    pose: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    static: bool = False
    world_xform: bool = False
    obb_grasp: bool = True
    collider: str = ""
    cavity: tuple[float, float, float, float, float, float] | None = None
    bbox: Vec3 | None = None
    density: float = 1000.0
    mass: float | None = None
    hydroelastic: bool = False
    kh: float = 1.0e11
    sdf_voxel: float | None = None
    friction: float | None = None
    contact_gap: float | None = None
    ke: float | None = None
    kd: float | None = None
    color: Vec3 | None = None
    finish: Finish | None = None
    visual_markers: tuple[tuple[Vec3, float, Vec3], ...] = ()

    def __post_init__(self) -> None:
        """Resolve :attr:`category` once, so there is one name for one fact.

        The fallback is kept and is purely defensive: an object named
        ``hammer_0`` with nothing declared is a ``hammer``. Checked before
        collapsing the two names -- across all 140 ``ObjectSpec`` instances
        in every shipped variant, ``category`` is never empty and never
        differed from what ``kind`` returned, so nothing in the corpus takes
        this branch.
        """
        if not self.category:
            object.__setattr__(self, "category", _category_of(self.name))


@dataclass(frozen=True, kw_only=True)
class StaticSpec:
    """Scenery: an arena asset, a table, a wall, a bin.

    Attributes:
        name: Key in handles.
        asset: The file to import, or ``None`` for a procedural box.
        xform: Where it goes.
        half_extents: Box half-extents [m] when :attr:`asset` is ``None``.
        visible: Whether it is drawn. A room shell that a browser viewer turns
            into vast untextured slabs is better hidden -- nothing collides with
            it, the arm cannot reach it, and the ground grid already gives the
            scene a floor.
        collide: Whether it collides.
        friction: Sliding friction for this piece's shapes, or ``None`` for the
            builder's default of 1.0.

            Worth stating for a **work surface**, which is the one static that
            everything a task is about rests on. A benchmark's table has a
            friction and it is rarely 1.0 -- SAPIEN's default is 0.3, and taking
            Newton's instead triples the tangential impulse a resting object's
            flickering contact set is allowed to produce. Measured on the
            SimplerEnv bridge table, a single-hull carrot nominally at rest slid
            11.0 mm over twelve seconds at 1.0 and 1.4 mm at 0.3, orientation and
            height steady throughout -- sliding, not rolling or bouncing.
        world_xform: This spec's :attr:`xform` is world coordinates rather than
            the scene's work-surface frame. For a static whose asset *is* the
            room -- a whole arena or stage the task imports because it predates
            the scene axis -- since a room is not positioned relative to itself.
        color: Display RGB in [0, 1], or ``None`` for the importer's own.
        strip_free_bodies: Remove free-jointed bodies from an MJCF before
            import, and its keyframes with them -- a keyframe indexes the qpos
            of the bodies just removed.
        meta: Path to a JSON sidecar carrying what MJCF has no place for
            (settled poses, compartment centres, joint names), relative to the
            asset.
        collapse_fixed_joints: Fold the asset's fixed joints. Set on a USD stage
            whose robot carries a welded wrist flange and gripper base: keeping
            them hands the solver zero-DOF articulations to maintain for nothing.
            MJCF arenas take this through ``strip_free_bodies`` instead, which is
            also what selects the collapsing entity.
        expose_declared_colliders: Import the colliders the stage declares in
            the two forms ``add_usd`` cannot follow -- inside an instance
            prototype, or with ``CollisionAPI`` on an Xform rather than on the
            mesh under it. USD only. Both are dropped silently, and on an Isaac
            gripper they drop *asymmetrically*: eight of eleven collision prims
            are unreachable, and the three that import give one inner finger two
            shapes and the other one. Off by default -- on the grocery stage that
            asymmetry measured harmless and recovering the eight measured worse.
            See ``expose_declared_colliders``.
        finish: How the shapes reflect light, or ``None`` for the finish keyed
            to their painted colour. Display only; the path-traced backend is
            the one that reads it. See :mod:`~robosimstudio.core.render.finish`.
        entity: Registry name of the entity class that realises this static, or
            empty for the generic one. The same seam :class:`ClothSpec` has, and
            it exists for the same reason: a benchmark scene sometimes needs a
            correction only its own suite can justify -- the grocery tote's
            collider has to become a hollow shell -- and that belongs beside the
            suite rather than as another flag here.
    """

    sim: SimRequirements | None = None
    """What this thing needs of physics -- its domain, what it touches, how it
    couples -- or ``None`` for the entity class's default. The asset record's
    declaration, carried onto the spec by the loader that built it, so the
    composer reads one thing."""

    stage: ClassVar[str] = "scenery"
    shape: ShapeSpec | None = None
    """A built-in primitive to draw instead of importing a file.

    ``None`` keeps every existing path exactly as it was: geometry comes from
    :attr:`asset`, or (for a static) from :attr:`half_extents`, or the thing is
    declared inside the cell's own file and is only found here."""
    name: str
    asset: AssetSpec | None = None
    xform: Xform = IDENTITY
    half_extents: Vec3 | None = None
    visible: bool = True
    collide: bool = True
    friction: float | None = None
    color: Vec3 | None = None
    finish: Finish | None = None
    world_xform: bool = False
    strip_free_bodies: bool = False
    meta: str = ""
    collapse_fixed_joints: bool = False
    expose_declared_colliders: bool = False
    entity: str = "static"


@dataclass(frozen=True, kw_only=True)
class JointSpec:
    """One named joint of an :class:`ArticulatedSpec`: where it starts, and how it is driven.

    Every field but :attr:`name` is ``None`` by default, and ``None`` keeps what
    the asset authored. That is why this is not a :class:`JointDriveSpec`, whose
    defaults are numbers: a cabinet's drawer arrives with its travel limits and
    its drive gains in the file, and a spec that only wanted to start the drawer
    open must not also zero its limits.

    Attributes:
        name: The joint's name, as the entity's handle lists it (SimReady's
            ``joint_drawer_01_joint_00`` is ``drawer_01``).
        q: Start coordinate, in the joint's own units (m for a prismatic joint,
            rad for a revolute one). An episode's ``joint_q`` overrides it.
        damping: Passive joint damping.
        friction: Static joint friction.
        armature: Added rotor inertia.
        target_ke: Drive stiffness towards the target coordinate. Setting it
            drives the joint, from :attr:`q` unless a controller writes a target.
        target_kd: Drive damping.
    """

    name: str
    q: float | None = None
    damping: float | None = None
    friction: float | None = None
    armature: float | None = None
    target_ke: float | None = None
    target_kd: float | None = None


@dataclass(frozen=True, kw_only=True)
class ArticulatedSpec:
    """A jointed object: a cabinet with drawers, a box with a hinged lid, a door.

    Its parts are the asset's rigid bodies and its joints the asset's own, so the
    asset is a USD stage that declares them -- NVIDIA's SimReady assets are the
    case it was written for. A stage whose joints belong to no articulation is
    given one root on import (:func:`~robosimstudio.scene.build.imports.articulate`).

    Attributes:
        name: Entity name.
        category: What it is, for goals written against categories. The name
            less a trailing ``_<digits>`` when empty, as for :class:`ObjectSpec`.
        asset: The USD stage.
        pose: Position of the asset's origin, in the scene's work-surface frame
            unless :attr:`world_xform`. SimReady assets put the origin at the base.
        quat: Orientation, xyzw.
        world_xform: :attr:`pose` is world coordinates; see :class:`ObjectSpec`.
        fixed_base: Bolt the root part to the world (a cabinet, a door frame).
            ``False`` leaves the root on a free joint, so the whole thing can be
            carried (a box with a hinged lid). An asset in separate parts that no
            joint holds together -- a jar and its lid, a rack and its boxes -- is
            one entity with each part its own articulation: ``True`` bolts the
            heaviest and frees the rest, ``False`` frees them all.
        joints: Per-joint start coordinates and drive overrides, by name.
        self_collide: Let the parts collide with each other. Off by default: a
            drawer sits inside its cabinet's collider, and colliding the two
            jams it on the first step.
        color: Display RGB for every part, or ``None`` for the asset's own.
        finish: How the parts reflect light, or ``None``; see :class:`ObjectSpec`.
    """

    sim: SimRequirements | None = None
    """What this thing needs of physics, or ``None`` for the entity class's default."""

    stage: ClassVar[str] = "objects"
    name: str
    category: str = ""
    asset: AssetSpec
    pose: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    world_xform: bool = False
    fixed_base: bool = True
    joints: tuple[JointSpec, ...] = ()
    self_collide: bool = False
    color: Vec3 | None = None
    finish: Finish | None = None
    entity: str = "articulated_object"

    def __post_init__(self) -> None:
        if not self.category:
            object.__setattr__(self, "category", _category_of(self.name))
        names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"articulated {self.name!r}: a joint is named twice in {names}")


@dataclass(frozen=True, kw_only=True)
class RegionSpec:
    """A goal region, in the frame of whatever it is attached to.

    The convention is BDDL's: ranges are ``(x0, y0, x1, y1)`` in the target's
    own frame, and containment is that rectangle plus a height window starting
    at the frame origin. A bin's origin is its body origin, so its regions are
    compartment coordinates and the window starts at the bin floor.

    Attributes:
        name: Key in handles and goals.
        frame: Name of the static or object whose frame the ranges are in. Empty is
            the work surface: the middle of the bench top, the frame ``at=`` uses.
        xy_ranges: ``(x0, y0, x1, y1)`` [m].
        z_range: Height window above the frame origin ``(lo, hi)`` [m]. The
            default 0 to 0.15 m is BDDL's literal window.
    """

    name: str
    frame: str
    xy_ranges: tuple[float, float, float, float]
    z_range: tuple[float, float] = (0.0, 0.15)
