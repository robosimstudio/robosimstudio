# SPDX-License-Identifier: Apache-2.0
"""Cloth in one file: a deformable sheet, from its spec to its registration.

What makes cloth a kind of thing the library can build, hold and measure, in reading order:

1. **What this file is**, and how it is found.
2. **The spec**: ``ClothMaterialSpec`` and ``ClothSpec``, a sheet as data.
3. **Materials**: the measured fabrics, which ``rss.materials`` holds.
4. **The entity**: ``EntityCloth`` builds the particles, settles them and reads them back, and
   ``EntityHeldCloth`` is a sheet a body holds from the start.
5. **The constructor**: ``sheet``.
6. **Measures and calibration**: what a goal may ask of a sheet, and a fabric's bending length.
7. **Registration**: the material records, ``register_domain`` and ``register_entity``.

This file is what ``rss.sheet`` builds, imported as ``robosimstudio.domains.cloth``.
Run ``00_towel.py`` beside it first. Copy this file to write a kind of your own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib.materials import CalibrationSpec, MaterialRecord, register_material, register_material_kind
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.vocabulary.measures import (
    FOLD_DRAG_WEIGHT,
    cloth_coverage,
    cloth_fold,
    cloth_fold_error,
    held_particles,
    mirror_pairs,
)
from robosimstudio.core.pipeline import ScenePipeline
from robosimstudio.core.probes import ParticleProbe
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.shell import grid_surface, mid_surface, prism_slab
from robosimstudio.domains.capabilities import ParticleMixin
from robosimstudio.domains.selection import (
    Selector,
    resolve_selector,
    selector_beyond,
    selector_extreme,
    selector_indices,
)
from robosimstudio.paths import resolve_asset
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import AssetSpec
from robosimstudio.specs.base import Quat, Vec3
from robosimstudio.specs.params import LIVE, REBUILD, RECAPTURE, ui
from robosimstudio.specs.physics import HostActor, SimRequirements, register_domain

__all__ = [
    "BETA_MAX",
    "DROOP_BAND",
    "GRAVITY",
    "REFERENCE_FABRICS",
    "SETTLE_QUANTILE",
    "SHIRT_FLEECE",
    "SHIRT_JERSEY",
    "SHIRT_MEMBRANE",
    "SHIRT_THICKNESS",
    "TAPE_MATERIAL",
    "TERRY_MEMBRANE",
    "TERRY_TOWELLING",
    "TOWEL_THICKNESS",
    "ClothMaterialSpec",
    "ClothSpec",
    "EntityCloth",
    "EntityHeldCloth",
    "HeldClothSpec",
    "ReferenceFabric",
    "Selector",
    "bending_length_from_droop",
    "crease_of",
    "droop_curve",
    "elastica_droop",
    "fold_score",
    "layer_separation",
    "materials",
    "resembles",
    "resolve_selector",
    "selector_beyond",
    "selector_extreme",
    "selector_indices",
    "sheet",
    "write_gripper_friction",
]


# =============================================================================
# 1. What this file is, and how it is found
# =============================================================================
# Importing this file registers cloth. The library finds it on the search path of
# ``robosimstudio.domains``: import it under that name, never by file path, or it registers twice.
# A folder of your own named in ``ROBOSIMSTUDIO_DOMAIN_ROOTS`` is found the same way.


# =============================================================================
# 2. The spec: what a sheet is, as data
# =============================================================================
# ``ClothMaterialSpec`` is a fabric and ``ClothSpec`` a sheet made of one. Both are frozen
# dataclasses, also exported as ``rss.ClothMaterialSpec`` and ``rss.ClothSpec``.


@dataclass(frozen=True, kw_only=True)
class ClothMaterialSpec:
    """Cloth constitutive parameters, in SI.

    :attr:`shell` picks the model. A ``"membrane"`` is a zero-thickness mid-surface with in-plane
    stiffness on its triangles and bending on its edges: a film or a tape. A ``"slab"`` offsets
    that surface to +/- ``thickness/2`` and fills the gap with tetrahedra, so it stacks when folded
    and squashes under a finger: a bath towel. Drape and thickness are separate dials: on a slab,
    how freely the two faces slide past each other sets the bending rigidity.

    Attributes:
        tri_ke: In-plane stretch stiffness [N/m], the 2D Lamé ``mu``. On a slab each face carries it.
        tri_ka: In-plane area stiffness [N/m], the 2D Lamé ``lambda``.
        tri_kd: In-plane damping [N·s/m].
        edge_ke: Bending stiffness [N·m/rad]. A membrane's whole drape. Near zero on a slab.
        edge_kd: Bending damping [N·m·s/rad].
        areal_density: Mass per unit area [kg/m²], on a slab too: see :attr:`bulk_density`.
        particle_radius: Collision radius per particle [m]. On a membrane it is the sheet's whole
            thickness in contact. On a slab it is a thin skin: see :meth:`default_particle_radius`.
        zero_rest_angle: Zero the edge rest angles after finalize, so the sheet is flat at rest.
        friction: Friction of the cloth against itself and against shapes. Keep it low on an
            adhesive: if friction takes the shear, the bonds never load and no peel front forms.
        contact_ke: Soft-contact stiffness [N/m].
        contact_kd: Soft-contact damping [N·s/m].
        shell: ``"membrane"`` or ``"slab"``.
        thickness: Thickness [m]. Required by a slab. A membrane records it, to size contact radii.
        core_shear_modulus: Tetrahedral Lamé ``mu`` [Pa], the drape knob. Slab only.
        core_lame_lambda: Tetrahedral Lamé ``lambda`` [Pa]. A squash feels ``lambda + 2 mu``. Slab only.
        core_damping: Tetrahedral viscous damping [Pa·s]. Slab only.
    """

    tri_ke: float = field(
        default=1.0e4,
        metadata=ui(tier=LIVE, unit="N/m", log=True, range=(1.0e2, 1.0e6), group="in-plane"),
    )
    tri_ka: float = field(
        default=1.0e4,
        metadata=ui(tier=LIVE, unit="N/m", log=True, range=(1.0e2, 1.0e6), group="in-plane"),
    )
    tri_kd: float = field(
        default=1.0e-2,
        metadata=ui(tier=LIVE, unit="N*s/m", log=True, range=(1.0e-4, 1.0e0), group="in-plane"),
    )
    edge_ke: float = field(
        default=1.0e-1,
        metadata=ui(
            tier=LIVE,
            unit="N*m/rad",
            log=True,
            range=(1.0e-5, 1.0e2),
            group="bending",
            doc="A membrane's whole drape; near zero on a slab, where the sandwich carries it.",
        ),
    )
    edge_kd: float = field(
        default=1.0e-3,
        metadata=ui(tier=LIVE, unit="N*m*s/rad", log=True, range=(1.0e-6, 1.0e-1), group="bending"),
    )
    areal_density: float = field(
        default=0.3,
        metadata=ui(
            tier=REBUILD,
            unit="kg/m^2",
            range=(0.05, 1.5),
            group="mass",
            doc="What fabric is sold by. Drape scales with it: a heavier sheet hangs in lazier folds.",
        ),
    )
    particle_radius: float = field(
        default=0.002,
        metadata=ui(
            tier=LIVE,
            unit="m",
            range=(0.0002, 0.01),
            group="contact",
            doc="Raising it far past the built value can overflow the contact buffers, which drops contacts silently.",
        ),
    )
    zero_rest_angle: bool = field(default=True, metadata=ui(tier=REBUILD, group="geometry"))
    friction: float = field(
        default=0.2,
        metadata=ui(tier=RECAPTURE, range=(0.0, 2.0), group="contact"),
    )
    contact_ke: float = field(
        default=1.0e4,
        metadata=ui(tier=RECAPTURE, unit="N/m", log=True, range=(1.0e2, 1.0e6), group="contact"),
    )
    contact_kd: float = field(
        default=1.0e-2,
        metadata=ui(tier=RECAPTURE, unit="N*s/m", log=True, range=(1.0e-3, 1.0e2), group="contact"),
    )
    shell: str = field(
        default="membrane",
        metadata=ui(tier=REBUILD, choices=("membrane", "slab"), group="geometry"),
    )
    thickness: float = field(
        default=0.0,
        metadata=ui(tier=REBUILD, unit="m", range=(0.0, 0.02), group="geometry"),
    )
    core_shear_modulus: float = field(
        default=1.0e5,
        metadata=ui(
            tier=LIVE,
            unit="Pa",
            log=True,
            range=(1.0e3, 1.0e7),
            group="core (slab)",
            doc="The drape knob. Soft core, limp towel; rigid core, cardboard.",
        ),
    )
    core_lame_lambda: float = field(
        default=0.0,
        metadata=ui(
            tier=LIVE,
            unit="Pa",
            range=(0.0, 1.0e7),
            group="core (slab)",
            doc="With mu, how hard the cloth is to squash through its thickness.",
        ),
    )
    core_damping: float = field(
        default=1.0e-2,
        metadata=ui(tier=LIVE, unit="Pa*s", log=True, range=(1.0e-4, 1.0e1), group="core (slab)"),
    )
    shell_young: float | None = field(
        default=None,
        metadata=ui(tier=REBUILD, unit="Pa", log=True, range=(1.0e3, 1.0e7), group="foreign engine"),
    )
    """Young's modulus [Pa] for an engine that wants one, or ``None`` for the island's.

    The fields above are VBD penalty stiffnesses and do not convert to a modulus. A sheet on
    SuperDex or Genesis uses this, or else its backend's ``shell_young``, shared by the island."""
    shell_poisson: float | None = field(
        default=None,
        metadata=ui(tier=REBUILD, range=(0.0, 0.49), group="foreign engine"),
    )
    """Poisson ratio to go with :attr:`shell_young`, or ``None`` for the island's."""

    def __post_init__(self) -> None:
        if self.shell not in ("membrane", "slab"):
            raise ValueError(f"cloth shell must be 'membrane' or 'slab', got {self.shell!r}")
        if self.shell == "slab" and not self.thickness > 0.0:
            raise ValueError("a slab cloth needs a positive thickness; that is the whole point of one")
        if self.thickness < 0.0:
            raise ValueError(f"cloth thickness cannot be negative, got {self.thickness}")

    @property
    def is_slab(self) -> bool:
        """Whether this cloth is built with geometric thickness."""
        return self.shell == "slab"

    @property
    def bulk_density(self) -> float:
        """Volumetric density [kg/m³] of a slab's tetrahedra: areal density over thickness."""
        if not self.thickness > 0.0:
            raise ValueError("bulk density needs a thickness")
        return self.areal_density / self.thickness

    def default_particle_radius(self) -> float:
        """The contact radius this cloth wants, from its thickness [m].

        Half the thickness on a membrane, whose particles are its thickness. A quarter on a slab,
        whose geometry carries it: half would rest two folded layers a full thickness apart.
        """
        if not self.thickness > 0.0:
            return self.particle_radius
        return 0.25 * self.thickness if self.is_slab else 0.5 * self.thickness

    def rest_exclusion_radius(self) -> float:
        """Rest-shape self-contact exclusion a slab needs [m].

        A slab's two faces rest one thickness apart and must not collide, or the solver inflates
        the sheet: pairs closer than 1.5 thicknesses at rest are excluded. A fold's contact is kept.
        """
        return 1.5 * self.thickness if self.is_slab else 0.0


@dataclass(frozen=True, kw_only=True)
class ClothSpec:
    """A deformable sheet: a grid or a mesh, a material, named particle sets and a settle.

    A sheet that needs more (adhesion, pins) subclasses this and registers its own entity.

    Attributes:
        name: Key in handles and particle-set names.
        mesh: ``"grid"`` or ``"mesh"``: where the geometry comes from.
        size: Sheet dimensions ``(length, width)`` [m], ``"grid"`` only.
        cells: Grid subdivisions ``(nx, ny)``, ``"grid"`` only.
        asset: Mesh to drape, ``"mesh"`` only.
        pose: Where the sheet's origin corner goes [m], in the cell's work-surface frame.
        quat: Orientation of the sheet's local frame, xyzw.
        material: Constitutive parameters.
        particle_sets: ``(name, selector)`` pairs. A selector takes the rest positions ``[n, 3]``
            and returns a boolean mask or an index array.
        simplify_faces: Coarsen the surface to about this many triangles, or 0 to keep the asset. A
            slab doubles the particle count, so halving a shirt's triangles buys a slab at about its
            membrane's cost. A non-manifold result is rejected: try another target.
        settle_frames: Frames of settling before the scene is ready.
        settle_substeps: Substeps per settle frame, finer than the run's own. At the run's rate a
            sheet sinks centimetres into the table before it settles, and a reset returns to that.
        color: Display RGB in [0, 1]. The tiled ray tracer ignores it and draws deformables white.
        opacity: Display alpha in (0, 1]. Below 1.0 the sheet draws translucent, as a film is.
        finish: How the sheet reflects light under the path-traced backend. ``None`` is a matte
            fabric at :attr:`opacity`, and a stated finish's opacity wins.
        decor: Named display detailing: ``""`` for none, ``"towel"`` for woven border bands.
    """

    sim: SimRequirements | None = None
    """What this sheet needs of physics, or ``None`` for the entity class's default."""

    stage: ClassVar[str] = "deformables"
    entity: str = "cloth"
    """Registry name of the entity class that builds this sheet."""
    name: str = "cloth"
    mesh: str = "grid"
    size: tuple[float, float] = (0.3, 0.3)
    cells: tuple[int, int] = (30, 30)
    asset: AssetSpec | None = None
    pose: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    material: ClothMaterialSpec = field(default_factory=ClothMaterialSpec)
    particle_sets: tuple[tuple[str, Callable[[Any], Any]], ...] = ()
    simplify_faces: int = 0
    settle_frames: int = 60
    settle_substeps: int = 10
    color: Vec3 = (0.85, 0.85, 0.88)
    opacity: float = 1.0
    finish: Finish | None = None
    decor: str = ""

    def __post_init__(self) -> None:
        if self.mesh not in ("grid", "mesh"):
            raise ValueError(f"cloth mesh must be 'grid' or 'mesh', got {self.mesh!r}")
        if self.mesh == "mesh" and self.asset is None:
            raise ValueError(f"cloth {self.name!r} is a mesh but names no asset")
        if not 0.0 < self.opacity <= 1.0:
            raise ValueError(f"cloth {self.name!r}: opacity must be in (0, 1], got {self.opacity}")


# =============================================================================
# 3. Materials: the fabrics, and what each was measured against
# =============================================================================
# Six fabrics and the two thicknesses they are built on. Section 7 registers each with the
# measurement it was calibrated against.


TOWEL_THICKNESS = 0.005
"""Towel thickness [m]: 5 mm, a 500 g/m^2 bath towel. Also the tightest radius the sheet creases
to, the gap two folded layers rest at, what a jaw squeezes before it bottoms out and, times 1.5,
the rest-shape exclusion between its faces."""

SHIRT_THICKNESS = 0.003
"""Garment thickness [m]: 3 mm at 350 g/m^2 is a fleece, not a tee. A tee is 0.5 mm: too thin for
a slab's core to add bending, so build one as a membrane."""


TERRY_TOWELLING = ClothMaterialSpec(
    shell="slab",
    thickness=TOWEL_THICKNESS,
    # A 500 g/m^2 bath towel: 100 kg/m^3 through 5 mm, a tenth of water.
    areal_density=0.5,
    # The faces: Newton's cloth values, nearly inextensible in-plane.
    tri_ke=1.0e4,
    tri_ka=1.0e4,
    tri_kd=1.5e-2,
    # The drape knob. Measured: see this fabric's calibration in section 7.
    core_shear_modulus=1.0e5,
    # Zero, so the core resists a squash with 2*mu and no more.
    core_lame_lambda=0.0,
    core_damping=1.0e-2,
    # The skin's bending adds to the core's. Near zero: at 5 mm the core alone lands in the band.
    # On a thin slab, where the core gives little, this term has to carry the drape.
    edge_ke=1.0e-4,
    edge_kd=1.0e-4,
    particle_radius=0.25 * TOWEL_THICKNESS,
    zero_rest_angle=True,
    # Cloth against itself: stacked layers neither slide off nor weld.
    friction=0.25,
    contact_ke=1.0e4,
    contact_kd=1.0e1,
)
"""Terry bath towelling: 5 mm thick, 500 g/m^2, built as a slab."""

SHIRT_FLEECE = replace(
    TERRY_TOWELLING,
    thickness=SHIRT_THICKNESS,
    # 350 g/m^2 sweatshirt fleece: 117 kg/m^3 through 3 mm.
    areal_density=0.35,
    # Ten times the towel's core. Bending goes with the square of the gap between the faces, so at
    # 3 mm the towel's core gives well under half its 5 mm rigidity. Measured on a clamped strip.
    core_shear_modulus=1.0e6,
    particle_radius=round(0.25 * SHIRT_THICKNESS, 5),
)
"""Sweatshirt fleece: 3 mm thick, 350 g/m^2, built as a slab."""

TERRY_MEMBRANE = replace(
    TERRY_TOWELLING,
    # For a fling at 2-3 m/s: half the particles, no core to invert, one surface for the air.
    shell="membrane",
    # The skin carries the whole drape. Calibrated at a 10 mm cell (section 7).
    edge_ke=6.6,
    edge_kd=1.0e-4,
    # Half the 5 mm thickness, the membrane convention.
    particle_radius=0.0025,
)
"""Terry bath towelling as a membrane, for flinging. See TERRY_TOWELLING."""

SHIRT_MEMBRANE = replace(
    SHIRT_FLEECE,
    shell="membrane",
    # The skin carries the whole drape. Calibrated at the shirt mesh's 6.4 mm mean edge (section 7).
    edge_ke=1.05,
    edge_kd=1.0e-4,
    # Half the 3 mm fabric thickness, the membrane convention.
    particle_radius=0.0015,
)
"""Sweatshirt fleece as a membrane, for flinging. See SHIRT_FLEECE."""

SHIRT_JERSEY = ClothMaterialSpec(
    # Newton's ``examples/cloth/example_cloth_franka.py``, converted from centimetre-gram-second
    # (gravity -981, a 65 cm shirt, `density=0.02`) and not retuned: both unit systems settle the
    # same mesh on the same table to within 0.3% of the garment's span. With mass in grams, tri_*
    # and contact_* are force/length, x 1e-3. edge_* is energy/length, x 1e-5, because VBD applies
    # `edge_ke * edge_rest_length` per hinge. areal_density is x 1e+1 and radii are x 1e-2.
    shell="membrane",
    tri_ke=1.0e4 * 1.0e-3,
    tri_ka=1.0e4 * 1.0e-3,
    # Newton's 1.5e-2, the same in-plane damping as TERRY_TOWELLING.
    tri_kd=1.5e-2 * 1.0e-3,
    # Newton's value, about four orders softer than the membranes above. A re-mesh moves it, since
    # stiffness per hinge is `edge_ke * edge length`: the mean edge went 11.7 -> 20.0 mm, 1.71x the
    # rigidity. Kept, not rescaled by 0.583: over three settles the sleeve-to-sleeve width is
    # 0.6939 m (spread 10.7 mm) on the 11.7 mm mesh and 0.6984 m (1.7 mm) on the 20.0 mm mesh, but
    # 0.6666 m (15.9 mm) there at 2.9e-5. A clamped strip collapses at this value (droop 1.00).
    edge_ke=5.0 * 1.0e-5,
    edge_kd=5.0e-1 * 1.0e-5,
    # 200 g/m^2, a midweight cotton jersey tee.
    areal_density=0.02 * 1.0e1,
    # 8 mm, Newton's 0.8 cm: a collision skin, not a thickness. Also what `layer_separation`
    # forgives and what sets the pinch gap. Kept absolute on the 20.0 mm mesh: Newton's 0.69 x
    # mean edge would be 13.8 mm, 72% more forgiveness and a pinch gap of 24.8 mm, not 14.4.
    # Spheres cover a triangle of edge L only for r >= 0.577 L, so use `full_surface_contact`.
    particle_radius=0.8 * 1.0e-2,
    # Flat at rest, so the worn-shirt mesh (27 cm of relief) relaxes onto a table.
    zero_rest_angle=True,
    # Newton's `self_contact_friction`: cloth against itself and against shapes. The gripper's is
    # separate: Newton raises its robot shapes to 1.5, and the folding scenes here pass 2.0 for
    # the pads and 1.2 for every other shape to `write_gripper_friction`.
    friction=0.25,
    contact_ke=1.0e4 * 1.0e-3,
    contact_kd=1.0e1 * 1.0e-3,
    # A record only: a real jersey tee is 0.5 mm. `particle_radius` sizes the contact.
    thickness=0.0005,
)
"""Newton's ``example_cloth_franka`` shirt in SI: a 200 g/m^2 jersey membrane with soft bending,
for the ``unisex_shirt.usd`` mesh that example ships."""

TAPE_MATERIAL = ClothMaterialSpec(
    # 20x a garment's stretch stiffness, as in the reference (2.0e5 against 1.0e4, in its units).
    tri_ke=2.0e4,
    tri_ka=2.0e4,
    tri_kd=1.0e-2,
    # Fitted so the settled tab reproduces the reference's droop: see its calibration in section 7.
    # 0.1 is also about what the reference's cgs edge stiffness converts to.
    edge_ke=0.1,
    edge_kd=1.0e-3,
    # The reference strip weighs 189 x 0.02 g = 3.78 g over 26 x 6 cm.
    areal_density=0.2423,
    particle_radius=0.002,
    zero_rest_angle=True,
    # Low: the bonds model the stickiness. At Newton's default, friction against the box takes the
    # shear and no peel front forms.
    friction=0.2,
    contact_ke=1.0e4,
    contact_kd=1.0e-2,
)
"""Packing tape, at a reference scene's converted values."""


materials = SimpleNamespace(
    SHIRT_FLEECE=SHIRT_FLEECE,
    SHIRT_JERSEY=SHIRT_JERSEY,
    SHIRT_MEMBRANE=SHIRT_MEMBRANE,
    SHIRT_THICKNESS=SHIRT_THICKNESS,
    TAPE_MATERIAL=TAPE_MATERIAL,
    TERRY_MEMBRANE=TERRY_MEMBRANE,
    TERRY_TOWELLING=TERRY_TOWELLING,
    TOWEL_THICKNESS=TOWEL_THICKNESS,
)
"""The fabrics above under one name, which is what ``rss.materials`` resolves to."""


# =============================================================================
# 4. The entity: build, settle, hold, read
# =============================================================================
# ``EntityCloth`` builds a membrane or a slab, settles it and snapshots the settled state for reset.
# The ``phase`` markers follow ``rss.Entity``: 1 build, 2 model finalized, 4 scene ready, 5 reset.


def write_gripper_friction(
    model: newton.Model,
    *,
    tokens: Sequence[str],
    gripper_friction: float,
    surface_friction: float,
) -> np.ndarray:
    """Give the gripper's shapes a different friction from every other shape.

    Newton combines cloth-against-shape friction as ``sqrt(soft_contact_mu * shape_material_mu)``
    and uses ``soft_contact_mu`` raw for cloth against itself. Raising the shared number until a
    hand holds the sheet also glues its folded layers together, so the pads are raised per shape.

    Args:
        model: The finalized model, before any solver is built and takes its copy of the frictions.
        tokens: Tails of the body labels of the gripper's shapes. Labels, because shape indices
            move in ``collapse_fixed_joints``.
        gripper_friction: ``shape_material_mu`` on those shapes.
        surface_friction: ``shape_material_mu`` on every other shape.

    Returns:
        The shape indices that were treated as the gripper's.
    """
    if model.shape_count == 0 or model.shape_material_mu is None:
        return np.zeros(0, dtype=np.int32)

    labels = list(model.shape_label)
    bodies = model.shape_body.numpy()
    body_labels = list(model.body_label)
    wanted = tuple(t.lower() for t in tokens)

    def is_gripper(shape: int) -> bool:
        body = int(bodies[shape])
        # A shape on the world has no body label, so its own label is matched.
        label = body_labels[body] if body >= 0 else labels[shape]
        tail = str(label).rsplit("/", 1)[-1].lower()
        return any(t in tail for t in wanted)

    mu = model.shape_material_mu.numpy().copy()
    mu[...] = surface_friction
    gripper = np.array([s for s in range(model.shape_count) if is_gripper(s)], dtype=np.int32)
    if len(gripper):
        mu[gripper] = gripper_friction
    model.shape_material_mu.assign(mu)
    return gripper


# ``Selector`` and its helpers are the library's, shared by every kind made of particles.


class EntityCloth(ParticleMixin, Entity):
    """One deformable sheet, built from ``spec`` into world ``world``."""

    spec: ClothSpec
    domain = "cloth"

    def __init__(self, spec: ClothSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global indices of this cloth's particles."""
        self.sets: dict[str, np.ndarray] = {}
        """Named subsets, keyed ``"<cloth>/<set>"``."""
        self.probe: ParticleProbe | None = None
        self.rest: dict[str, np.ndarray] = {}
        """The settled state a reset restores."""
        self.surface_count: int = 0
        """Vertices in the mid-surface: every particle of a membrane, half the particles of a slab.
        :meth:`mid_surface_positions` pairs a slab's faces by it. The model does not hold it."""

    # -------------------------------------------------------------- phase 1
    def surface(self) -> tuple[np.ndarray, np.ndarray]:
        """This cloth's mid-surface as ``(vertices, faces)``, scaled, in its own frame."""
        spec = self.spec
        if spec.mesh == "grid":
            return grid_surface(tuple(float(v) for v in spec.size), tuple(int(v) for v in spec.cells))

        import trimesh

        mesh = trimesh.load(str(resolve_asset(spec.asset)), force="mesh")
        if spec.simplify_faces and len(mesh.faces) > spec.simplify_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=int(spec.simplify_faces))
        # Scaled here, not by the builder: a slab's thickness must not scale with the garment.
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(spec.asset.scale)
        return vertices, np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        material = spec.material
        # ``pose`` is in the cell's work-surface frame, which has no rotation: ``quat`` is as given.
        pos = wp.vec3(*[float(v) for v in self.placed((*spec.pose, *spec.quat))[:3]])
        rot = wp.quat(*[float(v) for v in spec.quat])

        if material.is_slab:
            self.build_slab(builder, pos, rot)
        elif spec.mesh == "grid":
            length, width = (float(v) for v in spec.size)
            cells_x, cells_y = (int(v) for v in spec.cells)
            self.surface_count = (cells_x + 1) * (cells_y + 1)
            # Uniform mass per particle: the sheet's mass over its vertices.
            particle_mass = material.areal_density * (length * width) / self.surface_count
            builder.add_cloth_grid(
                pos=pos,
                rot=rot,
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=cells_x,
                dim_y=cells_y,
                cell_x=length / cells_x,
                cell_y=width / cells_y,
                mass=float(particle_mass),
                tri_ke=material.tri_ke,
                tri_ka=material.tri_ka,
                tri_kd=material.tri_kd,
                edge_ke=material.edge_ke,
                edge_kd=material.edge_kd,
                particle_radius=material.particle_radius,
                label=spec.name,
            )
        else:
            # Through ``surface``, so ``simplify_faces`` holds here too. Already scaled.
            vertices, faces = self.surface()
            self.surface_count = len(vertices)
            builder.add_cloth_mesh(
                pos=pos,
                rot=rot,
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*[float(c) for c in v]) for v in vertices],
                indices=[int(i) for i in faces.reshape(-1)],
                density=material.areal_density,
                tri_ke=material.tri_ke,
                tri_ka=material.tri_ka,
                tri_kd=material.tri_kd,
                edge_ke=material.edge_ke,
                edge_kd=material.edge_kd,
                particle_radius=material.particle_radius,
                label=spec.name,
            )

    def build_slab(self, builder: newton.ModelBuilder, pos, rot) -> None:
        """Build this cloth with a real thickness.

        Kept apart from the membrane branches: ``add_cloth_grid`` gives every particle the same
        mass and the mesh builder lumps mass by area, so a shared path would change every grid.
        """
        material = self.spec.material
        vertices, faces = self.surface()
        self.surface_count = len(vertices)
        slab, tets = prism_slab(vertices, faces, material.thickness)
        builder.add_soft_mesh(
            pos=pos,
            rot=rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*[float(c) for c in v]) for v in slab],
            indices=[int(i) for i in tets.reshape(-1)],
            # Volumetric, so the slab weighs what the fabric weighs at any thickness.
            density=material.bulk_density,
            k_mu=material.core_shear_modulus,
            k_lambda=material.core_lame_lambda,
            k_damp=material.core_damping,
            # The skin: zero by default in ``add_soft_mesh``, and what makes a slab resist bending.
            tri_ke=material.tri_ke,
            tri_ka=material.tri_ka,
            tri_kd=material.tri_kd,
            edge_ke=material.edge_ke,
            edge_kd=material.edge_kd,
            particle_radius=material.default_particle_radius(),
            label=self.spec.name,
        )

    def apply_model_materials(self, model: newton.Model) -> None:
        """Write this sheet's contact parameters, once every entity is bound."""
        self.apply_material(model)

    def host_actor(self) -> HostActor:
        """What a foreign engine needs of this sheet beyond the model arrays.

        Read only when ``cloth`` is routed to a host engine such as SuperDex or Genesis. A membrane
        is a ``shell`` actor. A slab is a ``tet`` actor, its Lamé pair read off ``tet_materials``.
        """
        material = self.spec.material
        thickness = float(material.thickness) if material.thickness > 0 else 2.0 * float(material.particle_radius)
        params: dict = {
            "friction": float(material.friction),
            "particle_radius": float(material.particle_radius),
            "self_contact": bool(self.sim_requirements().self_contact),
        }
        if material.is_slab:
            return HostActor(kind="tet", params={**params, "density": float(material.bulk_density)})
        shell = {**params, "thickness": thickness, "areal_density": float(material.areal_density)}
        # An absent key leaves the island's own modulus in force.
        if material.shell_young is not None:
            shell["young"] = float(material.shell_young)
        if material.shell_poisson is not None:
            shell["poisson"] = float(material.shell_poisson)
        return HostActor(kind="shell", params=shell)

    def apply_material(self, model: newton.Model) -> None:
        """Write the contact parameters onto the finalized model, before the solvers are built."""
        material = self.spec.material
        if material.zero_rest_angle and getattr(model, "edge_indices", None) is not None:
            # This sheet's edges only: another sheet in the model may keep its creases. The range
            # is the template's; a replicated model holds a copy of it in every world, at the
            # particle stride, and every copy is this sheet. Column 2 of an edge row is a vertex
            # on the edge, so it is always a real particle.
            lo, hi = self.range.particles
            count = int(getattr(self.worlds, "num_worlds", 1) or 1)
            stride = int(getattr(self.worlds, "particles_per_world", 0) or 0)
            on_edge = model.edge_indices.numpy().reshape(-1, 4)[:, 2]
            mine = np.zeros(len(on_edge), dtype=bool)
            for world in range(count):
                mine |= (on_edge >= lo + world * stride) & (on_edge < hi + world * stride)
            if mine.any():
                rest = model.edge_rest_angle.numpy()
                rest[mine] = 0.0
                model.edge_rest_angle.assign(rest)
        model.soft_contact_ke = material.contact_ke
        model.soft_contact_kd = material.contact_kd
        model.soft_contact_mu = material.friction

    # -------------------------------------------------------------- phase 2
    def tunables(self, pipeline=None):
        """This sheet's material, as something a panel may tune while the scene runs.

        ``LiveMaterial`` writes a field into the model's material arrays, which VBD reads every
        substep, so a write needs no notification. The spec is the one the sheet was built with.
        """
        from robosimstudio.core.tuning import Tunable
        from robosimstudio.scene.tuning import LiveMaterial, locate_cloth

        del pipeline
        if self.model is None:
            return ()
        live = LiveMaterial(self.model, locate_cloth(self.model, self))
        return (
            Tunable(
                name="material",
                spec=self.spec.material,
                write=live.write_field,
                summary=f"What {self.name} is made of.",
            ),
        )

    def resolve_sets(self, state: newton.State) -> dict[str, np.ndarray]:
        """Resolve the spec's particle selectors against the rest layout.

        Run once on the sheet as built, so a set names the same particles wherever the cloth goes.
        A selector sees positions in the work-surface frame, the frame ``pose`` is stated in.
        """
        if not len(self.particles):
            return {}
        # The rest layout, viewed in the work-surface frame. The particles stay in the world frame.
        rest = np.asarray(state.particle_q.numpy(), dtype=np.float64)[self.particles] - np.asarray(self.frame[:3])
        for set_name, selector in self.spec.particle_sets:
            key = f"{self.spec.name}/{set_name}"
            indices = resolve_selector(selector, rest, self.particles)
            self.sets[key] = indices
            self.add_set_every_world(key, indices)
        return dict(self.sets)

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Settle the sheet, then record the state every reset restores exactly.

        A settle steps the whole scene and a reset restores the last whole-array snapshot, so the
        particle things in a scene share one record on the pipeline: a settle one of them already
        ran is not run again, and each that has snapshotted snapshots again after one that is.
        """
        self.resolve_sets(pipeline.state_0)
        self.join_settle(pipeline, (int(self.spec.settle_frames), int(self.spec.settle_substeps)))

    def settle(self, pipeline: ScenePipeline) -> None:
        """Step the sheet at a fine substep dt, then restore the run's own."""
        spec = self.spec
        if spec.settle_frames <= 0:
            return
        with pipeline.substeps_at(spec.settle_substeps):
            for _ in range(int(spec.settle_frames)):
                pipeline.step()
        # A settle is scene construction, not simulated time.
        pipeline.sim_time = 0.0
        pipeline.step_count = 0

    # -------------------------------------------------------------- phase 5
    # ---------------------------------------------------------------- runtime
    def set_indices(self, name: str) -> np.ndarray:
        """One named set's global particle indices. A bare name is resolved inside this cloth."""
        key = name if "/" in name else f"{self.spec.name}/{name}"
        if key not in self.sets:
            raise KeyError(f"cloth {self.spec.name!r} has no particle set {name!r}; has {sorted(self.sets)}")
        return self.sets[key]

    def mid_surface_positions(self, state: newton.State) -> np.ndarray:
        """Where this cloth is, as one point per mid-surface vertex ``[n, 3]``.

        For a slab, the average of its two faces. Measure geometry on this, never on one face: a
        fold turns the sheet over, which adds a full thickness of error, 5 mm on a bath towel.
        """
        q = np.asarray(state.particle_q.numpy(), dtype=np.float64)[self.particles]
        return mid_surface(q, self.surface_count) if self.spec.material.is_slab else q

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None) -> ParticleProbe:
        """Read this step's particle positions. Once per control step.

        Args:
            state: The state to read.
            step: The control step. A second read in the same step is a no-op.
            readback: The pipeline's :class:`~robosimstudio.core.readback.HostSnapshot` for this
                step. When given, the probe reads it and makes no device-to-host copy of its own.
        """
        if self.probe is None:
            raise RuntimeError(f"cloth {self.spec.name!r} is not bound yet")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe


# -----------------------------------------------------------------------------
# A sheet held from the start
# -----------------------------------------------------------------------------
# A towel already in a hand, a flag on its pole: a patch of the sheet rides a body from the build
# onward. A subclass, because a held sheet contributes a substep kernel, and a kernel written for
# one world is what keeps a scene from being replicated. A free sheet stays free of that.


@dataclass(frozen=True, kw_only=True)
class HeldClothSpec(ClothSpec):
    """A sheet whose ``held`` particle set rides one body, kinematically, from the build onward.

    ``rss.sheet(held_by=...)`` returns one. Name the patch in :attr:`ClothSpec.particle_sets` as
    ``"held"``, and pose the body where the patch lies: each held particle keeps the offset from the
    body it had when the scene was ready, so a body far from the sheet holds it from afar.
    """

    entity: str = "held_cloth"
    held_by: str = ""
    """The holding body, by the last segment of its label: ``"left_link_6"`` is the YAM's left wrist."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.held_by:
            raise ValueError(f"cloth {self.name!r} is held by nothing: name a body in held_by")
        if "held" not in dict(self.particle_sets):
            raise ValueError(f"cloth {self.name!r} is held but declares no 'held' particle set")


@wp.kernel
def _hold_kernel(
    held: wp.array[wp.int32],
    offsets: wp.array[wp.vec3],
    count: wp.array[wp.int32],
    body: int,
    inv_dt: float,
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Put each held particle where the body's frame carries it, at the velocity that took it there."""
    i = wp.tid()
    if i >= count[0]:
        return
    p = held[i]
    goal = wp.transform_point(body_q[body], offsets[i])
    particle_qd[p] = (goal - particle_q[p]) * inv_dt
    particle_q[p] = goal


class EntityHeldCloth(EntityCloth):
    """A sheet whose ``held`` set rides a body: weightless to the solve, pinned after it.

    Three things make the hold. The numbers are from a towel whose edge a YAM wrist held this way
    while a script scribbled the hand about:

    * **The held particles weigh nothing.** Zero mass is Newton's kinematic particle, so VBD leaves
      them to the pin. Left with their mass, the solve fought the pin every substep and wrote
      0.6 m/s of chatter into the held edge.
    * **The pin reads the body every substep**, on the device, after the solve: the patch is where
      the body is now, not where it was at the start of the control step.
    * **What hangs from the body stops touching particles.** The patch rides the hand, so the
      fingers have nothing to do but slice the free cloth beside it, which tore or tangled the
      towel in 2 of 5 runs. Every shape on a body below the holder in the joint tree loses
      ``COLLIDE_PARTICLES``: for a hand, the fingers. The holder's own shapes keep it.
    """

    spec: HeldClothSpec

    def __init__(self, spec: HeldClothSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.holder = -1
        """The holding body's index."""
        self.held = np.zeros(0, dtype=np.int32)
        """Global indices of the held particles."""
        self.freed: list[int] = []
        """The shapes that stopped colliding with particles."""

    def bind(self, model: newton.Model) -> None:
        """Find the holder and the held patch, on the sheet as built, before the solvers exist."""
        super().bind(model)
        tail = self.spec.held_by
        hits = [b for b, label in enumerate(model.body_label) if str(label).rsplit("/", 1)[-1] == tail]
        if len(hits) != 1:
            raise ValueError(f"cloth {self.spec.name!r} is held by {tail!r}, which names {len(hits)} bodies")
        self.holder = hits[0]
        rest = model.particle_q.numpy().astype(np.float64)[self.particles] - np.asarray(self.frame[:3])
        self.held = resolve_selector(dict(self.spec.particle_sets)["held"], rest, self.particles)
        if not len(self.held):
            raise ValueError(f"cloth {self.spec.name!r}: its 'held' set selects no particle")
        device = model.device
        self._held = wp.array(self.held, dtype=wp.int32, device=device)
        self._offsets = wp.zeros(len(self.held), dtype=wp.vec3, device=device)
        self._count = wp.zeros(1, dtype=wp.int32, device=device)

    def apply_model_materials(self, model: newton.Model) -> None:
        """Weightless held particles and particle-blind fingers, before a solver copies either."""
        super().apply_model_materials(model)
        for array in (model.particle_mass, model.particle_inv_mass):
            values = array.numpy()
            values[self.held] = 0.0
            array.assign(values)
        parent = dict(zip(model.joint_child.numpy().tolist(), model.joint_parent.numpy().tolist(), strict=True))

        def carried(body: int) -> bool:
            while body in parent and parent[body] >= 0:
                body = parent[body]
                if body == self.holder:
                    return True
            return False

        bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
        self.freed = [s for s in range(model.shape_count) if flags[s] & bit and carried(int(bodies[s]))]
        flags[self.freed] &= ~bit
        model.shape_flags.assign(flags)

    def contribute_hooks(self, hooks) -> None:
        hooks.add("post_solve", self._hold, name=f"hold:{self.spec.name}")

    def _hold(self, state: newton.State, dt: float) -> None:
        wp.launch(
            _hold_kernel,
            dim=len(self.held),
            inputs=[self._held, self._offsets, self._count, self.holder, 0.0 if not dt else 1.0 / dt, state.body_q],
            outputs=[state.particle_q, state.particle_qd],
            device=self.model.device,
        )

    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Take hold where the patch lies, then settle the sheet hanging from it."""
        from robosimstudio.core.math import quat_to_mat

        state = pipeline.state_0
        pose = state.body_q.numpy()[self.holder].astype(np.float64)
        points = state.particle_q.numpy()[self.held].astype(np.float64)
        self._offsets.assign(((points - pose[:3]) @ quat_to_mat(pose[3:7])).astype(np.float32))
        self._count.assign(np.array([len(self.held)], dtype=np.int32))
        super().on_scene_ready(pipeline)


# =============================================================================
# 5. The constructor: ``sheet``
# =============================================================================
# ``sheet`` returns a ``ClothSpec`` for a rectangular grid. It is ``rss.sheet``.


def sheet(
    *,
    name: str = "sheet",
    size: tuple[float, float] = (0.3, 0.3),
    cells: tuple[int, int] = (30, 30),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = None,
    **kwargs,
) -> ClothSpec:
    """A rectangular sheet of cloth, generated as a grid.

    Cloth needs a particle solver, or the build raises. :data:`~robosimstudio.specs.physics.GENERIC`
    steps it with ``vbd``, and under its default coupling a hand can push a sheet but not pinch one.

    Args:
        name: Key in handles and particle-set names.
        size: Metres, ``(x, y)``.
        cells: Quads along each axis. Cost is quadratic: 30x30 is 961 particles, 100x100 is 10,201.
        at: Where the sheet's origin **corner** goes, in the cell's work-surface frame: ``z = 0.25``
            is 250 mm above the bench. The grid spans ``0`` to ``size``, so subtract half to centre it.
        material: A :class:`~robosimstudio.domains.cloth.ClothMaterialSpec` such as one of
            ``rss.materials``. ``None`` is the spec's default, a plain cloth that was not measured.
        **kwargs: Any other :class:`~robosimstudio.domains.cloth.ClothSpec` field. ``held_by``
            makes it a :class:`~robosimstudio.domains.cloth.HeldClothSpec`: a sheet whose ``held``
            particle set rides that body from the start.
    """
    extra = {} if material is None else {"material": material}
    spec = HeldClothSpec if kwargs.get("held_by") else ClothSpec
    return spec(name=name, size=size, cells=cells, pose=at, **extra, **kwargs)


# =============================================================================
# 6. Measures and calibration
# =============================================================================
# What a goal may ask of a sheet (``fold_score``, ``fold_extent``, ``settle_speed``, ``held``,
# ``cloth_coverage``), each with its calibration, then the bending length a fabric is judged by.


SETTLE_QUANTILE = 0.95
"""Which quantile of the particle speeds counts as the sheet's speed. Not the maximum: in
self-contact some particle is always being pushed out of a contact, so the maximum never settles.
0.95 is also the probe's own default."""


def crease_of(panel: dict[str, Any]) -> tuple[int, float]:
    """A panel's fold line, as ``(axis, origin)``.

    The panel's ``base`` must be the mirror of its ``moving`` through this line, or the score means
    something else: a shirt fold declared otherwise measured 0.44 where its geometry deserved 0.9.
    """
    crease = panel.get("crease") or {}
    return int(crease.get("axis", 1)), float(crease.get("at", 0.0))


def layer_separation(snapshot, cloth: str) -> float:
    """How far apart two stacked layers of this cloth rest [m]: its :func:`contact_thickness`.

    Zero, so nothing is forgiven, when the scene holds no entity for ``cloth``.
    """
    entity = snapshot.extras.get(cloth)
    return contact_thickness(getattr(getattr(entity, "spec", None), "material", None))


def contact_thickness(material) -> float:
    """How thick a sheet of this fabric is, to anything touching it [m].

    Two particle radii on a membrane, plus the geometric thickness on a slab: 7.5 mm for
    ``TERRY_TOWELLING``. Gripper pads closed nearer than this squeeze the particle between them out
    sideways and tear the mesh.
    """
    if material is None:
        return 0.0
    standoff = 2.0 * float(material.particle_radius)
    return float(material.thickness) + standoff if material.is_slab else standoff


def _pairs(snapshot, cloth: str, panel: dict[str, Any]) -> np.ndarray:
    """The mirror correspondence for one crease, built once and cached on the probe.

    Built against the settled baseline: pairs recomputed from the live cloth would report zero
    error for any shape. A reset restores that baseline and never rewrites it, so the cache holds.
    """
    probe = snapshot.particles[cloth]
    axis, origin = crease_of(panel)
    moving, base = f"{cloth}/{panel['moving']}", f"{cloth}/{panel['base']}"
    cache = getattr(probe, "_fold_pairs", None)
    if cache is None:
        cache = {}
        probe._fold_pairs = cache
    key = (moving, base, axis, origin)
    if key not in cache:
        cache[key] = mirror_pairs(probe, moving, base, axis=axis, origin=origin)
    return cache[key]


register_atom(
    AtomSpec(
        name="fold_score",
        summary="How well one crease was made, on SoftGym's ClothFold measure.",
        args=(
            ArgSpec(name="cloth", substrates=("cloth",), reads="handle"),
            ArgSpec(name="panel", kinds=("panel",), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("score", "group_dist", "fixation_dist", "softgym"),
        cost="expensive",
        reads_scene=True,
        provenance=(
            "SoftGym ClothFold (Lin et al., CoRL 2020), transcribed -- the 1.2 drag weight and "
            "the normalisation against the flat start are its own. The layer forgiveness is not "
            "SoftGym's and is reported separately as `softgym`, so the number stays comparable "
            "with everything else that reports one."
        ),
        calibration=AtomCalibration(
            positives=(
                "0.86-0.94 over four towel rollouts with the two arms filtered, three of them "
                "latching success; 0.71-0.93 with a press 3 cm further in."
            ),
            negatives=(
                "A sheet carried only halfway over scores exactly 0.75 by construction -- half "
                "the distance travelled is half the score -- and a shirt whose side fold hinged "
                "the bare sleeve at its root measured 0.17-0.49."
            ),
            gate=0.85,
            bench="a towel folded in half and a shirt folded in three creases, by two arms; the fold metric's unit tests",
            notes=(
                "The gate sits above the halfway-fold's 0.75 deliberately: a threshold below it "
                "would be satisfied by a sheet still standing open like a tent. On a 0.4 m towel "
                "whose halves start 0.22 m apart, 0.85 is a mean placement error of about 33 mm."
            ),
        ),
    )
)


@handler(
    "fold_score",
    kinds=("", "panel"),
    substrates=("cloth", ""),
    summary="The carried panel against where its mirror says it should have landed, less the drag.",
)
def fold_score(snapshot, cloth: str, panel: dict[str, Any]) -> dict[str, Any]:
    """One crease, scored twice: with the layer separation forgiven, and raw.

    The mirror is planar: it asks each carried particle to land at its target's rest height, but a
    folded flap lands on top of the standing half, turned over. With thickness ``h`` and contact
    standoff ``s``, the top face lands ``s`` off its target and the bottom face ``2h + s``: a mean
    of ``h + s``, which is :func:`layer_separation`. It is subtracted from both sides of the ratio
    (from one side only, a flat sheet scores 0.09) and scaled by the panel's ``forgiveness``, for
    a crease whose flap lands on an already-folded stack and so rests higher.
    """
    pairs = _pairs(snapshot, cloth, panel)
    ctx = snapshot.ctx
    moving, base = str(panel["moving"]), str(panel["base"])
    separation = layer_separation(snapshot, cloth) * float(panel.get("forgiveness", 1.0))
    score, group, fixation = cloth_fold_error(ctx, cloth, pairs, moving=moving, base=base, layer_thickness=separation)
    # SoftGym's formula with nothing forgiven, reported beside the gated score.
    raw, _, _ = cloth_fold_error(ctx, cloth, pairs, moving=moving, base=base)
    return {
        "score": float(score),
        "group_dist": float(group),
        "fixation_dist": float(fixation),
        "softgym": float(raw),
        "drag_weight": FOLD_DRAG_WEIGHT,
        # Keys that name their cloth and crease, so three creases on one garment stay apart.
        f"{cloth}/fold_score.{panel.get('name', 'crease')}": float(score),
        # Reported, not gated. The drag tells a missed flap from a dragged standing half.
        f"{cloth}/fold_score_softgym.{panel.get('name', 'crease')}": float(raw),
        f"{cloth}/fixation_dist.{panel.get('name', 'crease')}": float(fixation),
    }


register_atom(
    AtomSpec(
        name="fold_extent",
        summary="How far a sheet's span across its fold axis has closed, as a fraction of settled.",
        args=(
            ArgSpec(name="cloth", substrates=("cloth",), reads="handle"),
            ArgSpec(name="template", kinds=("template",), reads="feature"),
        ),
        directions=("check", "observe"),
        returns=("ratio", "extent"),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here. No published benchmark defines a fold extent for a task shaped like "
            "this one; the quantity is the obvious one and the gate is the part that had to be "
            "argued for."
        ),
        calibration=AtomCalibration(
            positives="A clean single fold halves it; a second fold takes it to about a quarter.",
            negatives="An untouched sheet reads 1.0, and a sheet dragged sideways without stacking also halves it.",
            gate=0.6,
            bench="a towel folded in half by two arms",
            notes=(
                "0.6 rather than 0.5 leaves room for the bead of material standing up along the "
                "crease, which no real fold is without. On its own the measure is worthless -- "
                "a drag halves it too -- which is what fold_score's drag term is for."
            ),
        ),
    )
)


@handler(
    "fold_extent",
    kinds=("", "template"),
    substrates=("cloth", ""),
    summary="The sheet's current span across the template's first crease axis, over its settled span.",
)
def _fold_extent(snapshot, cloth: str, template: dict[str, Any]) -> dict[str, Any]:
    """The span across the axis of the template's first crease, which the gate was calibrated on."""
    steps = template.get("steps") or []
    axis = int((steps[0].get("crease") or {}).get("axis", 1)) if steps else 1
    ratio, extent = cloth_fold(snapshot.ctx, cloth, axis=axis)
    return {
        "ratio": float(ratio),
        "extent": float(extent),
        "axis": axis,
        f"{cloth}/fold_ratio": float(ratio),
        f"{cloth}/fold_extent_m": float(extent),
    }


register_atom(
    AtomSpec(
        name="settle_speed",
        summary="How fast a deformable is still moving [m/s], as a high quantile over its particles.",
        args=(ArgSpec(name="cloth", substrates=("cloth",), reads="handle"),),
        directions=("check", "observe"),
        returns=("speed",),
        unit="m/s",
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here, and distinct from `speed` on purpose: that one reports a body's "
            "velocity magnitude, and a garment has no body. The quantile is the part that makes "
            "it usable -- see SETTLE_QUANTILE."
        ),
        calibration=AtomCalibration(
            positives="A towel that has come to rest on the bench reads well under 0.02 m/s at the 95th percentile.",
            negatives=(
                "A sheet still being carried, or springing back after release, reads an order of "
                "magnitude more. The MAXIMUM particle speed never falls below 0.02 at all, which "
                "is why this is a quantile."
            ),
            gate=0.02,
            bench="a towel folded in half and a shirt folded in three creases, by two arms",
        ),
    )
)


@handler(
    "settle_speed",
    substrates=("cloth",),
    summary="The 95th percentile of the sheet's particle speeds.",
)
def _settle_speed(snapshot, cloth: str) -> dict[str, Any]:
    probe = snapshot.particles[cloth]
    speed = float(probe.settle_speed(cloth, quantile=SETTLE_QUANTILE))
    # Reported, not gated: it tells "settled" from "one particle is stuck in a contact".
    max_speed = float(probe.max_speed(cloth))
    return {
        "speed": speed,
        "max_speed": max_speed,
        f"{cloth}/settle_speed": speed,
        f"{cloth}/max_speed": max_speed,
    }


register_atom(
    AtomSpec(
        name="held",
        summary="Whether a hand still has hold of a declared strip of a deformable.",
        args=(
            ArgSpec(name="cloth", substrates=("cloth",), reads="handle"),
            ArgSpec(name="strip", kinds=("set",), reads="feature", optional=True),
        ),
        directions=("check", "observe"),
        returns=("held", "count"),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "The cloth half of the repo's held_particles predicate, which reports a kinematic "
            "latch where there is one and a pinch radius where there is not. Two rigid fingers "
            "grip a sheet by friction under one-way coupling with full-surface contact; the "
            "latch is for a task that wants the strip held whatever the fingers do."
        ),
        calibration=AtomCalibration(
            positives="A latched carry holds the whole declared strip -- 93 particles on the towel's edge row.",
            negatives=(
                "A top-down jaw on a flat sheet carried 0 of those 93 while the one-way coupler "
                "handed VBD the arm's new pose as its previous one too, so friction saw the pads "
                "stand still. With the motion, a jaw closed with its fingertips at the bench keeps "
                "15 of 15 pinched particles through a 100 mm lift."
            ),
            gate=None,
            bench="a towel folded in half by two arms",
            notes="No gate. Held is a boolean the task negates; there is nothing to threshold.",
        ),
    )
)


@handler(
    "held",
    kinds=("", "set"),
    substrates=("cloth", ""),
    summary="A latch's grabbed set where there is one, particles inside the pinch radius where there is not.",
)
def _held_strip(snapshot, cloth: str, strip: dict[str, Any] | None = None) -> dict[str, Any]:
    """Held over one declared strip, or over the whole sheet when none is named.

    Name the strip: a garment's later folds grip material the first crease never touched.
    """
    name = str((strip or {}).get("name", "")) if strip else ""
    held, count = held_particles(snapshot.ctx, cloth, particle_set=name)
    return {"held": bool(held), "count": float(count), f"{cloth}/held.{name or 'any'}": bool(held)}


@handler(
    "held",
    substrates=("cloth",),
    summary="Held anywhere on the sheet, for a task with one strip and nothing to disambiguate.",
)
def _held_cloth(snapshot, cloth: str) -> dict[str, Any]:
    held, count = held_particles(snapshot.ctx, cloth)
    return {"held": bool(held), "count": float(count), f"{cloth}/held.any": bool(held)}


register_atom(
    AtomSpec(
        name="cloth_coverage",
        summary="How spread out a sheet is, as a fraction of the footprint it settled into.",
        args=(ArgSpec(name="cloth", substrates=("cloth",), reads="handle"),),
        directions=("check", "observe"),
        returns=("coverage",),
        cost="cheap",
        reads_scene=True,
        provenance=(
            "Designed here. 1.0 is 'as spread as it settled'; folding halves it and balling it up "
            "drives it toward 0. Reported by the fold tasks and gated by none of them -- the "
            "flattening task that would gate on it does not exist yet."
        ),
        calibration=AtomCalibration(
            positives="A garment laid out flat reads close to 1.0.",
            negatives="A folded towel reads about half, and a balled-up one a small fraction.",
            gate=None,
            bench="a towel folded in half by two arms",
            notes="No gate on purpose: nobody has calibrated one, and thresholding it uncalibrated "
            "is how a benchmark ends up measuring an arbitrary constant.",
        ),
    )
)


@handler("cloth_coverage", substrates=("cloth",), summary="The probe's occupancy ratio against its settled footprint.")
def _cloth_coverage(snapshot, cloth: str) -> dict[str, Any]:
    spread = float(cloth_coverage(snapshot.ctx, cloth))
    return {"coverage": spread, f"{cloth}/coverage": spread}


# ---------------------------------------------------------------------------
# Containment, height and speed, for a sheet
# ---------------------------------------------------------------------------
# These three atoms are shared by every kind. A body's centre is inside a region or it is not; a
# sheet answers with the share of its particles inside.

from robosimstudio.assetlib.vocabulary.motion import body_height, linear_speed
from robosimstudio.assetlib.vocabulary.regions import containment, region_bounds


@handler(
    "containment",
    kinds=("", "region"),
    substrates=("cloth", ""),
    summary="The share of a sheet's particles inside the box -- what a garment can actually answer.",
)
def _cloth_in_region(points, region: dict, *, origin=(0.0, 0.0, 0.0)) -> dict:
    lo, hi = region_bounds(region, origin)
    return {"fraction": containment(points, lo, hi)}


handler("height", kinds=("",), substrates=("cloth",), summary="Mean world z of a sheet's particles.")(body_height)
handler("speed", kinds=("",), substrates=("cloth",), summary="Linear speed of a sheet, over its mean.")(linear_speed)


# ``covers``, for a sheet over a rigid body and over another sheet.
from robosimstudio.assetlib.vocabulary.coverage import Covers, _covers

for _target in ("rigid", "cloth"):
    Covers.implement(
        kinds=("", ""),
        substrates=("cloth", _target),
        summary=f"A cloth body's footprint over a {_target} one's, from above.",
    )(_covers)


# -----------------------------------------------------------------------------
# Calibration: bending length
# -----------------------------------------------------------------------------
# Fabric is specified by its bending length ``c = (B / (rho_A g))^(1/3)`` [m]: clamp a strip, let
# it overhang and read how far the tip droops (ASTM D1388). ``B`` is the flexural rigidity [N m]
# and ``rho_A`` the areal density [kg/m^2].


GRAVITY = 9.81
"""Gravity the bending-length measurement assumes [m/s^2]."""


@dataclass(frozen=True)
class ReferenceFabric:
    """One row of the real-cloth table: typical thickness [mm] and areal density [g/m^2], and the
    published bending-length band [cm] a measurement is judged against."""

    name: str
    thickness_mm: float
    areal_gsm: float
    bending_cm: tuple[float, float]

    def covers(self, bending_cm: float) -> bool:
        """Whether a measurement falls inside this fabric's band."""
        return self.bending_cm[0] <= bending_cm <= self.bending_cm[1]


REFERENCE_FABRICS = (
    ReferenceFabric("tissue paper", 0.07, 22.0, (1.0, 1.5)),
    ReferenceFabric("cotton jersey (tee)", 0.5, 170.0, (1.0, 1.5)),
    ReferenceFabric("cotton poplin", 0.3, 120.0, (1.7, 2.2)),
    ReferenceFabric("terry bath towel", 5.0, 500.0, (2.0, 3.0)),
    ReferenceFabric("denim", 1.0, 380.0, (2.5, 3.5)),
)
"""Real fabrics to calibrate against. Bending length does not follow thickness: a jersey tee is
seven times thicker than tissue and droops the same, and a bath towel is ten times thicker than
jersey and only twice as stiff."""


def resembles(bending_cm: float) -> tuple[str, ...]:
    """Which reference fabrics a measurement is consistent with. Several, where bands overlap."""
    if not np.isfinite(bending_cm):
        return ()
    return tuple(f.name for f in REFERENCE_FABRICS if f.covers(bending_cm))


# --------------------------------------------------------------- the elastica
def elastica_droop(beta, *, steps: int = 400, shots: int = 80):
    """Tip droop of a heavy cantilever, as a fraction of its length.

    The strip is inextensible, clamped horizontally at ``s = 0`` and free at ``s = L``. With
    ``sigma = s/L``, ``theta`` the angle below horizontal and ``m = M L / B`` the scaled moment, it
    obeys ``dtheta/dsigma = m`` and ``dm/dsigma = beta (1 - sigma) cos(theta)``, with
    ``theta(0) = 0`` at the clamp and ``m(1) = 0`` at the free end. One group,
    ``beta = rho_A g L^3 / B``, sets the shape. Exact at any deflection, where the small-deflection
    formula is already 25% off once the tip has fallen a fifth of the overhang. Solved by shooting
    on ``m(0)``: the free-end moment is monotone in it, so bisection converges.

    Args:
        beta: ``rho_A g L^3 / B``, scalar or array. Zero is a rigid strip.
        steps: RK4 steps along the strip. The error goes as ``steps^-4``: 400 is past double precision.
        shots: Bisection iterations on the clamp moment.

    Returns:
        Tip droop over length in ``[0, 1)``, matching the shape of ``beta``.
    """
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    sigma = np.linspace(0.0, 1.0, steps + 1)[:-1]
    d = 1.0 / steps

    def integrate(m0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """March every strip from clamp to tip; return ``(m(1), droop)``."""
        theta = np.zeros_like(b)
        m = m0.copy()
        z = np.zeros_like(b)
        for s in sigma:
            # k = (dtheta, dm, dz) at four stages. Only the load term depends on sigma explicitly.
            k1 = (m, b * (1.0 - s) * np.cos(theta), np.sin(theta))
            t2, m2 = theta + 0.5 * d * k1[0], m + 0.5 * d * k1[1]
            k2 = (m2, b * (1.0 - s - 0.5 * d) * np.cos(t2), np.sin(t2))
            t3, m3 = theta + 0.5 * d * k2[0], m + 0.5 * d * k2[1]
            k3 = (m3, b * (1.0 - s - 0.5 * d) * np.cos(t3), np.sin(t3))
            t4, m4 = theta + d * k3[0], m + d * k3[1]
            k4 = (m4, b * (1.0 - s - d) * np.cos(t4), np.sin(t4))
            theta = theta + (d / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            m = m + (d / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            z = z + (d / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        return m, -z

    # m(0) is negative and at most the rigid strip's beta/2 in magnitude, so [-beta, 0] brackets it.
    lo, hi = -b, np.zeros_like(b)
    for _ in range(shots):
        mid = 0.5 * (lo + hi)
        under = integrate(mid)[0] < 0.0
        lo = np.where(under, mid, lo)
        hi = np.where(under, hi, mid)
    droop = integrate(0.5 * (lo + hi))[1]
    droop = np.where(b <= 0.0, 0.0, droop)
    return droop if np.ndim(beta) else float(droop[0])


_CURVE: tuple[np.ndarray, np.ndarray] | None = None


BETA_MAX = 100.0
"""The largest ``beta`` the shooting is trusted at, where the table stops. Past it the strip curls
beyond vertical, the free-end moment is no longer monotone in the clamp moment and bisection fails:
droop falls back to 0.28 at ``beta = 300``, and finer integration does not help. At 100 the strip
already droops to 94% of its length."""

DROOP_BAND = (0.02, 0.80)
"""Droop ratios the inversion is well conditioned over. Below the band the reading is mostly
discretisation. Above it, fabrics a factor of two apart in rigidity look alike."""


def droop_curve() -> tuple[np.ndarray, np.ndarray]:
    """The tabulated ``(beta, droop)`` curve, solved once and reused.

    Raises:
        RuntimeError: If the curve is not monotone: the shooting broke down inside the table's range.
    """
    global _CURVE  # noqa: PLW0603
    if _CURVE is None:
        betas = np.logspace(-3.0, np.log10(BETA_MAX), 601)
        droops = elastica_droop(betas)
        if not np.all(np.diff(droops) > 0.0):
            bad = int(np.argmin(np.diff(droops)))
            raise RuntimeError(f"elastica curve is not monotone near beta={betas[bad]:.3g}; lower BETA_MAX")
        _CURVE = (betas, droops)
    return _CURVE


def bending_length_from_droop(droop_ratio: float, length: float) -> float:
    """Invert a measured droop into a bending length [m].

    ``c = L / beta^(1/3)``, with ``beta`` read off :func:`droop_curve`. The areal density cancels.

    Args:
        droop_ratio: Tip droop over the overhang length.
        length: The overhang length ``L`` [m].

    Returns:
        The bending length, or NaN, never a clamped value, when the droop is off the curve's ends.
    """
    betas, droops = droop_curve()
    if droop_ratio <= droops[0] or droop_ratio >= droops[-1]:
        return float("nan")
    beta = float(np.exp(np.interp(droop_ratio, droops, np.log(betas))))
    return float(length) / beta ** (1.0 / 3.0)


# =============================================================================
# 7. Registration
# =============================================================================
# The material kind and its six fabrics, then the kind itself and the entity that builds it. The
# measures of section 6 registered where they are defined.


register_material_kind("cloth", ClothMaterialSpec)


_DRAPE = "the cloth drape bench (ASTM D1388, clamped strip)"

register_material(
    MaterialRecord(
        name="terry_towelling",
        spec=TERRY_TOWELLING,
        notes="Terry bath towelling, 5 mm and 500 g/m^2, as a slab. The towel that two arms fold in half.",
        calibration=CalibrationSpec(
            fabric="terry bath towelling",
            bench=_DRAPE,
            knob="core_shear_modulus",
            bending_length_cm=2.63,
            target_cm=(2.0, 3.0),
            sweep=((3.0e4, 2.35), (1.0e5, 2.63), (3.0e5, 3.02)),
            notes="1e5 sits in the middle of the band rather than on an edge of it.",
        ),
    )
)

register_material(
    MaterialRecord(
        name="shirt_fleece",
        spec=SHIRT_FLEECE,
        notes="Sweatshirt fleece, 3 mm and 350 g/m^2, as a slab. A shirt for folding.",
        calibration=CalibrationSpec(
            fabric="sweatshirt fleece",
            bench=_DRAPE,
            knob="core_shear_modulus",
            notes=(
                "Measured on the bench like the towel's rather than scaled analytically, but the "
                "sweep was not recorded and fleece is not a row of the reference table."
            ),
        ),
    )
)

register_material(
    MaterialRecord(
        name="terry_membrane",
        spec=TERRY_MEMBRANE,
        notes="The same towelling as a zero-thickness membrane. A towel for flinging.",
        calibration=CalibrationSpec(
            fabric="terry bath towelling",
            bench=_DRAPE,
            knob="edge_ke",
            cell=0.010,
            bending_length_cm=2.65,
            target_cm=(2.0, 3.0),
            sweep=((1.0, 1.38), (6.0, 2.57), (6.6, 2.65), (7.2, 2.73), (10.0, 3.02)),
            notes=(
                "Measured at a 10 mm cell; the hinge model's rigidity scales with cell "
                "size, so this number is for this resolution. Dimensional estimates were off by "
                "orders of magnitude -- do not guess it."
            ),
        ),
    )
)

register_material(
    MaterialRecord(
        name="shirt_membrane",
        spec=SHIRT_MEMBRANE,
        notes="The shirt fleece as a membrane, for flinging.",
        calibration=CalibrationSpec(
            fabric="jersey tee",
            bench=_DRAPE,
            knob="edge_ke",
            cell=0.0064,
            bending_length_cm=1.28,
            target_cm=(1.0, 1.5),
            sweep=((0.3, 0.80), (0.7, 1.09), (1.05, 1.28), (1.5, 1.46)),
            notes="Measured at the shirt mesh's own 6.4 mm mean edge.",
        ),
    )
)

register_material(
    MaterialRecord(
        name="shirt_jersey",
        spec=SHIRT_JERSEY,
        notes=(
            "Newton's own example_cloth_franka garment: its parameters converted from "
            "centimetre-gram-second to SI, on unisex_shirt.usd as authored. A shirt for folding."
        ),
        calibration=CalibrationSpec(
            fabric="jersey tee",
            bench=_DRAPE,
            knob="edge_ke",
            cell=0.0117,
            # 0.0 stands for no reading: at 5e-5 the strip collapses (droop/L 1.001) and inverts to nan.
            bending_length_cm=0.0,
            target_cm=(1.0, 1.5),
            # At an 11.7 mm mean edge. The band is reached at edge_ke ~ 1e-3, 20x the shipped value.
            sweep=((5.0e-5, 0.0), (1.0e-3, 1.34), (1.0e-2, 2.92), (3.0e-2, 4.20), (1.0e-1, 5.89)),
            notes=(
                "Ported, not calibrated -- the value is Newton's cloth_franka demo number and is "
                "kept because reproducing that example's cloth is the point. It is stated rather "
                "than corrected: this fabric drapes further than a real tee, which is what makes "
                "the example's shirt settle so completely onto a table. Raise edge_ke to 1e-3 at "
                "this cell for a garment that measures as jersey; the sweep above is the map."
            ),
        ),
    )
)

register_material(
    MaterialRecord(
        name="tape",
        spec=TAPE_MATERIAL,
        notes="Packing tape, at the reference scene's exact converted values.",
        calibration=CalibrationSpec(
            fabric="packing tape",
            bench="a strip of tape bonded to a box, its free tab left to droop",
            knob="edge_ke",
            notes=(
                "Not a drape-bench fabric: fitted so the settled tab reproduces the reference's "
                "droop measured like for like (bond at frame 0, settle 60 frames, statistics over "
                "every tab particle). Reference mean/min/max 0.808/0.774/0.844 m, this scene "
                "0.810/0.774/0.844. An earlier value of 1.0 was fitted to a droop that was really "
                "the strip sliding across the box during a bond-free settle, a regime the reference "
                "never runs since it bonds first."
            ),
        ),
    )
)


DOMAIN = register_domain(
    "cloth",
    state=("particles",),
    reads="cloth",
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A membrane or slab of particles.",
)

# The names a spec asks for with ``entity="cloth"``, and a held sheet with ``entity="held_cloth"``.
register_entity("cloth")(EntityCloth)
register_entity("held_cloth")(EntityHeldCloth)


# Import-only: there is no ``__main__`` here, because running this file would register everything
# a second time. The scene to run is ``00_towel.py``.
