# SPDX-License-Identifier: Apache-2.0
"""Soft bodies in one file: a tetrahedral body, from its spec to its registration.

What makes a volumetric soft body a kind of thing the library can build, hold and read, in order:

1. **What this file is**, and how it is found.
2. **The spec**: ``SoftMaterialSpec`` and ``SoftSpec``, a sponge as data.
3. **Materials**: the two foams shipped here.
4. **The entity**: ``EntitySoft`` builds the tetrahedra, generated or read off a file by ``read_tets``.
5. **The constructor**: ``sponge``.
6. **Readings**: none of its own. What a soft body is measured as.
7. **Registration**: ``register_domain``, ``register_entity`` and the material records.

This file is what ``rss.sponge`` builds, imported as ``robosimstudio.domains.soft``.
Run ``00_sponge.py`` beside it first. Copy this file to write a kind of your own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib.materials import MaterialRecord, register_material, register_material_kind
from robosimstudio.core.math import lame_from_young_poisson, young_poisson_from_lame
from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.core.render.finish import Finish
from robosimstudio.domains.capabilities import ParticleMixin
from robosimstudio.domains.capabilities import host_bridge as _locate_bridge
from robosimstudio.domains.selection import resolve_selector
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.asset import AssetSpec
from robosimstudio.specs.base import Quat, Vec3
from robosimstudio.specs.params import LIVE, REBUILD, ui
from robosimstudio.specs.physics import HostActor, SimRequirements, register_domain

if TYPE_CHECKING:
    from robosimstudio.core.pipeline import ScenePipeline

__all__ = [
    "FIRM_FOAM",
    "FIRM_FOAM_MATERIAL",
    "SOFT_FOAM",
    "SOFT_FOAM_MATERIAL",
    "EntitySoft",
    "EntitySoftSuperdex",
    "SoftMaterialSpec",
    "SoftSpec",
    "lame_from_young_poisson",
    "read_tets",
    "sponge",
    "young_poisson_from_lame",
]


# =============================================================================
# 1. What this file is, and how it is found
# =============================================================================
# Importing this file registers soft bodies. The library finds it on the search path of
# ``robosimstudio.domains``, as it does a folder of yours named in ``ROBOSIMSTUDIO_DOMAIN_ROOTS``.
# Import it by that name, never by file path, or it registers twice. It imports no engine, no cloth.


# =============================================================================
# 2. The spec: what a soft body is, as data
# =============================================================================
# ``SoftMaterialSpec`` is bulk matter and ``SoftSpec`` a body made of it. The entity writes the
# Lamé pair into the model and the engine reads it back, so the model alone says what the body is.


@dataclass(frozen=True, kw_only=True)
class SoftMaterialSpec:
    """Bulk constitutive and contact parameters for a tetrahedral body, in SI.

    Elasticity is stated as Young's modulus and Poisson ratio and converted to the Lamé pair at
    build. Every field but :attr:`particle_radius` is ``REBUILD``: SuperDex copies the material when
    it creates the actor, so a change lands on the next build, not the next substep.

    Attributes:
        young: Young's modulus [Pa]: 1e4-1e5 for open-cell foam, 1e5-1e6 for silicone, rigid above 1e7.
        poisson: Poisson ratio. Open-cell foam crushes into its own pores: 0.0-0.3, not rubber's 0.49.
        density: Bulk density [kg/m^3]. Kitchen sponge foam is 30-100.
        mass_damping: SuperDex ``mass_damping_coefficient`` [1/s], a drag ``alpha * M * v``. Stops
            rigid-mode ringing, and damps free fall too.
        stiffness_damping: SuperDex ``stiffness_damping_coefficient`` [s], a force ``beta * K * v``.
            Stops a released sponge wobbling. 1e-3 to 1e-2 reads as foam.
        friction: Coulomb friction against colliders. SuperDex takes the geometric mean of the two
            sides, so zero here or on the gripper lets the sponge slip out of any pinch.
        contact_penalty: SuperDex contact ``penalty_coefficient`` [Pa/m]. The engine's 1e9 suits
            bodies of density 1000, and a 60 kg/m^3 foam chatters under it.
        contact_padding: SuperDex ``penalty_threshold_extra_padding`` [m], an anti-tunnelling margin
            for fast colliders. It applies because a soft actor has no collider of its own.
        particle_radius: Newton-side particle radius [m], for display and probes only.
    """

    via_model: ClassVar[tuple[str, ...]] = ("young", "poisson", "particle_radius")
    """Fields that reach the engine through the model, not :meth:`actor_params`:
    the Lame pair in ``model.tet_materials``, and a display-only radius."""

    young: float = field(
        default=2.0e4,
        metadata=ui(tier=REBUILD, unit="Pa", log=True, range=(1.0e3, 1.0e7), group="elasticity"),
    )
    poisson: float = field(
        default=0.25,
        metadata=ui(tier=REBUILD, range=(0.0, 0.49), group="elasticity"),
    )
    density: float = field(
        default=60.0,
        metadata=ui(tier=REBUILD, unit="kg/m^3", log=True, range=(10.0, 2000.0), group="mass"),
    )
    mass_damping: float = field(
        default=0.0,
        metadata=ui(tier=REBUILD, unit="1/s", range=(0.0, 10.0), group="damping"),
    )
    stiffness_damping: float = field(
        default=5.0e-3,
        metadata=ui(tier=REBUILD, unit="s", log=True, range=(1.0e-4, 1.0e-1), group="damping"),
    )
    friction: float = field(
        default=0.9,
        metadata=ui(
            tier=REBUILD,
            range=(0.0, 2.0),
            group="contact",
            doc="Geometric-mean combined with the collider's; zero on either side is zero, full stop.",
        ),
    )
    contact_penalty: float = field(
        default=8.0e7,
        metadata=ui(
            tier=REBUILD,
            unit="Pa/m",
            log=True,
            range=(1.0e5, 1.0e9),
            group="contact",
            doc="At 8e7 a 3.7 N fingertip press sits within about a millimetre of the collider; softer values read as penetration on camera.",
        ),
    )
    contact_padding: float = field(
        default=2.0e-3,
        metadata=ui(tier=REBUILD, unit="m", range=(0.0, 0.01), group="contact"),
    )
    particle_radius: float = field(
        default=0.002,
        metadata=ui(tier=LIVE, unit="m", range=(0.0002, 0.01), group="contact"),
    )
    collider: str = field(
        default="auto",
        metadata=ui(
            tier=REBUILD,
            choices=("auto", "sdf", "none"),
            group="contact",
            doc="What OTHER deformables can touch: 'none' is the engine's own default and nothing can; 'auto' gives an SDF when the body shares its scene with another deformable, and none on a rigid core, which cannot have one.",
        ),
    )
    contact_threshold: float | None = field(
        default=None,
        metadata=ui(
            tier=REBUILD,
            unit="m",
            range=(0.0, 0.002),
            group="contact",
            doc="Penalty onset distance when this body is the collider; None is the engine's 1 mm.",
        ),
    )
    contact_smoothing: float | None = field(
        default=None,
        metadata=ui(tier=REBUILD, unit="m", range=(0.0, 0.005), group="contact"),
    )
    friction_falloff: float | None = field(
        default=None,
        metadata=ui(
            tier=REBUILD,
            unit="m/s",
            log=True,
            range=(1.0e-4, 1.0e-1),
            group="contact",
            doc="Below this slip speed friction is proportional to speed: no static friction. None is the engine's 10 mm/s.",
        ),
    )
    boundary_samples: str | None = field(
        default=None,
        metadata=ui(
            tier=REBUILD,
            choices=(None, "P1Q1", "P1Q3", "P1Q6"),
            group="contact",
            doc="Contact quadrature per surface triangle; a small feature (a latch tooth) needs more than the default three.",
        ),
    )
    sdf_resolution: float | None = field(
        default=None,
        metadata=ui(
            tier=REBUILD,
            unit="m",
            log=True,
            range=(5.0e-5, 5.0e-3),
            group="contact",
            doc="Voxel of the collider SDF; None lets the engine pick a quarter of the mean edge.",
        ),
    )

    def __post_init__(self) -> None:
        lame_from_young_poisson(self.young, self.poisson)  # validates both
        if self.collider not in ("auto", "sdf", "none"):
            raise ValueError(f"collider must be 'auto', 'sdf' or 'none', got {self.collider!r}")
        if not self.density > 0.0:
            raise ValueError(f"density must be > 0 kg/m^3, got {self.density}")

    @property
    def is_slab(self) -> bool:
        """Always true, so the render path adds no display thickness: the particles carry the volume."""
        return True

    @property
    def lame(self) -> tuple[float, float]:
        """``(k_mu, k_lambda)`` [Pa], what ``tet_materials`` stores."""
        return lame_from_young_poisson(self.young, self.poisson)

    def actor_params(self) -> dict[str, float]:
        """What the SuperDex bridge needs beyond the model's own arrays, as JSON-able scalars.

        Elasticity is absent on purpose: the bridge reads ``(mu, lambda)`` off ``model.tet_materials``.
        """
        params: dict[str, Any] = {
            "density": float(self.density),
            "mass_damping": float(self.mass_damping),
            "stiffness_damping": float(self.stiffness_damping),
            "friction": float(self.friction),
            "contact_penalty": float(self.contact_penalty),
            "contact_padding": float(self.contact_padding),
            "collider": str(self.collider),
        }
        # ``None`` means the engine's own value: an absent key tells the bridge to leave it alone.
        for key in ("contact_threshold", "contact_smoothing", "friction_falloff", "sdf_resolution"):
            value = getattr(self, key)
            if value is not None:
                params[key] = float(value)
        if self.boundary_samples:
            params["boundary_samples"] = str(self.boundary_samples)
        return params


@dataclass(frozen=True, kw_only=True)
class SoftSpec:
    """A volumetric soft body: a generated lattice, or a tetrahedral mesh read from a file.

    Attributes:
        name: Key in handles, probes and particle-set names.
        size: Extents ``(x, y, z)`` [m] of the lattice.
        cells: Lattice cells per axis, five tetrahedra each. ``(10, 7, 4)`` is 440 particles and
            1400 tets, which SuperDex steps in a few milliseconds. Doubling every axis costs 8x.
        pose: Where the **bottom-face centre** goes [m], in the cell's work-surface frame. A mesh
            from a file is placed by its own origin: see :attr:`asset`.
        quat: Orientation, xyzw.
        material: Bulk constitutive and contact parameters.
        particle_sets: ``(name, selector)`` pairs over the rest positions.
        settle_frames: Frames stepped before the settled state is recorded for reset. A few suffice.
        settle_substeps: Newton substeps per settle frame, or 0 for the run's own. The soft solver
            steps once per control step either way.
        color: Display RGB in [0, 1].
        opacity: Display alpha in (0, 1].
        finish: Finish under the path-traced backend, or ``None`` for matte foam at :attr:`opacity`.
    """

    sim: SimRequirements | None = None
    """What this body needs of physics, or ``None`` for the default of its kind."""

    stage: ClassVar[str] = "deformables"
    entity: str = "soft"
    name: str = "sponge"
    size: Vec3 = (0.10, 0.07, 0.035)
    cells: tuple[int, int, int] = (10, 7, 4)
    pose: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    asset: AssetSpec | None = None
    """A tetrahedral mesh file, or ``None`` for the lattice :attr:`size` and :attr:`cells` describe.

    ``.npz`` with ``vertices`` and ``tets``, or anything ``meshio`` reads that holds ``tetra``
    cells. Its ``scale`` applies to the vertices, ``size`` and ``cells`` are ignored, and
    :attr:`pose` places the mesh's **own origin**, not a bottom-face centre."""
    material: SoftMaterialSpec = field(default_factory=SoftMaterialSpec)
    particle_sets: tuple[tuple[str, Callable[[Any], Any]], ...] = ()
    anchored: Callable[[Any], Any] | None = None
    """A selector over the rest positions (work-surface frame) naming nodes held **permanently**
    where they were built, such as a clip's base on its rail. ``None`` leaves the body free. Unlike
    a pin, the hold survives a reset."""
    rigid: Callable[[Any], Any] | None = None
    """A selector over the rest positions (work-surface frame) naming nodes that move as **one
    rigid body** while the rest of the mesh flexes about it: a plug's body and its cantilever latch.
    ``None`` leaves the whole mesh deformable.

    A stiff part needs one to move at all. SuperDex solves in single precision, and a free 2.6 g
    plug at 1e9 Pa, its rigid modes ~1e8 below its elastic ones, hung in mid-air (0.00 mm of fall
    in 1.5 s) while a gripper carried only the patch it pressed. On SuperDex such a body probes
    contact but is no collider: it touches another deformable only where its own surface samples
    meet that one's SDF, and one without an SDF passes through it (measured on a 30 mm block). So
    ``collider="sdf"`` is refused at build and ``"auto"`` resolves to ``"none"``, with a warning
    naming each deformable that passes through."""
    settle_frames: int = 30
    settle_substeps: int = 0
    color: Vec3 = (0.95, 0.80, 0.25)
    opacity: float = 1.0
    finish: Finish | None = None

    def __post_init__(self) -> None:
        if any(not float(v) > 0.0 for v in self.size):
            raise ValueError(f"soft body {self.name!r}: size must be positive, got {self.size}")
        if any(int(v) < 1 for v in self.cells):
            raise ValueError(f"soft body {self.name!r}: cells must be >= 1 per axis, got {self.cells}")
        if not 0.0 < self.opacity <= 1.0:
            raise ValueError(f"soft body {self.name!r}: opacity must be in (0, 1], got {self.opacity}")

    @property
    def particle_count(self) -> int:
        """Particles the lattice will allocate: ``prod(cells + 1)``."""
        nx, ny, nz = (int(v) for v in self.cells)
        return (nx + 1) * (ny + 1) * (nz + 1)


# =============================================================================
# 3. Materials: the two foams
# =============================================================================
# Two foams. Section 7 registers them under the names the material registry holds.


SOFT_FOAM = SoftMaterialSpec()
"""A yielding open-cell foam, like a kitchen sponge. Published ranges for open-cell polyurethane
(E ~ 2e4 Pa, nu ~ 0.25, 60 kg/m^3), not a measurement made here."""

FIRM_FOAM = SoftMaterialSpec(young=8.0e4, poisson=0.3, density=110.0, stiffness_damping=8.0e-3)
"""A denser upholstery-grade foam: holds its corners, still visibly squashes."""


# =============================================================================
# 4. The entity: build, settle, hold, read
# =============================================================================
# The particles and tetrahedra go into the ordinary Newton ``ModelBuilder``, so probes, renderers
# and measures read a sponge like any other particles. They are then flagged inactive to Newton's
# solvers, and the ``superdex`` backend writes ``particle_q`` and ``particle_qd`` each control step.


# -----------------------------------------------------------------------------
# Tetrahedra off a file
# -----------------------------------------------------------------------------
# ``.npz`` with ``vertices`` ``(n, 3)`` and ``tets`` ``(m, 4)``, or anything ``meshio`` reads: gmsh
# ``.msh``, VTK, Abaqus, Tetgen. A file with no ``tetra`` cells is refused by name.


def read_tets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (n, 3) float64, tets (m, 4) int64)`` from ``path``."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as z:
            missing = [k for k in ("vertices", "tets") if k not in z]
            if missing:
                raise ValueError(f"{path}: an .npz tet mesh needs 'vertices' and 'tets'; missing {missing}")
            vertices, tets = np.asarray(z["vertices"], dtype=np.float64), np.asarray(z["tets"], dtype=np.int64)
    else:
        try:
            import meshio
        except ImportError as exc:
            raise ImportError(f"{path}: reading a {path.suffix} tet mesh needs meshio (`uv add meshio`)") from exc
        mesh = meshio.read(path)
        blocks = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type == "tetra"]
        if not blocks:
            kinds = sorted({c.type for c in mesh.cells})
            raise ValueError(f"{path}: no 'tetra' cells (it holds {kinds}); a volumetric body needs tetrahedra")
        vertices, tets = np.asarray(mesh.points, dtype=np.float64), np.concatenate(blocks, axis=0)
    vertices = vertices.reshape(-1, 3)
    tets = tets.reshape(-1, 4)
    if len(vertices) < 4 or len(tets) < 1:
        raise ValueError(f"{path}: {len(vertices)} vertices and {len(tets)} tets is not a volume")
    if tets.min() < 0 or tets.max() >= len(vertices):
        raise ValueError(f"{path}: tet indices run {tets.min()}..{tets.max()} over {len(vertices)} vertices")
    return vertices, tets


# -----------------------------------------------------------------------------
# The entity
# -----------------------------------------------------------------------------


class EntitySoft(ParticleMixin, Entity):
    """One volumetric soft body, built from ``spec`` into world ``world``."""

    spec: SoftSpec
    domain = "soft"
    hosted: ClassVar[bool] = False
    """Whether a host engine integrates this body, so its particles are built inactive to Newton.

    Decided by which class the compiler picks for the arrangement, the only thing known at build:
    the solvers copy the flags when they are made, before scene-ready can find a bridge."""

    def __init__(self, spec: SoftSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global indices of this body's particles."""
        self.sets: dict[str, np.ndarray] = {}
        """Named subsets, keyed ``"<body>/<set>"``."""
        self.probe: ParticleProbe | None = None
        self.rest: dict[str, np.ndarray] = {}
        """The settled state a reset restores."""
        self.bridge = None
        """The SuperDex bridge simulating this body, found at scene-ready. Stays ``None`` when the
        arrangement puts ``soft`` on another solver."""

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        material = spec.material
        placed = self.placed((*spec.pose, *spec.quat))
        rot = wp.quat(*[float(v) for v in placed[3:7]])
        # ``pose`` is the bottom-face centre and ``add_soft_grid`` grows from a corner, so shift by
        # the half-extents in the body's own frame.
        from robosimstudio.core.math import transform_matrix

        sx, sy, _ = (float(v) for v in spec.size)
        r = np.asarray(transform_matrix(np.asarray(placed, dtype=np.float64)))[:3, :3]
        corner = np.asarray(placed[:3], dtype=np.float64) + r @ np.array([-sx / 2.0, -sy / 2.0, 0.0])
        k_mu, k_lambda = material.lame
        start = int(builder.particle_count)
        if spec.asset is not None:
            # A file's tetrahedra, in the file's own frame, placed as a whole.
            from robosimstudio.paths import resolve_asset

            vertices, tets = read_tets(resolve_asset(spec.asset))
            vertices = vertices * float(spec.asset.scale)
            builder.add_soft_mesh(
                pos=wp.vec3(*[float(v) for v in placed[:3]]),
                rot=rot,
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*[float(v) for v in row]) for row in vertices],
                indices=[int(i) for i in tets.ravel()],
                density=float(material.density),
                k_mu=float(k_mu),
                k_lambda=float(k_lambda),
                k_damp=float(material.stiffness_damping * k_mu),
                particle_radius=float(material.particle_radius),
                label=spec.name,
            )
        else:
            self._add_grid(builder, spec, material, corner, rot, k_mu, k_lambda)
        # Under a host engine, inactive to every Newton solver and to soft-contact generation, whose
        # kernels gate on this flag. VBD's velocity write-back and its self-contact triangle detector
        # do not, and are handled where the backend ticks. A mesh from a file must be flagged too, or
        # a whole-model VBD integrates it alongside SuperDex. On a Newton solver the body stays active:
        # VBD skips an inactive particle, and the body would never move.
        if self.hosted:
            flags = builder.particle_flags
            for index in range(start, int(builder.particle_count)):
                flags[index] = int(flags[index]) & ~int(newton.ParticleFlags.ACTIVE)
        self._anchor_nodes = self._resolve_nodes(builder, start, "anchored")
        self._rigid_nodes = self._resolve_nodes(builder, start, "rigid")
        if len(self._anchor_nodes) and len(self._rigid_nodes):
            raise ValueError(f"soft body {self.spec.name!r}: `anchored` and `rigid` together are not supported")

    def _resolve_nodes(self, builder: newton.ModelBuilder, start: int, field: str) -> np.ndarray:
        """Local indices of the nodes a :class:`SoftSpec` selector (``anchored``, ``rigid``) names.

        A mask of the wrong length raises, as for ``particle_sets``. It never anchors a shifted set.
        """
        selector = getattr(self.spec, field)
        if selector is None:
            return np.zeros(0, dtype=np.int64)
        rest = np.asarray([tuple(p) for p in builder.particle_q[start:]], dtype=np.float64) - np.asarray(self.frame[:3])
        try:
            local = resolve_selector(selector, rest, np.arange(len(rest))).astype(np.int64)
        except (ValueError, IndexError) as error:
            raise ValueError(f"soft body {self.spec.name!r}: `{field}` -- {error}") from error
        if not len(local):
            raise ValueError(f"soft body {self.spec.name!r}: `{field}` selected no nodes")
        return local.astype(np.int64)

    def _add_grid(self, builder, spec, material, corner, rot, k_mu, k_lambda) -> None:
        """The generated lattice: :attr:`SoftSpec.size` cut into :attr:`SoftSpec.cells`."""
        cells_x, cells_y, cells_z = (int(v) for v in spec.cells)
        sx, sy, sz = (float(v) for v in spec.size)
        builder.add_soft_grid(
            pos=wp.vec3(*[float(v) for v in corner]),
            rot=rot,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cells_x,
            dim_y=cells_y,
            dim_z=cells_z,
            cell_x=sx / cells_x,
            cell_y=sy / cells_y,
            cell_z=sz / cells_z,
            density=float(material.density),
            k_mu=float(k_mu),
            k_lambda=float(k_lambda),
            # A Kelvin-Voigt viscosity [Pa*s], for the model's record. SuperDex damps this body
            # from ``stiffness_damping`` itself.
            k_damp=float(material.stiffness_damping * k_mu),
            particle_radius=float(material.particle_radius),
            label=spec.name,
        )

    def host_actor(self) -> HostActor:
        """The JSON-able remainder a host engine needs beyond the model arrays, elasticity excluded."""
        params = dict(self.spec.material.actor_params())
        anchors = getattr(self, "_anchor_nodes", None)
        if anchors is not None and len(anchors):
            params["anchors"] = [int(i) for i in anchors]
        rigid = getattr(self, "_rigid_nodes", None)
        if rigid is not None and len(rigid):
            params["rigid_core"] = [int(i) for i in rigid]
        return HostActor(kind="tet", params=params)

    # -------------------------------------------------------------- phase 2
    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Settle the body, then record what a reset returns to.

        A settle steps the whole scene and a reset restores the last whole-array snapshot, so the
        particle things in a scene share one record on the pipeline: a settle one of them already
        ran is not run again, and each that has snapshotted snapshots again after one that is.
        """
        self.bridge = _locate_bridge(pipeline, self.name)
        self.resolve_sets(pipeline.state_0)
        self.join_settle(pipeline, (int(self.spec.settle_frames), int(self.spec.settle_substeps)))

    def settle(self, pipeline: ScenePipeline) -> None:
        """Let the body come to rest. Where it lands is the start state.

        Plain ``pipeline.step()`` advances the body too: the soft backend ticks inside it.
        """
        spec = self.spec
        if spec.settle_frames <= 0:
            return
        if spec.settle_substeps > 0:
            with pipeline.substeps_at(int(spec.settle_substeps)):
                for _ in range(int(spec.settle_frames)):
                    pipeline.step()
        else:
            for _ in range(int(spec.settle_frames)):
                pipeline.step()
        # A settle is scene construction, not simulated time.
        pipeline.sim_time = 0.0
        pipeline.step_count = 0

    # -------------------------------------------------------------- phase 5
    # ---------------------------------------------------------------- runtime
    def resolve_sets(self, state: newton.State) -> dict[str, np.ndarray]:
        """Resolve the spec's particle selectors against the rest layout, in the work-surface frame."""
        if not len(self.particles):
            return {}
        rest = np.asarray(state.particle_q.numpy(), dtype=np.float64)[self.particles] - np.asarray(self.frame[:3])
        for set_name, selector in self.spec.particle_sets:
            key = f"{self.spec.name}/{set_name}"
            indices = resolve_selector(selector, rest, self.particles)
            self.sets[key] = indices
            self.add_set_every_world(key, indices)
        return dict(self.sets)

    def set_indices(self, name: str) -> np.ndarray:
        """One named set's global particle indices, resolved inside this body."""
        key = name if "/" in name else f"{self.spec.name}/{name}"
        if key not in self.sets:
            raise KeyError(f"soft body {self.spec.name!r} has no particle set {name!r}; has {sorted(self.sets)}")
        return self.sets[key]

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None) -> ParticleProbe:
        """Read this step's particle positions, once per control step."""
        if self.probe is None:
            raise RuntimeError(f"soft body {self.spec.name!r} is not bound yet")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe


# =============================================================================
# 5. The constructor: ``sponge``
# =============================================================================
# ``sponge`` returns a ``SoftSpec`` for a rectangular lattice. It is ``rss.sponge``.


def sponge(
    *,
    name: str = "sponge",
    size: Vec3 = (0.10, 0.07, 0.035),
    cells: tuple[int, int, int] = (10, 7, 4),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = None,
    **kwargs,
) -> SoftSpec:
    """A rectangular sponge, generated as a lattice.

    Soft bodies run on a foreign engine: :data:`~robosimstudio.specs.physics.GENERIC` gives them to
    ``superdex``, and a build without the ``[superdex]`` extra raises and names it.

    Args:
        name: Key in handles, probes and particle-set names.
        size: Metres, ``(x, y, z)``.
        cells: Lattice cells per axis, five tetrahedra each. The default is 440 particles.
        at: Where the **bottom-face centre** goes, in the cell's work-surface frame: ``at=(0, 0, 0)``
            stands the sponge on the middle of the bench.
        material: A :class:`~robosimstudio.domains.soft.SoftMaterialSpec` such as ``SOFT_FOAM`` or
            ``FIRM_FOAM``. ``None`` is the spec's default, the soft one.
        **kwargs: Any other :class:`~robosimstudio.domains.soft.SoftSpec` field.
    """
    extra = {} if material is None else {"material": material}
    return SoftSpec(name=name, size=size, cells=cells, pose=at, **extra, **kwargs)


# =============================================================================
# 6. Readings
# =============================================================================
# None of its own. ``reads=("volume", "cloth")`` in section 7 says how a soft body is measured: as
# a volume where a measure implements that, and otherwise as a cloud of particles, as cloth is.


# =============================================================================
# 7. Registration
# =============================================================================
# The kind, the entity that builds it, then the materials.


DOMAIN = register_domain(
    "soft",
    state=("particles",),
    reads=("volume", "cloth"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A tetrahedral soft body.",
)

register_entity("soft")(EntitySoft)


class EntitySoftSuperdex(EntitySoft):
    """:class:`EntitySoft` for SuperDex, which integrates the body itself: Newton must leave it alone."""

    hosted = True


register_entity("soft", engine="superdex")(EntitySoftSuperdex)


# -----------------------------------------------------------------------------
# The material records
# -----------------------------------------------------------------------------
# Neither record carries a ``CalibrationSpec``: the foams were not measured here.


register_material_kind("soft", SoftMaterialSpec)

SOFT_FOAM_MATERIAL = register_material(
    MaterialRecord(
        name="soft_foam",
        spec=SOFT_FOAM,
        notes="A yielding open-cell foam -- the kitchen-sponge feel. E ~ 2e4 Pa, nu ~ 0.25, 60 kg/m^3.",
    )
)

FIRM_FOAM_MATERIAL = register_material(
    MaterialRecord(
        name="firm_foam",
        spec=FIRM_FOAM,
        notes="A denser upholstery-grade foam: holds its corners, still visibly squashes.",
    )
)


# Import-only: there is no ``__main__`` here, because running this file would register everything
# a second time. The scene to run is ``00_sponge.py``.
