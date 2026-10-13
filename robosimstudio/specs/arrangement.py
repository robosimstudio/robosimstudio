# SPDX-License-Identifier: Apache-2.0
"""What a physics *arrangement* is, as data: backends by domain, a coupler, a collision policy -- and the one the library ships.

A document stages physics by a versioned **name** and never by a number, and
what the name stands for is an :class:`ArrangementSpec`: rates, a
:class:`BackendSpec` per domain, a :class:`CouplerSpec`, a :class:`CollisionSpec`.
Built with :func:`backend` and :func:`coupler`; :data:`GENERIC` is what a scene
gets when nobody has staged one, and :data:`MUJOCO_500HZ` is the one a hand can
lift a rigid body in.

One of the four modules :mod:`~robosimstudio.specs.physics` was split into. It
reads the registries for one helper -- the parameter freezing a registered
backend's defaults and a stated backend's kwargs must agree on -- and names no
solver: a backend here is an alias or a dotted path, and resolving either is
the composer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .params import REBUILD, RECAPTURE, ui
from .registries import _frozen_params

__all__ = [
    "GENERIC",
    "MUJOCO_500HZ",
    "ArrangementSpec",
    "BackendSpec",
    "CollisionSpec",
    "CouplerSpec",
    "backend",
    "coupler",
]


# ---------------------------------------------------------------------------
# The data: a backend, a coupler, a collision policy, an arrangement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BackendSpec:
    """One solver, as an arrangement states it: a name and its constructor kwargs.

    Built with :func:`backend`. The parameters are passed to the solver's
    constructor verbatim (``cls(model=view, **params)``), which is what makes
    this open: nothing here knows what ``rigid_avbd_beta`` means, only that
    ``"vbd"`` takes it.

    Attributes:
        solver: A registered alias or a dotted path.
        params: Constructor keyword arguments, sorted by name.
        substeps: How many sub-steps this island takes per coupled step --
            :attr:`newton.solvers.SolverCoupled.Entry.substeps`. A stiff rod
            may run at twice the rate of the arm beside it.
    """

    solver: str
    params: tuple[tuple[str, Any], ...] = ()
    substeps: int = field(default=1, metadata=ui(tier=REBUILD, range=(1.0, 16.0), step=1.0, group="rates"))

    def __post_init__(self) -> None:
        if not self.solver:
            raise ValueError("a backend needs a solver alias or dotted path")
        if int(self.substeps) < 1:
            raise ValueError(f"backend {self.solver!r}: substeps must be >= 1, got {self.substeps}")
        object.__setattr__(self, "params", _frozen_params(self.params))
        names = [k for k, _ in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"backend {self.solver!r}: a parameter is stated twice: {names}")

    def kwargs(self) -> dict[str, Any]:
        """The constructor keyword arguments, as a fresh dict."""
        return dict(self.params)

    def get(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)

    def with_(self, **params: Any) -> BackendSpec:
        """A copy with some parameters changed or added."""
        merged = {**dict(self.params), **params}
        return replace(self, params=_frozen_params(merged))

    def with_substeps(self, substeps: int) -> BackendSpec:
        return replace(self, substeps=int(substeps))


def backend(kind: str, /, *, substeps: int = 1, **params: Any) -> BackendSpec:
    """A :class:`BackendSpec`: ``backend("vbd", iterations=40, rigid_avbd_beta=1e2)``.

    The alias is positional-only, so a solver's own ``solver=`` keyword --
    MuJoCo's constraint solver, ``backend("mujoco", solver="newton")`` --
    goes to the constructor like every other parameter.
    """
    return BackendSpec(solver=kind, params=_frozen_params(params), substeps=int(substeps))


@dataclass(frozen=True, kw_only=True)
class CouplerSpec:
    """How islands couple: a registered kind and its parameters.

    Attributes:
        method: ``"proxy"``, ``"admm"``, or a method a task registered.
        params: The coupler's own parameters, sorted by name. For ``"proxy"``:
            ``mode`` (``"staggered"`` | ``"lagged"``), ``iterations``,
            ``mass_scale``, ``collide_interval``, ``contact_matching``,
            ``proxy_relaxation``, ``one_way_mass_scale``; for ``"admm"``: the
            fields of :class:`newton.solvers.SolverCoupledADMM.Config`.
    """

    method: str = "proxy"
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("a coupler needs a kind")
        object.__setattr__(self, "params", _frozen_params(self.params))

    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)

    def get(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)

    def with_(self, **params: Any) -> CouplerSpec:
        return replace(self, params=_frozen_params({**dict(self.params), **params}))


def coupler(method: str = "proxy", /, **params: Any) -> CouplerSpec:
    """A :class:`CouplerSpec`: ``coupler("proxy", mode="staggered", iterations=2, mass_scale=20.0)``."""
    return CouplerSpec(method=method, params=_frozen_params(params))


@dataclass(frozen=True, kw_only=True)
class CollisionSpec:
    """The collision policy the composer itself consumes.

    Typed, unlike backend parameters, because these are read by the composer
    when it builds pipelines and by the scene builder before anything imports,
    not passed through to a solver.

    Attributes:
        soft_contact_margin: Collision-pipeline soft-contact margin [m]: how far
            from a shape a particle is *considered* for contact.
        full_surface_contact: Generate soft contacts over the whole soft-mesh
            surface -- edges and triangle interiors -- against rigid SDFs,
            rather than at vertices only. **This is what lets a parallel jaw
            hold cloth**: a pad narrower than a mesh cell finds no vertex
            between the jaws otherwise. Consumed only by VBD, and every
            participating rigid shape needs an SDF -- see :attr:`force_sdf`.
        force_sdf: Bake a signed-distance field for every shape that takes the
            builder's default config, before anything imports. Necessary beside
            :attr:`full_surface_contact` on a robot whose colliders are meshes.
        rigid_gap: Default contact gap for shapes that state none [m], applied
            to the builder before anything imports. ``None`` keeps Newton's.
        contact_ke: Contact stiffness for shapes that state none [N/m]. It
            sets how far a grasped object may sink into the fingers; the bound
            on going stiffer is stability (MuJoCo wants the ``solref``
            timeconst above twice the substep). ``None`` keeps Newton's 2.5e3.
        contact_kd: Contact damping for shapes that state none. Scale it with
            :attr:`contact_ke`. ``None`` keeps Newton's 100.
        rigid_contact_max: Contact-buffer ceiling for the main pipeline. 0
            keeps Newton's sizing, which is derived from the settled scene.
        collide_every: Run the main pipeline's narrow phase every this many
            substeps. 1 is exact; a scene whose contacts change slowly may run
            it less often, which is measured per arrangement, never assumed.
        contact_matching: Frame-to-frame contact matching for the main
            pipeline -- ``"disabled"``, ``"latest"`` or ``"sticky"``.
            ``"latest"`` is load-bearing for a rod that rolls.
        broad_phase: Broad phase of the main pipeline when the composer routes
            it an explicit pair list; the list is the broad phase, so
            ``"explicit"`` is the honest default.
    """

    soft_contact_margin: float = field(
        default=0.008, metadata=ui(tier=RECAPTURE, unit="m", range=(1.0e-4, 5.0e-2), group="soft contact")
    )
    full_surface_contact: bool = field(default=False, metadata=ui(tier=REBUILD, group="soft contact"))
    force_sdf: bool = field(default=False, metadata=ui(tier=REBUILD, group="soft contact"))
    rigid_gap: float | None = field(default=None, metadata=ui(tier=REBUILD, unit="m", group="rigid contact"))
    contact_ke: float | None = field(
        default=None, metadata=ui(tier=REBUILD, unit="N/m", log=True, range=(1.0e2, 1.0e6), group="rigid contact")
    )
    contact_kd: float | None = field(
        default=None, metadata=ui(tier=REBUILD, unit="N*s/m", log=True, range=(1.0e-3, 1.0e3), group="rigid contact")
    )
    rigid_contact_max: int = field(default=0, metadata=ui(tier=REBUILD, group="buffers"))
    collide_every: int = field(default=1, metadata=ui(tier=REBUILD, range=(1.0, 8.0), step=1.0, group="cadence"))
    contact_matching: str = field(
        default="disabled", metadata=ui(tier=REBUILD, choices=("disabled", "latest", "sticky"), group="rigid contact")
    )
    broad_phase: str = field(
        default="explicit", metadata=ui(tier=REBUILD, choices=("explicit", "sap", "nxn"), group="buffers")
    )

    def __post_init__(self) -> None:
        if int(self.collide_every) < 1:
            raise ValueError(f"collide_every must be >= 1, got {self.collide_every}")
        if self.contact_matching not in ("disabled", "latest", "sticky"):
            raise ValueError(
                f"contact_matching must be 'disabled', 'latest' or 'sticky', got {self.contact_matching!r}"
            )
        if self.broad_phase not in ("explicit", "sap", "nxn"):
            raise ValueError(f"broad_phase must be 'explicit', 'sap' or 'nxn', got {self.broad_phase!r}")


# ---------------------------------------------------------------------------
# The arrangement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ArrangementSpec:
    """What a staged physics name stands for: rates, a backend per domain, a coupler, a collision policy.

    Attributes:
        control_hz: Control rate [Hz]. One env step covers one period.
        substeps: Physics substeps per control step. Must be even -- the state
            swap is a Python rebind baked into the captured graph.
        gravity: Gravity [m/s^2], along -z.
        use_cuda_graph: Attempt to capture the substep block.
        backends: ``(domain, backend)`` pairs. ``"*"`` is the fallback for any
            domain not named. Two domains stating an *equal* backend share one
            solver island, which is how a cable and a towel end up in one VBD.
        coupling: How islands couple when there is more than one.
        collision: The collision policy.
        overrides: ``(entity name or role, rule or backend)`` pairs that amend
            what the assets declared -- a contact rule added or changed, a
            backend for one entity.
    """

    control_hz: float = field(default=50.0, metadata=ui(tier=REBUILD, unit="Hz", range=(10.0, 240.0), group="rates"))
    substeps: int = field(
        default=4,
        metadata=ui(tier=REBUILD, range=(2.0, 64.0), step=2.0, group="rates", doc="Even, so the swap comes out even."),
    )
    gravity: float = field(default=-9.81, metadata=ui(tier=REBUILD, unit="m/s^2", range=(-20.0, 0.0), group="runtime"))
    use_cuda_graph: bool = field(default=True, metadata=ui(tier=REBUILD, group="runtime"))
    backends: tuple[tuple[str, BackendSpec], ...] = (("*", BackendSpec(solver="mujoco")),)
    coupling: CouplerSpec = field(default_factory=CouplerSpec)
    collision: CollisionSpec = field(default_factory=CollisionSpec)
    overrides: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.substeps, bool) or int(self.substeps) != self.substeps or self.substeps <= 0:
            raise ValueError(f"substeps must be a positive whole number, got {self.substeps!r}")
        if self.substeps % 2:
            raise ValueError(f"substeps must be even so the state swap comes out even, got {self.substeps}")
        if not self.control_hz > 0.0:
            raise ValueError(f"control_hz must be positive, got {self.control_hz}")
        names = [domain for domain, _ in self.backends]
        if len(names) != len(set(names)):
            raise ValueError(f"a domain is given two backends: {names}")
        for domain, spec in self.backends:
            if not isinstance(spec, BackendSpec):
                raise TypeError(f"backends[{domain!r}] must be a BackendSpec, got {type(spec).__name__}")

    @property
    def sim_dt(self) -> float:
        """Substep timestep [s]."""
        return 1.0 / self.control_hz / self.substeps

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    def backend_for(self, domain: str) -> BackendSpec | None:
        """The backend serving ``domain``: an exact entry, else the ``"*"`` fallback, else ``None``."""
        table = dict(self.backends)
        return table.get(domain, table.get("*"))

    def domains(self) -> tuple[str, ...]:
        return tuple(domain for domain, _ in self.backends)

    def with_(self, **changes: Any) -> ArrangementSpec:
        """A copy with some fields changed -- ``dataclasses.replace`` under a shorter name."""
        return replace(self, **changes)

    def backends_replacing(self, domain: str, spec: BackendSpec) -> tuple[tuple[str, BackendSpec], ...]:
        """The backend table with ``domain`` served by ``spec`` (added when absent)."""
        out = [(d, spec if d == domain else b) for d, b in self.backends]
        if domain not in dict(self.backends):
            out.append((domain, spec))
        return tuple(out)

    def with_backend(self, domain: str, spec: BackendSpec) -> ArrangementSpec:
        return replace(self, backends=self.backends_replacing(domain, spec))

    def with_override(self, target: str, value: Any) -> ArrangementSpec:
        return replace(self, overrides=(*self.overrides, (target, value)))


# ---------------------------------------------------------------------------
# The arrangements the library ships
# ---------------------------------------------------------------------------

GENERIC = ArrangementSpec(
    control_hz=60.0,
    substeps=16,
    backends=(
        ("articulated", backend("mujoco")),
        # The soft domain is volumetric FEM, which no Newton solver here
        # serves; it goes to the SuperDex co-sim backend, a host island the
        # composer splits out before the coupler -- so its presence changes
        # nothing about how the entries below couple. Two engine steps per
        # control step: contact detection is at step endpoints, so a mirror
        # sweeping more than a few millimetres per engine step tunnels; 8 ms
        # steps double the safe approach speed and stay inside the engine's
        # own 10-25 ms comfort band.
        ("soft", backend("superdex", substeps=2)),
        ("*", backend("vbd", rigid_body_contact_buffer_size=256)),
    ),
    coupling=coupler("proxy"),
    collision=CollisionSpec(),
)
"""A scene steps without anyone having measured it. **Not a tuned configuration.**

Every arrangement in this tree but this one was measured against the content it
serves, and lives with that content. This is the other kind: what a scene gets
when nobody has said anything, chosen to be *legible and conservative* rather
than good. Read it, copy it, and change the numbers that matter for what you
are actually simulating -- it is a value for exactly that reason, so
``replace(GENERIC, substeps=4)`` is the whole edit.

**The robot on MuJoCo, everything else on VBD.** One island for whatever moves
freely -- a cube, a rope, a sheet, all three -- and a second only when there is
an arm to drive. That shape is the one the coupler is right for, and it was
found by measuring the alternatives rather than by preference. With free rigid
bodies on MuJoCo instead (the dataclass default), a cube beside a rope bounced
between 7 and 195 mm for as long as it was stepped, with ``body_qd`` reading
zero: a free body on the *source* side of a proxy channel has its pose
overwritten by the mirror. With free bodies on VBD but the coupler deciding the
soft side by the nodes' roles, a cube beside a robot sat frozen at its spawn: a
VBD island holding only ``objects`` read as a rigid island and got no channel
at all. :attr:`~robosimstudio.scene.physics.islands.Island.is_deformable` keys
on the solver's registered capability now, and under this table all four
scenes -- cube; cube and rope; cube and robot; all three -- land within half a
millimetre of each other.

    The dataclass default, ``mujoco`` for ``"*"``, is what a cloth and a rod
    fall into if you build an ``ArrangementSpec()`` by hand. Neither survives
    it: a cloth raises early and legibly, a rod passes the capability check and
    dies inside MuJoCo's own importer. Stage an arrangement, or start from this
    one.

**A 256-row per-body rigid contact buffer, not Newton's 64.** The default
overflows the moment a rod touches a mesh body -- ``Warning: Per-body rigid
contact buffer overflowed 97 > 64`` and then ``nan`` through the whole state,
which is a wrong answer arrived at through a printed warning nobody reads in
time. 256 is not a universal figure and is not presented as one: it is
the cable-routing suite's measurement, where a 42-capsule rod pinched by a
decomposed fingertip peaks at 232. A denser rod needs more, and the symptom is
the same overflow line.

**Sixteen substeps at 60 Hz: conservative, not tuned.** A cloth sheet at 240 Hz
tunnels through the table it is lying on -- a measured failure of the shipped
content, not a hypothetical -- so the default buys margin at the cost of speed.
A rigid-only scene does not need it and should say so::

    scene = rss.Scene(cell="bench_075", physics=replace(GENERIC, substeps=4))

**``proxy`` coupling pushes a deformable and does not pinch one.** A rod being
dragged needs two-way coupling and gets it here. **A cloth grasp does not work
under this default**, and no amount of tuning makes it: a hand closing on a
sheet needs edge and face contacts against the pads, and Newton's proxy harvest
refuses that force feedback. Both shipped cloth tasks stage ``one_way`` with
``full_surface_contact=True`` for precisely this reason. There is no coupling
that is right for a rod and for a cloth grasp both, and a default that inspected
your scene and picked one would be the hidden library knowledge this tier exists
to remove -- so it is written down here, in ``build(verbose=True)``, and in the
cloth example::

    physics=replace(GENERIC, coupling=coupler("one_way"),
                    collision=CollisionSpec(full_surface_contact=True))
"""


MUJOCO_500HZ = ArrangementSpec(
    control_hz=50.0,
    substeps=10,
    backends=(("*", backend("mujoco", nconmax=512, njmax=512, enable_multiccd=True)),),
    collision=CollisionSpec(contact_ke=6.0e4, contact_kd=500.0, rigid_gap=2.0e-3),
)
"""The arm and every free rigid body in one MuJoCo island, 50 Hz control over 500 Hz.

**The arrangement a hand can pick a rigid body up in.** :data:`GENERIC` puts free
bodies on VBD behind a proxy coupler, which pushes and cannot pinch: a Panda
closing on a 50 mm cube there lets it slide out or flings it. Here the fingers
stall on the faces and friction carries it, so a gripper closed from the
studio's Grip button holds what it closed on. The rates and the contact law are
the tool-sorting benchmark's own, measured there; tasks stage it by the name
``mujoco_500hz@v1``.

It is for rigid bodies. A sheet or a capsule-chain rope has no MuJoCo solver and
belongs on :data:`GENERIC`'s VBD island. A sponge joins it on its own engine::

    physics=MUJOCO_500HZ.with_backend("soft", backend("superdex", substeps=2))
"""
