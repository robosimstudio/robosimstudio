# SPDX-License-Identifier: Apache-2.0
"""The co-simulation bridge: one SuperDex scene per world of one host island.

What happens per control step, in order, all outside the CUDA graph:

1. This control step's Newton body poses are read for every mirrored shape --
   directly off the state by default, since the tick runs after the substeps
   and zero-lag coupling is worth one small device read; ``pose_source=
   "snapshot"`` trades a control period of staleness for the free batched
   host copy instead.
2. The mirrors are driven through those poses across ``substeps`` internal
   SuperDex steps, linearly interpolated -- the engine derives surface
   velocity from consecutive transforms, so interpolation is what friction
   sees as a sweep rather than a teleport. A pin that rides a body
   (:meth:`SuperdexBridge.ride`) is swept with them.
3. The deformed vertex positions are read back zero-copy, velocities are
   finite-differenced (SuperDex exposes no per-node velocity getter), and both
   Newton states' ``particle_q``/``particle_qd`` ranges are assigned -- both,
   so the substep double-buffer swap can never resurrect a stale copy.

The bridge is built from the finalized ``newton.Model`` plus a JSON-able
recipe (the manifest's host-island record), never from specs or entities --
the same construction the repro runner performs, so live and replayed scenes
share one build path.

A replicated model (``recipe["num_worlds"] > 1``) gets **one scene per world**,
each solved on its own: world ``w``'s actors, mirrors, pins and rest state all
live in ``scenes[w]``, and the tick steps every scene in turn. This is what
keeps a wave's physics the serial physics -- see :class:`SuperdexBridge`.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.engines.cosim import CosimRig
from robosimstudio.engines.feedback import WrenchAccumulator
from robosimstudio.engines.geometry import actor_builder, actor_geometry, builders_for, register_actor_builder
from robosimstudio.engines.pose import lerp_pose

from .mirror import build_mirror_actors

__all__ = ["SuperdexBridge", "ensure_initialized"]

log = logging.getLogger(__name__)

_ENGINE = {"threads": None}
"""Process-global engine state: the worker-thread count ``initialize`` ran with."""

_FOREIGN = -2
"""``contact_points``' body index for a side this island cannot name."""

_CONTACT_VECTORS = ("pos", "normal", "force", "v_soft", "v_other")


def _no_contacts(n: int = 0) -> dict[str, np.ndarray]:
    """The columns :meth:`SuperdexBridge.contact_points` returns, uninitialised."""
    out: dict[str, np.ndarray] = {name: np.empty((n, 3), dtype=np.float64) for name in _CONTACT_VECTORS}
    out["area"] = np.empty(n, dtype=np.float64)
    out["gap"] = np.empty(n, dtype=np.float64)
    out["body"] = np.empty(n, dtype=np.int64)
    return out


def _handle_value(actor) -> int:
    """An actor's engine handle as a plain int, the way ``pin_nodes`` takes it."""
    get_handle = getattr(actor, "get_handle", None)
    handle = get_handle() if callable(get_handle) else actor
    return int(handle.value)


def ensure_initialized(threads: int = -1):
    """Import and initialize SuperDex once per process; return the module.

    ``initialize``/``shutdown`` are process-global and thread-affine, and a
    shutdown from the wrong thread aborts -- so nothing here ever calls
    ``shutdown``; the process exit releases the engine. The first caller's
    thread count wins and later differing requests are logged rather than
    honoured, which is the honest reading of a process-global knob.

    Raises:
        ImportError: Naming the ``[superdex]`` extra when the wheel is absent.
    """
    try:
        import superdex.physics as physics
    except ImportError as error:
        raise ImportError(
            "the 'superdex' backend needs the superdex-physics wheel: install the [superdex] extra "
            "(uv sync --extra superdex). It ships for Python 3.12 only."
        ) from error
    if not physics.is_initialized():
        physics.initialize(num_worker_threads=int(threads))
        _ENGINE["threads"] = int(threads)

        def _forward(channel, tag, message, code):
            del code
            level = logging.WARNING if "WARN" in str(channel).upper() else logging.ERROR
            if "INFO" in str(channel).upper() or "VERBOSE" in str(channel).upper():
                level = logging.DEBUG
            log.log(level, "superdex[%s]: %s", tag, message)

        physics.set_log_callback(_forward)
    elif _ENGINE["threads"] is not None and int(threads) != _ENGINE["threads"]:
        log.debug("superdex already initialized with %d worker threads; ignoring %d", _ENGINE["threads"], threads)
    return physics


def _root_matrix(actor) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` of an actor's root transform: local -> its scene's world."""
    root = actor.get_root_transform()
    x, y, z, w = (float(v) for v in root.rotation.tolist())
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot, np.asarray(list(root.translation), float)


def _root_reader(actor):
    """Node positions in the scene's world frame for an actor whose root moves (a skinned soft body)."""

    def read() -> np.ndarray:
        rot, t = _root_matrix(actor)
        local = np.asarray(actor.get_node_positions_local(), dtype=np.float64).reshape(-1, 3)
        return local @ rot.T + t

    return read


def _contact_params(physics, params: dict, *, penalty: float, friction: float, padding: float) -> Any:
    """One actor's SuperDex ``ContactParams`` from its host params.

    The three figures every builder always stated, plus the three the engine's
    defaults were silently deciding. Each of those is ``None`` unless the
    entity states it, and ``None`` keeps the engine's own value, so a spec that
    says nothing builds exactly what it built before.

    * ``contact_threshold`` -- ``penalty_threshold_default`` [m], the distance
      at which the penalty *starts*. **The collider's value governs the pair**
      (measured: a free block on a static box rests 0.98 mm up with the
      threshold set on the block, 0.000 mm with it set on the box). The
      engine's 1 mm is a stand-off, and a connector with 0.15 mm of clearance
      in its jack is pressed by every wall at once -- 250 N of phantom grip.
    * ``contact_smoothing`` -- ``penalty_smoothing_half_distance`` [m].
    * ``friction_falloff`` -- ``friction_falloff_vel`` [m/s]: below it friction
      is proportional to slip speed, so there is no static friction at all.
    """
    contact = physics.ContactParams(
        penalty_coefficient=float(params.get("contact_penalty", penalty)),
        coulomb_friction_coefficient=float(params.get("friction", friction)),
        penalty_threshold_extra_padding=float(params.get("contact_padding", padding)),
    )
    for key, attr in (
        ("contact_threshold", "penalty_threshold_default"),
        ("contact_smoothing", "penalty_smoothing_half_distance"),
        ("friction_falloff", "friction_falloff_vel"),
    ):
        value = params.get(key)
        if value is not None:
            setattr(contact, attr, float(value))
    return contact


WELD_COINCIDENCE = 1.0e-3
"""How close a node-to-node weld's two nodes must be at rest, as a fraction of the local node spacing.

The engine's deformable-to-deformable constraint has no rest offset: a gap at
rest is closed by the first step. Measured on a cable joined into a 15 x 15 x
10 mm lattice: the joined nodes leave at about 1.1 m/s per millimetre of gap
and the body is dragged about a third of it, so the 8.7-16.6 mm gaps a
nearest-node join used to accept launched the body's nodes at 21 m/s and moved
it 34 mm. Nodes built at one point sit 0.06 um apart (float32 rounding near
1 m), and a thousandth of the spacing -- 10 um on that lattice, 1 um on a 1 mm
mesh -- admits that rounding while capping the launch near 0.01 m/s. Under a
0.06 mm spacing it no longer would, so :data:`WELD_FLOAT32_ULPS` floors it.
"""

WELD_FLOAT32_ULPS = 4
"""The floor under :data:`WELD_COINCIDENCE`: float32 units in the last place of a pair's largest coordinate.

The rest layouts are float32 (Newton's ``particle_q``), so two nodes built at
one point can differ by the rounding of each coordinate -- at most one ULP per
axis, sqrt(3) ULPs across three. Measured on that joined cable and lattice at
eight placements (z up to 2.25 m, x and y off the bench centre): every joined
pair either equal or one ULP apart on one axis (0.0596 um at x = 0.6 m). Four
ULPs is 0.24 um at 0.95 m and 0.48 um past 1 m -- a launch of about 0.5 mm/s
at the 1.1 m/s per millimetre above -- where a purely relative tolerance
would refuse those coincident nodes on any mesh finer than 0.06 mm.
"""


class SuperdexBridge:
    """One SuperDex scene, its actors, and the per-control-step tick.

    Args:
        model: The finalized Newton model -- rest positions, tets, materials,
            shape geometry all come from here.
        recipe: The manifest's host-island record: ``island`` (name),
            ``gravity`` [3], ``actors`` (per deformable entity:
            ``{"name", "kind", "particle_start", "particle_count", "params"}``)
            and ``mirrors`` (per Newton shape:
            ``{"shape", "body", "two_way"}``; ``body: -1`` is world-fixed).
        substeps: SuperDex steps per control step.
        threads: Worker threads for ``initialize`` -- first caller wins.
        feedback: ``"auto"`` harvests contact wrenches for two-way mirrors;
            ``"off"`` never does.
        pose_source: ``"fresh"`` (default) or ``"snapshot"`` -- see the module
            docstring's step 1.
        mirror_friction_falloff: ``friction_falloff_vel`` [m/s] for every
            mirror -- the speed at which regularised friction reaches its
            Coulomb cap, below which a held object creeps. It reaches no
            contact while the mirrors are static: SuperDex then uses the
            deformable's falloff. See
            :func:`~robosimstudio.engines.superdex.mirror.build_mirror_actors`.
        mirror_smoothing: ``penalty_smoothing_half_distance`` [m] for every
            mirrored collider -- see :func:`build_mirror_actors`. The engine's
            own 5 mm lets a fingertip sit 2 mm inside a foam block; this
            defaults to the 0.5 mm at which it rests on the surface instead.
        mirror_threshold: ``penalty_threshold_default`` [m] for every mirror,
            or ``None`` for the engine's 1 mm stand-off. See
            :func:`build_mirror_actors`.
        mirror_padding: ``penalty_threshold_extra_padding`` [m] for every
            mirror, which the engine adds only against a point-cloud actor (a
            rod). 0 keeps the engine's own. See :func:`build_mirror_actors`.
        mirror_exact_boxes: Mirror boxes as exact ``BOX`` colliders, not SDFs.

    **Actor builders.** Each recipe actor is built by whatever
    :func:`~robosimstudio.engines.geometry.register_actor_builder` holds for
    ``("superdex", kind)``, called as ``build(bridge, record, particle_q,
    tet_indices, tet_materials)``. This bundle registers ``tet`` and ``shell``
    (:attr:`BUILDERS`); a domain pack that ships another kind registers its own
    from the pack, and the lookup imports the packs before it refuses a kind. A
    builder returns the actor's entry -- ``name``, ``actor``, ``read`` (a
    callable giving the nodes' world positions ``[count, 3]``), ``start``,
    ``count`` and ``prev`` (the rest positions, in the engine's precision) --
    and may use what the bridge offers every builder: :attr:`physics`,
    :attr:`model`, :attr:`recipe`, :meth:`scene_of`, :meth:`actor_root`,
    :meth:`contact_params`, and :attr:`actors`, the entries built before it
    (each stamped with its ``entity`` and ``world``).

    **One scene per world.** A replicated model superimposes its worlds at the
    origin, which is right for Newton -- its broadphase pairs within a world --
    and impossible in one host scene, which has no world axis to pair within:
    sixteen sponges and sixteen grippers at the same coordinates are one pile.
    The first cut laid the worlds out ten metres apart in a single scene, and
    that had a subtler cost than the offsets it needed on every read and write:
    **one scene is one Newton solve**, with one iteration cap
    (``non_linear_solver.max_iter``, 4 by default and ``STOPPED`` at it on
    every contact step) and one line search shared by every world's contacts.
    Per-world accuracy therefore fell with the world count -- the same recorded
    grasp of a 9 g sponge, mirrors at 1 mm/s friction falloff, held 4/4 at four
    worlds, 8/8 at eight and 3/12 at twelve; raising the cap to 48 held 12/12 at
    36 ms per world-step, slower than the serial 28 ms. Each world now has its
    own scene and therefore serial's own four iterations, so what a wave
    measures about a grasp is what the serial env measures, and a world's
    divergence is that world's alone. The cost is ``W`` solves per control step
    instead of one larger one, and they run side by side (see ``_lanes``):
    twelve worlds of a surface-preparation scene step in 110 ms against the shared scene's
    113 ms, and 320 ms with the lanes stepped one after another. The engine
    keeps every scene's state resident and hands out handles that are unique
    across scenes.
    """

    def __init__(
        self,
        model: Any,
        recipe: dict,
        *,
        substeps: int = 1,
        threads: int = -1,
        feedback: str = "auto",
        pose_source: str = "fresh",
        mirror_smoothing: float = 5.0e-4,
        mirror_friction_falloff: float = 1.0e-2,
        mirror_threshold: float | None = None,
        mirror_padding: float = 0.0,
        mirror_exact_boxes: bool = True,
        linear_solver: str | None = None,
        preconditioner: str | None = None,
        newton_iterations: int | None = None,
        linear_iterations: int | None = None,
    ) -> None:
        physics = ensure_initialized(threads)
        self.physics = physics
        self.model = model
        self.recipe = recipe
        self.substeps = max(1, int(substeps))
        self.feedback = str(feedback)
        self.pose_source = str(pose_source)
        self.mirror_smoothing = float(mirror_smoothing)
        self.mirror_friction_falloff = float(mirror_friction_falloff)
        self.mirror_threshold = None if mirror_threshold is None else float(mirror_threshold)
        self.mirror_padding = float(mirror_padding)
        self.mirror_exact_boxes = bool(mirror_exact_boxes)
        self.num_worlds = max(1, int(recipe.get("num_worlds", 1)))
        self.rig: CosimRig | None = None  # built once the mirrors are known
        self._diverged_logged: set[int] = set()
        self._diverged_worlds: list[bool] = [False] * self.num_worlds
        self._mirror_bodies: dict[int, int] | None = None  # actor handle -> newton body
        island = recipe.get("island", "soft")
        gravity = [float(v) for v in recipe.get("gravity", (0.0, 0.0, -9.81))]
        self.scenes: list[Any] = []
        """One engine scene per world, indexed by world. See the class docstring."""
        for world in range(self.num_worlds):
            scene = physics.create_scene(f"rss:{island}" if self.num_worlds == 1 else f"rss:{island}#w{world}")
            scene.set_gravity(gravity)
            self._tune_solver(scene, linear_solver, preconditioner, newton_iterations, linear_iterations)
            self.scenes.append(scene)
        # One lane per world at tick time: a world's mirrors are posed and its
        # scene stepped on a thread of its own. Both calls release the GIL, and
        # the engine's solves overlap -- twelve surface-preparation scenes step in
        # 22 ms side by side against 106 ms one after another (each scene's
        # solve is small and leaves the engine's own worker pool mostly idle).
        self._lanes = (
            ThreadPoolExecutor(max_workers=self.num_worlds, thread_name_prefix="superdex")
            if self.num_worlds > 1
            else None
        )
        self.actors: list[dict] = []
        self._actors_per_world: dict[int, int] = {}
        for record in recipe.get("actors", ()):
            world = int(record.get("world", 0))
            self._actors_per_world[world] = self._actors_per_world.get(world, 0) + 1
        particle_q = model.particle_q.numpy()
        tet_indices = model.tet_indices.numpy().reshape(-1, 4) if model.tet_count else np.zeros((0, 4), np.int64)
        tet_materials = model.tet_materials.numpy() if model.tet_count else np.zeros((0, 3))
        for record in recipe.get("actors", ()):
            kind = str(record.get("kind", "tet"))
            build = actor_builder("superdex", kind)
            if build is None:
                raise NotImplementedError(
                    f"superdex actor kind {kind!r} for {record.get('name')!r} is not built here; "
                    f"this engine builds {builders_for('superdex')}. "
                    "Reading a kind off the model is engines.geometry.register_actor_kind; building it on "
                    "this engine is engines.geometry.register_actor_builder('superdex', kind, build)."
                )
            entry = build(self, record, particle_q, tet_indices, tet_materials)
            # Stamped here rather than in each builder: which entity an actor is
            # and which world it belongs to are properties of the *record*, and
            # every builder agreeing to copy two fields is that many chances to
            # disagree. ``name`` stays whatever the builder made it -- it is what
            # the engine shows in a log -- and the lookup below uses these.
            entry.setdefault("entity", str(record.get("entity", record.get("name", ""))))
            entry["world"] = int(record.get("world", 0))
            self.actors.append(entry)

        self.welds: list[Any] = []
        """The engine constraints that join one actor's nodes to another's -- see :meth:`_build_welds`."""
        self._build_welds()

        # Which deformables a mirror can possibly touch. A replicated model's
        # worlds are separate scenes, so a collider in world 3 against the
        # sponge in world 7 is not even a pair the engine knows, and the wrench
        # harvest below would otherwise ask it ``mirrors x actors`` times a step
        # -- at twelve worlds, 33 thousand engine calls a step of which eleven
        # twelfths are known zero.
        self.actors_by_world: dict[int, list[dict]] = {}
        for entry in self.actors:
            self.actors_by_world.setdefault(int(entry["world"]), []).append(entry)

        # Per-sample contact records, for a caller that needs to know WHERE a
        # deformable is being pressed and not only how hard. Registered here
        # rather than on demand because a query must be live BEFORE the step
        # that fills it: a lazy first call would read a step that computed
        # none, which is a zero that looks like an answer. Measured free --
        # 1.0-1.5 ms/step with it and 1.6 ms without, inside one scene's own
        # run-to-run spread.
        for entry in self.actors:
            actor = entry["actor"]
            if actor.is_query_supported(physics.QueryType.CONTACT_POINTS):
                actor.register_query(physics.QueryType.CONTACT_POINTS)
                entry["contacts"] = actor.get_contact_points_world

        self.mirrors = build_mirror_actors(
            physics,
            self.scenes,
            model,
            shapes=list(recipe.get("mirrors", ())),
            contact_smoothing=self.mirror_smoothing,
            friction_falloff=self.mirror_friction_falloff,
            contact_threshold=self.mirror_threshold,
            contact_padding=self.mirror_padding,
            exact_boxes=self.mirror_exact_boxes,
        )
        self._moving_bodies = sorted({m.body for m in self.mirrors if m.moving})
        self.rig = CosimRig(self._moving_bodies, substeps=self.substeps, pose_source=self.pose_source)
        self._rest_state: list | None = None  # one captured state per scene
        self._pins: dict[int, list] = {}
        self._pin_world: dict[int, int] = {}
        """Which world each pin was made for, so re-aiming one needs no argument."""
        self._riders: dict[int, tuple[int, np.ndarray]] = {}
        """Pin -> ``(Newton body, its nodes' targets [n, 3] in that body's frame)``: the pins the tick aims."""
        self._rider_prev: dict[int, np.ndarray] = {}
        """Each ridden body's pose at the end of the last tick: where this tick's sweep starts."""
        self._pin_seq = 0

        # Contact-wrench feedback: for every two-way *moving* mirror, the force
        # the deformables exert on it is harvested per pair and handed back as
        # a Newton ``body_f`` wrench -- (force, torque) about the body's COM,
        # world frame, applied by the solver's rigid_force stage across the
        # NEXT control step's substeps. A zero-order hold with one step of lag,
        # which is what an eager-phase coupling can honestly offer.
        self.feedback_enabled = (
            self.feedback != "off"
            and any(m.two_way and m.moving for m in self.mirrors)
            and int(getattr(model, "body_count", 0)) > 0
        )
        self.apply_wrenches = None
        """Set by the owning solver: ``(wrench[n_bodies, 6] float32) -> None``."""
        if self.feedback_enabled:
            for entry in self.actors:
                entry["actor"].register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
            self._acc = WrenchAccumulator(
                model,
                alpha=float(recipe.get("feedback_alpha", 0.35)),
                dv_max=float(recipe.get("feedback_dv_max", 0.5)),
            )
            # Transient shaping, and it is load bearing, not taste: the penalty
            # force at first contact is stiff (a millimetre of overlap across a
            # palm-sized face is tens of newtons) and a zero-order hold applies
            # it for a whole control step after the surfaces may already have
            # separated -- explicit Euler on a stiff spring, which launched a
            # 63 g cube at 8 m/s in the first integration test.
            self.feedback_alpha = self._acc.alpha
            self.feedback_dv_max = self._acc.dv_max

    # ------------------------------------------------------------------ build
    def _tune_solver(
        self,
        scene: Any,
        linear: str | None,
        preconditioner: str | None,
        iterations: int | None,
        linear_iterations: int | None = None,
    ) -> None:
        """The engine's linear solve, by name, or its own choice where ``None``.

        The engine picks ``AUTO`` with a per-actor preconditioner, which is
        right for foam. Against stiff polymer (E ~ 1e8-1e9) under 1e10 penalties
        that choice runs its 1000-iteration cap on every Newton step: measured
        on a scene of three clips, a plug and a cable, 419 ms per control step
        at the default, 145 ms at ``CG`` + ``BLOCK_JACOBI``, 179 ms at
        ``MINRES``, and 8.1 s at the sparse direct ``LDLT``.
        """
        if linear is None and preconditioner is None and iterations is None and linear_iterations is None:
            return
        params = scene.get_solver_params()
        if linear is not None:
            params.linear_solver.solver_type = getattr(self.physics.LinearSolverType, str(linear).upper())
        if preconditioner is not None:
            params.linear_solver.preconditioner_type = getattr(
                self.physics.PreconditionerType, str(preconditioner).upper()
            )
        if iterations is not None:
            params.non_linear_solver.max_iter = int(iterations)
        if linear_iterations is not None:
            # A cap, not a tolerance: an inexact Newton step is still a Newton
            # step, and an uncapped CG spent 1000 iterations on every contact.
            params.linear_solver.max_iter = int(linear_iterations)
            # Hitting a cap that was set on purpose is not news: the engine
            # otherwise logs it on every Newton step, thousands of lines a second.
            params.linear_solver.verbosity = self.physics.VerbosityLevel.ERROR
            params.non_linear_solver.verbosity = self.physics.VerbosityLevel.ERROR
        scene.set_solver_params(params)

    def _shares_world(self, record: dict) -> bool:
        """Whether another deformable actor lives in this record's world.

        What ``collider="auto"`` asks: a soft body alone with the mirrors needs
        no collider of its own (its samples find theirs), and an SDF mapped
        through the deformation is not free; a soft body sharing its scene with
        a cable or another body is one those can only touch if it has one.
        """
        return self._actors_per_world.get(int(record.get("world", 0)), 0) > 1

    def _kept_out(self, record: dict) -> list[str]:
        """The other deformables in this record's world that pass through it, a body on a rigid core.

        Such a body has no collider, so it touches another deformable only
        where its own surface samples meet that one's collider, and only a soft
        body's SDF is one they meet: measured, a 60 x 60 x 30 mm block on a
        rigid core held a soft brick with an SDF (29.63 mm up, 29.82 mm on the
        plain block) and let a cable, a sheet and a brick with no collider
        through to 3.65, 0.94 and 0.83 mm (32.15, 29.08 and 29.71 mm on the
        plain block). So every actor of that world but a soft body that will
        have an SDF -- stated, or ``"auto"`` off a rigid core, which resolves to
        one here -- and but one welded to it with contact switched off, which
        the weld keeps apart anyway (a cable moulded into a plug). A kind
        another package builds is named: its collider is not known here.
        """

        def name(r: dict) -> str:
            return str(r.get("entity", r.get("name", "")))

        def welded(a: dict, b: dict) -> bool:
            welds = a.get("params", {}).get("welds", ())
            return any(str(w.get("to")) == name(b) and not bool(w.get("contact", False)) for w in welds)

        def has_sdf(r: dict) -> bool:
            params = r.get("params", {})
            collider = str(params.get("collider", "auto"))
            core = np.asarray(params.get("rigid_core", ())).size > 0
            return str(r.get("kind", "tet")) == "tet" and not core and collider in ("sdf", "auto")

        world = int(record.get("world", 0))
        return [
            name(other)
            for other in self.recipe.get("actors", ())
            if int(other.get("world", 0)) == world
            and name(other) != name(record)
            and not has_sdf(other)
            and not welded(other, record)
            and not welded(record, other)
        ]

    def _entry(self, entity: str, world: int = 0) -> dict:
        for entry in self.actors:
            if entry.get("entity", entry["name"]) == entity and int(entry.get("world", 0)) == int(world):
                return entry
        raise KeyError(
            f"no superdex actor for entity {entity!r} in world {world}; has {[e.get('entity') for e in self.actors]}"
        )

    def _build_welds(self) -> None:
        """Join actors that an entity said are one piece: a cable moulded into its plug.

        Stated per actor as ``params["welds"]``, each ``{"to": entity, "nodes":
        [local], "stiffness", "damping", "contact"}``: every listed node of this
        actor is tied by a node-to-node constraint to the **nearest node of the
        other actor at rest**, and contact between the two is switched off --
        the welded run lies *inside* the other body, which would otherwise push
        it out with every sample. Built here, after every actor exists and
        before the rest state is captured, so a reset restores a composition
        that still has them.

        Nearest-at-rest is chosen here rather than by the engine's
        ``find_closest`` because that flag is refused unless the first actor is
        a soft body, and a cable is not.

        **Only coincident nodes are joined node to node.** That constraint has
        no rest offset, so each pair must already meet, within
        :data:`WELD_COINCIDENCE` of the local node spacing (and never less than
        :data:`WELD_FLOAT32_ULPS` of float32 rounding), and no two listed nodes
        may share a partner -- otherwise :meth:`_check_coincident` raises.
        A body on a rigid core takes the other branch, which keeps every node's
        rest offset to the core.
        """
        for entry in list(self.actors):
            record = next(
                (
                    r
                    for r in self.recipe.get("actors", ())
                    if str(r.get("entity", r.get("name"))) == entry.get("entity", entry["name"])
                    and int(r.get("world", 0)) == int(entry["world"])
                ),
                None,
            )
            if record is None:
                continue
            for weld in record.get("params", {}).get("welds", ()):
                other = self._entry(str(weld["to"]), int(entry["world"]))
                scene = self.scene_of(entry["world"])
                # The rest layouts, which every entry carries as ``prev``: a soft
                # actor's position query has nothing to answer before a step.
                mine = np.asarray(entry["prev"], dtype=np.float64).reshape(-1, 3)
                theirs = np.asarray(other["prev"], dtype=np.float64).reshape(-1, 3)
                nodes = np.asarray(weld.get("nodes", ()), dtype=np.int64)
                nearest = np.asarray(
                    [int(np.argmin(np.linalg.norm(theirs - mine[node], axis=1))) for node in nodes], dtype=np.int64
                )
                if "link" not in other:
                    self._check_coincident(entry, other, nodes, nearest, mine, theirs)
                for node, partner in zip(nodes, nearest, strict=True):
                    if "link" in other:
                        # A node-to-node constraint onto a skinned soft body --
                        # onto its core or off it -- silently switched off every
                        # contact the welded cable had: the fingers passed
                        # through it and it never left its rest pose (measured
                        # both ways). The weld goes to the rigid core's link.
                        self.welds.append(
                            scene.create_deformable_node_to_rigid_constraint(
                                stiffness=float(weld.get("stiffness", 1.0e5)),
                                damping=float(weld.get("damping", 1.0)),
                                saturation=-1.0,
                                deformable_node_index=int(node),
                                deformable_actor=entry["actor"].get_handle(),
                                rigid_actor=other["link"].get_handle(),
                                fix_to_deformable_pos=True,
                            )
                        )
                        continue
                    self.welds.append(
                        scene.create_deformable_node_to_deformable_node_constraint(
                            stiffness=float(weld.get("stiffness", 1.0e5)),
                            damping=float(weld.get("damping", 1.0)),
                            saturation=-1.0,
                            node_index_a=int(node),
                            node_index_b=int(partner),
                            actor_a=entry["actor"].get_handle(),
                            actor_b=other["actor"].get_handle(),
                        )
                    )
                if not bool(weld.get("contact", False)):
                    scene.enable_actor_contact_symmetric(
                        entry["actor"].get_handle(),
                        other["actor"].get_handle(),
                        False,
                        self.physics.IncludeNestedActors.NO,
                    )

    @staticmethod
    def _check_coincident(
        entry: dict, other: dict, nodes: np.ndarray, nearest: np.ndarray, mine: np.ndarray, theirs: np.ndarray
    ) -> None:
        """Refuse a node-to-node weld whose pairs do not already meet at rest.

        Raises:
            ValueError: Naming each pair's gap against its tolerance, and any
                partner node two listed nodes would share.
        """
        if not len(nodes):
            return
        from scipy.spatial import cKDTree  # a heavy import a registry scan must not pay for

        def spacing(points: np.ndarray, index: np.ndarray) -> np.ndarray:
            """Distance from each indexed node to the nearest other node of the same actor.

            A KD-tree, O(N) in memory: the dense (len(index), N, 3) difference
            this replaced peaked at 133 MB for 20 joined nodes on a 104k-node
            lattice (0.8 MB now). The second neighbour: the first is the node.
            """
            distance, _ = cKDTree(points).query(points[index], k=2)
            return distance[:, 1]

        gaps = np.linalg.norm(theirs[nearest] - mine[nodes], axis=1)
        relative = WELD_COINCIDENCE * np.minimum(spacing(mine, nodes), spacing(theirs, nearest))
        extent = np.maximum(np.abs(mine[nodes]).max(axis=1), np.abs(theirs[nearest]).max(axis=1))
        rounding = WELD_FLOAT32_ULPS * np.spacing(extent.astype(np.float32)).astype(np.float64)
        tolerance = np.maximum(relative, rounding)
        far = gaps > tolerance
        partners, counts = np.unique(nearest, return_counts=True)
        shared = partners[counts > 1]
        if not far.any() and not shared.size:
            return
        name, to = entry.get("entity", entry["name"]), other.get("entity", other["name"])
        problems = []
        if far.any():
            problems.append(
                f"its nodes {nodes[far].tolist()} lie "
                f"{', '.join(f'{g * 1e3:.3g}' for g in gaps[far])} mm from their nearest {to!r} nodes "
                f"{nearest[far].tolist()} (tolerance {', '.join(f'{t * 1e3:.2g}' for t in tolerance[far])} mm)"
            )
        for partner in shared:
            problems.append(f"nodes {nodes[nearest == partner].tolist()} would all weld to {to!r} node {int(partner)}")
        raise ValueError(
            f"{name!r} joins {to!r} node to node, and the engine's node-to-node weld has no rest offset: "
            f"{'; '.join(problems)}. The first step would yank each pair together. Put {to!r} on a rigid core "
            f"(SoftSpec.rigid=...), whose weld keeps every node's offset -- on this engine it then has no collider, "
            f"so a cable, a sheet or a soft body without an SDF passes through it -- or mesh it so a node sits on "
            f"each joined one."
        )

    @property
    def scene(self) -> Any:
        """The single-world scene, or a replicated model's world 0. Prefer
        :meth:`scene_of` wherever a world is known."""
        return self.scenes[0]

    def scene_of(self, world: int) -> Any:
        """The engine scene that solves ``world``."""
        return self.scenes[int(world) % self.num_worlds]

    def actor_root(self, record: dict) -> Any:
        """Where this record's actor sits in its scene: identity, always.

        Every world has its own scene, so an actor keeps its own rest
        coordinates and node positions read back *local* mean the Newton world
        frame -- there is nothing to translate on the way in or out.
        """
        del record
        return self.physics.TransformRT()

    def contact_params(self, params: dict, *, penalty: float, friction: float, padding: float) -> Any:
        """One actor's engine ``ContactParams`` from its host params; see :func:`_contact_params`.

        The method form of the module's own helper, so an actor builder that
        lives outside this bundle states contact exactly as the shell and soft
        builders do without importing the bundle to reach it.
        """
        return _contact_params(self.physics, params, penalty=penalty, friction=friction, padding=padding)

    def _build_shell_actor(self, record: dict, particle_q: np.ndarray, tet_indices=None, tet_materials=None) -> dict:
        """A membrane sheet as a SuperDex shell actor.

        Triangles come off the model exactly as the render path reads them;
        elastic moduli are the island's (``shell_young``/``shell_poisson`` on
        the backend -- cloth specs speak VBD stiffnesses, not Young/Poisson),
        converted by the engine's own helper into thickness-integrated shell
        stiffness plus an areal density.
        """
        physics = self.physics
        del particle_q  # the shared reader takes it off the model
        geom = actor_geometry(self.model, {**record, "kind": "shell"})
        start, count, params, rest, tris = geom.start, geom.count, geom.params, geom.rest, geom.tris

        thickness = float(params.get("thickness", 0.002))
        areal_density = float(params.get("areal_density", 0.3))
        # The sheet's own modulus when its fabric states one, the island's
        # otherwise. The island value is a backend parameter and therefore one
        # number for every actor on it, which is fine for a scene with one sheet
        # and wrong the moment there are two of different cloth.
        material = physics.experimental.shell_material_params_from3d_isotropic(
            float(params.get("young", self.recipe.get("shell_young", 1.0e5))),
            float(params.get("poisson", self.recipe.get("shell_poisson", 0.3))),
            areal_density / max(thickness, 1.0e-5),
            thickness,
        )
        shape = physics.create_tri_mesh_shape(
            coordinates=rest.ravel(),
            connectivity=tris.astype(np.int32).ravel(),
        )
        shell_params = physics.experimental.ShellActorParams(
            name=str(record.get("name", f"shell:{start}")),
            shape=shape,
            material=material,
            world_from_local=self.actor_root(record),
        )
        shell_params.point_cloud_collider.radius = float(params.get("particle_radius", 0.002))
        shell_params.point_cloud_collider.self_contact = bool(params.get("self_contact", False))
        actor = physics.experimental.create_shell_actor(self.scene_of(record.get("world", 0)), shell_params)
        try:
            actor.set_recentering_params(physics.RecenteringParams(use_recentering=False))
        except (physics.Error, RuntimeError):
            pass  # a shell that recenters differently still reads back through its root transform
        actor.register_query(physics.QueryType.NODE_POSITIONS)
        np_real = np.float64 if physics.uses_double_precision() else np.float32
        return {
            "name": str(record.get("name", "")),
            "actor": actor,
            "read": actor.get_node_positions_local,
            "start": start,
            "count": count,
            "prev": rest.astype(np_real).copy(),
        }

    def _build_soft_actor(
        self, record: dict, particle_q: np.ndarray, tet_indices: np.ndarray = None, tet_materials: np.ndarray = None
    ) -> dict:
        physics = self.physics
        del particle_q, tet_indices, tet_materials  # the shared reader takes them off the model
        geom = actor_geometry(self.model, {**record, "kind": "tet"})
        start, count, params, rest, tets = geom.start, geom.count, geom.params, geom.rest, geom.tets
        young, poisson = geom.young, geom.poisson

        shape = physics.create_tet_mesh_shape(
            coordinates=rest.ravel(),
            connectivity=tets.astype(np.int32).ravel(),
        )
        soft = physics.SoftActorParams()
        soft.name = str(record.get("name", f"soft:{start}"))
        soft.layer = "soft"
        # Rest coordinates are already world -- the entity baked its pose into
        # the builder -- so at one world the actor lives at identity and local IS
        # world for every read-back below. Replicated, the root carries this
        # world's placement instead, which leaves local meaning exactly what it
        # meant before: still the Newton world frame, and still what every
        # reader below expects.
        soft.world_from_local = self.actor_root(record)
        soft.shape = shape
        soft.material = physics.SoftMaterialParams(
            type=physics.SoftMaterialType.NEO_HOOKEAN,
            neo_hookean=physics.NeoHookeanMaterialParams(youngs_modulus=young, poisson_ratio=poisson),
            density=float(params.get("density", 60.0)),
            mass_damping_coefficient=float(params.get("mass_damping", 0.0)),
            stiffness_damping_coefficient=float(params.get("stiffness_damping", 5.0e-3)),
        )
        soft.contact = _contact_params(physics, params, penalty=8.0e7, friction=0.9, padding=2.0e-3)
        soft.has_gravity = True
        soft.has_inertia = True
        soft.has_stress = True
        samples = params.get("boundary_samples")
        if samples:
            soft.boundary_element_type = getattr(physics.ActorBoundaryElementType, str(samples))
        # **A soft body is created with no collider at all** unless one is asked
        # for: ``ColliderType.NONE``, which the engine documents as "other actors
        # cannot detect contact with this actor". Its own boundary samples still
        # find the mirrors, which is all a sponge between two pads needs -- and
        # why nobody saw it -- but a cable pressed into a soft clip went straight
        # through it (measured: 0.000 mm of clip deflection, contact force 0.00 N,
        # with the cable's skin 2 mm inside the clip's lips). An SDF mapped
        # through the deformation is the collider a deformable *can* have.
        collider = str(params.get("collider", "auto"))
        core = np.asarray(params.get("rigid_core", ()), dtype=np.int64)
        if core.size:
            # The skinned actor's soft params carry no collider, and its core
            # link is built with none: an SDF asked for here was silently dropped
            # and a cable fell through the body (measured: 4.59 mm off the bench
            # through a 30 mm block that holds it at 33.02 mm without the core).
            # 'none' is the only collider such a body has, so an explicit 'sdf'
            # is refused and 'auto' -- the default, which asks for the collider
            # the body *can* have -- resolves to it, and says who loses by it:
            # every deformable in its world with no SDF for its samples to meet.
            if collider == "sdf":
                raise ValueError(
                    f"soft body {soft.name!r} rides a rigid core and its material states collider='sdf', which "
                    f"this engine cannot give it: a rigid-core body is built as a soft-skinned actor, which no other "
                    f"deformable can detect; it touches another only where its own surface samples meet that one's "
                    f"collider, so a soft body with an SDF holds it and a cable, a sheet or a soft body without one "
                    f"passes through. Leave the collider at 'auto' or 'none' and accept that, or drop SoftSpec.rigid "
                    f"if one of those must touch it (a plain soft body takes the SDF)."
                )
            if collider == "auto" and (blind := self._kept_out(record)):
                warnings.warn(
                    f"soft body {soft.name!r} rides a rigid core, so it has no collider on this engine: no other "
                    f"deformable can detect it, and it touches one only where its own surface samples meet that "
                    f"one's collider (a soft body's SDF). {blind} in its world {'has' if len(blind) == 1 else 'have'} "
                    f"no SDF for those samples to meet and will pass through it; collider 'auto' resolved to 'none'. Drop "
                    f"SoftSpec.rigid if they must touch it.",
                    stacklevel=2,
                )
            return self._build_skinned_actor(record, soft, rest, tets, core, start, count)
        if collider == "auto":
            collider = "sdf" if self._shares_world(record) else "none"
        experimental = physics.experimental.ExperimentalSoftActorParams()
        experimental.use_recentering = False
        experimental.collider_type = {"sdf": physics.ColliderType.SDF, "none": physics.ColliderType.NONE}[collider]
        if collider == "sdf":
            sdf = experimental.sdf
            resolution = params.get("sdf_resolution")
            if resolution is not None:
                sdf.resolution_mode = physics.GridSdfResolutionMode.EXPLICIT
                sdf.resolution_delta = [float(resolution)] * 3
            sdf.boundary_padding_dist = float(params.get("sdf_padding", 3.0e-3))
        actor = physics.experimental.create_soft_actor(self.scene_of(record.get("world", 0)), soft, experimental)
        # A moving local frame would put every read-back on a moving target;
        # with recentering off (above) and an identity root, node positions are
        # world. Kept for an engine build that ignores the experimental flag.
        try:
            actor.set_recentering_params(physics.RecenteringParams(use_recentering=False))
        except (physics.Error, RuntimeError):
            pass  # the experimental params already turned it off; the component is not there to set
        anchors = np.asarray(params.get("anchors", ()), dtype=np.int64)
        if anchors.size:
            # Permanent: a clip's base is part of the rail, not a pin a reset
            # might clear, and it must survive ``restore_rest``.
            actor.add_boundary_condition_nodes_world_permanent(
                node_indices=anchors.astype(np.int32), node_positions_world=rest[anchors].astype(np.float64).ravel()
            )
        actor.register_query(physics.QueryType.NODE_POSITIONS)
        np_real = np.float64 if physics.uses_double_precision() else np.float32
        return {
            "name": str(record.get("name", "")),
            "actor": actor,
            "read": actor.get_node_positions_local,
            "start": start,
            "count": count,
            "prev": rest.astype(np_real).copy(),
        }

    def _build_skinned_actor(
        self, record: dict, soft: Any, rest: np.ndarray, tets: np.ndarray, core: np.ndarray, start: int, count: int
    ) -> dict:
        """A soft body on a rigid core: SuperDex's soft-skinned actor, with one free link.

        The core's nodes are the mesh's ``constrainedNodes``, bound to a single
        rigid link on a ``FREE`` joint; the link owns the rigid motion (gravity
        and inertia evaluated on the posed nodes) and the soft solve owns only
        the deformation about it. That split is the point: free, a 1e9 Pa plug
        could not move rigidly in single precision at all.

        The engine takes constrained nodes only from a shape file, so the mesh
        goes through a ``.mochi.json`` in a scratch directory. The skeleton's
        frame sits at the core's centroid -- a rigid body's six DoFs about a
        point a metre away are not the ones to hand a single-precision solve.
        """
        import json
        import tempfile

        physics = self.physics
        params = record.get("params", {})
        centre = rest[core].mean(axis=0)
        local = rest - centre
        doc = {
            "mesh": {
                "nodesPerElement": 4,
                "coordinates": local.astype(float).ravel().tolist(),
                "connectivity": tets.astype(int).ravel().tolist(),
            },
            "constrainedNodes": core.astype(int).tolist(),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".mochi.json", delete=False) as fh:
            json.dump(doc, fh)
            shape_path = fh.name
        try:
            soft.shape = physics.load_shape_from_file(file_path=shape_path, bake_scale=[1.0, 1.0, 1.0])
        finally:
            Path(shape_path).unlink(missing_ok=True)
        soft.world_from_local = physics.TransformRT()
        # The skeleton evaluates gravity and inertia on the posed nodes; the soft
        # part, only its elasticity -- the engine refuses anything else.
        soft.has_gravity = False
        soft.has_inertia = False
        soft.has_stress = True
        lo, hi = local[core].min(axis=0), np.maximum(local[core].max(axis=0), local[core].min(axis=0) + 1.0e-4)
        corners = np.array([[(lo, hi)[i >> 2 & 1][0], (lo, hi)[i >> 1 & 1][1], (lo, hi)[i & 1][2]] for i in range(8)])
        # Outward-wound triangles: the link's mass properties come from them.
        faces = np.array(
            [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
        )
        name = soft.name
        link = physics.ArticulatedLinkParams(
            name=f"{name}:core",
            parent_link=-1,
            shape=physics.create_tri_mesh_shape(
                coordinates=corners.astype(float).ravel(), connectivity=faces.astype(np.int32)
            ),
            layer=f"{name}:core",
            collider_type=physics.ColliderType.NONE,
            # The body's mass is its nodes'; the link is a frame, not a second body.
            density=float(params.get("core_density", 1.0)),
        )
        skeleton = physics.ArticulatedActorParams(name=f"{name}:skeleton")
        skeleton.joints = [physics.ArticulatedJointParams(name=f"{name}:free", type=physics.ArticulatedJointType.FREE)]
        skeleton.links = [link]
        skeleton.world_from_root = physics.TransformRT(translation=[float(v) for v in centre])
        skinned = self.scene_of(record.get("world", 0)).create_soft_skinned_actor(
            physics.SoftSkinnedActorParams(
                skeleton_params=skeleton,
                soft_params=[soft],
                soft_attach_links=[f"{name}:core"],
                has_gravity=True,
                has_inertia=True,
                has_stress=False,
                enable_colliding_links=False,
            )
        )
        scene = self.scene_of(record.get("world", 0))
        actor = scene.get_actor(skinned.get_nested_soft_actors()[0])
        link_actor = scene.get_actor(skinned.get_nested_link_actors()[0])
        actor.register_query(physics.QueryType.NODE_POSITIONS)
        np_real = np.float64 if physics.uses_double_precision() else np.float32
        return {
            "name": str(record.get("name", "")),
            "actor": actor,
            "skinned": skinned,
            "link": link_actor,
            "core": frozenset(int(i) for i in core),
            "read": _root_reader(actor),
            "start": start,
            "count": count,
            "prev": rest.astype(np_real).copy(),
        }

    BUILDERS: ClassVar[dict] = {
        "tet": _build_soft_actor,
        "shell": _build_shell_actor,
    }
    """The actor kinds *this bundle* builds, and how. Not the whole of what the engine can.

    The neutral half -- reading a kind off the Newton model -- is open through
    :func:`~robosimstudio.engines.geometry.register_actor_kind`. Building one is
    open the same way, through
    :func:`~robosimstudio.engines.geometry.register_actor_builder`, and these
    two are simply what the bundle registers for itself at the foot of this
    module. A kind a domain pack ships registers its own builder from the pack,
    as a function taking the bridge (see "Actor builders" on the class), and a
    kind nobody registered is refused by name rather than approximated by the
    nearest one there is.
    """

    # ------------------------------------------------------------------ wiring
    @property
    def _wrench_ema(self) -> np.ndarray:
        """The feedback filter's state, exposed for inspection in tests."""
        return self._acc.ema

    def attach_pipeline(self, pipeline: Any) -> None:
        """Hand the bridge the pipeline whose states it writes. Idempotent."""
        self.rig.attach_pipeline(pipeline)

    def attach_states(self, states: tuple[Any, ...]) -> None:
        """Pipeline-less wiring -- the repro runner's states, both of them."""
        self.rig.attach_states(states)

    @property
    def pipeline(self) -> Any:
        """The pipeline the rig writes into. Read by the soft entity."""
        return self.rig.pipeline

    # ------------------------------------------------------------------ tick
    def tick(self, state: Any, dt: float) -> None:
        """Advance SuperDex by one control period and write the results back."""
        rig = self.rig
        ridden = sorted({body for body, _ in self._riders.values()})
        if ridden:  # one read for both: the mirrors' bodies and the ones a pin rides
            poses = rig.body_poses(state, sorted({*rig.bodies, *ridden}))
            targets = {b: poses[b] for b in rig.bodies}
        else:
            targets = rig.body_poses(state)
        teleport = rig.teleport
        step_dt = rig.substep_dt(dt)
        for sub, posed in rig.substeps_of(self.mirrors, targets):
            if ridden:
                self._aim_riders(poses, teleport, (sub + 1.0) / self.substeps)
            lanes: list[list] = [[] for _ in range(self.num_worlds)]
            for mirror, pose in posed:
                lanes[mirror.world % self.num_worlds].append((mirror, pose))
            if self._lanes is None:
                self._drive_world(0, lanes[0], step_dt, teleport)
            else:
                # ``list`` so a lane's exception surfaces here rather than being
                # dropped with the future.
                list(
                    self._lanes.map(self._drive_world, range(self.num_worlds), lanes, repeat(step_dt), repeat(teleport))
                )
        rig.commit(targets)
        self._rider_prev = {b: poses[b] for b in ridden} if ridden else {}

        for world, scene in enumerate(self.scenes):
            stats = scene.get_solver_stats()
            diverged = "DIVERGED" in str(stats.convergence_status)
            self._diverged_worlds[world] = diverged
            if diverged and world not in self._diverged_logged:
                self._diverged_logged.add(world)
                log.warning(
                    "superdex island %r world %d diverged (residual %.3g); its state may be unusable",
                    self.recipe.get("island"),
                    world,
                    float(getattr(stats, "residual_norm", float("nan"))),
                )

        if self.feedback_enabled and self.apply_wrenches is not None:
            self._acc.begin()
            for mirror in self.mirrors:
                if not (mirror.two_way and mirror.moving):
                    continue
                # A diverged step's contact records are stale by contract --
                # for that world; its neighbours' are as good as ever.
                if self._diverged_worlds[mirror.world % self.num_worlds]:
                    continue
                on_soft = np.zeros(3)
                # This world's deformables only -- see ``actors_by_world``.
                for entry in self.actors_by_world.get(mirror.world, ()):
                    on_soft += np.asarray(
                        entry["actor"].get_contact_force_from_actor_world(mirror.actor), dtype=np.float64
                    )
                if not np.any(on_soft):
                    continue
                # SuperDex reports the force on the *soft* body, and offers
                # no torque: the couple is entirely the moment arm, which
                # the accumulator takes from the shape's world frame.
                self._acc.add_at_shape(mirror.body, -on_soft, targets[mirror.body], mirror.local)
            self.apply_wrenches(self._acc.shape(dt))

        for entry in self.actors:
            # Through the entry's own reader rather than one call for every
            # kind: a soft or shell actor answers ``get_node_positions_local``,
            # and not every actor type does -- the engine refuses NodePositions
            # for some and states them as a rest mesh plus a per-node
            # displacement instead. The tick has no business knowing which; the
            # builder that made the actor does.
            entry["prev"] = self.rig.write_particles(
                state, entry["start"], entry["count"], np.asarray(entry["read"]()), dt, entry["prev"]
            )

    def _drive_world(self, world: int, posed: list, step_dt: float, teleport: bool) -> None:
        """Pose one world's moving mirrors and step its scene."""
        physics = self.physics
        for mirror, pose in posed:
            # A shape that has not moved since the last substep is not
            # written, and on a real cell most of them have not: an arm
            # parks, a support hand holds, and a decomposed hull set is
            # dozens of shapes on one link. This is the tick's dominant
            # cost -- measured at 83 per cent of it on a surface-preparation scene at
            # two worlds, against 4 per cent for the engine's own solve --
            # because every write is three FFI constructions, and there are
            # ``mirrors x substeps`` of them per control step.
            #
            # Skipping is exact rather than approximate: the engine holds
            # the last transform it was given, so not writing an identical
            # one changes nothing. ``teleport`` never skips, because it also
            # zeroes the velocity.
            if not teleport and np.array_equal(pose, mirror.prev):
                continue
            mirror.prev = pose.copy()
            # The pose is Newton's and goes in unchanged: this mirror's
            # scene holds only its own world, so nothing else's gripper is
            # at the same coordinates.
            mirror.actor.set_root_transform(
                physics.TransformRT(
                    rotation=physics.Quaternion(*[float(v) for v in pose[3:7]]),
                    translation=[float(v) for v in pose[:3]],
                )
            )
            if teleport:
                mirror.actor.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # This world's solve, with the whole iteration budget.
        self.scenes[world].step(step_dt)

    # --------------------------------------------------------------- contacts
    def contact_points(self, entity_name: str, *, against: int | None = None, world: int = 0) -> dict[str, np.ndarray]:
        """Where a deformable is being pressed, how hard, how fast, how wide.

        The per-sample half of what :attr:`_wrench_ema` reports in aggregate.
        Wrench feedback answers *how hard the sponge is pushed* with one vector
        per pair, which is all a rigid body's equation of motion needs; a
        surface law -- an abrasion field, a wipe, anything Preston-shaped --
        needs the same quantities *per contact patch*, and no amount of
        summing recovers them.

        Columns, all world frame, all normalised so the **deformable is side
        A** whatever order the engine reported the pair in (it may report the
        queried actor as either):

        =========  ========  =========================================
        ``pos``    (n, 3)    Contact point on the deformable [m]
        ``normal`` (n, 3)    Unit contact normal, pointing INTO the deformable
        ``force``  (n, 3)    Force ON the deformable, from the other side [N]
        ``v_soft`` (n, 3)    Velocity of the contact point on the deformable [m/s]
        ``v_other``(n, 3)    Velocity of the contact point on the other side [m/s]
        ``area``   (n,)      The sample's quadrature weight [m^2]
        ``gap``    (n,)      Signed separation [m]: positive is a gap,
                             **negative is interpenetration**
        ``body``   (n,)      Newton body of the other side
        =========  ========  =========================================

        ``gap`` is the one to reach for when a grasp looks like it is sinking
        into a deformable rather than pressing it. Nothing else here can tell
        a pad that is 3 mm inside the foam from a pad that has pushed the foam
        3 mm out of its way: both report contact, both report a force, and the
        block's bounding box is nearly the same either way.

        ``body`` is ``-1`` for a world-fixed mirror (the bench, the ground) and
        ``-2`` for anything this island cannot name -- a second deformable, or
        an actor no mirror was built for. ``against`` filters to one Newton
        body, and filters on that cheap field *before* reading any vector,
        which is most of what this costs.

        **What it costs.** SuperDex hands back a span of Python objects with no
        buffer protocol -- ``memoryview`` on one raises -- so every field is an
        attribute hop. Measured at 420 samples: ~1.3 us per vector column and
        ~7.6 us per sample for all of them, against a 0.8 ms control step on a
        bare sponge scene. Read what you need, filter early, and put the call
        behind a cadence or a proximity gate if the patch is large.

        ``area`` is the quadrature weight of the manifold the engine called
        side A, so in the rare pairing where the deformable is reported second
        it is the *collider's* sample area rather than the foam's. The two
        describe one physical patch and are the same order; nothing here
        pretends they are identical.

        Returns:
            Empty columns when this world's last step diverged -- SuperDex states
            that a diverged step's contact records may be stale, and the wrench
            path above already declines to read them.
        """
        entry = next(
            (
                e
                for e in self.actors
                if e.get("entity", e["name"]) == entity_name and int(e.get("world", 0)) == int(world)
            ),
            None,
        )
        if entry is None:
            raise KeyError(
                f"no actor named {entity_name!r} on island {self.recipe.get('island')!r}; "
                f"this island carries {[e['name'] for e in self.actors]}"
            )
        read = entry.get("contacts")
        if read is None:
            raise RuntimeError(
                f"actor {entity_name!r} does not support superdex's CONTACT_POINTS query, so "
                "per-sample contact cannot be read here; the aggregate pair force is what the "
                "feedback path harvests, through get_contact_force_from_actor_world."
            )
        if self._diverged_worlds[int(world) % self.num_worlds]:
            return _no_contacts()

        span = read()
        if not len(span):
            return _no_contacts()

        mine = _handle_value(entry["actor"])
        bodies = self._bodies_by_handle()
        keep: list[int] = []
        first_side: list[bool] = []
        hit_body: list[int] = []
        for index, point in enumerate(span):
            side_a = point.actor_a.value
            soft_is_a = side_a == mine
            other = point.actor_b.value if soft_is_a else side_a
            body = bodies.get(other, _FOREIGN)
            if against is not None and body != int(against):
                continue
            keep.append(index)
            first_side.append(soft_is_a)
            hit_body.append(body)

        n = len(keep)
        out = _no_contacts(n)
        if not n:
            return out
        pos, normal, force = out["pos"], out["normal"], out["force"]
        v_soft, v_other = out["v_soft"], out["v_other"]
        area, gap = out["area"], out["gap"]
        out["body"][:] = hit_body
        for row, (index, soft_is_a) in enumerate(zip(keep, first_side, strict=True)):
            point = span[index]
            if soft_is_a:
                pos[row] = np.asarray(point.pos_a)
                normal[row] = np.asarray(point.normal)
                force[row] = np.asarray(point.force)
                v_soft[row] = np.asarray(point.point_velocity_a)
                v_other[row] = np.asarray(point.point_velocity_b)
            else:
                # Reported the other way round: the position is B's, and the
                # force is stated ON A, so the deformable feels its negation.
                # The normal turns with it, for the same reason.
                pos[row] = np.asarray(point.pos_b)
                normal[row] = -np.asarray(point.normal)
                force[row] = -np.asarray(point.force)
                v_soft[row] = np.asarray(point.point_velocity_b)
                v_other[row] = np.asarray(point.point_velocity_a)
            area[row] = point.int_weight
            # Unsigned by the reversal: a separation is a separation whichever
            # manifold the engine called A.
            gap[row] = point.distance
        return out

    def _bodies_by_handle(self) -> dict[int, int]:
        """``{superdex actor handle: newton body}`` for every mirror. Cached.

        The body is Newton's, so a replicated model's worlds answer with their
        own -- which is what makes ``against=`` a per-world question without
        anything here having to ask which world it is being asked about. One
        table for every scene: the engine's handles carry the scene in their
        high bits and never collide across scenes.
        """
        if self._mirror_bodies is None:
            self._mirror_bodies = {_handle_value(m.actor): int(m.body) for m in self.mirrors}
        return self._mirror_bodies

    # ----------------------------------------------------------------- solver
    def solver_stats(self, world: int = 0) -> Any:
        """The engine's statistics for ``world``'s last solve."""
        return self.scene_of(world).get_solver_stats()

    def get_solver_params(self) -> Any:
        """The solver parameters every scene runs with (read off world 0)."""
        return self.scenes[0].get_solver_params()

    def set_solver_params(self, params: Any) -> None:
        """Give every world's scene the same solver parameters."""
        for scene in self.scenes:
            scene.set_solver_params(params)

    @property
    def diverged_worlds(self) -> tuple[int, ...]:
        """Which worlds' last step diverged."""
        return tuple(w for w, bad in enumerate(self._diverged_worlds) if bad)

    # ------------------------------------------------------------------- pins
    def pin_nodes(
        self,
        entity_name: str,
        nodes: np.ndarray,
        targets: np.ndarray,
        *,
        stiffness: float = 2000.0,
        damping: float = 10.0,
        world: int = 0,
    ) -> int:
        """Pin an actor's nodes to drivable world targets; returns a pin id.

        The host-engine half of a cloth attachment: where a Newton island's pin
        rewrites ``particle_q`` in ``post_solve``, a host island pins through
        the engine's own compliant node constraints, so the pinned patch still
        argues with contact and inertia instead of teleporting.

        Args:
            entity_name: Which actor's nodes (the entity's name).
            nodes: Global model particle indices, as the callers already hold.
            targets: World positions ``[n, 3]``, in Newton's frame, which is
                also every scene's frame.
            world: Which world's actor. 0 when nothing replicated it.
        """
        entry = next(
            e for e in self.actors if e.get("entity", e["name"]) == entity_name and int(e.get("world", 0)) == int(world)
        )
        scene = self.scene_of(world)
        actor = entry["actor"]
        handle_fn = getattr(actor, "get_handle", None)
        handle = handle_fn() if callable(handle_fn) else actor
        constraints = []
        for node, target in zip(np.asarray(nodes, dtype=np.int64), np.asarray(targets, dtype=np.float64), strict=True):
            constraints.append(
                scene.create_deformable_node_position_constraint(
                    stiffness=float(stiffness),
                    damping=float(damping),
                    saturation=-1.0,
                    node_index=int(node - entry["start"]),
                    position=[float(v) for v in target],
                    actor=handle,
                )
            )
        self._pin_seq += 1
        self._pins[self._pin_seq] = constraints
        self._pin_world[self._pin_seq] = int(world)
        return self._pin_seq

    def set_pin_targets(self, pin: int, targets: np.ndarray) -> None:
        """Re-aim one pin's constraints, in the order they were created.

        The targets are Newton's, like the ones :meth:`pin_nodes` took. A pin
        that rides a body (:meth:`ride`) is re-aimed by the tick, which
        overwrites these at its next substep.
        """
        for constraint, target in zip(self._pins[pin], np.asarray(targets, dtype=np.float64), strict=True):
            constraint.set_target_position([float(v) for v in target])

    def ride(self, pin: int, body: int, local: np.ndarray) -> None:
        """Carry one pin on a Newton body: its targets are ``local`` ``[n, 3]`` in the body's frame.

        Aimed by :meth:`tick` before **every engine substep**, from the body's
        pose swept through the control step the way the mirrors' are -- so the
        nodes keep up with a moving hand. Re-aimed once a control step from the
        outside instead, from the hand as it stood before the step, they
        trailed it by that step's travel: 5.2 mm at 0.3 m/s on the cable demo.
        Calling it again moves the offsets. :meth:`unpin` ends the ride.
        """
        if pin not in self._pins:
            raise KeyError(f"no pin {pin}")
        local = np.asarray(local, dtype=np.float64).reshape(-1, 3)
        if len(local) != len(self._pins[pin]):
            raise ValueError(f"pin {pin} holds {len(self._pins[pin])} nodes; {len(local)} offsets given")
        self._riders[int(pin)] = (int(body), local)

    def _aim_riders(self, poses: dict[int, np.ndarray], teleport: bool, t: float) -> None:
        """Every riding pin onto its body, ``t`` of the way through this control step."""
        for pin, (body, local) in self._riders.items():
            now = poses[body]
            pose = now if teleport or body not in self._rider_prev else lerp_pose(self._rider_prev[body], now, t)
            self.set_pin_targets(pin, pose[:3] + local @ quat_to_mat(pose[3:7]).T)

    def has_pin(self, pin: int) -> bool:
        """Whether ``pin`` still stands: :meth:`unpin`, :meth:`unpin_all` and :meth:`restore_rest` each end one."""
        return int(pin) in self._pins

    def unpin(self, pin: int) -> None:
        scene = self.scene_of(self._pin_world.pop(int(pin), 0))
        self._riders.pop(int(pin), None)
        for constraint in self._pins.pop(pin, ()):
            scene.destroy_constraint(constraint)

    def unpin_all(self) -> None:
        for pin in list(self._pins):
            self.unpin(pin)

    # ------------------------------------------------------------------ reset
    def place_actor(self, entity: str, world: int, positions: np.ndarray) -> bool:
        """Teleport one world's actor onto ``positions``; ``False`` if it has none.

        The channel a *drawn* deformable needs. An actor's rest coordinates are
        read off ``model.particle_q`` when this bridge is built, which is once,
        before any episode is known -- so a task whose foam is somewhere new
        every episode used to have to rebuild the model to say so. It can say it
        here instead: the lattice is the same lattice, rigidly moved, which is
        what a placement *is*, and a soft body's rest shape is invariant under
        one.

        ``positions`` is this actor's nodes in the Newton world frame, in the
        order the model holds them. Local **is** world for every actor this
        bridge builds -- recentering is off and the root is identity -- which is
        the same assumption every read-back above already makes.

        Velocities go to zero and ``prev`` is refreshed from what the engine
        reports back, because a teleport is not motion: the first speed reading
        after one must not be the jump. The mirrors are marked for teleport for
        the same reason ``restore_rest`` marks them.
        """
        rows = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        touched: list[dict] = []
        moved = False
        for entry in self.actors_by_world.get(int(world), ()):
            if entry.get("entity") != str(entity):
                continue
            if len(rows) != int(entry["count"]):
                raise ValueError(f"{entity!r} in world {world} has {entry['count']} nodes, given {len(rows)}")
            actor = entry["actor"]
            if "skinned" in entry:
                # Its nodes are the skeleton's pose plus a deformation; a placement is a skeleton pose.
                raise NotImplementedError(f"{entity!r} has a rigid core: place it by its skeleton, not its nodes")
            actor.set_node_positions_local(rows.ravel())
            actor.set_node_velocities_local(np.zeros(rows.size, dtype=np.float64))
            touched.append(entry)
            moved = True
        if moved:
            # The registered queries still hold what the actor looked like before
            # the write, so a read here would hand back the OLD nodes -- silently,
            # and ``prev`` would then be a whole placement behind, which reads as
            # a metre-per-second slide on the first control step. ``restore_rest``
            # refreshes for the same reason and says so; this is that line.
            self.scene_of(world).step(0.0)
            for entry in touched:
                entry["prev"] = np.ascontiguousarray(
                    np.asarray(entry["read"]()).reshape(-1, 3), dtype=entry["prev"].dtype
                )
            self.rig.mark_teleport()
        return moved

    def capture_rest(self) -> None:
        """Record every scene's current state as what a reset returns to."""
        if self._rest_state is not None:
            for scene, rest in zip(self.scenes, self._rest_state, strict=True):
                scene.release_state(rest)
        self._rest_state = [scene.capture_state() for scene in self.scenes]

    def restore_rest(self) -> None:
        """Return every scene to its captured rest; the next tick teleports mirrors."""
        if self._rest_state is None:
            return
        # Restore demands the composition the capture saw: pins created since
        # are extra constraints and must go first, not survive the episode.
        self.unpin_all()
        for scene, rest in zip(self.scenes, self._rest_state, strict=True):
            scene.restore_state(rest, release_immediately=False)
            scene.step(0.0)  # refresh queries without advancing state
        for entry in self.actors:
            positions = np.asarray(entry["read"]()).reshape(-1, 3)
            entry["prev"] = np.ascontiguousarray(positions, dtype=entry["prev"].dtype)
        self.rig.mark_teleport()
        self._diverged_logged.clear()
        self._diverged_worlds = [False] * self.num_worlds
        if self.feedback_enabled and self.apply_wrenches is not None:
            self.apply_wrenches(self._acc.zero())


_CONSUMES = {
    # What each builder above actually reads off ``HostActor.params``; the
    # composer warns on anything an actor states outside its kind's set.
    "shell": ("areal_density", "particle_radius", "poisson", "self_contact", "thickness", "young"),
    "tet": (
        "anchors",
        "boundary_samples",
        "collider",
        "contact_padding",
        "contact_penalty",
        "contact_smoothing",
        "contact_threshold",
        "density",
        "friction",
        "friction_falloff",
        "mass_damping",
        "rigid_core",
        "sdf_padding",
        "sdf_resolution",
        "stiffness_damping",
        "welds",
    ),
}
for _kind, _build in SuperdexBridge.BUILDERS.items():
    register_actor_builder(
        "superdex", _kind, _build, consumes=_CONSUMES[_kind], summary=f"SuperDex builds a {_kind} actor."
    )
del _kind, _build
