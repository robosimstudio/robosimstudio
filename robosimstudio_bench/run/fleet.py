# SPDX-License-Identifier: Apache-2.0
"""Wave scheduling: one model, many worlds, one episode per world.

The serial runner is one episode at a time through one environment. This is the
same measurement rotated ninety degrees: reset ``W`` worlds to ``W`` episodes,
step them together, score each one as it finishes, and write the same rows. A
wave is the unit because a reset is a wave-wide operation -- every world's start
pose is written in one host pass and one device assign -- and because "one reset,
one outcome per world" is what keeps an episode's identity intact when its
neighbours are running too.

Three things this module holds and nothing else does:

* :func:`plan_waves` -- episodes to waves, grouped by ``model_key``. The grouping
  is defensive rather than load-bearing: no shipped task declares
  ``model_key_fields``, so every job is one group today. A job file whose sampler
  changes the model per episode would otherwise run in whichever model the wave
  happened to build first, silently.
* :class:`WorldSlot` -- one world's assignment for one wave, and where its
  outcome accumulates. Deliberately a plain dataclass: the runner mutates it,
  the recorder reads it, and nothing about it is device state.
* :class:`FleetSummary` -- :class:`RunSummary` with the two rate properties that
  overlap breaks. A batched run's episodes share wall time, so summing per-row
  ``wall_s`` counts the same seconds ``W`` times.

The protocols at the bottom are the seams to the two workstreams below this one:
what a batched environment must offer the runner, and what a batched policy must.
They are ``Protocol`` rather than base classes because the implementations live
in different packages and inherit from their own hierarchies already.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.specs import Episode, TaskSpec, replace
from robosimstudio_bench.goals.context import diverged_result, finite_reward
from robosimstudio_bench.taskbuild.episodes import make_episode, model_key_for

from .summary import RunSummary

__all__ = [
    "PLACEABLE_FIELDS",
    "FleetSummary",
    "WorldSlot",
    "fleet_build_task",
    "fleet_model_key",
    "plan_mixed_waves",
    "plan_waves",
    "spawn_fields_placed_per_world",
]

PLACEABLE_FIELDS = ("static_poses", "static_xforms")
"""Sampled fields a fleet places per world at reset rather than building for.

A welded static is world-hung scenery in a single-world model, so a sampler
that moves one declares it model-changing and the serial environment rebuilds
per episode. A replicated build gives every placeable static a kinematic body
of its own (:class:`~robosimstudio.domains.static.entity.EntityStatic`), so
two episodes that differ only in where such a static stands share one model
here -- and the wave planner keys them together."""


def static_places_per_world(spec) -> bool:
    """Whether the class that builds this static gives it a per-world body in a replicated build.

    Asked of the entity class, not of the spec: a suite may build a mesh
    static its own way (the usbc port welds the reference's chamfered STL),
    and only the class knows whether that build claims a body. A key that
    trusted the asset suffix alone shared a model between episodes whose port
    every world then kept at the build episode's pose.
    """
    from robosimstudio.domains.static.entity import EntityStatic
    from robosimstudio.scene.registry import resolve_entity_class

    try:
        cls = resolve_entity_class(spec)
    except (KeyError, ValueError):
        return False
    return isinstance(cls, type) and issubclass(cls, EntityStatic) and cls.places_per_world(spec)


def spawn_fields_placed_per_world(build: TaskSpec) -> frozenset[str]:
    """Sampled fields this task's entities *draw* per world rather than build for.

    :data:`PLACEABLE_FIELDS`' sibling, and the difference between the two is which
    layer can name the field. A static's episode pose is in every task's
    vocabulary -- ``static_poses`` means the same thing everywhere -- so the
    planner names it. A drawn SPAWN is the entity's own: ``cable_bow`` means
    nothing to a task with no rod, and what the four fractions do to a rod is in
    the rod's builder rather than in a pose. So the class that consumes the field
    is what names it, through ``fields_placed_per_world(spec)``, and only if it
    also *writes* it per world -- the same contract
    :func:`static_places_per_world` keeps, for the same reason: a key that dropped
    a field nothing places would share one model between episodes whose rods it
    cannot draw, and every world would silently run the build episode's.

    Asked of the arrangement's implementation of the domain, not of the generic
    one: the same rod spec is a capsule chain under Newton and a Cosserat
    centreline under SuperDex, and only one of those two writes a per-world spawn.
    """
    from robosimstudio.scene.registry import resolve_entity_class

    out: set[str] = set()
    for spec in getattr(build, "entities", ()):
        try:
            cls = resolve_entity_class(spec, getattr(build, "physics", None))
        except (KeyError, ValueError):
            continue
        answer = getattr(cls, "fields_placed_per_world", None)
        if callable(answer):
            out |= {str(name) for name in answer(spec)}
    return frozenset(out)


def fleet_build_task(task: TaskSpec) -> TaskSpec:
    """The task whose MODEL a fleet builds for ``task``: its fleet base, else itself.

    A document that declares ``(:fleet-base X)`` is an *instance* of ``X``: its
    objects are a subset of the base's, and a wave runs it as one world of the
    base's model with everything it does not name parked. The base is
    resolved by name here, when a wave is planned, and pinned to the
    document's own pairing -- an instance staged on a different table or
    driven by a different robot than its base would be an instance of a
    different model, and the base has to declare that pairing.

    Returns ``task`` itself when it declares no base, which is every document
    written before superset instances existed, so callers need not branch.

    Raises:
        KeyError: When the base is not registered, or does not declare the
            document's pairing.
    """
    name = str(getattr(task, "fleet_base", "") or "")
    if not name:
        return task
    from robosimstudio_bench.taskbuild.matrix import declared_key
    from robosimstudio_bench.taskbuild.registry import get_task

    base = get_task(name)
    # The base's own spelling of the pairing, so the robot's own hand named on
    # one side and left unstated on the other is the same binding.
    scene, embodiment, gripper = declared_key(base, task.scene, task.embodiment, task.gripper) or (
        task.scene,
        task.embodiment,
        task.gripper,
    )
    base.binding_for(scene, embodiment, gripper)
    return replace(
        base,
        scene=scene or base.scene,
        embodiment=embodiment or base.embodiment,
        gripper=gripper or base.gripper,
    )


def fleet_model_key(task: TaskSpec, episode: Episode) -> str:
    """The model an episode needs *in a fleet*: its ``model_key`` less what a wave places per world.

    Identical to :attr:`Episode.model_key` unless the episode moves statics and
    every static it moves is placeable
    (:func:`~robosimstudio.domains.static.entity.placeable_static`); an
    episode that moves an imported arena keeps its own key, because that pose
    is baked at conversion in any build.

    Less, as well, whatever its entities *draw* per world
    (:func:`spawn_fields_placed_per_world`) -- ``cable_route``'s four rod
    fractions, which are geometry rather than a pose and are therefore written
    into each world's rows at build time rather than placed at reset.

    Under ``(:fleet-base X)`` it is **the base's** key -- the base's model with
    this document's episode as state -- so every instance of one superset keys
    together and a wave holds twelve of them. The active set is never part of
    a key: it is the one override whose whole purpose is to differ between
    worlds of one model.
    """
    build = fleet_build_task(task)
    overrides = dict(episode.overrides)
    moved: set[str] = set()
    for field_name in PLACEABLE_FIELDS:
        moved |= set((overrides.get(field_name) or {}).keys())
    statics = {spec.name: spec for spec in getattr(build, "statics", ())}
    placeable = bool(moved) and all(name in statics and static_places_per_world(statics[name]) for name in moved)
    if placeable:
        for field_name in PLACEABLE_FIELDS:
            overrides.pop(field_name, None)
    # And the drawn spawns, independently: a task may place its statics per world
    # and draw nothing, or draw a rod and move no static. ``cable_route`` does
    # both, and either half left in the key is a hundred models for a hundred
    # documents just as surely as both would be.
    drawn = [name for name in spawn_fields_placed_per_world(build) if name in overrides]
    for field_name in drawn:
        overrides.pop(field_name, None)
    if build is task and not placeable and not drawn:
        return episode.model_key
    return model_key_for(build, overrides, instruction=episode.instruction)


# ------------------------------------------------------------------ scheduling
def plan_mixed_waves(work: Sequence[tuple[TaskSpec, int]], num_worlds: int) -> list[list[tuple[TaskSpec, Episode]]]:
    """Group ``(document, episode id)`` work into waves of at most ``num_worlds``.

    The general form of :func:`plan_waves`, and the one thing that makes a wave
    able to hold **several documents**: the grouping key is
    :func:`fleet_model_key`, which is the *base's* key for any document naming a
    ``(:fleet-base X)``, so thirty instances of one superset land in one group
    and run twelve to a wave. Two documents of one base are then as batchable as
    two episodes of one document, which is exactly what they are -- the model is
    the same and what differs is state.

    Measured on this tree before the runner could use it: two instance documents
    of ``tool_hanging/wrenches_level2`` produce the same ``fleet_model_key`` as
    each other and as the base, while their own ``Episode.model_key`` differ,
    because a document's own key begins with its ``full_name``.

    Grouping preserves first-seen key order. **Within** a key, members are packed
    by declared step budget, because a wave's own loop bound is the ``max`` of its
    slots' ``max_steps`` -- so a wave holding a 22 800-step document beside a
    91 200-step one is bounded by the second, and every world that finishes early
    is stepped for nothing until the longest is done. Sorting by budget puts
    documents that will be cut off at the same time in the same wave.

    The sort is **stable and keyed only on the budget**, so a single-document job
    is still ``episode_ids`` chunked in order and its rows sort the way a serial
    run's do: every episode of one document declares the same ``max_steps``, so
    the key is constant and the original order survives untouched.

    Measured on ``sharps_disposal/tray_clutter`` (30 documents, budgets
    45 600-91 200, episodes 4 860-16 677): one wave of thirty stepped 500 310
    world-steps to do 316 115 of work -- ``FleetSummary.utilization`` 0.6318, i.e.
    **37% of the wave spent stepping worlds that had already finished** -- and the
    wave lost to four serial processes 4 642 s against 3 853 s. Packed into three
    waves of ten it steps 397 760. The declared budget predicts the realised
    episode length at ``r = 0.775`` on that cell and ``r = 0.996`` on
    ``tool_sort/level1``; it cannot help a cell whose documents all declare the
    same budget (``tool_hanging``, ``sharps_disposal/tray_pair``), where there is
    no spread to sort by and the sort is a no-op.

    Args:
        work: ``(document, episode id)`` pairs, in the order the job asked for
            them. The same document may appear more than once, for several of
            its episodes.
        num_worlds: Worlds per wave. Must be positive.

    Returns:
        Waves, each a list of at most ``num_worlds`` ``(document, episode)``
        pairs, packed by declared step budget within a model key. A wave never
        mixes model keys; the last wave of a group may be short, and the runner
        parks the leftover worlds.
    """
    if num_worlds < 1:
        raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
    groups: dict[str, list[tuple[TaskSpec, Episode]]] = {}
    for task, episode_id in work:
        episode = make_episode(task, int(episode_id))
        groups.setdefault(fleet_model_key(task, episode), []).append((task, episode))
    waves: list[list[tuple[TaskSpec, Episode]]] = []
    for members in groups.values():
        # Stable, and keyed only on the budget: a group of one document has a
        # constant key, so its given order is preserved exactly.
        packed = sorted(members, key=lambda pair: pair[0].max_steps or 0)
        for start in range(0, len(packed), num_worlds):
            waves.append(packed[start : start + num_worlds])
    return waves


def plan_waves(task: TaskSpec, episode_ids: list[int] | tuple[int, ...], num_worlds: int) -> list[list[Episode]]:
    """Group one document's episodes into waves of at most ``num_worlds``.

    :func:`plan_mixed_waves` over one document, with the documents dropped on
    the way out -- the shape every caller written before mixed-document waves
    expects, and what ``vabar_eval.wave_widths`` asks for when it wants to know
    how wide a cell would run.

    Episodes are materialised here rather than in the runner because the grouping
    key is a property of the sampled episode, not of its id: two ids differ in
    model only if their ``model_key`` does.

    The key is :func:`fleet_model_key`, not the episode's own: a drawn board of
    welded pegs is one model to a fleet, which places each world's pegs at
    reset, and many models to the serial environment, which rebuilds.
    """
    return [
        [episode for _task, episode in wave] for wave in plan_mixed_waves([(task, i) for i in episode_ids], num_worlds)
    ]


@dataclass
class WorldSlot:
    """One world's assignment within one wave, and what it ended up doing.

    ``episode`` is ``None`` for a parked world -- the tail of a wave that did not
    divide evenly. A parked world still steps (it is in the same model) but is
    never scored and produces no row.

    Attributes:
        world: Index into the replicated model.
        episode: What this world is running, or ``None`` when parked.
        task: The DOCUMENT this world is running -- which in a mixed-document
            wave is not the job's, and is what the row is labelled with, what
            the evaluator is built from, and what the step budget comes from.
            ``None`` for a parked world.
        evaluator: This world's own scorer. Fresh per assignment, because task
            evaluators latch -- one instance shared across worlds would declare
            every world successful the moment any of them was.
        settled_body_q: The wave's start pose for this world, captured from the
            first snapshot after reset. Predicates that measure displacement read
            it, so it must be this world's own.
        live: Still stepping toward an outcome. Cleared the moment the world is
            scored terminal, which is what stops the policy working on it.
        steps: Control steps at the moment it was scored.
        wall_s: Seconds from the wave's start to this world finishing. Not an
            exclusive per-episode cost -- see :class:`FleetSummary`.
    """

    world: int
    episode: Episode | None = None
    task: Any = None
    evaluator: Any = None
    settled_body_q: np.ndarray | None = None
    live: bool = True
    steps: int = 0
    success: bool = False
    reward: float = 0.0
    close_kind: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def parked(self) -> bool:
        return self.episode is None

    def finish(self, *, success: bool, reward: float, close_kind: str, steps: int, wall_s: float, info: dict) -> None:
        """Record this world's outcome and take it out of the live set.

        ``info`` is also assigned on every scoring pass while the world is live,
        because that is what a graph reads its own progress from; this freezes
        the last one as the episode's record.
        """
        self.live = False
        self.success = bool(success)
        self.reward = float(reward)
        self.close_kind = str(close_kind)
        self.steps = int(steps)
        self.wall_s = round(float(wall_s), 3)
        self.info = info
        self.metrics = dict(info.get("metrics", {}))

    def row(self, task_name: str) -> dict[str, Any]:
        """This world's results row, in the shape the serial runner writes.

        Same keys, same types, plus ``world`` -- the precedent being the pool,
        which adds ``worker`` for exactly the same reason: a row has to say which
        executor produced it or a straggler is indistinguishable from a slow
        episode.

        ``task`` is **this world's document**, falling back to ``task_name`` for
        a slot that carries none. A mixed wave of thirty instances writes thirty
        differently-labelled rows out of one model, and labelling them all with
        the job's task would collapse the cell into one document repeated.
        """
        if self.episode is None:
            raise ValueError(f"world {self.world} is parked and has no row")
        own = getattr(self.task, "full_name", "") or task_name
        return {
            "task": own,
            "episode_id": self.episode.episode_id,
            "seed": self.episode.seed,
            "success": bool(self.success),
            "reward": float(self.reward),
            "episode_length": int(self.steps),
            "close_kind": self.close_kind or "unknown",
            # The serial row's key (``run/runner.py::rollout``): why a policy
            # asked out, not only that it did.
            "termination_reason": str(self.info.get("termination_reason", "")),
            "metrics": self.metrics,
            "wall_s": self.wall_s,
            "world": int(self.world),
            **self.extras,
        }


# -------------------------------------------------------------------- summary
@dataclass(kw_only=True)
class FleetSummary(RunSummary):
    """A batched run's rows, with the rates that overlap would otherwise break.

    :attr:`RunSummary.steps_per_s_stepping` divides total steps by the *sum* of
    per-episode wall times, which is right when episodes run back to back and
    wrong here: a wave's episodes share the same seconds, so the sum counts them
    ``W`` times over and the reported rate collapses by that factor. The fleet
    denominator is the wave wall clock instead -- the seconds that actually
    elapsed while any world was stepping.

    Attributes:
        num_worlds: Worlds per wave.
        waves: One entry per wave: episodes, steps, wall time, how they closed,
            and the utilisation that says how much of the fleet was doing work.
    """

    num_worlds: int = 1
    waves: list[dict[str, Any]] = field(default_factory=list)

    @property
    def wave_wall_s(self) -> float:
        """Seconds spent inside waves, which is where all stepping happened."""
        return sum(float(wave.get("wall_s", 0.0)) for wave in self.waves)

    @property
    def steps_per_s_stepping(self) -> float:
        """Control steps per second delivered while waves were running.

        Sums every world's own steps over the wall clock the waves actually took,
        so this is the fleet's throughput and comparing it against a serial run's
        is the speedup. Builds and the gaps between waves are out of it.
        """
        wall = self.wave_wall_s
        return self.steps / wall if wall > 0 else 0.0

    @property
    def utilization(self) -> float:
        """Fraction of world-steps that belonged to a live episode.

        A wave runs until its last world finishes, so every world that finished
        early keeps being stepped for nothing. This is that waste, measured:
        1.0 means every world worked to the last step, 0.25 means three quarters
        of the fleet's stepping was a straggler being waited on.
        """
        spent = sum(int(wave.get("world_steps", 0)) for wave in self.waves)
        capacity = sum(int(wave.get("steps", 0)) * self.num_worlds for wave in self.waves)
        return spent / capacity if capacity > 0 else 0.0


def wave_record(*, index: int, slots: list[WorldSlot], steps: int, wall_s: float, world_steps: int) -> dict[str, Any]:
    """One wave's entry for :attr:`FleetSummary.waves` and the run's provenance."""
    closed: dict[str, int] = {}
    for slot in slots:
        if slot.parked:
            continue
        kind = slot.close_kind or "unknown"
        closed[kind] = closed.get(kind, 0) + 1
    return {
        "wave": int(index),
        "episodes": sum(1 for slot in slots if not slot.parked),
        "steps": int(steps),
        "wall_s": round(float(wall_s), 2),
        "world_steps": int(world_steps),
        "utilization": round(world_steps / (steps * len(slots)), 4) if steps and slots else 0.0,
        "done_histogram": dict(sorted(closed.items(), key=lambda kv: -kv[1])),
    }


# ------------------------------------------------------------------- protocols
def waves_for(count: int, num_worlds: int) -> int:
    """How many waves ``count`` episodes need. Here so the arithmetic is named."""
    return math.ceil(count / num_worlds) if num_worlds > 0 else 0


def run_root(root: str | Path, name: str) -> Path:
    """Where a job's run directory goes, matching the serial runner's rule."""
    return Path(root) / name


# --------------------------------------------------------------------- scoring
def score_world(
    slot: WorldSlot,
    fleet: Any,
    snapshot: Any,
    *,
    step: int,
    limit: int,
    request: str = "",
) -> tuple[bool, dict[str, Any]]:
    """Decide whether one world's episode is over, and how it ended.

    The precedence is :meth:`~robosimstudio.core.env.EnvBase.evaluate`'s, per
    world: a diverged state beats everything, then a policy asking to stop, then
    the evaluator declaring success, then the step budget running out. Keeping the
    order identical is what makes a batched ``close_kind`` histogram comparable
    with a serial one -- the categories have to mean the same thing.

    The evaluator owns success and the policy does not, here as everywhere: a
    policy may say it is finished, and the world is still scored on what the
    evaluator saw.
    """
    diverged = bool(snapshot.diverged[slot.world])
    # A diverged world is scored as the serial ``Env.evaluate`` scores one:
    # the evaluator is not consulted on NaN state, and the verdict is the shared
    # ``diverged_result`` -- no success, no reward -- rather than whatever the
    # evaluator made of garbage numbers.
    result = (
        diverged_result()
        if diverged
        else fleet.evaluate(slot.world, slot.evaluator, snapshot, settled_body_q=slot.settled_body_q)
    )
    info: dict[str, Any] = {
        "success": bool(result.success),
        "reward": finite_reward(result.reward),
        "metrics": dict(result.metrics),
    }
    # Recorded on the slot here, on every pass, rather than only by `finish`.
    # A graph reads the evaluator's verdict through `sim.check_success` and
    # `sim.next_goal`, and a fleet hands it whatever is on the slot
    # (`robosimstudio_policy_agent.fleet.solution.SolutionGraphBatched._infos`). Stored
    # only when the world finished, `slot.info` carried no "metrics" for the
    # whole live episode, so `GoalCursor._satisfied` read an empty table and
    # **no goal ever looked satisfied**: a conjunctive graph re-picked its first
    # object until MAX_ATTEMPTS, abandoned every goal and exited "finished with
    # N remaining" at reward 0. Measured on `tool_sort/level1_1`, 1/1 serially
    # against 0/1 at W=2. Assigned rather than returned-and-assigned because the
    # caller forgetting is exactly what happened; the later returns mutate this
    # same dict, so the slot sees the pass's final content.
    slot.info = info

    if diverged:
        info["close_kind"] = "diverged"
        return True, info
    if request:
        info["close_kind"] = "policy_request"
        info["termination_reason"] = request
        return True, info
    # Only a task that says so ends on success (``TaskSpec.terminate_on_success``,
    # the single-world ``Env.evaluate`` rule): tool_hanging scores the wrench
    # where the graph LEAVES it, so a wave that closed the world at the first
    # ``encloses`` credited hangs the serial run went on to lose.
    # The WORLD's document, not the fleet's one task: in a mixed wave the two
    # differ, and ``terminate_on_success`` is a per-document decision (the
    # hanging suite states false because it scores the wrench where the graph
    # LEAVES it). ``slot.task`` for a mixed wave, the fleet's otherwise.
    document = slot.task if slot.task is not None else getattr(fleet, "task", None)
    if result.success and bool(getattr(document, "terminate_on_success", True)):
        info["close_kind"] = "success"
        return True, info
    if limit and step >= limit:
        info["close_kind"] = "truncated"
        return True, info
    return False, info


class RunnerFleet:
    """Runs a job's episodes in waves and writes the run directory.

    The serial runner's shape rotated ninety degrees. It resets many worlds,
    steps them together, scores each one as it finishes, and writes one row per
    episode -- the same rows, in the same order, with two keys added. What it does
    *not* do is change any of the decisions: the evaluator still owns success, the
    policy still only requests termination, and the categories an episode can
    close in are the ones :func:`score_world` enumerates.

    Args:
        job: What to run. ``job.num_worlds`` decides the wave width.
        root: Where run directories go. The run lands in ``root/<job.name>``.
        verbose: Print one line per episode.
    """

    _fleet: Any = None
    _policy: Any = None
    """What ``run_episodes`` ended with -- not what opened it, since a wave whose
    benches differ rebuilds."""

    def __init__(self, job, *, root: str | Path = "outputs", verbose: bool = True, run_dir: Path | None = None) -> None:
        from robosimstudio.record import EpisodeStore

        self.job = job
        # ``run_dir`` names the directory outright, for a pool worker writing into
        # its own shard rather than into the job's run directory -- which the
        # parent owns and merges into.
        self.root = Path(run_dir) if run_dir is not None else run_root(root, job.name)
        self.verbose = bool(verbose)
        self.store = EpisodeStore(self.root)
        self.rows: list[dict[str, Any]] = []
        self.num_worlds = max(1, int(getattr(job, "num_worlds", 1)))
        self.summary = FleetSummary(job=job, root=self.root, workers=1, num_worlds=self.num_worlds)
        self.on_wave_tick = None
        """Optional ``(fleet, slots, step) -> None``, for a live view."""
        self.recorder = None
        """Per-world recording, opened when the job asks for it."""
        self.view_port = 0
        """Non-zero opens a live grid of the wave on that port."""

    # ------------------------------------------------------------------ the run
    def run(self) -> list[dict[str, Any]]:
        import time
        from datetime import UTC, datetime

        from .jobs import job_task
        from .runner import _suite_provenance

        job = self.job
        started = datetime.now(UTC).isoformat()
        wall0 = time.perf_counter()

        task = job_task(job)
        work = self.job_work(job, task)
        # The first wave, so the model can be built for it. Planning is pure --
        # it materialises episodes and groups them -- so asking twice costs a
        # sampler call per episode and nothing else.
        planned = plan_mixed_waves(work, self.num_worlds)
        first = planned[0] if planned else None
        # Built from a document of the FIRST wave, not from ``job.task``: a
        # mixed job's ``task`` is only the first of its documents, and
        # ``FleetEnv`` resolves whichever it is given to its fleet base anyway.
        fleet, policy = self.open(
            job,
            episodes=[episode for _document, episode in first] if first else None,
            task=first[0][0] if first else None,
        )
        view = self._open_view(fleet, policy)
        self.summary.builds = [
            {
                "task": job.task,
                "device": str(fleet.model.device),
                "captured": bool(fleet.pipeline.captured),
                "capture_error": fleet.pipeline.capture_error,
                "substeps": int(fleet.pipeline.substeps),
                "control_hz": float(fleet.pipeline.control_hz),
                "num_worlds": self.num_worlds,
                "ik_tier": fleet.ik_plan.tier,
                "build_s": round(time.perf_counter() - wall0, 2),
            }
        ]
        waves0 = time.perf_counter()

        self._fleet, self._policy = fleet, policy
        try:
            self.run_work(fleet, policy, work)
        finally:
            # ``run_episodes`` may have rebuilt the fleet for a later wave's
            # benches; what has to be closed and reported on is what it ended
            # with, not what this frame opened.
            fleet, policy = self._fleet, self._policy
            self.summary.wall_s = time.perf_counter() - waves0
            step_mix = {
                "steps_replayed": int(fleet.pipeline.steps_replayed),
                "steps_eager": int(fleet.pipeline.steps_eager),
                "recapture_failures": int(fleet.pipeline.recapture_failures),
                "capture_error": fleet.pipeline.capture_error,
            }
            policy.close()
            fleet.close()
            if self.recorder is not None:
                self.recorder.close()
            if view is not None:
                view.close()

        self._write_meta(fleet, policy, started, wall0, step_mix, _suite_provenance)
        if self.verbose:
            n = len(self.rows)
            wins = sum(1 for row in self.rows if row.get("success"))
            print(
                f"[fleet] {job.name}: {wins}/{n} success ({wins / n * 100 if n else 0:.0f}%) "
                f"over {len(self.summary.waves)} wave(s), {self.summary.utilization * 100:.0f}% utilization -> {self.root}"
            )
        return self.rows

    def run_episodes(self, fleet, policy, task, episode_ids: list[int]) -> list[dict[str, Any]]:
        """Run a set of episodes in waves on an already-built fleet.

        Split out from :meth:`run` so a pool worker can build a fleet once and
        keep pulling work into it -- the same economy the worker already gets
        from reusing an environment across units, and the reason the model is
        not rebuilt per wave.

        One document's episodes; :meth:`run_work` is the general form.

        Returns:
            The rows produced, in episode order.
        """
        return self.run_work(fleet, policy, [(task, int(i)) for i in episode_ids])

    def run_work(self, fleet, policy, work: Sequence[tuple[Any, int]]) -> list[dict[str, Any]]:
        """Run ``(document, episode id)`` work in waves on an already-built fleet.

        The wave a document lands in is decided by :func:`fleet_model_key`, so a
        mixed wave is not a special case here: several instances of one superset
        key together and are chunked exactly as several episodes of one document
        are.

        Returns:
            The rows produced, in the order the waves ran.
        """
        first = len(self.rows)
        planned = plan_mixed_waves(list(work), self.num_worlds)
        for index, members in enumerate(planned, start=len(self.summary.waves)):
            episodes = [episode for _document, episode in members]
            # A scene that puts part of a world's layout in at build time is
            # built for ONE wave's benches. Reusing it for the next wave's would
            # run this wave's episodes on the last wave's model -- the exact
            # silent failure the model key used to prevent, back by a different
            # door. Rebuilding costs a build per wave and nothing else; a task
            # that places nothing answers ``built_for`` True and never rebuilds.
            if not fleet.built_for(episodes):
                fleet, policy = self._rebuild_for(fleet, policy, episodes, task=members[0][0])
                # At once rather than after the loop: a later wave that raises
                # must still leave the caller what is open, not what was closed.
                self._fleet, self._policy = fleet, policy
            self._run_wave(fleet, policy, index, members)
        self._fleet, self._policy = fleet, policy
        return self.rows[first:]

    @staticmethod
    def job_work(job, task) -> list[tuple[Any, int]]:
        """A job's ``(document, episode id)`` work, in the order it asked for it.

        Without ``JobSpec.documents`` this is the one task over
        ``job.episode_ids`` -- every job written before mixed waves existed.
        With it, it is **each named document** over those ids, which for the
        paper's instance documents (``(:count 1)``) is one episode each: thirty
        tool-sort documents become thirty world-slots that
        :func:`plan_mixed_waves` packs into three waves of twelve rather than
        thirty processes of one.
        """
        from .jobs import job_task

        names = tuple(getattr(job, "documents", ()) or ())
        if not names:
            return [(task, int(i)) for i in job.episode_ids]
        ids = [int(i) for i in job.episode_ids]
        # Resolved once per name, so every episode of one document shares one
        # spec (identity is what a wave compares) with the job's seed applied.
        documents = {name: job_task(job, name) for name in names}
        return [(documents[name], i) for name in names for i in ids]

    def _rebuild_for(self, fleet, policy, episodes: list[Episode], task=None):
        """Close this fleet and open one built for ``episodes``.

        Everything hanging off the old model goes with it -- the policy holds
        the arms, the recorder holds the shapes -- so they are re-opened
        together rather than left pointing at a model that no longer exists.

        ``task`` is a document of the wave being built for, which in a mixed
        wave is not the job's: rebuilding from ``job.task`` would put the
        rebuilt model back on the first document's, and every later wave of a
        drawn superset would run on it.
        """
        if self.verbose:
            print(f"[fleet] rebuilding for {len(episodes)} drawn bench(es)")
        policy.close()
        if self.recorder is not None:
            self.recorder.close()
        fleet.close()
        return self.open(self.job, episodes=list(episodes), task=task)

    def open(self, job=None, episodes: list[Episode] | None = None, task=None):
        """Build the fleet and its policy, warmed and ready to run waves.

        Separate from :meth:`run` so a pool worker can build once and keep
        pulling work into what it built. Building a wide fleet costs seconds,
        and rebuilding one per wave would cost more than the waves do.

        Returns:
            ``(fleet, policy)``.
        """
        from robosimstudio_bench.envs.fleet import FleetEnv

        from .jobs import job_task

        job = job or self.job
        # A fleet has no per-job action space or observation renderer to switch:
        # ``FleetEnv`` takes neither, and a job that set one would otherwise run
        # with the task's own while its rows claimed the job's.
        unsupported = {
            name: value for name in ("control_mode", "obs_renderer") if (value := getattr(job, name, "") or "")
        }
        if unsupported:
            raise ValueError(
                f"job {job.name!r} sets {unsupported}, which a batched run (num_worlds > 1) cannot honour: "
                "FleetEnv has no per-job control mode or observation renderer. Run it with num_worlds=1, "
                "or leave these empty to use the task's own"
            )
        fleet = FleetEnv(
            task if task is not None else job_task(job),
            num_worlds=self.num_worlds,
            scene=job.scene,
            embodiment=job.embodiment,
            gripper=job.gripper,
            device=job.device,
            max_steps=job.max_steps,
            eval_every=int(getattr(job, "eval_every", 1)),
            # The rig the job asked for, which used to be dropped here and still
            # written into every row and run_meta.json -- a batched run claimed a
            # renderer it never built, and a vision graph saw no cameras. Refused
            # by name when the backend has no batched form, rather than rendering
            # world 0 for every world, which runs and produces a wave of
            # identical pictures labelled with different episode ids.
            renderer=job.renderer,
            verbose=self.verbose,
            # The wave this model is for, when the caller knows it. A task that
            # draws something built into the model needs every world placed
            # before the solvers and any host engine are built on top of it --
            # see ``_place_worlds``. A task that draws nothing passes nothing
            # and gets the replicated scene it always got.
            episodes=episodes,
        )
        policy = self._load_policy(fleet)
        self.recorder = self._open_recorder(fleet)
        # Warm before timing, exactly as the serial runner does: the first reset
        # is where kernel compilation and the capture happen, so doing it here
        # keeps the first wave timed like the others.
        fleet.reset_wave([None] * self.num_worlds)
        return fleet, policy

    def _run_wave(self, fleet, policy, index: int, members: Sequence[tuple[Any, Episode]]) -> None:
        """One wave: reset, step to exhaustion, score as worlds finish.

        ``members`` are ``(document, episode)`` pairs, one per live world. Each
        slot keeps its own document, so its evaluator, its step budget, its
        ``terminate_on_success`` and the ``task`` on its row are that
        document's -- which is what lets one wave hold thirty instances of one
        superset and score each as itself.
        """
        import time

        wave0 = time.perf_counter()
        padded: list[tuple[Any, Episode] | None] = list(members) + [None] * (self.num_worlds - len(members))
        slots = [
            WorldSlot(
                world=w,
                episode=None if pair is None else pair[1],
                task=None if pair is None else pair[0],
                evaluator=None if pair is None else self._evaluator(pair[0], pair[1]),
            )
            for w, pair in enumerate(padded)
        ]
        self._check_graph(slots)
        fleet.reset_wave([slot.episode for slot in slots], [slot.task for slot in slots])

        snapshot = fleet.snapshot()
        for slot in slots:
            # This world's own start pose, the one the fleet captured at reset,
            # so the evaluator and the policy's tools measure from one baseline.
            slot.settled_body_q = fleet.settled_body_q(slot.world)
            if slot.evaluator is not None:
                slot.evaluator.reset(fleet.context(slot.world, snapshot, settled_body_q=slot.settled_body_q))
        policy.reset_wave(slots)
        if self.recorder is not None:
            self.recorder.begin_wave(slots, snapshot)

        live = np.array([not slot.parked for slot in slots], dtype=bool)
        # Per world, because a wave may hold documents with different budgets:
        # the sharps tray documents span 22 800 to 91 200 steps and tool_sort's
        # 9 600 to 28 800. One wave-wide limit truncated the generous ones early
        # or ran the tight ones long, and either way the episode length a row
        # reports would be the wave's and not the document's. The loop runs to
        # the longest; each world is truncated at its own.
        limits = [int(self.job.max_steps or getattr(slot.task, "max_steps", 0) or 10_000) for slot in slots]
        # A negative budget would skip the loop entirely and close nothing. A
        # positive one ends every live world inside the loop, on the step equal
        # to its own limit, so nothing is left live once the loop exits.
        if min(limits) <= 0:
            raise ValueError(f"max_steps must be positive, got {min(limits)}")
        limit = max(limits)
        own_limits = frozenset(limits)
        cadence = fleet.eval_every
        world_steps = 0
        step = 0

        while live.any() and step < limit:
            policy.act(live)
            fleet.flush()
            fleet.step()
            step += 1
            world_steps += int(live.sum())
            # Each world's own limit is a scoring step too, not just the cadence
            # and the wave's longest: a world whose budget falls between two
            # cadence steps would otherwise run past it and report the overrun
            # as its episode length.
            due = step % cadence == 0 or step >= limit or step in own_limits
            snapshot = fleet.snapshot() if due else None
            policy.on_step(snapshot)
            if snapshot is None:
                continue
            if self.recorder is not None:
                self.recorder.step(snapshot, slots, live)
            requests = policy.termination_requests()
            for world in np.flatnonzero(live):
                slot = slots[world]
                done, info = score_world(
                    slot, fleet, snapshot, step=step, limit=limits[world], request=requests.get(int(world), "")
                )
                if not done:
                    continue
                slot.extras = dict(policy.row_extras(int(world)))
                slot.finish(
                    success=info["success"],
                    reward=info["reward"],
                    close_kind=info["close_kind"],
                    steps=step,
                    wall_s=time.perf_counter() - wave0,
                    info=info,
                )
                live[world] = False
                if self.recorder is not None:
                    spans = getattr(policy, "raw_spans", None)
                    if callable(spans):
                        # Before the write, not after: `finish_world` is what
                        # puts this world's tape on disk, and a span handed over
                        # afterwards would belong to an episode already closed.
                        self.recorder.record_spans(int(world), spans(int(world)))
                    self.recorder.finish_world(int(world), slot, extra=slot.extras)
                # The row goes to disk now, as the serial runner writes each
                # episode's the moment it ends. A wave used to write all of its
                # rows after the last world finished, so a SIGTERM anywhere in
                # a 35-minute wave lost twelve scored episodes at once -- and
                # the study read "the runner wrote no results rows" as a broken
                # policy. Then the world itself is taken out of service.
                self._close_slot(slot, fleet, policy, index)
                retire = getattr(policy, "retire_world", None)
                if callable(retire):
                    retire(int(world))
            if self.on_wave_tick is not None:
                # A view is an observation of a run, not part of it. One that
                # raises takes itself out rather than the measurement with it --
                # losing the picture is a nuisance, losing a wave that had
                # already scored half its episodes is not.
                try:
                    self.on_wave_tick(fleet, slots, step)
                except Exception as exc:
                    print(f"[fleet] live view failed ({type(exc).__name__}: {exc}); continuing without it")
                    self.on_wave_tick = None

        self.summary.waves.append(
            wave_record(
                index=index, slots=slots, steps=step, wall_s=time.perf_counter() - wave0, world_steps=world_steps
            )
        )

    # ------------------------------------------------------------------ helpers
    def _check_graph(self, slots: list[WorldSlot]) -> None:
        """Refuse a mixed wave whose documents resolve the job's graph differently.

        One graph drives a whole wave -- ``SolutionGraphBatched`` resolves it
        once, from the fleet's task
        (``robosimstudio_policy_agent.fleet.solution._resolve_graph``) -- which is right,
        and is why the documents have to agree about what that name means.
        ``tool_hanging``'s level-1 and level-2 documents both ship
        ``gap_perception`` and ship *different* directories for it, so a wave
        that mixed them would run every world on whichever document happened to
        build the model, and report the others as that graph's failures.

        Silent otherwise, including for a job that names no graph: a routine or
        an idle wave has nothing to disagree about.
        """
        graph = str(dict(self.job.solution_kwargs).get("graph") or "")
        if not graph:
            return
        resolved: dict[str, list[str]] = {}
        for slot in slots:
            if slot.task is None:
                continue
            try:
                where = str(slot.task.graph_path(graph))
            except KeyError:
                # A document that does not declare this graph is the
                # ``--allow-undeclared-graph`` case, which the runner has
                # already ruled on; nothing to compare.
                continue
            resolved.setdefault(where, []).append(slot.task.full_name)
        if len(resolved) > 1:
            detail = "; ".join(f"{where!r} for {', '.join(sorted(names))}" for where, names in sorted(resolved.items()))
            raise ValueError(
                f"this wave's documents disagree about what graph {graph!r} is: {detail}. One graph drives "
                "a wave, so documents share a wave only when they ship the same one"
            )

    def _evaluator(self, task, episode: Episode | None = None):
        """A fresh evaluator per world per wave, built for that world's episode.

        Fresh because task evaluators latch: one instance shared across worlds
        would declare every world successful the moment any of them was, and
        reusing one across waves would carry the previous episode's verdict in.
        Per episode because a superset instance is scored on the objects it
        runs: the goal is masked to the episode's active set
        (:func:`~robosimstudio_bench.goals.context.evaluator_for`), so a
        four-object instance of an eight-object base reads 4/4 and not 4/8.
        A parked world gets a scorer too, and never uses it.
        """
        from robosimstudio_bench.goals.context import evaluator_for

        return evaluator_for(task, episode)

    def _open_recorder(self, fleet):
        """Per-world recording and capture, or ``None``.

        Recording reads only the eval snapshot; capture also asks the fleet for
        the two arrays scoring does not read. See
        :class:`~robosimstudio.record.fleet.FleetRecorder`.
        """
        if not (self.job.record.enabled or self.job.capture.enabled):
            return None
        from robosimstudio.record.fleet import FleetRecorder

        return FleetRecorder(
            self.root,
            self.job.record,
            task=fleet.task,
            num_worlds=self.num_worlds,
            fleet=fleet,
            capture=self.job.capture,
        )

    def _open_view(self, fleet, policy):
        """A live grid of the wave, or ``None``. Never fatal.

        A view is an observation of a run, not part of it: a machine without a
        browser stack should still produce the numbers. So a failure here says
        so and the run continues.
        """
        if not self.view_port:
            return None
        try:
            from robosimstudio_studio.fleet_view import FleetView

            view = FleetView(fleet, port=int(self.view_port))
            view.bind(policy)
            self.on_wave_tick = view.on_tick
            return view
        except Exception as exc:
            print(f"[fleet] could not open the live view ({type(exc).__name__}: {exc}); running headless")
            return None

    def _close_slot(self, slot, fleet, policy, index: int) -> None:
        """Write one finished world's row, the moment it is finished."""
        row = {
            **slot.row(fleet.task.full_name),
            "solution": policy.name,
            "renderer": self.job.renderer,
            "device": self.job.device,
            "wave": index,
        }
        self.store.append_result(row)
        self.rows.append(row)
        self.summary.rows.append(row)
        if self.verbose:
            print(
                f"[fleet] {row['task']} episode {slot.episode.episode_id} (world {slot.world}): "
                f"{'SUCCESS' if slot.success else 'failure':>7} "
                f"reward {slot.reward:.2f} steps {slot.steps} ({slot.close_kind})"
            )

    def _load_policy(self, fleet):
        from ..solutions.fleet import load_fleet_solution

        return load_fleet_solution(self.job.solution, fleet, **dict(self.job.solution_kwargs))

    def _write_meta(self, fleet, policy, started: str, wall0: float, step_mix: dict, suite_provenance) -> None:
        import time
        from datetime import UTC, datetime

        from robosimstudio.record import RunMeta

        job = self.job
        lo, hi = self.summary.interval
        self.store.write_run_meta(
            RunMeta(
                run_name=job.name,
                task=job.task,
                scene=fleet.task.scene,
                embodiment=fleet.task.embodiment,
                gripper=fleet.task.gripper,
                solution=policy.name,
                renderer=job.renderer,
                device=job.device,
                started=started,
                finished=datetime.now(UTC).isoformat(),
                episodes=len(self.rows),
                successes=sum(1 for row in self.rows if row.get("success")),
                # Every key a serial run writes, so one run directory reads the
                # same way whoever produced it, plus what only a fleet has.
                provenance={
                    "pooled": False,
                    "workers": 1,
                    "undeclared_graph": bool(job.allow_undeclared_graph),
                    "wall_s": round(self.summary.wall_s, 2),
                    "run_wall_s": round(time.perf_counter() - wall0, 2),
                    "episodes_per_min": round(self.summary.episodes_per_min, 2),
                    "success_rate": round(self.summary.success_rate, 4),
                    "wilson95": [round(lo, 4), round(hi, 4)],
                    "close_kinds": self.summary.close_kinds,
                    "unrun_episodes": list(self.summary.unrun),
                    "control_steps": self.summary.steps,
                    "steps_per_s": round(self.summary.steps_per_s, 1),
                    "steps_per_s_stepping": round(self.summary.steps_per_s_stepping, 1),
                    "cuda_graph": self.summary.captured,
                    "cuda_graph_steps": step_mix,
                    "builds": self.summary.builds,
                    "suite_provenance": suite_provenance(job.task),
                    # A fleet run's own facts. ``batched`` is the flag anything
                    # comparing two runs should read first: per-episode wall time
                    # means "from the wave's start", not "this episode alone".
                    "batched": True,
                    # Which documents the wave held. Without this a mixed run's
                    # directory says only its first document's name, and the
                    # rows -- which do carry their own -- would be the only
                    # place the cell's membership was recorded.
                    "documents": list(getattr(job, "documents", ()) or ()),
                    "num_worlds": self.num_worlds,
                    "eval_every": fleet.eval_every,
                    "ik_tier": fleet.ik_plan.tier,
                    "ik_reason": fleet.ik_plan.reason,
                    "utilization": round(self.summary.utilization, 4),
                    "waves": self.summary.waves,
                    "policy_provenance": dict(policy.provenance()),
                },
            )
        )
