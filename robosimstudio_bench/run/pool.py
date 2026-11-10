# SPDX-License-Identifier: Apache-2.0
"""Many environments at once: one process each, one shared queue of episodes.

Processes, not worlds -- and the two compose. A graph run holds a runtime
thread, a bus and a tool registry per environment, and every ``robot.*`` tool
blocks until *its* arm converges; replicated worlds in one model share one step
call, so a wave advances at the fleet's pace while a world's graph thinks, and
that world holds still and spends budget doing it (``robosimstudio_policy_agent/fleet``
is the batched runtime, ``_FleetWorker`` below is how a pooled process runs
one). One process per environment gives each graph its own clock and its own
crash domain, and costs only a CUDA context -- which is why a sweep over many
*documents* (each its own model) fans out processes, and a wave is for many
episodes of one document.

What this adds to :class:`~robosimstudio_bench.run.runner.RunnerEval` is
scheduling, not measurement: the episodes go into a queue, each worker builds its
environment once and pulls until the queue is empty, and every rollout still goes
through the same :func:`~robosimstudio_bench.run.runner.rollout`. So a pooled
number is comparable to a serial one by construction -- which is the whole point
of measuring this way rather than with a second rollout loop.

Three things about the schedule are deliberate:

* **Pull, don't shard.** Equal contiguous shards look tidy and finish badly: a
  failure runs to the step ceiling while a success terminates early, so a fixed
  split leaves most workers idle behind the slowest one. Workers pull, so a
  worker that draws four quick successes goes back for more work.
* **An environment is built once per worker, not once per episode.** A scene
  build is tens of seconds; a reset of the same variant is free (``Env.reset``
  re-poses unless ``model_key`` changed). Units are therefore ordered job by job,
  and a worker only rebuilds when the task, the policy or the renderer changes.
* **The starts are staggered.** N processes cold-compiling the same Warp kernels
  collide on the cache's file locks and multiply peak host memory for nothing.

The measurement caveat is the same one the graph README states, and pooling makes
it louder: a graph's outcome is not pinned by the episode id, because how many
control steps elapse between tool calls depends on when the runtime thread got
scheduled. Oversubscribe the CPU and every worker's graph thread runs late --
episodes stretch, some truncate that would not have, and the ``SimBus`` watchdog
(30 s of *wall* clock, so it is contention-sensitive) starts firing. Size
:func:`run_pooled` so a worker gets a core or two, raise ``bus_timeout`` when it
cannot, and read the ``close_kind`` histogram before believing a rate.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robosimstudio.core.gpu_turn import TURN_ENV
from robosimstudio.record import EpisodeStore, RunMeta
from robosimstudio.specs import JobSpec

from .jobs import FLEET_RUNTIME, apply_runtime, job_task, runtime_provenance
from .summary import RunSummary, wilson_interval

__all__ = ["PoolResult", "Watchdog", "run_pooled", "wilson_interval"]

_POLL_S = 1.0
"""How often the parent checks whether a silent worker is still alive."""

WATCHDOG_ENV = "ROBOSIMSTUDIO_POOL_WATCHDOG"
"""The watchdog for callers that do not pass one: ``off``, or ``build=900,step=300,retries=1``.

What a run started through a vendored runner (VABAR's ``vabar bench``, which
calls :func:`run_pooled` without the ``--watchdog-*`` flags) is tuned with."""

TEST_HANG_ENV = "ROBOSIMSTUDIO_POOL_TEST_HANG"
"""Test seam: ``build:<worker>`` or ``step:<worker>``, with ``:always`` to hang every attempt.

Makes that worker sleep forever as it starts building its first unit, or at the
third control step of its first episode -- the two places a real hang has been
seen -- so the watchdog can be shown end to end through a real run. Only the
first process to hold that worker index hangs unless ``:always`` is given, so the
replacement finishes the unit. Nothing reads it but :func:`_worker`."""


@dataclass(frozen=True)
class Watchdog:
    """How long a pool worker may make no progress before it is killed and its unit retried.

    **A worker that hangs holds its unit forever, and the pool waits for it.**
    Nothing else ever re-queues work a worker took off the queue, and
    :func:`_collect` waits for a ``done`` that a hung process never sends. The
    case that made this concrete is the RTX first frame: a scene's first
    ``step`` waited 1,800 s for an OVRTX streaming client that never reported
    idle (bounded since by
    :data:`~robosimstudio.solvers.render.rtx.STREAMING_TIMEOUT_S`), and on a
    stress rig building VABAR's tool-sorting scenes 16 at a time one build in
    904 never produced its first frame at all -- 40 minutes at 9 cores, nothing
    logged, until it was killed.

    **Progress, not CPU.** Each worker stamps a shared clock when it takes a
    unit, after every control step (a step observer on the environment, a wave
    tick on a fleet) and when it sends a row. CPU cannot tell a hang from the
    longest legitimate silence, a cold shader compile: both burned 3-9 cores on
    that rig. So the budgets are wall time without a stamp:

    Attributes:
        build_s: From taking a unit to its first control step -- the scene
            build, the warming reset and the first frame, or an episode's own
            reset. 900 s: a cold RTX compile took 330-550 s to a first frame on
            that rig (warm builds 83-126 s at the median), and a retried unit
            gets twice as long, so a compile slower than anyone measured is
            killed at most once. 0 turns this phase off.
        step_s: Between control steps. 300 s, raised for a job to its graph's
            ``bus_timeout`` plus 120 s: under ``--clock lockstep`` the
            simulation stands still while a node thinks, for up to that long,
            and the graph's own watchdog ends the episode after it. 0 turns this
            phase off.
        retries: How many times a killed unit runs again on a fresh worker
            before its episodes are written as ``close_kind="watchdog"``
            error rows.

        wave_steps: A wave every world of which a graph aborted by this control
            step is one the pool does not believe yet; see :func:`_wave_aborted`.
            0 turns that off.
        wave_worlds: The narrowest wave that rule applies to.

    A batched (fleet) unit is one wave, so a kill re-runs every world in it, and
    a wave's reset or mid-unit rebuild has no stamp of its own: a fleet unit gets
    the larger of the two budgets throughout.
    """

    build_s: float = 900.0
    step_s: float = 300.0
    retries: int = 1
    wave_steps: int = 500
    wave_worlds: int = 6

    @property
    def enabled(self) -> bool:
        return self.build_s > 0 or self.step_s > 0

    @property
    def guards_waves(self) -> bool:
        return self.enabled and self.wave_steps > 0

    @classmethod
    def from_env(cls, value: str | None = None) -> Watchdog:
        """The watchdog :data:`WATCHDOG_ENV` describes, or the default when it is unset.

        Raises:
            ValueError: A value that names nothing, so a typo does not silently
                run with the defaults.
        """
        text = (os.environ.get(WATCHDOG_ENV, "") if value is None else value).strip().lower()
        if not text:
            return cls()
        if text in ("off", "0", "false", "no"):
            return cls(build_s=0.0, step_s=0.0)
        fields_: dict[str, float] = {}
        for part in text.split(","):
            key, sep, number = part.partition("=")
            key = key.strip()
            if not sep or key not in ("build", "step", "retries", "wave_steps", "wave_worlds"):
                raise ValueError(
                    f"{WATCHDOG_ENV}={text!r}: expected 'off' or 'build=S,step=S,retries=N,wave_steps=N,wave_worlds=N'"
                )
            fields_[key] = float(number)
        return cls(
            build_s=fields_.get("build", cls.build_s),
            step_s=fields_.get("step", cls.step_s),
            retries=int(fields_.get("retries", cls.retries)),
            wave_steps=int(fields_.get("wave_steps", cls.wave_steps)),
            wave_worlds=int(fields_.get("wave_worlds", cls.wave_worlds)),
        )

    def budget(self, job: JobSpec, phase: int, attempt: int) -> float:
        """Seconds a worker on ``job`` may stay silent in ``phase``, on its ``attempt``-th try (0 first); 0 is unwatched."""
        build = float(self.build_s) * (1 + max(0, attempt))
        bus = float(dict(job.solution_kwargs).get("bus_timeout", 0.0) or 0.0)
        step = float(self.step_s)
        if step > 0 and bus > 0:
            step = max(step, bus + 120.0)
        if int(getattr(job, "num_worlds", 1)) > 1:
            return max(build if self.build_s > 0 else 0.0, step)
        if phase == _STEPPING:
            return step
        if phase in (_BUILDING, _CLOSING):
            return build if self.build_s > 0 else 0.0
        return 0.0

    def provenance(self) -> dict[str, Any]:
        return {
            "build_s": self.build_s,
            "step_s": self.step_s,
            "retries": self.retries,
            "wave_steps": self.wave_steps,
            "wave_worlds": self.wave_worlds,
        }


def _aborted_early(row: dict[str, Any], steps: int) -> bool:
    """Whether a graph ended this episode by control step ``steps``, rather than the task or the clock."""
    return bool(row.get("graph_error")) and not row.get("success") and int(row.get("episode_length", 0)) <= steps


def _wave_aborted(rows: Sequence[dict[str, Any]], width: int, steps: int, worlds: int) -> bool:
    """Whether a finished wave looks like a fault of its worker rather than a result of its graph.

    **Now and then a renderer draws a material without its diffuse colour, for
    its whole life, and says nothing.** On ``cable_routing`` that material holds
    the cable and the bench: they come out black, the graph's first perception
    finds no cable, and all 15 worlds of the wave abort at control step 85 with
    "No feasible grasp relative to the fixed anchor", while the other worker's
    wave in the same bench is fine. A worker's fault takes its whole wave, and
    a graph's early failures do not line up like that:

    Of the 131 waves at least 6 wide in the pooled batched runs on disk, 123 had
    no world abort within 500 steps and 8 had every world abort, with nothing in
    between. All 8 were the worker: 4 of these bad renderers at step 85 and 4
    ``BusTimeout: render was not answered within 180 s`` at step 291. A graph's
    own failures showed up later, and are left alone by the 500: whole waves
    aborted from step 807 and from step 8,040.

    Width is the evidence, so a wave narrower than ``worlds`` never qualifies --
    a serial episode that aborts says nothing about its worker -- and neither
    does a wave whose rows do not cover its work, which raised part way and is
    reported as what it is.
    """
    return len(rows) >= max(1, worlds) and len(rows) >= width and all(_aborted_early(row, steps) for row in rows)


# What a worker is doing, as its progress slot says. Idle is never watched: it
# covers the start-up stagger and the wait on the queue.
_IDLE, _BUILDING, _STEPPING, _CLOSING = 0, 1, 2, 3
_PHASES = {_IDLE: "idle", _BUILDING: "building", _STEPPING: "stepping", _CLOSING: "closing"}
_SLOT = 3
"""Doubles per worker in the shared progress array: last stamp (monotonic s), phase, unit index."""


@dataclass(kw_only=True)
class PoolResult(RunSummary):
    """One job's pooled outcome: a :class:`RunSummary`, plus what pooling costs.

    Everything a serial run can also report is inherited, deliberately -- a
    pooled rate and a serial one are the same measurement and are read the same
    way.

    Attributes:
        pool_wall_s: Wall time of the whole pool, builds and stagger included --
            what the run actually cost, whatever it was spent on.
        wave_wall_s: For a batched job, every wave's wall time, from every
            worker that ran one. Empty for a serial job.
        watchdog: The pool's :class:`Watchdog` settings, every worker it killed
            on this job's units (``kills``) and every wave it ran twice
            (``wave_retries``), for the run's provenance.
    """

    pool_wall_s: float = 0.0
    wave_wall_s: list[float] = field(default_factory=list)
    watchdog: dict[str, Any] = field(default_factory=dict)

    @property
    def steps_per_s_stepping(self) -> float:
        """As :attr:`RunSummary.steps_per_s_stepping`, over waves for a batched job.

        A wave's rows share its seconds -- each row's ``wall_s`` runs from the
        wave's start to that world finishing -- so summing them counts one wave
        up to ``num_worlds`` times over. A batched job divides by the waves' own
        wall time instead, as :class:`~robosimstudio_bench.run.fleet.FleetSummary`
        does, times the workers that ran them side by side.
        """
        if not self.wave_wall_s:
            return super().steps_per_s_stepping
        wall = sum(self.wave_wall_s)
        return self.steps * self.workers / wall if wall > 0 else 0.0


# --------------------------------------------------------------------- parent
def run_pooled(
    jobs: JobSpec | Sequence[JobSpec],
    *,
    root: str | Path = "outputs",
    workers: int = 4,
    devices: Sequence[str] = (),
    stagger: float = 4.0,
    threads_per_worker: int = 0,
    preload: str = "",
    verbose: bool = True,
    watchdog: Watchdog | None = None,
) -> dict[str, PoolResult]:
    """Run every job's episodes across ``workers`` processes.

    Args:
        jobs: One job, or several. Units are queued job by job, so a worker
            rebuilds only when it crosses a boundary.
        root: Where run directories go. A job lands in ``root/<job.name>``, with
            its workers' episodes under ``root/<job.name>/shards/<worker>``.
        workers: Processes to run. Each holds one environment, one policy and
            -- for a graph -- one runtime thread, so this wants sizing against
            *cores*, not against episodes. Four is a safe default on a
            sixteen-core box; a graph policy starved of CPU does not merely run
            slower, it scores differently (see the module docstring).
        devices: Warp devices to spread workers over, round robin, e.g.
            ``("cuda:0", "cuda:1")``. Empty keeps each job's own ``device``.
        stagger: Seconds to delay each successive worker's first build.
        threads_per_worker: ``OMP_NUM_THREADS`` for the workers. 0 divides the
            visible cores among them, which is what keeps N processes of a
            thread-happy BLAS from oversubscribing the machine.
        preload: A job file to import inside each worker. Needed when the jobs
            name a task the *job file itself* registered: a worker is a fresh
            interpreter and only has the suites the package ships.
        verbose: Print a line per finished episode, then a summary per job.
        watchdog: When to kill a worker that has stopped making progress, and
            how often to retry its unit; see :class:`Watchdog`. ``None`` reads
            :data:`WATCHDOG_ENV`, and without it uses the defaults; a
            ``Watchdog(build_s=0, step_s=0)`` turns it off.

    Returns:
        ``job.name -> PoolResult``.
    """
    job_list = [jobs] if isinstance(jobs, JobSpec) else list(jobs)
    if not job_list:
        return {}
    # The serial worker builds ``job.task`` and nothing else, so a job naming
    # documents there would run ``job.task`` once per document under its name
    # -- refused here as ``bench_run.check_document_waves`` refuses it, so the
    # library and the CLI agree.
    serial = [job.name for job in job_list if getattr(job, "documents", ()) and int(job.num_worlds) <= 1]
    if serial:
        raise ValueError(
            "documents share a WAVE, so they need num_worlds above 1; the serial worker builds one "
            f"model from job.task and would ignore the rest. Job(s) asking for both: {', '.join(serial)}"
        )
    workers = max(1, int(workers))
    root = Path(root)

    units = _work_units(job_list)
    if not units:
        return {job.name: PoolResult(job=job, root=root / job.name, workers=workers) for job in job_list}
    workers = min(workers, len(units))
    worker_devices = [devices[i % len(devices)] if devices else "" for i in range(workers)]

    if verbose:
        tasks = ", ".join(sorted({job.task for job in job_list}))
        episodes = sum(_job_work_total(job) for job in job_list)
        widths = {int(getattr(job, "num_worlds", 1)) for job in job_list}
        # A unit is a wave when the jobs are batched and an episode when they are
        # not, and the count a reader wants is the one they asked for -- so a
        # batched run reports both, and a serial one does not repeat itself.
        shape = f"{len(units)} waves ({episodes} episodes)" if widths != {1} else f"{episodes} episodes"
        print(f"[pool] {workers} workers x {shape} of {tasks}{' on ' + ','.join(worker_devices) if devices else ''}")

    ctx = multiprocessing.get_context("spawn")
    work: Any = ctx.Queue()
    results: Any = ctx.Queue()
    # Each unit carries its index, which is how a worker's progress slot names
    # the unit it holds -- and so what the watchdog re-runs if it is killed.
    for uid, (job_index, assigned) in enumerate(units):
        work.put((job_index, assigned, uid))
    for _ in range(workers):
        work.put(None)

    watch = watchdog if watchdog is not None else Watchdog.from_env()
    progress = _progress_array(ctx, workers)
    # ``(steps, worlds, per-job off flags)``; the flags are how a job whose re-run
    # failed too tells its workers to stop asking. See :meth:`_Watcher.wave`.
    wave_off = _wave_off_array(ctx, len(job_list))
    wave_guard = (watch.wave_steps, watch.wave_worlds, wave_off) if watch.guards_waves else None
    threads = threads_per_worker or max(1, (os.cpu_count() or workers) // workers)
    # Several workers on a GPU take turns between their renders and their physics:
    # see robosimstudio.core.gpu_turn. A caller's own setting stands, off included.
    turn_dir = tempfile.mkdtemp(prefix="rss-gpu-turn-") if workers > 1 and not os.environ.get(TURN_ENV) else ""

    def spawn(index: int, *, stagger_s: float = 0.0, generation: int = 0, first: Any = None):
        """A worker process, not yet started; a replacement passes the unit it inherits as ``first``."""
        return ctx.Process(
            target=_worker,
            args=(index, job_list, str(root), work, results),
            kwargs={
                "device": worker_devices[index],
                "stagger": stagger_s,
                "preload": preload,
                "progress": progress,
                "generation": generation,
                "first": first,
                "wave_guard": wave_guard,
            },
            name=f"pool-{index}" if not generation else f"pool-{index}r{generation}",
        )

    def start(processes) -> None:
        # Set on the parent's environ so the spawned children inherit it at
        # interpreter start, which is the only point early enough to matter: a
        # thread pool sized at import time cannot be resized afterwards. Put back
        # once they have started, so a later pool sizes itself afresh and nothing
        # else this process starts inherits a cap meant for these workers. A
        # replacement the watchdog starts later gets the same cap the same way.
        saved_threads = _limit_threads(threads, override=bool(threads_per_worker))
        saved_turn = {TURN_ENV: os.environ.get(TURN_ENV)}
        if turn_dir:
            os.environ[TURN_ENV] = turn_dir
        try:
            for proc in processes:
                proc.start()
        finally:
            # A spawned child copies the environment at ``start``, so it is safe
            # to restore now.
            _restore_env(saved_threads)
            _restore_env(saved_turn)

    started = datetime.now(UTC).isoformat()
    wall0 = time.perf_counter()
    procs = [spawn(index, stagger_s=stagger * index) for index in range(workers)]
    start(procs)
    watcher = (
        _Watcher(
            watch, job_list, units, progress, spawn, start, devices=worker_devices, verbose=verbose, wave_off=wave_off
        )
        if watch.enabled
        else None
    )

    episode_total = sum(_job_work_total(job) for job in job_list)
    try:
        rows, builds, spans, waves = _collect(
            procs, results, total=episode_total, wall0=wall0, verbose=verbose, watcher=watcher
        )
    except BaseException:
        # Ctrl-C, or anything else that ends the pool early: no worker outlives
        # it. Otherwise one hung in native code survives the parent's unwinding
        # -- an MPS daemon's ``quit`` waits on it as a client, and the
        # interpreter's exit joins it for ever. Seen: SIGINT to ``vabar bench``
        # with one worker hung did nothing until that worker was SIGKILLed, and
        # the run's private MPS daemon was left behind.
        _stop(procs, grace=5.0)
        raise
    finally:
        if turn_dir:
            shutil.rmtree(turn_dir, ignore_errors=True)
    wall = time.perf_counter() - wall0
    _stop(procs, grace=30.0)

    out: dict[str, PoolResult] = {}
    for index, job in enumerate(job_list):
        job_rows = [row for job_index, row in rows if job_index == index]
        first, last = spans.get(index, (0.0, 0.0))
        kills = [kill for kill in (watcher.kills if watcher else []) if kill["job"] == index]
        result = PoolResult(
            job=job,
            root=root / job.name,
            rows=job_rows,
            wall_s=max(0.0, last - first),
            pool_wall_s=wall,
            workers=workers,
            builds=[build for job_index, build in builds if job_index == index],
            wave_wall_s=list(waves.get(index, [])),
            watchdog=(
                {**watch.provenance(), "kills": kills, "wave_retries": watcher.wave_retries.get(index, [])}
                if watcher is not None
                else {"off": True}
            ),
        )
        _write_pooled(result, started=started)
        out[job.name] = result
    return out


def _stop(procs, *, grace: float) -> None:
    """Join every worker, then TERM what is left, then KILL what ignores that.

    TERM alone was the old end of a pool, and a process blocked in a native wait
    can outlive it; KILL cannot be ignored. Never SIGUSR1 or a faulthandler
    dump, which have crashed workers before.
    """
    deadline = time.monotonic() + grace
    for proc in procs:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    for proc in procs:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)


def _progress_array(ctx, workers: int):
    """``_SLOT`` doubles per worker, shared with the workers: last stamp, phase, unit.

    Without a lock: each slot has one writer, a stamp only moves forward, and a
    stale read costs the watchdog one poll. A context without shared arrays (a
    test's stand-in) gets a plain list, which an in-process worker can still write.
    """
    make = getattr(ctx, "Array", None)
    return make("d", _SLOT * workers, lock=False) if make is not None else [0.0] * (_SLOT * workers)


def _wave_off_array(ctx, jobs: int):
    """One flag per job, set by the parent and read by the workers, so no lock; as :func:`_progress_array`."""
    make = getattr(ctx, "Array", None)
    return make("b", jobs, lock=False) if make is not None else [0] * jobs


class _Watcher:
    """The parent's half of the :class:`Watchdog`: read the stamps, kill the silent, re-run their units.

    A killed worker is replaced at once, at the same index and device, so the
    pool keeps its width and the replacement takes the ``None`` the dead one
    never reached. The unit the dead one held goes to the replacement as its
    first unit, directly, rather than back on the queue, where it would sit
    behind the ``None``s every worker stops at. A unit out of retries is written
    as ``close_kind="watchdog"`` error rows, so it reads as a result and not as
    a gap, and the replacement goes on with the queue.
    """

    def __init__(
        self, watchdog: Watchdog, jobs, units, progress, spawn, start, *, devices, verbose: bool, wave_off: Any = None
    ) -> None:
        self.watchdog = watchdog
        self.jobs = list(jobs)
        self.units = units
        self.progress = progress
        self.spawn = spawn
        self.start = start
        self.devices = list(devices)
        self.verbose = verbose
        self.tries: dict[int, int] = {}
        self.generation: dict[int, int] = {}
        self.kills: list[dict[str, Any]] = []
        self.rerun: set[tuple[int, str, int]] = set()
        """``(job index, task, episode id)`` of every unit handed to a replacement."""
        self.wave_off = wave_off if wave_off is not None else [0] * len(self.jobs)
        self.wave_retries: dict[int, list[dict[str, Any]]] = {}
        """Job index -> what :meth:`wave` did about each wave it ran twice, for the provenance."""
        self.held: dict[int, dict[str, Any]] = {}
        """Unit -> the first attempt of a wave being re-run: its rows, its walls and its provenance record."""
        self._next = 0.0

    def slot(self, index: int) -> tuple[float, int, int]:
        base = _SLOT * index
        return float(self.progress[base]), int(self.progress[base + 1]), int(self.progress[base + 2])

    def check(self, procs: list) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        """Kill and replace every worker silent past its budget.

        Returns:
            ``(rows, gone)``: error rows for units out of retries, paired with
            their job index, and how many killed workers will never send the
            ``done`` the pool counts (one that hung while closing down).
        """
        now = time.monotonic()
        if now < self._next:
            return [], 0
        self._next = now + _POLL_S
        rows: list[tuple[int, dict[str, Any]]] = []
        gone = 0
        for index, proc in enumerate(procs):
            stamp, phase, uid = self.slot(index)
            if phase == _IDLE or stamp <= 0.0 or not proc.is_alive():
                continue
            held = self.units[uid] if 0 <= uid < len(self.units) else None
            job = self.jobs[held[0]] if held is not None else self.jobs[0]
            attempt = self.tries.get(uid, 0)
            budget = self.watchdog.budget(job, phase, attempt)
            silent = now - stamp
            if budget <= 0.0 or silent <= budget:
                continue
            # Read again: a worker that finished at the deadline has moved on.
            stamp, phase, uid = self.slot(index)
            if phase == _IDLE or now - stamp <= budget:
                continue
            proc.kill()
            proc.join(timeout=30.0)
            base = _SLOT * index
            self.progress[base + 1] = _IDLE
            self.progress[base] = 0.0
            kill: dict[str, Any] = {
                "worker": index,
                "job": held[0] if held is not None else -1,
                "unit": uid,
                "phase": _PHASES.get(phase, str(phase)),
                "silent_s": round(silent, 1),
                "budget_s": round(budget, 1),
                "episodes": [[name or job.task, int(ep)] for name, ep in held[1]] if held is not None else [],
            }
            first = None
            if held is None:
                kill["action"] = "closed" if phase == _CLOSING else "replaced"
                if phase == _CLOSING:
                    # It had taken its ``None`` and was shutting down: nothing to
                    # re-run, nothing to replace, and no ``done`` to wait for.
                    self.kills.append(kill)
                    gone += 1
                    self._say(index, kill, job)
                    continue
            else:
                self.tries[uid] = attempt + 1
                if attempt < self.watchdog.retries:
                    kill["action"] = "retried"
                    first = (held[0], held[1], uid)
                    self.rerun.update((held[0], name or job.task, int(ep)) for name, ep in held[1])
                else:
                    kill["action"] = "error rows"
                    if uid in self.held:
                        # The re-run of a wave hung for good. Its first attempt
                        # is still the only result that wave has.
                        kill["action"] = "first attempt stands"
                        rows.extend((held[0], row) for row in self._settle(uid, "re-run hung"))
                    else:
                        rows.extend((held[0], self._error_row(job, index, name, ep, kill)) for name, ep in held[1])
            self.kills.append(kill)
            self._say(index, kill, job)
            self._replace(procs, index, first)
        return rows, gone

    def _replace(self, procs: list, index: int, first: Any) -> None:
        self.generation[index] = self.generation.get(index, 0) + 1
        procs[index] = self.spawn(index, generation=self.generation[index], first=first)
        self.start([procs[index]])

    def wave(
        self,
        procs: list,
        index: int,
        job_index: int,
        uid: int,
        rows: list[dict[str, Any]],
        walls: list[float],
        *,
        leaving: int = 0,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Decide what a finished wave's rows come to; returns the rows and wave times that count.

        **A wave that looks like its worker's fault is run once more, on a new
        process, and the two runs together decide.** A healthy wave passes
        straight through, and costs nothing but the check.

        **What the re-run must not do is flatter a graph that really fails.** So
        its rows only replace the first attempt's if it came back mostly clean
        -- fewer than half its worlds aborted early, which a broken graph does
        not manage and a fresh renderer does. Otherwise the first attempt's rows
        stand, exactly as if this rule did not exist, and the job stops asking:
        a graph that aborts everywhere scores what it scored before this was
        written, for the price of one extra build and wave per worker that got
        there before the first verdict came back.

        The worst it can flatter is a graph whose worlds abort independently,
        which needs every world to abort in the first attempt and most not to
        in the second. Whatever the abort rate, that replaces at most 1 wave in
        4,000 at a width of 15 and 1 in 110 at a width of 6, the narrowest
        :attr:`Watchdog.wave_worlds` lets through (the worst case is a rate
        near 70%, and no graph on disk aborts part of a wave that early: see
        :func:`_wave_aborted`).

        Args:
            procs: The workers, updated in place when ``index`` is replaced.
            index: The worker that ran the wave.
            job_index: Its job.
            uid: Its unit; below 0 for a caller's own, never re-run.
            rows: A row per world scored.
            walls: The wall times of the fleet waves that scored them.
            leaving: The pid of a worker that judged its wave by
                :func:`_wave_aborted` and left (see
                :meth:`_FleetWorker.send_wave`), 0 if it did not. It needs a
                replacement whatever is decided here, unless the watchdog
                already gave it one.
        """
        steps = self.watchdog.wave_steps
        job = self.jobs[job_index]
        unit = self.units[uid] if 0 <= uid < len(self.units) else None
        width = len(unit[1]) if unit is not None else len(rows)
        gone = bool(leaving) and getattr(procs[index], "pid", None) == leaving
        first = None
        if uid in self.held:
            early = sum(_aborted_early(row, steps) for row in rows)
            self.held[uid]["record"]["rerun"] = _wave_digest(rows, steps, width)
            # Added so a row that arrives later for this wave (the error rows of
            # a re-run that raised) cannot sit beside the one that stood.
            self.rerun.update((job_index, str(row.get("task", "")), int(row.get("episode_id", -1))) for row in rows)
            if len(rows) >= width and early * 2 < len(rows):
                self.held[uid]["record"].update(outcome="cleared", stood="rerun")
                self.held.pop(uid)
                self._say_wave(index, f"re-run of unit {uid} came back clean ({early} of {len(rows)} aborted early)")
            else:
                self.wave_off[job_index] = 1
                walls = list(self.held[uid]["walls"])
                rows = self._settle(uid, "failed again")
                self._say_wave(
                    index, f"re-run of unit {uid} failed too: its first attempt stands, and this job re-runs no more"
                )
        elif gone and unit is not None and not self.wave_off[job_index]:
            record = {
                "worker": index,
                "unit": uid,
                "shard": _shard_root(Path(), job, index, self.generation.get(index, 0)).name,
                "episodes": [[name or job.task, int(ep)] for name, ep in unit[1]],
                "first": _wave_digest(rows, steps, width),
            }
            self.wave_retries.setdefault(job_index, []).append(record)
            self.held[uid] = {"rows": list(rows), "walls": list(walls), "record": record, "job": job_index}
            self.rerun.update((job_index, str(row.get("task", "")), int(row.get("episode_id", -1))) for row in rows)
            first = (job_index, unit[1], uid)
            self._say_wave(
                index, f"every world of unit {uid} aborted early, {record['first']['error']!r}; re-running it"
            )
            rows, walls = [], []
        if gone:
            # The same replacement a kill gets; the slot goes to the replacement
            # now, so nothing of this worker may still be stamping it.
            procs[index].kill()
            procs[index].join(timeout=30.0)
            base = _SLOT * index
            self.progress[base + 1] = _IDLE
            self.progress[base] = 0.0
            self._replace(procs, index, first)
        return rows, walls

    def _settle(self, uid: int, outcome: str) -> list[dict[str, Any]]:
        """Let a wave's first attempt stand; returns its rows."""
        held = self.held.pop(uid)
        held["record"].update(outcome=outcome, stood="first")
        return held["rows"]

    def unsettled(self) -> list[tuple[int, dict[str, Any]]]:
        """The first attempts of waves whose re-run never reported: the worker died, and these are still results."""
        rows = [(held["job"], row) for uid, held in list(self.held.items()) for row in self._settle(uid, "re-run died")]
        return rows

    def _say_wave(self, index: int, what: str) -> None:
        if self.verbose:
            print(f"[pool] wave guard: w{index}: {what}", flush=True)

    def _error_row(self, job: JobSpec, index: int, name: str, episode_id: int, kill: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": name or job.task,
            "episode_id": int(episode_id),
            "success": False,
            "reward": 0.0,
            "episode_length": 0,
            "close_kind": "watchdog",
            "error": (
                f"watchdog: no progress for {kill['silent_s']:.0f} s while {kill['phase']} "
                f"(budget {kill['budget_s']:.0f} s), {self.tries.get(kill['unit'], 1)} attempt(s)"
            ),
            "solution": job.solution,
            "renderer": job.renderer,
            "worker": index,
            "device": self.devices[index] if index < len(self.devices) and self.devices[index] else job.device,
        }

    def _say(self, index: int, kill: dict[str, Any], job: JobSpec) -> None:
        if not self.verbose:
            return
        what = ", ".join(f"{task.rpartition('/')[2]} ep {ep}" for task, ep in kill["episodes"]) or "no unit"
        then = {
            "retried": "re-running the unit on a fresh worker",
            "error rows": "out of retries; writing its episodes as watchdog errors",
            "first attempt stands": "out of retries; the wave's first attempt stands",
            "replaced": "replacing it",
            "closed": "it was closing down",
        }[kill["action"]]
        print(
            f"[pool] watchdog: w{index} made no progress for {kill['silent_s']:.0f} s while {kill['phase']} "
            f"({what}; budget {kill['budget_s']:.0f} s); killed it, {then}",
            flush=True,
        )


def _wave_digest(rows: Sequence[dict[str, Any]], steps: int, width: int) -> dict[str, Any]:
    """What the provenance keeps of one attempt at a wave; the rows themselves stay in its shard."""
    errors = [str(row.get("graph_error", "")) for row in rows if row.get("graph_error")]
    lengths = [int(row.get("episode_length", 0)) for row in rows]
    return {
        "worlds": width,
        "rows": len(rows),
        "aborted_early": sum(_aborted_early(row, steps) for row in rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "steps": [min(lengths), max(lengths)] if lengths else [],
        "error": errors[0][:400] if errors else "",
    }


def _collect(procs, results, *, total: int, wall0: float, verbose: bool, watcher: _Watcher | None = None):
    """Read rows until every worker has reported done or died.

    Liveness is polled rather than trusted: a worker killed by the driver (an
    illegal access, an OOM) never sends its ``done``, and a pool that waits for
    one hangs a run that has already produced most of its data. A worker that
    is alive and hung is the ``watcher``'s (:class:`_Watcher`): ``procs`` is
    updated in place as it replaces them.

    Returns:
        ``(rows, builds, spans, waves)`` -- rows and builds each paired with the
        job index they belong to, ``spans`` mapping a job index to the
        ``(start, finish)`` of its episodes, relative to the pool's start, and
        ``waves`` mapping a batched job's index to its waves' wall times.
    """
    rows: list[tuple[int, dict[str, Any]]] = []
    builds: list[tuple[int, dict[str, Any]]] = []
    spans: dict[int, tuple[float, float]] = {}
    waves: dict[int, list[float]] = {}
    seen: set[tuple[int, str, int]] = set()
    live = len(procs)

    def take(worker: int, job_index: int, row: dict[str, Any]) -> None:
        key = (job_index, str(row.get("task", "")), int(row.get("episode_id", -1)))
        if watcher is not None and key in watcher.rerun and key in seen:
            # A worker killed just as it sent its row, and the re-run sent
            # another: the episode is counted once.
            return
        seen.add(key)
        rows.append((job_index, row))
        # A job's span runs from the moment its first episode started to the
        # moment its last one finished. Timed on arrival here rather than in
        # the row, because the row is the results file's schema and this is
        # bookkeeping about the schedule.
        done_at = time.perf_counter() - wall0
        began_at = done_at - float(row.get("wall_s", 0.0))
        first, last = spans.get(job_index, (began_at, done_at))
        spans[job_index] = (min(first, began_at), max(last, done_at))
        if verbose:
            _print_row(worker, row, done=len(rows), total=total, wall0=wall0)

    while live > 0:
        if watcher is not None:
            failed, gone = watcher.check(procs)
            live -= gone
            for job_index, row in failed:
                rows.append((job_index, row))
                seen.add((job_index, str(row.get("task", "")), int(row.get("episode_id", -1))))
                if verbose:
                    _print_row(int(row["worker"]), row, done=len(rows), total=total, wall0=wall0)
            if live <= 0:
                break
        try:
            message = results.get(timeout=_POLL_S)
        except queue.Empty:
            if not any(proc.is_alive() for proc in procs):
                break
            continue
        kind = message[0]
        if kind == "row":
            _, worker, job_index, row = message
            take(worker, job_index, row)
        elif kind == "wave":
            _, worker, job_index, uid, wave_rows, walls, leaving = message
            if watcher is not None:
                wave_rows, walls = watcher.wave(procs, worker, job_index, uid, wave_rows, walls, leaving=leaving)
            for row in wave_rows:
                take(worker, job_index, row)
            if walls:
                waves.setdefault(job_index, []).extend(float(wall) for wall in walls)
        elif kind == "ready":
            _, worker, job_index, build = message
            builds.append((job_index, {**build, "worker": worker}))
            if verbose:
                print(
                    f"[pool] w{worker} ready in {build['build_s']:.1f} s: {build['task']} on "
                    f"{build['device']}, {'cuda graph captured' if build['captured'] else 'EAGER (no capture)'}, "
                    f"{build['control_hz']:.0f} Hz x {build['substeps']} substeps"
                )
        elif kind == "done":
            live -= 1
        elif kind == "error":
            print(f"[pool] worker {message[1]} stopped: {message[2]}")
    if watcher is not None:
        for job_index, row in watcher.unsettled():
            take(-1, job_index, row)
    return rows, builds, spans, waves


def _print_row(worker: int, row: dict[str, Any], *, done: int, total: int, wall0: float) -> None:
    elapsed = max(1e-6, time.perf_counter() - wall0)
    verdict = "SUCCESS" if row.get("success") else "failure"
    # The document too, when a job holds several: a mixed wave's episodes are
    # every document's episode 0, so the id alone printed "ep 0" four times for
    # four different problems. The row already carries which one it was.
    what = f"ep {row.get('episode_id')}"
    task = str(row.get("task") or "")
    if task:
        what = f"{task.rpartition('/')[2]} {what}"
    print(
        f"[pool] w{worker} {what}: {verdict:>7} "
        f"steps {row.get('episode_length')} ({row.get('close_kind')}) "
        f"| {done}/{total} done, {60.0 * done / elapsed:.1f} ep/min"
    )


def _write_pooled(result: PoolResult, *, started: str) -> None:
    """Write the pooled run directory: the rows, the meta, and links to episodes.

    The recorded episodes live under the worker that produced them, so the
    pooled directory symlinks them into place. That is what makes it a run
    directory rather than a summary: ``EpisodeStore.episode_ids`` finds them,
    and so do ``render_offline`` and ``export_lerobot``.
    """
    store = EpisodeStore(result.root)
    result.root.mkdir(parents=True, exist_ok=True)
    for row in sorted(result.rows, key=lambda r: int(r.get("episode_id", -1))):
        store.append_result(row)
    if result.job.record.enabled:
        _link_episodes(result)
    lo, hi = result.interval
    store.write_run_meta(
        RunMeta(
            run_name=result.job.name,
            task=result.job.task,
            scene=result.job.scene,
            embodiment=result.job.embodiment,
            gripper=result.job.gripper,
            solution=str(result.rows[0].get("solution", "")) if result.rows else result.job.solution,
            renderer=result.job.renderer,
            device=",".join(sorted({str(row.get("device", "")) for row in result.rows} - {""})),
            started=started,
            finished=datetime.now(UTC).isoformat(),
            episodes=result.episodes,
            successes=result.successes,
            provenance={
                "pooled": True,
                "workers": result.workers,
                # See runner.py's twin, and JobSpec.allow_undeclared_graph.
                "undeclared_graph": bool(result.job.allow_undeclared_graph),
                "wall_s": round(result.wall_s, 2),
                "pool_wall_s": round(result.pool_wall_s, 2),
                "episodes_per_min": round(result.episodes_per_min, 2),
                "success_rate": round(result.success_rate, 4),
                "wilson95": [round(lo, 4), round(hi, 4)],
                "close_kinds": result.close_kinds,
                "unrun_episodes": list(result.unrun),
                "control_steps": result.steps,
                "steps_per_s": round(result.steps_per_s, 1),
                "steps_per_s_stepping": round(result.steps_per_s_stepping, 1),
                "cuda_graph": result.captured,
                "builds": result.builds,
                # The settings, and every worker it killed on this job's units:
                # a re-run episode's row is otherwise indistinguishable.
                "watchdog": result.watchdog,
                # What a batched job ran under is the fleet's barrier, whatever
                # the job asked for; see RunnerFleet._write_meta.
                **(FLEET_RUNTIME if result.job.num_worlds > 1 else runtime_provenance(result.job)),
            },
        )
    )


def _link_episodes(result: PoolResult) -> None:
    shards = result.root / "shards"
    if not shards.is_dir():
        return
    _set_aside_attempts(result)
    # Every episode is tried, so one failed link does not drop the rest from
    # the run directory; the failures are reported once, at the end.
    failed: list[str] = []
    total = 0
    for shard in sorted(p for p in shards.iterdir() if p.is_dir()):
        for episode in sorted(p for p in shard.iterdir() if (p / "meta.json").exists()):
            link = result.root / episode.name
            if os.path.lexists(link):
                continue
            total += 1
            try:
                link.symlink_to(Path("shards") / shard.name / episode.name, target_is_directory=True)
            except OSError as exc:
                failed.append(f"{episode}: {exc}")
    if failed:
        print(f"[pool] could not link {len(failed)} of {total} episodes; first: {failed[0]}")


def _set_aside_attempts(result: PoolResult) -> None:
    """Move the recorded episodes of each wave's discarded first attempt out of :func:`_link_episodes`'s way.

    The first shard to hold an episode wins its link, and that is the discarded
    attempt's whenever the re-run's rows stood. Moved rather than deleted: it is
    the evidence of what went wrong with that worker.
    """
    for record in result.watchdog.get("wave_retries", []):
        if record.get("stood") != "rerun":
            continue
        shard = result.root / "shards" / str(record.get("shard", ""))
        for _task, episode_id in record.get("episodes", []):
            episode = shard / f"{int(episode_id):04d}"
            if (episode / "meta.json").exists():
                (shard / "discarded").mkdir(exist_ok=True)
                episode.rename(shard / "discarded" / episode.name)


# --------------------------------------------------------------------- worker
def _work_units(job_list: Sequence[JobSpec]) -> list[tuple[int, tuple[tuple[str, int], ...]]]:
    """What a worker pulls: the ``(document, episode id)`` work of one job it should run next.

    One episode wide for a serial job, ``num_worlds`` wide for a batched one --
    because a wave is the unit a fleet resets and scores, and handing a worker
    one episode at a time would rebuild a 256-world model per episode.

    The ids themselves rather than a span, because a job may name a strided or
    listed selection (``JobSpec.episode_list``) and a span would run everything
    between them. The worker keeps the job whole -- ``_env_key`` reads every
    other field unchanged -- and iterates the work it was handed.

    The document rides beside the id, empty for ``job.task``. A job naming
    several documents (``JobSpec.documents`` -- instances of one superset) is
    the cross product of those with the ids, so a worker's unit is a wave's
    worth of *work* rather than of one document's episodes. Empty rather than
    the task's own name so a job that names none produces exactly the units it
    always did.

    Pulling stays the load balancer either way. A worker that finishes a wave
    takes the next one, so a job whose waves differ in length still spreads.
    """
    units: list[tuple[int, tuple[tuple[str, int], ...]]] = []
    for index, job in enumerate(job_list):
        ids = [int(i) for i in job.episode_ids]
        if not ids:
            continue
        names = tuple(getattr(job, "documents", ()) or ()) or ("",)
        work = [(str(name), i) for name in names for i in ids]
        width = max(1, int(getattr(job, "num_worlds", 1)))
        for start in range(0, len(work), width):
            units.append((index, tuple(work[start : start + width])))
    return units


def _job_work_total(job: JobSpec) -> int:
    """How many episodes one job runs: its ids, once per document it names."""
    return len(list(job.episode_ids)) * max(1, len(tuple(getattr(job, "documents", ()) or ())))


def _env_key(job: JobSpec) -> tuple:
    """What forces a worker to rebuild. Episodes of one job never do.

    The pairing is part of it, and was missing: a sweep of one task across
    several robots is the ordinary shape of a job file now, and a key that named
    only the task let the second pairing's episodes run in the first pairing's
    environment -- reported under the second one's name.

    So is the document set, for the same reason one step removed: a mixed job's
    fleet is built from a document of its first wave, so two jobs that agree
    about ``task`` and name different documents are asking for two models. The
    wave would be refused rather than mismeasured (``FleetEnv._check_documents``
    compares every document against the model that was built), which is why
    this is consistency rather than a live defect -- but a key that decides what
    gets built should name everything that decides it. Within one job every unit
    carries the same set, so this forces no rebuild the ordinary path did not
    already do.
    """
    return (
        job.task,
        job.documents,
        job.scene,
        job.embodiment,
        job.gripper,
        job.solution,
        job.solution_kwargs,
        job.renderer,
        # Both reach ``Env`` in ``_build``: a different control mode is a
        # different action space, and reusing the env across it ran one job's
        # actions through the other's.
        job.obs_renderer,
        job.control_mode,
        # Applied to the documents the worker resolves (``job_task``), so a
        # different seed is a different set of episodes from the same build.
        job.seed,
        job.max_steps,
        job.device,
        job.record,
        job.capture,
        job.num_worlds,
        # ``render_every`` reaches ``Env`` in ``_build``; the clock and the
        # observation mode reach the policy there, and a reused policy would
        # keep the previous job's.
        job.render_every,
        job.clock,
        job.obs_render,
    )


def _worker(
    index: int,
    jobs: list[JobSpec],
    root: str,
    work,
    results,
    *,
    device: str,
    stagger: float,
    preload: str,
    progress: Any = None,
    generation: int = 0,
    first: Any = None,
    wave_guard: Any = None,
) -> None:
    """One process: build an environment, then pull episodes until the queue dries.

    Every import that touches Warp happens inside a function rather than at
    module scope, so the parent never initialises CUDA and the whole device stays
    available to the workers.

    ``progress`` is the pool's shared stamp array (:class:`_Progress`), and a
    replacement the :class:`Watchdog` started runs ``first`` -- the unit its
    killed predecessor held -- before it pulls from the queue, into a shard of
    its own (``generation``) so a half-written episode of the dead one is never
    mixed with the re-run. ``wave_guard`` says when a batched worker should stop
    trusting itself; see :meth:`_FleetWorker.send_wave`.
    """
    from .jobs import load_jobs
    from .runner import rollout

    mark = _Progress(progress, index)
    hang = _test_hang(index, generation)
    if preload:
        load_jobs(preload)
    if stagger > 0:
        time.sleep(stagger)

    key: tuple | None = None
    env = policy = recorder = None
    store = None
    episodes = 0
    # The device the build resolved -- ``job.device`` may be "auto" -- for the rows.
    built_device = ""
    fleet_worker = _FleetWorker()
    fleet_worker.mark = mark
    fleet_worker.generation = generation
    fleet_worker.wave_guard = wave_guard
    steps = [0]

    def stepped(*_args, **_kwargs) -> None:
        # A completed control step, tool-internal ones included: progress.
        mark(_STEPPING)
        steps[0] += 1
        if hang == "step" and steps[0] == 3:
            _hang_forever()

    wall0 = time.perf_counter()
    try:
        while True:
            unit, first = (first, None) if first is not None else (work.get(), None)
            if unit is None:
                break
            # NOT ``work``: that name is the queue this worker pulls from, two
            # lines up. Binding the unit's payload to it made the *second*
            # ``work.get()`` a tuple lookup, so every worker ran exactly one
            # unit and then died with ``'tuple' object has no attribute 'get'``
            # -- after its rows were already written, which is what made it look
            # like a teardown problem rather than the loop ending early.
            # ``(job index, work, unit index)``; a unit without an index (a
            # caller that queues its own) is simply never re-run.
            job_index, assigned, *uid = unit
            # From here to the first control step is the building phase: the
            # previous environment's teardown, the build, the warming reset and
            # the first frame -- where every hang seen so far sat, between
            # episodes as well as at a cell's start.
            mark(_BUILDING, uid[0] if uid else -1)
            if hang == "build":
                _hang_forever()
            job = jobs[job_index]
            if device:
                job = replace(job, device=device)
            if job.num_worlds > 1:
                try:
                    fleet_worker.uid = uid[0] if uid else -1
                    episodes += fleet_worker.run(job, assigned, Path(root), index, results, job_index)
                except Exception as exc:
                    # As the serial path below: a wave that raised is rebuilt
                    # rather than trusted, and every episode of the unit that
                    # scored no row gets an error row instead of reading as a gap.
                    errors = fleet_worker.error_rows(job, assigned, exc, worker=index)
                    shard = EpisodeStore(_shard_root(Path(root), job, index, generation))
                    for error in errors:
                        shard.append_result(error)
                        results.put(("row", index, job_index, error))
                    episodes += len(fleet_worker.sent) + len(errors)
                    fleet_worker.close()
                mark(_IDLE, -1)
                continue
            ((_document, episode_id),) = assigned
            if key != _env_key(job):
                _teardown(env, policy, recorder)
                env = policy = recorder = None
                env, policy, recorder, build = _build(job, Path(root), index, generation)
                observe = getattr(env, "add_step_observer", None)
                if callable(observe):
                    observe(stepped)
                built_device = str(build["device"])
                store = EpisodeStore(_shard_root(Path(root), job, index, generation))
                key = _env_key(job)
                results.put(("ready", index, job_index, build))
            # The episode's own reset may rebuild the model and render a first
            # frame, so it is building time until a control step says otherwise.
            mark(_BUILDING)
            episode0 = time.perf_counter()
            try:
                result = rollout(
                    env,
                    policy,
                    episode_id=episode_id,
                    recorder=recorder,
                    max_steps=job.max_steps,
                )
                row = {**result.row, "solution": policy.name, "renderer": job.renderer}
            except Exception as exc:
                # An episode that raised may have left the device in a state the
                # next one cannot trust, so the environment is rebuilt rather
                # than reused -- and the row is written either way, because an
                # episode that errored is a result and must not read as a gap.
                row = {
                    "task": job.task,
                    "episode_id": episode_id,
                    "success": False,
                    "reward": 0.0,
                    "episode_length": 0,
                    "close_kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "solution": job.solution,
                    "renderer": job.renderer,
                }
                _teardown(env, policy, recorder)
                env = policy = recorder = None
                key = None
            row = {
                **row,
                "worker": index,
                "device": built_device or job.device,
                "wall_s": round(time.perf_counter() - episode0, 3),
            }
            if store is not None:
                store.append_result(row)
            results.put(("row", index, job_index, row))
            mark(_IDLE, -1)
            episodes += 1
    except BaseException as exc:
        # A build that fails will fail the same way for the next unit, so the
        # worker reports and exits; the parent names the episodes nobody ran.
        results.put(("error", index, f"{type(exc).__name__}: {exc}"))
    finally:
        # Closing down is watched too: a renderer that hangs in its teardown
        # would otherwise keep the pool waiting for this ``done``.
        mark(_CLOSING, -1)
        fleet_worker.close()
        _teardown(env, policy, recorder)
        results.put(("done", index, episodes, time.perf_counter() - wall0))


class _FleetWorker:
    """One worker's batched environment, kept alive across the units it pulls.

    A wave is the unit a worker takes, but the *model* is not per wave: building
    a 256-world scene costs seconds and rebuilding it per wave would dominate
    the run -- and repeated builds in one process are what make the physics
    capture fail. So this holds the fleet, the policy and the runner between
    units, exactly as the worker already holds an environment, and rebuilds only
    when the environment key says the job changed.

    The batched runner is reused whole rather than reimplemented: a pooled run
    and a single-process one have to score episodes the same way, and the way to
    guarantee that is for there to be one implementation of it.
    """

    def __init__(self) -> None:
        self.key: tuple | None = None
        self.runner: Any = None
        self.fleet: Any = None
        self.policy: Any = None
        self.task: Any = None
        self.job: Any = None
        self._docs: dict[str, Any] = {}
        # The rows the last ``run`` put on the results queue, including when it
        # raised part way: what ``error_rows`` leaves out.
        self.sent: list[dict[str, Any]] = []
        # The worker's progress slot and its watchdog generation; see ``_worker``.
        self.mark: _Progress = _Progress(None, 0)
        self.generation = 0
        self.uid = -1
        self.wave_guard: Any = None

    def _tick(self, _fleet: Any, _slots: Any, _step: int) -> None:
        """``RunnerFleet.on_wave_tick``: a wave that scores a step is making progress."""
        self.mark(_STEPPING)

    def run(
        self, job: JobSpec, work: tuple[tuple[str, int], ...], root: Path, index: int, results, job_index: int
    ) -> int:
        from .fleet import RunnerFleet, plan_mixed_waves

        self.sent = []
        key = _env_key(job)
        if key != self.key:
            build0 = time.perf_counter()
            self.close()
            self.runner = RunnerFleet(job, verbose=False, run_dir=_shard_root(root, job, index, self.generation))
            # A wave stamps progress on every scoring step. A pool worker never
            # serves a live view (``--view`` needs one process), so the hook is free.
            self.runner.on_wave_tick = self._tick
            self.runner.tick_reads_verdicts = False
            self.task = job_task(job)
            self.job = job
            # Documents resolve with the job's seed applied, so a cache filled
            # under the previous job's key does not carry over.
            self._docs = {}
            # Built for this unit's first wave, as ``RunnerFleet.run`` builds:
            # a scene that places per world at build time would otherwise be
            # built for episode 0, then thrown away by ``run_work``'s
            # ``built_for`` and built again before the first wave ran.
            planned = plan_mixed_waves(
                [(self._document(name), episode_id) for name, episode_id in work], self.runner.num_worlds
            )
            first = planned[0] if planned else None
            self.fleet, self.policy = self.runner.open(
                job,
                episodes=[episode for _document, episode in first] if first else None,
                task=first[0][0] if first else None,
            )
            self.runner._fleet, self.runner._policy = self.fleet, self.policy
            self.key = key
            # The same keys a serial worker reports, so the parent's progress
            # line and the run's ``builds`` provenance read one way whichever
            # kind of worker produced them.
            results.put(
                (
                    "ready",
                    index,
                    job_index,
                    {
                        "task": job.task,
                        "device": str(self.fleet.model.device),
                        "captured": bool(self.fleet.pipeline.captured),
                        "capture_error": self.fleet.pipeline.capture_error,
                        "substeps": int(self.fleet.pipeline.substeps),
                        "control_hz": float(self.fleet.pipeline.control_hz),
                        "num_worlds": int(job.num_worlds),
                        "build_s": round(time.perf_counter() - build0, 2),
                    },
                )
            )
        # ``(document, id)`` work, the document resolved here rather than in
        # the runner so the registry lookup is cached per worker: a mixed unit
        # names a different document per world and the same handful of names
        # over and over across a job's units.
        pairs = [(self._document(name), episode_id) for name, episode_id in work]
        first = len(self.runner.rows)
        waves0 = len(self.runner.summary.waves)
        try:
            self.runner.run_work(self.fleet, self.policy, pairs)
        finally:
            # ``run_work`` may have rebuilt for a wave's benches, closing what
            # it was handed; what this worker holds and later closes is what
            # the runner ended with.
            self.fleet = getattr(self.runner, "_fleet", None) or self.fleet
            self.policy = getattr(self.runner, "_policy", None) or self.policy
            # Whether or not a wave raised: a row is in the shard's store the
            # moment its world is scored, so one an earlier wave (or an earlier
            # world of the failing one) produced is a result and is sent.
            self.sent = self.runner.rows[first:]
            # The waves' own wall time, for ``PoolResult.steps_per_s_stepping``:
            # the rows' ``wall_s`` overlap within a wave and cannot be summed.
            walls = [float(wave.get("wall_s", 0.0)) for wave in self.runner.summary.waves[waves0:]]
            self.send_wave(results, index, job_index, work, walls, device=self._device(job))
        return len(self.sent)

    def send_wave(
        self, results, index: int, job_index: int, work: tuple[tuple[str, int], ...], walls: list[float], *, device: str
    ) -> None:
        """Send the parent a finished unit's rows, and leave if they say this worker cannot be trusted.

        Whole and in one message, because :meth:`_Watcher.wave` decides whether
        they count, and they must not be counted a row at a time before it has.

        **A worker whose wave aborted everywhere leaves at once, without closing
        anything**, exactly as a killed one does and for the same reasons: a
        renderer's teardown can hang, the parent gives its progress slot to a
        replacement straight away, and the renderer it would be reusing is the
        suspect. Rebuilding one in this process is not an option: "repeated
        builds in one process are what make the physics capture fail" (above).
        """
        rows = [{**row, "worker": index, "device": device} for row in self.sent]
        guard = self.wave_guard
        leaving = (
            guard is not None
            and self.uid >= 0
            and not guard[2][job_index]
            and _wave_aborted(rows, len(work), guard[0], guard[1])
        )
        results.put(("wave", index, job_index, self.uid, rows, walls, os.getpid() if leaving else 0))
        if leaving:
            self.mark(_IDLE, -1)
            # The queue's feeder thread holds the message until the pipe takes
            # it, and it is what ``os._exit`` would otherwise lose.
            results.close()
            results.join_thread()
            os._exit(0)

    def error_rows(
        self, job: JobSpec, work: tuple[tuple[str, int], ...], exc: Exception, *, worker: int
    ) -> list[dict[str, Any]]:
        """An error row for each ``(document, id)`` of ``work`` that :attr:`sent` has no row for.

        Matched on the row's ``task`` and ``episode_id``, as
        :attr:`RunSummary.unrun` matches them.
        """
        ran = {(str(row.get("task", "")), int(row.get("episode_id", -1))) for row in self.sent}
        rows: list[dict[str, Any]] = []
        for name, episode_id in work:
            try:
                label = getattr(job_task(job, name), "full_name", "") or name or job.task
            except Exception:
                label = name or job.task
            if (label, int(episode_id)) in ran:
                continue
            rows.append(
                {
                    "task": label,
                    "episode_id": int(episode_id),
                    "success": False,
                    "reward": 0.0,
                    "episode_length": 0,
                    "close_kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "solution": job.solution,
                    "renderer": job.renderer,
                    "worker": worker,
                    "device": self._device(job),
                }
            )
        return rows

    def _device(self, job: JobSpec) -> str:
        """The device the fleet was built on, which ``job.device`` -- possibly ``auto`` -- only asked for."""
        model = getattr(self.fleet, "model", None)
        return str(model.device) if model is not None else job.device

    def _document(self, name: str):
        """The document a unit names, or the job's own task when it names none."""
        if not name:
            return self.task
        cached = self._docs.get(name)
        if cached is None:
            cached = job_task(self.job, name)
            self._docs[name] = cached
        return cached

    def close(self) -> None:
        # Through ``_teardown``, which reports a close that raises rather than
        # raising it: this runs after a failed wave and in the worker's
        # ``finally``, and either way the worker must still forget the fleet.
        _teardown(self.fleet, self.policy, None)
        self.key = self.runner = self.fleet = self.policy = None


def _shard_root(root: Path, job: JobSpec, index: int, generation: int = 0) -> Path:
    """A worker's shard; a watchdog replacement (``generation`` above 0) writes beside its predecessor's."""
    return root / job.name / "shards" / (f"{index:02d}" if not generation else f"{index:02d}r{generation}")


class _Progress:
    """A worker's slot in the pool's shared progress array: what the :class:`Watchdog` reads.

    Calling it stamps the slot with now, and optionally sets the phase and the
    unit held. Stores only -- no lock, no queue, no syscall -- so stamping every
    control step costs nothing a step would notice. Without an array (a worker
    run in-process by a test) it does nothing.
    """

    def __init__(self, array: Any, index: int) -> None:
        self.array = array
        self.base = _SLOT * int(index)

    def __call__(self, phase: int | None = None, unit: int | None = None) -> None:
        array = self.array
        if array is None:
            return
        if phase is not None:
            array[self.base + 1] = float(phase)
        if unit is not None:
            array[self.base + 2] = float(unit)
        array[self.base] = time.monotonic()


def _test_hang(index: int, generation: int) -> str:
    """``build``, ``step`` or ``""``: where :data:`TEST_HANG_ENV` makes this worker hang."""
    spec = os.environ.get(TEST_HANG_ENV, "").strip()
    if not spec:
        return ""
    phase, _, rest = spec.partition(":")
    worker, _, always = rest.partition(":")
    if phase not in ("build", "step") or not worker.strip().isdigit() or int(worker) != int(index):
        return ""
    return phase if (generation == 0 or always.strip() == "always") else ""


def _hang_forever() -> None:
    """What :data:`TEST_HANG_ENV` asks for: block like a native wait that never returns."""
    print(f"[pool] {TEST_HANG_ENV}: this worker now hangs on purpose", flush=True)
    while True:
        time.sleep(3600.0)


def _build(job: JobSpec, root: Path, index: int, generation: int = 0):
    """Build one worker's environment, and warm it before the first episode.

    The warming reset is not spare work. ``EnvBase.reset`` is where
    :meth:`~robosimstudio.core.pipeline.ScenePipeline.warmup_and_capture` runs,
    so doing it here rather than inside the first rollout means (a) the first
    episode is timed like every other one instead of carrying kernel compilation
    and the capture, and (b) the worker can report *whether the capture took*
    before it starts. A pool is where a silent fall-back to eager hides best:
    one worker running several times slower still produces perfectly ordinary
    rows, and only the pooled rate looks off.
    """
    from robosimstudio.record import EpisodeRecorder
    from robosimstudio_bench.envs.env import Env

    from ..solutions import load_solution

    wall0 = time.perf_counter()
    # Exactly the fields ``JobSpec`` carries and ``Env`` takes. This call had
    # drifted from both: it read ``job.render_every``, ``job.renderer_kwargs``,
    # ``job.perception_renderer`` and ``job.perception_renderer_kwargs``, none
    # of which are ``JobSpec`` fields, and passed ``perception_renderer=``
    # where ``Env`` names the parameter ``obs_renderer``. So every serial
    # pooled run died on its first unit with ``'JobSpec' object has no
    # attribute 'render_every'`` and reported "N episodes never ran".
    # ``control_mode`` is passed for the same reason the others were dropped:
    # the job carries it and this is where it belongs.
    env = Env(
        job_task(job),
        scene=job.scene,
        embodiment=job.embodiment,
        gripper=job.gripper,
        renderer=job.renderer,
        obs_renderer=job.obs_renderer,
        device=job.device,
        max_steps=job.max_steps,
        control_mode=job.control_mode,
        render_every=job.render_every,
    )
    policy = load_solution(job.solution, env, **dict(job.solution_kwargs))
    recorder = (
        EpisodeRecorder(_shard_root(root, job, index, generation), job.record, task=env.task, capture=job.capture)
        if job.record.enabled or job.capture.enabled
        else None
    )
    apply_runtime(job, env, policy)
    env.reset()
    build = {
        "task": job.task,
        "device": str(env.model.device),
        "captured": bool(env.pipeline.captured),
        "capture_error": env.pipeline.capture_error,
        "substeps": int(env.pipeline.substeps),
        "control_hz": float(env.pipeline.control_hz),
        "build_s": round(time.perf_counter() - wall0, 2),
    }
    return env, policy, recorder, build


def _teardown(env, policy, recorder) -> None:
    for closeable in (policy, recorder, env):
        if closeable is not None:
            try:
                closeable.close()
            except Exception as exc:
                print(f"[pool] close failed on {type(closeable).__name__}: {exc}")


def _limit_threads(threads: int, *, override: bool) -> dict[str, str | None]:
    """Cap the BLAS/OpenMP pools the children will inherit.

    An asked-for count wins over the environment; the count this derives from the
    core count does not, so a machine whose profile already sets these keeps its
    own answer.

    Returns:
        Each variable's value before this, ``None`` where it was unset -- what
        :func:`_restore_env` puts back.
    """
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        if override:
            os.environ[name] = str(max(1, threads))
        else:
            os.environ.setdefault(name, str(max(1, threads)))
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    """Put environment variables back as :func:`_limit_threads` found them."""
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
