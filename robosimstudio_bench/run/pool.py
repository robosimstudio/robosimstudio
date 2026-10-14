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
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robosimstudio.record import EpisodeStore, RunMeta
from robosimstudio.specs import JobSpec

from .jobs import job_task
from .summary import RunSummary, wilson_interval

__all__ = ["PoolResult", "run_pooled", "wilson_interval"]

_POLL_S = 1.0
"""How often the parent checks whether a silent worker is still alive."""


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
    """

    pool_wall_s: float = 0.0
    wave_wall_s: list[float] = field(default_factory=list)

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
    for unit in units:
        work.put(unit)
    for _ in range(workers):
        work.put(None)

    started = datetime.now(UTC).isoformat()
    wall0 = time.perf_counter()
    # Set on the parent's environ so the spawned children inherit it at
    # interpreter start, which is the only point early enough to matter: a
    # thread pool sized at import time cannot be resized afterwards. Put back
    # once they have started, so a later pool sizes itself afresh and nothing
    # else this process starts inherits a cap meant for these workers.
    saved_threads = _limit_threads(
        threads_per_worker or max(1, (os.cpu_count() or workers) // workers),
        override=bool(threads_per_worker),
    )
    try:
        procs = [
            ctx.Process(
                target=_worker,
                args=(index, job_list, str(root), work, results),
                kwargs={
                    "device": worker_devices[index],
                    "stagger": stagger * index,
                    "preload": preload,
                },
                name=f"pool-{index}",
            )
            for index in range(workers)
        ]
        for proc in procs:
            proc.start()
    finally:
        # A spawned child copies the environment at ``start``, so it is safe
        # to restore now.
        _restore_env(saved_threads)

    episode_total = sum(_job_work_total(job) for job in job_list)
    rows, builds, spans, waves = _collect(procs, results, total=episode_total, wall0=wall0, verbose=verbose)
    wall = time.perf_counter() - wall0
    for proc in procs:
        proc.join(timeout=30.0)
        if proc.is_alive():
            proc.terminate()

    out: dict[str, PoolResult] = {}
    for index, job in enumerate(job_list):
        job_rows = [row for job_index, row in rows if job_index == index]
        first, last = spans.get(index, (0.0, 0.0))
        result = PoolResult(
            job=job,
            root=root / job.name,
            rows=job_rows,
            wall_s=max(0.0, last - first),
            pool_wall_s=wall,
            workers=workers,
            builds=[build for job_index, build in builds if job_index == index],
            wave_wall_s=list(waves.get(index, [])),
        )
        _write_pooled(result, started=started)
        out[job.name] = result
    return out


def _collect(procs, results, *, total: int, wall0: float, verbose: bool):
    """Read rows until every worker has reported done or died.

    Liveness is polled rather than trusted: a worker killed by the driver (an
    illegal access, an OOM) never sends its ``done``, and a pool that waits for
    one hangs a run that has already produced most of its data.

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
    live = len(procs)
    while live > 0:
        try:
            message = results.get(timeout=_POLL_S)
        except queue.Empty:
            if not any(proc.is_alive() for proc in procs):
                break
            continue
        kind = message[0]
        if kind == "row":
            _, worker, job_index, row = message
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
        elif kind == "ready":
            _, worker, job_index, build = message
            builds.append((job_index, {**build, "worker": worker}))
            if verbose:
                print(
                    f"[pool] w{worker} ready in {build['build_s']:.1f} s: {build['task']} on "
                    f"{build['device']}, {'cuda graph captured' if build['captured'] else 'EAGER (no capture)'}, "
                    f"{build['control_hz']:.0f} Hz x {build['substeps']} substeps"
                )
        elif kind == "waves":
            _, _worker_index, job_index, walls = message
            waves.setdefault(job_index, []).extend(float(wall) for wall in walls)
        elif kind == "done":
            live -= 1
        elif kind == "error":
            print(f"[pool] worker {message[1]} stopped: {message[2]}")
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
            },
        )
    )


def _link_episodes(result: PoolResult) -> None:
    shards = result.root / "shards"
    if not shards.is_dir():
        return
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
) -> None:
    """One process: build an environment, then pull episodes until the queue dries.

    Every import that touches Warp happens inside a function rather than at
    module scope, so the parent never initialises CUDA and the whole device stays
    available to the workers.
    """
    from .jobs import load_jobs
    from .runner import rollout

    if preload:
        load_jobs(preload)
    if stagger > 0:
        time.sleep(stagger)

    key: tuple | None = None
    env = policy = recorder = None
    store = None
    episodes = 0
    fleet_worker = _FleetWorker()
    wall0 = time.perf_counter()
    try:
        while True:
            unit = work.get()
            if unit is None:
                break
            # NOT ``work``: that name is the queue this worker pulls from, two
            # lines up. Binding the unit's payload to it made the *second*
            # ``work.get()`` a tuple lookup, so every worker ran exactly one
            # unit and then died with ``'tuple' object has no attribute 'get'``
            # -- after its rows were already written, which is what made it look
            # like a teardown problem rather than the loop ending early.
            job_index, assigned = unit
            job = jobs[job_index]
            if device:
                job = replace(job, device=device)
            if job.num_worlds > 1:
                try:
                    episodes += fleet_worker.run(job, assigned, Path(root), index, results, job_index)
                except Exception as exc:
                    # As the serial path below: a wave that raised is rebuilt
                    # rather than trusted, and every episode of the unit that
                    # scored no row gets an error row instead of reading as a gap.
                    errors = fleet_worker.error_rows(job, assigned, exc, worker=index)
                    shard = EpisodeStore(_shard_root(Path(root), job, index))
                    for error in errors:
                        shard.append_result(error)
                        results.put(("row", index, job_index, error))
                    episodes += len(fleet_worker.sent) + len(errors)
                    fleet_worker.close()
                continue
            ((_document, episode_id),) = assigned
            if key != _env_key(job):
                _teardown(env, policy, recorder)
                env, policy, recorder, build = _build(job, Path(root), index)
                store = EpisodeStore(_shard_root(Path(root), job, index))
                key = _env_key(job)
                results.put(("ready", index, job_index, build))
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
            row = {**row, "worker": index, "device": job.device, "wall_s": round(time.perf_counter() - episode0, 3)}
            if store is not None:
                store.append_result(row)
            results.put(("row", index, job_index, row))
            episodes += 1
    except BaseException as exc:
        # A build that fails will fail the same way for the next unit, so the
        # worker reports and exits; the parent names the episodes nobody ran.
        results.put(("error", index, f"{type(exc).__name__}: {exc}"))
    finally:
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

    def run(
        self, job: JobSpec, work: tuple[tuple[str, int], ...], root: Path, index: int, results, job_index: int
    ) -> int:
        from .fleet import RunnerFleet, plan_mixed_waves

        self.sent = []
        key = _env_key(job)
        if key != self.key:
            build0 = time.perf_counter()
            self.close()
            self.runner = RunnerFleet(job, verbose=False, run_dir=_shard_root(root, job, index))
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
            for row in self.sent:
                results.put(("row", index, job_index, {**row, "worker": index, "device": job.device}))
            # The waves' own wall time, for ``PoolResult.steps_per_s_stepping``:
            # the rows' ``wall_s`` overlap within a wave and cannot be summed.
            walls = [float(wave.get("wall_s", 0.0)) for wave in self.runner.summary.waves[waves0:]]
            if walls:
                results.put(("waves", index, job_index, walls))
        return len(self.sent)

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
                    "device": job.device,
                }
            )
        return rows

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


def _shard_root(root: Path, job: JobSpec, index: int) -> Path:
    return root / job.name / "shards" / f"{index:02d}"


def _build(job: JobSpec, root: Path, index: int):
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
    )
    policy = load_solution(job.solution, env, **dict(job.solution_kwargs))
    recorder = (
        EpisodeRecorder(_shard_root(root, job, index), job.record, task=env.task, capture=job.capture)
        if job.record.enabled or job.capture.enabled
        else None
    )
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
