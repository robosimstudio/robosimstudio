# SPDX-License-Identifier: Apache-2.0
"""What a set of episode rows says, whoever produced them.

One summary type for the serial runner and the process pool, because the two are
the same measurement with different scheduling: every rollout goes through
:func:`~robosimstudio_bench.run.runner.rollout` either way, and a number that
was reported differently depending on how the episodes were scheduled would
invite exactly the comparison it cannot support.

So everything here is derived from the rows plus the little the scheduler knows
about itself -- how long it took, how many processes it used, what each of them
built. :class:`~robosimstudio_bench.run.pool.PoolResult` extends it with the
one quantity that only exists for a pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robosimstudio.specs import JobSpec

__all__ = ["RunSummary", "wilson_interval"]


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Wilson rather than normal-approximation because the rates worth reporting
    here sit near 0 or 1 with a handful of trials, which is exactly where the
    textbook interval leaves the unit interval.

    Args:
        successes: Successful trials.
        trials: Total trials.
        z: Standard-normal quantile; 1.96 is 95%.

    Returns:
        ``(lo, hi)``, both 0 when there were no trials.
    """
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(kw_only=True)
class RunSummary:
    """One job's outcome, read off its rows.

    Attributes:
        job: What was run.
        root: The run directory.
        rows: Every episode row, in the order they finished.
        wall_s: Span of *this job's* episodes, from the first one's start to the
            last one's finish. Per job rather than per run, so a run of several
            jobs does not report each of them at the whole run's rate.
        workers: How many processes ran them. 1 for a serial run.
        builds: One entry per environment that was built: its wall time, and
            whether the substep block captured into a CUDA graph.
    """

    job: JobSpec
    root: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    workers: int = 1
    builds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def episodes(self) -> int:
        return len(self.rows)

    @property
    def successes(self) -> int:
        return sum(1 for row in self.rows if row.get("success"))

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.rows else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson interval on the success rate.

        Quoted rather than implied: 3/3 and 27/32 are not the same claim, and a
        rate over a handful of episodes is mostly interval.
        """
        return wilson_interval(self.successes, self.episodes)

    @property
    def mean_length(self) -> float:
        lengths = [float(row.get("episode_length", 0)) for row in self.rows]
        return sum(lengths) / len(lengths) if lengths else 0.0

    @property
    def episodes_per_min(self) -> float:
        return 60.0 * self.episodes / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def steps(self) -> int:
        return sum(int(row.get("episode_length", 0)) for row in self.rows)

    @property
    def steps_per_s(self) -> float:
        """Aggregate control steps per second delivered on this job.

        Over the job's own span, so the resets and the gaps between episodes are
        in it and any other job's time is not. Compare it against
        :attr:`steps_per_s_stepping` to see what the gaps cost, and against a
        one-worker run's to see whether workers are actually adding throughput.
        """
        return self.steps / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def steps_per_s_stepping(self) -> float:
        """Aggregate control steps per second while episodes were running.

        Sums each episode's own wall time, so builds and stagger are out of it:
        this over :attr:`workers` is what one environment sustains, and how far
        it falls below a single process's rate is the contention. Zero when the
        rows carry no per-episode wall time.
        """
        episode_wall = sum(float(row.get("wall_s", 0.0)) for row in self.rows)
        return self.steps * self.workers / episode_wall if episode_wall > 0 else 0.0

    @property
    def captured(self) -> dict[str, int]:
        """How many environments captured a CUDA graph, and how many fell back.

        A pool is the easiest place to lose capture without noticing -- one
        worker that fell back runs several times slower and drags the rate down
        while every episode still looks fine.
        """
        counts = {"captured": 0, "eager": 0}
        for build in self.builds:
            counts["captured" if build.get("captured") else "eager"] += 1
        return counts

    @property
    def close_kinds(self) -> dict[str, int]:
        """How the episodes ended, most common first. Read this before the rate:
        a run whose failures are all ``truncated`` was starved, not beaten."""
        counts: dict[str, int] = {}
        for row in self.rows:
            kind = str(row.get("close_kind", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def unrun(self) -> tuple[int | tuple[str, int], ...]:
        """Episode ids no row reported.

        A worker that died takes its in-flight episode with it, and a rate is
        wrong if that is silent.

        A job naming ``documents`` runs every id once *per document*, so there
        the unit is ``(document, episode_id)`` -- matched against each row's
        ``task`` -- and that pair is what is reported. Matching on the id alone
        called an id run as soon as any one document had reported it.
        """
        documents = tuple(getattr(self.job, "documents", ()) or ())
        if not documents:
            ran = {int(row.get("episode_id", -1)) for row in self.rows}
            return tuple(e for e in self.job.episode_ids if e not in ran)
        done = {(str(row.get("task", "")), int(row.get("episode_id", -1))) for row in self.rows}
        return tuple((name, int(e)) for name in documents for e in self.job.episode_ids if (name, int(e)) not in done)
