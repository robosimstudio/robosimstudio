# SPDX-License-Identifier: Apache-2.0
"""What a run records, and what a job asks for.

The one spec group nothing else in the IR depends on: these describe an
*invocation* rather than a thing in a scene, which is why a job file is the
whole configuration format for a batch of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CaptureSpec", "JobSpec", "RecordSpec"]


@dataclass(frozen=True, kw_only=True)
class CaptureSpec:
    """Raw simulator evidence recorded for offline consumers.

    Capture is deliberately separate from :class:`RecordSpec`: videos are a
    presentation artefact, while this is the typed state tape from which a
    downstream system can build point observations and action spans.  Nothing
    here describes or interprets the scene.

    Attributes:
        enabled: Write the raw tape and its static name/index manifest.
        every: Record one control step in every N.  Self-learning uses 1 so a
            tool span always resolves to exact endpoints.
        contacts: Persist raw contact pairs at recorded steps when a contact
            probe is available.
        tool_spans: Record begin/end step markers for graph tools tagged
            ``sim_step``.
    """

    enabled: bool = False
    every: int = 1
    contacts: bool = True
    tool_spans: bool = True

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"capture.every must be >= 1, got {self.every}")
        if self.tool_spans and self.every != 1:
            raise ValueError(
                "capture.tool_spans requires capture.every=1 so every span has exact "
                "begin/end points; disable tool_spans to subsample the point tape"
            )


@dataclass(frozen=True, kw_only=True)
class RecordSpec:
    """What an episode recorder writes.

    Attributes:
        enabled: Whether to record at all.
        cameras: Camera names to stream as mp4. Empty records every camera the
            environment renders.
        video_fps: Frame rate of the mp4s. Frames are sampled against simulation
            time, so slow rendering or planning never changes playback speed.
        record_video: Write camera videos as well as state arrays.
        record_depth: Also write a colourised depth preview per camera.
        privileged: Record ground-truth state (``body_q``, ``particle_q``) --
            what an offline re-render replays, and what a policy must never see.
        particle_limit: Record every particle up to this many, and a
            deterministic subsample beyond it. A cloth of tens of thousands of
            particles at 50 Hz is gigabytes an episode.
        every: Record one step in ``every``. 1 records all.
        start_on_motion: Skip leading idle frames until the commanded robot
            action first changes from its reset hold target. Useful for short
            demonstration videos of policies whose perception models warm up
            before issuing a motion command.
    """

    enabled: bool = True
    cameras: tuple[str, ...] = ()
    video_fps: int = 30
    record_video: bool = True
    record_depth: bool = False
    privileged: bool = True
    particle_limit: int = 4096
    every: int = 1
    start_on_motion: bool = False


@dataclass(frozen=True, kw_only=True)
class JobSpec:
    """One evaluation job: a task, a solution, and how many episodes.

    Job files are Python and declare ``JOBS: list[JobSpec]``, which is the whole
    configuration format for a run.

    Attributes:
        name: Run directory name.
        task: ``"<suite>/<variant>"``.
        solution: Dotted path to a solution class -- a routine, a graph -- or a registered name.
        solution_kwargs: Constructor arguments for it.
        episodes: Episode ids to run, as ``(start, stop)`` half-open.
        renderer: Which backend to render with, or ``""`` for none.
        scene: Which registered place to stage the task in, or ``""`` for its
            own. Must be a pairing the task declares.
        embodiment: Which registered robot, or ``""`` for the task's own.
        gripper: Which registered hand to put on every arm, or ``""`` for the
            robot's own. Same rule.
        record: What to record.
        seed: Overrides the task's base seed (``EpisodeSpec.seed``) when not
            -1, in every runner: episode ``i`` then draws at ``seed + i``, and
            that effective seed is what each results row's ``seed`` records.
        max_steps: Overrides the task's own when not 0.
        control_mode: Overrides the task's own when not empty. Which action
            space a policy speaks belongs to the job, since it is a fact about
            the checkpoint rather than about the scene.
        capture: Raw simulator evidence to persist independently of ordinary
            recording. This is a point/span tape and never rendered language.
        allow_undeclared_graph: Run a shipped graph on a pairing whose binding
            does not claim it.

            Off by default, and the refusal is the point. A shipped graph is
            written against one robot in one place, so a run of it elsewhere is
            an *experiment* rather than a measurement -- and the two are
            indistinguishable in ``outputs/`` unless something says which one
            happened. Setting this is that saying-so: it unblocks the run and
            stamps ``undeclared_graph`` into the run's provenance, so a number
            from a pairing nobody has claimed is labelled as one in the artifact
            rather than only in whoever ran it's memory. Exploring is how a
            pairing earns its ``graphs`` entry; this is what keeps the exploring
            from being mistaken for the claim.
        device: Warp device.
        num_worlds: Episodes to run at once, in one replicated model. 1 is the
            serial runner, unchanged. Above 1 the job runs in waves of this many
            -- the benchmark tier's fleet runner. The
            rows are the same either way; the run's provenance records which
            produced them, because per-episode wall time means something
            different when episodes overlap.
        eval_every: Control steps between scoring passes in a batched run. 1
            scores every step, which is what a run being compared against a
            serial baseline wants. Higher trades up to that many steps of latency
            in noticing an episode ended for proportionally less host work.
            Ignored when ``num_worlds`` is 1.
    """

    name: str
    task: str
    scene: str = ""
    embodiment: str = ""
    gripper: str = ""
    solution: str = ""
    solution_kwargs: tuple[tuple[str, Any], ...] = ()
    episodes: tuple[int, int] = (0, 1)
    renderer: str = ""
    obs_renderer: str = ""
    record: RecordSpec = field(default_factory=RecordSpec)
    capture: CaptureSpec = field(default_factory=CaptureSpec)
    seed: int = -1
    max_steps: int = 0
    control_mode: str = ""
    allow_undeclared_graph: bool = False
    device: str = "cuda"
    num_worlds: int = 1
    eval_every: int = 1
    documents: tuple[str, ...] = ()
    """Several ``"<suite>/<variant>"`` documents to run in one job, or empty for ``task`` alone.

    What makes a **mixed-document wave**. Every named document must resolve to
    the same model -- which in practice means they all declare the same
    ``(:fleet-base X)`` -- and the job then runs ``episode_ids`` of each of
    them, packed into waves by
    the fleet runner's wave planner. Thirty instance
    documents of one superset at ``(:count 1)`` become thirty world-slots and
    three waves of twelve, where naming them one at a time is thirty processes.

    ``task`` stays the job's identity (its run directory, its provenance) and is
    conventionally the first of these. Each episode is still scored on its own
    document's goal and written with its own ``task`` in the row, so a mixed
    run's results are indistinguishable from the same documents run
    one-by-one."""
    episode_list: tuple[int, ...] = ()
    """The exact episode ids to run, when they are not the whole of ``episodes``.

    Empty for a plain range, which is what every job file writes. A benchmark
    instance set is often *strided* through its base document -- the paper's
    tool-hanging wrenches are that document's episodes 0, 4, 8, ... -- and a
    job that could only name a range would run four episodes for every one it
    wanted. ``episodes`` still brackets the list, so everything that reads the
    span for a name or a size keeps working; ``episode_ids`` is what the
    runners iterate and it prefers the list when there is one."""

    @property
    def episode_ids(self) -> range | tuple[int, ...]:
        if self.episode_list:
            return tuple(self.episode_list)
        return range(self.episodes[0], self.episodes[1])
