# SPDX-License-Identifier: Apache-2.0
"""The native episode store: npz, JSONL and mp4, and nothing else.

This is the canonical format. It needs numpy, imageio and the standard library,
which means an episode written on a collection machine can be read anywhere --
including three years from now, by something that is not this package. Exporting
to a training format (LeRobot, RLDS) is a *converter*, shipped as an example
over this API and never as a library module, because a format that a training
stack owns is a format that changes when that stack does.

Layout::

    <run>/
      run_meta.json                   what was run, with what, on what
      episode_results.jsonl           one row per episode -- the results file
      <episode_id>/
        meta.json                     task, seed, instruction, outcome
        steps.npz                     action, joint_q, ee_pose, reward, sim_time
        privileged.npz                body_q, particle_q, bond_active, anchors
        capture_manifest.json         stable names -> simulator indices/geometry
        capture.npz                   raw state at canonical control steps
        events.jsonl                  episode point markers
        spans.jsonl                   graph tool begin/end step markers
        contacts.jsonl                raw contact pairs at captured steps
        cameras/<camera>.mp4          one video per camera
        gap/                          a graph run's trace, when there was one

``privileged.npz`` is named as a warning, not as a description: it is
ground-truth simulator state, it is what an offline re-render replays, and a
policy that reads it is not being evaluated on anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["EpisodeData", "EpisodeStore", "RunMeta"]

SCHEMA = "robosimstudio.episode.v1"
"""Schema tag written into every row and every episode's metadata.

Write-only: nothing here validates it on read, so the layout is what a reader
actually depends on and a run recorded under an older tag still opens. The tag
is provenance for whoever is looking at the files, and the thing to bump when
the *layout* changes -- which it has not.
"""


@dataclass
class RunMeta:
    """What a run was, written once at the top of the run directory."""

    schema: str = SCHEMA
    run_name: str = ""
    task: str = ""
    # Which of the task's declared pairings this run used.
    scene: str = ""
    embodiment: str = ""
    gripper: str = ""
    solution: str = ""
    renderer: str = ""
    device: str = ""
    started: str = ""
    finished: str = ""
    episodes: int = 0
    successes: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeData:
    """One episode, read back.

    Attributes:
        path: The episode directory.
        meta: Its ``meta.json``.
        steps: The per-step arrays.
        privileged: Ground-truth state arrays, empty if none was recorded.

    :meth:`capture` opens the separate raw point/span interface when the run
    enabled it. It is separate from ``steps`` because its cadence includes
    control steps made inside blocking graph tools.
    """

    path: Path
    meta: dict[str, Any]
    steps: dict[str, np.ndarray]
    privileged: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def episode_id(self) -> int:
        return int(self.meta.get("episode_id", -1))

    @property
    def success(self) -> bool:
        return bool(self.meta.get("success", False))

    @property
    def length(self) -> int:
        actions = self.steps.get("action")
        return 0 if actions is None else len(actions)

    def videos(self) -> dict[str, Path]:
        """Camera name to mp4 path, for whichever cameras were streamed."""
        camera_dir = self.path / "cameras"
        if not camera_dir.is_dir():
            return {}
        return {p.stem: p for p in sorted(camera_dir.glob("*.mp4"))}

    def capture(self):
        """Open the episode's raw point/span capture."""
        from .capture import EpisodeCapture

        return EpisodeCapture(self.path)

    def replay_steps(self) -> Iterator[dict[str, np.ndarray]]:
        """Per-step ground-truth state, for
        :class:`~robosimstudio.core.render.offline.OfflineRenderer`.

        Narrowed to the replayable channels, because ``privileged.npz`` also
        carries the episode's *anchors* -- ``<cloth>/bond_index``,
        ``<cloth>/bond_anchor``, ``<cloth>/bond_row``, ``particle_subset`` --
        which are recorded once rather than per step. Passing those on would
        take a length with them, and
        :func:`~robosimstudio.core.render.offline.steps_from_arrays` truncates
        to its shortest array: a 340-step tape peel with 45 bonds replayed as 45
        steps, which does not fail, it just re-renders the first eighth of the
        episode and stops.
        """
        from robosimstudio.core.render.offline import REPLAY_CHANNELS, steps_from_arrays

        source = {k: v for k, v in self.privileged.items() if k in REPLAY_CHANNELS}
        if "joint_q" in self.steps:
            source.setdefault("joint_q", self.steps["joint_q"])
        yield from steps_from_arrays(source)


class EpisodeStore:
    """The single read API over a run directory.

    Everything downstream -- replay, offline re-render, export, analysis -- goes
    through this rather than reading npz files directly, so the layout has one
    consumer to change if it ever moves.

    Args:
        root: The run directory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ write
    def write_run_meta(self, meta: RunMeta) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "run_meta.json"
        path.write_text(json.dumps(asdict(meta), indent=2, default=str))
        return path

    def append_result(self, row: dict[str, Any]) -> None:
        """Append one episode row to ``episode_results.jsonl``.

        Appended per episode rather than written at the end, so a run killed
        halfway still has results for everything that finished.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "episode_results.jsonl", "a") as f:
            f.write(json.dumps({"schema": SCHEMA, **row}, default=str) + "\n")

    # ------------------------------------------------------------------- read
    @property
    def run_meta(self) -> dict[str, Any]:
        path = self.root / "run_meta.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def results(self) -> list[dict[str, Any]]:
        """Every episode row, in the order they were written."""
        path = self.root / "episode_results.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def episode_ids(self) -> list[str]:
        """Every episode directory name, sorted; empty for a run not yet written."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "meta.json").exists())

    def episode(self, episode_id: int | str) -> EpisodeData:
        """One episode, with its arrays loaded."""
        path = self.root / (episode_id if isinstance(episode_id, str) else f"{int(episode_id):04d}")
        if not (path / "meta.json").exists():
            raise FileNotFoundError(f"no episode at {path}")
        meta = json.loads((path / "meta.json").read_text())
        steps = _load_npz(path / "steps.npz")
        privileged = _load_npz(path / "privileged.npz")
        return EpisodeData(path=path, meta=meta, steps=steps, privileged=privileged)

    def episodes(self) -> Iterator[EpisodeData]:
        """Every episode, lazily -- a run's arrays do not all fit at once."""
        for name in self.episode_ids():
            yield self.episode(name)

    def summary(self) -> dict[str, Any]:
        """Success rate and episode count over the run's results."""
        rows = self.results()
        successes = sum(1 for r in rows if r.get("success"))
        return {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else 0.0,
            "mean_length": float(np.mean([r.get("episode_length", 0) for r in rows])) if rows else 0.0,
        }


def episode_row(
    task: Any,
    episode: Any,
    *,
    request: Any = None,
    success: bool,
    reward: float,
    episode_length: int,
    close_kind: str,
    termination_reason: str,
    metrics: dict[str, Any],
    wall_s: float,
    sim_s: float,
    recorded_steps: int,
    captured_steps: int,
    video_frames: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One episode's results row and ``meta.json``: the same keys from every recorder.

    Provenance comes off ``task`` when there is one and off ``request``
    otherwise: a bare ``Sim`` has no task and still has a scene worth writing
    down. A request spells the place ``cell`` where a task spells it ``scene``,
    and its embodiment may be a spec rather than a name, so it is written by
    name.
    """
    source = task if task is not None else request
    embodiment = getattr(source, "embodiment", "")
    wall_s, sim_s = float(wall_s), float(sim_s)
    return {
        "schema": SCHEMA,
        "task": getattr(source, "full_name", "") if task is not None else "",
        "scene": getattr(source, "scene", "") or getattr(source, "cell", ""),
        "embodiment": str(getattr(embodiment, "name", embodiment) or ""),
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "instruction": episode.instruction,
        "model_key": episode.model_key,
        "success": bool(success),
        "reward": float(reward),
        "episode_length": int(episode_length),
        "close_kind": close_kind or "unknown",
        "termination_reason": termination_reason or "",
        "metrics": metrics,
        "wall_s": round(wall_s, 3),
        "sim_s": round(sim_s, 3),
        "realtime_factor": round(sim_s / wall_s, 3) if wall_s > 0 else 0.0,
        "recorded_steps": int(recorded_steps),
        "captured_steps": int(captured_steps),
        "video_frames": int(video_frames),
        **(extra or {}),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
