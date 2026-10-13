# SPDX-License-Identifier: Apache-2.0
"""Recording a wave: one episode directory per world, from one read.

The single-world recorder's contract holds here unchanged and for the same
reason: **recording must not change what is recorded**. It is handed what the
step already produced and never asks the fleet for anything. In a wave that is
not merely good manners -- an extra device-to-host copy per world per step would
be the dominant cost of the run, and the recorded episodes would no longer be
evidence about the run that was measured.

So this reads exclusively from the fleet snapshot
the runner already takes for scoring. Nothing is copied that scoring did not
already copy; what a world contributes is a slice of an array that is on the
host anyway. That is what makes recording five hundred worlds cost about what
recording one does.

Two consequences worth stating:

* **State for every world, on the eval cadence.** The arrays land in the same
  ``steps.npz`` / ``privileged.npz`` / ``meta.json`` layout an episode has
  always had, so :class:`~robosimstudio.record.store.EpisodeStore`,
  ``render_offline`` and ``export_lerobot`` read a fleet run without knowing it
  was one. A wave recorded at ``eval_every`` samples the trajectory at that
  cadence, which the episode's own ``meta.json`` records.
* **Video is deliberately not here.** Rendering is per-world camera work and the
  fleet does not render yet; the blessed way to get video out of a fleet run is
  to record state and replay it through ``render_offline``, which turns "record
  only the failures" from a prediction into a query.

Like the single-world recorder, this disables itself rather than raising. A full
disk must lose the recording, not a wave that has already scored half its
episodes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.specs import RecordSpec, TaskSpec

from .capture import CAPTURE_SCHEMA, capture_manifest
from .store import EpisodeStore, episode_row

__all__ = ["FleetRecorder"]


class FleetRecorder:
    """Per-world episode recording, driven from the wave's own snapshots.

    Args:
        root: The run directory. Episodes land in ``root/<episode_id:04d>/``.
        spec: What to record.
        task: The variant, for the metadata rows.
        num_worlds: Worlds in the fleet.
    """

    def __init__(
        self,
        root: str | Path,
        spec: RecordSpec,
        *,
        task: TaskSpec | None = None,
        num_worlds: int = 1,
        fleet=None,
        capture=None,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.task = task
        self.num_worlds = int(num_worlds)
        from robosimstudio.specs.run import CaptureSpec

        self.capture_spec = capture or CaptureSpec()
        self.capture_enabled = bool(self.capture_spec.enabled)
        """Whether a raw point tape is written beside the recording.

        Capture is the one thing here that *does* ask the fleet for something.
        A point needs ``joint_qd`` and the deformables, which scoring does not
        read, so an enabled capture costs two or three device reads a tick --
        for the whole wave, not per world, which is the same read the serial
        path makes per episode. Off by default and stated here rather than
        quietly paid, because this module's whole contract is that recording
        does not change what is recorded."""

        self.enabled = bool(spec.enabled) or self.capture_enabled
        self.failures: dict[str, str] = {}
        self._steps: list[dict[str, list[np.ndarray]]] = [{} for _ in range(self.num_worlds)]
        self._priv: list[dict[str, list[np.ndarray]]] = [{} for _ in range(self.num_worlds)]
        self._anchors: list[dict[str, np.ndarray]] = [{} for _ in range(self.num_worlds)]
        """Per-episode constants written beside the privileged arrays: which
        shapes the recorded live colours address, by index and by label."""
        self._dirs: list[Path | None] = [None] * self.num_worlds
        self._slots: list[Any] = [None] * self.num_worlds
        self._wall0 = 0.0
        self._recorded = 0
        self._store = EpisodeStore(self.root)
        self._arms = self._arm_layout(fleet)
        self._grip = dict(getattr(fleet, "grippers", {}) or {})
        self._fleet = fleet
        self._capture: list[dict[str, list[np.ndarray]]] = [{} for _ in range(self.num_worlds)]
        self._events: list[list[dict[str, Any]]] = [[] for _ in range(self.num_worlds)]
        self._spans: list[list[dict[str, Any]]] = [[] for _ in range(self.num_worlds)]
        self._manifest: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _arm_layout(fleet) -> list[tuple[str, list[int]]]:
        """Each arm's coordinates, in *action-vector* order.

        Declaration order, not role order: an action vector's slots belong to
        the arms the way the model declares them, and reordering here would
        write one hand's targets into the other hand's slice.
        """
        embodiment = getattr(getattr(fleet, "scene", None), "embodiment", None)
        if embodiment is None:
            return []
        return [(name, list(index.arm_coords)) for name, index in embodiment.arms.items()]

    def _action(self, snapshot, world: int) -> np.ndarray:
        """The joint-position action this world was commanded, per arm.

        Written because it is what an episode *is*, for anything that learns
        from one: :class:`~robosimstudio.record.store.EpisodeData` reads
        its length from it, and a policy trained on these predicts exactly this
        vector. The graph commands joint targets, so the targets are the action
        -- there is nothing to reconstruct.

        Layout matches the joint-position action space: each arm's coordinates
        followed by its gripper command, arms in declaration order.
        """
        if not self._arms:
            return np.zeros(0, dtype=np.float32)
        targets = snapshot.targets(world)
        parts: list[np.ndarray] = []
        for name, coords in self._arms:
            grip = self._grip.get(name)
            command = float(grip.commands()[world]) if grip is not None else 0.0
            parts.append(np.asarray(targets[coords], dtype=np.float32))
            parts.append(np.asarray([command], dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------- wave
    def begin_wave(self, slots: list, snapshot=None) -> None:
        """Open a directory per assigned world. Parked worlds get none.

        Args:
            snapshot: The wave's start state, if the caller has it. A capture
                tape's first point is the pose the episode *began* in -- the
                single-world recorder takes one in ``begin`` for exactly that --
                and without it a tape starts one control step in, which is a
                state delta measured from the wrong end.
        """
        if not self.enabled:
            return
        self._wall0 = time.perf_counter()
        self._steps = [{} for _ in range(self.num_worlds)]
        self._priv = [{} for _ in range(self.num_worlds)]
        self._anchors = [{} for _ in range(self.num_worlds)]
        self._capture = [{} for _ in range(self.num_worlds)]
        self._events = [[] for _ in range(self.num_worlds)]
        self._spans = [[] for _ in range(self.num_worlds)]
        self._dirs = [None] * self.num_worlds
        self._slots = [None] * self.num_worlds
        for slot in slots:
            if slot.parked:
                continue
            try:
                # The same ``<run>/<episode_id:04d>`` the single-world recorder
                # writes, so every reader of a run directory finds a fleet run's
                # episodes where it has always looked for them.
                path = self._store.root / f"{slot.episode.episode_id:04d}"
                path.mkdir(parents=True, exist_ok=True)
                self._dirs[slot.world] = path
                self._slots[slot.world] = slot
                if self.capture_enabled:
                    # Per world, not per wave. The indices and geometry are the
                    # template's and identical everywhere, but the episode id,
                    # seed and sampled overrides are this world's -- and one
                    # manifest for the wave labels every tape with world zero's
                    # episode, which is a tape that lies about what it is
                    # evidence of. Caught by diffing a batched tape against a
                    # serial one, where episode 1 claimed to be episode 0.
                    self._manifest[slot.world] = capture_manifest(self._fleet, slot.episode)
                    self._events[slot.world].append(
                        {
                            "schema": CAPTURE_SCHEMA,
                            "event": "episode_start",
                            "step": int(getattr(snapshot, "step", 0) or 0),
                            "sim_time": float(getattr(snapshot, "sim_time", 0.0) or 0.0),
                        }
                    )
            except Exception as exc:
                self._disable(f"begin world {slot.world}", exc)
        if self.capture_enabled and snapshot is not None:
            live = np.zeros(self.num_worlds, dtype=bool)
            for slot in slots:
                if not slot.parked:
                    live[int(slot.world)] = True
            try:
                self._capture_points(snapshot, slots, live)
            except Exception as exc:
                self._disable("begin capture", exc)

    def step(self, snapshot, slots: list, live: np.ndarray) -> None:
        """Record one cadence tick, for every world still running.

        Reads only the snapshot. The arrays it slices are host views the runner
        took for scoring, so a world's contribution is a copy of a few hundred
        floats and nothing else.
        """
        if not self.enabled:
            return
        try:
            # The deformables, once for the wave, when the tape is privileged:
            # the serial recorder writes ``particle_q`` beside ``body_q`` and
            # the offline renderer replays exactly those two, so a wave that
            # wrote only bodies re-rendered as a robot closing on a sponge that
            # never moved -- the tape held the particles, in ``capture.npz``,
            # and the replay does not read that file.
            #
            # One read a tick, shared with capture when both want it.
            capture_due = self.capture_enabled and snapshot.step % max(1, int(self.capture_spec.every)) == 0
            has_particles = bool(int(getattr(self._fleet.model, "particle_count", 0) or 0))
            arrays = self._fleet_arrays() if capture_due or (self.spec.privileged and has_particles) else None
            particles = arrays[1] if arrays is not None and self.spec.privileged else None
            # The colours a running scene rewrites -- a stripe wiped off a bar
            # -- are state as much as a pose is, and the serial tape carries
            # them (``shape_color_live``). A wave that wrote poses and
            # particles alone re-rendered with every stripe as built, fully
            # dirty, under a hand that had scored 0.92 of it clean.
            colours = self._live_colours() if self.spec.privileged else None
            for world in np.flatnonzero(live):
                index = int(world)
                if self._dirs[index] is None:
                    continue
                self._append(index, "action", self._action(snapshot, index))
                self._append(index, "sim_time", np.float32(snapshot.sim_time))
                self._append(index, "step", np.int32(snapshot.step))
                self._append(index, "joint_target_q", np.asarray(snapshot.targets(index), dtype=np.float32))
                self._append(index, "joint_q", np.asarray(snapshot.coords(index), dtype=np.float32))
                slot = self._slots[index]
                if slot is not None:
                    self._append(index, "success", np.uint8(bool(slot.success)))
                if self.spec.privileged:
                    self._append_priv(index, "body_q", np.asarray(snapshot.body(index), dtype=np.float32))
                    if particles is not None:
                        self._append_priv(
                            index,
                            "particle_q",
                            np.asarray(snapshot.worlds.particle_view(particles, index), dtype=np.float32),
                        )
                    if colours is not None:
                        table, live_shapes, labels = colours
                        rows = live_shapes + index * int(snapshot.worlds.shapes_per_world)
                        self._append_priv(index, "shape_color_live", table[rows])
                        anchors = self._anchors[index]
                        if "shape_color_live_index" not in anchors:
                            anchors["shape_color_live_index"] = live_shapes
                            if labels is not None:
                                anchors["shape_color_live_label"] = labels
            if capture_due:
                self._capture_points(snapshot, slots, live, arrays=arrays)
            self._recorded += 1
        except Exception as exc:
            self._disable("step", exc)

    def _capture_points(self, snapshot, slots: list, live: np.ndarray, *, arrays=None) -> None:
        """One raw state point per live world. No relations or labels derived.

        The same keys the single-world tape carries, each taken from this
        world's view of the wave -- so a reader of ``capture.npz`` cannot tell
        which runner wrote it, which is the whole point: the derived channels a
        self-learning loop reads are computed from this file and must mean the
        same thing either way.
        """
        settled = {int(s.world): s.settled_body_q for s in slots if getattr(s, "settled_body_q", None) is not None}
        joint_qd, particle_q, particle_qd = arrays if arrays is not None else self._fleet_arrays()
        worlds = snapshot.worlds
        wall = time.time()
        for world in np.flatnonzero(live):
            index = int(world)
            if self._dirs[index] is None:
                continue
            point = {
                "step": np.asarray(int(snapshot.step), dtype=np.int64),
                "sim_time": np.asarray(float(snapshot.sim_time), dtype=np.float64),
                "wall_time": np.asarray(wall, dtype=np.float64),
                "body_q": np.asarray(snapshot.body(index), dtype=np.float32),
                "body_qd": np.asarray(snapshot.body_vel(index), dtype=np.float32),
                "joint_q": np.asarray(snapshot.coords(index), dtype=np.float32),
                "joint_target_q": np.asarray(snapshot.targets(index), dtype=np.float32),
            }
            start = settled.get(index)
            # The wave's start pose for this world, which is what the serial
            # tape's ``settled_body_q`` is. Before a wave has one, this step's
            # own pose is the honest answer rather than zeros.
            point["settled_body_q"] = np.asarray(start if start is not None else point["body_q"], dtype=np.float32)
            if joint_qd is not None:
                point["joint_qd"] = np.asarray(worlds.dof_view(joint_qd, index), dtype=np.float32)
            if particle_q is not None:
                point["particle_q"] = np.asarray(worlds.particle_view(particle_q, index), dtype=np.float32)
                point["particle_qd"] = np.asarray(worlds.particle_view(particle_qd, index), dtype=np.float32)
            for key, value in point.items():
                self._capture[index].setdefault(key, []).append(value.copy())

    def _live_colours(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """The stepping model's colour table, the template's live shapes, their labels.

        Once for the wave: one device read of the whole table, a few hundred
        rows of which each world then slices at its own block --
        ``live + world * shapes_per_world``, the same arithmetic the taint
        field paints them with. The anchors are the template's, which is what
        a re-render (one world, rebuilt) resolves against -- by label first,
        since a convex decomposition upstream of the decals may cut a different
        number of hulls in the rebuild, then positionally.
        """
        live = getattr(self._fleet.scene.handles, "live_shapes", None)
        table = getattr(self._fleet.model, "shape_color", None)
        if live is None or not len(live) or table is None or not hasattr(table, "numpy"):
            return None
        live = np.asarray(live, dtype=np.int32)
        colours = np.asarray(table.numpy(), dtype=np.float32)[:, :3]
        labels = getattr(self._fleet.model, "shape_label", None)
        names = None
        if labels is not None and len(labels) > int(live.max()):
            names = np.asarray([str(labels[int(s)]) for s in live])
        return colours, live, names

    def _fleet_arrays(self) -> tuple[Any, Any, Any]:
        """The arrays a point needs that scoring did not already read.

        Once for the wave. ``joint_qd`` and the deformables are not in the
        snapshot because nothing that scores a world looks at them, so this is
        the cost capture adds and the reason it is off unless asked for.
        """
        state = self._fleet.pipeline.state_0
        joint_qd = np.asarray(state.joint_qd.numpy()) if getattr(state, "joint_qd", None) is not None else None
        if not int(getattr(self._fleet.model, "particle_count", 0) or 0):
            return joint_qd, None, None
        return joint_qd, np.asarray(state.particle_q.numpy()), np.asarray(state.particle_qd.numpy())

    def record_spans(self, world: int, rows: list[dict[str, Any]]) -> None:
        """Attach one world's raw tool lifecycle rows, supplied by its policy."""
        if self.capture_enabled and self.capture_spec.tool_spans and 0 <= int(world) < self.num_worlds:
            self._spans[int(world)] = [dict(row) for row in rows]

    def finish_world(self, world: int, slot, *, extra: dict[str, Any] | None = None) -> None:
        """Write one world's arrays and metadata as its episode ends.

        Written when the world finishes rather than when the wave does, so a
        long wave held open by one straggler still has every finished episode on
        disk -- and a run that dies mid-wave keeps what it had scored.
        """
        index = int(world)
        if not self.enabled or self._dirs[index] is None:
            return
        path = self._dirs[index]
        try:
            steps = {k: np.stack(v) for k, v in self._steps[index].items() if v}
            if steps:
                np.savez_compressed(path / "steps.npz", **steps)
            priv = {k: np.stack(v) for k, v in self._priv[index].items() if v}
            if priv:
                priv.update(self._anchors[index])
                np.savez_compressed(path / "privileged.npz", **priv)
            if self.capture_enabled:
                self._events[index].append(
                    {
                        "schema": CAPTURE_SCHEMA,
                        "event": "episode_end",
                        "step": int(getattr(slot, "steps", 0) or 0),
                        "sim_time": self._sim_s(slot),
                    }
                )
                tape = {k: np.stack(v) for k, v in self._capture[index].items() if v}
                if tape:
                    np.savez_compressed(path / "capture.npz", **tape)
                (path / "capture_manifest.json").write_text(
                    json.dumps(self._manifest.get(index, {}), indent=2, default=str)
                )
                self._write_jsonl(path / "events.jsonl", self._events[index])
                self._write_jsonl(path / "spans.jsonl", self._spans[index])
            (path / "meta.json").write_text(json.dumps(self._meta(slot, extra or {}), indent=2, default=str))
        except Exception as exc:
            self._disable(f"end world {index}", exc)
        finally:
            self._steps[index] = {}
            self._priv[index] = {}
            self._capture[index] = {}
            self._events[index] = []
            self._spans[index] = []
            self._dirs[index] = None

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))

    def _task_of(self, slot) -> Any:
        """The document this world ran -- a mixed wave's is not the job's -- else the job's."""
        task = getattr(slot, "task", None)
        return task if task is not None else self.task

    def _sim_s(self, slot) -> float:
        """Simulated seconds the world ran: its control steps at its control rate."""
        physics = getattr(self._task_of(slot), "physics", None)
        hz = float(getattr(physics, "control_hz", 0.0) or 0.0)
        return int(getattr(slot, "steps", 0) or 0) / hz if hz > 0 else 0.0

    def _meta(self, slot, extra: dict[str, Any]) -> dict[str, Any]:
        """The episode's metadata: the serial recorder's row, plus ``world``."""
        index = int(slot.world)
        info = getattr(slot, "info", None) or {}
        return episode_row(
            self._task_of(slot),
            slot.episode,
            success=bool(slot.success),
            reward=float(slot.reward),
            episode_length=int(slot.steps),
            close_kind=slot.close_kind,
            termination_reason=info.get("termination_reason", ""),
            metrics=slot.metrics,
            wall_s=float(slot.wall_s),
            sim_s=self._sim_s(slot),
            # This world's recorded ticks: what the serial recorder counts per
            # episode, and what ``steps.npz`` holds.
            recorded_steps=len(self._steps[index].get("sim_time", [])),
            captured_steps=len(self._capture[index].get("step", ())) if self.capture_enabled else 0,
            # The fleet does not render; video comes from ``render_offline``.
            video_frames=0,
            extra={"world": index, **extra},
        )

    def _append(self, world: int, key: str, value) -> None:
        self._steps[world].setdefault(key, []).append(value)

    def _append_priv(self, world: int, key: str, value) -> None:
        self._priv[world].setdefault(key, []).append(value)

    def _disable(self, where: str, exc: Exception) -> None:
        """Lose the recording, not the run."""
        self.enabled = False
        self.failures[where] = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        """Report once what recording cost, if anything went wrong."""
        if self.failures:
            where, why = next(iter(self.failures.items()))
            print(f"[fleet-record] recording was disabled after {len(self.failures)} failure(s): {where}: {why}")
