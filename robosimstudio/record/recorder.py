# SPDX-License-Identifier: Apache-2.0
"""Recording an episode, without perturbing it.

The contract this class exists to honour: **recording must not change what is
recorded**. An evaluation with recording on and one with it off have to produce
the same trajectory and the same success bit, or the recorded episodes are not
evidence about the run that was measured.

Two things follow, and both are structural rather than aspirational:

* **The recorder never steps or renders the environment.** Ordinary recording
  consumes the observation/action/info already produced. Raw capture reuses the
  evaluator's ``last_context`` and takes joint/particle arrays from the step's
  shared host snapshot; that costs no copy of its own, and cannot advance
  physics or change a command.
* **Raw capture observes the environment's canonical step stream.** Blocking
  graph tools perform many control steps inside one outer policy call, so a
  rollout-loop hook would record only their endpoints. A read-only environment
  observer records those internal steps too.
* **It disables itself rather than raising.** A full disk, an mp4 encoder that
  will not start, a malformed array: any of those must lose the recording, not
  the run. Failures are counted and reported once at :meth:`close`, so a run
  that recorded nothing says so instead of looking like a run with no episodes.

Cameras stream to mp4 as they arrive rather than being buffered, so an episode's
memory cost is bounded by one frame and a run that dies mid-episode still has
the video up to that point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import newton
import numpy as np

from robosimstudio.core.render.video import VideoSet
from robosimstudio.specs import CaptureSpec, Episode, RecordSpec, TaskSpec

from .capture import CAPTURE_SCHEMA, capture_manifest
from .store import EpisodeStore, episode_row

__all__ = ["EpisodeRecorder"]


class EpisodeRecorder:
    """Streams one run's episodes into an :class:`~robosimstudio.record.EpisodeStore`.

    Args:
        root: The run directory.
        spec: What to record.
        task: Accepted and stored, **never read**. The provenance a row carries
            comes off the environment it is handed at :meth:`end` -- its task
            when it has one, its request otherwise -- which is what lets a scene
            assembled by hand record a take without inventing a task to pass
            here. Kept in the signature so the call sites stay as they are.
    """

    def __init__(
        self,
        root: str | Path,
        spec: RecordSpec,
        *,
        task: TaskSpec | None = None,
        capture: CaptureSpec | None = None,
    ) -> None:
        self.store = EpisodeStore(root)
        self.spec = spec
        self.capture_spec = capture or CaptureSpec()
        self.task = task
        self.record_enabled = bool(spec.enabled)
        self.capture_enabled = bool(self.capture_spec.enabled)
        self.enabled = self.record_enabled or self.capture_enabled
        self.failures: list[str] = []

        self._episode: Episode | None = None
        self._dir: Path | None = None
        self._videos: VideoSet | None = None
        self._steps: dict[str, list[np.ndarray]] = {}
        self._privileged: dict[str, list[np.ndarray]] = {}
        self._anchors: dict[str, np.ndarray] = {}
        self._capture: dict[str, list[np.ndarray]] = {}
        self._capture_manifest: dict[str, Any] = {}
        self._capture_events: list[dict[str, Any]] = []
        self._capture_spans: list[dict[str, Any]] = []
        self._capture_contacts: list[dict[str, Any]] = []
        self._particle_subset: np.ndarray | None = None
        self._start_wall = 0.0
        self._recorded = 0
        self._capture_env: Any = None
        self._capture_observed = False
        self._motion_started = False
        self._action_baseline: np.ndarray | None = None
        self._next_video_sim_time: float | None = None
        self._video_frames = 0

    # ---------------------------------------------------------------- lifecycle
    def begin(self, episode: Episode, env) -> None:
        """Open an episode directory and start its videos."""
        if not self.enabled:
            return
        try:
            self._episode = episode
            self._dir = self.store.root / f"{episode.episode_id:04d}"
            self._dir.mkdir(parents=True, exist_ok=True)
            self._steps = {}
            self._privileged = {}
            self._anchors = {}
            self._capture = {}
            self._capture_events = []
            self._capture_spans = []
            self._capture_contacts = []
            self._recorded = 0
            self._motion_started = not self.spec.start_on_motion
            self._action_baseline = None
            self._next_video_sim_time = None
            self._video_frames = 0
            self._start_wall = time.perf_counter()
            self._videos = (
                VideoSet(self._dir / "cameras", fps=self.spec.video_fps)
                if self.record_enabled and self.spec.record_video
                else None
            )
            self._particle_subset = self._choose_particles(env)
            if self.record_enabled:
                self._record_anchors(env)
        except Exception as exc:
            self._disable("begin", exc)
            return
        if not self.capture_enabled:
            return
        try:
            self._capture_manifest = capture_manifest(env)
            self._record_capture(env)
            self._capture_events.append(
                {
                    "schema": CAPTURE_SCHEMA,
                    "event": "episode_start",
                    "step": int(env.step_count),
                    "sim_time": float(env.sim_time),
                }
            )
            add_observer = getattr(env, "add_step_observer", None)
            if add_observer is not None:
                add_observer(self._capture_step)
                self._capture_env = env
                self._capture_observed = True
        except Exception as exc:
            self._disable_capture("begin", exc)

    def _choose_particles(self, env) -> np.ndarray | None:
        """Which particles to record, subsampling deterministically past the limit.

        A cloth of tens of thousands of particles at control rate is gigabytes
        an episode. The subsample is a fixed stride rather than a random draw,
        so two runs of the same task record the same particles and their
        trajectories are comparable.
        """
        n = env.model.particle_count
        if n == 0 or n <= self.spec.particle_limit:
            return None
        stride = int(np.ceil(n / self.spec.particle_limit))
        return np.arange(0, n, stride, dtype=np.int32)

    def _record_anchors(self, env) -> None:
        """Record the things that are constant for an episode, once.

        Bond anchors and particle-set memberships do not change during an
        episode, so writing them per step would be the largest thing in the
        file and the least informative.
        """
        entities = getattr(env.scene, "by_name", getattr(env.scene, "cloths", {}))
        for name, cloth in ((n, e) for n, e in entities.items() if getattr(e, "carries_particles", False)):
            bonds = getattr(cloth, "bonds", None)
            if bonds is not None and bonds.bond_idx is not None:
                self._anchors[f"{name}/bond_index"] = np.asarray(bonds.bond_idx.numpy(), dtype=np.int32)
                self._anchors[f"{name}/bond_anchor"] = np.asarray(bonds.bond_anchor.numpy(), dtype=np.float32)
                self._anchors[f"{name}/bond_row"] = np.asarray(bonds.rows, dtype=np.int32)
        if self._particle_subset is not None:
            self._anchors["particle_subset"] = self._particle_subset

    # -------------------------------------------------------------------- step
    def step(
        self,
        env,
        action: np.ndarray | None,
        obs: dict[str, Any],
        reward: float,
        info: dict[str, Any],
        *,
        stage: str = "",
    ) -> None:
        """Record one control step. Reads only what the step already produced."""
        if not self.enabled or self._dir is None:
            return
        # Raw capture runs on its own cadence and its own gate: it is the
        # canonical tape, so neither the demonstration recorder being off nor
        # the start-on-motion trigger may thin it. Its failures turn off capture
        # alone, and ordinary recording carries on.
        if (
            self.capture_enabled
            and not self._capture_observed
            and env.step_count % max(1, self.capture_spec.every) == 0
        ):
            try:
                self._record_capture(env)
            except Exception as exc:
                self._disable_capture("capture step", exc)
        if not self.record_enabled or env.step_count % max(1, self.spec.every):
            return
        try:
            if not self._motion_started:
                # Trigger from the commanded robot target, not measured
                # proprioception. During perception/model warm-up the controller
                # keeps sending the reset hold target, while passive gripper
                # settling can move measured joints enough to look like motion.
                # The first changed action is therefore the first intentional
                # motion command and is the frame a demonstration should begin on.
                current_action = (
                    np.zeros(0, dtype=np.float64)
                    if action is None
                    else np.asarray(action, dtype=np.float64).reshape(-1)
                )
                if self._action_baseline is None:
                    self._action_baseline = current_action.copy()
                    return
                changed = current_action.shape != self._action_baseline.shape or (
                    current_action.size > 0 and float(np.max(np.abs(current_action - self._action_baseline))) > 1.0e-4
                )
                if not changed:
                    return
                self._motion_started = True
            self._append(
                "action", np.zeros(0, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
            )
            self._append("reward", np.float32(reward))
            self._append("sim_time", np.float32(env.sim_time))
            self._append("success", np.uint8(bool(info.get("success"))))
            proprio = obs.get("proprio") or {}
            for key, value in proprio.items():
                self._append(key.replace("/", "_"), np.asarray(value, dtype=np.float32))
            for key, value in (obs.get("objects") or {}).items():
                self._append(f"obj_{key}", np.asarray(value, dtype=np.float32))

            if self.spec.privileged:
                self._record_privileged(env)
            frames = obs.get("video_cameras") or obs.get("cameras")
            if frames and self._videos is not None and self._video_frame_due(env.sim_time):
                selected = (
                    frames if not self.spec.cameras else {k: v for k, v in frames.items() if k in self.spec.cameras}
                )
                self._videos.append_frames(_overlay_stage(selected, stage), depth=self.spec.record_depth)
                self._video_frames += 1
            self._recorded += 1
        except Exception as exc:
            self._disable("step", exc)

    def _video_frame_due(self, sim_time: float) -> bool:
        """Sample video on the simulation clock, never on wall time.

        A 60 Hz controller written directly into a 30 FPS container otherwise
        plays at half speed. Planning and model inference do not advance
        ``sim_time``, so they naturally contribute no frames or playback pause.
        """
        fps = max(1, int(self.spec.video_fps))
        now = float(sim_time)
        if self._next_video_sim_time is None:
            self._next_video_sim_time = now + 1.0 / fps
            return True
        if now + 1.0e-9 < self._next_video_sim_time:
            return False
        # Advance the deadline on its own grid, not from this frame's time:
        # anchoring to ``now`` would emit one frame every ceil(hz / fps) steps
        # (50 Hz into 30 FPS gives 25 FPS, played back 1.2x fast). A step that
        # skips several deadlines still emits one frame, and the deadline jumps
        # to the first grid point after it, so there is never a burst.
        period = 1.0 / fps
        missed = int(np.floor((now + 1.0e-9 - self._next_video_sim_time) / period)) + 1
        self._next_video_sim_time += missed * period
        return True

    def _capture_step(self, env, action, obs, reward: float, info: dict[str, Any]) -> None:
        """Environment observer for the canonical stream, including tool steps."""
        del action, obs, reward, info
        if not self.enabled or not self.capture_enabled:
            return
        if env.step_count % max(1, self.capture_spec.every):
            return
        try:
            self._record_capture(env)
        except Exception as exc:
            self._disable_capture("capture step", exc)

    def _record_capture(self, env) -> None:
        """Append one raw state point. No relations or labels are derived."""
        context = getattr(env, "last_context", None) or env.make_context()
        # The step's host copy, already read for the evaluator: no device read here.
        snap = env.pipeline.snapshot
        values: dict[str, np.ndarray] = {
            "step": np.asarray(int(env.step_count), dtype=np.int64),
            "sim_time": np.asarray(float(env.sim_time), dtype=np.float64),
            "wall_time": np.asarray(time.time(), dtype=np.float64),
            "body_q": np.asarray(context.body_q, dtype=np.float32),
            "body_qd": np.asarray(context.body_qd, dtype=np.float32),
            "settled_body_q": np.asarray(context.settled_body_q, dtype=np.float32),
            "joint_q": np.asarray(snap.joint_q, dtype=np.float32),
            "joint_qd": np.asarray(snap.joint_qd, dtype=np.float32),
            "joint_target_q": np.asarray(context.joint_target_q, dtype=np.float32),
        }
        if env.model.particle_count:
            values["particle_q"] = np.asarray(snap.particle_q, dtype=np.float32)
            values["particle_qd"] = np.asarray(snap.particle_qd, dtype=np.float32)
        for key, value in values.items():
            self._capture.setdefault(key, []).append(value.copy())

        # ``CaptureSpec.contacts`` is the switch: it was also an environment
        # variable while a bisect attributed peel's CUDA 700 to this snapshot
        # (it calls mujoco-warp's contact_force during film tearing).
        if self.capture_spec.contacts and context.contacts is not None:
            # A restricted :class:`ContactView` (a parked shape was in contact)
            # has no ``update``: it was cut from a probe the env already
            # refreshed for this step before building the context.
            if hasattr(context.contacts, "update"):
                context.contacts.update(env.step_count)
            shape_a, shape_b = context.contacts._shape_pairs()
            pairs = [
                {
                    "shape_a": int(shape_a[i]),
                    "shape_b": int(shape_b[i]),
                    "impulse": [float(v) for v in context.contacts.impulses[i]],
                    "distance": float(context.contacts.depths[i]),
                }
                for i in range(len(shape_a))
            ]
            if pairs:
                self._capture_contacts.append({"step": int(env.step_count), "pairs": pairs})

    def record_spans(self, rows: list[dict[str, Any]]) -> None:
        """Attach raw tool lifecycle rows supplied by a policy."""
        if self.capture_enabled and self.capture_spec.tool_spans:
            self._capture_spans = [dict(row) for row in rows]

    def _record_privileged(self, env) -> None:
        state: newton.State = env.state
        self._append_priv("body_q", np.asarray(state.body_q.numpy(), dtype=np.float32))
        if env.model.particle_count:
            particle_q = np.asarray(state.particle_q.numpy(), dtype=np.float32)
            if self._particle_subset is not None:
                particle_q = particle_q[self._particle_subset]
            self._append_priv("particle_q", particle_q)
        for name, cloth in ((n, e) for n, e in env.scene.by_name.items() if e.carries_particles):
            bonds = getattr(cloth, "bonds", None)
            if bonds is not None and bonds.bond_active is not None:
                self._append_priv(f"{name}_bond_active", np.asarray(bonds.bond_active.numpy(), dtype=np.uint8))
        # The colours a running scene rewrites -- a mark polished off, a seam
        # parting -- are state as much as a pose is, and the only state a
        # re-render through a backend that bakes colour at load cannot recover
        # from the poses. Recorded for the live shapes alone: a few hundred
        # rows, not the whole model's table every step.
        live = getattr(env.scene.handles, "live_shapes", None)
        colors = getattr(env.model, "shape_color", None)
        if live is not None and len(live) and colors is not None and hasattr(colors, "numpy"):
            if "shape_color_live_index" not in self._anchors:
                self._anchors["shape_color_live_index"] = np.asarray(live, dtype=np.int32)
                # By label as well as by index: a shape's index is not stable
                # across builds of the same task (a convex decomposition
                # upstream of the decals may cut a different number of hulls),
                # its label is, and the re-render rebuilds the scene.
                labels = getattr(env.model, "shape_label", None)
                if labels is not None and len(labels) > int(np.max(live)):
                    self._anchors["shape_color_live_label"] = np.asarray([str(labels[int(s)]) for s in live])
            self._append_priv("shape_color_live", np.asarray(colors.numpy(), dtype=np.float32)[live, :3])

    def _append(self, key: str, value: np.ndarray) -> None:
        self._steps.setdefault(key, []).append(value)

    def _append_priv(self, key: str, value: np.ndarray) -> None:
        self._privileged.setdefault(key, []).append(value)

    # --------------------------------------------------------------------- end
    def end(self, env, info: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Close the episode, write its arrays, and return its results row.

        The row is returned even when recording is disabled, so a run without
        recording still produces results -- the row is the *measurement*, the
        arrays are the evidence.
        """
        self._detach_capture_observer()
        wall = time.perf_counter() - self._start_wall if self._start_wall else 0.0
        row = self._results_row(env, info, wall, extra or {})
        if not self.enabled or self._dir is None:
            self._episode = None
            return row
        try:
            if self.capture_enabled:
                try:
                    self._end_capture(env)
                except Exception as exc:
                    self._disable_capture("end", exc)
                # The row is also an index into the evidence. Rebuild it after
                # adding a missing terminal endpoint so ``captured_steps`` is
                # the number actually written to capture.npz.
                row = self._results_row(env, info, wall, extra or {})
            if self._videos is not None:
                self._videos.close()
                self._videos = None
            if self.record_enabled:
                np.savez_compressed(self._dir / "steps.npz", **_stack_steps(self._steps))
            if self.record_enabled and (self._privileged or self._anchors):
                arrays = {k: np.stack(v) for k, v in self._privileged.items() if v}
                arrays.update(self._anchors)
                np.savez_compressed(self._dir / "privileged.npz", **arrays)
            if self.capture_enabled:
                try:
                    self._write_capture(self._dir)
                except Exception as exc:
                    self._disable_capture("end", exc)
                    row = self._results_row(env, info, wall, extra or {})
            (self._dir / "meta.json").write_text(json.dumps(row, indent=2, default=str))
        except Exception as exc:
            self._disable("end", exc)
        finally:
            self._steps, self._privileged, self._anchors, self._capture = {}, {}, {}, {}
            self._dir = None
            self._episode = None
        return row

    def _end_capture(self, env) -> None:
        recorded_steps = [int(v) for v in self._capture.get("step", [])]
        if not recorded_steps or recorded_steps[-1] != int(env.step_count):
            self._record_capture(env)
        self._capture_events.append(
            {
                "schema": CAPTURE_SCHEMA,
                "event": "episode_end",
                "step": int(env.step_count),
                "sim_time": float(env.sim_time),
            }
        )

    def _write_capture(self, directory: Path) -> None:
        np.savez_compressed(directory / "capture.npz", **{k: np.stack(v) for k, v in self._capture.items() if v})
        (directory / "capture_manifest.json").write_text(json.dumps(self._capture_manifest, indent=2, default=str))
        self._write_jsonl(directory / "events.jsonl", self._capture_events)
        self._write_jsonl(directory / "spans.jsonl", self._capture_spans)
        self._write_jsonl(directory / "contacts.jsonl", self._capture_contacts)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))

    def _results_row(self, env, info: dict[str, Any], wall: float, extra: dict[str, Any]) -> dict[str, Any]:
        return episode_row(
            getattr(env, "task", None),
            env.episode,
            request=getattr(env, "request", None),
            success=bool(info.get("success", False)),
            reward=float(info.get("reward", 0.0)),
            episode_length=int(env.step_count),
            close_kind=info.get("close_kind", "unknown"),
            termination_reason=info.get("termination_reason", ""),
            metrics=info.get("metrics", {}),
            wall_s=wall,
            sim_s=float(env.sim_time),
            recorded_steps=self._recorded,
            captured_steps=len(self._capture.get("step", ())) if self.capture_enabled else 0,
            video_frames=self._video_frames,
            extra=extra,
        )

    def close(self) -> None:
        """Finish the run. Reports once if anything was dropped."""
        self._detach_capture_observer()
        if self._videos is not None:
            self._videos.close()
            self._videos = None
        if self.failures:
            what = "capture" if all(f.startswith("capture ") for f in self.failures) else "recording"
            print(f"[record] {what} was disabled after {len(self.failures)} failure(s): {self.failures[0]}")

    def _detach_capture_observer(self) -> None:
        if self._capture_env is not None:
            remove = getattr(self._capture_env, "remove_step_observer", None)
            if remove is not None:
                remove(self._capture_step)
        self._capture_env = None
        self._capture_observed = False

    def _disable_capture(self, where: str, exc: Exception) -> None:
        """Turn off raw capture alone; ordinary recording, if on, carries on."""
        self.capture_enabled = False
        self._detach_capture_observer()
        self._capture = {}
        self.failures.append(f"capture {where}: {type(exc).__name__}: {exc}")
        self.enabled = self.record_enabled

    def _disable(self, where: str, exc: Exception) -> None:
        self.enabled = False
        self._detach_capture_observer()
        self.failures.append(f"{where}: {type(exc).__name__}: {exc}")
        if self._videos is not None:
            try:
                self._videos.close()
            except Exception:
                pass
            self._videos = None


def _stack_steps(steps: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """One array per step key, ragged vectors padded rather than refused.

    An episode's ``action`` is empty on a held (``None``) step and a full
    vector otherwise, and a policy that switches control modes changes its
    length mid-episode. Such a key is padded with NaN to its longest step and
    a ``<key>_len`` array says how much of each row is real.
    """
    arrays: dict[str, np.ndarray] = {}
    for key, values in steps.items():
        if not values:
            continue
        shapes = {np.shape(v) for v in values}
        if len(shapes) == 1 or any(np.ndim(v) > 1 for v in values):
            arrays[key] = np.stack(values)
            continue
        flat = [np.asarray(v).reshape(-1) for v in values]
        lengths = np.asarray([len(v) for v in flat], dtype=np.int32)
        padded = np.full((len(flat), int(lengths.max())), np.nan, dtype=np.result_type(*flat, np.float32))
        for i, row in enumerate(flat):
            padded[i, : len(row)] = row
        arrays[key] = padded
        arrays[f"{key}_len"] = lengths
    return arrays


def _overlay_stage(frames: dict[str, Any], stage: str) -> dict[str, Any]:
    """Copy RGB frames and draw a compact graph-stage label at top right."""
    label = str(stage).strip()
    if not label:
        return frames
    from PIL import Image, ImageDraw, ImageFont

    labelled: dict[str, Any] = {}
    for name, channels in frames.items():
        copied = dict(channels)
        rgb = channels.get("rgb")
        if rgb is not None:
            image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.load_default(size=max(14, image.height // 28))
            except TypeError:
                font = ImageFont.load_default()
            box = draw.multiline_textbbox((0, 0), label, font=font, spacing=3)
            width, height = box[2] - box[0], box[3] - box[1]
            padding = max(5, image.height // 100)
            x = max(padding, image.width - width - 2 * padding)
            y = padding
            draw.rounded_rectangle(
                (x - padding, y - padding, x + width + padding, y + height + padding),
                radius=padding,
                fill=(0, 0, 0),
            )
            draw.multiline_text((x, y), label, fill=(255, 255, 255), font=font, spacing=3)
            copied["rgb"] = np.asarray(image)
        labelled[name] = copied
    return labelled
