# SPDX-License-Identifier: Apache-2.0
"""The studio's own controls: recording and export.

Where :mod:`~robosimstudio_studio.panels` is composed from what the
*scene* has -- an arm panel per arm, a cloth panel if there is cloth -- these are
what the *page* does, and they are the same whatever it is showing: keep what
you drove as episodes, film what you watched.

The other two controls a page can have -- the task menu and the policy runner --
need a task, so they live with the page's benchmark half in
``robosimstudio_studio.bench_panels``. ``PanelTask``, ``PanelSolution`` and
``ROUTINE_CHOICE`` still resolve as attributes of this module where that half is
installed, because this is where they were defined and callers import them from
here; :func:`~robosimstudio_studio.seams.lazy_names` is how, and it is why this
module imports without a benchmark.

They obey the same threading contract as the scene panels: callbacks assign
plain attributes and nothing else, and every Warp, GL and file touch happens on
the main thread inside ``drain``, ``capture`` or ``export``.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.render.offline import OfflineRenderer
from robosimstudio.core.render.video import VideoSet
from robosimstudio.record import EpisodeRecorder, EpisodeStore, RunMeta
from robosimstudio.specs import RecordSpec

from .panels import Panel
from .seams import lazy_names

__all__ = ["PanelRecord", "PanelVideo"]

# The names that moved to the page's benchmark half, resolved on access where
# it is installed and left out of ``__all__`` so a star-import does not ask.
__getattr__ = lazy_names(
    __name__,
    {"ROUTINE_CHOICE": "bench_panels", "PanelSolution": "bench_panels", "PanelTask": "bench_panels"},
)


class _NoResult:
    """What a scene with no evaluator has said about itself: nothing.

    The recorder's row wants a success bit, a reward and a metrics table, and
    ``Env`` keeps those in ``last_result``. A bare ``Sim`` has no evaluator and
    therefore no ``last_result``; this stands in for it with the values "not
    scored" honestly maps to. A fresh instance per read, so the metrics dict a
    row carries is never shared with the next one.
    """

    success = False
    reward = 0.0

    def __init__(self) -> None:
        self.metrics: dict = {}


class PanelRecord(Panel):
    """Keep what you just drove: takes, written as a run directory.

    A demonstration is data collection that cannot be batched -- a person is in
    the loop by definition -- so it is the one collection path that belongs in
    the page rather than in ``scripts/bench_run.py``. What it writes is an ordinary run
    directory (``0000/``, ``0001/``, ``episode_results.jsonl``, ``run_meta.json``),
    so anything that reads a benchmark run directory reads a take with nothing
    taught about where it came from.

    **A take is a take on whatever the page is showing.** Start and Stop
    bracket one; the next Start opens the next number, and a session reopening
    a directory that already has takes counts on from the highest rather than
    over it. The recorder is handed an :class:`~robosimstudio.specs.Episode`
    renumbered by take, because the episode *id* here identifies the take while
    the seed, the instruction and the model stay the loaded episode's -- both
    facts land in the row. A scene assembled by hand has the synthetic
    ``Episode(0, 0)`` the compiler gave it, which is all the recorder needs.

    **A bare scene's take is valid and not re-renderable.** It is the same
    ``steps.npz``/``meta.json``/``cameras/*.mp4`` a benchmark take is -- but
    re-rendering one means rebuilding its scene from a registry task id, and a
    hand-assembled scene has none. The run's
    ``run_meta.json`` says so (``provenance.re_renderable``) rather than
    carrying a task name that resolves to nothing.

    **What is recorded as the action is the target, not the measurement.** In
    teleop nobody forms an action: the arm panels solve IK straight into
    ``joint_target_q``. So the panel reads that back through
    :meth:`~robosimstudio.solvers.control.action.ActionJointPosition.target_action`
    and records it as a ``joint_position`` action, which is the command that
    reproduces the take. Recording ``joint_q`` instead would record where the arm
    *got to*, and replaying that as a command lags one servo step every step. A
    take driven by a policy records that policy's own action instead, in its own
    mode -- the same bytes ``scripts/bench_run.py`` would have written.
    """

    def __init__(self, session, *, spec: RecordSpec | None = None, root: str | Path = "") -> None:
        super().__init__(session)
        self.spec = spec or RecordSpec(enabled=True)
        self.root = Path(root) if root else session.run_dir / "demos"
        self.store = EpisodeStore(self.root)
        self.recorder: EpisodeRecorder | None = None
        self.take = self._next_take()
        self.steps = 0
        # Read off the directory rather than started over: a reset that
        # recompiled the scene rebuilds this panel over the same run, and a
        # fresh count would say 0 kept beside takes already written, and the
        # next take's run meta would claim the run began with it.
        meta = self.store.run_meta if self.root.is_dir() else {}
        self.kept = len(self.store.results()) if self.root.is_dir() else 0
        self.mode = ""
        self.started = ""
        """When the current take opened."""
        self.run_started = str(meta.get("started", "") or "")
        """When the first take of this run opened; what ``run_meta.json`` says."""
        self.start_requested = False
        self.stop_requested = False

    def _next_take(self) -> int:
        """One past the highest take already in the directory.

        The usual case is a directory that does not exist yet -- nothing creates
        it until a take is recorded -- and a store asked to list a path that is
        not there raises, which is right for the post-hoc tools and wrong here.
        """
        if not self.root.is_dir():
            return 0
        ids = [int(name) for name in self.store.episode_ids() if str(name).isdigit()]
        return max(ids) + 1 if ids else 0

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def build(self, gui) -> None:
        with gui.add_folder("Record", expand_by_default=False):
            start = gui.add_button("Start take")
            start.on_click(lambda _: setattr(self, "start_requested", True))
            stop = gui.add_button("Stop take")
            stop.on_click(lambda _: setattr(self, "stop_requested", True))
            self._state = gui.add_text("Recording", initial_value="idle", disabled=True)
            self._count = gui.add_text("Kept", initial_value=f"{self.kept} takes", disabled=True)

    # ---------------------------------------------------------------- lifecycle
    def drain(self) -> None:
        if self.stop_requested:
            self.stop_requested = False
            self.stop()
        if self.start_requested:
            self.start_requested = False
            self.start()

    def start(self) -> None:
        """Open a take on the episode the page is holding."""
        if self.recording:
            self.stop()
        self.recorder = EpisodeRecorder(self.root, self.spec, task=getattr(self.env, "task", None))
        # The recorder names the directory from the episode it is handed, and
        # what identifies a take is the take. Everything else about the episode
        # -- seed, instruction, model key -- is the loaded one's, and stays.
        self.recorder.begin(replace(self.env.episode, episode_id=self.take), self.env)
        self.steps = 0
        self.started = datetime.now(UTC).isoformat()
        self.run_started = self.run_started or self.started
        self.mode = ""
        self._state.value = f"take {self.take:04d}: recording"
        self.session.status = f"recording take {self.take:04d}"

    def capture(self, action: np.ndarray | None = None, mode: str = "") -> None:
        """Record the step that just happened. Called by the loop, main thread.

        Args:
            action: What drove the step, when a policy drove it. ``None`` means
                teleop, and the joint target the panels wrote is read back
                instead.
            mode: The action space ``action`` is in.
        """
        if self.recorder is None:
            return
        driven = action is not None
        if not driven:
            action, mode = self._teleop_action()
        if mode and not self.mode:
            self.mode = mode
        # A policy step has already observed -- ``env.step`` returns it -- and a
        # teleop step has not, because ``step_teleop`` does not observe. The
        # recorder is forbidden from asking for anything itself, so the front end
        # asks once, here, and only while a take is open.
        obs = self.session.obs if driven else self.env.observe()
        result = self._result()
        info = {"success": bool(result.success), "reward": float(result.reward), "metrics": result.metrics}
        self.recorder.step(self.env, action, obs, float(result.reward), info)
        self.steps += 1

    def _result(self):
        """The last scoring, or the honest nothing a scene with no evaluator has."""
        result = getattr(self.env, "last_result", None)
        return _NoResult() if result is None else result

    def _teleop_action(self) -> tuple[np.ndarray | None, str]:
        """The joint target the panels wrote, as a ``joint_position`` action."""
        space = self.env.controllers.get("joint_position")
        target = getattr(space, "target_action", None)
        if target is None:
            return None, ""
        return target(self.env.control), "joint_position"

    def stop(self) -> None:
        """Close the take, write its row, and keep the run directory readable."""
        if self.recorder is None:
            return
        recorder, self.recorder = self.recorder, None
        result = self._result()
        info = {
            "success": bool(result.success),
            "reward": float(result.reward),
            "metrics": result.metrics,
            "close_kind": "operator",
        }
        row = recorder.end(
            self.env,
            info,
            extra={
                "episode_id": self.take,
                "take": self.take,
                "source_episode_id": self.env.episode.episode_id,
                "policy": f"studio:{self.mode}" if self.mode else "studio:teleop",
                "control_mode": self.mode or "joint_position",
            },
        )
        recorder.close()
        self.store.append_result(row)
        self.kept += 1
        self._write_run_meta()
        self._state.value = f"take {self.take:04d}: {self.steps} steps -> {self.root}"
        self.session.status = f"kept take {self.take:04d} ({self.steps} steps)"
        print(f"[studio] take {self.take:04d}: {self.steps} steps -> {self.root}")
        self.take += 1

    def _write_run_meta(self) -> None:
        """Rewrite the run meta after every take.

        Per take rather than at close, because a session is ended by closing a
        browser tab or a Ctrl-C as often as by anything orderly, and a run
        directory without ``run_meta.json`` cannot be re-rendered -- it is where
        the task name lives.
        """
        rows = self.store.results()
        request = self.env.request
        benchmarked = getattr(self.env, "task", None) is not None
        embodiment = getattr(request, "embodiment", "")
        provenance: dict[str, Any] = {
            "source": "studio",
            "control_hz": request.physics.control_hz,
            # A re-render rebuilds the scene from ``task`` through the
            # registry. A scene assembled by hand has no registry entry, so the
            # run says it cannot be re-rendered rather than naming a task that
            # resolves to nothing.
            "re_renderable": benchmarked,
        }
        if not benchmarked:
            provenance["note"] = "recorded from a scene assembled by hand; it cannot be rebuilt from a task id"
        self.store.write_run_meta(
            RunMeta(
                run_name=self.root.name,
                task=request.full_name if benchmarked else "",
                scene=getattr(request, "scene", "") or getattr(request, "cell", ""),
                embodiment=str(getattr(embodiment, "name", embodiment) or ""),
                solution="studio",
                renderer=self.env.renderer_name,
                device=str(self.env.model.device),
                started=self.run_started,
                finished=datetime.now(UTC).isoformat(),
                episodes=len(rows),
                successes=sum(1 for row in rows if row.get("success")),
                provenance=provenance,
            )
        )

    def refresh(self) -> None:
        if self.recording:
            self._state.value = f"take {self.take:04d}: {self.steps} steps"
        self._count.value = f"{self.kept} takes -> {self.root}"

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        # A reset starts a different trajectory. Keeping the take open would
        # splice two of them into one episode file.
        if self.recording:
            self.stop()

    def close(self) -> None:
        super().close()
        self.stop()


class PanelVideo(Panel):
    """Capture state as it runs; render it to mp4 when you ask.

    Rendering is not something this page does. It is something it *defers*:
    capturing is copying ``body_q`` and ``particle_q`` into a ring buffer, which
    costs microseconds, and the render happens on the button, through
    :class:`~robosimstudio.core.render.offline.OfflineRenderer` -- a replay of
    the captured state, with no solver re-run. So the video
    is the trajectory you watched, photographed afterwards, and deciding to film
    the last thirty seconds is a decision you make *after* seeing them.

    Rendering every step through the backend you want the video in would instead
    pay the render's full cost inside the control loop, at the resolution and the
    convergence a *video* wants rather than the ones a preview settles for: a
    converged 1920x1080 path-traced frame measures 56 ms against a control step's
    15, so filming the session as it runs is deciding to run it at a quarter
    speed, for footage you have not seen yet and may not keep.

    There is one button because there is one decision. The video is rendered
    with the renderer (``rtx``; the collider-only tracer stands in where it is
    not installed). Frame rate, cadence and buffer depth are
    :class:`~robosimstudio_studio.StudioConfig` fields: chosen once for a
    session, not fiddled with while it runs, because a page of knobs for them
    made the studio read like a render console.

    The buffer is a ring, so a long session holds its last ``max_frames`` rather
    than growing without bound. What that drops is said out loud in the readout,
    because a silently truncated video looks exactly like a short one.
    """

    def __init__(self, session, *, backend: str = "", fps: int = 30, every: int = 2, max_frames: int = 900):
        super().__init__(session)
        from robosimstudio.solvers.render import choose_renderer

        device = getattr(getattr(self.env, "model", None), "device", None)
        self.backend = choose_renderer(backend, device)[0]
        self.fps = int(fps)
        self.every = max(1, int(every))
        self.buffer: deque[dict[str, np.ndarray]] = deque(maxlen=max(30, int(max_frames)))
        self.dropped = 0
        self.exports = 0
        self.export_requested = False

    def build(self, gui) -> None:
        with gui.add_folder("Video", expand_by_default=False):
            export = gui.add_button("Export video")
            export.on_click(lambda _: setattr(self, "export_requested", True))
            self._count = gui.add_text("Buffered", initial_value="0", disabled=True)
            self._result = gui.add_text("Last export", initial_value=f"{self.backend} at {self.fps} fps", disabled=True)

    # ------------------------------------------------------------------ capture
    def capture(self) -> None:
        """Copy one step's replayable state. Called by the loop, main thread."""
        if self.env.step_count % self.every:
            return
        if len(self.buffer) == self.buffer.maxlen:
            self.dropped += 1
        state = self.env.state
        # Either channel can be absent: a scene of cloth alone has no bodies,
        # and Newton leaves ``body_q`` as None rather than an empty array.
        step: dict[str, np.ndarray] = {}
        if state.body_q is not None:
            step["body_q"] = np.array(state.body_q.numpy(), dtype=np.float64)
        if self.env.model.particle_count:
            step["particle_q"] = np.array(state.particle_q.numpy(), dtype=np.float32)
        self.buffer.append(step)

    def drain(self) -> None:
        if self.export_requested:
            self.export_requested = False
            self.export()

    def refresh(self) -> None:
        text = f"{len(self.buffer)} frames"
        if self.dropped:
            text += f" (oldest {self.dropped} dropped)"
        self._count.value = text

    # ------------------------------------------------------------------- export
    def export(self) -> None:
        """Render the buffer to mp4. Blocks the loop, which is the honest cost."""
        driver = self.session.driver
        if driver is not None and driver.running:
            self._result.value = "stop the policy first -- a long render times a graph's bus out"
            return
        if not self.buffer:
            self._result.value = "nothing captured yet"
            return
        backend = self.backend
        steps = list(self.buffer)
        self.exports += 1
        target = self.session.run_dir / f"video_{self.exports:03d}_{backend}"
        self._result.value = f"rendering {len(steps)} frames through {backend}..."

        renderer = None
        try:
            cameras = list(self.env.cameras)
            renderer = OfflineRenderer(
                self.env.model,
                # Lit by the scene's look, via `Sim.look`. A video exported from
                # the studio and one written by the harness are then the same
                # picture by construction rather than by convention.
                self.env.make_renderer(backend, cameras=cameras, channels=("rgb",)),
                # Not strict: an episode load that changed the model leaves the
                # buffer holding arrays the current one has no room for, and
                # naming the skipped channel beats refusing the whole video.
                strict=False,
            )
            t0 = time.perf_counter()
            frames = 0
            with VideoSet(target, fps=self.fps) as videos:
                for frame_set in renderer.replay(steps):
                    videos.append_frames(frame_set)
                    frames += 1
                    if frames % 10 == 0:
                        self._result.value = f"{frames}/{len(steps)} frames through {backend}..."
            meta = {
                "task": self.env.request.full_name,
                "episode_id": self.env.episode.episode_id,
                "instruction": self.env.episode.instruction,
                "backend": backend,
                "channels": ["rgb"],
                "fps": self.fps,
                "frames": frames,
                "capture_every": self.every,
                "control_hz": self.env.request.physics.control_hz,
                "cameras": {c.name: [c.width, c.height] for c in cameras},
                "dropped_from_buffer": self.dropped,
                "skipped_channels": sorted(renderer.replay_state.skipped),
            }
            (target / "render_meta.json").write_text(json.dumps(meta, indent=2))
            self._result.value = f"{frames} frames -> {target}"
            self.session.status = f"wrote {frames} frames to {target}"
            print(f"[studio] {frames} frames through {backend} in {time.perf_counter() - t0:.1f} s -> {target}")
        except Exception as exc:
            # A renderer that will not build -- no GPU, a contended card -- is a
            # bad afternoon, not a dead session.
            self._result.value = f"{type(exc).__name__}: {exc}"[:200]
            print(f"[studio] export failed: {type(exc).__name__}: {exc}")
        finally:
            if renderer is not None:
                renderer.close()
