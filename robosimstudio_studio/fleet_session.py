# SPDX-License-Identifier: Apache-2.0
"""The studio, over a wave: many worlds running at once, in one page.

A sibling of :class:`~robosimstudio_studio.session.StudioSession` rather
than a mode inside it, and the split is the honest one. The studio's interactive
parts -- the six-DOF arm gizmo, scene picking, dragging a handle -- are
single-world *concepts*: they drive one arm, on one scene, that a person is
looking at. There is no sensible bimodal version of "drag this handle" across
three hundred worlds, so this offers what a fleet actually has to offer instead:
a grid of worlds each running its own episode under its own policy, with the
playback controls that make watching one useful.

What carries over is everything that was never about a single world -- the
viewer, the lighting rig, the stage, the speed control, the frame pacing that
keeps drawing from eating the loop. What does not carry over is left out rather
than stubbed.

Two facts about watching a fleet, both measured, both shaping this:

* **Realtime factor is the binding constraint, not draw cost.** Steady-state
  drawing at a couple of frames a second is inside the noise up to 64 worlds and
  about 7% at 256. But each world advances more slowly as the fleet widens --
  ``tool_sort`` runs 256 worlds at 1.01x wall clock and 512 at 0.78x, while a
  single towel is already 0.58x. So the width a page can *usefully* show is set
  by how fast the scene runs, and the default speed is ``max`` because a fleet
  is normally already slower than the clock.
* **The first frame is expensive and the rest are not.** Handing a replicated
  model to the browser uploads every world's geometry, which is a few seconds
  whatever the width. It happens once, at load.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robosimstudio_bench.run.fleet import WorldSlot, plan_waves, score_world

from .fleet_view import FleetView

__all__ = ["FleetSession"]

SPEEDS: dict[str, float] = {"max": 0.0, "1x": 1.0, "0.5x": 0.5, "0.25x": 0.25, "0.1x": 0.1}
"""Playback rates. ``max`` leads, unlike the single-world page's, because a wave
wide enough to be worth watching is usually already slower than the clock."""


class FleetSession:
    """A page showing one wave, stepped and scored live.

    Args:
        task: Which variant to run.
        num_worlds: Worlds in the wave.
        scene: Stage it somewhere else, or ``""`` for the task's own.
        embodiment: Run it on another robot, or ``""``.
        gripper: Put another hand on every arm, or ``""``. The three pairing
            axes are carried through because the task menu offers them at any
            world count -- a dropdown that silently did nothing above one world
            would be worse than not offering it.
        policy: Dotted path to a fleet policy, or a shorthand the runner knows.
            Empty holds every world still, which is the baseline.
        solution_kwargs: Constructor arguments for it.
        episode: First episode id; world *w* runs ``episode + w``.
        port: Where to serve.
        share: Whether to open a public tunnel.
        max_worlds: Draw at most this many, whatever the wave's width.
        device: Warp device.
        eval_every: Control steps between scoring passes. Also the policy's
            completion cadence -- the two have to agree, since a motion is only
            noticed as finished on a scoring pass and a policy counting its
            settle in a different unit waits the wrong number of steps.
        fps: Draw rate ceiling.
        viewer: An existing viewer to draw into, borrowed rather than owned.
            The studio passes its own, so setting the world count reloads the
            scene in the tab that is already open.
        loop: Start the next wave when one finishes, advancing the episode ids.
            A page left open is usually one somebody wants to *watch*, and an
            episode is a few seconds -- so without this the grid is frozen on a
            finished wave within a minute of opening the link.
    """

    def __init__(
        self,
        task: str,
        *,
        num_worlds: int = 16,
        scene: str = "",
        embodiment: str = "",
        gripper: str = "",
        policy: str = "",
        solution_kwargs: dict[str, Any] | None = None,
        episode: int = 0,
        port: int = 8080,
        share: bool = False,
        max_worlds: int = 64,
        device: str | None = None,
        eval_every: int = 1,
        fps: float = 2.0,
        loop: bool = False,
        viewer: Any = None,
    ) -> None:
        from robosimstudio_bench.envs.fleet import FleetEnv
        from robosimstudio_bench.solutions.fleet import load_fleet_solution

        t0 = time.perf_counter()
        self.num_worlds = int(num_worlds)
        self.episode0 = int(episode)
        self.fleet = FleetEnv(
            task,
            num_worlds=self.num_worlds,
            scene=scene,
            embodiment=embodiment,
            gripper=gripper,
            device=device,
            eval_every=eval_every,
            verbose=True,
        )
        self.task = self.fleet.task
        # The policy checks motion completion on the fleet's scoring pass, so it
        # is told the same cadence rather than left on its own default -- with
        # the two disagreeing, motions finish late and settles are counted in
        # the wrong unit, which reads as a wave that simply never succeeds.
        kwargs = dict(solution_kwargs or {})
        kwargs.setdefault("cadence", int(eval_every))
        self.policy = load_fleet_solution(policy, self.fleet, **kwargs)
        self.view = FleetView(self.fleet, port=port, share=share, max_worlds=max_worlds, fps=fps, viewer=viewer)
        self.view.bind(self.policy)

        self.speed = 0.0
        self.paused = False
        self.loop = bool(loop)
        self.done = False
        """Every assigned world has been scored. A page idles on it; a budgeted
        run stops, since a finished wave never steps again and a step count it
        cannot reach would idle forever."""
        self.quit_requested = False
        self.reset_requested = False
        self.status = ""
        self.slots: list[WorldSlot] = []
        self.step_count = 0
        self.limit = int(self.task.max_steps or 10_000)
        self._dt = 1.0 / float(self.fleet.pipeline.control_hz)
        self._next_frame = time.perf_counter()
        self._rate_window = (time.perf_counter(), 0)
        self.rate = 0.0
        self.gui = None
        """Set when the studio composes this page's controls under its own."""
        self.reset_wave()
        print(
            f"[fleet-studio] {self.task.full_name} x {self.num_worlds} worlds ready in {time.perf_counter() - t0:.1f} s"
        )

    # ---------------------------------------------------------------------- gui
    def build_gui(self, gui=None) -> None:
        """The controls a grid needs, and no more.

        Deliberately small. A control that would act on "the" scene has no
        meaning here, so this offers the ones that act on the *wave*: whether it
        is running, how fast, and starting it over. Takes a ``gui`` so the studio
        can compose it under its own task menu rather than owning the page.
        """
        if gui is None:
            server = getattr(self.view.viewer, "server", None)
            if server is None or not hasattr(server, "gui"):
                self.gui = None
                return
            gui = server.gui
        self.gui = gui
        with gui.add_folder(f"Wave ({self.num_worlds} worlds)"):
            # No task name here: the Task folder above already says which one,
            # and repeating it put two things called "task" on the page.
            self._progress = gui.add_text("progress", initial_value="", disabled=True)
            self._rate_text = gui.add_text("rate", initial_value="", disabled=True)
            pause = gui.add_button("pause / resume")
            reset = gui.add_button("restart wave")
            loop = gui.add_checkbox("loop waves", initial_value=self.loop)
            speed = gui.add_dropdown("speed", tuple(SPEEDS), initial_value="max")

        @pause.on_click
        def _(_event) -> None:
            self.paused = not self.paused

        @reset.on_click
        def _(_event) -> None:
            # Flagged rather than done here: this fires on the server's thread
            # and a reset writes the device state the loop is stepping.
            self.reset_requested = True

        @loop.on_update
        def _(_event) -> None:
            self.loop = bool(loop.value)

        @speed.on_update
        def _(_event) -> None:
            self.speed = SPEEDS.get(speed.value, 0.0)

    def _refresh_gui(self) -> None:
        if self.gui is None:
            return  # controls not composed yet, or no browser to compose them in
        live = sum(1 for slot in self.slots if slot.live and not slot.parked)
        done = [slot for slot in self.slots if not slot.parked and not slot.live]
        wins = sum(1 for slot in done if slot.success)
        self._progress.value = f"step {self.step_count}/{self.limit} | {live} running, {len(done)} done, {wins} success"
        self._rate_text.value = f"{self.rate:.1f} steps/s ({self.rate * self._dt:.2f}x realtime)"

    # -------------------------------------------------------------------- wave
    def reset_wave(self) -> None:
        """Start a fresh wave: world *w* takes episode ``episode0 + w``."""
        ids = list(range(self.episode0, self.episode0 + self.num_worlds))
        waves = plan_waves(self.task, ids, self.num_worlds)
        episodes = waves[0] if waves else []
        padded = list(episodes) + [None] * (self.num_worlds - len(episodes))
        self.slots = [
            WorldSlot(world=w, episode=episode, evaluator=self._evaluator()) for w, episode in enumerate(padded)
        ]
        self.fleet.reset_wave(padded)
        snapshot = self.fleet.snapshot()
        for slot in self.slots:
            # The start pose the fleet captured at reset, the one baseline the
            # evaluator and the policy's tools both measure "has it moved" from.
            slot.settled_body_q = self.fleet.settled_body_q(slot.world)
            if slot.evaluator is not None:
                slot.evaluator.reset(self.fleet.context(slot.world, snapshot, settled_body_q=slot.settled_body_q))
        self.policy.reset_wave(self.slots)
        self.step_count = 0
        self.done = False
        self.status = f"wave of {len(episodes)} episodes from {self.episode0}"

    def _evaluator(self):
        from robosimstudio_bench.goals.context import EvaluatorNone

        return self.task.evaluator(self.task) if self.task.evaluator is not None else EvaluatorNone()

    # -------------------------------------------------------------------- loop
    def tick(self) -> bool:
        """One iteration. True when the wave actually stepped."""
        if self.reset_requested:
            self.reset_requested = False
            self.reset_wave()
            self._next_frame = time.perf_counter()
            return False
        if self.paused:
            self.view.on_tick(self.fleet, self.slots, self.step_count)
            time.sleep(0.02)
            return False

        live = np.array([slot.live and not slot.parked for slot in self.slots], dtype=bool)
        if not live.any() or self.step_count >= self.limit:
            # The wave is over. A page keeps showing it -- a finished grid is the
            # result -- so this idles rather than exiting, and ``done`` is what
            # tells a *budgeted* run to stop instead of idling forever.
            self.done = True
            if self.loop:
                # On to the next W episodes, so a page left open keeps showing
                # work rather than a frozen result.
                self.episode0 += self.num_worlds
                self.reset_wave()
                return False
            self.view.on_tick(self.fleet, self.slots, self.step_count)
            self._refresh_gui()
            time.sleep(0.05)
            return False

        self.policy.act(live)
        self.fleet.flush()
        self.fleet.step()
        self.step_count += 1

        due = self.step_count % self.fleet.eval_every == 0
        snapshot = self.fleet.snapshot() if due else None
        self.policy.on_step(snapshot)
        if snapshot is not None:
            self._score(snapshot, live)
        self.view.on_tick(self.fleet, self.slots, self.step_count)
        self._tick_rate()
        self._pace()
        return True

    def _score(self, snapshot, live: np.ndarray) -> None:
        """Score the live worlds, the way a batched run scores them.

        Through the runner's own :func:`score_world`, so a page and a run agree
        about what "success" and "truncated" mean -- watching a wave should show
        the same verdicts the same wave would be reported with.
        """
        requests = self.policy.termination_requests()
        for world in np.flatnonzero(live):
            slot = self.slots[world]
            done, info = score_world(
                slot, self.fleet, snapshot, step=self.step_count, limit=self.limit, request=requests.get(int(world), "")
            )
            if done:
                slot.finish(
                    success=info["success"],
                    reward=info["reward"],
                    close_kind=info["close_kind"],
                    steps=self.step_count,
                    wall_s=0.0,
                    info=info,
                )
        self._refresh_gui()

    def _tick_rate(self) -> None:
        started, frames = self._rate_window
        now = time.perf_counter()
        if now - started >= 1.0:
            self.rate = (frames + 1) / (now - started)
            self._rate_window = (now, 0)
        else:
            self._rate_window = (started, frames + 1)

    def _pace(self) -> None:
        """Hold the wave to the selected speed, dropping debt rather than catching
        up -- a scene already too slow gets slower, and a stutter becomes a stall."""
        if self.speed <= 0.0:
            return
        self._next_frame += self._dt / self.speed
        sleep = self._next_frame - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            self._next_frame = time.perf_counter()

    def run(self, *, steps: int = 0) -> None:
        """Serve until the browser goes away, or for a fixed number of steps.

        Args:
            steps: Stop after this many *stepped* iterations. Zero serves until
                asked to stop. Counting steps rather than passes means a paused
                page does not spend the budget.
        """
        stepped = 0
        wall0 = time.perf_counter()
        try:
            while self.view.viewer.is_running() and not self.quit_requested:
                if not self.tick():
                    if steps and self.done:
                        return
                    continue
                stepped += 1
                if steps and stepped >= steps:
                    return
        finally:
            self.report(stepped, time.perf_counter() - wall0)
            self.close()

    def report(self, stepped: int, wall: float) -> None:
        """Say what the wave did, on the way out.

        A page somebody is watching reports itself continuously in the browser;
        a page driven by ``--profile-steps`` has nowhere to say it but here, and
        a measured run that printed nothing is a measurement nobody has.
        """
        rate = stepped / max(wall, 1e-9)
        done = [slot for slot in self.slots if not slot.parked and not slot.live]
        wins = sum(1 for slot in done if slot.success)
        assigned = sum(1 for slot in self.slots if not slot.parked)
        print(
            f"[fleet-studio] {stepped} steps x {self.num_worlds} worlds in {wall:.1f} s = "
            f"{rate:.1f} steps/s ({rate * self._dt:.2f}x realtime, {self.num_worlds * rate:.0f} world-steps/s)"
        )
        print(f"[fleet-studio] {wins}/{assigned} success, {len(done)} of them finished within the budget")

    def close(self) -> None:
        """Give back what this page owned. A borrowed viewer stays open."""
        self.policy.close()
        self.view.close()
        self.fleet.close()
