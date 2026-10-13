# SPDX-License-Identifier: Apache-2.0
"""The environment skeleton: a lifecycle around a pipeline, and nothing else.

:class:`EnvBase` is deliberately thin. It owns the loop -- apply an action, step
the pipeline, run the post-step hooks, gather an observation -- and it owns the
four things that loop needs to be honest about:

* **A set of controllers, not a control mode.** An arm has one target buffer and
  several ways to write it, and which one is right changes *within* an episode: a
  planner's joint trajectory to cross a workcell, normalized Cartesian deltas for
  the part that needs feedback, a scripted waypoint for the part that needs
  precision. So every mode the scene supports is built and one is *active*, and a
  command may name its own mode (``step(action, mode="osc_pose")``). Selecting a
  mode re-seeds it -- see :meth:`EnvBase.use_control`, which is where the reason
  lives.

* **Observation providers.** A dict of ``name -> callable(env) -> value``, so a
  camera rig, a proprio block or a task's own summary all arrive the same way
  and a scene can add one without this class knowing what it is.
* **External termination.** A policy may *request* that an episode end
  (``request_termination("graph reported done")``) and the environment decides
  what to do about it. What a policy may never do is declare *success* -- that
  is the evaluator's, and the split is what keeps a benchmark number meaningful.
* **A finite-state guard.** A diverged solver keeps stepping happily on NaN,
  and does so at a very *stable* rate, because NaN arithmetic costs the same as
  any other and the contact solver stops finding work. Timing that yields a
  number that looks like an excellent result and means nothing, so a run has to
  check rather than assume.

No specs, no registries, no notion of a task, a reward or an episode id. The
toolkit layer's ``Env`` subclasses this and adds all of that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import newton
import numpy as np

from .pipeline import ScenePipeline

__all__ = ["CLOSE_PRECEDENCE", "ActionSpace", "EnvBase", "StepResult", "beats"]


@runtime_checkable
class ActionSpace(Protocol):
    """The four things an environment asks of whatever drives it.

    Stated as a protocol rather than imported, and that is the whole seam: the
    action spaces this stack ships are the robot library's, built over an IK
    solver and a gripper, and ``core`` must not know that. What ``core`` needs is
    smaller than any of them -- how wide is the vector, seed yourself from this
    state, apply this vector, and what does holding still look like -- so that is
    what it names.

    Anything answering these four is drivable here, which is what lets a test
    hand ``EnvBase`` a stub and a fleet hand it something batched.
    """

    @property
    def action_dim(self) -> int:
        """How many numbers one action is."""

    def reset(self, state) -> None:
        """Seed internal setpoints from the scene as it stands."""

    def apply(self, action, control, state) -> None:
        """Write one action into the control buffers."""

    def hold_action(self, state):
        """The action that holds the scene where it is."""


ObsProvider = Callable[["EnvBase"], Any]
"""Fills one key of an observation, given the environment."""
PostStepHook = Callable[["EnvBase"], None]
"""Runs once per control step, after physics and before the observation."""

CLOSE_PRECEDENCE: tuple[str, ...] = ("diverged", "policy_request", "success", "truncated")
"""Why an episode ended, strongest claim first.

Stated once because it was implemented twice and the two disagreed. On the step
where the goal is met *and* the step budget runs out, the single-world path
recorded ``"truncated"`` and the batched one recorded ``"success"`` -- one
episode with two labels depending on how many worlds it ran in, which is exactly
what a benchmark number must not have.

The order is a claim about what the run *means*. ``diverged`` says the numbers
stopped meaning anything; ``policy_request`` says nobody was still trying;
``success`` says the goal was met, and an episode that met it on its last
allowed step succeeded rather than ran out of time; ``truncated`` is what is
left when none of the others applies.
"""


def beats(reason: str, current: str | None) -> bool:
    """Whether ``reason`` should replace ``current`` as the close reason."""
    if not current:
        return True
    return CLOSE_PRECEDENCE.index(reason) < CLOSE_PRECEDENCE.index(current)


StepResult = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]
"""``(obs, reward, terminated, truncated, info)`` -- the gym-shaped return."""


class EnvBase:
    """A steppable scene: reset, step, observe, close.

    Args:
        pipeline: The scene's :class:`~robosimstudio.core.pipeline.ScenePipeline`.
        action_space: How an action becomes joint targets. One
            :class:`ActionSpace`, or a
            ``{mode: ActionSpace}`` mapping of every mode this scene can switch
            between -- see :meth:`use_control`. ``None`` means the environment is
            stepped without actions, which is what a settle or a replay wants.
        control_mode: Which mode starts active. Empty takes the first.
        max_steps: Control steps before an episode truncates. 0 never truncates.
        warmup: Run :meth:`ScenePipeline.warmup_and_capture` on the first reset.
    """

    def __init__(
        self,
        pipeline: ScenePipeline,
        *,
        action_space: ActionSpace | dict[str, ActionSpace] | None = None,
        control_mode: str = "",
        max_steps: int = 0,
        warmup: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.model = pipeline.model
        self.max_steps = int(max_steps)
        self.warmup = bool(warmup)

        if action_space is None:
            self.controllers: dict[str, ActionSpace] = {}
        elif isinstance(action_space, dict):
            self.controllers = dict(action_space)
        else:
            self.controllers = {control_mode or "default": action_space}
        self.control_mode = control_mode or next(iter(self.controllers), "")
        if self.controllers and self.control_mode not in self.controllers:
            raise ValueError(
                f"control_mode {self.control_mode!r} is not among the modes this scene "
                f"supports: {sorted(self.controllers)}"
            )

        self.obs_providers: dict[str, ObsProvider] = {}
        self.post_step_hooks: list[PostStepHook] = []
        self._episode_hook_floor: tuple[int, int] | None = None
        """How many hooks the *scene* installed, snapshotted at the first reset.

        Everything past that mark was installed by a policy during an episode,
        and :meth:`reset` drops it. See that method for the run this cost.
        """
        self.teleop_hooks: list[PostStepHook] = []
        """Run once per :meth:`step_teleop`, not per :meth:`step`."""

        self.step_count = 0
        self.termination_request: str | None = None
        self._warmed = False

    # ------------------------------------------------------------------ shapes
    @property
    def state(self) -> newton.State:
        """The state holding the current frame."""
        return self.pipeline.state

    @property
    def control(self) -> newton.Control:
        return self.pipeline.control

    @property
    def sim_time(self) -> float:
        return self.pipeline.sim_time

    @property
    def action_space(self) -> ActionSpace | None:
        """The mode currently driving the arm."""
        return self.controllers.get(self.control_mode)

    @property
    def action_dim(self) -> int:
        space = self.action_space
        return 0 if space is None else space.action_dim

    def use_control(self, mode: str) -> ActionSpace:
        """Make ``mode`` the active controller, re-seeding it from the live state.

        The re-seed is the whole reason this is a method rather than an
        assignment, and it is the lesson the reference runtime's own controller
        swap records: a controller carries a *setpoint*, and an incoming one whose
        setpoint is where it last left off will drag the arm back there on the
        first step after the handover. ``osc_pose``'s setpoint is a Cartesian pose
        that leads the hand; ``ee_delta``'s is an integrated pose that may be a
        whole episode stale. Re-seeding costs one state read and makes a handover
        continuous.

        Switching to the mode already active does nothing, so a caller may name
        its mode on every command without paying for it.

        Args:
            mode: One of the control modes the action spaces declare, that
                this scene supports.

        Returns:
            The now-active action space.
        """
        if mode == self.control_mode and self.action_space is not None:
            return self.action_space
        if mode not in self.controllers:
            raise KeyError(f"this scene has no control mode {mode!r}; it supports {sorted(self.controllers)}")
        space = self.controllers[mode]
        space.reset(self.state)
        self.control_mode = mode
        return space

    # ---------------------------------------------------------------- lifecycle
    def add_obs_provider(self, name: str, provider: ObsProvider) -> None:
        """Register an observation key. A duplicate name raises rather than wins."""
        if name in self.obs_providers:
            raise KeyError(f"observation key {name!r} is already provided")
        self.obs_providers[name] = provider

    def reset_scene(self) -> None:
        """Put the scene in its start state. Subclasses do the actual work."""

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore the start state and return the first observation.

        Warm-up runs *before* the first reset takes effect, since it steps real
        physics: capture first, then restore, so the measured episode starts
        from the scene's start state rather than from wherever the warm-up left
        it.
        """
        if self.warmup and not self._warmed:
            self.reset_scene()
            self.pipeline.warmup_and_capture()
            self._warmed = True

        self._retire_episode_hooks()
        self.reset_scene()
        self.pipeline.sim_time = 0.0
        self.pipeline.step_count = 0
        self.step_count = 0
        if hasattr(self.pipeline, "reset_finite_flags"):
            self.pipeline.reset_finite_flags()
        if hasattr(self.pipeline, "mark_dirty"):
            self.pipeline.mark_dirty()
        self.termination_request = None
        # Every mode, not just the active one: a script that switches mid-episode
        # would otherwise hand over to a controller still holding a setpoint from
        # the *previous* episode's layout.
        for space in self.controllers.values():
            space.reset(self.state)
        return self.observe(), self.reset_info()

    def _retire_episode_hooks(self) -> None:
        """Drop every hook installed since the first reset, so none outlives its episode.

        A tool a policy runs may install a substep hook -- a kinematic pin on a
        particle, say, in ``pipeline.hooks.post_solve`` -- and a follower in
        :attr:`post_step_hooks`, and release both when it finishes. An episode
        that ends mid-tool (a success, a truncation) never reaches that release,
        and without this the next episode would begin with the last one's pin
        still holding.

        The floor is taken at the **first** reset rather than in ``__init__``,
        because entities install their own hooks while the scene is being built
        -- the conveyor's belt drive, the gravity ramp -- and those are the
        scene, not the episode. Anything appended after that mark belongs to a
        policy that has now finished. Kept here rather than left to each policy:
        a lifetime that depends on every tool remembering to unregister is one
        that will be got wrong.

        A captured CUDA graph holds the substep hooks it was captured with, so
        removing one also drops the graph and asks for a re-capture on the next
        step; otherwise the graph would go on replaying the retired hook.
        """
        floor = self._episode_hook_floor
        if floor is None:
            self._episode_hook_floor = (len(self.pipeline.hooks.post_solve), len(self.post_step_hooks))
            return
        retired = len(self.pipeline.hooks.post_solve) > floor[0]
        del self.pipeline.hooks.post_solve[floor[0] :]
        del self.post_step_hooks[floor[1] :]
        invalidate = getattr(self.pipeline, "invalidate_graph", None)
        if retired and invalidate is not None:
            invalidate(recapture=True)

    def reset_info(self) -> dict[str, Any]:
        """Whatever a caller should know about the episode it just started."""
        return {}

    def request_termination(self, reason: str) -> None:
        """Ask for the episode to end. A policy may do this; declare success it may not."""
        if self.termination_request is None:
            self.termination_request = str(reason)

    # --------------------------------------------------------------------- step
    def apply_action(self, action: Sequence[float] | None, mode: str = "") -> None:
        """Write an action onto the control targets.

        Args:
            action: The action, in ``mode``'s layout.
            mode: Which controller this action is for. Empty uses the active one.
                Naming a different mode switches to it first, re-seeded --
                see :meth:`use_control`. That is what lets one script alternate
                joint-space and Cartesian commands: the mode belongs to the
                *command*, not to the environment.
        """
        space = self.use_control(mode) if mode else self.action_space
        if action is None or space is None:
            return
        space.apply(action, self.control, self.state)

    def _step_once(self) -> None:
        """One control period of physics and the bookkeeping every loop owes it.

        The step count and :attr:`post_step_hooks`, in that order. Every door
        that advances the scene -- :meth:`step`, :meth:`step_teleop`,
        ``Sim.run``, ``rss.draw`` -- goes through here, so a hook a scene
        installs runs whichever of them is driving.
        """
        self.pipeline.step()
        self.step_count += 1
        for hook in self.post_step_hooks:
            hook(self)

    def step(self, action: Sequence[float] | None = None, mode: str = "") -> StepResult:
        """One control period: apply, step, hook, observe."""
        self.apply_action(action, mode)
        self._step_once()
        obs = self.observe()
        terminated, truncated, info = self.evaluate()
        return obs, float(info.get("reward", 0.0)), terminated, truncated, info

    def step_teleop(self) -> None:
        """Advance one control period from whatever the targets already hold.

        The teleop loop's step: a front end writes ``joint_target_q`` directly
        through an IK or a slider, so there is no action vector to apply and
        re-deriving one would round-trip the setpoint through the arm's actual
        position and lag behind the gizmo.
        """
        self._step_once()
        for hook in self.teleop_hooks:
            hook(self)

    def idle_action(self, mode: str = "") -> np.ndarray | None:
        """An action that holds the scene where it is, in ``mode``'s layout.

        Args:
            mode: Empty reads the active controller. Naming one does *not* switch
                to it -- a caller asking what a hold looks like in another layout
                is not asking to hand the arm over.
        """
        space = self.controllers.get(mode) if mode else self.action_space
        return None if space is None else space.hold_action(self.state)

    # -------------------------------------------------------------- observation
    def observe(self) -> dict[str, Any]:
        """Gather every registered observation key.

        Providers read back from device, so a caller should not ask for this
        more than once per step.
        """
        return {name: provider(self) for name, provider in self.obs_providers.items()}

    def evaluate(self) -> tuple[bool, bool, dict[str, Any]]:
        """Decide whether the episode is over. Subclasses add scoring.

        Returns:
            ``(terminated, truncated, info)``. The base implementation
            terminates on a non-finite state or an external request, and
            truncates at :attr:`max_steps`.
        """
        info: dict[str, Any] = {"success": False, "reward": 0.0}
        terminated = False
        if not self.state_is_finite():
            info["close_kind"] = "diverged"
            terminated = True
        elif self.termination_request is not None:
            info["close_kind"] = "policy_request"
            info["termination_reason"] = self.termination_request
            terminated = True
        truncated = bool(self.max_steps and self.step_count >= self.max_steps)
        if truncated and not terminated:
            info["close_kind"] = "truncated"
        return terminated, truncated, info

    def state_is_finite(self) -> bool:
        """Whether the simulation is still producing real numbers.

        Checked rather than assumed: see the module docstring on why a diverged
        run looks like a fast one.
        """
        readback = getattr(self.pipeline, "readback", None)
        if readback is not None:
            # The device flagged it during the step's own read-back: no extra
            # copy, and the flag is sticky across the steps since the last reset.
            return self.pipeline.snapshot.is_finite()
        state = self.state
        # A deformable-only scene -- a material rig -- has no rigid state at
        # all, and those arrays are None rather than empty.
        arrays = [state.joint_q, state.joint_qd, state.body_q] if self.model.body_count else []
        if self.model.particle_count:
            arrays.append(state.particle_q)
        return all(bool(np.isfinite(a.numpy()).all()) for a in arrays)

    def close(self) -> None:
        """Release whatever the environment opened. Idempotent."""
