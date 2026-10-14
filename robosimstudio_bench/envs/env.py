# SPDX-License-Identifier: Apache-2.0
"""The spec-driven environment: a task, an episode, an action, an observation.

:class:`Env` is what a policy, a runner or a teleop session actually holds. It is
a :class:`~robosimstudio.sim.Sim` -- a compiled scene, its control modes, its
camera rig and the loop over them -- plus everything that needs a *benchmark* to
have declared it: which task, which episode, what the evaluator says, whether the
initial state holds, and what an episode's instruction is.

**What it does not add is the check that the split is in the right place.** It
does not override :meth:`~robosimstudio.sim.Sim.read`,
:meth:`~robosimstudio.sim.Sim.make_context`, or any observation provider except
the instruction, because there is nothing task-shaped left to put in them: an
arm's IK does not know what it is reaching for and a camera does not know what it
is photographing.

Two behaviours are worth reading before using it:

* **Reset rebuilds only when the model changed.** Episodes of the same variant
  usually differ only in where the objects start, and rebuilding a scene per
  episode makes a hundred-episode evaluation cost an hour instead of minutes.
  So a reset compares the new episode's ``model_key`` against the current one
  and re-poses when they match. What counts as a model change is the *task's*
  declaration (``EpisodeSpec.model_key_fields``), not a guess made here.
* **``info["success"]`` is the evaluator's, always.** A policy may call
  :meth:`~robosimstudio.core.env.EnvBase.request_termination`, which ends the
  episode and is recorded as such -- but the success bit next to it is what the
  task's evaluator said at that step.

The observation contract::

    obs = {
        "proprio":     {"<arm>/joint_q", "<arm>/gripper", "<arm>/ee_pose"},
        "objects":     {"<name>": (px, py, pz, qx, qy, qz, qw)},
        "deformables": {"<name>": {"centroid_x", ..., "footprint_area"}},
        "cameras":     {"<camera>": {"rgb", "depth", "seg"}},
        "instruction": str,
        "step": int, "sim_time": float,
    }

``cameras`` is present only when a renderer is attached, and only on the steps a
``render_every`` cadence renders on.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robosimstudio.core.env import beats
from robosimstudio.scene import CompiledScene, compile_scene
from robosimstudio.sim import Sim
from robosimstudio.specs import Episode, TaskSpec, replace
from robosimstudio_bench.goals.context import EvalResult, diverged_result, evaluator_for, finite_reward
from robosimstudio_bench.taskbuild.episodes import make_episode
from robosimstudio_bench.taskbuild.matrix import declared_key
from robosimstudio_bench.taskbuild.registry import get_task

if TYPE_CHECKING:
    from robosimstudio.scene.state import WorldState

__all__ = ["Env", "make_env"]

REPRO_EXTRA = (
    "repro export is not included in this release: it needs the robosimstudio_repro package, "
    "which is not distributed with robosimstudio"
)


def _repro(module: str):
    """``robosimstudio_repro.<module>``, or ``None`` when that package is absent.

    The replay bundle is a separate top-level package, not a layer of the
    library, and this release does not ship it. Only *its* absence is
    answered with ``None``: an ``ImportError`` raised from inside the package is
    a bug in it and stays loud.
    """
    import importlib

    try:
        return importlib.import_module(f"robosimstudio_repro.{module}")
    except ModuleNotFoundError as missing:
        if missing.name == "robosimstudio_repro":
            return None
        raise


class Env(Sim):
    """One task variant, steppable and scored.

    Args:
        task: The variant, or ``"<suite>/<variant>"``.
        scene: Which registered place to stage it in, or ``""`` for the task's
            own. Must be one the task declares a binding for -- an undeclared
            pairing raises here rather than compiling on a guessed mount, since
            a robot placed at a pose nobody chose builds a scene that looks
            plausible and reaches nothing.
        embodiment: Which registered robot, or ``""`` for the task's own. Same
            rule.
        gripper: Which registered hand to put on every arm, or ``""`` for
            whichever the robot declares. Same rule again -- and the hand has to
            be one that can actually get onto this arm, which
            :func:`~robosimstudio_bench.taskbuild.matrix.admits` decides and the
            binding records.
        episode: Which episode to start at.
        renderer: Backend name for the camera rig, ``"auto"`` for the machine's
            (see :func:`~robosimstudio.solvers.render.choose_renderer`), or ``""``
            for no rendering.
        render_every: Render one control step in this many. A policy that only
            looks at pixels every fourth step should not pay for four renders.
        renderer_kwargs: Forwarded to the backend.
        device: Warp device.
        max_steps: Overrides the task's own when non-zero.
        control_mode: Which mode starts active, overriding the task's own when
            non-empty. Every mode the scene supports is built regardless -- this
            names the one a bare ``step(action)`` speaks, and a caller may name
            another per command (``step(action, mode="osc_pose")``) or hand the
            arm over for a while with
            :meth:`~robosimstudio.core.env.EnvBase.use_control`.
        use_cuda_graph: Overrides the physics spec's own setting.
        verbose: Print what was compiled.
    """

    def __init__(
        self,
        task: TaskSpec | str,
        *,
        scene: str = "",
        embodiment: str = "",
        gripper: str = "",
        episode: int = 0,
        renderer: str = "",
        obs_renderer: str = "",
        render_every: int = 1,
        renderer_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        max_steps: int = 0,
        control_mode: str = "",
        use_cuda_graph: bool | None = None,
        verbose: bool = False,
    ) -> None:
        self.task: TaskSpec = get_task(task) if isinstance(task, str) else task
        if scene or embodiment or gripper:
            # The declared spelling of the pairing, so naming the robot's own
            # hand finds the binding that left it unstated, as admits() does.
            scene, embodiment, gripper = declared_key(self.task, scene, embodiment, gripper) or (
                scene,
                embodiment,
                gripper,
            )
            # Validate before rewriting: binding_for raises with the admissible
            # list, which is the message a caller who guessed a pairing needs.
            self.task.binding_for(scene, embodiment, gripper)
            self.task = replace(
                self.task,
                scene=scene or self.task.scene,
                embodiment=embodiment or self.task.embodiment,
                gripper=gripper or self.task.gripper,
            )
        self.device = device
        self.verbose = bool(verbose)
        self._use_cuda_graph = use_cuda_graph

        self.episode: Episode = make_episode(self.task, episode)
        scene: CompiledScene = compile_scene(
            self.task, self.episode, device=device, use_cuda_graph=use_cuda_graph, verbose=verbose
        )

        # Everything a *scene* needs -- control modes, the camera rig, the
        # contact probe, the observation providers -- is Sim's, and it needs
        # none of it explained: a TaskSpec satisfies every attribute Sim reads
        # off a request, which is the check that the cut is in the right place.
        super().__init__(
            scene,
            control_mode=control_mode or self.task.control_mode,
            max_steps=max_steps or self.task.max_steps,
            renderer=renderer,
            obs_renderer=obs_renderer,
            render_every=render_every,
            renderer_kwargs=renderer_kwargs,
            verbose=verbose,
        )

        self.evaluator = evaluator_for(self.task, self.episode)
        """This episode's scorer. Built for the episode's active set and rebuilt
        when a reset changes it -- a superset instance is scored on the objects
        it runs, not on every object its model holds."""
        self.repro_tape = self._make_repro_tape()
        self.last_result = EvalResult()
        self._init_reported = False
        """Whether this env has already said its init does not hold.

        Once per environment, not per episode: whether a task starts where it
        says it does is a fact about the task."""
        self._step_observers: list[Any] = []
        """Read-only instrumentation called after every control step.

        This includes steps made inside a blocking graph tool, which never
        return through the outer rollout loop. Observers therefore see the
        canonical control-step stream rather than a policy-dependent subset.
        """
        self.last_context: WorldState | None = None
        """The context the last :meth:`evaluate` gathered, or ``None`` before one.

        ``None`` rather than an empty context, because "nothing has been scored
        yet" and "everything read zero" are different answers and only one of
        them is true here."""
        self.last_init: Any = None
        """The last reset's :class:`~robosimstudio_bench.goals.initial.InitReport`.

        ``info["init"]`` carries three scalars off it, which is what an episode
        row wants. *Which* constraints failed is a list, and anything showing an
        init clause by clause -- a page, an authoring loop -- needs the list
        rather than a sentence it has to parse back. ``None`` before the first
        reset, and for an evaluator that cannot answer at all."""

        # The one observation a scene cannot answer: an instruction is what the
        # *episode* says this run is for.
        self.add_obs_provider("instruction", lambda env: env.episode.instruction)

    # ------------------------------------------------------------------ build
    def _make_repro_tape(self):
        """The always-on command recorder :meth:`export_repro` drains.

        Installed below the episode hook floor so it survives resets. Its
        per-step cost rides on readbacks evaluation already pays for; a hot
        loop that begrudges even that sets ``env.repro_tape.enabled = False``.

        ``None`` when ``robosimstudio_repro`` is absent: the env then records
        nothing, and every reader of ``repro_tape`` is guarded for it.
        """
        tape_module = _repro("tape")
        if tape_module is None:
            return None
        tape = tape_module.ReproTape(self)
        tape.install()
        return tape

    # -------------------------------------------------------------- lifecycle
    def reset(self, episode_id: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start an episode, rebuilding the scene only if the model changed."""
        if episode_id is not None:
            new_episode = make_episode(self.task, episode_id)
            if new_episode.model_key != self.episode.model_key:
                self._rebuild(new_episode)
            if new_episode.active != self.episode.active:
                # The active set is state, not model -- the scene is re-posed
                # and the rest parked -- but the goal tree was masked to the
                # previous episode's objects, and an evaluator carries no
                # per-episode rebuild of its own.
                self.evaluator = evaluator_for(self.task, new_episode)
            self.episode = new_episode

        # ``Sim.reset`` under this: it restores the scene, captures the
        # articulation this run started from, and takes the settled poses.
        obs, info = super().reset()
        if self.repro_tape is not None:
            self.repro_tape.on_reset()
        readback = self.pipeline.snapshot
        for entity in self.deformables().values():
            entity.update_probe(self.state, readback=readback)
        self.evaluator.reset(self.make_context())
        self.last_result = EvalResult()
        # Dropped with the result it belongs to: a context carried across a reset
        # describes the layout the previous episode ended in.
        self.last_context = None
        self._verify_init(info)
        if self.obs_renderer is not None:
            self.frames = self.obs_renderer.render(self.state)
            obs["cameras"] = self.frames
        elif self.renderer is not None:
            self.frames = self.renderer.render(self.state)
            obs["cameras"] = self.frames
        if self.renderer is not None and self.obs_renderer is not None:
            self.video_frames = self.renderer.render(self.state)
            obs["video_cameras"] = self.video_frames
        return obs, info

    def _rebuild(self, episode: Episode) -> None:
        """Recompile for an episode that needs a different model."""
        if self.obs_renderer is not None:
            self.obs_renderer.close()
            self.obs_renderer = None
        if self.renderer is not None:
            self.renderer.close()
        self.scene = compile_scene(
            self.task, episode, device=self.device, use_cuda_graph=self._use_cuda_graph, verbose=self.verbose
        )
        self.pipeline = self.scene.pipeline
        self.model = self.pipeline.model
        # Every ref resolves through the scene it is bound to, so one handed out
        # before this rebuild -- ``env[name]`` caches them -- went on reading the
        # model before it, frozen at its last state. Rebound, a held ref follows,
        # as Sim.__init__ binds the refs it is given.
        for ref in self.refs.values():
            ref._scene = self.scene
        self.controllers = self._make_controllers()
        self.contacts = self._make_contact_probe()
        # Both of these are *indices into a model*, and the model just changed.
        # A camera's ``parent_body`` was resolved against the old one, so an
        # eye-in-hand rig would ride whichever body now happens to hold that
        # index -- and a clearance probe built lazily and kept would go on
        # answering about the scene before this one, which is the worst shape a
        # wrong answer can take: plausible, and about the wrong world.
        self._cameras = self._resolve_cameras()
        self._clearance = None
        self._parking = None
        self._restricted = None
        self._warmed = False
        self._start_joint_q = None
        if self.obs_renderer_name and self.cameras:
            self.obs_renderer = self._make_renderer(self.obs_renderer_name)
        if self.renderer_name and self.cameras:
            self.renderer = self._make_renderer(self.renderer_name)

    def reset_info(self) -> dict[str, Any]:
        return {
            "task": self.task.full_name,
            "episode_id": self.episode.episode_id,
            "seed": self.episode.seed,
            "instruction": self.episode.instruction,
            "model_key": self.episode.model_key,
        }

    def export_repro(self, out_dir: str | Path, **kwargs: Any) -> Path:
        """Write this run as a standalone repro bundle; return the bundle directory.

        Needs the separate ``robosimstudio_repro`` package, which this release
        does not include; without it this raises ``RuntimeError`` saying so.
        """
        export_module = _repro("export")
        if export_module is None:
            raise RuntimeError(REPRO_EXTRA)
        return export_module.export_repro(self, out_dir, **kwargs)

    def _verify_init(self, info: dict[str, Any]) -> None:
        """Ask the evaluator whether the settled scene is where the task said.

        A document's ``:init`` is a claim about the starting state, and settling
        is physics: a hammer placed inside ``bin1_region`` can bounce out of it
        on the way to rest. Unchecked, that episode runs from a start its own
        task has misdescribed and the result gets attributed to the policy. The
        same read answers the other question worth asking at that instant --
        whether the goal is *already* satisfied, which is a task measuring
        nothing.

        The first sweep found one, and it was not a settle at all:
        ``cable_route/port_easy`` declares ``(anchor-held cable)`` and reads 1.0 after
        compile and 0.0 after the first reset, because ``EntityRod.reset``
        engages the anchor latch and then zeroes the whole flag array two lines
        later to clear the working hand's slot. Present since the entity was
        written and invisible until something asked -- which is the argument for
        asking on every reset rather than in a test somebody remembers to run.

        Asked here because this is the first moment it can be: after the warm-up,
        the re-pose and the settle, with the evaluator reset against the same
        scene it is about to score.

        **Duck-typed, and that is the layer law rather than laziness.** Verifying
        an init means measuring a constraint with the atoms the goal uses, which
        lives a layer above this one; a hand-written evaluator has no such
        method and contributes nothing here. So this asks whether the object it
        already holds can answer, exactly as :attr:`doc` asks whether the
        evaluator has a document.

        **Reports, never raises.** A task whose init does not hold is worth
        knowing about immediately and is not always worth refusing to run -- a
        settle that lands a can 2 mm outside its region is a threshold question,
        and an environment that would not start is a poor place to find that out.
        The verdict rides in ``info["init"]`` every reset, for whoever is
        recording; the *log* happens once per environment, because whether a
        task's init holds is a property of the task and printing it sixteen times
        during a sweep would train everyone to scroll past it.

        The **report itself** is kept on :attr:`last_init`, not only its summary.
        ``info`` carries three scalars because that is what an episode row wants;
        which constraints failed is a list, and a reader that has to parse it back
        out of a sentence is a reader with a second copy of the format. A page
        showing the init clause by clause needs the list, and it is right here.
        """
        check = getattr(self.evaluator, "check_start", None)
        if check is None:
            return
        report = check(self.make_context())
        if report is None:
            return
        self.last_init = report
        info["init"] = {"holds": bool(report.holds), "trivial": bool(report.trivial), "summary": report.summary()}
        if not report and not self._init_reported:
            self._init_reported = True
            print(f"[env] {self.task.full_name}: {report.summary()}")

    # -------------------------------------------------------------------- step
    def step(self, action: Sequence[float] | None = None, mode: str = ""):
        """One control period. ``mode`` names the controller this action is for.

        Empty uses whichever is active, so a single-mode policy never mentions it.
        """
        obs, _, terminated, truncated, info = super().step(action, mode)
        reward = float(info["reward"])
        for observer in tuple(self._step_observers):
            observer(self, action, obs, reward, info)
        return obs, reward, terminated, truncated, info

    def add_step_observer(self, observer: Any) -> None:
        """Observe each completed control step, including tool-internal ones."""
        if observer not in self._step_observers:
            self._step_observers.append(observer)

    def remove_step_observer(self, observer: Any) -> None:
        """Remove an observer previously added with :meth:`add_step_observer`."""
        if observer in self._step_observers:
            self._step_observers.remove(observer)

    def evaluate(self) -> tuple[bool, bool, dict[str, Any]]:
        """Score the step, then fold in the base class's termination reasons."""
        if self.contacts is not None:
            self.contacts.update(self.step_count)
        readback = self.pipeline.snapshot
        for entity in self.deformables().values():
            entity.update_probe(self.state, self.step_count, readback=readback)
        if self.obs_render_enabled and self.step_count % self.render_every == 0:
            if self.obs_renderer is not None:
                self.frames = self.obs_renderer.render(self.state)
            elif self.renderer is not None:
                self.frames = self.renderer.render(self.state)
            if self.renderer is not None and self.obs_renderer is not None:
                self.video_frames = self.renderer.render(self.state)

        ctx = self.make_context()
        # Kept, not just used. `make_context` reads `body_q`, `body_qd` and the
        # joint targets off the step's one batched read-back, so anything
        # wanting to *describe* this step -- a recorder, a studio panel --
        # reuses the one already made. The recorder's contract is that it
        # "never asks the environment for anything", which it can only honour
        # if the gathering has already happened.
        self.last_context = ctx
        if _state_diverged(ctx.body_q):
            # The physics has diverged (a stiff contact at high kp can NaN the
            # whole rigid state in one step). Scoring garbage as numbers put
            # NaN rewards into episode rows; ending the episode with its own
            # close_kind says what actually happened and spends no more budget
            # stepping a broken world.
            self.last_result = diverged_result()
            terminated, truncated, info = super().evaluate()
            info["success"] = self.last_result.success
            info["reward"] = self.last_result.reward
            info["metrics"] = dict(self.last_result.metrics)
            info["close_kind"] = "diverged"
            return True, truncated, info
        result = self.evaluator(ctx)
        self.last_result = result

        terminated, truncated, info = super().evaluate()
        info["success"] = bool(result.success)
        info["reward"] = finite_reward(result.reward)
        info["metrics"] = result.metrics
        if result.success and self.task.terminate_on_success and beats("success", info.get("close_kind")):
            # **Success beats truncation, and loses to everything else.** On the
            # step where the goal is met *and* the step budget runs out, the base
            # class has already written "truncated" -- and the guard here used to
            # be ``"close_kind" not in info``, so it stayed. The batched path
            # never agreed: ``run/fleet.py::score_world`` checks
            # ``diverged -> policy_request -> success -> budget`` and records
            # "success" for the same episode. One episode had two labels
            # depending on how many worlds it ran in, which is exactly what a
            # benchmark number must not have.
            #
            # An episode that met its goal on its last allowed step succeeded;
            # calling that "truncated" understates it. ``diverged`` and
            # ``policy_request`` still win, because neither is a claim about the
            # goal: one says the numbers stopped meaning anything and the other
            # says nobody was still trying.
            info["close_kind"] = "success"
            terminated = True
        return terminated, truncated, info

    # -------------------------------------------------------------- observation
    # ------------------------------------------------------------------ access
    @property
    def doc(self):
        """The document this task was defined by, or ``None``.

        Off the evaluator, which is the object a running env already holds and
        which was built from the document in the first place. ``None`` for a task
        assembled some other way -- an honest answer rather than an error, since
        a caller asking what the task *says* about its objects is asking about
        something that may not exist.

        A property rather than a spec field: ``TaskSpec`` is the IR, and putting
        the document on it would make the two disagree the moment anything edited
        one. Named here because it was reachable only as a ``getattr`` spelled out
        at each call site, which is a lookup nobody can find by reading the class.
        """
        return getattr(self.evaluator, "doc", None)

    @property
    def goal(self):
        """The compiled goal tree, or ``None`` for a hand-written evaluator.

        What lets a caller ask which clauses there are and what each one is about
        -- ``bench.goals.goal.leaves`` and ``placements`` walk exactly this -- rather
        than pattern-matching the flat metric keys an ``EvalResult`` carries.
        """
        return getattr(self.evaluator, "goal", None)


def _state_diverged(body_q) -> bool:
    """Whether the rigid state has left the reals.

    One non-finite entry poisons every quantity downstream -- evaluator
    clauses, capture facts, rewards -- so the check is all-or-nothing over the
    pose table and cheap (the host readback has already been paid for).
    """
    if body_q is None or len(body_q) == 0:
        return False
    return not bool(np.isfinite(np.asarray(body_q)).all())


def make_env(task: str, **kwargs) -> Env:
    """Build an environment for ``"<suite>/<variant>"``."""
    return Env(task, **kwargs)
