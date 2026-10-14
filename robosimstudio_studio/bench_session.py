# SPDX-License-Identifier: Apache-2.0
"""The page's benchmark half: everything a session does that needs a *task*.

:class:`~robosimstudio_studio.session.StudioSession` is one class over two
kinds of scene, and this is the half of it that only one of them has. A scene
somebody assembled by hand is adopted as it stands; a task has to be *resolved*
-- an id looked up, a pairing admitted or refused, a document compiled into an
``Env`` -- and once it is in the page it brings a goal to read, policies to run
and a world count that can turn the page into a grid. All of that names the
benchmark tier, so all of it is here, in a module the session reaches only
through :func:`robosimstudio_studio.seams.benchmark` and that a release without
the tier leaves out by path.

Functions over a session rather than methods on a subclass, because the page is
*one* class to everybody who opens it: ``StudioSession("suite/variant")`` and
``StudioSession(sim)`` are the same constructor, a task menu can swap one kind
of scene for the other under a browser that stays connected, and
``robosimstudio_real2sim_agent`` subclasses the session for a reason that has nothing to
do with which kind it is showing. Everything here therefore goes back through
the session's own methods (``adopt``, ``teardown``) wherever it has one, so an
override of those still applies.

What is here, by who asks for it:

* the session's task door -- :func:`load_task`, with its two refusals
  (:func:`unbuildable`, :func:`refuse`) and its grid (:func:`load_fleet`,
  :func:`build_fleet_gui`);
* the three panels that need a document -- :func:`document_panels`;
* what a goal tree says about the page -- :func:`opening_subject`, and for the
  overlay :func:`feature_payload` and :func:`feature_holds`;
* the registry's task list, for a launcher's ``--list`` -- :func:`list_tasks`.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.robots.registry import get_embodiment
from robosimstudio.scene.compiler import CannotReplicate
from robosimstudio.specs import TaskSpec
from robosimstudio_bench.envs.env import Env
from robosimstudio_bench.goals.bind import Operand, read
from robosimstudio_bench.goals.goal import leaves
from robosimstudio_bench.taskbuild.matrix import admits, declared_key
from robosimstudio_bench.taskbuild.registry import get_task, list_tasks
from robosimstudio_bench.taskdoc.ast import Ref

from .bench_panels import PanelSolution, PanelTask
from .goal_panel import PanelGoal
from .session import Refused

__all__ = [
    "build_fleet_gui",
    "document_panels",
    "feature_holds",
    "feature_payload",
    "list_tasks",
    "load_fleet",
    "load_task",
    "opening_subject",
    "refuse",
    "unbuildable",
]


def _pairing(task: str, scene: str, embodiment: str, gripper: str = "") -> str:
    """How a pairing is written in the page's status line.

    The task alone stopped naming what is loaded once one task could be staged
    in several places with several robots, but most pages run a native pairing
    and appending two empty strings to every message would be noise. So the
    extra axes appear only when they are not the task's own.
    """
    extra = " / ".join(x for x in (scene, embodiment, gripper) if x)
    return f"{task} @ {extra}" if extra else task


# ---------------------------------------------------------------- task load
def load_task(
    session,
    task: str | TaskSpec,
    episode: int,
    scene: str = "",
    embodiment: str = "",
    gripper: str = "",
    num_worlds: int | None = None,
) -> bool:
    """Compile a task into a page, replacing whatever was in it.

    ``task`` is normally a registry name. It may also be a built
    :class:`~robosimstudio.specs.TaskSpec`, which is how a rebuild
    carries edits the registry does not know about -- a tuned material, most
    obviously. Such a spec already holds the pairing it was compiled with, so
    the axis arguments are left empty and nothing is re-resolved.

    A selection the task does not declare is synthesized through
    :meth:`~robosimstudio.specs.TaskSpec.with_pairing` when
    :func:`~robosimstudio_bench.taskbuild.matrix.admits` allows it. That is the
    explicit opt-out from ``Env``'s refusal to compile undeclared pairings,
    and the task menu is exactly the caller it exists for: its
    ``(untried)`` options *are* these combinations, offered because trying
    one is the point of a page you can drive by hand -- without this, every
    one of them bounced off ``binding_for``'s KeyError and the menu snapped
    back. What ``admits`` refuses is still refused, in its own words rather
    than compiled on a guessed mount.

    The new scene is compiled **before** the old one is given up, because a
    menu can select a task that cannot be built and a page that dies on a
    mis-click is a page that cannot be trusted with a menu. Compiling first
    costs both scenes being alive for a moment and buys the failure being a
    message.

    A pairing that is *refused* never reaches the compile at all, and is
    reported as its reason rather than as a stack -- see
    :class:`~robosimstudio_studio.session.Refused`. Two things get refused:
    what ``admits`` turns down, and a robot whose
    :attr:`~robosimstudio.specs.EmbodimentSpec.unbuildable` says it cannot
    be compiled yet. Only the second can arrive on a pairing the task
    declares, which is why it is checked here and not left to ``admits``.

    Args:
        session: The page. This is
            :meth:`~robosimstudio_studio.session.StudioSession.load_task`,
            which hands itself over.
        task: A registry name or a built spec.
        episode: Which episode to start on.
        scene: Stage it somewhere else, or ``""`` for the task's own.
        embodiment: Run it on another robot, or ``""``.
        gripper: Put another hand on every arm, or ``""``.
        num_worlds: Change the page's world count with this load, or ``None``
            to keep it.

    Returns:
        Whether the task was loaded. On ``False`` the page is untouched.
    """
    t0 = time.perf_counter()
    config = session.config
    label = task if isinstance(task, str) else task.full_name
    # Put back on every ``False``: the count is the page's, and a load that
    # did not happen must not leave the next rebuild asking for its width.
    worlds = session.num_worlds
    if num_worlds is not None:
        session.num_worlds = max(1, int(num_worlds))
    # Ahead of the world count, because a robot that cannot be compiled
    # cannot be compiled at any width: the fleet path builds its own scenes
    # and would otherwise answer the same click with the same traceback.
    why = unbuildable(task, embodiment)
    if why:
        session.num_worlds = worlds
        return refuse(session, label, why, scene, embodiment, gripper)
    if session.num_worlds > 1:
        try:
            if load_fleet(session, task, episode, label, t0, scene, embodiment, gripper):
                return True
            session.num_worlds = worlds
            return False
        except CannotReplicate as exc:
            # The selection is a task; the world count is a throughput
            # preference on it. Most scenes cannot be replicated yet, so
            # refusing here made most of the menu dead on a page that was
            # showing a grid -- the dropdown snapped back to what was
            # loaded and the reason was a truncated line in a disabled
            # text box. Read from the browser that is not a refusal, it is
            # a menu that will not take a selection.
            #
            # So the request is honoured at the width it can be honoured
            # at, which is one. Not silently: the Worlds field is written
            # back to what the page actually built, so it never claims a
            # width it does not have, and the reason is on the status line.
            session.num_worlds = 1
            session._narrowed = str(exc)
    try:
        task, load_scene, load_embodiment, load_gripper = resolve_pairing(task, scene, embodiment, gripper)
        env = Env(
            task,
            scene=load_scene,
            embodiment=load_embodiment,
            gripper=load_gripper,
            episode=episode,
            renderer=config.obs_renderer,
            render_every=config.obs_render_every,
            device=config.device,
            max_steps=config.max_steps,
            control_mode=config.control_mode,
            verbose=True,
        )
    except Refused as exc:
        session.num_worlds = worlds
        return refuse(session, label, str(exc), scene, embodiment, gripper)
    except Exception as exc:
        session.num_worlds = worlds
        message = f"{type(exc).__name__}: {exc}"
        session.status = f"{_pairing(label, scene, embodiment, gripper)} did not compile -- {message}"
        print(f"[studio] {label} did not compile:")
        traceback.print_exc()
        if session.task_panel is not None:
            session.task_panel.report_failure(label, message)
        # Nothing loaded, so there is no narrowing to report; leaving it set
        # would attach this refusal to whatever loads next.
        session._narrowed = ""
        return False

    session.adopt(env, label=label, episode=episode)
    loaded = f"loaded {_pairing(label, scene, embodiment, gripper)} in {time.perf_counter() - t0:.1f} s"
    narrowed, session._narrowed = session._narrowed, ""
    if narrowed:
        # Said on the status line, where a person is looking, and said as
        # what happened rather than as an error: the task they picked is in
        # the page, at one world, because that is the only width it has.
        session.status = f"{loaded} -- at 1 world: {narrowed}"
        if session.task_panel is not None:
            session.task_panel.note(f"1 world -- {narrowed}")
        print(f"[studio] {label} narrowed to 1 world: {narrowed}")
    else:
        session.status = loaded
    print(f"[studio] {session.env.task.full_name} episode {episode}: {session.env.episode.instruction}")
    return True


def resolve_pairing(
    task: str | TaskSpec, scene: str = "", embodiment: str = "", gripper: str = ""
) -> tuple[str | TaskSpec, str, str, str]:
    """The task and overrides a load hands ``Env`` or ``FleetEnv`` for a pairing.

    A pairing the task declares -- in either spelling of the hand, ``""`` or the
    one the robot already wears -- comes back as that binding's own key, so the
    load finds the declared binding (its graphs, its mount) rather than having
    :meth:`~robosimstudio.specs.TaskSpec.with_pairing` append a bare one beside
    it. Anything else :func:`admits` allows comes back as a spec pinned to it,
    with empty overrides; what it refuses raises
    :class:`~robosimstudio_studio.session.Refused` in its words. No overrides at
    all is the task as it stands.
    """
    if not (scene or embodiment or gripper):
        return task, scene, embodiment, gripper
    spec = get_task(task) if isinstance(task, str) else task
    key = declared_key(spec, scene, embodiment, gripper)
    if key is not None:
        return (task, *key)
    want = (scene or spec.scene, embodiment or spec.embodiment, gripper or spec.gripper)
    ok, why = admits(spec, *want)
    if not ok:
        raise Refused(f"inadmissible pairing: {why}")
    return spec.with_pairing(*want), "", "", ""


def unbuildable(task: str | TaskSpec, embodiment: str) -> str:
    """Why the robot this load would compile cannot be, or ``""``.

    The one refusal a page has to make for itself. ``admits`` lets a pairing
    the task *declares* straight through, and rightly: a declaration is
    normally somebody saying they have run it. A suite may declare one that
    nobody has, as the record of a break rather than a claim it works -- and
    on a page whose whole interaction is the click, a record has to read
    back as its own sentence instead of as the ``KeyError`` it is recording.

    Returns the embodiment's own words unchanged, the way ``admits``'
    reasons are: the string names its robot, so a prefix would only say
    the robot twice in one status line.
    """
    try:
        wanted = embodiment or (task.embodiment if isinstance(task, TaskSpec) else get_task(task).embodiment)
        return get_embodiment(wanted).unbuildable if wanted else ""
    except KeyError:
        # An unregistered task or robot is not this check's business: the
        # load itself fails on it and says which name, with the stack that
        # found it. Guessing here would replace that with a worse message.
        return ""


def refuse(session, label: str, message: str, scene: str, embodiment: str, gripper: str) -> bool:
    """Report a load that will not be attempted, and leave the page alone.

    What is shared is the *reporting*, not the decision -- see
    :class:`~robosimstudio_studio.session.Refused` for why a refusal prints one
    sentence where a compile that broke prints a stack. Both must put the menu
    back, or the next drain requests the same failing load and the one after
    that.
    """
    session.status = f"{_pairing(label, scene, embodiment, gripper)} was not loaded -- {message}"
    print(f"[studio] {label} was not loaded: {message}")
    if session.task_panel is not None:
        session.task_panel.report_failure(label, message)
    # Nothing loaded, so there is no narrowing to report; leaving it set
    # would attach this refusal to whatever loads next.
    session._narrowed = ""
    return False


# --------------------------------------------------------------------- fleet
def load_fleet(
    session,
    task: str | TaskSpec,
    episode: int,
    label: str,
    t0: float,
    scene: str = "",
    embodiment: str = "",
    gripper: str = "",
) -> bool:
    """Load the page as a grid of worlds, in the tab already open.

    The fleet borrows the session's viewer rather than opening its own, so
    setting the world count and reloading swaps what the page *is* without
    the browser losing its connection -- which is the whole reason the count
    is a scene parameter here rather than a separate command.
    """
    from .fleet_session import FleetSession

    config = session.config
    try:
        # Resolved as the single-world load resolves it: ``FleetEnv`` refuses an
        # undeclared pairing like ``Env`` does, so an ``(untried)`` option has to
        # arrive as a spec that declares it, and a declared one in its own
        # spelling.
        task, scene_, embodiment_, gripper_ = resolve_pairing(task, scene, embodiment, gripper)
        base_declares(task, scene_, embodiment_, gripper_)
        fleet = FleetSession(
            task,
            num_worlds=session.num_worlds,
            scene=scene_,
            embodiment=embodiment_,
            gripper=gripper_,
            policy=config.fleet_solution,
            solution_kwargs=dict(config.fleet_solution_kwargs),
            episode=int(episode),
            max_worlds=int(config.show_worlds),
            device=config.device,
            loop=bool(config.fleet_loop),
            viewer=session.viewer,
        )
    except Refused as exc:
        return refuse(session, label, str(exc), scene, embodiment, gripper)
    except CannotReplicate:
        # Not a failed load -- a width this scene does not have. The caller
        # answers it by narrowing the page, so it must not be reported as a
        # task that would not compile, and must not put the menu back.
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        session.status = f"{label} x{session.num_worlds} did not load -- {message}"
        print(f"[studio] {label} x{session.num_worlds} did not load:")
        traceback.print_exc()
        if session.task_panel is not None:
            session.task_panel.report_failure(label, message)
        return False

    session.teardown()
    session.fleet = fleet
    session.episode_id = int(episode)
    session.lighting.apply()
    # The task menu is built here too, and that is what makes the world count
    # a *setting* rather than a mode: without it the page that many worlds
    # produce has no way back to one, which is not a unification at all.
    build_fleet_gui(session)
    session.status = f"loaded {label} x{session.num_worlds} worlds in {time.perf_counter() - t0:.1f} s"
    print(f"[studio] {label} x{session.num_worlds} worlds")
    return True


def base_declares(task: str | TaskSpec, scene: str = "", embodiment: str = "", gripper: str = "") -> None:
    """Refuse the width, not the pairing, when a fleet base does not declare it.

    A document with ``(:fleet-base X)`` is built at more than one world as
    ``X``'s model, pinned to the document's pairing, and
    :func:`~robosimstudio_bench.run.fleet.fleet_build_task` insists the base
    declares that pairing -- an instance on another robot is an instance of
    another model. An ``(untried)`` option arrives here as a spec that declares
    the pairing *itself* (:func:`resolve_pairing`), so it passes ``FleetEnv``'s
    own check and then dies on the base's ``binding_for`` as a ``KeyError``.
    The single-world load never consults the base, so this is a width the
    pairing does not have, and the page narrows to one world rather than
    reporting a load that failed.

    Raises:
        CannotReplicate: When the base is unknown or does not declare the pairing.
    """
    from robosimstudio.specs import replace
    from robosimstudio_bench.run.fleet import fleet_build_task

    spec = get_task(task) if isinstance(task, str) else task
    if not getattr(spec, "fleet_base", ""):
        return
    if scene or embodiment or gripper:
        spec = replace(
            spec,
            scene=scene or spec.scene,
            embodiment=embodiment or spec.embodiment,
            gripper=gripper or spec.gripper,
        )
    try:
        fleet_build_task(spec)
    except KeyError as exc:
        raise CannotReplicate(
            f"its fleet base {spec.fleet_base!r} does not declare this pairing ({exc.args[0] if exc.args else exc})"
        ) from exc


def build_fleet_gui(session) -> None:
    """The page's controls when it is showing a grid.

    The task menu, unchanged -- family, variant, pairing, worlds, reload --
    followed by what a grid has instead of a scene's controls: the wave, its
    pacing, and how tightly the worlds are packed. The panels that drive *a*
    scene are simply not built; they have nothing to act on.
    """
    gui = session.server.gui
    gui.configure_theme(control_layout="collapsible", dark_mode=session.config.dark_mode)
    session.task_panel = PanelTask(session)
    session.panels = [session.task_panel]
    session.task_panel.build(gui)
    session.fleet.build_gui(gui)
    with gui.add_folder("Appearance", expand_by_default=False):
        spacing = gui.add_slider(
            "World spacing", min=0.0, max=8.0, step=0.05, initial_value=round(session.fleet.view.spacing, 2)
        )
        # Live, unlike the world count: offsets are applied at draw time and
        # move nothing in the simulation, so there is nothing to rebuild. The
        # callback only assigns; the next draw applies it on the main thread.
        spacing.on_update(lambda _: setattr(session.fleet.view, "pending_spacing", float(spacing.value)))
        session.lighting.build(gui)


# -------------------------------------------------------------------- panels
def document_panels(session) -> tuple[PanelTask, PanelGoal, PanelSolution]:
    """The three panels that need a *document*: the task menu, the goal readout, the policy runner.

    Built here because each of them names the benchmark, and handed back rather
    than assigned because where they sit in the sidebar -- and whether the
    selection keeps them at all -- is the session's to decide.

    Returns:
        ``(task_panel, goal_panel, driver)``, in the order the page reads them.
    """
    config = session.config
    driver = PanelSolution(
        session,
        settle_steps=config.settle_steps,
        privileged=config.privileged,
        roll=config.roll,
        motion=config.motion,
    )
    return PanelTask(session), PanelGoal(session), driver


# ---------------------------------------------------------------------- goal
def opening_subject(index, goal):
    """The first thing a goal is about, as the page's opening selection.

    The goal half of
    :meth:`~robosimstudio_studio.session.StudioSession._opening_subject`, which
    says why this and not the first thing in the scene. Falls back to the
    index's first subject when no operand names one the scene has.

    Args:
        index: The page's :class:`~robosimstudio_studio.selection.SceneIndex`.
        goal: The environment's goal tree.
    """
    for leaf in leaves(goal):
        for operand in leaf.about[1]:
            found = index.by_name(operand.split(".", 1)[0])
            if found is not None:
                return found
    return index.first()


def feature_payload(context, subject, record, spec) -> dict[str, Any] | None:
    """One declared feature's payload in world coordinates, through the goal's reader.

    :func:`~robosimstudio_bench.goals.bind.read` with ``form="feature"`` is the
    reader every *measure* resolves its operands through, so what the overlay
    draws from this is literally what is measured, in the frame it is measured
    in.

    Args:
        context: The step's evaluation context -- ``env.last_context``.
        subject: The :class:`~robosimstudio_studio.selection.Subject` the
            feature hangs off.
        record: That subject's asset declaration.
        spec: The feature, as the record declares it.

    Returns:
        The payload, or ``None`` for anything unplaceable -- a feature on a
        body that has no pose, a snapshot not taken yet. A drawing that cannot
        be made is a drawing that is not made.
    """
    operand = Operand(
        ref=Ref(base=subject.name, path=(spec.name,)),
        asset=record.name,
        substrate=record.substrate,
        kind=spec.kind,
        feature=spec,
    )
    try:
        return read(Snapshot(ctx=context), operand, "feature")
    except Exception:
        return None


def feature_holds(goal, metrics, wanted: str) -> bool:
    """Whether a goal clause about ``wanted`` currently holds.

    Free: the verdict is already in ``metrics`` under the clause's label, so
    this is a lookup per leaf rather than an evaluation.

    Args:
        goal: The environment's goal tree.
        metrics: The last result's metrics table.
        wanted: The operand, as ``"<subject>.<feature>"``.
    """
    return any(wanted in leaf.about[1] and metrics.get(leaf.label) for leaf in leaves(goal))
