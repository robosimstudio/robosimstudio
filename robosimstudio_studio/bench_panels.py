# SPDX-License-Identifier: Apache-2.0
"""The controls a *task* page has and a bare scene does not: the task menu, the policy runner.

Both need a document. The menu lists what the registry resolves and offers the
pairings ``admits`` will let you try; the runner steps one of the policies the
task ships, exactly the way a headless rollout steps it. A scene somebody
assembled by hand has no variants to switch between and no shipped policy to
run, so the session builds neither over one
(:meth:`~robosimstudio_studio.session.StudioSession.build_panels`).

That makes this module part of the page's benchmark half -- one of
:data:`robosimstudio_studio.seams.ADDONS` -- and it names the benchmark tier at
module scope, which no other panel module may. It is reached through
:mod:`~robosimstudio_studio.seams` and through the lazy names on
:mod:`robosimstudio_studio` and :mod:`robosimstudio_studio.controls`, where
these classes lived until the page learned to open without a benchmark.

They obey the same threading contract as every other panel: callbacks assign
plain attributes and nothing else, and every Warp, GL and file touch happens on
the main thread inside ``drain``.
"""

from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio_bench.taskbuild.registry import list_tasks

from .panels import Panel

__all__ = ["ROUTINE_CHOICE", "PanelSolution", "PanelTask"]


def _label(options: dict[str, str], value: str) -> str:
    """The label a dropdown should start on for one axis value.

    Falls back to the first option rather than raising: a task whose loaded
    pairing is somehow not among its declared ones would otherwise take the page
    down over a menu default.
    """
    for label, candidate in options.items():
        if candidate == value:
            return label
    return next(iter(options))


def _families() -> dict[str, list[str]]:
    """Registered tasks as ``{suite: [variant, ...]}``, both in registry order.

    A task id is already two things joined by a slash -- the suite, which decides
    what the scene is, and the variant, which decides what is being asked of it.
    This is that split made explicit, and it is derived from
    :func:`~robosimstudio_bench.taskbuild.registry.list_tasks` rather than from the
    suite registry so the menu cannot offer a pairing the registry would not
    resolve.
    """
    families: dict[str, list[str]] = {}
    for full in list_tasks():
        family, variant = full.split("/", 1)
        families.setdefault(family, []).append(variant)
    return families


class PanelTask(Panel):
    """The task menu: pick one and the page loads it.

    Selecting *is* the command -- there is no second button to press, because
    the only thing a two-step "choose, then load" buys is a chance to choose
    something you did not want, and the page has a Reload for the one case a
    button is right (rebuilding what is already selected).

    The load itself is deferred to the loop rather than done in the callback:
    it compiles a scene and swaps the viewer's model, which is Warp work, and
    viser fires callbacks on its own server threads. The callback sets a flag;
    :meth:`drain` -- main thread -- turns it into a request the loop services.
    A compile costs seconds to a minute, so the readout says what is loading
    before the loop blocks on it, and that message reaches the browser while
    the main thread is busy because viser's transport is not on it.

    **The episode is here**, and it is the one axis of the pairing that does not
    reload: it goes through ``env.reset``, which rebuilds the model only when the
    new episode needs a different one. It sits with the scene and the robot
    because that is what it is -- part of what the page has open.
    """

    select_requested = False
    reload_requested = False
    run_requested = False
    family_changed = False
    """Set by the Family dropdown; ``drain`` turns it into a new Variant list.

    Class attributes so a drain that somehow beats ``build`` reads False rather
    than raising -- the panel is composed before the GUI is.
    """
    embodiment_changed = False
    """Set by the Embodiment dropdown; ``drain`` re-homes the Gripper first.

    A hand is chosen *for* a robot -- an attachable hand bolts to one named
    flange -- so carrying one robot's hand onto another makes the very pairing
    the change asks for inadmissible before anything is tried. Falling back to
    the new robot's own hand is the same move a family change makes on the
    variant list.
    """
    episode_changed = False
    """Set by the Episode field; ``drain`` turns it into a reset."""
    snapshot_requested = False
    """Set by Save snapshot; ``drain`` writes the file.

    In this folder rather than under Record because it is not a recording: a
    take is an episode you meant to keep, and this is one step you did not
    understand. It sits with the thing it is about -- whatever is loaded.
    """
    repro_requested = False
    """Set by Export repro; ``drain`` writes the bundle.

    Beside the snapshot because it is the same gesture escalated: a snapshot
    is one step written down for *us*, a repro bundle is the whole run made
    runnable for the Newton developers, with the studio withheld.
    """
    _episode = None
    """The field itself, absent on a grid, where there is no one episode for it
    to be about."""

    def build(self, gui) -> None:
        self.select_requested = False
        self.reload_requested = False
        self.run_requested = False
        self.family_changed = False
        self.embodiment_changed = False
        self._families = _families()
        task = self.session.current_task
        current = task.full_name
        family, variant = current.split("/", 1)
        with gui.add_folder("Task"):
            # Two dropdowns rather than one list of every id, because a task name
            # is already two things: the suite, which decides what the scene *is*,
            # and the variant, which decides which question is being asked of it.
            # A flat menu makes finding the second one a scan of every one of the
            # first, and it grows the wrong way -- a suite that gains a variant
            # lengthens a list that every other suite has to be read past.
            self._family = gui.add_dropdown("Family", options=tuple(self._families), initial_value=family)
            self._variant = gui.add_dropdown("Variant", options=tuple(self._families[family]), initial_value=variant)
            # Changing family retargets the variant list and selects that suite's
            # first variant, which is a real selection -- so it falls through to
            # the same request the variant dropdown makes.
            self._family.on_update(lambda _: setattr(self, "family_changed", True))
            self._variant.on_update(lambda _: setattr(self, "select_requested", True))
            self._scenes = self._axis_options(task, axis="scene")
            self._embodiments = self._axis_options(task, axis="embodiment")
            self._grippers = self._axis_options(task, axis="gripper")
            self._scene = gui.add_dropdown(
                "Scene", options=tuple(self._scenes), initial_value=_label(self._scenes, task.scene)
            )
            self._scene.on_update(lambda _: setattr(self, "select_requested", True))
            self._embodiment = gui.add_dropdown(
                "Embodiment",
                options=tuple(self._embodiments),
                initial_value=_label(self._embodiments, task.embodiment),
            )
            self._embodiment.on_update(lambda _: setattr(self, "embodiment_changed", True))
            self._gripper = gui.add_dropdown(
                "Gripper",
                options=tuple(self._grippers),
                initial_value=_label(self._grippers, task.gripper),
            )
            self._gripper.on_update(lambda _: setattr(self, "select_requested", True))
            # A fourth axis of the pairing, and the same shape as the other
            # three: set it, reload, get that scene. One is the interactive
            # page; above one the page is a grid of worlds each running its own
            # episode, and the tools that drive *a* scene go away with it.
            #
            # Deliberately not live: a world count change rebuilds the model, so
            # it waits for Reload the way a scene or a robot does, rather than
            # recompiling on every keystroke.
            self._worlds = gui.add_number(
                "Worlds", initial_value=int(getattr(self.session, "num_worlds", 1)), min=1, max=4096, step=1
            )
            # Only on a page showing one scene. A grid runs a different episode
            # in every world by construction, so there is no one episode there
            # for a field to be about.
            self._episode = None
            if self.session.env is not None:
                count = max(1, int(getattr(self.session.env.task.episodes, "count", 1)))
                self._episode = gui.add_number(
                    "Episode", initial_value=min(int(self.session.episode_id), count - 1), min=0, max=count - 1, step=1
                )
                self._episode.on_update(lambda _: setattr(self, "episode_changed", True))
            reload_ = gui.add_button("Reload")
            reload_.on_click(lambda _: setattr(self, "reload_requested", True))
            # Beside the task rather than only in the Policy folder: "load this
            # and run what it ships" is one gesture, and it is the first thing
            # anyone does with a task they have just picked. Which policy is
            # still the Policy folder's to say -- this runs whatever is selected
            # there, and that panel reports it if nothing is.
            run = gui.add_button("Run policy")
            run.on_click(lambda _: setattr(self, "run_requested", True))
            # Deliberately takeable while paused, while a policy runs, and
            # mid-drag: the states worth a snapshot are the ones you cannot
            # reproduce, and a button that needed the scene to be idle first
            # would only ever capture the states that were already fine.
            snap = gui.add_button("Save snapshot")
            snap.on_click(lambda _: setattr(self, "snapshot_requested", True))
            # Same policy as the snapshot: takeable whenever, because the run
            # worth shipping to a Newton developer is the one going wrong right
            # now. The export drains the repro tape, which has been recording
            # since the last reset regardless of what pressed this.
            # Only where the replay package is installed: without it the export
            # has nothing to write with, and a button that always fails is noise.
            has_repro = importlib.util.find_spec("robosimstudio_repro") is not None
            if has_repro:
                repro = gui.add_button("Export repro")
                repro.on_click(lambda _: setattr(self, "repro_requested", True))
            self._readout = gui.add_text(
                "Loaded", initial_value=f"{current} episode {self.session.episode_id}", disabled=True
            )
            self._snapshot = gui.add_text("Snapshot", initial_value="none taken", disabled=True)
            self._repro = gui.add_text("Repro", initial_value="none exported", disabled=True) if has_repro else None

    def _axis_options(self, task, *, axis: str) -> dict[str, str]:
        """Label to value for one axis: the declared pairings, then the admissible ones.

        Declared pairings come first and unannotated -- somebody has compiled
        those. After them come the pairings
        :func:`~robosimstudio_bench.taskbuild.matrix.admits` will *let* you try:
        checked on what the specs can answer (arm count, a raised surface, a
        gripper span) and on nothing that needs measuring. They are marked
        ``(untried)`` because that is exactly what they are, and they are worth
        offering because trying one is the point of a page you can drive by
        hand -- a robot that has never held this task's objects is a question,
        and the answer is a drag on its gizmo away.

        Each candidate is tested one swap away from the pairing the page is
        holding -- ``task.scene / embodiment / gripper``, which the load wrote
        the selection into -- not from the native pairing. The panel is rebuilt
        with every load, so this is what makes the lists answer for *this*
        page: tested against the native pairing, loading the UR5 still showed
        an empty gripper menu, because the two hands that bolt to its flange
        were being checked against a YAM's wrists, which can take neither.

        What is *not* offered is what ``admits`` rejects -- which includes a
        robot that says it cannot be compiled yet, so a half-landed embodiment
        stays out of the menu instead of being a traceback one click away.
        viser's dropdown
        disables as a whole and has no per-option flag, so an inadmissible
        combination cannot be shown struck through -- it can only be absent,
        and absent is the honest rendering of a pairing that cannot be built.

        A pairing that ships no policy says so in its label. It is still
        selectable -- it compiles and can be teleoperated, which is most of what
        this page is for -- but a reader should not have to open MATRIX.md to
        find out that pressing Run will offer nothing.
        """
        from robosimstudio.robots.registry import attachable_grippers, list_embodiments
        from robosimstudio.scene.registry import list_cells
        from robosimstudio.specs import Binding
        from robosimstudio_bench.taskbuild.matrix import admits, pairing_key

        slot = ("scene", "embodiment", "gripper").index(axis)
        graphs: dict[str, bool] = {}
        for binding in task.bindings or (task.native_binding,):
            value = binding.key(task)[slot]
            # Keyed by value, not by label: several bindings share one scene
            # when only the embodiment differs, and keying by label listed that
            # scene once plainly and once annotated. A value carries a graph if
            # *any* pairing using it does.
            graphs[value] = graphs.get(value, False) or bool(binding.graphs)
        options = {
            (value or "(task default)") + ("" if has else "  (no graph)"): value for value, has in graphs.items()
        }

        declared = set(graphs)
        current = (task.scene, task.embodiment, task.gripper)
        catalogue = {"scene": list_cells, "embodiment": list_embodiments, "gripper": attachable_grippers}[axis]
        # A hand is the same pairing spelled two ways -- ``""`` and the hand the
        # robot already wears -- so the catalogue's name for a declared
        # binding's own hand is that binding, not an untried one beside it.
        same = set()
        if axis == "gripper":
            for binding in task.bindings or (task.native_binding,):
                try:
                    same.add(pairing_key(task, binding))
                except LookupError:
                    pass  # a robot the registry does not know cannot match one
        for value in catalogue():
            if value in declared:
                continue
            pairing = list(current)
            pairing[slot] = value
            if same:
                try:
                    if pairing_key(task, Binding(scene=pairing[0], embodiment=pairing[1], gripper=value)) in same:
                        continue
                except LookupError:
                    pass
            if admits(task, *pairing)[0]:
                options[f"{value}  (untried)"] = value
        return options

    def drain(self) -> None:
        if self.episode_changed:
            # A reset rather than a load: the scene is already compiled, and an
            # episode that needs a different model is `env.reset`'s business to
            # notice. The guard is what makes `on_reset` able to write the field
            # back without the write asking for the reset it is reporting.
            self.episode_changed = False
            wanted = int(self._episode.value) if self._episode is not None else int(self.session.episode_id)
            if wanted != int(self.session.episode_id):
                self.session.episode_id = wanted
                self.session.reset_requested = True
        if self.run_requested:
            # Forwarded rather than started here: the policy panel owns building
            # and resetting, and this panel is drained *before* it, so the flag
            # is serviced on this same control step rather than the next one.
            self.run_requested = False
            driver = getattr(self.session, "driver", None)
            if driver is not None:
                driver.start_requested = True
        if self.snapshot_requested:
            # Main thread, and it has to be: this reads body poses, contacts and
            # the shape table back off the device, and viser fires its callbacks
            # on server threads.
            self.snapshot_requested = False
            self._save_snapshot()
        if self.repro_requested:
            # Main thread for the same reason: the export walks every model
            # array back off the device.
            self.repro_requested = False
            self._export_repro()
        if self.family_changed:
            # Retarget the variant list before anything reads it. Assigning
            # ``options`` fires the variant dropdown's own update, which sets
            # select_requested -- so the load happens on this same drain.
            self.family_changed = False
            family = str(self._family.value)
            variants = self._families.get(family, [])
            if variants:
                self._variant.options = tuple(variants)
                if str(self._variant.value) not in variants:
                    self._variant.value = variants[0]
                self.select_requested = True
        if self.embodiment_changed:
            # Re-home the gripper before the selection is read: the hand on the
            # menu was admitted against the *previous* robot, and may not bolt
            # onto the one just picked. Guarded on the robot actually changing,
            # so report_failure's reset of this dropdown -- which fires the
            # same event -- keeps the loaded hand rather than discarding it.
            self.embodiment_changed = False
            embodiment = self._embodiments.get(str(self._embodiment.value), "")
            if embodiment != self.session.current_task.embodiment and self._grippers.get(str(self._gripper.value), ""):
                self._gripper.value = _label(self._grippers, "")
            self.select_requested = True
        if not (self.select_requested or self.reload_requested):
            return
        selected = f"{self._family.value}/{self._variant.value}"
        scene = self._scenes.get(str(self._scene.value), "")
        embodiment = self._embodiments.get(str(self._embodiment.value), "")
        gripper = self._grippers.get(str(self._gripper.value), "")
        task = self.session.current_task
        unchanged = (
            selected == task.full_name
            and scene == task.scene
            and embodiment == task.embodiment
            and gripper == task.gripper
        )
        if self.select_requested and not self.reload_requested and unchanged:
            self.select_requested = False
            return  # a dropdown fired without changing anything
        episode = self.session.episode_id
        if selected != task.full_name:
            # Another task's pairings are its own, and the three dropdowns still
            # show this one's. Load it on its native pairing; its own options
            # arrive with the panel the load rebuilds. Its episodes are its own
            # too, and it may have fewer than the one this task is on.
            scene = embodiment = gripper = ""
            episode = 0
        self.select_requested = self.reload_requested = False
        pairing = " @ ".join(x for x in (selected, " / ".join(y for y in (scene, embodiment, gripper) if y)) if x)
        self._readout.value = f"loading {pairing} ..."
        self.session.status = f"loading {pairing}"
        # The session rebuilds this panel as part of the load, so nothing after
        # this line may touch self.
        self.session.load_request = (
            selected,
            episode,
            scene,
            embodiment,
            gripper,
            int(self._worlds.value),
        )

    def _save_snapshot(self) -> None:
        """Write one step of the loaded scene to ``<run>/snapshots/``.

        ``evaluate=False``: the goal panel is the one caller allowed to score a
        step, because scoring latches ``once`` and advances ``hold-for``, and a
        button that quietly satisfied a five-second hold by being pressed would
        be worse than no button. The readings the panel is already showing are
        the ones to read off the page.

        Failures are reported into the readout rather than raised. A snapshot is
        a debugging aid, and taking the studio down while somebody is trying to
        work out why a scene is wrong is the least useful thing it could do.
        """
        from robosimstudio.scene.snapshot import summarise, write_snapshot

        env = self.session.env
        if env is None:
            self._snapshot.value = "nothing loaded"
            return
        try:
            path = write_snapshot(env, self.session.run_dir / "snapshots", note=self.session.status)
            summary = summarise(json.loads(path.read_text()))
        except Exception as failure:
            self._snapshot.value = f"failed -- {failure}"[:200]
            return
        self._snapshot.value = f"{path.name}: {summary}"[:200]

    def _export_repro(self) -> None:
        """Write a repro bundle of the loaded run to ``<run>/repro/``.

        Same failure policy as the snapshot: reported into the readout, never
        raised. Refusals are the interesting failures here — a task whose
        physics needs its declared standalone module says so in this box, and
        that message is the instruction for what to declare.
        """
        env = self.session.env
        if env is None:
            self._repro.value = "nothing loaded"
            return
        try:
            path = env.export_repro(self.session.run_dir / "repro")
            steps = len(getattr(env, "repro_tape", []) or [])
            size_mb = sum(f.stat().st_size for f in path.iterdir()) / 1e6
            self._repro.value = f"{path.name}: {steps} steps, {size_mb:.1f} MB"[:200]
        except Exception as failure:
            self._repro.value = f"failed -- {failure}"[:200]

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs
        # An episode load is a reset rather than a task load, so this is where
        # the menu learns which episode is actually in the page. Written back
        # rather than assumed, because most resets do not come from this field:
        # the transport's Reset, a policy starting, a task load. A field naming
        # an episode the page is not running is worse than no field.
        self._readout.value = f"{self.session.current_task.full_name} episode {info.get('episode_id', '?')}"
        loaded = info.get("episode_id")
        if self._episode is not None and isinstance(loaded, int):
            self._episode.value = loaded

    def note(self, message: str) -> None:
        """Append a qualification to the readout of a load that *succeeded*.

        Distinct from :meth:`report_failure`, which puts the menu back: this is
        for a load that happened but not on the terms asked for, so the menu
        keeps the selection and only the readout says what was different.
        """
        self._readout.value = f"{self._readout.value} ({message})"[:200]

    def report_failure(self, task: str, message: str) -> None:
        """Say that a selection did not load, and put the menu back.

        Every dropdown is returned to what the page is actually running, not
        only the task one: a menu showing a selection that is not loaded is
        worse than no menu, and leaving the scene or embodiment on a pairing
        that just failed makes the next drain request the same failing load
        again, and the one after that. The assignments' own update events are
        absorbed by the no-op guard in :meth:`drain`.
        """
        self._readout.value = f"{task} failed -- {message}"[:200]
        current = self.session.current_task
        # ``rpartition``, not ``split("/", 1)``: a name with no slash in it is a
        # ValueError there, and the menu putting itself back is exactly the path
        # that must not raise -- a failed load would take the page with it. An
        # unregistered variant is also prepended to its own dropdown, because
        # viser refuses a dropdown whose initial value is not among its options.
        family, _, variant = current.full_name.rpartition("/")
        options = tuple(self._families.get(family, ()))
        self._family.value = family
        self._variant.options = options if variant in options else (variant, *options)
        self._variant.value = variant
        self.family_changed = False
        self._scene.value = _label(self._scenes, current.scene)
        self._embodiment.value = _label(self._embodiments, current.embodiment)
        self.embodiment_changed = False
        self._gripper.value = _label(self._grippers, current.gripper)


ROUTINE_CHOICE = "routine"
"""The shipped scripted policy, as it appears in the policy menu."""


class PanelSolution(Panel):
    """Run any of this task's policies -- its graphs or its scripted routine.

    One control rather than one per kind, because a graph and a routine differ in
    what they *are* and not in how an episode is driven: both are a
    :class:`~robosimstudio_bench.solutions.SolutionBase`, and both get stepped
    here exactly the way :func:`~robosimstudio_bench.run.rollout` steps them --
    ``env.step(policy.get_action(obs), mode=policy.control_mode)``, which
    evaluates, renders and terminates. That identity is the point of the panel: a
    policy that behaved differently here than under ``scripts/bench_run.py`` would make
    this page a liar about the numbers it shows.

    The two kinds still differ in what they need, and both are handled:

    * A graph runs the pinned runtime on its own thread and drives the
      environment through the connector's ``robot.*`` tools, so this loop pumps
      the bus between steps to keep one writer, and the policy owns a bus that
      must be closed. Privileged ``sim.*`` tools are on by default, as they are
      there -- a graph that may ask where the hammer is has not been evaluated on
      finding it -- and the checkbox is how you check it runs on perception alone.
    * A routine is pure per-step computation with nothing to close, and may
      contribute substep hooks (the peel's particle latch), which its own
      ``reset`` releases when this panel stops it.

    The graph-only controls below (privileged, settle, workflow dir) are ignored
    when the routine is selected rather than hidden, because a dropdown whose
    neighbours appear and disappear is harder to read than one whose neighbours
    are sometimes irrelevant.
    """

    def __init__(
        self,
        session,
        *,
        settle_steps: int = 60,
        privileged: bool = True,
        roll: str = "auto",
        motion: str = "",
    ) -> None:
        super().__init__(session)
        # What this *pairing* supports, not every policy the task ships -- see
        # :func:`robosimstudio_bench.solutions.declared.declared_solutions`, which
        # is also what the record tool and the certifier ask.
        self.settle_steps = int(settle_steps)
        self.privileged = bool(privileged)
        self.roll = str(roll or "auto")
        self.motion = str(motion)
        self.policy: Any = None
        self.running = False
        self.runs = 0
        self.steps = 0
        self.run_dir = session.run_dir
        self.start_requested = False
        self.stop_requested = False
        self.message = ""
        self.suspended: list[Any] = []

    @property
    def choices(self) -> tuple[str, ...]:
        """Every policy this *pairing* supports, routine first."""
        from robosimstudio_bench.solutions.declared import ROUTINE, declared_solutions

        task = self.env.task
        declared = declared_solutions(task, task.binding_for())
        return tuple(ROUTINE_CHOICE if label == ROUTINE else label for label in declared)

    @property
    def control_mode(self) -> str:
        """The action space the running policy emits into, or ``""``."""
        if self.policy is None:
            return ""
        return self.policy.control_mode or self.env.control_mode

    def build(self, gui) -> None:
        with gui.add_folder("Policy"):
            options = self.choices or ("(none for this pairing)",)
            self._choice = gui.add_dropdown("Policy", options=options, initial_value=options[0])
            self._custom = gui.add_text("Workflow dir", initial_value="")
            self._privileged = gui.add_checkbox("Privileged sim.* tools", initial_value=self.privileged)
            self._settle = gui.add_number("Settle steps", initial_value=self.settle_steps, min=0, step=10)
            run = gui.add_button("Run")
            run.on_click(lambda _: setattr(self, "start_requested", True))
            stop = gui.add_button("Stop")
            stop.on_click(lambda _: setattr(self, "stop_requested", True))
            self._state = gui.add_text("Policy", initial_value=self._initial_message(), disabled=True)
            self._progress = gui.add_text("Progress", initial_value="", disabled=True)

    def _initial_message(self) -> str:
        if self.choices:
            return f"ships {', '.join(self.choices)}"
        return "ships no policy -- give a workflow directory"

    # ---------------------------------------------------------------- lifecycle
    def drain(self) -> None:
        if self.stop_requested:
            self.stop_requested = False
            self.stop("stopped")
        if self.start_requested:
            self.start_requested = False
            self.start()

    def start(self, selected: str = "") -> None:
        """Reset the episode and hand the arm to the selected policy.

        The reset is not optional: a policy plans from the scene it is given, and
        one that starts from wherever teleop left the arm is being asked a
        different question than the one ``scripts/bench_run.py`` asks.

        Args:
            selected: The policy by name -- :data:`ROUTINE_CHOICE`, a graph
                label, or a workflow directory -- for a caller with no page,
                such as the certifier stepping a task headlessly. Empty reads
                the page's own menu and its custom-path field.
        """
        if not selected:
            custom = str(self._custom.value).strip()
            selected = custom or (str(self._choice.value) if self.choices else "")
        if not selected or selected.startswith("("):
            self._state.value = "no policy: pick one or give a workflow directory"
            return
        self.stop("")
        try:
            self.runs += 1
            self.run_dir = self.session.run_dir / f"run_{self.runs:03d}"
            self.policy = self._build(selected)
        except Exception as exc:
            # Two common cases, both reported rather than raised: the gap runtime
            # is a pinned checkout that may not be importable, and a routine may
            # want a controller this embodiment has no way to build.
            self.policy = None
            self._state.value = f"{type(exc).__name__}: {exc}"[:200]
            return

        session = self.session
        model = getattr(session.env, "model", None)
        try:
            # An episode whose model key differs recompiles the scene in here,
            # as in ``reset_now``, and a failed recompile pauses the page the
            # same way rather than ending its loop.
            obs, info = session.reset_env()
        except Exception as exc:
            self.stop(f"{type(exc).__name__}: {exc}"[:200])
            session.set_paused(True)
            print(f"[studio] episode {session.episode_id} did not reset for the policy; paused:")
            traceback.print_exc()
            return
        if getattr(session.env, "model", None) is not model:
            # The Run was forwarded in the same drain that asked for a new
            # episode, and this reset is the one that recompiled: the page is
            # rebuilt over the new model the way ``reset_now`` rebuilds it --
            # which retires this panel with the rest -- and the policy, built
            # against the old one, is not started on it.
            self.stop("the episode recompiled the scene; press Run again")
            session._drop_panels()
            session._bind(session.env)
            session.obs = obs
            session.on_reset(obs, info)
            session.status = self.message
            return
        session.obs = obs
        # Panels first, then the policy: on_reset re-syncs the gizmos to the arm
        # the reset just moved, and this panel's own on_reset is a no-op while
        # ``running`` is still False -- which is why it is set last.
        self.session.on_reset(obs, info)
        self.policy.reset(obs, info)
        # The policy owns the arm; a gizmo move or gizmo-tracking IK on top of
        # it would fight every waypoint it commands. Only the arms still
        # driven by their gizmos: one a leader holds is the leader's to give
        # back, as in ``routine.py``.
        self.suspended = [panel for panel in self.session.arms.values() if not panel.suspended]
        for panel in self.suspended:
            panel.suspend()
        self.running = True
        self.steps = 0
        self.message = f"running {Path(selected).name} ({self.control_mode})"
        self._state.value = self.message
        self.session.status = self.message

    def _build(self, selected: str):
        """The policy named by the menu: the shipped routine, or a graph."""
        if selected == ROUTINE_CHOICE:
            # Offered only where the routine module is installed (see ``choices``),
            # so this names no module a graphs-only release lacks.
            from robosimstudio_bench.solutions.base import load_solution
            from robosimstudio_bench.solutions.declared import ROUTINE, solution_source

            path, kwargs = solution_source(ROUTINE)
            return load_solution(path, self.env, **kwargs)
        from robosimstudio_policy_agent.graph_solution import SolutionGraph

        return SolutionGraph(
            self.env,
            graph=selected,
            privileged=bool(self._privileged.value),
            settle_steps=int(self._settle.value),
            trace_dir=str(self.run_dir),
            roll=self.roll,
            motion=self.motion,
        )

    def step(self) -> tuple[np.ndarray | None, str]:
        """One control period driven by the policy -- ``rollout``'s step, exactly.

        Including the mode tag: the policy names the action space it emits into,
        so a scripted Cartesian routine and a joint-space graph run on this one
        scene without either dictating the other's action space.

        Returns:
            What the policy emitted and the mode it was in, for the recorder.
            Returned rather than left on the panel because the episode may end
            inside this call, and a take must keep the action that drove its
            last step rather than the state a stopped policy left behind.
        """
        env = self.env
        try:
            mode = self.policy.control_mode
            action = self.policy.get_action(self.session.obs)
            obs, reward, terminated, truncated, info = env.step(action, mode=mode)
            self.session.obs = obs
            self.policy.on_step(obs, reward, info)
        except Exception as exc:
            # Reported the way a build failure is, and the arm given back:
            # raised, it would end the page's loop, or on a page that catches
            # it, fire again every tick with the gizmos still suspended.
            print("[studio] the policy failed:")
            traceback.print_exc()
            self.stop(f"{type(exc).__name__}: {exc}"[:200])
            return None, env.control_mode
        self.steps += 1
        if terminated or truncated:
            self.finish(info)
        return (None if action is None else np.asarray(action, dtype=np.float32)), mode or env.control_mode

    def finish(self, info: dict[str, Any]) -> None:
        """The episode ended under the policy. Report the verdict.

        The graph's trace is the policy's own to keep -- ``SolutionGraph`` writes
        it into ``<run>/<episode>/graph_result.json`` when it is closed, which is
        the same place and the same file a headless run produces.
        """
        verdict = "SUCCESS" if info.get("success") else "no success"
        close_kind = str(info.get("close_kind", "unknown"))
        self.stop(f"{verdict} in {self.steps} steps ({close_kind})")

    def stop(self, message: str) -> None:
        """Release whatever the policy held and give the arm back to the gizmos.

        ``close`` is the uniform seam: a graph policy joins its worker thread and
        closes its bus, a scripted one has nothing to do, and a routine that
        latched onto a deformable releases the latch through its own reset.
        """
        if self.policy is not None:
            routine = getattr(self.policy, "routine", None)
            if routine is not None and hasattr(routine, "reset"):
                routine.reset()
            self.policy.close()
            self.policy = None
        was_running, self.running = self.running, False
        if was_running:
            for panel in self.suspended:
                panel.resume()
        self.suspended = []
        if message:
            self.message = message
            self._state.value = message
            self.session.status = message

    def refresh(self) -> None:
        if not self.running:
            return
        # A routine narrates which leg it is flying and which object it is on;
        # a graph's own progress is its trace. Whatever the policy says, if it
        # says anything, is worth more than a step count on its own.
        routine = getattr(self.policy, "routine", None)
        said = " | ".join(str(v) for v in (getattr(routine, "phase", ""), getattr(routine, "message", "")) if v)
        progress = f"{self.steps} steps, {self.env.step_count} in episode"
        self._progress.value = f"{progress} -- {said}" if said else progress

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        # A reset from anywhere else -- the Reset button, an episode load --
        # invalidates a running policy, which planned against the scene it saw.
        if self.running:
            self.stop("stopped by a reset")

    def close(self) -> None:
        super().close()
        self.stop("")
