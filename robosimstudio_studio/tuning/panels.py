# SPDX-License-Identifier: Apache-2.0
"""Material tuning, as panels the studio mounts beside everything else.

These sit in the main studio rather than on a page of their own, and that is the
whole point. A bare strip can tell you a bending length; it cannot tell you
whether the shirt folds, whether the jaws keep hold of it, or whether the graph
gets through its second crease. Those are the questions worth tuning against, and
they only exist in a task with the real mesh, the robot and the routine running
-- so the sliders belong next to the policy runner and the episode score.

The tuning state itself is in :mod:`.tuner`, which knows nothing about viser.
What is here is the page: which cloth, which preset, what each change costs, and
the one button that pays the cost a slider cannot.

Threading is the studio's own contract -- a viser callback assigns an attribute
and nothing else; :meth:`Panel.drain` does the work on the main thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robosimstudio.assetlib.materials import MaterialRecord, list_materials, write_material

from .arrangement import ArrangementBinder
from .binder import TIER_MARK, SpecBinder
from .metrics import declared_states, measure_sheet, relax
from .tuner import MaterialTuner

__all__ = ["PanelReadings", "PanelTune"]


class Panel:
    """The studio's panel protocol, duck-typed to avoid importing its module."""

    @property
    def profile_name(self) -> str:
        """What this panel is called in a profile of the studio's loop.

        Part of the protocol rather than of the studio's own base class for the
        same reason as everything else here: these panels are handed to a page
        that never imports them back. Kept identical to the studio's spelling so
        one phase table reads consistently across both.
        """
        return type(self).__name__.removeprefix("Panel").lower()

    def build(self, gui: Any) -> None: ...
    def drain(self) -> None: ...
    def refresh(self) -> None: ...
    def on_reset(self, obs: Any, info: Any) -> None: ...
    def on_select(self, subject: Any) -> None: ...
    def close(self) -> None: ...


class PanelTune(Panel):
    """Folders ``Material`` and ``Physics``: what the scene is made of and how it is stepped.

    The two halves are gated separately. A scene has an arrangement whatever is
    in it, but only some deformables have a material this panel can tune, so the
    ``Material`` folder can be absent from a page that still wants ``Physics``.
    """

    def __init__(self, session: Any, *, material_dir: str = "assets/materials") -> None:
        self.session = session
        self.tuner = MaterialTuner(session.env)
        self.material_dir = material_dir
        # ``None`` when nothing in the scene has a fabric material -- a SuperDex
        # sponge is bulk matter, and the tuner drops it rather than mapping
        # Young's modulus onto ``tri_materials``.
        self.binder = SpecBinder(self.tuner.material) if self.tuner else None
        # Off the request rather than off a task, for the reason
        # :class:`~robosimstudio_studio.panels.PanelArm` reads its control rate
        # the same way: an arrangement is a fact about how the scene is stepped,
        # and a page over a scene nobody wrote a document for has one.
        self.physics = ArrangementBinder(session.env.request.physics, getattr(session.env.scene, "program", None))
        self.cloth_requested = ""
        self.preset_requested = ""
        self.revert_requested = False
        self.rebuild_requested = False
        self.save_requested = False
        self._cloth: Any = None
        self._preset: Any = None
        self._presets: tuple[str, ...] = ()
        self._calibration: Any = None
        self._pending: Any = None
        self._save_name: Any = None
        self._saved: Any = None

    # ------------------------------------------------------------------ build
    def build(self, gui: Any) -> None:
        if self.tuner:
            self._build_material(gui)
        # Not gated on the tuner, for the same reason the binder above is built
        # off the request: the arrangement is a fact about how the scene is
        # stepped. A scene whose only deformable is a foreign engine's is
        # precisely the one whose backend parameters are worth reaching, and it
        # is the one that has no ``Material`` folder.
        with gui.add_folder("Physics", expand_by_default=False):
            gui.add_markdown(
                "The arrangement this scene was composed from: every backend's parameters and the coupler's. "
                f"`{TIER_MARK['recapture'].strip()}` applies live and recaptures; `{TIER_MARK['rebuild'].strip()}` needs rebuild."
            )
            self.physics.build(gui)
            if not self.tuner:
                self._rebuild_controls(gui)
        self.refresh()

    def _build_material(self, gui: Any) -> None:
        with gui.add_folder("Material", expand_by_default=False):
            if len(self.tuner.names) > 1:
                self._cloth = gui.add_dropdown("Cloth", options=self.tuner.names, initial_value=self.tuner.selected)
                self._cloth.on_update(lambda _: setattr(self, "cloth_requested", str(self._cloth.value)))
            self._presets = ("(as built)", *list_materials())
            self._preset = gui.add_dropdown("Preset", options=self._presets, initial_value=self._presets[0])
            self._preset.on_update(lambda _: setattr(self, "preset_requested", str(self._preset.value)))
            revert = gui.add_button("Revert to as built")
            revert.on_click(lambda _: setattr(self, "revert_requested", True))
            self._calibration = gui.add_text("Calibration", initial_value="--", disabled=True)
            self._rebuild_controls(gui)
            gui.add_markdown(
                f"`{TIER_MARK['recapture'].strip()}` recapture &nbsp; "
                f"`{TIER_MARK['rebuild'].strip()}` needs rebuild &nbsp; rest lands next substep."
            )
            self.binder.build(gui)
            self._save_name = gui.add_text("Name", initial_value=f"{self.tuner.selected or 'cloth'}_tuned")
            save = gui.add_button("Save as preset file")
            save.on_click(lambda _: setattr(self, "save_requested", True))
            self._saved = gui.add_text("Last saved", initial_value="--", disabled=True)

    def _rebuild_controls(self, gui: Any) -> None:
        """What is waiting on a recompile, and the button that pays for it.

        Built into whichever folder is on the page rather than into ``Material``:
        both halves accumulate rebuild-tier edits and one recompile applies them
        together, so a page with only ``Physics`` still needs them.
        """
        self._pending = gui.add_text("Pending rebuild", initial_value="none", disabled=True)
        rebuild = gui.add_button("Apply & rebuild")
        rebuild.on_click(lambda _: setattr(self, "rebuild_requested", True))

    # ------------------------------------------------------------------ drain
    def drain(self) -> None:
        if self.tuner:
            if self.cloth_requested:
                name, self.cloth_requested = self.cloth_requested, ""
                self.tuner.select(name)
                self.binder.set_spec(self.tuner.material)
                self._sync_preset()
                self.refresh()
                return
            if self.preset_requested:
                preset, self.preset_requested = self.preset_requested, ""
                # "(as built)" is the built material, so picking it is a revert
                # rather than nothing.
                if preset == "(as built)":
                    self.revert_requested = True
                else:
                    self.tuner.adopt(preset)
                    self.binder.set_spec(self.tuner.material)
                    self.refresh()
                    return
            if self.revert_requested:
                self.revert_requested = False
                self.tuner.revert()
                self.binder.set_spec(self.tuner.material)
                self._sync_preset()
                self.refresh()
                return
            if self.save_requested:
                self.save_requested = False
                self._save()
                return
        # Outside that guard: with no ``Material`` folder the ``Physics`` one
        # carries this button, and its own rebuild-tier fields are what it is
        # for. ``task_with_edits`` substitutes nothing when there is no cloth,
        # which is the right answer rather than a special case.
        if self.rebuild_requested:
            self.rebuild_requested = False
            self._rebuild()
            return

        changed = self.binder.drain() if self.binder is not None else {}
        for field, value in changed.items():
            self.tuner.set_field(field, value)
        if changed or self.physics.drain():
            self.refresh()

    def _sync_preset(self) -> None:
        """Show the selected cloth's preset in the dropdown.

        Otherwise it keeps naming one the cloth no longer wears, and since a
        dropdown only reports a change, picking that preset again does nothing.
        The assignment fires ``on_update`` like a click does, so the request it
        leaves behind is dropped here rather than adopted next drain.
        """
        if self._preset is None:
            return
        preset = self.tuner.preset_name()
        self._preset.value = preset if preset in self._presets else self._presets[0]
        self.preset_requested = ""

    def _rebuild(self) -> None:
        """Ask the loop to recompile this task with the tuned material.

        Handed over rather than done here: a load calls ``gui.reset()`` and
        builds a fresh set of panels, so a panel that rebuilt inside its own
        drain would return into a loop iterating handles that no longer exist.

        Only a *task* can be recompiled this way: the request is a spec the
        page's task loader builds again. A scene somebody assembled by hand
        arrived already built, and the script that built it is the only thing
        that can build it differently -- so there the button says that on the
        status line, rather than raising out of the loop's drain.
        """
        from dataclasses import replace

        if getattr(self.session.env, "task", None) is None:
            self.session.status = (
                "nothing to rebuild from -- this scene was assembled by hand, so a rebuild-tier edit "
                "belongs in the script that built it"
            )
            return
        task = self.tuner.task_with_edits()
        if self.physics.spec != task.physics:
            task = replace(task, physics=self.physics.spec)
        # The spec already carries the pairing it was compiled with, so the axes
        # are left empty rather than restated. The world count is not an axis and
        # is restated: the page is showing some number of worlds and a rebuild
        # that quietly dropped to one would be a different page.
        self.session.load_request = (task, self.session.episode_id, "", "", "", self.session.num_worlds)

    def _save(self) -> None:
        name = str(self._save_name.value).strip() or "tuned"
        origin = self.tuner.preset_name()
        # A task names itself; a scene assembled by hand has only the name its
        # request was given, and a preset tuned on one is still worth keeping.
        env = self.session.env
        task = getattr(env, "task", None)
        where = task.full_name if task is not None else getattr(env.request, "full_name", "") or "a hand-built scene"
        record = MaterialRecord(
            name=name,
            spec=self.tuner.material,
            notes=f"Tuned in the studio on {where}" + (f", from {origin!r}." if origin else "."),
            calibration=None,
        )
        self._saved.value = str(write_material(record, Path(self.material_dir) / f"{name}.json"))

    # ---------------------------------------------------------------- refresh
    def refresh(self) -> None:
        if self._pending is None:
            return
        pending = list(self.tuner.pending())
        pending += list(self.physics.pending_rebuild())
        self._pending.value = ", ".join(pending) if pending else "none"
        if self._calibration is None:
            return
        calibration = self.tuner.calibration()
        if calibration is None or not calibration.bending_length_cm:
            self._calibration.value = "no recorded measurement for this material"
        else:
            band = calibration.target_cm
            self._calibration.value = f"{calibration.bending_length_cm} cm at cell {calibration.cell * 1000:.1f} mm" + (
                f" (real fabric {band[0]}-{band[1]} cm)" if band else ""
            )

    def on_reset(self, obs: Any, info: Any) -> None:
        del obs, info

    def close(self) -> None: ...


class PanelReadings(Panel):
    """Folder ``Readings``: what the cloth is doing, in physical units.

    Complements the episode score rather than repeating it. An evaluator says
    whether the fold passed; this says how limp the fabric is and whether it has
    stopped moving -- which is what you change a material *for*.
    """

    def __init__(self, session: Any) -> None:
        self.session = session
        self.measure_requested = False
        self.resettle_requested = False
        self._rows: dict[str, Any] = {}
        self._note: Any = None
        self._entity: Any = None
        self._gui: Any = None
        self._folder: Any = None

    def _cloth(self) -> Any:
        deformables = {n: e for n, e in self.session.env.scene.by_name.items() if e.carries_particles}
        return next(iter(deformables.values()), None) if deformables else None

    def _read(self) -> list[Any]:
        entity = self._cloth()
        env = self.session.env
        if entity is None:
            return []
        # Sheet only. `measure_cantilever` stays in `metrics.py`, tested
        # and exported: the cloth drape bench runs the same measurement its own
        # way, and the panel can grow the branch back when a scene asks for it
        # through something typed.
        readings = measure_sheet(entity, env.state)
        # Plus whatever the garment declares it should reach. Those gates are
        # calibrated numbers with a paragraph of provenance each, sitting in the
        # asset, invisible on the page where somebody is dragging stiffness
        # sliders trying to hit them.
        return readings + declared_states(self._record(), self._snapshot())

    def _record(self) -> Any:
        """The selected cloth's declaration, or ``None`` for a scratch sheet.

        ``None`` is the common answer here, not the exceptional one: this page
        exists to tune a fabric, and most of what it is pointed at is a test
        strip nothing declares.
        """
        env = self.session.env
        doc = getattr(env, "doc", None)
        entity = self._cloth()
        if doc is None or entity is None:
            return None
        from robosimstudio.assetlib.registry import ASSETS

        types = {o.name: o.type for o in doc.objects}
        try:
            return ASSETS.get(types[entity.spec.name])
        except (AttributeError, KeyError):
            return None

    def _snapshot(self) -> Any:
        """The step's own gather, or ``None`` before anything has been scored."""
        context = getattr(self.session.env, "last_context", None)
        if context is None:
            return None
        from robosimstudio.assetlib.semantics.snapshot import Snapshot

        return Snapshot(ctx=context).freeze()

    def build(self, gui: Any) -> None:
        if self._cloth() is None:
            return
        self._gui = gui
        with gui.add_folder("Readings", expand_by_default=False) as self._folder:
            measure = gui.add_button("Measure (settle first)")
            measure.on_click(lambda _: setattr(self, "measure_requested", True))
            resettle = gui.add_button("Re-settle & re-baseline")
            resettle.on_click(lambda _: setattr(self, "resettle_requested", True))
            self._note = gui.add_text("State", initial_value="live", disabled=True)
            for reading in self._read():
                self._rows[reading.name] = gui.add_text(reading.name, initial_value=reading.text(), disabled=True)

    def _busy(self) -> str:
        """Why the pipeline cannot be stepped from here just now, or ``""``.

        Measure and re-settle step the pipeline directly, around ``env.step``:
        no step count, no evaluator, nothing filmed, and particle velocities
        zeroed under whatever is holding the cloth.
        """
        session = self.session
        driver = getattr(session, "driver", None)
        if driver is not None and driver.running:
            return "a policy is driving; stop it first"
        record = getattr(session, "record", None)
        if record is not None and record.recording:
            return "a take is recording; stop it first"
        return ""

    def drain(self) -> None:
        if self.measure_requested or self.resettle_requested:
            busy = self._busy()
            if busy:
                self.measure_requested = self.resettle_requested = False
                self._note.value = busy
                return
        if self.measure_requested:
            self.measure_requested = False
            settled, steps = relax(self.session.env.pipeline)
            self._note.value = f"settled in {steps} steps" if settled else f"still moving after {steps} steps"
            self.refresh()
        if self.resettle_requested:
            self.resettle_requested = False
            entity = self._cloth()
            if entity is not None:
                pipeline = self.session.env.pipeline
                entity.settle(pipeline)
                # The settle stepped the whole particle array, so every thing that
                # shares it takes its baseline again, not only this cloth.
                for thing in getattr(pipeline, "particle_settles", {}).get("things", [entity]):
                    thing.snapshot_rest(pipeline)
                self._note.value = "re-settled; baseline moved with it"
            self.refresh()

    def refresh(self) -> None:
        for reading in self._read():
            row = self._rows.get(reading.name)
            if row is not None:
                row.value = reading.text()
            elif self._folder is not None:
                # A garment's declared states only read once a step has been
                # scored, which is after ``build`` on every load.
                with self._folder:
                    self._rows[reading.name] = self._gui.add_text(
                        reading.name, initial_value=reading.text(), disabled=True
                    )

    def on_reset(self, obs: Any, info: Any) -> None:
        del obs, info
        if self._note is not None:
            self._note.value = "live"

    def close(self) -> None: ...
