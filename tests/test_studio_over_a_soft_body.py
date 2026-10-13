# SPDX-License-Identifier: Apache-2.0
"""The tuning page over a scene whose deformable is not a fabric.

A deformable is not a cloth. ``cloth_fold`` folds a towel, ``surface_prep``
wipes with a block of foam on a foreign engine, and both arrive in the scene's
``deformables`` bin because both are particles -- but only one of them has a material
this page can write. Bulk matter is a Young's modulus and a Poisson ratio at
rebuild tier; the ``Material`` folder maps a fabric onto ``tri_materials`` and
friends, and there is nothing for it to bind.

``MaterialTuner`` knew that and dropped the sponge. What it had no way to say was
that the page it feeds must then be *half* built -- and the half that survives is
the one that matters most on such a scene, because a foreign engine's backend
parameters are reached through ``Physics``.
"""

from __future__ import annotations

import unittest

from robosimstudio.domains.cloth import ClothMaterialSpec

from robosimstudio.specs import ArrangementSpec, backend
from tests.fixtures.viser_stubs import _StubGui

TIER = "fast"
"""No GPU and no scene: stub deformables against the viser stubs."""


class TestADeformableThatIsNotCloth(unittest.TestCase):
    """A SuperDex sponge is a deformable and it is not a fabric.

    ``MaterialTuner`` drops it on purpose -- bulk matter has a Young's modulus,
    not a ``tri_ke``, and nothing on this panel writes one. What it could not do
    was say so: it reports an empty scene by leaving the selection empty, and
    ``PanelTune`` bound ``tuner.material`` outright, which is ``working[""]``. So
    **every** task whose only deformable is a soft body died on ``KeyError: ''``
    inside ``build_panels``, before a single folder was drawn -- a sponge task
    could not be opened in the studio at all, and swapping to it from the task
    menu took the session loop down with it.

    The two halves are gated separately now, which is the split the panel always
    implied. A scene has an arrangement whatever is in it, and a scene whose only
    deformable is a foreign engine's is exactly the one whose backend parameters
    are worth reaching -- so ``Physics`` is the folder that must *not* be gated
    on there being a fabric.
    """

    def _panel(self, material):
        from types import SimpleNamespace

        from robosimstudio_studio.tuning.panels import PanelTune

        entity = SimpleNamespace(
            spec=SimpleNamespace(name="thing", material=material),
            # What `locate_cloth` reads off a built cloth. A model with neither
            # index array is a membrane with no core, which is enough to bind.
            range=SimpleNamespace(particles=(0, 4), tris=(0, 2)),
            carries_particles=True,
        )
        env = SimpleNamespace(
            scene=SimpleNamespace(by_name={"thing": entity}, program=None),
            model=SimpleNamespace(),
            request=SimpleNamespace(
                physics=ArrangementSpec(control_hz=60.0, substeps=10, backends=(("*", backend("mujoco")),))
            ),
        )
        panel, gui = PanelTune(SimpleNamespace(env=env)), _StubGui()
        panel.build(gui)
        return panel, gui

    def _sponge(self):
        from robosimstudio.domains.soft import SoftMaterialSpec

        return self._panel(SoftMaterialSpec())

    def _folders(self, gui) -> list[str]:
        return [f.label for f in gui.folders if f.label is not None]

    def _labels(self, gui) -> list[str]:
        return [label for label, _ in gui.widgets]

    def test_a_page_over_a_sponge_builds(self) -> None:
        """The construction that raised. Nothing is bound because there is
        nothing this panel knows how to tune."""
        panel, _ = self._sponge()
        self.assertEqual(panel.tuner.names, ())
        self.assertFalse(panel.tuner)
        self.assertIsNone(panel.binder, "a binder over a material the panel cannot write")

    def test_it_offers_the_arrangement_and_no_material_folder(self) -> None:
        _, gui = self._sponge()
        self.assertEqual(self._folders(gui), ["Physics"])

    def test_a_backend_parameter_still_lands(self) -> None:
        """The other half of the same gate. ``drain`` returned early on an empty
        tuner, so every control in ``Physics`` was inert on exactly the pages
        where it is the only thing on offer."""
        panel, gui = self._sponge()
        widget = next(w for label, w in gui.widgets if label.startswith("substeps"))
        widget.fire(12)
        panel.drain()
        self.assertEqual(panel.physics.spec.substeps, 12)
        self.assertIn("rates:substeps", panel._pending.value, "the readout never heard about the edit")

    def test_the_rebuild_controls_come_with_the_folder_that_needs_them(self) -> None:
        """They lived in ``Material``, and a rebuild-tier backend parameter has
        no other way to be applied."""
        _, gui = self._sponge()
        self.assertEqual(self._labels(gui).count("Pending rebuild"), 1)
        self.assertIn("Apply & rebuild", self._labels(gui))

    def test_a_fabric_still_gets_both_folders_and_one_of_each_control(self) -> None:
        """The gate is on the material, not on the panel: a cloth page is
        unchanged, and the rebuild controls are not built twice."""
        panel, gui = self._panel(ClothMaterialSpec())
        self.assertEqual(panel.tuner.names, ("thing",))
        self.assertEqual(self._folders(gui), ["Material", "Physics"])
        self.assertEqual(self._labels(gui).count("Pending rebuild"), 1)
        self.assertEqual(self._labels(gui).count("Apply & rebuild"), 1)
        self.assertEqual(self._labels(gui).count("Calibration"), 1)

    def test_a_deformable_whose_spec_declares_no_material_is_dropped_not_raised(self) -> None:
        """The third kind, which is the one the filter was written for.

        A sponge has the wrong material; this has none. ``hasattr`` guards only
        its last attribute access, so reaching ``e.spec.material`` to test it
        raised straight out of the expression that exists to tolerate exactly
        this. Both shipped deformables declare a material, so nothing found it.
        """
        panel, gui = self._panel_without_material()
        self.assertEqual(panel.tuner.names, (), "a spec with no material is not on this panel")
        self.assertIn("Physics", self._folders(gui), "and the arrangement is still reachable")

    def _panel_without_material(self):
        from types import SimpleNamespace

        from robosimstudio_studio.tuning.panels import PanelTune

        entity = SimpleNamespace(
            spec=SimpleNamespace(name="thing"),  # no `material` at all
            range=SimpleNamespace(particles=(0, 4), tris=(0, 2)),
            carries_particles=True,
        )
        env = SimpleNamespace(
            scene=SimpleNamespace(by_name={"thing": entity}, program=None),
            model=SimpleNamespace(),
            request=SimpleNamespace(
                physics=ArrangementSpec(control_hz=60.0, substeps=10, backends=(("*", backend("mujoco")),))
            ),
        )
        panel, gui = PanelTune(SimpleNamespace(env=env)), _StubGui()
        panel.build(gui)
        return panel, gui


class TestTheTuningPageOverASceneWithNoTask(unittest.TestCase):
    """Two buttons that reached for ``env.task``, on a page that has none.

    A scene somebody assembled by hand gets the ``Material`` and ``Physics``
    folders like any other -- the whole page is the default for
    ``StudioSession(sim)`` -- and both act from inside the loop's ``drain``. So
    an ``AttributeError`` there is not a message in a text box, it is the
    session loop ending under a browser that is still connected.
    """

    def _page(self):
        from types import SimpleNamespace

        from robosimstudio_studio.tuning.panels import PanelTune

        entity = SimpleNamespace(
            spec=SimpleNamespace(name="thing", material=ClothMaterialSpec()),
            range=SimpleNamespace(particles=(0, 4), tris=(0, 2)),
            carries_particles=True,
        )
        env = SimpleNamespace(  # a ``Sim``: a scene, a model, a request -- and no ``task``
            scene=SimpleNamespace(by_name={"thing": entity}, program=None),
            model=SimpleNamespace(),
            request=SimpleNamespace(
                full_name="a_towel_on_a_bench",
                physics=ArrangementSpec(control_hz=60.0, substeps=10, backends=(("*", backend("mujoco")),)),
            ),
        )
        session = SimpleNamespace(env=env, status="", load_request=None, episode_id=0, num_worlds=1)
        panel, gui = PanelTune(session), _StubGui()
        panel.build(gui)
        return panel, session

    def test_apply_and_rebuild_says_there_is_nothing_to_rebuild_from(self) -> None:
        panel, session = self._page()
        panel.rebuild_requested = True
        panel.drain()  # raised ``'Sim' object has no attribute 'task'``
        self.assertIsNone(session.load_request, "a rebuild was requested of a scene nothing can recompile")
        self.assertIn("assembled by hand", session.status)

    def test_a_preset_saved_there_names_the_scene_it_was_tuned_on(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        panel, _ = self._page()
        with tempfile.TemporaryDirectory() as tmp:
            panel.material_dir = tmp
            panel._save_name.value = "towel_tuned"
            panel.save_requested = True
            panel.drain()  # raised the same ``AttributeError``
            written = Path(tmp) / "towel_tuned.json"
            self.assertTrue(written.exists(), panel._saved.value)
            self.assertIn("a_towel_on_a_bench", json.dumps(json.loads(written.read_text())))
