# SPDX-License-Identifier: Apache-2.0
"""Small contracts in the core tier, each one pinned where it once broke.

The bus withdraws a request its worker gave up on; a settle does not eat a
pending re-capture; every door that steps a scene runs its post-step hooks;
``Scene.add`` binds a ref for whatever it accepts and refuses a second of
anything; ``rss.view`` keeps the fields of a config it is handed. None of them
needs a scene, so all of them run on the CPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import threading
import time
import types
import unittest
from unittest import mock

TIER = "fast"
"""CPU only: fakes and a one-body model, no scene build."""


class TestABusRequestThatTimedOutNeverRuns(unittest.TestCase):
    def test_a_late_pump_skips_it(self) -> None:
        from robosimstudio.core.bus import BusTimeout, SimBus

        bus = SimBus(timeout=0.05)
        bus.owner_thread = -1  # never this thread, so the call really queues
        ran: list[int] = []
        with self.assertRaises(BusTimeout):
            bus.call(lambda: ran.append(1))
        self.assertEqual(bus.pump(), 0, "a withdrawn request is not serviced")
        self.assertEqual(ran, [], "the caller was told it timed out, so it must not run behind its back")

    def test_an_interrupt_on_the_owner_propagates_and_releases_the_worker(self) -> None:
        from robosimstudio.core.bus import BusClosed, SimBus

        bus = SimBus(timeout=5.0)
        bus.owner_thread = -1
        caught: list[BaseException] = []

        def interrupt() -> None:
            raise KeyboardInterrupt

        def worker() -> None:
            try:
                bus.call(interrupt)
            except BaseException as exc:
                caught.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        while bus._queue.empty():
            time.sleep(0.001)
        with self.assertRaises(KeyboardInterrupt):
            bus.pump()
        thread.join(2.0)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], BusClosed, "the worker must not be handed the owner's interrupt")
        self.assertTrue(bus.closed)


class TestASettleKeepsAPendingRecapture(unittest.TestCase):
    def _executor(self):
        from robosimstudio.core.schedule import Schedule, ScheduleExecutor
        from tests.test_schedule import _model

        return ScheduleExecutor(_model(), Schedule(), control_hz=50.0, substeps=2, use_cuda_graph=False)

    def test_a_recapture_asked_for_before_the_settle_waits_for_it(self) -> None:
        """Captured at the settle's substep count and then discarded with the
        block, the re-capture used to be spent, and the run went eager for good."""
        executor = self._executor()
        executor.invalidate_graph()
        with executor.substeps_at(8):
            executor.step()
        self.assertTrue(executor._recapture, "the re-capture must still be pending at the run's own count")
        self.assertEqual(executor.substeps, 2)

    def test_stages_changed_inside_the_settle_drop_the_graph_set_aside(self) -> None:
        executor = self._executor()
        executor._graph = object()  # stands in for a captured graph
        with executor.substeps_at(8):
            executor.invalidate_graph()
        self.assertIsNone(executor._graph, "the graph set aside no longer matches the stages")
        self.assertTrue(executor._recapture)

    def test_an_after_stage_added_later_still_makes_the_step_wait(self) -> None:
        from robosimstudio.core.schedule import Stage

        executor = self._executor()
        executor.schedule.add(Stage(name="tick", fn=lambda s, dt: None, phase="after", reads=frozenset({"body_q"})))
        with mock.patch("robosimstudio.core.schedule.wp.synchronize_device") as sync:
            executor.step()
        sync.assert_called_once()


class TestEveryDoorRunsThePostStepHooks(unittest.TestCase):
    def _sim(self):
        from robosimstudio.sim.sim import Sim

        sim = Sim.__new__(Sim)
        sim.pipeline = types.SimpleNamespace(step=lambda: None)
        sim.step_count = 0
        sim.post_step_hooks = []
        sim._warned_nonfinite = True  # nothing to read the finite flags from
        return sim

    def test_run(self) -> None:
        sim = self._sim()
        seen: list[int] = []
        sim.post_step_hooks.append(lambda s: seen.append(s.step_count))
        sim.run(3)
        self.assertEqual(seen, [1, 2, 3])

    def test_draw(self) -> None:
        draw = importlib.import_module("robosimstudio.sim.draw")  # the package exports the function by that name

        sim = self._sim()
        seen: list[int] = []
        sim.post_step_hooks.append(lambda s: seen.append(s.step_count))
        page = mock.MagicMock()
        page.__enter__.return_value = page
        page.is_running.return_value = True
        with mock.patch.object(draw, "viewer", return_value=page):
            draw.draw(sim, steps=2)
        self.assertEqual(seen, [1, 2])
        self.assertEqual(sim.step_count, 2)


def _region(name: str):
    from robosimstudio.specs import RegionSpec

    return RegionSpec(name=name, frame="", xy_ranges=(-0.1, -0.1, 0.1, 0.1))


class TestSceneAddBindsAndRefuses(unittest.TestCase):
    def _scene(self):
        from robosimstudio.sim.scene import Scene

        return Scene(cell="bench_075")

    def test_robot_region_and_camera_refs_are_bound_by_build(self) -> None:
        from robosimstudio.sim.scene import robot
        from robosimstudio.specs import CameraSpec

        scene = self._scene()
        refs = [
            scene.add(robot("panda")),
            scene.add(_region("bin")),
            scene.add(CameraSpec(name="top")),
        ]
        # Sim binds exactly the refs the scene holds, so a ref left out stays on None.
        self.assertEqual([scene._refs[r.name] for r in refs], refs)
        compiled = types.SimpleNamespace(embodiment="the robot")
        refs[0]._scene = compiled
        self.assertEqual(refs[0].entity, "the robot")
        self.assertEqual(refs[2].spec.name, "top")
        with self.assertRaisesRegex(ValueError, "camera"):
            _ = refs[2].pose

    def test_a_second_robot_is_refused(self) -> None:
        from robosimstudio.sim.scene import robot

        scene = self._scene()
        scene.add(robot("panda"))
        with self.assertRaisesRegex(ValueError, "already has a robot"):
            scene.add(robot("ur5e"))

    def test_duplicate_names_are_refused_for_every_kind(self) -> None:
        from robosimstudio.domains.rigid import box
        from robosimstudio.specs import CameraSpec

        scene = self._scene()
        scene.add(CameraSpec(name="top"))
        scene.add(_region("bin"))
        for spec in (CameraSpec(name="top"), _region("bin"), box(name="top")):
            with self.assertRaises(ValueError):
                scene.add(spec)

    def test_remove_takes_out_whatever_add_put_in(self) -> None:
        from robosimstudio.sim.scene import robot
        from robosimstudio.specs import CameraSpec

        scene = self._scene()
        scene.add(robot("panda"))
        scene.add(_region("bin"))
        scene.add(CameraSpec(name="top"))
        for name in ("panda", "bin", "top"):
            scene.remove(name)
        self.assertEqual((scene._robot, scene.regions, scene.cameras, scene._refs), (None, [], [], {}))
        scene.add(robot("ur5e"))  # the slot is free again


class TestSmallerContracts(unittest.TestCase):
    def test_substeps_must_be_a_positive_whole_number(self) -> None:
        from robosimstudio.specs import ArrangementSpec

        for bad in (0, -2, 3.5):
            with self.assertRaises(ValueError):
                ArrangementSpec(substeps=bad)
        with self.assertRaises(ValueError):
            ArrangementSpec(control_hz=0.0)

    def test_set_body_attribute_counts_a_generator(self) -> None:
        from robosimstudio.core.builder import set_body_attribute

        attr = types.SimpleNamespace(values={})
        builder = types.SimpleNamespace(custom_attributes={"k": attr})
        self.assertEqual(set_body_attribute(builder, "k", 1.0, (b for b in (3, 4))), 2)
        self.assertEqual(attr.values, {3: 1.0, 4: 1.0})

    def test_arm_name_on_a_scene_without_a_task(self) -> None:
        from robosimstudio import arms

        sim = types.SimpleNamespace(request=types.SimpleNamespace(full_name="bare"))
        with mock.patch.object(arms, "arm_order", return_value=()):
            with self.assertRaisesRegex(RuntimeError, "'bare' has no arms"):
                arms.arm_name(sim, 0)

    def test_build_order_is_newtons_palette(self) -> None:
        """The copy in ``palette`` is only safe while it still equals the original."""
        import newton

        from robosimstudio.core.render.palette import BUILD_ORDER

        self.assertEqual(tuple(map(tuple, newton.ModelBuilder._SHAPE_COLOR_PALETTE)), BUILD_ORDER)


@dataclasses.dataclass
class _Config:
    panels: object = "all"
    render_every: int = 4
    port: int = 8080
    share: bool = False
    handles: int = 1
    speed: str = "1x"
    dark_mode: bool = False
    cloth_look: str = "fluffy"
    lighting: object = None
    show_scenery: bool = False


class TestViewKeepsTheConfigItIsGiven(unittest.TestCase):
    def _view(self, **kwargs) -> _Config:
        from robosimstudio.sim.draw import view

        made: list[_Config] = []

        class Session:
            def __init__(self, sim, *, config) -> None:
                made.append(config)

            def run(self) -> None:
                pass

        fake = types.SimpleNamespace(StudioConfig=_Config, StudioSession=Session)
        with mock.patch.dict(sys.modules, {"robosimstudio_studio.session": fake}):
            view(object(), **kwargs)
        return made[0]

    def test_no_config_gets_this_doors_defaults_and_no_tunnel(self) -> None:
        config = self._view()
        self.assertEqual((config.panels, config.render_every, config.share), ("drive", 1, False))

    def test_a_given_config_keeps_its_fields(self) -> None:
        config = self._view(config=_Config(panels="all", render_every=4, share=True))
        self.assertEqual((config.panels, config.render_every, config.share), ("all", 4, True))

    def test_a_keyword_still_lands_on_top_of_it(self) -> None:
        config = self._view(config=_Config(), panels="drive", share=True)
        self.assertEqual((config.panels, config.render_every, config.share), ("drive", 4, True))


if __name__ == "__main__":
    unittest.main()
