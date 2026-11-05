# SPDX-License-Identifier: Apache-2.0
"""Each RTX viewer opens a stage named for itself, and its renderer on the simulation's GPU.

Newton names every viewer's OVStage ``"newton.ViewerRTX"``. Two of them in one
process were not kept apart: a second renderer on a twin's scene drew three
mugs of one material near black while the first kept their colours, from two
stages that were identical line for line (see ``_stage_of_its_own``). What is
pinned here is the wrap round Newton's ``_init_ovrtx`` that names each stage
apart, and the one beside it that puts the renderer on the sim's GPU, against
stand-in ``ovstage`` and ``ovrtx`` modules: no GPU, no OVRTX.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from robosimstudio.solvers.render import rtx

TIER = "fast"


def _modules():
    """Stand-ins for the two OVRTX modules: a stage that keeps its name, a renderer that keeps its config."""
    ovstage = types.ModuleType("ovstage")

    class Stage:
        def __init__(self, name=None, config=None):
            self.name = name
            self.config = config

    ovstage.Stage = Stage
    ovrtx = types.ModuleType("ovrtx")

    class Renderer:
        def __init__(self, config=None):
            self.config = config

    ovrtx.Renderer = Renderer
    return {"ovstage": ovstage, "ovrtx": ovrtx}


class EachViewerNamesItsOwnStage(unittest.TestCase):
    def test_two_stages_asked_for_by_one_name_are_named_apart(self) -> None:
        fake = _modules()
        with mock.patch.dict(sys.modules, fake):
            import ovstage

            original = ovstage.Stage
            with rtx._stage_of_its_own():
                first = ovstage.Stage("newton.ViewerRTX")
            with rtx._stage_of_its_own():
                second = ovstage.Stage("newton.ViewerRTX", config="kept")
            self.assertIs(ovstage.Stage, original, "the module's own class is back after each window")
        self.assertNotEqual(first.name, second.name)
        for stage in (first, second):
            self.assertTrue(stage.name.startswith("newton.ViewerRTX."), stage.name)
            self.assertIsInstance(stage, original, "a stage made in the window is still an ovstage.Stage")
        self.assertEqual(second.config, "kept", "what else the caller passed goes through")

    def test_a_failure_inside_the_window_puts_the_class_back(self) -> None:
        with mock.patch.dict(sys.modules, _modules()):
            import ovstage

            original = ovstage.Stage
            with self.assertRaises(RuntimeError), rtx._stage_of_its_own():
                raise RuntimeError("Failed to create OVRTX renderer")
            self.assertIs(ovstage.Stage, original)

    def test_without_ovstage_there_is_nothing_to_name(self) -> None:
        with mock.patch.dict(sys.modules, {"ovstage": None}), rtx._stage_of_its_own():
            pass


class TheRendererGoesOnTheSimsGpu(unittest.TestCase):
    def test_an_unset_device_is_filled_and_a_stated_one_is_kept(self) -> None:
        with mock.patch.dict(sys.modules, _modules()):
            import ovrtx

            original = ovrtx.Renderer
            with rtx._renderer_on_gpu(3):
                filled = ovrtx.Renderer(config=types.SimpleNamespace(active_cuda_gpus=None))
                kept = ovrtx.Renderer(config=types.SimpleNamespace(active_cuda_gpus="0"))
            self.assertIs(ovrtx.Renderer, original)
        self.assertEqual(filled.config.active_cuda_gpus, "3")
        self.assertEqual(kept.config.active_cuda_gpus, "0")

    def test_a_cpu_device_leaves_ovrtx_unimported(self) -> None:
        with mock.patch.dict(sys.modules, {"ovrtx": None}), rtx._renderer_on_gpu(None):
            pass


class TheViewerWrapsNewtonsOwnInit(unittest.TestCase):
    def test_newtons_init_meets_both_wraps_and_the_decoration_comes_first(self) -> None:
        cls = rtx._viewer_class()
        base = cls.__mro__[1]
        order: list[str] = []
        made: dict[str, object] = {}

        def newton_init(self) -> None:
            # The part of Newton's own _init_ovrtx the wraps are for.
            import ovrtx
            import ovstage

            order.append("newton")
            made["renderer"] = ovrtx.Renderer(config=types.SimpleNamespace(active_cuda_gpus=None))
            made["stage"] = ovstage.Stage("newton.ViewerRTX")

        viewer = cls.__new__(cls)
        viewer._decorate = lambda v: order.append("decorate")
        viewer.device = types.SimpleNamespace(is_cuda=True, ordinal=2)
        with mock.patch.dict(sys.modules, _modules()), mock.patch.object(base, "_init_ovrtx", newton_init):
            viewer._init_ovrtx()
            again = cls.__new__(cls)
            again._decorate = None
            again.device = types.SimpleNamespace(is_cuda=True, ordinal=2)
            first = made["stage"].name
            again._init_ovrtx()
        self.assertEqual(order, ["decorate", "newton", "newton"])
        self.assertEqual(made["renderer"].config.active_cuda_gpus, "2")
        self.assertNotEqual(first, made["stage"].name, "two viewers in one process, two stage names")

    def test_two_viewers_built_at_once_on_two_threads_leave_the_modules_as_they_were(self) -> None:
        """Each saves the module's class, swaps its own in, and puts back what it saved: unserialised, the second
        thread saved the first one's stand-in and put that back for good."""
        import threading
        import time

        cls = rtx._viewer_class()
        base = cls.__mro__[1]
        names: list[str] = []

        def newton_init(self) -> None:
            import ovstage

            time.sleep(0.05)  # long enough for the other thread to reach its own swap
            names.append(ovstage.Stage("newton.ViewerRTX").name)

        def build() -> None:
            viewer = cls.__new__(cls)
            viewer._decorate = None
            viewer.device = types.SimpleNamespace(is_cuda=True, ordinal=0)
            viewer._init_ovrtx()

        fake = _modules()
        with mock.patch.dict(sys.modules, fake), mock.patch.object(base, "_init_ovrtx", newton_init):
            original_stage, original_renderer = fake["ovstage"].Stage, fake["ovrtx"].Renderer
            threads = [threading.Thread(target=build) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertIs(fake["ovstage"].Stage, original_stage, "a stand-in Stage was left installed")
            self.assertIs(fake["ovrtx"].Renderer, original_renderer, "a stand-in Renderer was left installed")
        self.assertEqual(2, len(set(names)))
        for name in names:
            self.assertEqual(2, name.count("."), f"a suffix on a suffix: {name}")


if __name__ == "__main__":
    unittest.main()
