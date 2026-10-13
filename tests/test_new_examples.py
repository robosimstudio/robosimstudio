# SPDX-License-Identifier: Apache-2.0
"""The newest numbered examples run whole, and the two "your own" ones build what they say.

``tests/test_examples.py`` names the numbered scripts it covers one by one, so
the eight here are not seen by it. Each is loaded by file path, fresh, under the
module name a reader's own load would give it, with the studio replaced by a
stub that records what it was handed; ``main()`` then runs to its last line.

The assertions are about the scene rather than the page: that it was built,
that it steps and stays finite, and -- for a file that registers a second
implementation of a kind -- that the thing in the scene was built by *that*
file's class and not by the shipped one. An example that needs an engine this
checkout does not carry is skipped rather than failed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path

import numpy as np

TIER = "medium"
"""Builds and steps real scenes on a GPU, and renders two stills."""

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

SCENES = (
    "00_getting_started/00_hello_world.py",
    "00_getting_started/01_core_concepts.py",
    "02_cloth/01_fabrics.py",
    "02_cloth/04_your_own_cloth.py",
    "03_soft/02_your_own_soft.py",
    "04_extending/00_add_a_robot.py",
    "05_studio/00_your_own_panel.py",
)
"""The seven that end by opening the studio on a scene."""

HEADLESS = "05_studio/01_render.py"
"""The one that opens nothing and writes files instead."""


def _load(rel: str):
    """One example as a module, loaded by file path, fresh every time.

    The name is a function of the path, so a second load of the same file is a
    second module under the *same* name -- which is what a reader's ``python
    examples/...`` does too, and what the two registering examples promise to
    survive. It is the name :func:`tests.fixtures.examples._load` gives the file
    as well: a process that runs both (the grasp tests load the own-cloth example
    there) then sees one class shipped twice, which the registry accepts, rather
    than two classes under one entity name, which it refuses.
    """
    name = "_rss_example_" + re.sub(r"\W", "_", rel.removesuffix(".py"))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _stub_studio():
    """``rss.view`` replaced for the duration; what it was handed is yielded."""
    import robosimstudio as rss

    opened: list[tuple] = []

    def view(sim, **kwargs):
        opened.append((sim, kwargs))

    real = rss.view
    rss.view = view
    try:
        yield opened
    finally:
        rss.view = real


def _run(case: unittest.TestCase, rel: str):
    """Load and run one example; ``(module, what it opened the studio on, what it printed)``."""
    module = _load(rel)
    text = io.StringIO()
    try:
        with _stub_studio() as opened, contextlib.redirect_stdout(text):
            module.main()
    except ImportError as exc:
        if "extra" in str(exc):
            case.skipTest(f"examples/{rel} needs something this checkout does not carry: {exc}")
        raise
    return module, opened, text.getvalue()


def _one_scene(case: unittest.TestCase, rel: str):
    module, opened, text = _run(case, rel)
    case.assertEqual(1, len(opened), f"examples/{rel} opened {len(opened)} pages, not one")
    sim, kwargs = opened[0]
    case.assertTrue(hasattr(sim, "pipeline"), f"examples/{rel} handed the studio something else")
    return module, sim, kwargs, text


class TestTheNewExamplesRun(unittest.TestCase):
    def test_each_scene_example_builds_steps_and_stays_finite(self) -> None:
        for rel in SCENES:
            with self.subTest(example=rel):
                _, sim, _, _ = _one_scene(self, rel)
                sim.run(20).sync()
                self.assertGreaterEqual(sim.step_count, 20, f"examples/{rel} did not step")
                self.assertTrue(sim.state_is_finite(), f"examples/{rel} diverged")
                sim.close()

    def test_the_first_three_print_what_they_build(self) -> None:
        _, sim, _, text = _one_scene(self, "00_getting_started/00_hello_world.py")
        self.assertIn("the cube came to rest at z=0.77", text)
        self.assertEqual({"cube", "table"}, set(sim.names()) & {"cube", "table"})
        sim.close()

        _, sim, _, text = _one_scene(self, "00_getting_started/01_core_concepts.py")
        self.assertIn("void: the work surface is at z=0.000 m", text)
        self.assertIn("sim['cube'] is cube: True", text)
        sim.close()

        _, sim, _, _ = _one_scene(self, "02_cloth/01_fabrics.py")
        self.assertEqual(
            {"jersey", "tape", "fleece", "towelling"},
            {n for n in sim.names() if n in ("jersey", "tape", "fleece", "towelling")},
        )
        self.assertEqual(
            6 * 25 * 25, sim.model.particle_count, "two membranes, and two slabs with a particle on each face"
        )
        sim.close()

    def test_the_render_example_writes_a_still_and_a_video(self) -> None:
        _, opened, text = _run(self, HEADLESS)
        self.assertEqual([], opened, "a headless example opened a page")
        stills = [Path(line.split(" written to ", 1)[1]) for line in text.splitlines() if " written to " in line]
        self.assertEqual(["still"], [path.stem for path in stills])
        import imageio.v3 as iio

        still = stills[0]
        self.assertTrue(still.is_file(), f"{still} was announced and not written")
        self.assertEqual(b"\x89PNG", still.read_bytes()[:4])
        rgb = iio.imread(still)
        self.assertEqual((540, 960, 3), rgb.shape)
        self.assertGreater(float(rgb.std()), 10.0, "a blank frame")
        videos = [Path(line.split("video: ", 1)[1]) for line in text.splitlines() if line.startswith("video: ")]
        self.assertEqual(1, len(videos), text)
        self.assertTrue(videos[0].is_file(), f"{videos[0]} was announced and not written")
        self.assertGreater(videos[0].stat().st_size, 1000, "an empty video")


class TestTheOwnImplementations(unittest.TestCase):
    """The two files that register a second implementation of a shipped kind."""

    def test_the_sheet_is_built_by_the_files_own_class(self) -> None:
        from robosimstudio.domains.cloth import EntityCloth

        import robosimstudio as rss

        module, sim, _, text = _one_scene(self, "02_cloth/04_your_own_cloth.py")
        sheet = sim["sheet"]
        # The file's class by name: the registry keeps the first copy of an identical
        # class, so after a second load the entity's class is the first module's twin.
        built = type(sheet.entity)
        self.assertEqual((module.__name__, "MyCloth"), (built.__module__, built.__qualname__))
        self.assertNotIsInstance(sheet.entity, EntityCloth, "the shipped class built it")
        self.assertIsInstance(sheet.spec, rss.ClothSpec, "the stock spec, with entity= pointing at the file's class")
        self.assertEqual(441, len(sheet.particles))
        self.assertEqual(800, sim.model.tri_count, "two triangles per cell of a 20 x 20 grid")
        self.assertIn("MyCloth built 441 particles", text)
        surface = rss.get_cell("bench_075").surface_z
        self.assertGreater(float(sheet.particle_q[:, 2].min()), surface, "the sheet went through the bench")
        self.assertLess(float(sheet.particle_q[:, 2].min()), surface + 0.02, "the hem never reached the bench")
        sim.close()

    def test_the_block_is_built_by_the_files_own_class(self) -> None:
        from robosimstudio.domains.soft import EntitySoft

        import robosimstudio as rss

        module, sim, _, text = _one_scene(self, "03_soft/02_your_own_soft.py")
        block = sim["block"]
        built = type(block.entity)
        self.assertEqual((module.__name__, "MySoft"), (built.__module__, built.__qualname__))
        self.assertNotIsInstance(block.entity, EntitySoft, "the shipped class built it")
        self.assertIsInstance(block.spec, rss.SoftSpec)
        self.assertEqual(440, len(block.particles))
        self.assertEqual(1680, sim.model.tet_count, "six tetrahedra per cell of a 10 x 7 x 4 lattice")
        self.assertEqual("vbd", sim.request.physics.backend_for("soft").solver, "on Newton, with no extra")
        self.assertIn("1680 tetrahedra", text)
        column = block.particle_q[:, 2]
        surface = rss.get_cell("bench_075").surface_z
        self.assertGreater(float(column.min()), surface, "the block went through the bench")
        self.assertAlmostEqual(
            0.040, float(column.max() - column.min()), delta=0.003, msg="the block did not keep its height"
        )
        sim.close()

    def test_loading_either_file_twice_does_not_fail(self) -> None:
        from robosimstudio.scene.registry import get_entity

        for rel, key, cls in (
            ("02_cloth/04_your_own_cloth.py", "my_cloth", "MyCloth"),
            ("03_soft/02_your_own_soft.py", "my_soft", "MySoft"),
        ):
            with self.subTest(example=rel):
                first = _load(rel)
                second = _load(rel)
                self.assertIsNot(first, second, "the loader cached the module")
                registered = get_entity(key)
                self.assertEqual((second.__name__, cls), (registered.__module__, registered.__qualname__))


class TestTheRobotExample(unittest.TestCase):
    def test_the_inline_robot_is_unregistered_indexed_and_solves(self) -> None:
        from robosimstudio.robots.registry import EMBODIMENTS

        _, sim, _, text = _one_scene(self, "04_extending/00_add_a_robot.py")
        self.assertNotIn("my_panda", EMBODIMENTS)
        self.assertEqual("my_panda", sim.scene.embodiment.spec.name)
        self.assertIn("my_panda stands with its tool centre at", text)
        self.assertEqual(["right"], sorted(sim.handles.arms))
        self.assertEqual(7, len(sim.handles.arms["right"].arm_coords))
        seed = np.asarray(sim.pipeline.snapshot.joint_q, dtype=np.float64)
        target = np.array([0.60, 0.0, 0.95])
        solved = sim.arm().solve_q(seed, target, (0.0, 0.0, -1.0))
        landed, _ = sim.arm().fk_ee(np.asarray(solved, dtype=np.float32))
        self.assertLess(float(np.linalg.norm(landed - target)), 0.02)
        sim.close()


class TestThePanelExample(unittest.TestCase):
    def test_the_button_tosses_the_cube_and_the_readout_follows(self) -> None:
        from tests.fixtures.viser_stubs import _StubGui, _StubScene

        _, sim, kwargs, _ = _one_scene(self, "05_studio/00_your_own_panel.py")
        page = kwargs.get("page")
        self.assertIsNotNone(page, "the example stopped putting its folder on the page")

        class _Studio:
            def __init__(self) -> None:
                self.gui = _StubGui()
                self.scene = _StubScene()
                self.steps: list = []

            def each_step(self, fn) -> None:
                self.steps.append(fn)

        studio = _Studio()
        page(studio)
        self.assertIn("Cube", [f.label for f in studio.gui.folders])
        sim.run(120).sync()  # the example opens the page with the cube still in the air
        for fn in studio.steps:
            fn()
        readout = studio.gui.find("height above the bench (mm)")
        self.assertAlmostEqual(25.0, readout.value, delta=2.0, msg="a 50 mm cube resting on the bench")

        studio.gui.find("Toss the cube").fire()  # what a click does: the callback only sets a flag
        for fn in studio.steps:
            fn()  # the main-thread hook writes the velocity
        sim.run(6).sync()  # 0.12 s at 50 Hz: 1.5 m/s up puts it 108 mm over where it lay
        for fn in studio.steps:
            fn()
        self.assertGreater(readout.value, 80.0, "the cube was not tossed")
        self.assertTrue(sim.state_is_finite())
        sim.close()


if __name__ == "__main__":
    unittest.main()
