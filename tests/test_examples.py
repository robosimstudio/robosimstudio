# SPDX-License-Identifier: Apache-2.0
"""The examples run -- all of ``main()``, against a studio that is not there.

An example that no longer runs is worse than no example: it is the first thing
a reader tries, and nothing else in the suite touches ``examples/``.

Each scene script assembles something and hands it to
:func:`robosimstudio.view`, so there is no loop of its own to drive and no
headless switch to flip. ``view`` is replaced instead, and what it was handed is
the thing worth asserting on: an example's claim is *what it put in the scene*,
and the interaction over that scene is the studio's and is tested with the
studio (``tests/test_studio_panels.py``, ``tests/test_robot_drag.py``).

So the assertions here are about the scene rather than about the page. A towel
that stopped being built would still open a page; it would just be a page with
nothing on it, which is exactly the failure that a "did it render" test cannot
see and the one that matters.

**The layout is viser's**: numbered category folders holding numbered scripts
(``examples/02_cloth/00_towel.py``), each of which runs alone. A numbered folder
is not an importable package, so every example here is loaded by file path --
which is also how a reader runs one (``uv run examples/02_cloth/00_towel.py``).
The tables of scripts and the loader are :mod:`tests.fixtures.examples`.

**This module is the numbered examples**, the ones every tree ships, the rope's
in ``examples/10_rope/`` among them. Four older flat rope scripts remain under
``examples/scenes/`` in a tree that has them. Where a rule here is every scene
script's -- it runs, it ships two arms, the page can take hold of it, it parses
no command line -- they are held to it by the same loop, and the loop covers the
numbered scripts alone in a tree that does not have them. What only those older
scripts claim is tested beside them, in a tree that has them.

``scripts/studio.py`` -- the studio's launcher, in a tree that has it -- is not
checked here: it takes a registered task id and about fifty flags, which makes
it a tool rather than an example.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from textwrap import dedent

from tests.fixtures.examples import DEMOS, EXAMPLES, HAS_ROPE, NUMBERED, ROOT, SCRIPTS, _load, _run

TIER = "medium"
"""Builds and steps real scenes on a GPU."""


class TestTheExamplesRun(unittest.TestCase):
    def test_each_one_builds_a_scene_and_opens_the_studio_on_it(self) -> None:
        for name in SCRIPTS:
            with self.subTest(example=name):
                sim, _ = _run(name)
                self.assertTrue(hasattr(sim, "pipeline"), f"examples/{name} handed the studio something else")
                sim.run(20).sync()
                self.assertTrue(sim.state_is_finite(), f"examples/{name} diverged in twenty steps")

    def test_each_one_ships_a_robot_you_can_drive(self) -> None:
        """Two arms, so half of what there is to do with a rope or a towel is there.

        The examples were rebuilt around this: a page with nothing actuated on it
        can be dragged but not *driven*, and the difference is the whole second
        half of the interaction.
        """
        for name in SCRIPTS:
            with self.subTest(example=name):
                sim, _ = _run(name)
                self.assertEqual(["left", "right"], sorted(sim.handles.arms))

    def test_each_one_builds_the_thing_it_is_about(self) -> None:
        """The assertion a "did the page render" test cannot make."""
        sim, _ = _run("01_rigid/00_drag_things.py")
        self.assertEqual({"ball", "cube", "pin"}, {n for n in sim.names() if n in {"ball", "cube", "pin"}})

        sim, _ = _run("02_cloth/00_towel.py")
        self.assertGreater(sim.model.particle_count, 500, "a sheet is a cloud of particles")

        sim, _ = _run("02_cloth/02_engines.py")
        self.assertIn("towel", sim.names())
        self.assertEqual(441, sim.model.particle_count, "a 20 x 20 sheet is 21 x 21 particles")

        sim, _ = _run("03_soft/01_tet_mesh.py")
        self.assertEqual(140, sim.model.tet_count, "28 rings of five tetrahedra, wound closed, read off a file")
        self.assertEqual(112, sim.model.particle_count, "four vertices a ring")

        sim, _ = _run("04_extending/01_new_kind.py")
        self.assertGreater(len(sim["chain"].bodies), 12, "the kind this file registered")

        sim, _ = _run("10_rope/00_cable.py")
        self.assertEqual(28, len(sim["cable"].bodies), "a cable is a chain of capsules")

        sim, _ = _run("10_rope/01_between_posts.py")
        self.assertEqual(24, len(sim["cable"].bodies))
        self.assertLessEqual({"post_start", "post_end"}, set(sim.names()))

    def test_a_cable_clamped_by_both_ends_sags_between_them(self) -> None:
        """``anchors`` is the whole of ``01_between_posts``: the two ends stay where
        they were built, just over the posts, and the span between them hangs."""
        import numpy as np

        sim, _ = _run("10_rope/01_between_posts.py")
        sim.run(120).sync()
        bodies = np.asarray(sim["cable"].bodies).ravel()
        z = np.asarray(sim.pipeline.snapshot.body_q)[bodies, 2]
        import robosimstudio as rss

        built = rss.get_cell("bench_075").surface_z + 0.16 + 0.008 + 0.002  # post, radius, clearance
        self.assertAlmostEqual(built, float(z[0]), delta=0.002, msg="the clamped start moved")
        self.assertAlmostEqual(built, float(z[-1]), delta=0.002, msg="the clamped end moved")
        sag = built - float(z.min())
        self.assertGreater(sag, 0.005, "a span clamped at both ends hangs under its own weight")
        self.assertLess(sag, 0.08, "the clamps did not hold the span up")

    def test_cloth_stages_the_coupling_a_pinch_needs(self) -> None:
        """``GENERIC``'s proxy coupling can push a sheet and not pinch one, so a
        page with two grippers on it would look like broken hardware."""
        sim, _ = _run("02_cloth/00_towel.py")
        physics = sim.request.physics
        self.assertEqual("one_way", physics.coupling.method)
        self.assertTrue(physics.collision.full_surface_contact)

    def test_the_engines_example_changes_one_entry_and_nothing_else(self) -> None:
        """``02_engines`` is about one string: on its default the sheet is on
        Newton's own solver, the arm on MuJoCo, and the rest is ``GENERIC``'s."""
        import robosimstudio as rss

        sim, _ = _run("02_cloth/02_engines.py")
        physics = sim.request.physics
        self.assertEqual("vbd", physics.backend_for("cloth").solver)
        self.assertEqual("mujoco", physics.backend_for("articulated").solver)
        self.assertEqual(rss.GENERIC.coupling, physics.coupling)
        self.assertEqual(
            {"newton", "superdex", "genesis"}, set(_load("02_cloth/02_engines.py").SOLVERS), "the three it offers"
        )

    def test_the_tet_mesh_keeps_the_soft_body_on_newton(self) -> None:
        """So that an example about *geometry* needs no extra: no entry routes
        ``soft`` anywhere, and it falls through the wildcard onto VBD."""
        sim, _ = _run("03_soft/01_tet_mesh.py")
        physics = sim.request.physics
        self.assertNotIn("soft", [kind for kind, _ in physics.backends])
        self.assertEqual("vbd", physics.backend_for("soft").solver)


class TestTheMultiEngineDemos(unittest.TestCase):
    """The scripts that are the whole multi-engine claim, and were in no test.

    ``03_soft/00_sponge``, ``02_cloth/03_genesis`` and ``10_rope/02_superdex`` are
    the smallest thing that proves a foreign engine simulates a deformable beside
    a Newton arm -- one per engine and kind -- and until then nothing ran them.
    They cannot join :data:`SCRIPTS`: the scene scripts are held to a bimanual
    robot and to running on a bare checkout, and these are held to an engine
    instead.
    """

    def test_the_sponge_is_a_soft_body_on_the_foreign_engine(self) -> None:
        """SuperDex's acceptance scene: it builds, it steps, and the foam is particles."""
        import importlib.util

        # The module is ``superdex``; the distribution that ships it is
        # ``superdex-physics``. Guarding on the distribution name skips silently
        # on a checkout that has the engine, which is a test that never runs.
        if importlib.util.find_spec("superdex") is None:
            self.skipTest("[superdex] is not installed; the engine cannot be reached")
        sim, _ = _run("03_soft/00_sponge.py")
        self.assertGreater(sim.model.particle_count, 100, "a sponge is a tetrahedral cloud of particles")
        self.assertEqual(
            ["superdex"], sorted({b.solver for _, b in sim.request.physics.backends if b.solver == "superdex"})
        )
        sim.run(20).sync()
        self.assertTrue(sim.state_is_finite(), "examples/03_soft/00_sponge.py diverged in twenty steps")

    def test_the_genesis_cloth_refuses_by_name_without_its_wheel(self) -> None:
        """Genesis's acceptance scene, on a venv that deliberately lacks the engine.

        The refusal *is* the tested behaviour here: this checkout keeps
        ``[genesis]`` out of its main venv on purpose (installing it moves twenty
        packages), so what a reader running this example meets is the error, and
        an error that did not name the extra would send them to the wrong place.
        """
        import importlib.util

        if importlib.util.find_spec("genesis") is not None:
            self.skipTest("[genesis] is installed; the refusal path is not reachable")
        with self.assertRaises(ImportError) as caught:
            _run("02_cloth/03_genesis.py")
        self.assertIn("[genesis]", str(caught.exception), "the refusal must name the extra to install")

    def test_the_superdex_cable_is_a_centerline_on_the_foreign_engine(self) -> None:
        """The rope's acceptance scene on SuperDex: one spec, now a Cosserat
        centerline -- a particle per node and no capsule -- that the page grips."""
        import importlib.util

        if importlib.util.find_spec("superdex") is None:
            self.skipTest("[superdex] is not installed; the engine cannot be reached")
        sim, _ = _run("10_rope/02_superdex.py")
        cable = sim["cable"]
        self.assertEqual("superdex", sim.request.physics.backend_for("rod").solver)
        self.assertEqual("mujoco", sim.request.physics.backend_for("articulated").solver)
        self.assertEqual(29, len(cable.particles), "28 elements are 29 nodes")
        self.assertEqual(0, len(cable.entity.owned_bodies()), "a centerline owns no capsule")
        sim.run(20).sync()
        self.assertTrue(sim.state_is_finite(), "examples/10_rope/02_superdex.py diverged in twenty steps")

    def test_each_demo_stages_its_engine_for_the_domain_it_is_about(self) -> None:
        """Static, so it holds whichever wheels are present: the arrangement is data."""
        for name, engine in zip(DEMOS, ("superdex", "genesis", "superdex"), strict=True):
            with self.subTest(demo=name):
                text = (EXAMPLES / name).read_text()
                ast.parse(text)
                self.assertIn(engine, text, "the demo must name the engine it demonstrates")


class TestThePageCanTakeHoldOfIt(unittest.TestCase):
    """The claim the rewrite is *for*, checked on a real page rather than a stub.

    Every one of these examples exists so that what it put in the scene can be
    clicked and dragged. A stub records what a panel added and never refreshes,
    so it cannot tell a page that offers handles from one that offers none --
    and ``01_new_kind``'s chain, which is the whole point of that file, was
    unclickable while every stubbed test passed.
    """

    PORT = 8981

    def _page(self, name: str):
        from robosimstudio_studio.session import StudioConfig, StudioSession

        sim, kwargs = _run(name)
        TestThePageCanTakeHoldOfIt.PORT += 1
        session = StudioSession(sim, config=StudioConfig(port=self.PORT, panels=kwargs.get("panels", "drive")))
        if kwargs.get("page"):
            kwargs["page"](session)
        return sim, session

    def test_every_example_offers_a_handle_on_what_it_is_about(self) -> None:
        rope = (("scenes/cable.py", {"rope"}), ("scenes/cables.py", {"chain"}), ("scenes/ported.py", {"hose"}))
        for name, expected in (
            ("01_rigid/00_drag_things.py", {"cube", "ball", "pin"}),
            ("04_extending/01_new_kind.py", {"chain"}),
            ("10_rope/00_cable.py", {"cable"}),
            ("10_rope/01_between_posts.py", {"cable"}),
            *(rope if HAS_ROPE else ()),
        ):
            with self.subTest(example=name):
                _, session = self._page(name)
                try:
                    for _ in range(4):
                        session.tick()
                    self.assertIsNotNone(session.objects, f"examples/{name} has nothing draggable")
                    self.assertLessEqual(expected, set(session.objects.names))
                finally:
                    session.teardown()

    def test_every_example_offers_a_handle_on_the_robot(self) -> None:
        for name in SCRIPTS:
            with self.subTest(example=name):
                _, session = self._page(name)
                try:
                    session.tick()
                    self.assertEqual(["left", "right"], sorted(session.objects.arm_bodies))
                finally:
                    session.teardown()

    def test_the_deformable_pages_offer_a_grip_on_what_they_are_about(self) -> None:
        for name, thing in (
            ("02_cloth/00_towel.py", "towel"),
            ("02_cloth/02_engines.py", "towel"),
            ("03_soft/01_tet_mesh.py", "gasket"),
        ):
            with self.subTest(example=name):
                _, session = self._page(name)
                try:
                    for _ in range(4):
                        session.tick()
                    self.assertIsNotNone(session.deformables, f"the {thing} cannot be grabbed")
                    self.assertIn(thing, session.deformables.names)
                finally:
                    session.teardown()


def _numbered_scripts() -> list[str]:
    """Every numbered script under a numbered folder, whoever wrote it."""
    return sorted(path.relative_to(EXAMPLES).as_posix() for path in EXAMPLES.glob("[0-9]*_*/[0-9]*_*.py"))


class TestTheShape(unittest.TestCase):
    """What an example looks like, held as rules rather than as a habit.

    The layout and the docstring are viser's: a numbered folder per category, a
    numbered script per example, and a docstring the docs page is generated
    from -- a title, one sentence, a short paragraph, and the API it shows.
    """

    MAX_LINES = 70
    """A scene script is short enough to read in one screen and a half.
    ``01_new_kind`` is exempt: it implements a kind, which is its subject."""

    LONG = ("04_extending/01_new_kind.py",)

    def test_no_scene_script_parses_a_command_line(self) -> None:
        """The shape is pyroki's: a parameter is a local in ``main()``, not a flag.

        ``studio`` is deliberately not in this list. It is the launcher -- it
        takes a registered task id -- and a command line is what makes it one.
        """
        for name in (*SCRIPTS, *DEMOS):
            with self.subTest(example=name):
                source = (EXAMPLES / name).read_text()
                self.assertNotIn("argparse", source)
                tree = ast.parse(source)
                mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
                self.assertEqual(1, len(mains), "one main(), like a pyroki example")
                self.assertEqual([], mains[0].args.args, "main() takes no arguments")

    def test_no_scene_script_writes_its_own_loop(self) -> None:
        """What the rewrite was for. Every one of these used to open a bare
        ``rss.viewer``, spin ``while viewer.is_running()`` and report a number,
        which is a drop test with a camera on it."""
        for name in (*SCRIPTS, *DEMOS):
            with self.subTest(example=name):
                source = (EXAMPLES / name).read_text()
                self.assertNotIn("rss.viewer", source)
                self.assertNotIn("is_running", source)

    def test_each_numbered_script_ends_by_opening_the_studio(self) -> None:
        """``import robosimstudio as rss`` at the top, ``rss.view(...)`` as the last
        thing ``main()`` does, and ``main()`` called only when the file is run."""
        for name in (*NUMBERED, *DEMOS):
            with self.subTest(example=name):
                tree = ast.parse((EXAMPLES / name).read_text())
                imports = [a for n in tree.body if isinstance(n, ast.Import) for a in n.names]
                self.assertIn(("robosimstudio", "rss"), [(a.name, a.asname) for a in imports])
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                last = main.body[-1]
                self.assertIsInstance(last, ast.Expr)
                self.assertEqual("rss.view", ast.unparse(last.value.func), "main() ends by opening the page")
                guard = tree.body[-1]
                self.assertIsInstance(guard, ast.If)
                self.assertEqual("__name__ == '__main__'", ast.unparse(guard.test))
                self.assertEqual("main()", ast.unparse(guard.body[0]).strip())

    def test_each_numbered_script_is_short(self) -> None:
        for name in (*NUMBERED, *DEMOS):
            if name in self.LONG:
                continue
            with self.subTest(example=name):
                count = len((EXAMPLES / name).read_text().splitlines())
                self.assertLessEqual(count, self.MAX_LINES, f"examples/{name} is {count} lines")

    def test_each_docstring_is_the_page_it_becomes(self) -> None:
        """Title, blank line, one sentence, at most one paragraph, then the API.

        ``docs/generate_example_docs.py`` turns line 1 into the page title and
        the sentence after it into the card under the picture, so these are the
        rules of a rendered page and not of taste: a title of two to five words
        with no full stop; one sentence on one line saying what you will see and
        can do; at most one short paragraph; then ``**Key features:**`` and
        three to five bullets, each naming the API it shows.
        """
        for name in (*NUMBERED, *DEMOS):
            with self.subTest(example=name):
                doc = ast.get_docstring(ast.parse((EXAMPLES / name).read_text())) or ""
                blocks = [block.strip() for block in doc.split("\n\n")]
                title = blocks[0]
                self.assertNotIn("\n", title, "line 1 is the title, and line 2 is blank")
                self.assertTrue(2 <= len(title.split()) <= 5, f"a title of two to five words, not {title!r}")
                self.assertFalse(title.endswith("."), "a title takes no full stop")
                sentence = blocks[1]
                self.assertNotIn("\n", sentence, "the description is one sentence on one line")
                self.assertTrue(sentence.endswith("."), sentence)
                self.assertIn("**Key features:**", blocks, "the API list is introduced the way viser's is")
                at = blocks.index("**Key features:**")
                self.assertLessEqual(at - 2, 1, "at most one paragraph between the sentence and the features")
                self.assertEqual(at + 2, len(blocks), "the bullets close the docstring")
                bullets = [line for line in blocks[at + 1].splitlines() if line.startswith("* ")]
                self.assertEqual(len(bullets), len(blocks[at + 1].splitlines()), "one bullet per line")
                self.assertTrue(3 <= len(bullets) <= 5, f"three to five features, not {len(bullets)}")
                for bullet in bullets:
                    self.assertRegex(
                        bullet, r":(func|class|meth|attr|data|mod):`|``", "a feature names the API it shows"
                    )

    def test_every_numbered_script_has_a_title_the_docs_can_use(self) -> None:
        """The rule that holds for a script this module does not run -- one that
        needs hardware, or another renderer -- because the docs page is generated
        from every numbered script alike: a docstring whose first line is a
        title with no full stop, and a blank line under it."""
        found = _numbered_scripts()
        self.assertLessEqual(set(NUMBERED) | set(DEMOS), set(found), "the tables above name a script that is gone")
        for name in found:
            with self.subTest(example=name):
                doc = ast.get_docstring(ast.parse((EXAMPLES / name).read_text())) or ""
                lines = doc.splitlines()
                self.assertTrue(
                    lines and lines[0].strip(), "an example opens with a docstring, and line 1 is its title"
                )
                self.assertFalse(lines[0].rstrip().endswith("."), "a title takes no full stop")
                self.assertTrue(len(lines) == 1 or not lines[1].strip(), "line 2 is blank")

    def test_no_numbered_script_is_a_pack(self) -> None:
        """A kind's folder is on ``robosimstudio.domains``'s path, and what keeps
        its examples from being scanned as kinds is their name: it starts with a
        digit, so it is not an identifier (``tests/test_domain_packs.py``)."""
        for name in _numbered_scripts():
            with self.subTest(example=name):
                self.assertFalse(Path(name).stem.isidentifier())


ALONE = dedent(
    """
    import importlib.util, json, runpy, sys
    import robosimstudio as rss

    opened = []
    rss.view = lambda sim, **kwargs: opened.append(sim)
    runpy.run_path(sys.argv[1], run_name="__main__")
    (sim,) = opened
    sim.run(20).sync()
    other = "robosimstudio.domains." + sys.argv[2]
    print(json.dumps({
        "finite": bool(sim.state_is_finite()),
        "particles": int(sim.model.particle_count),
        "other_loaded": other in sys.modules,
        "other_findable": importlib.util.find_spec(other) is not None,
    }))
    """
).strip()


class TestEachKindsFolderStandsAlone(unittest.TestCase):
    """Each kind is its own folder and an independent example: one runs with the others gone.

    The cloth, soft and rope folders are each a root of
    ``robosimstudio.domains``. Here a fresh process is given *one* of them as its
    only root -- so the other kind is not merely unimported but not there to
    import -- and runs an example out of that folder exactly as a reader would,
    by path, through to twenty steps. ``soft.py`` once reached into ``cloth.py``
    for a helper inside two of its methods, which no import-time check could
    see and the first build did.
    """

    def _alone(self, example: str, other: str) -> dict:
        from robosimstudio import roots

        folder = (EXAMPLES / example).parent
        env = {**os.environ, roots.DOMAIN_ROOTS_ENV: str(folder), f"{roots.DOMAIN_ROOTS_ENV}_ONLY": "1"}
        proc = subprocess.run(
            [sys.executable, "-c", ALONE, str(EXAMPLES / example), other],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        self.assertEqual(0, proc.returncode, proc.stderr[-3000:])
        return json.loads([line for line in proc.stdout.splitlines() if line.startswith("{")][-1])

    def test_a_soft_example_builds_and_steps_with_no_cloth(self) -> None:
        got = self._alone("03_soft/01_tet_mesh.py", "cloth")
        self.assertFalse(got["other_findable"], "the cloth folder was meant to be off the path")
        self.assertFalse(got["other_loaded"])
        self.assertEqual(112, got["particles"])
        self.assertTrue(got["finite"])

    def test_a_cloth_example_builds_and_steps_with_no_soft(self) -> None:
        got = self._alone("02_cloth/00_towel.py", "soft")
        self.assertFalse(got["other_findable"], "the soft folder was meant to be off the path")
        self.assertFalse(got["other_loaded"])
        self.assertGreater(got["particles"], 500)
        self.assertTrue(got["finite"])

    def test_a_rope_example_builds_and_steps_with_no_cloth(self) -> None:
        got = self._alone("10_rope/00_cable.py", "cloth")
        self.assertFalse(got["other_findable"], "the cloth folder was meant to be off the path")
        self.assertFalse(got["other_loaded"])
        self.assertEqual(0, got["particles"], "a capsule chain is bodies, not particles")
        self.assertTrue(got["finite"])


class TestTheDoorsDifferOnPurpose(unittest.TestCase):
    """``rss.view`` and ``StudioConfig`` disagree on two defaults, both deliberately.

    ``view`` is the door a *person* opens by hand, so it shows the drive
    selection and draws every step. A test or a batch job builds the config
    itself: it gets every panel and keeps its step budget. Neither door shares:
    a share URL puts the page on the internet, so it is asked for, never assumed.
    """

    def test_no_door_shares_by_default(self) -> None:
        import inspect

        import robosimstudio as rss
        from robosimstudio.sim.draw import _UNSET
        from robosimstudio_studio.session import StudioConfig

        self.assertFalse(StudioConfig().share, "a config built directly must not open a tunnel")
        self.assertIs(_UNSET, inspect.signature(rss.view).parameters["share"].default)
        for door in (rss.viewer, rss.draw):
            with self.subTest(door=door.__name__):
                self.assertIs(False, inspect.signature(door).parameters["share"].default)

    def test_view_draws_every_step_and_a_bare_config_does_not(self) -> None:
        """The split default with a number behind it.

        The browser is redrawn once per ``render_every`` control steps, so on a
        scene that steps slowly the frame rate is the *step* rate divided by it.
        A coupled cloth scene steps at about 30 Hz: at 4 that is 5 fps of a towel
        somebody is trying to drag, measured at 5.7 against 26.4 at 1. The
        throughput it costs (33.3 steps/s to 26.4) is what a benchmark keeps and
        a page being watched does not need.
        """
        from robosimstudio.sim.draw import _VIEW_DEFAULTS
        from robosimstudio_studio.session import StudioConfig

        self.assertEqual({"panels": "drive", "render_every": 1}, _VIEW_DEFAULTS)
        self.assertEqual(4, StudioConfig().render_every, "the measured page must keep its step budget")
        self.assertEqual("all", StudioConfig().panels)


class TestTheScriptSeam(unittest.TestCase):
    """``01_new_kind`` adds a readout of its own through ``page=``, which is the
    supported way to put something on the studio's page without forking it."""

    def test_the_new_kind_hands_view_a_page_hook_that_works(self) -> None:
        from tests.fixtures.viser_stubs import _StubGui, _StubScene

        sim, kwargs = _run("04_extending/01_new_kind.py")
        page = kwargs.get("page")
        self.assertIsNotNone(page, "01_new_kind stopped adding its own readout")

        class _Studio:
            def __init__(self) -> None:
                self.gui = _StubGui()
                self.scene = _StubScene()
                self.steps: list = []

            def each_step(self, fn) -> None:
                self.steps.append(fn)

        studio = _Studio()
        page(studio)
        self.assertTrue(studio.steps, "the hook registered no per-step work")
        sim.run(5).sync()
        for fn in studio.steps:
            fn()  # reads this file's own ``ChainRef.span``; must not raise


if __name__ == "__main__":
    unittest.main()
