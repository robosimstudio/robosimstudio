# SPDX-License-Identifier: Apache-2.0
"""A scene built from nothing but specs, stepped, with no benchmark in the process.

The hello-world this stack could not write. ``StaticSpec.half_extents`` has
always drawn a procedural box, but only a welded one: an ``ObjectSpec`` with no
asset means *"declared by the cell's own import, only find it"*, so
``EntityRigid.build`` returned early and **there was no way to drop a cube on a
table** without shipping a mesh for it. ``rigid`` is the most-used domain in the
corpus and it was the one kind you could not make from nothing.

Three claims, and the third is the one that caught a real bug:

* a request with no ``TaskSpec`` compiles and steps;
* the benchmark is never imported to do it;
* **a procedural body lands where the frame says it should.** ``_placed``
  returned the pose unchanged whenever ``asset is None``, because that used to
  mean exactly one thing. A ``ShapeSpec`` object has no asset *and* is placed
  right here, so it was spawned at the surface frame's origin -- inside the
  bench -- and fell through to the floor. A test that only asserted "it steps"
  would have passed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from robosimstudio.domains.cloth import ClothSpec
from robosimstudio.domains.rod import RodSpec

from robosimstudio.scene import BuildRequest, compile_scene
from robosimstudio.scene.state import WorldState
from robosimstudio.specs import GENERIC, ArrangementSpec, ObjectSpec, StaticSpec
from robosimstudio.specs.asset import ShapeSpec

TIER = "medium"
"""Builds and steps real scenes on a GPU."""

ROOT = Path(__file__).resolve().parents[1]

BENCH = "robosimstudio_bench.taskbuild.discovery"
"""Importing this is what executes every task directory on disk."""


def _drop(cell: str, *, steps: int = 150):
    """One cube, released above the work surface, settled."""
    request = BuildRequest(
        name="bare",
        cell=cell,
        entities=(
            ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0.0, 0.0, 0.20)),
            StaticSpec(
                name="post",
                shape=ShapeSpec(kind="cylinder", radius=0.02, half_height=0.06),
                xform=(0.0, 0.25, 0.06, 0.0, 0.0, 0.0, 1.0),
            ),
        ),
        physics=ArrangementSpec(),
    )
    scene = compile_scene(request)
    for _ in range(steps):
        scene.pipeline.step()
    scene.pipeline.sync()
    return scene


class TestASceneNeedsNoBenchmark(unittest.TestCase):
    def test_a_request_compiles_and_steps(self) -> None:
        scene = _drop("void")
        self.assertEqual(1, scene.model.body_count, "one free body: the cube")
        self.assertTrue(scene.pipeline.sim_time > 0.0)

    def test_a_reading_needs_no_task(self) -> None:
        scene = _drop("void")
        state = WorldState(handles=scene.handles, state=scene.pipeline.state_0)
        self.assertEqual(0, state.step)


class TestAProceduralBodyLandsWhereTheFrameSaysIt(unittest.TestCase):
    """The frame is the whole reason a scene can move to a taller table."""

    def _settled_z(self, cell: str) -> float:
        scene = _drop(cell)
        body = scene.handles.objects["cube"].body
        return float(scene.pipeline.state_0.body_q.numpy()[body][2])

    def test_on_bare_ground(self) -> None:
        self.assertAlmostEqual(0.025, self._settled_z("void"), delta=0.005)

    def test_on_a_bench_the_frame_composes(self) -> None:
        """0.775, not 0.025. The pose is stated in the work-surface frame, so a
        cube released 200 mm over a 750 mm bench rests 25 mm above its top --
        and a procedural body that skipped the frame fell through the bench
        instead, which is exactly what this number catches."""
        self.assertAlmostEqual(0.775, self._settled_z("bench_075"), delta=0.005)


class TestADeformableNeedsNoTaskEither(unittest.TestCase):
    """The other two kinds somebody would reach for first.

    A rod and a sheet are the reason this refactor exists -- "load a cable and
    watch it fall" was five files and a BDDL dialect away -- so each is built
    here the way the README will tell someone to build it: a spec, ``GENERIC``,
    and nothing else. No asset file, no registered entity name of anybody's, no
    robot.

    :data:`~robosimstudio.specs.physics.GENERIC` rather than ``ArrangementSpec()``
    on purpose, and the two failures it avoids are different from each other. A
    sheet under the dataclass default **raises**: MuJoCo integrates no particles
    and ``build_islands`` says so. A rod under it **does not** -- bodies and
    joints are exactly what MuJoCo owns, so the chain composes silently and
    bends like a row of hinges. Only one of those two is a bug you would find.
    """

    def _settle(self, spec, *, steps: int = 120):
        request = BuildRequest(name="bare", cell="bench_075", entities=(spec,), physics=GENERIC)
        scene = compile_scene(request)
        for _ in range(steps):
            scene.pipeline.step()
        scene.pipeline.sync()
        return scene

    def test_a_rod_falls_and_settles(self) -> None:
        """No ``entity=``: ``RodSpec`` defaults to the library's own ``"rod"``.

        It defaulted to ``"cable_rod"``, which only the cable suite's own task
        directory registers, so this scene raised ``unknown entity 'cable_rod'`` unless a
        benchmark had been imported first.

        **The rod's start is in the work-surface frame**, like every rigid
        pose: released 150 mm over the surface it settles *on the bench* at its
        own radius, not on the floor. This asserts the bench height precisely
        because a rod that skipped the frame would land 750 mm lower and fail
        here -- which is what it did before ``EntityRod.build`` ran ``start``
        through :meth:`~robosimstudio.scene.entity.Entity.placed`.
        """
        scene = self._settle(RodSpec(name="rope", start=(0.0, -0.2, 0.15), segments=16, length=0.4))
        z = scene.pipeline.state_0.body_q.numpy()[:, 2]
        self.assertEqual(16, len(z), "one body per capsule")
        # Released 150 mm over a 750 mm bench and settled onto it: every capsule
        # rests near 0.761 (the surface plus the rod's radius), and none went
        # through the bench to the floor.
        self.assertLess(float(z.max()), 0.78, "the rod never fell")
        self.assertGreater(float(z.min()), 0.74, "a capsule fell through the bench")

    def test_a_sheet_settles(self) -> None:
        """Procedural: ``mesh="grid"`` needs no file. The particles are the check
        -- a cloth that built but never solved keeps its released height.

        Work-surface frame like the rod; see :meth:`test_a_rod_falls_and_settles`."""
        scene = self._settle(ClothSpec(name="towel", size=(0.3, 0.3), cells=(16, 16), pose=(0.0, 0.0, 0.12)))
        z = scene.pipeline.state_0.particle_q.numpy()[:, 2]
        self.assertEqual(17 * 17, len(z), "a 16x16 grid has 17x17 particles")
        self.assertLess(float(z.max()), 0.78, "the sheet never fell")
        self.assertGreater(float(z.min()), 0.74, "the sheet fell through the bench")


class TestBuildReturnsSomethingWarm(unittest.TestCase):
    """``build()`` compiles the kernels and captures the substep block.

    Compiling a *scene* does not compile its kernels: ``warmup_and_capture``
    does, and it runs on the first ``reset``. Every caller of an ``Env`` resets
    it to pick an episode, so the benchmark path was always warm and the gap
    never showed. ``scene.build().run(1000)`` has no such moment, and stepped
    **eager** -- a supported way to run, about an order of magnitude slower, with
    nothing anywhere saying so.

    Pinned on ``_warmed`` rather than on the graph itself, because capture is
    allowed to fail: ``warmup_and_capture`` prints and falls back to eager on a
    device that cannot capture, and a test that demanded a graph would be a test
    about the hardware.
    """

    def _sim(self, **kwargs):
        from robosimstudio.sim import Scene

        scene = Scene(cell="bench_075")
        cube = scene.add(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0.0, 0.0, 0.20)))
        return scene.build(**kwargs), cube

    def test_it_is_warm(self) -> None:
        sim, _ = self._sim()
        self.assertTrue(sim._warmed, "build() returned a Sim that would step eager")

    def test_warmup_false_is_the_opt_out(self) -> None:
        """For a caller who wants the model and does not intend to step it."""
        sim, _ = self._sim(warmup=False)
        self.assertFalse(sim._warmed)

    def test_it_changes_nothing_about_where_things_land(self) -> None:
        """The warm-up steps real physics and restores the scene afterwards, so
        the result is the same either way -- which is what makes it safe to do
        without being asked."""
        warm, cube_warm = self._sim()
        cold, cube_cold = self._sim(warmup=False)
        warm.run(150).sync()
        cold.run(150).sync()
        self.assertAlmostEqual(float(cube_warm.position[2]), float(cube_cold.position[2]), places=4)


class TestRunSaysWhenItDiverged(unittest.TestCase):
    """A diverged run looks exactly like a fast one.

    It steps, it returns, and every number after it is ``nan``. An evaluator
    catches that -- ``scripts/bench_smoke.py`` and half the tools call
    ``state_is_finite`` -- but the caller most likely to hit it is the one with
    no evaluator: somebody stepping a scene they assembled themselves.

    The NaN is **planted**, not provoked. This used to build a scene that
    genuinely diverged -- a rod against a mesh body with Newton's default
    64-row contact buffer -- and the moment ``GENERIC`` named 256 rows, no
    recipe short of absurd stiffness would diverge any more. A test whose
    premise is "this arrangement blows up" is a test about a solver's buffer
    sizes; what is under test here is the *reporting*, and writing a NaN into
    ``body_q`` exercises exactly that path and nothing else.
    """

    def _poisoned(self):
        import numpy as np

        from robosimstudio.sim import Scene

        scene = Scene(cell="void")
        scene.add(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0.0, 0.0, 0.20)))
        sim = scene.build()
        body_q = sim.pipeline.state_0.body_q.numpy()
        body_q[0, 2] = np.nan
        sim.pipeline.state_0.body_q.assign(body_q)
        return sim

    def test_it_says_so_once(self) -> None:
        import contextlib
        import io

        sim = self._poisoned()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sim.run(5)
            sim.run(5)
        self.assertFalse(sim.state_is_finite(), "a planted NaN did not propagate; the read-back flag has moved")
        said = [line for line in out.getvalue().splitlines() if line.startswith("[sim]")]
        self.assertEqual(1, len(said), f"expected one line, got {said}")
        self.assertIn("nan", said[0])

    def test_a_healthy_scene_says_nothing(self) -> None:
        import contextlib
        import io

        from robosimstudio.sim import Scene

        scene = Scene(cell="bench_075")
        scene.add(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0.0, 0.0, 0.20)))
        sim = scene.build()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sim.run(120)
        self.assertTrue(sim.state_is_finite())
        self.assertEqual([], [ln for ln in out.getvalue().splitlines() if ln.startswith("[sim]")])


class TestTheBenchmarkIsNeverImported(unittest.TestCase):
    """The claim the whole tier split rests on, made where it can be made.

    A cell and a robot the *library* ships used to drag in every task directory
    on disk, because the only populator those registries had was the task-root
    scanner: ``get_cell("bench_075")`` is a question about
    ``scene/builtin/benches.py`` and it executed thirty thousand lines of
    benchmark content to answer it.

    **In a subprocess, and that is not fastidiousness.** ``scripts/testrun.py``
    hands every module in a tier to one ``python -m unittest``, so ``sys.modules``
    is shared across the run: asserted in-process this passes alone, passes in a
    small selection, and fails in the full tier the moment any earlier module
    touches the benchmark. That is a test measuring the run order rather than the
    property -- the failure mode ``tests/known/known_red.json`` already records
    once -- and a fresh interpreter is what makes the claim about the library.
    """

    def test_a_bare_scene_does_not_execute_the_task_directories(self) -> None:
        script = textwrap.dedent(
            f"""
            import sys
            from robosimstudio.domains.rod import RodSpec
            from robosimstudio.scene import BuildRequest, compile_scene
            from robosimstudio.specs import GENERIC, ObjectSpec
            from robosimstudio.specs.asset import ShapeSpec

            compile_scene(BuildRequest(
                name="bare", cell="bench_075", physics=GENERIC,
                entities=(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0, 0, 0.2)),
                          RodSpec(name="rope", start=(0.0, -0.2, 0.15), segments=16, length=0.4))))
            leaked = sorted(m for m in sys.modules if m.startswith({BENCH!r}))
            print("LEAKED:" + ",".join(leaked))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=900, check=False
        )
        self.assertEqual(0, result.returncode, f"the bare scene failed to build:\n{result.stderr[-3000:]}")
        line = next((ln for ln in result.stdout.splitlines() if ln.startswith("LEAKED:")), None)
        self.assertIsNotNone(line, f"the probe printed nothing:\n{result.stdout[-2000:]}")
        self.assertEqual(
            "LEAKED:",
            line,
            "building a bare scene executed the task directories; a library cell or a library "
            "entity name is reaching the benchmark's discovery again",
        )


class TestTheSceneApi(unittest.TestCase):
    """``Scene`` -> ``Sim`` -> ``Ref``, on a stepped scene.

    The authoring half is checked without a GPU in
    :class:`tests.test_core_fixes.TestSceneAddBindsAndRefuses`; what needs one is that the ref you were handed
    at ``add`` time reads the state that exists *now*.
    """

    def _scene(self):
        from robosimstudio.sim import Scene

        scene = Scene(cell="bench_075", name="probe")
        cube = scene.add(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0.0, 0.0, 0.20)))
        return scene, cube

    def test_the_ref_from_add_reads_the_built_scene(self) -> None:
        scene, cube = self._scene()
        sim = scene.build()
        sim.run(150).sync()
        self.assertAlmostEqual(0.775, float(cube.position[2]), delta=0.005)
        self.assertEqual(1, cube.bodies.size)
        self.assertTrue(cube.shapes.size)

    def test_the_ref_follows_a_rebuild(self) -> None:
        """A handle that silently points at the build before last is the hardest
        kind of wrong to see: every number it returns is plausible."""
        scene, cube = self._scene()
        first = scene.build()
        first.run(150).sync()
        settled = float(cube.position[2])
        second = scene.build()
        second.sync()
        self.assertGreater(float(cube.position[2]), settled + 0.1, "the ref still reads the first build")

    def test_the_sim_names_what_the_scene_holds(self) -> None:
        scene, _ = self._scene()
        sim = scene.build()
        self.assertIn("cube", sim.names())
        self.assertIn("table", sim.names(), "the cell's own furniture is reachable too")
        self.assertEqual("cube", sim["cube"].name)
        with self.assertRaises(KeyError):
            sim["nothing"]

    def test_read_gives_the_same_world_state_a_measure_gets(self) -> None:
        scene, _ = self._scene()
        sim = scene.build()
        sim.run(10)
        state = sim.read()
        self.assertIsInstance(state, WorldState)
        self.assertEqual(10, state.step)
        self.assertGreater(state.sim_time, 0.0)


class TestEveryCombinationLands(unittest.TestCase):
    """A cube, a rope and a robot, in every combination, all land.

    The second thing anybody tries after a cube on a bench is a cube *and* a
    rope, and for a while it was the one scene the default arrangement got
    wrong: the cube bounced between 7 and 195 mm forever with ``body_qd`` at
    zero. Moving free bodies onto VBD fixed that and broke cube-and-robot
    instead -- frozen at spawn -- because the coupler decided the soft side of a
    channel by the nodes' *roles*, and a VBD island holding only a cube read as
    rigid. ``Island.is_deformable`` keys on the solver's capability now, and
    this is the matrix that says all four shapes agree.

    0.775 is the bench top plus the half extent; the tolerance covers the
    half-millimetre by which VBD's and MuJoCo's contact resolution differ.
    """

    def _cube_z(self, *, rope: bool, robot: bool) -> float:
        from robosimstudio.domains.rod import RodSpec

        from robosimstudio.sim import Scene

        scene = Scene(cell="bench_075")
        cube = scene.add(ObjectSpec(name="cube", shape=ShapeSpec(half_extents=(0.025,) * 3), pose=(0.0, 0.0, 0.20)))
        if rope:
            scene.add(RodSpec(name="rope", start=(-0.1, -0.2, 0.08), segments=16, length=0.4))
        if robot:
            from robosimstudio.sim import robot as place

            scene.add(place("panda"))
        sim = scene.build()
        sim.run(200).sync()
        self.assertTrue(sim.state_is_finite())
        return float(cube.position[2])

    def test_cube(self) -> None:
        self.assertAlmostEqual(0.775, self._cube_z(rope=False, robot=False), delta=0.002)

    def test_cube_and_rope(self) -> None:
        self.assertAlmostEqual(0.775, self._cube_z(rope=True, robot=False), delta=0.002)

    def test_cube_and_robot(self) -> None:
        self.assertAlmostEqual(0.775, self._cube_z(rope=False, robot=True), delta=0.002)

    def test_cube_rope_and_robot(self) -> None:
        self.assertAlmostEqual(0.775, self._cube_z(rope=True, robot=True), delta=0.002)


if __name__ == "__main__":
    unittest.main()
