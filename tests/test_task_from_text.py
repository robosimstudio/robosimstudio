# SPDX-License-Identifier: Apache-2.0
"""A task written as a string, the examples that teach it, and a release with graphs only.

``robosimstudio_bench.task`` lowers a document through the same function a folder
of documents goes through; the first class checks what it registers and what it
refuses, with no scene built. The second runs in a subprocess with the scripted
routine module refused, which is the shape of a release that leaves it out. The
third holds the task and policy examples to the docstring and length rules the
other examples keep.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TIER = "fast"

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = """
import robosimstudio as rss
from robosimstudio.assetlib import register_asset
from robosimstudio.domains.rigid import Body

register_asset(Body.from_spec(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.1, 0.03)), name="text_probe_cube"))
"""

DOCUMENT = """
(define (problem text_probe)
  (:language put the cube in the square)
  (:staging (:scene bench_075) (:embodiment panda) (:physics mujoco_500hz@v1))
  (:objects cube - text_probe_cube)
  (:regions (square (:ranges ((0.05 -0.20 0.15 -0.10)))))
  (:goal (>= (containment cube square) 1.0))
  (:artifacts (:graph pick "pick_place")))
"""


def _run(script: str, *, refuse: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """``script`` in a fresh interpreter that scans no task folder, with ``refuse`` unimportable."""
    blocker = textwrap.dedent(
        f"""
        import importlib.abc, sys
        class _Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if any(name == r or name.startswith(r + ".") for r in {refuse!r}):
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
                return None
        sys.meta_path.insert(0, _Refuse())
        """
    )
    env = {
        **os.environ,
        "ROBOSIMSTUDIO_TASK_ROOTS": "",
        "ROBOSIMSTUDIO_TASK_ROOTS_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    return subprocess.run(
        [sys.executable, "-c", blocker + PREAMBLE + textwrap.dedent(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


class TestATaskFromAString(unittest.TestCase):
    def test_it_registers_and_its_graph_resolves_beside_the_caller(self) -> None:
        done = _run(
            f"""
            import robosimstudio_bench as bench
            from robosimstudio_bench.taskbuild.artifacts import graph_dir
            task = bench.task({DOCUMENT!r}, home="/somewhere/else")
            assert task.full_name == "text_probe/text_probe", task.full_name
            assert bench.get_task("text_probe") is task
            assert graph_dir(task.suite, task.graph_path("pick")) == "/somewhere/else/pick_place"
            again = bench.task({DOCUMENT!r}, home="/somewhere/else")
            assert bench.get_task("text_probe") is again, "stating it again replaces it"
            print("ok")
            """
        )
        self.assertEqual(0, done.returncode, done.stderr[-3000:])
        self.assertIn("ok", done.stdout)

    def test_without_home_a_script_s_graphs_resolve_beside_it(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="home-"))
        script = folder / "probe.py"
        script.write_text(
            PREAMBLE
            + "import robosimstudio_bench as bench\n"
            + "from robosimstudio_bench.taskbuild.artifacts import graph_dir\n"
            + f"task = bench.task({DOCUMENT!r})\n"
            + "print(graph_dir(task.suite, task.graph_path('pick')))\n"
        )
        env = {
            **os.environ,
            "ROBOSIMSTUDIO_TASK_ROOTS": "",
            "ROBOSIMSTUDIO_TASK_ROOTS_ONLY": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        done = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False
        )
        self.assertEqual(0, done.returncode, done.stderr[-3000:])
        self.assertEqual(str(folder.resolve() / "pick_place"), done.stdout.strip().splitlines()[-1])

    def test_a_measure_nothing_implements_is_refused_when_stated(self) -> None:
        bad = DOCUMENT.replace("(containment cube square)", "(sparkle cube)")
        done = _run(
            f"""
            import robosimstudio_bench as bench
            try:
                bench.task({bad!r})
            except Exception as exc:
                print("refused:", type(exc).__name__, exc)
            else:
                print("accepted")
            """
        )
        self.assertEqual(0, done.returncode, done.stderr[-3000:])
        self.assertIn("refused:", done.stdout)
        self.assertIn("sparkle", done.stdout)


class TestTheSurface(unittest.TestCase):
    def test_the_lazy_names_are_listed(self) -> None:
        """What the API page is built from: dir() names every flat name, imported or not."""
        import robosimstudio_bench
        import robosimstudio_policy_agent

        self.assertLessEqual({"task", "Env", "rollout", "register_sampler"}, set(dir(robosimstudio_bench)))
        self.assertLessEqual(
            {"SolutionGraph", "SolutionGraphBatched", "check_graph"}, set(dir(robosimstudio_policy_agent))
        )


class TestAReleaseWithGraphsOnly(unittest.TestCase):
    """With ``robosimstudio_bench.solutions.routines`` refused, every door says so in one sentence."""

    ROUTINES = ("robosimstudio_bench.solutions.routines",)

    def test_each_door_refuses_a_routine_and_offers_graphs(self) -> None:
        routine = DOCUMENT.replace('(:graph pick "pick_place")', '(:graph pick "pick_place") (:routine "x.y:Z")')
        done = _run(
            f"""
            import importlib.util
            import robosimstudio_bench as bench
            from robosimstudio_bench.solutions.declared import ROUTINES_ABSENT, declared_solutions, routines_installed
            assert not routines_installed()
            try:
                from robosimstudio_bench.solutions import SolutionRoutine
            except ImportError as exc:
                assert str(exc) == ROUTINES_ABSENT, exc
            else:
                raise AssertionError("SolutionRoutine imported")
            try:
                bench.task({routine!r})
            except Exception as exc:
                assert ROUTINES_ABSENT in str(exc), exc
            else:
                raise AssertionError("a document naming a routine was lowered")
            task = bench.task({DOCUMENT!r})
            assert declared_solutions(task) == ("pick",), declared_solutions(task)
            spec = importlib.util.spec_from_file_location("bench_run", "scripts/bench_run.py")
            run = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run)
            assert "routine" not in run.SOLUTIONS and "scripted" not in run.SOLUTION_ALIASES, run.SOLUTIONS
            import contextlib, io, sys
            sys.argv = ["bench_run.py", "--task", "text_probe", "--solve", "routine"]
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = run.main()
            assert code == 2 and ROUTINES_ABSENT in err.getvalue(), (code, err.getvalue())
            print("ok")
            """,
            refuse=self.ROUTINES,
        )
        self.assertEqual(0, done.returncode, done.stderr[-3000:])
        self.assertIn("ok", done.stdout)


class TestTheTaskAndPolicyExamples(unittest.TestCase):
    """The rules ``tests/test_examples.py`` holds the scene examples to, over these two folders."""

    FOLDERS = ("07_tasks", "08_policies")
    HEADLESS = ("08_policies/03_many_worlds.py",)
    """A batched run of sixteen worlds has no single scene to open."""

    def scripts(self) -> list[str]:
        found = [f"{f}/{p.name}" for f in self.FOLDERS for p in sorted((ROOT / "examples" / f).glob("[0-9]*.py"))]
        self.assertGreaterEqual(len(found), 7)
        return found

    def test_each_is_short_and_runs_alone(self) -> None:
        for name in self.scripts():
            with self.subTest(example=name):
                source = (ROOT / "examples" / name).read_text()
                self.assertLessEqual(len(source.splitlines()), 70)
                self.assertNotIn("argparse", source)
                tree = ast.parse(source)
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                self.assertEqual([], main.args.args)
                if name not in self.HEADLESS:
                    self.assertEqual("rss.view", ast.unparse(main.body[-1].value.func))
                self.assertEqual("__name__ == '__main__'", ast.unparse(tree.body[-1].test))

    def test_each_docstring_is_the_page_it_becomes(self) -> None:
        for name in self.scripts():
            with self.subTest(example=name):
                doc = ast.get_docstring(ast.parse((ROOT / "examples" / name).read_text())) or ""
                blocks = [block.strip() for block in doc.split("\n\n")]
                self.assertTrue(2 <= len(blocks[0].split()) <= 5 and not blocks[0].endswith("."), blocks[0])
                self.assertNotIn("\n", blocks[1])
                self.assertTrue(blocks[1].endswith("."), blocks[1])
                at = blocks.index("**Key features:**")
                self.assertLessEqual(at - 2, 1)
                bullets = blocks[at + 1].splitlines()
                self.assertTrue(3 <= len(bullets) <= 5 and all(b.startswith("* ") for b in bullets), bullets)

    def test_no_example_ships_a_task_document_file(self) -> None:
        for folder in self.FOLDERS:
            found = [p for p in (ROOT / "examples" / folder).rglob("*") if p.suffix in (".task", ".suite")]
            self.assertEqual([], found, "a task is a string in its script; no task document ships")


if __name__ == "__main__":
    unittest.main()
