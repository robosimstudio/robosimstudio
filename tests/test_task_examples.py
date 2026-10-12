# SPDX-License-Identifier: Apache-2.0
"""The task and policy examples run whole, and the graphs in them do what they say.

Each example is loaded by file path and its ``main()`` run with the studio
replaced by a stub, as ``tests/test_new_examples.py`` does for the scene
examples. The task examples need no extra. The policy examples need the graph
runtime (the ``[gap]`` extra) and are skipped without it; with it, each graph
must solve its task -- a pick and place that does not place is a broken
example, not a slow one.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path

TIER = "medium"
"""Builds scenes and runs graphs on a GPU."""

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

HAS_GAP = importlib.util.find_spec("gap") is not None
NO_GAP = "the graph runtime is not installed (the [gap] extra)"


def _run(rel: str) -> tuple[str, list]:
    """Load one example fresh, run ``main()`` with ``rss.view`` stubbed; return what it printed and opened."""
    import robosimstudio as rss

    name = "_rss_task_example_" + re.sub(r"\W", "_", rel.removesuffix(".py"))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    opened: list = []
    real = rss.view
    rss.view = lambda sim, **kwargs: opened.append(sim)
    out = io.StringIO()
    try:
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        rss.view = real
    return out.getvalue(), opened


class TestTheTaskExamples(unittest.TestCase):
    def test_the_first_task_opens_an_env_whose_goal_does_not_hold_yet(self) -> None:
        _, opened = _run("07_tasks/00_your_first_task.py")
        self.assertEqual(1, len(opened))
        env = opened[0]
        self.assertTrue(hasattr(env, "task"), "the studio was handed the benchmark's Env")
        self.assertIn("square", env.handles.regions)
        self.assertFalse(env.evaluate()[-1]["success"])

    def test_the_goal_example_reads_partial_credit(self) -> None:
        printed, _ = _run("07_tasks/01_goals_and_rewards.py")
        found = re.search(r"success=(\w+) reward=([\d.]+)", printed)
        self.assertIsNotNone(found, printed)
        self.assertEqual("False", found.group(1))
        self.assertAlmostEqual(2 / 3, float(found.group(2)), places=2, msg="still and clear, not in the square")

    def test_the_same_episode_starts_the_same_way(self) -> None:
        printed, _ = _run("07_tasks/02_episodes.py")
        starts = re.findall(r"episode (\d+): the cube starts at (.+)", printed)
        self.assertEqual(["0", "1", "0"], [k for k, _ in starts])
        self.assertEqual(starts[0][1], starts[2][1])
        self.assertNotEqual(starts[0][1], starts[1][1])


@unittest.skipUnless(HAS_GAP, NO_GAP)
class TestThePolicyExamples(unittest.TestCase):
    def test_pick_and_place_places(self) -> None:
        printed, opened = _run("08_policies/00_pick_and_place.py")
        self.assertIn("success=True", printed)
        self.assertEqual(1, len(opened))

    def test_check_and_retry_solves_its_turned_cubes(self) -> None:
        printed, _ = _run("08_policies/01_check_and_retry.py")
        wins = int(re.search(r"(\d+) of 4 episodes succeeded", printed).group(1))
        self.assertGreaterEqual(wins, 3, printed)

    def test_the_camera_graph_places_without_privilege(self) -> None:
        printed, _ = _run("08_policies/02_without_privilege.py")
        self.assertIn("success=True", printed)

    def test_many_worlds_reports_a_rate(self) -> None:
        printed, _ = _run("08_policies/03_many_worlds.py")
        wins = int(re.search(r"(\d+) of 16 succeeded", printed).group(1))
        self.assertGreaterEqual(wins, 14, printed)


class TestTheGoalAndTheRetry(unittest.TestCase):
    """What the examples' docstrings promise beyond one clean run."""

    def test_the_first_goal_turns_true_with_the_cube_in_the_square(self) -> None:
        import robosimstudio as rss
        import robosimstudio_bench as bench
        from robosimstudio.assetlib import register_asset
        from robosimstudio.domains.rigid import Body

        _run("07_tasks/00_your_first_task.py")  # registers nothing a second load would clash with
        inside = rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.10, -0.15, 0.03))
        register_asset(Body.from_spec(inside, name="cube_in_square_probe"))
        module = sys.modules["_rss_task_example_07_tasks_00_your_first_task"]
        env = bench.Env(bench.task(module.TASK.replace("cube_50mm", "cube_in_square_probe"), suite="first_inside"))
        env.reset()
        for _ in range(25):
            env.step()
        self.assertTrue(env.evaluate()[-1]["success"], "the same document, the cube resting in the square")

    @unittest.skipUnless(HAS_GAP, NO_GAP)
    def test_a_missed_grasp_routes_to_the_second_pick(self) -> None:
        """The retry graph with its first grasp aimed beside the cube: the check sees air, the second pick places it."""
        import json
        import shutil
        import tempfile

        import robosimstudio_bench as bench
        from robosimstudio_policy_agent import SolutionGraph

        _run("08_policies/00_pick_and_place.py")  # its cube type and its document
        module = sys.modules["_rss_task_example_08_policies_00_pick_and_place"]
        tmp = Path(tempfile.mkdtemp(prefix="retry-"))
        graph = tmp / "pick_place_retry"
        shutil.copytree(EXAMPLES / "08_policies" / "pick_place_retry", graph)
        # A real miss: the first plan aims 8 cm to the side of the cube, so the hand shuts
        # on air and the graph's own grip check has to notice.
        plan = graph / "scripts" / "plan_pick.py"
        flag = tmp / "miss_once"
        flag.write_text("")
        plan.write_text(
            plan.read_text()
            .replace("import math\n", "import math\nimport os\n")
            .replace(
                '    p, q = cube["position"], cube["rotation"]\n',
                '    p, q = dict(cube["position"]), cube["rotation"]\n'
                f"    if os.path.exists({str(flag)!r}):\n"
                f"        os.remove({str(flag)!r})\n"
                '        p["y"] += 0.08\n',
            )
        )
        env = bench.Env(bench.task(module.TASK, suite="retry_probe"))
        trace = tmp / "trace"
        result = bench.rollout(env, SolutionGraph(env, graph=str(graph), roll="held", trace_dir=str(trace)))
        self.assertFalse(flag.exists(), "the first plan aimed beside the cube")
        self.assertTrue(result.success, "the second pick placed the cube")
        ran = json.dumps(json.loads(next(trace.rglob("dag_trace.json")).read_text()))
        self.assertIn("pick_again", ran)


if __name__ == "__main__":
    unittest.main()
