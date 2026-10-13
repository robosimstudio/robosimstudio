# SPDX-License-Identifier: Apache-2.0
"""Task documents and runs: the small rules that fail quietly when broken.

Each test here pins a behaviour whose failure produced a plausible number or a
confusing refusal rather than an error: a job seed nothing read, a one-clause
staging section refused, a reference the reader split in two, a gate that
scored 1.0 every step, a peak that thresholded the wrong quantity. No solver,
no scene, no GPU.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio.scene.state import WorldState
from robosimstudio.specs import EpisodeSpec, JobSpec, TaskSpec
from robosimstudio_bench.goals import query as query_module
from robosimstudio_bench.goals.combinators import Integral, Leaf, Peak
from robosimstudio_bench.goals.context import diverged_result, finite_reward
from robosimstudio_bench.goals.reward import Progress, _clip, build_reward
from robosimstudio_bench.run.jobs import job_task
from robosimstudio_bench.run.pool import _env_key
from robosimstudio_bench.run.summary import RunSummary
from robosimstudio_bench.taskbuild.episodes import make_episode, pick_instruction, with_base_seed
from robosimstudio_bench.taskbuild.staging import Staging, parse_staging
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp
from robosimstudio_bench.taskdoc.typecheck import _check_reward

TIER = "fast"


def _task(seed: int = 5) -> TaskSpec:
    return TaskSpec(name="t", episodes=EpisodeSpec(count=3, seed=seed))


def _ctx(t: float = 0.0) -> WorldState:
    return WorldState(handles=None, state=None, sim_time=t)


class TestJobSeed(unittest.TestCase):
    """``JobSpec.seed`` overrides the base seed, and the row's seed is the effective one."""

    def test_unset_keeps_the_task_itself(self):
        task = _task()
        self.assertIs(with_base_seed(task, -1), task)
        self.assertIs(job_task(JobSpec(name="j", task="x"), task), task)

    def test_override_reaches_the_episode_seed(self):
        task = job_task(JobSpec(name="j", task="x", seed=100), _task())
        self.assertEqual(task.episodes.seed, 100)
        self.assertEqual(make_episode(task, 2).seed, 102)

    def test_a_sampler_that_enumerates_reads_the_seed(self):
        """``pick_instruction`` cycles by the seed, so instance documents differ."""
        sample = pick_instruction("a", "b", "c")
        self.assertEqual(sample(_task(seed=0), 1, None)["instruction"], "b")
        self.assertEqual(sample(_task(seed=1), 1, None)["instruction"], "c")


class TestPoolEnvKey(unittest.TestCase):
    """Everything that reaches ``Env`` in ``_build`` forces a rebuild."""

    def test_control_mode_obs_renderer_and_seed_are_in_the_key(self):
        base = JobSpec(name="j", task="x")
        for changed in (
            JobSpec(name="j", task="x", control_mode="joint"),
            JobSpec(name="j", task="x", obs_renderer="tiled"),
            JobSpec(name="j", task="x", seed=3),
        ):
            self.assertNotEqual(_env_key(base), _env_key(changed))


class TestUnrun(unittest.TestCase):
    """A ``documents`` job runs each id once per document; ``unrun`` matches both."""

    def test_documents_are_matched_by_document_and_id(self):
        job = JobSpec(name="j", task="s/a", documents=("s/a", "s/b"), episodes=(0, 2))
        rows = [{"task": "s/a", "episode_id": 0}, {"task": "s/a", "episode_id": 1}, {"task": "s/b", "episode_id": 0}]
        self.assertEqual(RunSummary(job=job, root=None, rows=rows).unrun, (("s/b", 1),))

    def test_a_plain_job_still_reports_ids(self):
        job = JobSpec(name="j", task="s/a", episodes=(0, 3))
        self.assertEqual(RunSummary(job=job, root=None, rows=[{"episode_id": 1}]).unrun, (0, 2))


class TestRewardGate(unittest.TestCase):
    """A gate <= 0 is refused at build time, where it used to score 1.0 every step."""

    @staticmethod
    def _section(gate):
        return {"progress": [["depth", "plug"], [":gate", gate]]}

    def test_build_refuses_zero_and_negative_gates(self):
        for gate in (0, 0.0, -0.5):
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "must be > 0"):
                build_reward(self._section(gate), lambda form: Leaf(fn=lambda ctx: (True, 1.0, {})))

    def test_a_positive_gate_builds(self):
        clause = build_reward(self._section(0.02), lambda form: Leaf(fn=lambda ctx: (True, 1.0, {})))
        self.assertIsInstance(clause, Progress)

    def test_the_checker_reports_it(self):
        doc = SimpleNamespace(reward={"progress": [[":gate", 0]]})
        problems = _check_reward(doc, catalog=None)
        self.assertTrue(any("must be a number > 0" in str(p) for p in problems), problems)


class TestNonFinite(unittest.TestCase):
    def test_clip_reads_nan_as_zero(self):
        self.assertEqual(_clip(float("nan")), 0.0)
        self.assertEqual(_clip(0.4), 0.4)

    def test_finite_reward(self):
        self.assertEqual(finite_reward(float("nan")), 0.0)
        self.assertEqual(finite_reward(math.inf), 0.0)
        self.assertEqual(finite_reward(0.25), 0.25)

    def test_diverged_result(self):
        result = diverged_result()
        self.assertEqual((result.success, result.reward, result.metrics), (False, 0.0, {"diverged": True}))


class TestPeakAndIntegralMissingKey(unittest.TestCase):
    """A key the measure did not report raises, as the reward clause does."""

    def test_both_raise(self):
        leaf = Leaf(fn=lambda ctx: (True, 1.0, {"other": 2.0}))
        for node in (Peak(of=leaf, key="value", value=1.0), Integral(of=leaf, key="value", value=1.0)):
            node.reset(_ctx())
            with self.subTest(node=type(node).__name__), self.assertRaisesRegex(KeyError, "'value'"):
                node(_ctx(0.1))


class TestStagingSubSections(unittest.TestCase):
    """``_plain`` unwraps a single value; a one-clause sub-section is still one clause."""

    def test_one_clause(self):
        self.assertEqual(parse_staging({"episodes": [":count", 50]}).episodes, (("count", 50),))
        self.assertEqual(parse_staging({"episodes": [":sampler", "fixed"]}).episodes, (("sampler", "fixed"),))
        self.assertEqual(parse_staging({"arm-actuation": [":ke", 100]}).arm_actuation, (("ke", 100),))

    def test_two_clauses(self):
        staging = parse_staging({"episodes": [[":count", 50], [":seed", 3]]})
        self.assertEqual(staging.episodes, (("count", 50), ("seed", 3)))
        gains = parse_staging({"arm-actuation": [[":ke", 100], [":kd", 7]]}).arm_actuation
        self.assertEqual(gains, (("ke", 100), ("kd", 7)))

    def test_through_the_reader(self):
        """The same sections as written in a document."""
        from robosimstudio_bench.taskdoc.parse import _plain

        (form,) = parse_sexp("(:staging (:episodes (:count 50)) (:arm-actuation (:ke 100)))")
        staging = parse_staging(_plain(form[1:]))
        self.assertEqual((staging.episodes, staging.arm_actuation), ((("count", 50),), (("ke", 100),)))

    def test_an_explicit_false_survives_a_profile_that_says_true(self):
        document = parse_staging({"scene-resident": "false"})
        merged = document.merged_over(Staging(scene_resident=True))
        self.assertIs(merged.scene_resident, False)
        self.assertIs(Staging().merged_over(Staging(scene_resident=True)).scene_resident, True)


class TestParameterisedReference(unittest.TestCase):
    """``cable.point(0.63)`` is one reference token, in a document as in code."""

    def test_reader_keeps_it_whole(self):
        (form,) = parse_sexp("(>= (winding-turns cable.point(0.63) peg0.axis) 0.5)")
        self.assertEqual(form, [">=", ["winding-turns", "cable.point(0.63)", "peg0.axis"], 0.5])

    def test_it_parses_to_a_reference_with_its_argument(self):
        (form,) = parse_sexp("(>= (winding-turns cable.point(0.63) peg0.axis) 0.5)")
        condition = parse_condition(form)
        self.assertEqual(condition.args[0].base, "cable")
        self.assertEqual(condition.args[0].arg, 0.63)
        self.assertIsNone(condition.args[1].arg)

    def test_forms_are_untouched(self):
        """A paren after whitespace, or after a number, is still a form."""
        (form,) = parse_sexp("(a x.y (0.5) 0.5(1) b(2) (c d))")
        self.assertEqual(form, ["a", "x.y", [0.5], 0.5, [1], "b", [2], ["c", "d"]])


class TestQueryRefusals(unittest.TestCase):
    """Dispatch refuses with a plain ``LookupError``; the query API promises ``QueryRefused``."""

    def test_applicable_skips_an_unimplemented_atom(self):
        leaf = SimpleNamespace(name="threaded", args=(1, 2))
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[leaf]),
            mock.patch.object(query_module, "dispatch", side_effect=LookupError("no handler")),
        ):
            operand = SimpleNamespace(kind="", substrate="rigid")
            self.assertEqual(query_module.applicable("family", operand, operand), [])

    def test_query_converts_it(self):
        spec = SimpleNamespace(args=(), reads_scene=False)
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", return_value=spec),
            mock.patch.object(query_module, "dispatch", side_effect=LookupError("no handler")),
            mock.patch.object(query_module, "_layouts", return_value={}),
            self.assertRaises(query_module.QueryRefused),
        ):
            query_module.query(SimpleNamespace(), "threaded", doc=None)

    def test_an_unknown_atom_is_refused(self):
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", side_effect=LookupError("unknown atom")),
            self.assertRaises(query_module.QueryRefused),
        ):
            query_module.query(SimpleNamespace(), "nope", doc=None)


if __name__ == "__main__":
    unittest.main()
