# SPDX-License-Identifier: Apache-2.0
"""Wave scheduling and fleet bookkeeping: the batched contracts, without a GPU.

Everything here is arithmetic on episodes and rows. The parts that need a device
-- replication, batched IK, a stepping fleet -- are exercised by the smoke path;
what this file guards is that a batched run's *rows* are the same claim a serial
run's are, and that the rates derived from them survive episodes overlapping.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robosimstudio.core.env import CLOSE_PRECEDENCE, beats
from robosimstudio.core.worlds import WorldMap
from robosimstudio.scene import compiler
from robosimstudio.scene.compiler import CannotReplicate
from robosimstudio.specs import EpisodeSpec, JobSpec, TaskSpec
from robosimstudio_bench.goals.context import EvalResult
from robosimstudio_bench.run.fleet import (
    FleetSummary,
    WorldSlot,
    plan_waves,
    score_world,
    wave_record,
    waves_for,
)

TIER = "fast"
"""No GPU. Gates every commit."""


def _task(**kwargs: Any) -> TaskSpec:
    return TaskSpec(name="t", **kwargs)


class TestWorldMap(unittest.TestCase):
    """The stride arithmetic every batched read and write goes through."""

    MAP = WorldMap(
        num_worlds=4,
        bodies_per_world=3,
        shapes_per_world=5,
        joints_per_world=2,
        coords_per_world=7,
        dofs_per_world=6,
        particles_per_world=0,
    )

    def _model(self, num_worlds: int = 4, **overrides: int) -> Any:
        counts = {
            "body_count": 3 * num_worlds,
            "shape_count": 5 * num_worlds,
            "joint_count": 2 * num_worlds,
            "joint_coord_count": 7 * num_worlds,
            "joint_dof_count": 6 * num_worlds,
            "particle_count": 0,
            "world_count": num_worlds,
        }
        return SimpleNamespace(**{**counts, **overrides})

    def test_indices_offset_by_world(self):
        self.assertEqual(self.MAP.body(2, 0), 2)
        self.assertEqual(self.MAP.body(2, 3), 11)
        self.assertEqual(self.MAP.coord(6, 2), 20)
        self.assertEqual(self.MAP.dof(0, 1), 6)

    def test_a_world_view_indexes_like_the_template(self):
        """The claim the whole design rests on: an index resolved once against
        the single-world model is right for every world's slice."""
        body_q = np.arange(4 * 3 * 7, dtype=np.float32).reshape(4 * 3, 7)
        for world in range(4):
            view = self.MAP.body_view(body_q, world)
            self.assertEqual(view.shape, (3, 7))
            # Template body 2 is this world's body 2, wherever the world sits.
            np.testing.assert_array_equal(view[2], body_q[self.MAP.body(2, world)])

    def test_validate_accepts_a_uniform_replication(self):
        self.MAP.validate(self._model())

    def test_validate_rejects_a_count_that_is_not_a_multiple(self):
        """One extra body -- a ground plane replicated where it should not be,
        an entity built into the fleet builder -- makes every stride wrong."""
        with self.assertRaises(ValueError) as caught:
            self.MAP.validate(self._model(body_count=13))
        self.assertIn("body_count", str(caught.exception))

    def test_validate_rejects_a_world_count_disagreement(self):
        with self.assertRaises(ValueError):
            self.MAP.validate(self._model(world_count=3))

    def test_tile_coords_builds_the_flat_model_vector(self):
        home = np.arange(7, dtype=np.float32)
        tiled = self.MAP.tile_coords(home)
        self.assertEqual(tiled.shape, (28,))
        for world in range(4):
            np.testing.assert_array_equal(tiled[self.MAP.coords(world)], home)

    def test_tile_coords_rejects_the_wrong_width(self):
        with self.assertRaises(ValueError):
            self.MAP.tile_coords(np.zeros(6, dtype=np.float32))

    def test_stack_coords_writes_each_world_its_own_row(self):
        """A wave reset: W different start postures, one flat vector, one assign."""
        rows = [np.full(7, world, dtype=np.float32) for world in range(4)]
        flat = self.MAP.stack_coords(rows)
        self.assertEqual(flat.shape, (28,))
        for world in range(4):
            np.testing.assert_array_equal(flat[self.MAP.coords(world)], rows[world])

    def test_stack_coords_rejects_a_short_wave(self):
        with self.assertRaises(ValueError):
            self.MAP.stack_coords([np.zeros(7, dtype=np.float32)] * 3)


class TestPlanWaves(unittest.TestCase):
    """Episodes to waves: in order, at most W wide, never mixing models."""

    def test_even_and_uneven_splits(self):
        task = _task(episodes=EpisodeSpec(count=16))
        self.assertEqual([len(w) for w in plan_waves(task, list(range(10)), 4)], [4, 4, 2])
        self.assertEqual([len(w) for w in plan_waves(task, list(range(8)), 4)], [4, 4])
        self.assertEqual([len(w) for w in plan_waves(task, list(range(3)), 8)], [3])
        self.assertEqual(plan_waves(task, [], 4), [])

    def test_every_episode_lands_exactly_once_in_order(self):
        """The flattened waves are the requested ids, unchanged.

        This is what makes a fleet run's ``episode_results.jsonl`` sort the way a
        serial run's does, which is the whole basis for comparing two runs.
        """
        task = _task(episodes=EpisodeSpec(count=32))
        ids = [3, 1, 4, 1, 5, 9, 2, 6]
        flat = [e.episode_id for wave in plan_waves(task, ids, 3) for e in wave]
        self.assertEqual(flat, ids)

    def test_waves_never_mix_model_keys(self):
        """A sampler that changes the model splits the waves, rather than one
        wave silently running in whichever model it built first."""

        def sampler(task, episode_id, rng):
            del task, rng
            return {"layout": "a" if episode_id % 2 == 0 else "b"}

        task = _task(episodes=EpisodeSpec(count=8, sampler=sampler, model_key_fields=("layout",)))
        waves = plan_waves(task, list(range(8)), 3)
        for wave in waves:
            self.assertEqual(len({e.model_key for e in wave}), 1, "a wave mixed two models")
        self.assertEqual(sorted(e.episode_id for wave in waves for e in wave), list(range(8)))

    def test_shipped_style_task_is_one_group(self):
        """No shipped suite declares model_key_fields, so grouping is a no-op:
        the waves are the ids chunked in order."""
        task = _task(episodes=EpisodeSpec(count=8))
        waves = plan_waves(task, list(range(8)), 3)
        self.assertEqual([[e.episode_id for e in w] for w in waves], [[0, 1, 2], [3, 4, 5], [6, 7]])

    def test_zero_worlds_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_waves(_task(), [0], 0)

    def test_waves_for_matches_the_planner(self):
        task = _task(episodes=EpisodeSpec(count=32))
        for count, width in ((10, 4), (8, 4), (1, 8), (17, 16)):
            self.assertEqual(waves_for(count, width), len(plan_waves(task, list(range(count)), width)))


class TestWorldSlot(unittest.TestCase):
    """One world's outcome, in the row shape the serial runner writes."""

    def _slot(self) -> WorldSlot:
        task = _task(episodes=EpisodeSpec(count=4))
        (wave,) = plan_waves(task, [2], 4)
        return WorldSlot(world=5, episode=wave[0])

    def test_row_carries_the_serial_keys_plus_world(self):
        """The runner's own row keys, so a fleet run reads like any other run."""
        slot = self._slot()
        slot.finish(
            success=True, reward=1.5, close_kind="success", steps=42, wall_s=3.14159, info={"metrics": {"depth": 2.0}}
        )
        row = slot.row("suite/variant")
        serial_keys = {
            "task",
            "episode_id",
            "seed",
            "success",
            "reward",
            "episode_length",
            "close_kind",
            "termination_reason",
            "metrics",
            "wall_s",
        }
        self.assertTrue(serial_keys <= set(row), f"row is missing serial keys: {serial_keys - set(row)}")
        self.assertEqual(set(row) - serial_keys, {"world"})
        self.assertEqual(row["episode_id"], 2)
        self.assertEqual(row["episode_length"], 42)
        self.assertEqual(row["metrics"], {"depth": 2.0})
        self.assertEqual(row["wall_s"], 3.142)
        self.assertEqual(row["termination_reason"], "")

    def test_row_says_why_the_policy_asked_out(self):
        """``termination_reason``, as the serial row carries it."""
        slot = self._slot()
        info = {"close_kind": "policy_request", "termination_reason": "graph finished"}
        slot.finish(success=False, reward=0.0, close_kind="policy_request", steps=3, wall_s=1.0, info=info)
        self.assertEqual(slot.row("suite/variant")["termination_reason"], "graph finished")

    def test_finishing_clears_live(self):
        slot = self._slot()
        self.assertTrue(slot.live)
        slot.finish(success=False, reward=0.0, close_kind="truncated", steps=9, wall_s=1.0, info={})
        self.assertFalse(slot.live)

    def test_a_parked_world_has_no_row(self):
        """The tail of an uneven wave produces no results row at all."""
        slot = WorldSlot(world=7)
        self.assertTrue(slot.parked)
        with self.assertRaises(ValueError):
            slot.row("suite/variant")

    def test_row_extras_land_at_the_top_level(self):
        slot = self._slot()
        slot.extras = {"graph_nodes": 12}
        slot.finish(success=True, reward=1.0, close_kind="success", steps=1, wall_s=0.5, info={})
        self.assertEqual(slot.row("t")["graph_nodes"], 12)


class TestFleetSummary(unittest.TestCase):
    """The rates overlap breaks, fixed where they are computed."""

    def _summary(self, rows: list[dict], waves: list[dict], num_worlds: int) -> FleetSummary:
        job = JobSpec(name="j", task="t", episodes=(0, len(rows)))
        return FleetSummary(job=job, root=Path("/tmp/x"), rows=rows, num_worlds=num_worlds, waves=waves, workers=1)

    def test_stepping_rate_uses_wave_wall_not_summed_episode_wall(self):
        """Four worlds each running 100 steps over the same 10 seconds deliver
        40 steps/s, not 4 x that -- which is what summing per-row wall gives."""
        rows = [{"episode_id": i, "episode_length": 100, "wall_s": 10.0, "success": True} for i in range(4)]
        waves = [{"wave": 0, "steps": 100, "wall_s": 10.0, "world_steps": 400}]
        summary = self._summary(rows, waves, num_worlds=4)
        self.assertEqual(summary.steps, 400)
        self.assertAlmostEqual(summary.steps_per_s_stepping, 40.0)

    def test_utilization_reports_straggler_waste(self):
        """A wave held open by one world reports the fraction actually worked."""
        waves = [{"wave": 0, "steps": 100, "wall_s": 10.0, "world_steps": 130}]
        summary = self._summary([], waves, num_worlds=4)
        self.assertAlmostEqual(summary.utilization, 130 / 400)

    def test_empty_run_reports_zero_rather_than_dividing(self):
        summary = self._summary([], [], num_worlds=8)
        self.assertEqual(summary.steps_per_s_stepping, 0.0)
        self.assertEqual(summary.utilization, 0.0)

    def test_inherited_success_reporting_still_works(self):
        """A fleet summary is still a run summary: the rate and interval that
        every run is read by come from the rows, unchanged."""
        rows = [{"episode_id": i, "episode_length": 10, "wall_s": 1.0, "success": i < 3} for i in range(4)]
        summary = self._summary(rows, [{"wave": 0, "steps": 10, "wall_s": 1.0, "world_steps": 40}], num_worlds=4)
        self.assertEqual(summary.successes, 3)
        self.assertAlmostEqual(summary.success_rate, 0.75)
        lo, hi = summary.interval
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)


class TestScoreWorld(unittest.TestCase):
    """Per-world termination, in the precedence the single-world path uses."""

    def _fixture(self, *, success=False, diverged=False, reward=None):

        slot = WorldSlot(world=1, episode=plan_waves(_task(), [0], 2)[0][0])
        slot.settled_body_q = np.zeros((1, 7))
        if reward is None:
            reward = 2.0 if success else 0.0
        result = EvalResult(success=success, reward=reward, metrics={"depth": 1.0})
        self.evaluated = 0

        def evaluate(*a, **k):
            self.evaluated += 1
            return result

        fleet = SimpleNamespace(evaluate=evaluate)
        snapshot = SimpleNamespace(diverged=np.array([False, diverged]))
        return slot, fleet, snapshot

    def test_success_terminates_and_is_the_evaluators_call(self):

        slot, fleet, snapshot = self._fixture(success=True)
        done, info = score_world(slot, fleet, snapshot, step=5, limit=100)
        self.assertTrue(done)
        self.assertEqual(info["close_kind"], "success")
        self.assertTrue(info["success"])
        self.assertEqual(info["metrics"], {"depth": 1.0})

    def test_divergence_beats_everything(self):
        """A world producing NaN has lost its episode however it was scored."""

        slot, fleet, snapshot = self._fixture(success=True, diverged=True)
        done, info = score_world(slot, fleet, snapshot, step=5, limit=100, request="graph finished")
        self.assertTrue(done)
        self.assertEqual(info["close_kind"], "diverged")

    def test_a_diverged_world_scores_as_the_serial_env_does(self):
        """No success, no reward, and the evaluator never consulted on NaN state --
        ``Env.evaluate``'s verdict, not whatever the evaluator made of garbage."""

        slot, fleet, snapshot = self._fixture(success=True, diverged=True)
        _, info = score_world(slot, fleet, snapshot, step=5, limit=100)
        self.assertFalse(info["success"])
        self.assertEqual(info["reward"], 0.0)
        self.assertEqual(info["metrics"], {"diverged": True})
        self.assertEqual(self.evaluated, 0)
        self.assertIs(slot.info, info)

    def test_a_nan_reward_reaches_the_row_as_zero(self):
        slot, fleet, snapshot = self._fixture(reward=float("nan"))
        _, info = score_world(slot, fleet, snapshot, step=5, limit=100)
        self.assertEqual(info["reward"], 0.0)

    def test_a_policy_request_ends_the_world_but_does_not_win_it(self):

        slot, fleet, snapshot = self._fixture(success=False)
        done, info = score_world(slot, fleet, snapshot, step=5, limit=100, request="graph finished")
        self.assertTrue(done)
        self.assertEqual(info["close_kind"], "policy_request")
        self.assertFalse(info["success"])
        self.assertEqual(info["termination_reason"], "graph finished")

    def test_the_budget_truncates_last(self):

        slot, fleet, snapshot = self._fixture()
        self.assertEqual(score_world(slot, fleet, snapshot, step=99, limit=100)[0], False)
        done, info = score_world(slot, fleet, snapshot, step=100, limit=100)
        self.assertTrue(done)
        self.assertEqual(info["close_kind"], "truncated")


class TestWaveRecord(unittest.TestCase):
    """What a wave reports about itself in the run's provenance."""

    def test_histogram_counts_only_assigned_worlds(self):
        task = _task(episodes=EpisodeSpec(count=8))
        (wave,) = plan_waves(task, [0, 1, 2], 4)
        slots = [WorldSlot(world=i, episode=wave[i] if i < 3 else None) for i in range(4)]
        for slot, kind in zip(slots[:3], ("success", "success", "truncated"), strict=True):
            slot.finish(success=kind == "success", reward=0.0, close_kind=kind, steps=50, wall_s=1.0, info={})
        record = wave_record(index=0, slots=slots, steps=80, wall_s=12.0, world_steps=180)
        self.assertEqual(record["episodes"], 3)
        self.assertEqual(record["done_histogram"], {"success": 2, "truncated": 1})
        self.assertAlmostEqual(record["utilization"], 180 / 320)


class TestCannotReplicate(unittest.TestCase):
    """The refusal a page answers by narrowing rather than by refusing.

    Most scenes cannot be built at more than one world yet. On an interactive
    page that is not a failed load -- the selection is a *task*, and the world
    count is a preference on it -- so the studio catches this one type and
    rebuilds at one world. Which only works if it stays distinguishable from
    every other way a compile can fail, and stays catchable by the handlers
    written before it had a name.
    """

    def test_it_is_a_kind_of_not_implemented_error(self):
        self.assertTrue(issubclass(CannotReplicate, NotImplementedError))
        with self.assertRaises(NotImplementedError):
            raise CannotReplicate("towel")

    def test_it_does_not_catch_unrelated_failures(self):
        # An inadmissible pairing or a missing asset is not fixed by narrowing
        # the grid, so those must not be answered the way this one is.
        for other in (ValueError("inadmissible pairing"), KeyError("no such asset"), RuntimeError("boom")):
            self.assertNotIsInstance(other, CannotReplicate)

    def test_it_is_exported(self):
        self.assertIn("CannotReplicate", compiler.__all__)


class TestTheEmptyContextIsCoveredByTheRefusal(unittest.TestCase):
    """`FleetEnv.context` fills no `particles`, and only per-world `entities`, safely.

    Safely only because `fleet.scoring_refusal` declines every scene that would
    read anything else; an entity is handed over only when it declares
    `world_aware` (the gear board: per-world latch and spin windows, and
    a `published_states` that reads `WorldState.world`). Lifting the rest with
    `entities` short would take every published state with it: `bind._published`
    reads that dict, so a peel would raise `BindError` on every episode.
    Structural on purpose: reading the source keeps this in the fast tier.
    """

    def _refusal_source(self) -> str:
        import inspect

        from robosimstudio_bench.envs import fleet

        return inspect.getsource(fleet.scoring_refusal)

    def test_the_refusal_still_declines_scenes_with_deformables(self):
        """It asks what a thing *is*, not which view held it.

        The clause read ``scene.deformables`` -- a bin -- so a publisher that
        owned a handle sat in ``objects`` and slipped past it. Both halves ask
        the entity now, which is what this pins."""
        source = self._refusal_source()
        self.assertIn("carries_particles", source)
        self.assertIn("published_states", source)
        # And the one way through: a publisher that answers per world says so.
        self.assertIn("world_aware", source)

    def test_the_compilers_refusal_is_about_the_physics_alone(self):
        """The two refusals answer different questions and must stay apart.

        A deformable's physics replicates -- four worlds to within 2e-10 -- so
        the compiler has nothing to say about it; what cannot be replicated is a
        substep kernel with world-0 indices baked into it. Scoring is this
        class's problem, and saying so twice would leave the fleet's own gap
        depending on the compiler remembering it.
        """
        import inspect

        source = inspect.getsource(compiler._replication_refusal)
        self.assertIn("contribute_hooks", source)
        self.assertNotIn("base.deformables", source)

    def test_the_context_says_what_must_land_with_a_per_world_probe(self):
        """If someone fills `particles` here, this points at `entities` too."""
        import inspect

        from robosimstudio_bench.envs.fleet import FleetEnv

        doc = inspect.getdoc(FleetEnv.context) or ""
        self.assertIn("entities", doc)
        self.assertIn("scoring_refusal", doc)


class TestTheWorldViewAnswersWhatTheToolsAsk(unittest.TestCase):
    """``WorldView`` is a duck, and a duck that falls behind takes a wave down.

    Every ``sim.*`` and suite tool is built with one ``env`` and reads a surface
    off it. On a single world that is an ``Env``; on a wave it is a
    :class:`~robosimstudio_bench.envs.world_view.WorldView` presenting
    one world. Nothing declares the surface, so ``Env`` can grow a property and
    the view not notice -- and it did, twice, in one file:
    ``sim.next_goal`` reached ``env.evaluator`` and ``sim.clearance`` reached
    ``env.clearance``, neither of which a view had. A missing name here is not a
    degraded answer, it is ``AttributeError`` inside a node, which the runtime
    turns into a failed subgraph -- **every world, on its first goal**, with the
    wave still dutifully reporting 0/N.

    So the surface is derived from the source rather than listed by hand: what
    the graph package reads off ``self.env`` is what a view has to answer.
    """

    DRIVER_ONLY = frozenset({"control", "frames", "last_result", "request_termination", "reset", "step"})
    """Names a *wave* never asks a view, because the fleet replaces their asker.

    Driving one env -- resetting it, stepping it, writing its control, its
    frame buffer, its last verdict -- is what ``SolutionGraphBatched`` and the
    fleet facade do batched, in their own code. Held as an exemption rather
    than dropped so the next name to appear here has to be argued for, and
    checked against where it is actually read.
    """

    DRIVER_FILES = frozenset({"facade.py", "graph_solution.py"})
    """The single-world driver: the two modules a fleet run replaces wholesale."""

    def _reads(self) -> dict[str, set[str]]:
        """Every ``self.env.<name>`` the graph package writes, by file."""
        import ast
        from pathlib import Path

        import robosimstudio_policy_agent

        root = Path(robosimstudio_policy_agent.__file__).parent
        found: dict[str, set[str]] = {}
        for path in sorted(root.rglob("*.py")):
            if "nodes" in path.relative_to(root).parts:
                continue  # node scripts read through ``ctx``, never ``self.env``
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Attribute):
                    continue
                inner = node.value
                if inner.attr != "env" or not isinstance(inner.value, ast.Name) or inner.value.id != "self":
                    continue
                found.setdefault(node.attr, set()).add(path.name)
        return found

    def test_the_scan_finds_the_surface_at_all(self) -> None:
        """A scan that matched nothing would make the test below vacuous."""
        reads = self._reads()
        self.assertGreater(len(reads), 10, "the graph tools read far more off env than this")
        self.assertIn("handles", reads)

    def test_every_name_a_tool_reads_is_one_a_world_can_answer(self) -> None:
        from robosimstudio_bench.envs.world_view import WorldView

        missing = {
            name: sorted(files)
            for name, files in sorted(self._reads().items())
            if name not in self.DRIVER_ONLY and not hasattr(WorldView, name)
        }
        self.assertEqual(missing, {}, f"a tool reads off env what a world view cannot answer: {missing}")

    def test_the_exempt_names_are_only_read_by_the_driver(self) -> None:
        """The exemption is "the fleet replaces that code", so it has to be true.

        A ``sim.*`` tool growing a read of ``env.step`` would be exempted by the
        list above and still break every wave; this is what refuses that.
        """
        reads = self._reads()
        for name in sorted(self.DRIVER_ONLY):
            with self.subTest(name=name):
                elsewhere = sorted(reads.get(name, set()) - self.DRIVER_FILES)
                self.assertEqual(elsewhere, [], f"{name!r} is exempt as driver-only but is read in {elsewhere}")


if __name__ == "__main__":
    unittest.main()


class TestOneEpisodeGetsOneLabel(unittest.TestCase):
    """Why an episode ended may not depend on how many worlds it ran in.

    It did. On the step where the goal is met *and* the step budget runs out,
    the single-world path recorded ``"truncated"`` -- its guard was
    ``"close_kind" not in info``, and the base class had already written that --
    while ``score_world`` checks success before the budget and recorded
    ``"success"``. Same episode, two labels, decided by ``--num-worlds``.
    """

    def test_success_beats_truncation_and_loses_to_the_rest(self) -> None:
        """The rule, stated once and checked directly.

        ``diverged`` and ``policy_request`` still win, because neither is a
        claim about the goal: one says the numbers stopped meaning anything,
        the other says nobody was still trying."""
        self.assertTrue(beats("success", None))
        self.assertTrue(beats("success", "truncated"))
        self.assertFalse(beats("success", "policy_request"))
        self.assertFalse(beats("success", "diverged"))
        self.assertFalse(beats("truncated", "success"))

    def test_the_batched_path_closes_in_that_same_order(self) -> None:
        """Read off ``score_world``'s source, because the order *is* the
        if-chain -- there is no value to assert, only a sequence of branches,
        and a chain reordered by a well-meaning edit is exactly the regression
        this pins."""
        import inspect
        import re

        from robosimstudio_bench.run import fleet as fleet_module

        written = re.findall(r'info\["close_kind"\] = "([a-z_]+)"', inspect.getsource(fleet_module.score_world))
        self.assertEqual(written, list(CLOSE_PRECEDENCE))
