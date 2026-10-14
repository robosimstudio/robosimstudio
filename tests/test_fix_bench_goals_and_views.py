# SPDX-License-Identifier: Apache-2.0
"""Goal readers over welded fixtures, published-state clause keys, a world's
array read, and a batched env refusing an undeclared pairing up front."""

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.scene.handles import EntityHandle, SceneHandles
from robosimstudio.specs import TaskSpec
from robosimstudio_bench.goals.bind import Operand, _published_reader, read
from robosimstudio_bench.goals.goal import about_key, reading
from robosimstudio_bench.taskdoc.ast import Ref
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"


class TestAWeldedFixtureReadsWhereItWasPut(unittest.TestCase):
    """A single-world static is shapes on body -1: its pose is its ``xform``."""

    XFORM = (0.1, -0.15, 0.75, 0.0, 0.0, 0.0, 1.0)

    def setUp(self):
        pad = EntityHandle(name="pad", category="static", body=-1, static=True, xform=self.XFORM)
        ctx = SimpleNamespace(
            handles=SceneHandles(statics={"pad": pad}),
            body_q=np.zeros((3, 7)),
            body_qd=np.zeros((3, 6)),
            entities={},
        )
        self.snapshot = Snapshot(ctx=ctx)
        self.operand = Operand(ref=Ref(base="pad"), asset="pad_120mm", substrate="rigid")

    def test_position(self):
        np.testing.assert_allclose(read(self.snapshot, self.operand, "position"), self.XFORM[:3])

    def test_pose(self):
        np.testing.assert_allclose(read(self.snapshot, self.operand, "pose"), self.XFORM)

    def test_velocity(self):
        np.testing.assert_allclose(read(self.snapshot, self.operand, "velocity"), np.zeros(6))


class TestAPublishedStateClauseKeysLikeAnAtom(unittest.TestCase):
    def _leaf(self, text: str):
        condition = parse_condition(parse_sexp(text)[0])
        operand = Operand(ref=condition.args[0], asset="tape_asset")
        record = SimpleNamespace(name="tape_asset", states=[], published=[SimpleNamespace(name="bond-fraction")])
        return _published_reader(condition, (operand,), None, lambda _name: record)

    def test_about_key_finds_the_verdict(self):
        leaf = self._leaf("(<= (bond-fraction tape) 0.05)")
        snapshot = Snapshot(ctx=SimpleNamespace(entities={"tape": {"bond_fraction": 0.02}}))
        held, _score, metrics = leaf.fn(snapshot)
        self.assertEqual(leaf.about, ("bond_fraction", ("tape",)))
        self.assertIs(metrics[about_key(*leaf.about)], held)
        self.assertTrue(held)

    def test_two_operands_report_apart(self):
        a, b = self._leaf("(<= (bond-fraction tape_a) 0.05)"), self._leaf("(<= (bond-fraction tape_b) 0.05)")
        snapshot = Snapshot(
            ctx=SimpleNamespace(entities={"tape_a": {"bond_fraction": 0.02}, "tape_b": {"bond_fraction": 0.5}})
        )
        metrics = {**a.fn(snapshot)[2], **b.fn(snapshot)[2]}
        self.assertTrue(metrics[about_key(*a.about)])
        self.assertFalse(metrics[about_key(*b.about)])

    def test_the_panel_still_reads_the_number(self):
        leaf = self._leaf("(<= (bond-fraction tape) 0.05)")
        leaf.label = "goal"
        _held, _score, metrics = leaf.fn(Snapshot(ctx=SimpleNamespace(entities={"tape": {"bond_fraction": 0.02}})))
        self.assertEqual(reading(metrics, leaf), "0.02")


class TestAWorldReadCopiesOnlyItsWorld(unittest.TestCase):
    def test_numpy_reads_the_slice_not_the_fleet(self):
        import warp as wp

        from robosimstudio_bench.envs.world_view import _ArraySlice

        whole = wp.array(np.arange(12, dtype=np.float32), device="cpu")

        class _NoWholeRead:
            """The fleet's array, refusing to be copied whole."""

            def __getitem__(self, key):
                return whole[key]

            def numpy(self):
                raise AssertionError("copied every world to read one")

        np.testing.assert_array_equal(_ArraySlice(_NoWholeRead(), 2, 3).numpy(), [6, 7, 8])


class TestABatchedEnvRefusesAnUndeclaredPairing(unittest.TestCase):
    def test_scene_override_on_a_task_without_bindings(self):
        from robosimstudio_bench.envs.fleet import FleetEnv

        spec = TaskSpec(name="t", scene="bench_a", embodiment="franka")
        with self.assertRaises(KeyError) as caught:
            FleetEnv(spec, num_worlds=2, scene="bench_b", device="cpu")
        self.assertIn("declares no binding", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
