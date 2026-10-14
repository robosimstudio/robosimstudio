"""Regressions: writers that share an array, a mass that is not one, an object a
later episode does not name, and raw capture reading the step's snapshot.

No GPU: warp arrays on the CPU and stand-ins for everything else.
"""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import warp as wp

from robosimstudio.core.tuning import BodyWriter, ShapeWriter

TIER = "fast"


class _Array:
    """A Warp array's two-method surface: ``numpy()`` and ``assign()``."""

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def numpy(self):
        return self.values

    def assign(self, values) -> None:
        self.values = np.array(values, copy=True)

    def __len__(self) -> int:
        return len(self.values)


class TestTwoWritersOverOneArray(unittest.TestCase):
    def test_neither_puts_back_the_others_rows(self) -> None:
        model = SimpleNamespace(shape_material_mu=_Array([0.5, 0.5, 0.5, 0.5]))
        a = ShapeWriter(model, [0, 1])
        b = ShapeWriter(model, [2, 3])
        a.read(), b.read()
        b("mu", 1.2)
        a("mu", 0.1)
        np.testing.assert_allclose(model.shape_material_mu.numpy(), [0.1, 0.1, 1.2, 1.2])

    def test_an_outside_write_survives_the_next_one(self) -> None:
        model = SimpleNamespace(shape_material_mu=_Array([0.5, 0.5, 0.5, 0.5]))
        a = ShapeWriter(model, [0, 1])
        a("mu", 0.2)
        model.shape_material_mu.values[3] = 0.9
        a("mu", 0.3)
        np.testing.assert_allclose(model.shape_material_mu.numpy(), [0.3, 0.3, 0.5, 0.9])


class TestTwoClothsOverOneMaterialArray(unittest.TestCase):
    def _target(self, name, tris):
        from robosimstudio.scene.tuning import ClothTarget

        empty = np.zeros(0, dtype=np.int32)
        return ClothTarget(name=name, particles=empty, tris=np.asarray(tris, dtype=np.int32), edges=empty, tets=empty)

    def test_neither_puts_back_the_others_rows(self) -> None:
        from robosimstudio.scene.tuning import LiveMaterial

        model = SimpleNamespace(tri_materials=_Array(np.full((4, 3), 100.0)))
        a = LiveMaterial(model, self._target("a", [0, 1]))
        b = LiveMaterial(model, self._target("b", [2, 3]))
        self.assertTrue(a.write_field("tri_ke", 5.0))
        self.assertTrue(b.write_field("tri_ke", 7.0))
        model.tri_materials.values[0, 1] = 9.0
        self.assertTrue(a.write_field("tri_kd", 3.0))
        np.testing.assert_allclose(model.tri_materials.numpy()[:, 0], [5.0, 5.0, 7.0, 7.0])
        np.testing.assert_allclose(model.tri_materials.numpy()[:, 1], [9.0, 100.0, 100.0, 100.0])
        np.testing.assert_allclose(model.tri_materials.numpy()[:, 2], [3.0, 3.0, 100.0, 100.0])


class TestAMassThatIsNotOne(unittest.TestCase):
    def test_zero_and_negative_are_refused(self) -> None:
        model = SimpleNamespace(
            body_mass=_Array([2.0]),
            body_inv_mass=_Array([0.5]),
            body_inertia=_Array([np.eye(3) * 2.0]),
            body_inv_inertia=_Array([np.eye(3) * 0.5]),
        )
        writer = BodyWriter(model, [0])
        for mass in (0.0, -1.0, float("nan")):
            self.assertFalse(writer("mass", mass))
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 2.0)
        self.assertAlmostEqual(float(model.body_inv_mass.numpy()[0]), 0.5)
        np.testing.assert_allclose(model.body_inertia.numpy()[0], np.eye(3) * 2.0)
        self.assertTrue(writer("mass", 4.0))
        self.assertAlmostEqual(float(model.body_inv_mass.numpy()[0]), 0.25)


class TestAnEpisodeThatDoesNotNameTheObject(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio.domains.rigid.entity import EntityRigid
        from robosimstudio.specs import ObjectSpec

        wp.init()
        self.entity = EntityRigid(ObjectSpec(name="a", pose=(0.1, 0.2, 0.03)))
        self.entity.coord_start = 2
        self.pipeline = SimpleNamespace(
            state_0=SimpleNamespace(joint_q=wp.zeros(9, dtype=wp.float32, device="cpu")),
            state_1=SimpleNamespace(joint_q=wp.zeros(9, dtype=wp.float32, device="cpu")),
        )

    def _start(self, episode) -> np.ndarray:
        self.entity.reset(self.pipeline, episode)
        return self.pipeline.state_0.joint_q.numpy()[2:5].copy()

    def test_it_returns_to_its_declared_pose(self) -> None:
        expected = self._start(None)
        moved = self._start({"object_poses": {"a": (0.5, 0.5, 0.03, 0.0)}})
        self.assertFalse(np.allclose(moved, expected))
        np.testing.assert_allclose(self._start({"object_poses": {}}), expected, atol=1e-6)
        np.testing.assert_allclose(self.entity.declared_q[:3], (0.1, 0.2, 0.03))

    def test_only_its_own_coordinates_are_written(self) -> None:
        states = (_SliceOnly(9), _SliceOnly(9))
        pipeline = SimpleNamespace(
            state_0=SimpleNamespace(joint_q=states[0]), state_1=SimpleNamespace(joint_q=states[1])
        )
        self.entity.reset(pipeline, {"object_poses": {"a": (0.5, 0.5, 0.03, 0.0)}})
        self.entity.reset(self.pipeline, {"object_poses": {"a": (0.5, 0.5, 0.03, 0.0)}})
        expected = self.pipeline.state_0.joint_q.numpy()[2:9]
        for joint_q in states:
            self.assertEqual(len(joint_q.assigns), 1)
            key, values = joint_q.assigns[0]
            self.assertEqual(key, slice(2, 9))
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.shape, (7,))
            np.testing.assert_allclose(values, expected, atol=1e-6)


class _SliceOnly:
    """A ``joint_q`` that refuses a whole-vector read and records slice assigns."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.assigns: list[tuple[slice, np.ndarray]] = []

    def numpy(self):
        raise AssertionError("copied the whole joint_q to the host")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key):
        owner = self

        class _View:
            def assign(self, values) -> None:
                owner.assigns.append((key, np.array(values, copy=True)))

        return _View()


class _Unreadable:
    def numpy(self):
        raise AssertionError("read the device array instead of the step's snapshot")


class TestRawCaptureReadsTheSnapshot(unittest.TestCase):
    def test_joint_and_particle_arrays_come_from_the_snapshot(self) -> None:
        from robosimstudio.record import EpisodeRecorder
        from robosimstudio.specs import CaptureSpec, RecordSpec

        snap = SimpleNamespace(
            joint_q=np.array([1.0, 2.0]),
            joint_qd=np.array([3.0]),
            particle_q=np.ones((2, 3)),
            particle_qd=np.zeros((2, 3)),
        )
        context = SimpleNamespace(
            body_q=np.zeros((1, 7)),
            body_qd=np.zeros((1, 6)),
            settled_body_q=np.zeros((1, 7)),
            joint_target_q=np.zeros(2),
            contacts=None,
        )
        unreadable = _Unreadable()
        env = SimpleNamespace(
            last_context=context,
            pipeline=SimpleNamespace(snapshot=snap),
            state=SimpleNamespace(
                joint_q=unreadable, joint_qd=unreadable, particle_q=unreadable, particle_qd=unreadable
            ),
            model=SimpleNamespace(particle_count=2),
            step_count=3,
            sim_time=0.05,
        )
        with TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory, RecordSpec(record_video=False), capture=CaptureSpec(enabled=True, contacts=False)
            )
            recorder._capture = {}
            recorder._record_capture(env)
        np.testing.assert_allclose(recorder._capture["joint_q"][0], [1.0, 2.0])
        self.assertEqual(recorder._capture["joint_q"][0].dtype, np.float32)
        np.testing.assert_allclose(recorder._capture["particle_q"][0], np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
