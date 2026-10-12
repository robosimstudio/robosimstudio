# SPDX-License-Identifier: Apache-2.0
"""Writing into a built model, and telling the solver you did.

The material studio could tune a cloth and nothing else, and the reason was not
that a rigid body has nothing to tune -- it is that the mechanism happened to be
written for cloth. These tests cover the generalisation, and they are shaped
around the two ways it can be wrong rather than around the code.

**A write that does not land.** VBD reads Newton's cloth arrays directly, so a
write *is* the change. MuJoCo converts Newton's model into an ``mjm`` at
construction and reads the copy, so a write nothing tells it about lands in an
array it stopped consulting -- a slider that moves, a number that changes, and a
simulation behaving exactly as before. `notify_model_changed` is the fix and the
flag has to be the right one, so the flags are asserted against newton's own
enum rather than trusted as literals.

**A write that lands wrong.** Mass is the case: writing `body_mass` alone leaves
a body that weighs a tenth of what it did and resists rotation exactly as much,
which tumbles like nothing real. Scaling the tensor with it is the claim, and a
welded body staying welded is the other half.

No GPU: the writers take anything with the right attributes, and a fake model of
numpy arrays exercises every path a real one does. What a real model would add is
whether Warp accepts the assignment, which is warp's contract and not this
module's.
"""

from __future__ import annotations

import unittest

import numpy as np
from newton import ModelFlags

from robosimstudio.core.tuning import ArrayWriter, BodyWriter, JointWriter, ShapeWriter, notify
from robosimstudio.specs import BodySpec, ContactMaterialSpec, JointDriveSpec
from robosimstudio.specs.params import RECAPTURE, iter_params, undescribed

TIER = "fast"
"""No GPU, no scene. Gates every commit."""


class _Array:
    """A Warp array's two-method surface: ``numpy()`` and ``assign()``."""

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.assigns = 0

    def numpy(self):
        return self.values

    def assign(self, values) -> None:
        self.values = np.array(values, copy=True)
        self.assigns += 1

    def __len__(self) -> int:
        return len(self.values)


class _Model:
    """A model made of the arrays a writer touches, and nothing else."""

    def __init__(self, **arrays) -> None:
        for name, values in arrays.items():
            setattr(self, name, _Array(values))


class _Solver:
    def __init__(self) -> None:
        self.told: list[int] = []

    def notify_model_changed(self, flags: int) -> None:
        self.told.append(int(flags))


class _Program:
    """What a pipeline's composed physics answers to, here: fan out and count.

    One island or five, the notification goes to every solver -- which is the
    property this stands in for, and the reason a program rather than two named
    attributes: an arrangement with a rod, a sheet and an arm has three, and a
    pipeline that named them would have to grow a field per island.
    """

    def __init__(self, *names: str) -> None:
        self.islands = {name: _Solver() for name in (names or ("mjc", "vbd"))}

    def notify(self, flags: int) -> None:
        for solver in self.islands.values():
            solver.notify_model_changed(int(flags))

    def solvers(self) -> dict[str, _Solver]:
        return dict(self.islands)


class _Pipeline:
    def __init__(self, *, captured: bool = False) -> None:
        self.program = _Program()
        self.captured = captured
        self.invalidated = 0

    def invalidate_graph(self) -> None:
        self.invalidated += 1

    def told(self, island: str = "mjc") -> list[int]:
        return self.program.islands[island].told


def _shape_model(count: int = 3) -> _Model:
    return _Model(**{attr: np.zeros(count) for attr in ShapeWriter.FIELDS.values()})


class TestTheFlagsAreNewtonsOwn(unittest.TestCase):
    """Spelled as bits in the module so it costs no newton import to define, and
    pinned here so the bits cannot quietly stop meaning what they say.

    Getting one wrong is the quietest possible failure: the write still happens,
    the solver is still notified, and it refreshes a category that did not
    change while the one that did goes unread."""

    def test_a_shape_writer_declares_shape_properties(self) -> None:
        self.assertEqual(ShapeWriter.flags, int(ModelFlags.SHAPE_PROPERTIES))

    def test_a_body_writer_declares_body_inertial_properties(self) -> None:
        self.assertEqual(BodyWriter.flags, int(ModelFlags.BODY_INERTIAL_PROPERTIES))

    def test_a_joint_writer_declares_joint_dof_properties(self) -> None:
        self.assertEqual(JointWriter.flags, int(ModelFlags.JOINT_DOF_PROPERTIES))

    def test_the_arrays_each_flag_names_are_the_ones_written(self) -> None:
        """Newton documents the array list per flag. These are that list."""
        for attr in ("shape_material_mu", "shape_material_ke", "shape_material_kd"):
            self.assertIn(attr, ShapeWriter.FIELDS.values())
        for attr in ("joint_target_ke", "joint_target_kd", "joint_limit_lower", "joint_limit_upper"):
            self.assertIn(attr, JointWriter.FIELDS.values())


class TestNotifyingTheSolvers(unittest.TestCase):
    def test_every_solver_is_told(self) -> None:
        """A coupled arrangement runs several solvers over one model, and a
        shape whose friction moved is one any of them may be resolving. The
        program fans it out, so this layer does not carry a list of islands."""
        pipeline = _Pipeline()
        notify(pipeline, ShapeWriter.flags)
        for island in pipeline.program.islands:
            self.assertEqual(pipeline.told(island), [ShapeWriter.flags], island)

    def test_a_captured_graph_is_dropped(self) -> None:
        """The notification is host work a replay does not perform."""
        pipeline = _Pipeline(captured=True)
        notify(pipeline, ShapeWriter.flags)
        self.assertEqual(pipeline.invalidated, 1)

    def test_an_eager_run_keeps_its_graph_because_it_has_none(self) -> None:
        pipeline = _Pipeline(captured=False)
        notify(pipeline, ShapeWriter.flags)
        self.assertEqual(pipeline.invalidated, 0)

    def test_no_pipeline_is_a_no_op_rather_than_a_crash(self) -> None:
        """So a writer can be exercised against a bare model."""
        notify(None, ShapeWriter.flags)

    def test_nothing_written_notifies_nobody(self) -> None:
        pipeline = _Pipeline()
        writer = ShapeWriter(_shape_model(), [], pipeline)
        self.assertFalse(writer("mu", 0.5))
        self.assertEqual(pipeline.told(), [])


class TestWritingAContactMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _shape_model(4)
        self.pipeline = _Pipeline()
        self.writer = ShapeWriter(self.model, [1, 2], self.pipeline)

    def test_it_writes_only_this_entitys_shapes(self) -> None:
        self.assertTrue(self.writer("mu", 2.0))
        np.testing.assert_array_equal(self.model.shape_material_mu.numpy(), [0.0, 2.0, 2.0, 0.0])

    def test_it_notifies_after_a_write(self) -> None:
        self.writer("mu", 2.0)
        self.assertEqual(self.pipeline.told(), [int(ModelFlags.SHAPE_PROPERTIES)])

    def test_a_field_it_does_not_own_is_refused_rather_than_guessed(self) -> None:
        self.assertFalse(self.writer("tri_ke", 1.0))
        self.assertEqual(self.pipeline.told(), [])

    def test_it_reads_what_the_model_holds(self) -> None:
        """Read back, never remembered. After an import or a collider swap the
        model is the only thing that knows what it holds."""
        self.model.shape_material_mu.values[1] = 1.7
        self.assertAlmostEqual(ShapeWriter(self.model, [1, 2]).read()["mu"], 1.7)

    def test_the_read_fills_a_spec(self) -> None:
        spec = ContactMaterialSpec(**self.writer.read())
        self.assertEqual(spec.mu, 0.0)

    def test_one_upload_per_write_rather_than_a_round_trip(self) -> None:
        """The whole reason for the host mirror: a drag is one upload a notch,
        not a download and an upload a notch."""
        for value in (0.1, 0.2, 0.3):
            self.writer("mu", value)
        self.assertEqual(self.model.shape_material_mu.assigns, 3)


class TestWritingAMass(unittest.TestCase):
    def _model(self, mass: float = 2.0) -> _Model:
        return _Model(
            body_mass=[mass],
            body_inv_mass=[1.0 / mass if mass else 0.0],
            body_inertia=[np.eye(3) * mass],
            body_inv_inertia=[np.eye(3) / mass if mass else np.zeros((3, 3))],
        )

    def test_the_mass_lands(self) -> None:
        model = self._model(2.0)
        self.assertTrue(BodyWriter(model, [0])("mass", 4.0))
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 4.0)

    def test_the_inverse_lands_with_it(self) -> None:
        model = self._model(2.0)
        BodyWriter(model, [0])("mass", 4.0)
        self.assertAlmostEqual(float(model.body_inv_mass.numpy()[0]), 0.25)

    def test_the_inertia_scales_by_the_same_factor(self) -> None:
        """The same shape, less dense. Leaving the tensor alone gives a body a
        tenth the weight and the same resistance to rotation, which is not a
        lighter object -- it is one that tumbles like nothing real."""
        model = self._model(2.0)
        BodyWriter(model, [0])("mass", 4.0)
        np.testing.assert_allclose(model.body_inertia.numpy()[0], np.eye(3) * 4.0)
        np.testing.assert_allclose(model.body_inv_inertia.numpy()[0], np.eye(3) * 0.25)

    def test_a_welded_body_stays_welded(self) -> None:
        """Zero inverse mass is what marks a static fixed. Quietly giving it one
        would unweld it from the world -- the failure `preserve_body_inertia`
        exists to avoid on the other side of the build."""
        model = self._model(0.0)
        self.assertFalse(BodyWriter(model, [0])("mass", 4.0))
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 0.0)
        self.assertAlmostEqual(float(model.body_inv_mass.numpy()[0]), 0.0)

    def test_nothing_but_mass_is_offered(self) -> None:
        self.assertFalse(BodyWriter(self._model(), [0])("com", 1.0))


class TestWritingAJointDrive(unittest.TestCase):
    def test_a_gain_lands_on_this_entitys_dofs(self) -> None:
        model = _Model(**{attr: np.zeros(3) for attr in JointWriter.FIELDS.values()})
        self.assertTrue(JointWriter(model, [2])("target_ke", 500.0))
        np.testing.assert_array_equal(model.joint_target_ke.numpy(), [0.0, 0.0, 500.0])

    def test_a_travel_limit_lands(self) -> None:
        model = _Model(**{attr: np.zeros(3) for attr in JointWriter.FIELDS.values()})
        JointWriter(model, [0])("limit_upper", 0.4)
        self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[0]), 0.4)


class TestAnArrayThatIsNotThere(unittest.TestCase):
    """A model need not carry every array a writer knows about, and a missing one
    is a `False` rather than an `AttributeError` -- which is what lets a panel say
    "this entity has nothing to write that to" instead of a traceback."""

    def test_a_missing_array_reports_that_nothing_landed(self) -> None:
        self.assertFalse(ShapeWriter(_Model(), [0])("mu", 1.0))

    def test_an_empty_array_reports_that_nothing_landed(self) -> None:
        self.assertFalse(ShapeWriter(_Model(shape_material_mu=[]), [0])("mu", 1.0))

    def test_reading_a_missing_array_answers_a_default(self) -> None:
        self.assertEqual(ShapeWriter(_Model(), [0]).read()["mu"], 0.0)

    def test_a_base_writer_declares_no_flags(self) -> None:
        self.assertEqual(ArrayWriter.flags, 0)


class TestTheSpecsAreFullyDescribed(unittest.TestCase):
    """Every field carries a `ui()`, which is the assertion `params.undescribed`
    exists to make possible -- a panel is generated from these, so an undescribed
    field is a control that silently never appears."""

    def test_nothing_is_undescribed(self) -> None:
        for spec in (ContactMaterialSpec, BodySpec, JointDriveSpec):
            with self.subTest(spec.__name__):
                self.assertEqual(undescribed(spec), ())

    def test_they_are_recapture_tier(self) -> None:
        """The assignment is live; the notification is host work a captured graph
        does not replay. Free while the studio runs eager, which it does."""
        for spec in (ContactMaterialSpec, BodySpec, JointDriveSpec):
            for f, described in iter_params(spec):
                with self.subTest(f"{spec.__name__}.{f.name}"):
                    self.assertEqual(described.tier, RECAPTURE)

    def test_every_spec_field_has_somewhere_to_be_written(self) -> None:
        """A described field no writer maps is a slider that does nothing."""
        for spec, writer in ((ContactMaterialSpec, ShapeWriter), (JointDriveSpec, JointWriter)):
            for f, _ in iter_params(spec):
                with self.subTest(f"{spec.__name__}.{f.name}"):
                    self.assertIn(f.name, writer.FIELDS)


if __name__ == "__main__":
    unittest.main()
