# SPDX-License-Identifier: Apache-2.0
"""A jointed object from a USD stage: placed per world, its joints named and readable.

``ArticulatedSpec`` builds a SimReady-shaped cabinet -- a body, a prismatic
drawer, a revolute door, no articulation root in the file -- on a fixed base
(the importer's fixed joint from the world) or a free one (a FREE joint on the
root part). Its handle names the parts and joints the asset declares, a joint's
coordinate is read off where its two bodies are, a reset places one world's copy
and starts its joints, and the interaction graph does not take it for the robot
although it is in the robot's domain. A goal reads a drawer through the ``joint``
feature kind and the ``opening`` atom.

Built on a CPU builder from stages :mod:`tests.fixtures.simready` writes; the
reset runs against a stub pipeline, and nothing is stepped.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np

from robosimstudio.core.worlds import WorldMap
from robosimstudio.domains.articulated.entity import EntityArticulated, joint_name, part_name
from robosimstudio.domains.articulated.measures import _joint_opening
from robosimstudio.scene.handles import ArticulatedHandle
from robosimstudio.scene.physics.interactions import build_graph
from robosimstudio.specs import GENERIC, ArticulatedSpec, AssetSpec, JointSpec
from tests.fixtures.simready import Joint, Part, write_stage

TIER = "fast"
"""Builds stages into a CPU builder and finalizes them; no solver, no GPU."""

POSE = (0.5, 0.2, 0.0)
BODY_Z = 0.3
"""How far above the asset's origin its root part's centre is: the root is placed by that offset."""


def _spec(path: Path, **kwargs) -> ArticulatedSpec:
    return ArticulatedSpec(name=kwargs.pop("name", "cabinet"), asset=AssetSpec(path=str(path)), pose=POSE, **kwargs)


def _built(spec: ArticulatedSpec):
    entity = EntityArticulated(spec)
    builder = newton.ModelBuilder()
    entity.build_into(builder, None)
    model = builder.finalize(device="cpu")
    entity.bind(model)
    return entity, model


class _Cabinet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = write_stage(
            Path(cls._dir.name) / "cabinet.usda",
            [
                Part("body_01", (0.0, 0.0, BODY_Z), (0.2, 0.2, 0.3)),
                Part("drawer_01", (0.25, 0.0, 0.4), (0.05, 0.15, 0.05)),
                Part("door_01", (0.22, 0.0, 0.15), (0.01, 0.15, 0.1)),
            ],
            [
                Joint("drawer_01", "prismatic", "body_01", "drawer_01", (0.25, 0.0, 0.4), "X", (0.0, 0.3)),
                Joint("door_01", "revolute", "body_01", "door_01", (0.22, -0.15, 0.15), "Z", (0.0, 90.0)),
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()


class TestItStandsOnTheBaseItAsks(_Cabinet):
    def test_a_fixed_base_is_one_fixed_joint_from_the_world(self) -> None:
        entity, model = _built(_spec(self.path))
        types = model.joint_type.numpy().tolist()
        parents = model.joint_parent.numpy().tolist()
        self.assertEqual(1, model.articulation_count)
        self.assertEqual(2, model.joint_dof_count)
        roots = [t for t, p in zip(types, parents, strict=True) if p == -1]
        self.assertEqual([int(newton.JointType.FIXED)], roots)
        self.assertEqual(entity.weld_joint, parents.index(-1))
        self.assertEqual(-1, entity.coord_start)
        np.testing.assert_allclose(model.body_q.numpy()[entity.body][:3], (0.5, 0.2, BODY_Z), atol=1e-6)

    def test_a_free_base_is_a_free_joint_on_the_root_part(self) -> None:
        entity, model = _built(_spec(self.path, fixed_base=False))
        self.assertEqual(8, model.joint_dof_count)
        self.assertEqual(int(newton.JointType.FREE), int(model.joint_type.numpy()[entity._root_joint]))
        self.assertGreaterEqual(entity.coord_start, 0)
        self.assertEqual(-1, entity.weld_joint)
        np.testing.assert_allclose(entity.episode_start_q(None)[:3], (0.5, 0.2, BODY_Z), atol=1e-6)


class TestItsPartsAndJointsAreNamed(_Cabinet):
    def test_simready_labels_become_names(self) -> None:
        self.assertEqual("drawer_01", part_name("/RootNode/Geometry/drawer_01_obj_00"))
        self.assertEqual("drawer_01", joint_name("/RootNode/Joints/joint_drawer_01_joint_00"))
        self.assertEqual("hinge", joint_name("a/hinge"))

    def test_the_handle_lists_them(self) -> None:
        entity, model = _built(_spec(self.path))
        handle = entity.handle(model)
        self.assertIsInstance(handle, ArticulatedHandle)
        self.assertEqual({"body_01", "drawer_01", "door_01"}, set(handle.parts))
        self.assertEqual({"drawer_01", "door_01"}, set(handle.joints))
        self.assertEqual("prismatic", handle.joints["drawer_01"].kind)
        self.assertEqual("revolute", handle.joints["door_01"].kind)
        self.assertAlmostEqual(np.pi / 2, handle.joints["door_01"].upper, places=4)
        self.assertEqual(handle.parts["body_01"], handle.body)
        with self.assertRaisesRegex(KeyError, "drawer_01"):
            handle.joint("drawer_09")

    def test_two_cabinets_keep_their_own(self) -> None:
        first, second = (
            EntityArticulated(_spec(self.path, name="left")),
            EntityArticulated(_spec(self.path, name="right")),
        )
        builder = newton.ModelBuilder()
        first.build_into(builder, None)
        second.build_into(builder, None)
        model = builder.finalize(device="cpu")
        a, b = first.handle(model), second.handle(model)
        self.assertNotEqual(a.joints["drawer_01"].joint, b.joints["drawer_01"].joint)
        self.assertNotEqual(a.body, b.body)


class TestTheSpecStartsAndDrivesItsJoints(_Cabinet):
    def test_the_start_and_the_overrides_land(self) -> None:
        spec = _spec(
            self.path,
            joints=(JointSpec(name="drawer_01", q=0.12, damping=3.0, target_ke=200.0, target_kd=20.0),),
        )
        entity, model = _built(spec)
        handle = entity.handle(model)
        drawer = handle.joints["drawer_01"]
        self.assertAlmostEqual(0.12, float(model.joint_q.numpy()[drawer.q_start]), places=6)
        self.assertAlmostEqual(3.0, float(model.joint_damping.numpy()[drawer.qd_start]))
        self.assertAlmostEqual(200.0, float(model.joint_target_ke.numpy()[drawer.qd_start]))
        self.assertEqual(int(newton.JointTargetMode.POSITION), int(model.joint_target_mode.numpy()[drawer.qd_start]))
        self.assertAlmostEqual(0.12, float(model.joint_target_q.numpy()[drawer.q_start]), places=6)

    def test_a_joint_it_does_not_have_is_named(self) -> None:
        with self.assertRaisesRegex(ValueError, r"no joint 'drawer_09'.*drawer_01"):
            _built(_spec(self.path, joints=(JointSpec(name="drawer_09", q=0.1),)))

    def test_a_joint_named_twice_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "named twice"):
            _spec(self.path, joints=(JointSpec(name="drawer_01"), JointSpec(name="drawer_01")))

    def test_a_start_outside_the_travel_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"starts drawer_01 at 0.5, outside its travel \[0.0, 0.3"):
            _built(_spec(self.path, joints=(JointSpec(name="drawer_01", q=0.5),)))


class TestAPartRestingOutsideItsTravelStartsAtTheNearerLimit(unittest.TestCase):
    """The sterilization tray's lock: authored at rest at 0, between limits of 5 and 120 degrees."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = write_stage(
            Path(cls._dir.name) / "tray.usda",
            [
                Part("body_01", (0.0, 0.0, 0.02), (0.2, 0.1, 0.02)),
                Part("lock_01", (0.2, 0.0, 0.05), (0.01, 0.03, 0.02)),
            ],
            [Joint("lock_01", "revolute", "body_01", "lock_01", (0.2, 0.0, 0.04), "Y", (5.0, 120.0))],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_it_starts_at_the_limit_and_says_so(self) -> None:
        with self.assertWarnsRegex(UserWarning, r"lock_01 at 0 outside \[0.0872665, 2.0944\]"):
            entity, model = _built(_spec(self.path, name="tray"))
        lock = entity.handle(model).joints["lock_01"]
        self.assertAlmostEqual(np.radians(5.0), float(model.joint_q.numpy()[lock.q_start]), places=5)

    def test_an_episode_start_outside_it_is_refused(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entity, _ = _built(_spec(self.path, name="tray"))
        with self.assertRaisesRegex(ValueError, "outside its travel"):
            entity.joint_starts({"joint_q": {"tray": {"lock_01": 0.0}}})


class TestAJointIsReadFromItsBodies(_Cabinet):
    def test_both_kinds_agree_with_joint_q(self) -> None:
        entity, model = _built(_spec(self.path, joints=(JointSpec(name="drawer_01", q=0.07),)))
        handle = entity.handle(model)
        joint_q = model.joint_q.numpy()
        joint_q[handle.joints["door_01"].q_start] = 0.5
        state = model.state()
        newton.eval_fk(model, _wp(joint_q), model.joint_qd, state)
        body_q = state.body_q.numpy()
        self.assertAlmostEqual(0.07, handle.joints["drawer_01"].coordinate_from_bodies(body_q), places=5)
        self.assertAlmostEqual(0.5, handle.joints["door_01"].coordinate_from_bodies(body_q), places=5)
        self.assertAlmostEqual(0.07, handle.joints["drawer_01"].coordinate(joint_q), places=6)

    def test_published_states_carry_each_coordinate(self) -> None:
        entity, model = _built(_spec(self.path, joints=(JointSpec(name="drawer_01", q=0.2),)))
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        published = entity.published_states(SimpleNamespace(body_q=state.body_q.numpy()))
        self.assertAlmostEqual(0.2, published["drawer_01.q"], places=5)
        self.assertAlmostEqual(0.0, published["door_01.q"], places=5)


class TestATurnPastHalfIsReadOnItsTravel(unittest.TestCase):
    def test_a_lid_open_past_pi_reads_its_own_angle(self) -> None:
        from robosimstudio.core.math import quat_from_axis_angle
        from robosimstudio.scene.handles import JointHandle

        lid = JointHandle(
            name="lid",
            joint=0,
            kind="revolute",
            q_start=0,
            qd_start=0,
            parent=-1,
            child=0,
            axis=(0.0, 0.0, 1.0),
            X_p=(0, 0, 0, 0, 0, 0, 1),
            X_c=(0, 0, 0, 0, 0, 0, 1),
            lower=0.0,
            upper=3.5,
        )
        for angle in (0.2, 3.3, 3.5):
            body_q = np.array([[0.0, 0.0, 0.0, *quat_from_axis_angle((0.0, 0.0, 1.0), angle)]])
            self.assertAlmostEqual(angle, lid.coordinate_from_bodies(body_q), places=6)


def _wp(values: np.ndarray):
    import warp as wp

    return wp.array(values, dtype=float, device="cpu")


class _Pipeline:
    """The three things a reset touches, on one model: two states, a control, and a program to notify."""

    def __init__(self, model) -> None:
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        self.notified: list[int] = []
        self.program = SimpleNamespace(notify=self.notified.append)


class TestAResetPlacesOneWorld(_Cabinet):
    def _fleet(self, spec: ArticulatedSpec):
        entity = EntityArticulated(spec)
        base = newton.ModelBuilder()
        entity.build_into(base, None)
        fleet = newton.ModelBuilder()
        fleet.replicate(base, 2, spacing=(0.0, 0.0, 0.0))
        template, model = base.finalize(device="cpu"), fleet.finalize(device="cpu")
        entity.worlds = WorldMap.of(template, model, 2)
        entity.stepping_model = model
        entity.bind(template)
        pipeline = _Pipeline(model)
        newton.eval_fk(model, model.joint_q, model.joint_qd, pipeline.state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, pipeline.state_1)
        return entity, model, pipeline

    def _check(self, entity, model, pipeline) -> None:
        worlds = entity.worlds
        handle = entity.handle(entity.model)
        before = pipeline.state_0.body_q.numpy()[worlds.bodies(0)].copy()
        episode = {"object_poses": {"cabinet": (0.9, -0.3, 0.0, 0.0)}, "joint_q": {"cabinet": {"drawer_01": 0.25}}}
        entity.reset(pipeline, episode, world=1)
        for state in (pipeline.state_0, pipeline.state_1):
            body_q = state.body_q.numpy()
            np.testing.assert_allclose(body_q[worlds.bodies(0)], before, atol=1e-6)
            one = worlds.body_view(body_q, 1)
            np.testing.assert_allclose(one[handle.body][:3], (0.9, -0.3, BODY_Z), atol=1e-5)
            self.assertAlmostEqual(0.25, handle.joints["drawer_01"].coordinate_from_bodies(one), places=5)
            self.assertAlmostEqual(
                0.0, handle.joints["drawer_01"].coordinate_from_bodies(worlds.body_view(body_q, 0)), places=5
            )

    def test_a_fixed_base_moves_through_its_weld(self) -> None:
        entity, model, pipeline = self._fleet(_spec(self.path))
        self._check(entity, model, pipeline)
        self.assertEqual([int(newton.ModelFlags.JOINT_PROPERTIES)], pipeline.notified)

    def test_a_free_base_moves_through_its_coordinates(self) -> None:
        entity, model, pipeline = self._fleet(_spec(self.path, fixed_base=False))
        self._check(entity, model, pipeline)

    def test_an_episode_naming_a_joint_it_does_not_have_is_refused(self) -> None:
        entity, _, pipeline = self._fleet(_spec(self.path))
        with self.assertRaisesRegex(ValueError, "drawer_09"):
            entity.reset(pipeline, {"joint_q": {"cabinet": {"drawer_09": 0.1}}}, world=0)


class TestALooseAssetIsOneEntityOfFreeParts(unittest.TestCase):
    """A jar and its lid: two rigid bodies and no joint, as 39 SimReady assets are."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = write_stage(
            Path(cls._dir.name) / "jar.usda",
            [
                Part("jar_01", (0.0, 0.0, 0.05), (0.04, 0.04, 0.05), mass=4.0),
                Part("lid_01", (0.0, 0.0, 0.11), (0.04, 0.04, 0.01)),
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_a_fixed_base_bolts_the_heavier_part_and_frees_the_other(self) -> None:
        entity, model = _built(_spec(self.path, name="jar"))
        parts = entity.handle(model).parts
        self.assertEqual(parts["jar_01"], entity.body)
        kinds = {int(model.joint_child.numpy()[j]): int(model.joint_type.numpy()[j]) for j in range(model.joint_count)}
        self.assertEqual(int(newton.JointType.FIXED), kinds[parts["jar_01"]])
        self.assertEqual(int(newton.JointType.FREE), kinds[parts["lid_01"]])

    def test_every_part_is_free_and_named(self) -> None:
        entity, model = _built(_spec(self.path, name="jar", fixed_base=False))
        self.assertEqual([int(newton.JointType.FREE)] * 2, model.joint_type.numpy().tolist())
        self.assertEqual({"jar_01", "lid_01"}, set(entity.handle(model).parts))

    def test_a_reset_places_both_parts_with_the_asset(self) -> None:
        for fixed_base in (False, True):
            with self.subTest(fixed_base=fixed_base):
                self._reset_places_both(fixed_base)

    def _reset_places_both(self, fixed_base: bool) -> None:
        entity = EntityArticulated(_spec(self.path, name="jar", fixed_base=fixed_base))
        base = newton.ModelBuilder()
        entity.build_into(base, None)
        fleet = newton.ModelBuilder()
        fleet.replicate(base, 2, spacing=(0.0, 0.0, 0.0))
        template, model = base.finalize(device="cpu"), fleet.finalize(device="cpu")
        entity.worlds, entity.stepping_model = WorldMap.of(template, model, 2), model
        entity.bind(template)
        pipeline = _Pipeline(model)
        entity.reset(pipeline, {"object_poses": {"jar": (0.9, -0.3, 0.0, 0.0)}}, world=1)
        handle = entity.handle(template)
        one = entity.worlds.body_view(pipeline.state_0.body_q.numpy(), 1)
        np.testing.assert_allclose(one[handle.parts["jar_01"]][:3], (0.9, -0.3, 0.05), atol=1e-5)
        np.testing.assert_allclose(one[handle.parts["lid_01"]][:3], (0.9, -0.3, 0.11), atol=1e-5)


class TestAWorldDocumentMovesItsBase(TestAResetPlacesOneWorld):
    def test_the_reset_after_it_places_from_that_document(self) -> None:
        entity, _, pipeline = self._fleet(_spec(self.path))
        moved = dataclasses.replace(entity.spec, pose=(-0.4, 0.6, 0.0))
        entity.place_from_document(pipeline, moved, world=1, document="moved")
        entity.reset(pipeline, {}, world=1)
        one = entity.worlds.body_view(pipeline.state_0.body_q.numpy(), 1)
        np.testing.assert_allclose(one[entity.body][:3], (-0.4, 0.6, BODY_Z), atol=1e-5)
        # Taken for that wave only: the next reset is from the spec's own pose again.
        entity.reset(pipeline, {}, world=1)
        one = entity.worlds.body_view(pipeline.state_0.body_q.numpy(), 1)
        np.testing.assert_allclose(one[entity.body][:3], (*POSE[:2], BODY_Z), atol=1e-5)

    def test_a_document_making_it_from_another_asset_is_refused(self) -> None:
        entity, _, pipeline = self._fleet(_spec(self.path))
        other = dataclasses.replace(entity.spec, asset=AssetSpec(path="elsewhere.usda"))
        with self.assertRaisesRegex(ValueError, "elsewhere.usda"):
            entity.place_from_document(pipeline, other, world=1, document="other")


class TestAnAuthoredDriveHoldsTheStart(unittest.TestCase):
    """A drawer whose asset drives it towards 0: started open, its target is where it starts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = write_stage(
            Path(cls._dir.name) / "driven.usda",
            [
                Part("body_01", (0.0, 0.0, BODY_Z), (0.2, 0.2, 0.3)),
                Part("drawer_01", (0.25, 0.0, 0.4), (0.05, 0.15, 0.05)),
            ],
            [Joint("drawer_01", "prismatic", "body_01", "drawer_01", (0.25, 0.0, 0.4), "X", (0.0, 0.3), drive=50.0)],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_the_built_target_is_the_start(self) -> None:
        entity, model = _built(_spec(self.path, joints=(JointSpec(name="drawer_01", q=0.15),)))
        drawer = entity.handle(model).joints["drawer_01"]
        self.assertGreater(float(model.joint_target_ke.numpy()[drawer.qd_start]), 0.0, "the asset's drive")
        self.assertAlmostEqual(0.15, float(model.joint_target_q.numpy()[drawer.q_start]), places=6)

    def test_a_reset_moves_the_target_with_the_episode(self) -> None:
        entity, model = _built(_spec(self.path))
        pipeline = _Pipeline(model)
        entity.reset(pipeline, {"joint_q": {"cabinet": {"drawer_01": 0.2}}})
        drawer = entity.handle(model).joints["drawer_01"]
        self.assertAlmostEqual(0.2, float(pipeline.control.joint_target_q.numpy()[drawer.q_start]), places=6)


class TestARootItCannotPlaceIsRefused(unittest.TestCase):
    def test_a_door_hinged_to_the_world(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = write_stage(
                Path(folder) / "door.usda",
                [Part("door_01", (0.0, 0.0, 0.5), (0.02, 0.4, 0.5))],
                [Joint("door_01", "revolute", "", "door_01", (0.0, -0.4, 0.5), "Z", (0.0, 90.0))],
            )
            for fixed_base in (True, False):
                with self.subTest(fixed_base=fixed_base), self.assertRaisesRegex(ValueError, "jointed to it otherwise"):
                    _built(_spec(path, name="door", fixed_base=fixed_base))


class TestItIsNotTheRobot(_Cabinet):
    def test_its_node_plays_objects(self) -> None:
        entity, model = _built(_spec(self.path))
        graph = build_graph(model, {"cabinet": entity}, None, GENERIC)
        self.assertEqual(frozenset({"objects"}), graph.nodes["cabinet"].roles)
        self.assertIsNone(graph.robot())


class TestAGoalReadsHowFarItIsOpen(_Cabinet):
    def test_the_reader_and_the_handler(self) -> None:
        from robosimstudio.assetlib.semantics.features import FeatureSpec
        from robosimstudio_bench.goals.bind import Operand, read

        entity, model = _built(_spec(self.path, joints=(JointSpec(name="drawer_01", q=0.15),)))
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        handle = entity.handle(model)
        snapshot = SimpleNamespace(
            body_q=state.body_q.numpy(),
            handles=SimpleNamespace(entity=lambda name: {"cabinet": handle}[name]),
            entities={"cabinet": entity},
        )
        operand = Operand(
            ref=SimpleNamespace(base="cabinet", arg=None, path=("drawer",)),
            substrate="articulated",
            kind="joint",
            feature=FeatureSpec(name="drawer", kind="joint", payload={"joint": "drawer_01"}),
        )
        joint = read(snapshot, operand, form="feature")
        self.assertAlmostEqual(0.15, joint["coordinate"], places=5)
        self.assertEqual((0.0, 0.3), (joint["lower"], round(joint["upper"], 6)))
        self.assertAlmostEqual(0.5, _joint_opening(joint)["fraction"], places=4)


if __name__ == "__main__":
    unittest.main()
