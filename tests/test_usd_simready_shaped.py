# SPDX-License-Identifier: Apache-2.0
"""A SimReady-shaped USD prop imports as one rigid body, on the stage's Newton variant.

NVIDIA's SimReady assets carry a variant set per physics engine with PhysX
enabled, and most of their multi-part props hold the parts with fixed joints that
belong to no articulation, which Newton's ``finalize`` refuses. The import selects
the Newton variant, gives such joints one articulation root, and folds a prop
whose joints are all fixed into one body -- all on a throwaway session layer, and
folding only what this import added. A stage with moving joints is refused by the
rigid path with a pointer to ``ArticulatedSpec``. A stage whose jointed parts form
two trees under one root imports as two articulations, and one whose joints close
a loop imports with the closing joint outside the tree. A mesh bound to two
materials is drawn in both, and an asked-for collider approximation is applied.

Built on a CPU builder from stages :mod:`tests.fixtures.simready` writes; nothing
is stepped.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import newton
import numpy as np
import warp as wp

from robosimstudio.domains.articulated.entity import EntityArticulated
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.imports import (
    add_usd_stage,
    articulate,
    open_session_stage,
    select_engine_variant,
    split_articulations,
)
from robosimstudio.specs import ArticulatedSpec, AssetSpec, ObjectSpec
from tests.fixtures.simready import NEWTON_MASS, PHYSX_MASS, Joint, Part, write_stage

TIER = "fast"
"""Builds stages into a CPU builder and finalizes them; no solver, no GPU."""


def _rigid(path: Path, name: str = "prop") -> EntityRigid:
    return EntityRigid(ObjectSpec(name=name, asset=AssetSpec(path=str(path)), pose=(0.0, 0.0, 0.1), obb_grasp=False))


class _Stages(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.TemporaryDirectory()
        root = Path(cls._dir.name)
        cls.box = write_stage(root / "box.usda", [Part("box_01")])
        cls.pallet = write_stage(root / "pallet.usda", [Part("pallet_01", (0.0, 0.0, 0.3), (0.5, 0.5, 0.3))])
        cls.bottle = write_stage(
            root / "bottle.usda",
            [
                Part("body_01", (0.0, 0.0, 0.08), (0.03, 0.03, 0.08)),
                Part("cap_01", (0.0, 0.0, 0.17), (0.02, 0.02, 0.01)),
            ],
            [Joint("cap_01", "fixed", "body_01", "cap_01", (0.0, 0.0, 0.16))],
        )
        cls.cabinet = write_stage(
            root / "cabinet.usda",
            [
                Part("body_01", (0.0, 0.0, 0.3), (0.2, 0.2, 0.3)),
                Part("drawer_01", (0.25, 0.0, 0.4), (0.05, 0.15, 0.05)),
            ],
            [Joint("drawer_01", "prismatic", "body_01", "drawer_01", (0.25, 0.0, 0.4), "X", (0.0, 0.3))],
        )
        cls.split = write_stage(
            root / "split.usda",
            [
                Part("body_01", mass=3.0),
                Part("hinge_01", (0.1, 0.0, 0.0)),
                Part("lid_01", (0.0, 0.0, 0.2)),
                Part("lock_01", (0.1, 0.0, 0.2)),
            ],
            [
                Joint("hinge_01", "revolute", "body_01", "hinge_01", (0.1, 0.0, 0.0), "Y", (-90.0, 90.0)),
                Joint("lock_01", "revolute", "lid_01", "lock_01", (0.1, 0.0, 0.2), "Y", (-90.0, 90.0)),
            ],
        )
        # The rongeurs' linkage: each jaw pivots on its handle and on the other jaw.
        cls.pliers = write_stage(
            root / "pliers.usda",
            [
                Part("handleLeft_01", (0.0, 0.0, 0.0), (0.1, 0.01, 0.01)),
                Part("handleRight_01", (0.0, 0.03, 0.0), (0.1, 0.01, 0.01)),
                Part("jawLeft_01", (0.15, 0.0, 0.0), (0.03, 0.01, 0.01)),
                Part("jawRight_01", (0.15, 0.03, 0.0), (0.03, 0.01, 0.01)),
            ],
            [
                Joint(
                    "jawRight_01", "revolute", "handleRight_01", "jawRight_01", (0.12, 0.03, 0.0), "Z", (-15.0, 15.0)
                ),
                Joint(
                    "handleRight_01",
                    "revolute",
                    "handleLeft_01",
                    "handleRight_01",
                    (0.1, 0.015, 0.0),
                    "Z",
                    (-30.0, 30.0),
                ),
                Joint("jawLeft_01", "revolute", "handleLeft_01", "jawLeft_01", (0.12, 0.0, 0.0), "Z", (-15.0, 15.0)),
                Joint("jawLeft_01", "revolute", "jawRight_01", "jawLeft_01", (0.15, 0.015, 0.0), "Z", (-15.0, 15.0)),
            ],
        )
        cls.kettle = write_stage(
            root / "kettle.usda", [Part("kettle_01", materials=((0.8, 0.8, 0.8), (0.5, 0.3, 0.1)))]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()


class TestTheNewtonVariantIsSelected(_Stages):
    def test_the_rigid_path_takes_the_newton_mass(self) -> None:
        entity = _rigid(self.box)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        self.assertAlmostEqual(float(builder.body_mass[entity.body]), NEWTON_MASS, places=5)

    def test_no_engine_variant_keeps_the_shipped_selection(self) -> None:
        builder = newton.ModelBuilder()
        add_usd_stage(builder, str(self.box), floating=True, engine_variant=None)
        self.assertAlmostEqual(float(builder.body_mass[0]), PHYSX_MASS, places=5)

    def test_a_stage_without_engine_variants_is_not_touched(self) -> None:
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.SetDefaultPrim(stage.DefinePrim("/RootNode", "Xform"))
        self.assertEqual([], select_engine_variant(stage, "Newton"))


class TestAFixedJointPropIsOneBody(_Stages):
    def test_the_bottle_and_its_cap_are_one_body_with_both_masses(self) -> None:
        entity = _rigid(self.bottle)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        model = builder.finalize(device="cpu")
        self.assertEqual(1, model.body_count)
        self.assertEqual([int(newton.JointType.FREE)], model.joint_type.numpy().tolist())
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 2 * NEWTON_MASS, places=4)

    def test_the_fold_leaves_what_was_built_before_it_alone(self) -> None:
        cabinet = lambda: EntityArticulated(ArticulatedSpec(name="cabinet", asset=AssetSpec(path=str(self.cabinet))))  # noqa: E731
        alone = newton.ModelBuilder()
        cabinet().build_into(alone, None)
        both = newton.ModelBuilder()
        cabinet().build_into(both, None)
        prop = _rigid(self.bottle)
        prop.build_into(both, None)
        bodies, joints, shapes = alone.body_count, alone.joint_count, alone.shape_count
        np.testing.assert_allclose(np.asarray(alone.body_q), np.asarray(both.body_q)[:bodies])
        self.assertEqual(list(alone.joint_type), list(both.joint_type)[:joints])
        self.assertEqual(list(alone.joint_parent), list(both.joint_parent)[:joints])
        self.assertEqual(list(alone.shape_body), list(both.shape_body)[:shapes])
        self.assertEqual(bodies, prop.body)
        both.finalize(device="cpu")


class TestABodyAboveTheStageOriginStandsWhereTheAssetDoes(_Stages):
    """The pallet's body frame is 0.3 m above its stage origin, its box hanging to the origin."""

    def test_the_free_joint_holds_the_body_frame_not_the_pose(self) -> None:
        entity = _rigid(self.pallet)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        c = entity.coord_start
        np.testing.assert_allclose(list(builder.joint_q[c : c + 3]), (0.0, 0.0, 0.1 + 0.3), atol=1e-6)
        np.testing.assert_allclose(entity.episode_start_q(None)[:3], (0.0, 0.0, 0.4), atol=1e-6)

    def test_an_episode_pose_moves_the_asset_origin(self) -> None:
        entity = _rigid(self.pallet)
        entity.build_into(newton.ModelBuilder(), None)
        moved = entity.episode_start_q({"object_poses": {"prop": (1.0, 2.0, 0.0, np.pi / 2)}})
        np.testing.assert_allclose(moved[:3], (1.0, 2.0, 0.3), atol=1e-6)

    def test_a_body_at_the_origin_has_no_offset(self) -> None:
        entity = _rigid(self.box)
        entity.build_into(newton.ModelBuilder(), None)
        self.assertIsNone(entity.body_offset)


class TestAJointedStageIsNotARigidBody(_Stages):
    def test_it_is_refused_with_the_spec_that_takes_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "ArticulatedSpec"):
            _rigid(self.cabinet).build_into(newton.ModelBuilder(), None)

    def test_static_is_refused_with_the_same_pointer(self) -> None:
        spec = ObjectSpec(name="box", asset=AssetSpec(path=str(self.box)), static=True, obb_grasp=False)
        with self.assertRaisesRegex(ValueError, "ArticulatedSpec with fixed_base=True"):
            EntityRigid(spec).build_into(newton.ModelBuilder(), None)


class TestTheFileIsNotEdited(_Stages):
    def test_the_root_layer_is_clean_and_the_bytes_unchanged(self) -> None:
        from pxr import Sdf

        before = self.cabinet.read_bytes()
        builder = newton.ModelBuilder()
        add_usd_stage(builder, str(self.cabinet), floating=False)
        self.assertEqual(before, self.cabinet.read_bytes())
        self.assertFalse(Sdf.Layer.FindOrOpen(str(self.cabinet)).dirty)

    def test_a_second_open_sees_the_stage_as_shipped(self) -> None:
        from pxr import UsdPhysics

        first = open_session_stage(self.cabinet)
        self.assertEqual(["/RootNode"], articulate(first))
        second = open_session_stage(self.cabinet)
        self.assertFalse(second.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI))


class TestEachTreeIsItsOwnArticulation(_Stages):
    """The storage box: a box on two hinges, and a lid with a lock that no joint holds to it."""

    def test_free_the_two_trees_are_two_free_articulations(self) -> None:
        builder = newton.ModelBuilder()
        add_usd_stage(builder, str(self.split), floating=True)
        model = builder.finalize(device="cpu")
        self.assertEqual(2, model.articulation_count)
        roots = [j for j in range(builder.joint_count) if builder.joint_parent[j] == -1]
        self.assertEqual([int(newton.JointType.FREE)] * 2, [int(builder.joint_type[j]) for j in roots])

    def test_fixed_the_heaviest_tree_stands_and_the_other_is_free(self) -> None:
        builder = newton.ModelBuilder()
        results = add_usd_stage(builder, str(self.split), floating=False)
        builder.finalize(device="cpu")
        roots = {
            builder.body_label[builder.joint_child[j]].rsplit("/", maxsplit=1)[-1]: int(builder.joint_type[j])
            for j in range(builder.joint_count)
            if builder.joint_parent[j] == -1
        }
        self.assertEqual(
            {"body_01_obj_00": int(newton.JointType.FIXED), "lid_01_obj_00": int(newton.JointType.FREE)}, roots
        )
        self.assertEqual(4, len(results["path_body_map"]))

    def test_the_stage_is_split_on_the_session_layer_only(self) -> None:
        from pxr import UsdPhysics

        stage = open_session_stage(self.split)
        articulate(stage)
        changed = split_articulations(stage)
        self.assertEqual(
            ["/RootNode/Geometry/body_01_obj_00", "/RootNode/Geometry/lid_01_obj_00"],
            sorted(changed["articulation_roots_split"]),
        )
        self.assertFalse(stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI))
        fresh = open_session_stage(self.split)
        self.assertFalse(fresh.GetPrimAtPath("/RootNode/Geometry/lid_01_obj_00").HasAPI(UsdPhysics.ArticulationRootAPI))


class TestALoopClosesOutsideTheTree(_Stages):
    def test_the_second_joint_onto_a_jaw_is_the_one_left_out(self) -> None:
        stage = open_session_stage(self.pliers)
        articulate(stage)
        self.assertEqual(
            ["/RootNode/Joints/joint_jawLeft_01_joint_01"], split_articulations(stage)["loop_joints_excluded"]
        )

    def test_two_hinges_on_one_pair_are_no_loop(self) -> None:
        path = write_stage(
            Path(self._dir.name) / "shoulder.usda",
            [Part("torso_01"), Part("arm_01", (0.2, 0.0, 0.0))],
            [
                Joint("arm_01", "revolute", "torso_01", "arm_01", (0.1, 0.0, 0.0), "Y", (-90.0, 90.0)),
                Joint("arm_01", "revolute", "torso_01", "arm_01", (0.1, 0.0, 0.0), "Z", (-90.0, 90.0)),
            ],
        )
        stage = open_session_stage(path)
        articulate(stage)
        self.assertEqual([], split_articulations(stage)["loop_joints_excluded"])

    def test_it_builds_with_that_joint_in_no_articulation(self) -> None:
        builder = newton.ModelBuilder()
        add_usd_stage(builder, str(self.pliers), floating=False)
        model = builder.finalize(device="cpu")
        self.assertEqual(1, model.articulation_count)
        loose = [builder.joint_label[j] for j in range(builder.joint_count) if builder.joint_articulation[j] < 0]
        self.assertEqual(["/RootNode/Joints/joint_jawLeft_01_joint_01"], loose)

    def test_the_entity_names_the_tree_joints_only(self) -> None:
        entity = EntityArticulated(ArticulatedSpec(name="pliers", asset=AssetSpec(path=str(self.pliers))))
        entity.build_into(newton.ModelBuilder(), None)
        self.assertEqual(["handleRight_01", "jawLeft_01", "jawRight_01"], sorted(entity.joints))


class TestEachCollapseFoldsOnlyItsOwnImport(_Stages):
    """The robot and static paths, which used to collapse every fixed joint in the builder."""

    def _welded_box(self, builder: newton.ModelBuilder) -> int:
        body = builder.add_link(xform=wp.transform((1.0, 0.0, 0.5), wp.quat_identity()), label="fixture")
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([builder.add_joint_fixed(-1, body)])
        return body

    def test_a_usd_collapse_keeps_an_earlier_weld(self) -> None:
        builder = newton.ModelBuilder()
        fixture = self._welded_box(builder)
        add_usd_stage(builder, str(self.bottle), floating=True, collapse_fixed_joints=True)
        self.assertEqual("fixture", builder.body_label[fixture])
        self.assertEqual(int(newton.JointType.FIXED), int(builder.joint_type[0]))
        self.assertEqual(2, builder.body_count)

    def test_an_mjcf_collapse_keeps_an_earlier_weld(self) -> None:
        from robosimstudio.scene.build.imports import add_mjcf_collapsed

        mjcf = """<mujoco><worldbody><body name="arm"><freejoint/><geom type="box" size="0.1 0.1 0.1"/>
            <body name="flange" pos="0 0 0.2"><geom type="box" size="0.05 0.05 0.05"/></body></body></worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        fixture = self._welded_box(builder)
        add_mjcf_collapsed(builder, mjcf)
        self.assertEqual("fixture", builder.body_label[fixture])
        self.assertEqual(2, builder.body_count)


class TestAMeshInTwoMaterialsIsDrawnInBoth(_Stages):
    def test_two_visual_shapes_and_a_hidden_collider_of_the_same_mass(self) -> None:
        entity = _rigid(self.kettle)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        visible, colliding = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
        flags = [int(builder.shape_flags[s]) for s in range(builder.shape_count)]
        self.assertEqual(1, sum(1 for f in flags if f & colliding))
        self.assertFalse(next(f for f in flags if f & colliding) & visible)
        drawn = [s for s, f in enumerate(flags) if f & visible]
        self.assertEqual(2, len(drawn))
        # Two colours, one per material (Newton states them in sRGB, so not the authored numbers).
        self.assertEqual(2, len({tuple(round(float(c), 2) for c in builder.shape_color[s]) for s in drawn}))
        self.assertAlmostEqual(float(builder.body_mass[entity.body]), NEWTON_MASS, places=5)


class TestAnAskedApproximationIsApplied(_Stages):
    def test_a_usd_prop_takes_its_bounding_box_at_its_authored_mass(self) -> None:
        spec = ObjectSpec(
            name="prop",
            asset=AssetSpec(path=str(self.box), approximate="bounding_box"),
            pose=(0.0, 0.0, 0.1),
            obb_grasp=False,
        )
        entity = EntityRigid(spec)
        builder = newton.ModelBuilder()
        entity.build_into(builder, None)
        colliding = [
            s for s in range(builder.shape_count) if builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        self.assertEqual([int(newton.GeoType.BOX)], [int(builder.shape_type[s]) for s in colliding])
        self.assertAlmostEqual(float(builder.body_mass[entity.body]), NEWTON_MASS, places=5)

    def test_it_is_applied_once(self) -> None:
        # A second pass over the first's hulls, with no mass held, is what the
        # rigid build's own approximation would have been for a USD prop.
        spec = ObjectSpec(name="prop", asset=AssetSpec(path=str(self.box), approximate="bounding_box"), obb_grasp=False)
        original = newton.ModelBuilder.approximate_meshes
        with mock.patch.object(newton.ModelBuilder, "approximate_meshes", autospec=True, side_effect=original) as calls:
            EntityRigid(spec).build_into(newton.ModelBuilder(), None)
        asked = [c for c in calls.call_args_list if c.kwargs.get("method") == "bounding_box"]
        self.assertEqual(1, len(asked), "the importer's own pass is the stage's convexHull; this one is the spec's")


if __name__ == "__main__":
    unittest.main()
