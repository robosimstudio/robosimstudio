# SPDX-License-Identifier: Apache-2.0
"""``rss.robot(arms_at=...)``: the argument, its way to the compiler, and the placement on a builder.

The GPU half (``test_arms_at``) measures a built ``yam_bimanual``; this half
holds the plumbing and every refusal, none of which needs a device:

* ``arms_at`` is spelled as ``at`` is, per arm, and reaches the binding in world
  coordinates -- absolute, so it does not compose with ``at``;
* an arm the robot does not have is refused naming the arms it has, and a
  one-armed robot takes its one name;
* on a builder, the named arm's base lands on its pose before and after the
  fixed-joint collapse, an arm left out stays where the file put it, and the
  two layouts that cannot be placed one arm at a time are refused.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

TIER = "fast"
"""Specs, a compiler helper and a CPU builder; no device."""

H = math.sqrt(0.5)
LEFT = (-0.36, 0.20, 0.06)
RIGHT = (0.12, -0.36, 0.22, 0.0, 0.0, H, H)


class TestTheArgument(unittest.TestCase):
    def test_a_position_stays_a_position(self) -> None:
        """Not filled out to a pose: a position keeps the base's heading, which only the build knows."""
        import robosimstudio as rss

        placed = rss.robot("yam_bimanual", arms_at={"left": LEFT, "right": RIGHT})
        self.assertEqual((("left", LEFT), ("right", RIGHT)), placed.arms_at)

    def test_a_placement_built_directly_reads_the_same_and_hashes(self) -> None:
        import robosimstudio as rss

        direct = rss.Robot(name="yam_bimanual", arms_at={"left": list(LEFT)})
        self.assertEqual((("left", LEFT),), direct.arms_at)
        hash(direct)

    def test_unstated_is_empty(self) -> None:
        import robosimstudio as rss

        self.assertEqual((), rss.robot("yam_bimanual").arms_at)

    def test_a_bad_pose_names_the_arm(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(ValueError, r"arms_at\['left'\] is a position"):
            rss.robot("yam_bimanual", arms_at={"left": (0.1, 0.2)})
        with self.assertRaisesRegex(TypeError, r"arms_at\['left'\] is None"):
            rss.robot("yam_bimanual", arms_at={"left": None})

    def test_a_zero_rotation_is_refused_before_the_frame_hides_it(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(ValueError, r"arms_at\['right'\] has a zero quaternion"):
            rss.robot("yam_bimanual", arms_at={"right": (0.1, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0)})

    def test_not_a_mapping_of_names_is_refused(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(TypeError, "maps an arm's name"):
            rss.robot("yam_bimanual", arms_at=[LEFT])
        with self.assertRaisesRegex(TypeError, "maps an arm's name"):
            rss.robot("yam_bimanual", arms_at=LEFT)  # spelled like at= by mistake


class TestTheBinding(unittest.TestCase):
    def _binding(self, cell: str = "bench_075", **robot):
        import robosimstudio as rss

        scene = rss.Scene(cell=cell)
        scene.add(rss.robot("yam_bimanual", **robot))
        return scene.request.binding

    def test_arms_at_reaches_the_binding_in_world(self) -> None:
        """The bench's work-surface frame is 750 mm up and centred at world x = 0.6."""
        mounts = dict(self._binding(arms_at={"left": LEFT, "right": RIGHT}).arm_mounts)
        np.testing.assert_allclose(mounts["left"], (0.24, 0.20, 0.81), atol=1e-12)
        np.testing.assert_allclose(mounts["right"], (0.72, -0.36, 0.97, 0.0, 0.0, H, H), atol=1e-12)

    def test_arms_at_is_absolute_not_relative_to_at(self) -> None:
        binding = self._binding(at=(-0.2, 0.1, 0.05), arms_at={"left": LEFT})
        np.testing.assert_allclose(binding.mount, (0.40, 0.10, 0.80, 0.0, 0.0, 0.0, 1.0), atol=1e-12)
        np.testing.assert_allclose(dict(binding.arm_mounts)["left"], (0.24, 0.20, 0.81), atol=1e-12)

    def test_without_arms_at_the_binding_is_untouched(self) -> None:
        self.assertIsNone(self._binding())
        self.assertEqual((), self._binding(at=(0.0, 0.0, 0.0)).arm_mounts)

    def test_arm_mounts_without_a_robot_are_refused(self) -> None:
        import robosimstudio as rss
        from robosimstudio.specs import Binding

        scene = rss.Scene(cell="bench_075", binding=Binding(scene="bench_075", arm_mounts=(("left", LEFT),)))
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.1)))
        with self.assertRaisesRegex(ValueError, r"places arms \['left'\], and the scene has no robot"):
            scene.build(device="cpu", warmup=False)

    def test_arms_at_and_a_binding_saying_the_same_are_refused(self) -> None:
        import robosimstudio as rss
        from robosimstudio.specs import Binding

        binding = Binding(scene="bench_075", embodiment="yam_bimanual", arm_mounts=(("left", (0, 0, 1, 0, 0, 0, 1)),))
        scene = rss.Scene(cell="bench_075", binding=binding)
        scene.add(rss.robot("yam_bimanual", arms_at={"left": LEFT}))
        with self.assertRaisesRegex(ValueError, "both say where the arms stand"):
            _ = scene.request


class TestTheNames(unittest.TestCase):
    def _mounts(self, robot: str, **arms):
        import robosimstudio as rss
        from robosimstudio.robots.registry import get_embodiment
        from robosimstudio.scene.compiler import _arm_mounts

        scene = rss.Scene(cell="bench_075")
        scene.add(rss.robot(robot, arms_at=arms))
        request = scene.request
        return _arm_mounts(request, request.binding_for(), get_embodiment(robot))

    def test_the_robots_own_arms_pass_through_in_world(self) -> None:
        mounts = self._mounts("yam_bimanual", left=LEFT, right=RIGHT)
        self.assertEqual(["left", "right"], sorted(mounts))
        self.assertEqual(7, len(mounts["right"]))

    def test_an_arm_the_robot_does_not_have_is_refused_with_its_arms(self) -> None:
        with self.assertRaisesRegex(ValueError, r"arm 'middle'.*has arms \['left', 'right'\]"):
            self._mounts("yam_bimanual", middle=LEFT)

    def test_a_one_armed_robot_takes_its_one_name(self) -> None:
        self.assertEqual(["right"], sorted(self._mounts("panda", right=LEFT)))
        with self.assertRaisesRegex(ValueError, r"has arms \['right'\]"):
            self._mounts("panda", left=LEFT)

    def test_a_pose_without_a_rotation_is_refused(self) -> None:
        """A zero quaternion normalizes to nothing and would carry every body to a degenerate frame."""
        from robosimstudio.robots.registry import get_embodiment
        from robosimstudio.scene.compiler import _arm_mounts
        from robosimstudio.scene.request import BuildRequest
        from robosimstudio.specs import Binding

        binding = Binding(scene="bench_075", embodiment="panda", arm_mounts=(("right", (0, 0, 1, 0, 0, 0, 0)),))
        request = BuildRequest(cell="bench_075", embodiment="panda", binding=binding)
        with self.assertRaisesRegex(ValueError, "non-zero quaternion"):
            _arm_mounts(request, binding, get_embodiment("panda"))

    def test_a_binding_may_state_them_as_a_mapping(self) -> None:
        from robosimstudio.specs import Binding

        binding = Binding(scene="bench_075", embodiment="panda", arm_mounts={"right": [0, 0, 1, 0, 0, 0, 1]})
        self.assertEqual((("right", (0, 0, 1, 0, 0, 0, 1)),), binding.arm_mounts)
        hash(binding)
        with self.assertRaisesRegex(TypeError, "Binding.arm_mounts is"):
            Binding(arm_mounts=("right",))
        self.assertEqual((), Binding(arm_mounts=None).arm_mounts)

    def test_a_robot_inside_the_scenes_import_is_refused(self) -> None:
        from dataclasses import replace

        from robosimstudio.robots.registry import get_embodiment
        from robosimstudio.scene.compiler import _arm_mounts
        from robosimstudio.scene.request import BuildRequest
        from robosimstudio.specs import Binding

        binding = Binding(scene="bench_075", embodiment="panda", scene_resident=True, arm_mounts=(("right", RIGHT),))
        request = BuildRequest(cell="bench_075", embodiment="panda", binding=binding)
        with self.assertRaisesRegex(ValueError, "inside the scene's own import"):
            _arm_mounts(request, binding, get_embodiment("panda"))
        resident = replace(get_embodiment("panda"), asset=None)
        with self.assertRaisesRegex(ValueError, "inside the scene's own import"):
            _arm_mounts(request, replace(binding, scene_resident=False), resident)


ARM = """
      <body name="{p}_l1" pos="0 0 0.05">
        <joint name="{p}_j1" axis="0 0 1"/>
        <geom type="capsule" size="0.02 0.05"/>
        <body name="{p}_l2" pos="0 0 0.1">
          <joint name="{p}_j2" axis="0 1 0"/>
          <geom type="capsule" size="0.02 0.05"/>
        </body>
      </body>"""


def _arm(p: str, at: str) -> str:
    return f'<body name="{p}_base" pos="{at}"><geom type="cylinder" size="0.03 0.01"/>{ARM.format(p=p)}</body>'


APART = f"<mujoco><worldbody>{_arm('a', '0 0.2 0')}{_arm('b', '0 -0.2 0')}</worldbody></mujoco>"
"""Two arms, each on a base welded to the world: the YAM's shape."""
SHARED = (
    f'<mujoco><worldbody><body name="torso">{_arm("a", "0 0.2 0")}{_arm("b", "0 -0.2 0")}</body></worldbody></mujoco>'
)
"""The same two arms on one torso."""

MOUNTED = (
    '<mujoco><worldbody><body name="a_base" pos="0 0.2 0"><geom type="cylinder" size="0.03 0.01"/>'
    '<body name="a_mount" pos="0.05 0 0.02" quat="0.9238795 0 0 0.3826834"><geom type="box" size="0.01 0.01 0.01"/>'
    f"{ARM.format(p='a')}</body></body></worldbody></mujoco>"
)
"""One arm on a mount plate welded between its root and its first joint, yawed 45 degrees: the
weld's auto-named joint, ``a_mount_joint``, matches the arm's ``a_`` prefix and comes first."""
LIFTED = (
    '<mujoco><worldbody><body name="stand"><geom type="box" size="0.1 0.1 0.02"/>'
    '<body name="torso" pos="0 0 0.3"><joint name="lift" type="slide" axis="0 0 1"/>'
    '<geom type="box" size="0.05 0.05 0.05"/>'
    f"{_arm('a', '0 0.2 0')}</body></body></worldbody></mujoco>"
)
"""One arm welded to a torso that rides a lift on a stand. On the root itself a lift would be
dropped: the importer welds the root of a fixed-base robot to the world in its joint's place."""

WELDED = APART.replace("</worldbody>", '</worldbody><equality><weld body1="a_l2"/></equality>')
"""The two arms, the left one's wrist welded to the world by an equality."""

WANT = (0.3, 0.4, 0.6, 0.0, 0.0, H, H)


class TestThePlacement(unittest.TestCase):
    def _build(
        self,
        xml: str,
        *,
        collapse: bool,
        mounts: dict,
        base_body: str = "",
        prefix: bool = False,
        reverse: bool = False,
        mount: tuple = (0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
    ):
        import newton

        from robosimstudio.robots.entity import EntityEmbodiment
        from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "two.xml"
        path.write_text(xml)
        arms = tuple(
            ArmSpec(
                name=p,
                joint_names=() if prefix else (f"{p}_j2", f"{p}_j1") if reverse else (f"{p}_j1", f"{p}_j2"),
                joint_prefix=f"{p}_" if prefix else "",
                ee_body_name=f"{p}_l2",
                base_body_name=base_body if p == "a" else "",
            )
            for p in ("a", "b")
            if f'"{p}_j1"' in xml
        )
        spec = EmbodimentSpec(name="two", asset=AssetSpec(path=str(path)), arms=arms, collapse_fixed_joints=collapse)
        builder = newton.ModelBuilder()
        entity = EntityEmbodiment(spec, pose=mount, arm_mounts=mounts)
        entity.build_into(builder, None)
        self._entities = [entity]
        return builder

    @staticmethod
    def _body(builder, name: str) -> np.ndarray:
        from robosimstudio.scene.entity import label_index

        return np.asarray(builder.body_q[label_index(builder.body_label, name)], dtype=np.float64)

    @staticmethod
    def _parent_frame(builder, joint: str) -> np.ndarray:
        from robosimstudio.scene.entity import label_index

        return np.asarray(builder.joint_X_p[label_index(builder.joint_label, joint)], dtype=np.float64)

    def test_the_base_lands_and_the_other_arm_stays(self) -> None:
        builder = self._build(APART, collapse=False, mounts={"a": WANT})
        np.testing.assert_allclose(self._body(builder, "a_base"), WANT, atol=1e-6)
        np.testing.assert_allclose(self._body(builder, "b_base"), (0.1, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0), atol=1e-6)
        # The link rides its base: 50 mm up the base's own z.
        np.testing.assert_allclose(self._body(builder, "a_l1")[:3], (0.3, 0.4, 0.65), atol=1e-6)

    def test_the_collapse_folds_the_moved_weld(self) -> None:
        """Collapsed, the base is the world's; its first joint and its collider carry the pose."""
        import newton

        builder = self._build(APART, collapse=True, mounts={"a": WANT})
        self.assertNotIn("a_base", [label.split("/")[-1] for label in builder.body_label])
        np.testing.assert_allclose(self._parent_frame(builder, "a_j1"), (0.3, 0.4, 0.65, 0.0, 0.0, H, H), atol=1e-6)
        np.testing.assert_allclose(
            self._parent_frame(builder, "b_j1"), (0.1, -0.2, 0.55, 0.0, 0.0, 0.0, 1.0), atol=1e-6
        )
        puck = [
            s
            for s, label in enumerate(builder.shape_label)
            if label.split("/")[-2:-1] == ["a_base"] and int(builder.shape_type[s]) == int(newton.GeoType.CYLINDER)
        ]
        self.assertEqual(1, len(puck))
        np.testing.assert_allclose(np.asarray(builder.shape_transform[puck[0]], dtype=np.float64), WANT, atol=1e-6)

    def test_a_position_keeps_the_heading(self) -> None:
        """On a robot turned a quarter by its mount, a position moves the base and does not turn it."""
        turned = (0.1, 0.0, 0.5, 0.0, 0.0, H, H)
        builder = self._build(APART, collapse=False, mounts={"a": (0.3, 0.4, 0.6)}, mount=turned)
        np.testing.assert_allclose(self._body(builder, "a_base"), (0.3, 0.4, 0.6, 0.0, 0.0, H, H), atol=1e-6)

    def test_the_first_joint_is_the_first_in_the_tree(self) -> None:
        """``joint_names`` listed wrist first still finds the base the arm stands on."""
        builder = self._build(APART, collapse=False, mounts={"a": WANT}, reverse=True)
        np.testing.assert_allclose(self._body(builder, "a_base"), WANT, atol=1e-6)

    def test_an_equality_to_what_stays_is_refused(self) -> None:
        # The importer carries the weld as an equality and as a loop joint; either refusal holds.
        with self.assertRaisesRegex(ValueError, "would hold the arm where the file put it"):
            self._build(WELDED, collapse=False, mounts={"a": WANT})
        # The other arm's placement does not carry the welded wrist, so it is free to move.
        builder = self._build(WELDED, collapse=False, mounts={"b": WANT})
        np.testing.assert_allclose(self._body(builder, "b_base"), WANT, atol=1e-6)

    def test_a_second_joint_to_what_stays_is_refused(self) -> None:
        """A loop-closing joint from the world to the wrist would hold the arm where the file put it."""
        import warp as wp

        from robosimstudio.scene.entity import label_index

        builder = self._build(APART, collapse=False, mounts={})
        entity = self._entities[0]
        wrist = label_index(builder.body_label, "a_l2")
        builder.add_joint_ball(parent=-1, child=wrist, parent_xform=wp.transform_identity(), label="a_tie")
        entity.arm_mounts = {"a": WANT}
        with self.assertRaisesRegex(ValueError, "joint 'a_tie' ties the moved bodies to the world"):
            entity._place_arms(builder, 0)

    def test_a_name_the_spec_does_not_have_is_refused_by_the_entity_too(self) -> None:
        with self.assertRaisesRegex(ValueError, r"places arm\(s\) \['c'\], and its arms are \['a', 'b'\]"):
            self._build(APART, collapse=False, mounts={"c": WANT})

    def test_the_handle_reads_the_body_arms_at_placed(self) -> None:
        """One rule for both: the finalized model's base_body is the body the placement moved."""
        import warp as wp

        from robosimstudio.robots.entity import EntityEmbodiment

        for xml, kwargs in ((MOUNTED, {"prefix": True}), (APART, {"reverse": True})):
            with self.subTest(**kwargs):
                builder = self._build(xml, collapse=False, mounts={"a": WANT}, **kwargs)
                entity = next(e for e in self._entities if e.spec.name == "two")
                model = builder.finalize(device=wp.get_device("cpu"))
                EntityEmbodiment.bind(entity, model)
                state = model.state()
                np.testing.assert_allclose(state.body_q.numpy()[entity.arms["a"].base_body], WANT, atol=1e-6)

    def test_an_arm_sharing_its_root_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"arm 'a' stands on .*torso.* with arm\(s\) \['b'\]"):
            self._build(SHARED, collapse=False, mounts={"a": WANT})

    def test_a_base_on_a_moving_joint_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "a_l1' hangs on the moving joint"):
            self._build(APART, collapse=False, mounts={"a": WANT}, base_body="a_l1")

    def test_a_base_under_a_moving_ancestor_names_the_ancestor(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"a_base' is welded to '[^']*torso', which hangs on the moving joint '[^']*lift'"
        ):
            self._build(LIFTED, collapse=False, mounts={"a": WANT})

    def test_the_base_is_what_the_first_moving_joint_hangs_on(self) -> None:
        """Not the root: a weld between them matches the prefix too, and the plate is the base."""
        for collapse in (False, True):
            with self.subTest(collapse=collapse):
                builder = self._build(MOUNTED, collapse=collapse, mounts={"a": WANT}, prefix=True)
                # Either way the first joint's frame is the plate's, 50 mm up the plate's own z.
                np.testing.assert_allclose(
                    self._parent_frame(builder, "a_j1")[:3] if collapse else self._body(builder, "a_l1")[:3],
                    (0.3, 0.4, 0.65),
                    atol=1e-6,
                )
                if not collapse:
                    np.testing.assert_allclose(self._body(builder, "a_mount"), WANT, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
