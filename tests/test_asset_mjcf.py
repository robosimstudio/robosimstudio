# SPDX-License-Identifier: Apache-2.0
"""Every MJCF under ``assets/`` compiles in MuJoCo, and names exactly the meshes beside it.

The robots' files are imported through Newton, which reads what it needs and
can step past a geom whose mesh was never declared; MuJoCo's own compiler does
not. ``yam_i2rt_arena_narrow_arms.xml`` shipped with twelve finger geoms naming
six meshes its ``<asset>`` never declared, and ``yam_bimanual_arms.xml``
declared a 9 MB scenery mesh no geom drew. Both read fine to everything that
loads them, which is why it takes a compiler to see them.

So, per file: it compiles; every mesh it declares is drawn by some geom; and
every mesh file in the embodiment directories is declared by some MJCF or
named by some URDF, so nothing is shipped that nothing loads.

Static: compiles models, steps nothing.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

TIER = "fast"
"""No GPU, no scene. Gates every commit."""

ASSETS = Path(__file__).resolve().parents[1] / "assets"
MESH_SUFFIXES = (".stl", ".obj", ".msh")


def mjcf_files() -> list[Path]:
    return sorted(ASSETS.rglob("*.xml"))


class TestBundledMjcf(unittest.TestCase):
    def test_there_are_files_to_check(self) -> None:
        self.assertTrue(mjcf_files(), f"no MJCF under {ASSETS}")

    def test_every_mjcf_compiles(self) -> None:
        import mujoco

        for path in mjcf_files():
            with self.subTest(path=str(path.relative_to(ASSETS))):
                model = mujoco.MjModel.from_xml_path(str(path))
                self.assertGreater(model.nbody, 1)

    def test_every_declared_mesh_is_drawn(self) -> None:
        import mujoco

        for path in mjcf_files():
            with self.subTest(path=str(path.relative_to(ASSETS))):
                model = mujoco.MjModel.from_xml_path(str(path))
                drawn = {
                    int(model.geom_dataid[g])
                    for g in range(model.ngeom)
                    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                }
                unused = [model.mesh(i).name for i in range(model.nmesh) if i not in drawn]
                self.assertEqual(unused, [], "meshes declared and drawn by no geom")

    def test_every_shipped_mesh_is_loaded_by_something(self) -> None:
        import mujoco

        named: set[Path] = set()
        for path in mjcf_files():
            spec = mujoco.MjSpec.from_file(str(path))
            meshdir = path.parent / (spec.meshdir or "")
            named.update((meshdir / mesh.file).resolve() for mesh in spec.meshes)
        for path in sorted(ASSETS.rglob("*.urdf")):
            for filename in re.findall(r'<mesh\s+filename="([^"]+)"', path.read_text()):
                named.add((path.parent / filename.removeprefix("package://")).resolve())
        shipped = [p for p in sorted(ASSETS.rglob("*")) if p.suffix.lower() in MESH_SUFFIXES]
        orphans = [str(p.relative_to(ASSETS)) for p in shipped if p.resolve() not in named]
        self.assertEqual(orphans, [], "mesh files no MJCF or URDF under assets/ loads")


class TestTheYamHandsGripLikeTheRealOnes(unittest.TestCase):
    """The YAM hands' edits in the checked-in files and their specs' clamp (``assets/PROVENANCE.md``).

    The files' colliders are at friction 1, their fingers tied by a stiff equality and their finger
    actuators capped at the driver's 50 N a jaw -- edits no generator makes, so a regenerated file
    that drops them fails here. The servo is each hand spec's own: stiff enough to reach the cap at
    once, with 230 N*s/m a finger as the closing-speed limit at it.
    """

    ARENA = ("embodiments/yam/yam_bimanual_arms.xml", "embodiments/yam_i2rt_arena/yam_i2rt_arena_narrow_arms.xml")
    I2RT = "embodiments/yam_i2rt/yam_i2rt_arms.xml"
    I2RT_BAKED = "embodiments/yam_i2rt/yam_i2rt_bimanual.xml"

    @staticmethod
    def _tree(path: str):
        import xml.etree.ElementTree as ET

        return ET.parse(ASSETS / path).getroot()

    def test_the_arena_hand_is_at_friction_one_on_stiffly_tied_fingers(self) -> None:
        for path in self.ARENA:
            with self.subTest(path=path):
                root = self._tree(path)
                for name in ("collision", "sphere_collision"):
                    geom = next(d for d in root.iter("default") if d.get("class") == name).find("geom")
                    self.assertEqual(1.0, float(geom.get("friction").split()[0]), name)
                ties = root.find("equality").findall("joint")
                self.assertEqual(2, len(ties))
                self.assertTrue(all(t.get("solref") == "0.004 1" for t in ties), "a stiff equality")
                servo = next(d for d in root.iter("default") if d.get("class") == "finger").find("position")
                self.assertEqual("-100 100", servo.get("forcerange"), "50 N a jaw: the equality halves it")

    def test_the_i2rt_hand_has_its_blades_tied_and_capped_at_50_n(self) -> None:
        for path in (self.I2RT, self.I2RT_BAKED):
            with self.subTest(path=path):
                root = self._tree(path)
                servos = [a for a in root.find("actuator") if re.search(r"joint[78]$", a.get("joint", ""))]
                self.assertEqual(4, len(servos))
                self.assertTrue(all(a.get("forcerange") == "-50 50" for a in servos))
                ties = root.find("equality").findall("joint")
                pairs = {(t.get("joint1"), t.get("joint2"), t.get("polycoef")) for t in ties}
                expected = {("right_joint7", "right_joint8", "0 1 0 0 0"), ("left_joint7", "left_joint8", "0 1 0 0 0")}
                self.assertEqual(expected, pairs)

    def test_each_hand_servo_states_its_assets_cap_and_reaches_it_within_a_millimetre(self) -> None:
        from robosimstudio.robots.grippers.linear_4310 import LINEAR_4310
        from robosimstudio.robots.grippers.yam_linear_arena import YAM_LINEAR_ARENA

        arena = next(d for d in self._tree(self.ARENA[0]).iter("default") if d.get("class") == "finger")
        i2rt = next(a for a in self._tree(self.I2RT).find("actuator") if a.get("joint", "").endswith("joint7"))
        for hand, servo in ((YAM_LINEAR_ARENA, arena.find("position")), (LINEAR_4310, i2rt)):
            with self.subTest(hand=hand.name):
                cap = float(servo.get("forcerange").split()[1])
                clamp = hand.actuation
                self.assertEqual(cap, clamp.effort_limit, "the asset's cap, restated for any solver")
                self.assertGreaterEqual(clamp.target_ke * 0.001, cap, "the cap within a millimetre of contact")
                self.assertEqual(230.0, clamp.joint_damping, "the closing-speed limit")


if __name__ == "__main__":
    unittest.main()
