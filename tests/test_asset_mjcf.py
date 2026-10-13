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


if __name__ == "__main__":
    unittest.main()
