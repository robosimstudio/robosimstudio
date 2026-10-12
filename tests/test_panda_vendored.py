# SPDX-License-Identifier: Apache-2.0
"""The Panda ships with the library, whole, and has not drifted from its source.

``panda`` and ``panda_robotiq`` used to import their arm out of the VABAR
collection's tool-sort export -- a file the library read by path and did not
ship, so ``rss.robot("panda")`` built in a checkout and failed to resolve
anywhere the collection was not installed. The two MJCFs and the meshes they
name are vendored under ``assets/embodiments/panda/`` now, as byte copies.

A copy is a second place for a file to change, so this holds four things still:
the specs point at the vendored files; every mesh they name is there; nothing
else is (40 MB is enough); and while the collection's originals are on disk the
copies are byte-identical to them.

Static: reads files, builds nothing.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from robosimstudio import paths as asset_paths
from robosimstudio.robots.registry import get_embodiment

TIER = "fast"
"""No GPU, no scene. Gates every commit."""

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "assets" / "embodiments" / "panda"
COLLECTION = REPO / "vabar" / "assets" / "vab_tool_sort"
"""Where the files were copied from. Absent from an install without the collection."""

MJCFS = {
    "panda": "panda_standalone.xml",
    "panda_robotiq": "panda_nohand_standalone.xml",
    "panda_robotiq_roll": "panda_nohand_standalone.xml",
}
"""Which file each registered Panda imports."""


def _meshes(mjcf: Path) -> set[str]:
    """Every ``file=`` an MJCF names, relative to its own directory (``meshdir="."``)."""
    return set(re.findall(r'\bfile="([^"]+)"', mjcf.read_text()))


class TestThePandaIsTheLibrarysOwn(unittest.TestCase):
    def test_each_spec_names_the_vendored_file(self) -> None:
        for name, filename in MJCFS.items():
            with self.subTest(embodiment=name):
                spec = get_embodiment(name)
                self.assertEqual(f"embodiments/panda/{filename}", spec.asset.path)
                self.assertEqual(VENDORED / filename, asset_paths.resolve_asset(spec.asset))

    def test_it_resolves_with_no_collection_searched(self) -> None:
        """The claim the move makes: the checkout's own ``assets/`` is enough.

        Pointed at that one root, exactly as ``ROBOSIMSTUDIO_ASSETS`` points a
        deployment, so a collection's ``assets/`` beside it cannot answer.
        """
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {asset_paths._ENV_VAR: str(REPO / "assets")}):
            self.assertEqual([REPO / "assets"], asset_paths._candidate_roots())
            for filename in sorted(set(MJCFS.values())):
                found = asset_paths.resolve_path(f"embodiments/panda/{filename}")
                for mesh in sorted(_meshes(found)):
                    self.assertTrue((found.parent / mesh).is_file(), f"{filename} names {mesh}, which is not vendored")

    def test_the_mjcfs_resolve_their_meshes_beside_themselves(self) -> None:
        """Why the copies needed no edit, and what would silently break one: the
        compiler says ``meshdir="."`` and every mesh is ``assets/<file>``."""
        for filename in sorted(set(MJCFS.values())):
            with self.subTest(mjcf=filename):
                text = (VENDORED / filename).read_text()
                self.assertIn('meshdir="."', text)
                self.assertTrue(all(mesh.startswith("assets/") for mesh in _meshes(VENDORED / filename)))

    def test_exactly_the_named_meshes_are_vendored(self) -> None:
        named = set().union(*(_meshes(VENDORED / filename) for filename in set(MJCFS.values())))
        shipped = {str(path.relative_to(VENDORED)) for path in (VENDORED / "assets").iterdir()}
        self.assertEqual(sorted(named), sorted(shipped), "a mesh is missing, or one nothing names came along")
        self.assertEqual(61, len(named))

    def test_it_says_where_it_came_from(self) -> None:
        """A directory that can be handed to somebody: geometry, licence, attribution."""
        self.assertIn("MIT License", (VENDORED / "LICENSE").read_text())
        note = (VENDORED / "README.md").read_text()
        for word in ("robosuite", "franka_ros", "Apache-2.0"):
            self.assertIn(word, note)


@unittest.skipUnless(COLLECTION.is_dir(), "the VABAR collection is not in this checkout")
class TestTheCopyHasNotDrifted(unittest.TestCase):
    """The collection keeps the originals, and its own scenes read them there."""

    def test_every_vendored_file_is_a_byte_copy(self) -> None:
        files = [VENDORED / name for name in sorted(set(MJCFS.values()))]
        files += sorted((VENDORED / "assets").iterdir())
        for path in files:
            with self.subTest(file=path.name):
                original = COLLECTION / path.relative_to(VENDORED)
                self.assertTrue(original.is_file(), f"{original} is gone; the copy has no source to be checked against")
                self.assertEqual(
                    original.read_bytes(),
                    path.read_bytes(),
                    f"{path.relative_to(REPO)} differs from {original.relative_to(REPO)}; copy the regenerated file over",
                )


if __name__ == "__main__":
    unittest.main()
