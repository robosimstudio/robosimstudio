# SPDX-License-Identifier: Apache-2.0
"""The MJCF fixups in ``scene/build/imports`` that need no Newton model to check.

``strip_free_bodies`` is the one this module exists for. It is pure
``ElementTree`` over a file, so it is checkable in milliseconds -- and it was
checked by nothing until it moved into ``core``, at which point it lost two of
its imports and every GPU-tier scene that stages an arena died on a ``NameError``
that 1,583 green fast tests had no way to see. A function whose only coverage is
"some heavy scene happens to call it" is a function with no coverage.

Static: no model, no scene, no GPU.
"""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from robosimstudio.scene.build.imports import strip_free_bodies

TIER = "fast"
"""No GPU. Gates every commit."""


def _scene(body_xml: str) -> Path:
    src = Path(tempfile.mkdtemp()) / "scene.xml"
    src.write_text(f"<mujoco><worldbody>{body_xml}</worldbody></mujoco>")
    return src


class TestStripFreeBodies(unittest.TestCase):
    def test_both_spellings_of_a_free_joint_are_removed(self):
        """MJCF says it two ways, and reading only one fails *open*.

        A check for ``<freejoint/>`` alone returns a copy of the scene it was
        asked to empty: nothing raises, nothing is removed. Every robosuite
        export here uses the attribute form, so the tag form is the one a test
        has to assert on.
        """
        src = _scene(
            '<body name="fixed"><geom size="1"/></body>'
            '<body name="tag"><freejoint/><geom size="1"/></body>'
            '<body name="attribute"><joint type="free"/><geom size="1"/></body>'
        )
        root = ET.parse(strip_free_bodies(src)).getroot()
        kept = [b.get("name") for b in root.find("worldbody").findall("body")]
        self.assertEqual(kept, ["fixed"])

    def test_the_keyframes_go_with_them(self):
        """A keyframe indexes the qpos of the bodies just removed."""
        src = _scene('<body name="loose"><freejoint/><geom size="1"/></body>')
        src.write_text(src.read_text().replace("</worldbody>", '</worldbody><keyframe><key qpos="0 0 0"/></keyframe>'))
        root = ET.parse(strip_free_bodies(src)).getroot()
        self.assertEqual(root.findall("keyframe"), [])

    def test_a_read_only_source_strips_and_still_finds_its_meshes(self):
        """A shared site-packages or a hermetic build installs the assets
        read-only, so the copy cannot go beside its source. It goes in the temp
        root and spells out the source-relative ``meshdir``, so MuJoCo still
        loads the same mesh from it."""
        import mujoco

        root = Path(tempfile.mkdtemp())
        (root / "meshes").mkdir()
        (root / "meshes" / "tet.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        )
        src = root / "scene.xml"
        src.write_text(
            '<mujoco><compiler meshdir="meshes"/><asset><mesh name="tet" file="tet.obj"/></asset>'
            '<worldbody><body name="fixed"><geom type="mesh" mesh="tet"/></body>'
            '<body name="loose"><freejoint/><geom size="1"/></body></worldbody></mujoco>'
        )
        for path in (root / "meshes", root):
            path.chmod(0o555)
        try:
            stripped = Path(strip_free_bodies(src))
        finally:
            for path in (root, root / "meshes"):
                path.chmod(0o755)
        self.assertNotEqual(stripped.parent, root)
        model = mujoco.MjModel.from_xml_path(str(stripped))
        self.assertEqual(model.nmesh, 1)
        self.assertEqual(model.nbody, 2)  # world and the fixed body; the free one is gone


if __name__ == "__main__":
    unittest.main()
