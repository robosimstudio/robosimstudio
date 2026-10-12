# SPDX-License-Identifier: Apache-2.0
"""One search over the named handle tables, and the two readers that share it.

``SceneHandles`` keeps a table per kind -- objects, statics, deformables, regions,
arms -- because its consumers read them by kind: every object's pose for an
observation, every deformable's mesh for a renderer, every region for the
overlay, an arm by name to drive it. What did not need to exist twice was the
walk over four of those tables for a bare name, which ``Ref.handle`` and
``Sim.__getitem__`` each carried a copy of. :meth:`SceneHandles.find` is that
walk, written once, and this pins what it answers.

Structural: stub handles, no scene, no GPU.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from robosimstudio.scene.handles import ArmHandle, DeformableHandle, EntityHandle, RegionHandle, SceneHandles
from robosimstudio.scene.ref import Ref

TIER = "fast"
"""No GPU. Gates every commit."""


def _handles() -> SceneHandles:
    return SceneHandles(
        objects={"hammer": EntityHandle(name="hammer", body=1)},
        statics={"bench": EntityHandle(name="bench", static=True)},
        deformables={"towel": DeformableHandle(name="towel")},
        regions={"bin": RegionHandle(name="bin", frame="bench")},
        arms={"right": ArmHandle(name="right")},
    )


class TestFindIsTheOneSearch(unittest.TestCase):
    def test_a_name_in_any_named_table_is_found_as_the_object_in_that_table(self) -> None:
        handles = _handles()
        for name, table in (
            ("hammer", handles.objects),
            ("bench", handles.statics),
            ("towel", handles.deformables),
            ("bin", handles.regions),
        ):
            with self.subTest(name=name):
                self.assertIs(handles.find(name), table[name])

    def test_the_kind_is_on_the_handle(self) -> None:
        """What the four classes are for: which table it came from is what you got back."""
        handles = _handles()
        self.assertIsInstance(handles.find("hammer"), EntityHandle)
        self.assertFalse(handles.find("hammer").static)
        self.assertTrue(handles.find("bench").static)
        self.assertIsInstance(handles.find("towel"), DeformableHandle)
        self.assertIsInstance(handles.find("bin"), RegionHandle)

    def test_an_unknown_name_is_none_not_an_error(self) -> None:
        """Both readers branch on ``None``: one returns it, the other declines to make a ``Ref``."""
        self.assertIsNone(_handles().find("nothing"))

    def test_an_arm_is_not_found_by_a_bare_name(self) -> None:
        """An arm is addressed by role through ``arm()``; neither reader ever resolved one."""
        handles = _handles()
        self.assertIsNone(handles.find("right"))
        self.assertIs(handles.arm("right"), handles.arms["right"])


class TestRefHandleReadsThroughFind(unittest.TestCase):
    def test_ref_handle_is_what_find_returns(self) -> None:
        handles = _handles()
        scene = SimpleNamespace(handles=handles, by_name={})
        for name in ("hammer", "bench", "towel", "bin"):
            with self.subTest(name=name):
                self.assertIs(Ref(scene, name).handle, handles.find(name))
        self.assertIsNone(Ref(scene, "nothing").handle)


if __name__ == "__main__":
    unittest.main()
