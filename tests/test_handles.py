# SPDX-License-Identifier: Apache-2.0
"""The studio's grab handles: allocation, and what a linked drag does.

No GPU and no browser. What is checked here is the part of direct manipulation
that is pure bookkeeping -- which slot a click takes, which one it recycles when
they are all taken, and the rigid transform that carries the other handles when
one is dragged -- because that is the half that can be silently wrong. Whether
the wrench then lifts the box is physics, and lives in ``tests.test_robot_drag``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_to_mat, quat_xyzw_to_wxyz
from robosimstudio_studio.handles import HandleSet

TIER = "fast"
"""No GPU. Gates every commit."""


class _StubNode:
    """A viser scene node that remembers its callbacks, so a drag can be faked."""

    def __init__(self, name: str, visible: bool) -> None:
        self.name = name
        self.position = (0.0, 0.0, 0.0)
        self.wxyz = (1.0, 0.0, 0.0, 0.0)
        self.visible = bool(visible)
        self.removed = False
        self.callbacks: list = []

    def on_update(self, fn):
        self.callbacks.append(fn)
        return fn

    def remove(self) -> None:
        self.removed = True

    def fire(self, position=None, quat=None, phase: str = "update") -> None:
        """What the browser does: move the node, then tell the server about it."""
        if position is not None:
            self.position = tuple(float(v) for v in position)
        if quat is not None:
            self.wxyz = tuple(float(v) for v in quat_xyzw_to_wxyz(quat))
        for fn in list(self.callbacks):
            fn(SimpleNamespace(target=self, phase=phase))


class _StubScene:
    def __init__(self) -> None:
        self.nodes: list[_StubNode] = []

    def _node(self, name, visible) -> _StubNode:
        node = _StubNode(name, visible)
        self.nodes.append(node)
        return node

    def add_transform_controls(self, name, **kwargs):
        return self._node(name, kwargs.get("visible", True))

    def add_icosphere(self, name, **kwargs):
        return self._node(name, kwargs.get("visible", True))


def _set(handles: HandleSet, count: int = 2, **kwargs) -> HandleSet:
    del handles
    return HandleSet(_StubScene(), "/test/handle", count=count, **kwargs)


def _drag(handle, position=None, quat=None) -> None:
    """One gesture: grab where it is, move, let go."""
    handle.gizmo.fire(phase="start")
    handle.gizmo.fire(position=position, quat=quat, phase="update")


class TestAllocation(unittest.TestCase):
    """Which slot a click gets, which is what makes two handles usable."""

    def test_a_placed_handle_is_visible_and_a_cleared_one_is_not(self):
        """Visibility is the readout: a gizmo on screen is one holding something."""
        handles = _set(None)
        slot = handles.acquire()
        self.assertFalse(slot.gizmo.visible, "a slot is invisible until it is placed")
        slot.place([0.1, 0.2, 0.3], label="towel")
        self.assertTrue(slot.live)
        self.assertTrue(slot.gizmo.visible)
        self.assertTrue(slot.marker.visible, "the bead that names the slot must come with it")
        np.testing.assert_allclose(slot.gizmo.position, [0.1, 0.2, 0.3])
        slot.clear()
        self.assertFalse(slot.live)
        self.assertFalse(slot.gizmo.visible)
        self.assertFalse(slot.marker.visible)

    def test_free_slots_come_first(self):
        """Two clicks give two handles rather than moving one."""
        handles = _set(None, count=2)
        first = handles.acquire()
        first.place([0.0, 0.0, 0.0])
        second = handles.acquire()
        self.assertIsNot(second, first, "the second click reused the first handle")
        second.place([0.5, 0.0, 0.0])
        self.assertEqual(len(handles.live), 2)

    def test_a_full_set_recycles_the_oldest(self):
        """A third click works. Refusing it would be the button dance again."""
        handles = _set(None, count=2)
        first, second = handles.acquire(), handles.acquire()
        first.place([0.0, 0.0, 0.0], label="a")
        second.place([1.0, 0.0, 0.0], label="b")
        third = handles.acquire()
        self.assertIs(third, first, "a full set must give up its oldest handle, not its newest")

    def test_a_key_re_seats_its_own_handle(self):
        """Clicking a body twice moves its handle instead of adding a second wrench."""
        handles = _set(None, count=2)
        first = handles.acquire(key=7)
        first.place([0.0, 0.0, 0.0], key=7)
        again = handles.acquire(key=7)
        self.assertIs(again, first)
        self.assertEqual(len(handles.live), 1)

    def test_newest_tracks_placement_order(self):
        """Release last has to mean the one you just put down."""
        handles = _set(None, count=2)
        first, second = handles.acquire(), handles.acquire()
        first.place([0.0, 0.0, 0.0])
        second.place([1.0, 0.0, 0.0])
        self.assertIs(handles.newest(), second)
        second.clear()
        self.assertIs(handles.newest(), first)
        first.clear()
        self.assertIsNone(handles.newest(), "an empty set has no newest handle to release")

    def test_release_all_and_close(self):
        """A task swap has to take the previous scene's handles down with it."""
        handles = _set(None, count=2)
        for _ in range(2):
            handles.acquire().place([0.0, 0.0, 0.0])
        handles.release_all()
        self.assertEqual(handles.live, [])
        nodes = [n for h in handles for n in (h.gizmo, h.marker)]
        handles.close()
        self.assertTrue(all(n.removed for n in nodes), "closing left scene nodes behind")


class TestLinkedDrag(unittest.TestCase):
    """Moving two handles with one mouse, which is the whole point of two.

    The gate is that the followers move by the *rigid transform* the dragged
    handle underwent. Anything less -- translation only, or a per-update delta
    that drifts -- looks right on a slow straight pull and is wrong the moment
    the gesture rotates.
    """

    def _pair(self, linked: bool):
        handles = _set(None, count=2, linked=linked)
        first, second = handles.acquire(), handles.acquire()
        first.place([0.0, 0.0, 0.0])
        second.place([0.2, 0.0, 0.0])
        return handles, first, second

    def test_unlinked_leaves_the_other_where_it_was(self):
        """One grip pulling is a pinch, and has to stay one."""
        handles, first, second = self._pair(linked=False)
        _drag(first, position=[0.0, 0.0, 0.3])
        handles.drain()
        np.testing.assert_allclose(second.position, [0.2, 0.0, 0.0])
        np.testing.assert_allclose(first.position, [0.0, 0.0, 0.3])

    def test_linked_translation_carries_the_other(self):
        """Both corners rise together, so the sheet is lifted rather than tented."""
        handles, first, second = self._pair(linked=True)
        _drag(first, position=[0.0, 0.0, 0.3])
        handles.drain()
        np.testing.assert_allclose(second.position, [0.2, 0.0, 0.3], atol=1e-12)
        np.testing.assert_allclose(second.gizmo.position, [0.2, 0.0, 0.3], atol=1e-12)

    def test_linked_rotation_swings_the_other_around(self):
        """A twist about the dragged handle carries the pair as one body.

        Rotating each handle in place instead would leave the separation
        pointing the way it started, which on a two-corner hold means the sheet
        is asked to twist without turning.
        """
        handles, first, second = self._pair(linked=True)
        quarter = quat_from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0)
        _drag(first, quat=quarter)
        handles.drain()
        np.testing.assert_allclose(second.position, [0.0, 0.2, 0.0], atol=1e-9)
        np.testing.assert_allclose(quat_to_mat(second.quat), quat_to_mat(quarter), atol=1e-9)
        self.assertAlmostEqual(
            float(np.linalg.norm(second.position - first.position)), 0.2, places=9, msg="the pair was stretched"
        )

    def test_the_dragged_handle_is_never_written_back(self):
        """The browser owns the pose it is dragging; writing it fights the client."""
        handles, first, second = self._pair(linked=True)
        del second
        _drag(first, position=[0.0, 0.0, 0.3])
        first.gizmo.position = (99.0, 99.0, 99.0)  # what the client would show next
        handles.drain()
        self.assertEqual(first.gizmo.position, (99.0, 99.0, 99.0))

    def test_a_finished_drag_stops_carrying(self):
        """Ending the gesture ends the link, so the next drain is not a second pull."""
        handles, first, second = self._pair(linked=True)
        _drag(first, position=[0.0, 0.0, 0.3])
        handles.drain()
        first.gizmo.fire(phase="end")
        second.position = np.array([0.2, 0.0, 0.3])
        handles.drain()
        np.testing.assert_allclose(second.position, [0.2, 0.0, 0.3], atol=1e-12)

    def test_a_long_drag_does_not_accumulate(self):
        """Every update is measured from the anchor, so drops cannot compound it."""
        handles, first, second = self._pair(linked=True)
        first.gizmo.fire(phase="start")
        for step in range(1, 6):
            first.gizmo.fire(position=[0.0, 0.0, 0.05 * step], phase="update")
            handles.drain()
        np.testing.assert_allclose(second.position, [0.2, 0.0, 0.25], atol=1e-12)

    def test_a_handle_placed_mid_drag_is_not_dragged_with_it(self):
        """A handle that was not in the anchor set is not part of the gesture."""
        handles = _set(None, count=2, linked=True)
        first = handles.acquire()
        first.place([0.0, 0.0, 0.0])
        first.gizmo.fire(phase="start")
        second = handles.acquire()
        second.place([0.2, 0.0, 0.0])
        first.gizmo.fire(position=[0.0, 0.0, 0.3], phase="update")
        handles.drain()
        np.testing.assert_allclose(second.position, [0.2, 0.0, 0.0])

    def test_summary_names_the_live_slots(self):
        """The readout is how you tell which bead is holding what."""
        handles = _set(None, count=2)
        self.assertEqual(handles.summary("nothing"), "nothing")
        handles.acquire().place([0.0, 0.0, 0.0], label="towel (12)")
        self.assertEqual(handles.summary(), "1: towel (12)")


if __name__ == "__main__":
    unittest.main()


class TestSceneryThatIsNotEntityStatic(unittest.TestCase):
    """The statics view is membership by domain, not by build stage.

    ``view_of`` asks whether the entity's domain declares ``world_fixed``, where
    the old bin asked whether its spec declared ``stage = "scenery"``. Those
    agree for everything shipped and part for a novel kind: a welded thing with
    its own entity had no stage, so it fell to ``extras`` and was invisible;
    it is classified correctly now, which means the handle loop reaches it --
    and the loop gathered ``entity.meta``, which is ``EntityStatic``'s JSON
    sidecar and not something ``Entity`` declares.

    So the classification was right and the consequence was a crash, for exactly
    the kind of thing the derived views exist to stop being second-class.
    """

    TIER = "fast"

    def test_a_novel_welded_kind_is_scenery(self) -> None:
        from robosimstudio.scene.compiler import view_of
        from robosimstudio.scene.entity import Entity
        from robosimstudio.specs.physics import register_domain

        register_domain(
            "probe_plinth",
            state=("shapes",),
            reads="rigid",
            role="fixtures",
            world_fixed=True,
            summary="A novel welded kind, declared here.",
        )

        class _Spec:
            name, entity, sim = "plinth", "probe_plinth", None

        class NovelScenery(Entity):
            domain = "probe_plinth"

        made = NovelScenery(_Spec())
        self.assertEqual("statics", view_of(made))
        self.assertFalse(hasattr(made, "meta"), "`meta` is EntityStatic's, so the loop must not assume it")
