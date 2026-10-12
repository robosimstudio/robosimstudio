# SPDX-License-Identifier: Apache-2.0
"""Turning a click into a thing, and holding onto which thing it was.

The page had nowhere for "what you are looking at" to live. A click resolved a
ray to a `ShapeHit`, discarded the shape index, and used only `.body`; two panels
each kept a private `selected` string, and nothing else could ask. So the tests
here are about the lookup that did not exist and the one ambiguity it turns up.

Structural, against stub handles: what is under test is which subject an index
answers with, and that is a pure function of the handles a compiled scene already
carries. Building a scene to ask it would put this in a tier that gates nothing.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from robosimstudio.scene.handles import ArmHandle, DeformableHandle, EntityHandle, RegionHandle, SceneHandles
from robosimstudio_studio.selection import KINDS, SceneIndex, Subject

TIER = "fast"
"""No GPU, no scene. Gates every commit."""


def _handles(**kwargs) -> SceneHandles:
    return SceneHandles(**kwargs)


def _doc(**types_by_name) -> object:
    return types.SimpleNamespace(objects=[types.SimpleNamespace(name=n, type=t) for n, t in types_by_name.items()])


class TestEverythingInASceneIsASubject(unittest.TestCase):
    """Objects, deformables, arms, scenery and the boxes a goal is measured
    against. The old page could name the first and, sometimes, the third."""

    def setUp(self) -> None:
        self.index = SceneIndex(
            _handles(
                objects={"hammer_0": EntityHandle(name="hammer_0", body=1, shapes=np.array([10, 11]))},
                deformables={"towel": DeformableHandle(name="towel", particles=np.array([0, 1, 2]))},
                arms={"right": ArmHandle(name="right", shapes=np.array([20]))},
                statics={"bench": EntityHandle(name="bench", static=True, shapes=np.array([30]))},
                regions={"bin2": RegionHandle(name="bin2", frame="bin2")},
            ),
            doc=_doc(hammer_0="hammer", towel="towel"),
        )

    def test_all_five_kinds_are_present(self) -> None:
        self.assertEqual({s.kind for s in self.index.subjects}, set(KINDS))

    def test_they_are_ordered_movable_first(self) -> None:
        """Somebody opening a page came to click on a thing, not on the bench."""
        self.assertEqual([s.kind for s in self.index.subjects], list(KINDS))

    def test_the_first_subject_is_what_a_page_opens_on(self) -> None:
        self.assertEqual(self.index.first().name, "hammer_0")

    def test_a_subject_is_typed_through_the_document(self) -> None:
        self.assertEqual(self.index.by_name("hammer_0").asset, "hammer")

    def test_something_the_document_does_not_type_has_no_asset(self) -> None:
        """A legitimate state, not a failure -- the same degradation `describe`
        makes for a scene nobody has written a task for."""
        self.assertEqual(self.index.by_name("bench").asset, "")

    def test_the_summary_counts_what_is_there(self) -> None:
        self.assertEqual(self.index.summary(), "1 object · 1 cloth · 1 arm · 1 static · 1 region")


class TestAClickFindsItsOwner(unittest.TestCase):
    """The reverse lookup. `EntityHandle.shapes` has always been there; going
    from a shape back to what owns it was open-coded twice as `{body: name}`
    dicts covering only the movable rigid bodies."""

    def setUp(self) -> None:
        self.index = SceneIndex(
            _handles(
                objects={"hammer_0": EntityHandle(name="hammer_0", body=1, shapes=np.array([10, 11]))},
                deformables={"towel": DeformableHandle(name="towel", particles=np.array([5, 6, 7]))},
                statics={"bench": EntityHandle(name="bench", static=True, shapes=np.array([30, 31]))},
            )
        )

    def test_a_shape_resolves_to_its_object(self) -> None:
        self.assertEqual(self.index.at_shape(11).name, "hammer_0")

    def test_scenery_is_selectable_now(self) -> None:
        """The bench used to report a bare label and nothing else."""
        self.assertEqual(self.index.at_shape(30).name, "bench")

    def test_a_particle_resolves_to_its_cloth(self) -> None:
        self.assertEqual(self.index.at_particle(6).name, "towel")

    def test_a_body_resolves_when_a_shape_is_unknown(self) -> None:
        self.assertEqual(self.index.at_body(1).name, "hammer_0")

    def test_an_unknown_shape_answers_nothing_rather_than_guessing(self) -> None:
        self.assertIsNone(self.index.at_shape(999))
        self.assertIsNone(self.index.at_particle(999))


class TestBothArmsClaimTheWholeRobot(unittest.TestCase):
    """Measured, and the reason the tiebreak exists.

    `ArmHandle.shapes` is filled for contact *exclusion*, where "everything
    belonging to this arm" sensibly means the whole robot -- on `yam_bimanual`
    the left and right lists are the same 112 shapes. Read as identity that makes
    every click on a link ambiguous, and breaking the tie alphabetically would
    select the left arm on one robot and the right on another, which is the exact
    class of bug `SceneHandles.work_arm` was introduced for.
    """

    def _index(self, work: str) -> SceneIndex:
        robot = np.array([40, 41, 42])
        return SceneIndex(
            _handles(
                arms={"left": ArmHandle(name="left", shapes=robot), "right": ArmHandle(name="right", shapes=robot)},
                work_arm=work,
            )
        )

    def test_an_ambiguous_link_resolves_to_the_arm_doing_the_task(self) -> None:
        self.assertEqual(self._index("right").at_shape(41).name, "right")

    def test_and_follows_the_binding_rather_than_the_alphabet(self) -> None:
        """The same scene with the roles swapped answers differently, which is
        the whole point -- alphabetically both would say `left`."""
        self.assertEqual(self._index("left").at_shape(41).name, "left")

    def test_a_scene_with_no_declared_work_arm_still_answers(self) -> None:
        self.assertIsNotNone(self._index("").at_shape(41))


class TestAnObjectBeatsWhatItStandsOn(unittest.TestCase):
    def test_a_shape_two_subjects_claim_goes_to_the_movable_one(self) -> None:
        """Somebody clicking a tool resting in a bin means the tool."""
        index = SceneIndex(
            _handles(
                objects={"tool": EntityHandle(name="tool", shapes=np.array([7]))},
                statics={"bench": EntityHandle(name="bench", static=True, shapes=np.array([7]))},
            )
        )
        self.assertEqual(index.at_shape(7).name, "tool")


class TestASuiteChainIsASubject(unittest.TestCase):
    """The cable, and whatever the next suite carries as a many-bodied thing.

    A chain had **no** handle: it is built as a suite-owned extra whose segments
    the scene never names, so ``_build_handles`` had no row for it and the one
    movable thing in a cable scene could not be clicked while every peg around
    it could. The page answered that by synthesising a parallel handle type off
    the live entity, for the one bin the compiler skipped.

    The compiler builds it now -- ``EntityRod.handle`` returns an
    ``EntityHandle`` whose ``body`` is the first segment and whose ``bodies`` is
    the run -- so a chain is an ordinary object and the synthesised type is gone.
    These pin what that buys, which is what they always pinned: a subject, a
    claim on *every* segment body, and the front seat a movable thing gets.
    """

    def _chain(self) -> EntityHandle:
        return EntityHandle(name="cable", body=5, bodies=(5, 6, 7), shapes=np.array([40, 41, 42]))

    def _index(self, **handles) -> SceneIndex:
        handles.setdefault("objects", {"cable": self._chain()})
        return SceneIndex(_handles(**handles), scene=types.SimpleNamespace(by_name={}))

    def test_the_chain_is_an_object_subject(self) -> None:
        subject = self._index().by_name("cable")
        self.assertIsNotNone(subject)
        self.assertEqual(subject.kind, "object")

    def test_every_segment_body_resolves_to_it(self) -> None:
        """The reason ``bodies`` exists beside ``body``: a click lands on a
        segment, not on the first one."""
        index = self._index()
        for body in (5, 6, 7):
            self.assertEqual(index.at_body(body).name, "cable", body)

    def test_every_segment_shape_resolves_to_it(self) -> None:
        index = self._index()
        for shape in (40, 41, 42):
            self.assertEqual(index.at_shape(shape).name, "cable", shape)

    def test_a_chain_scene_opens_on_the_chain(self) -> None:
        """A cable task has no plain objects at all; the page used to open on
        an arm because the one movable thing was not a subject."""
        index = self._index(arms={"right": ArmHandle(name="right", shapes=np.array([20]))})
        self.assertEqual(index.first().name, "cable")

    def test_a_thing_with_no_bodies_publishes_no_handle(self) -> None:
        """The tape's film, an assembly's virtual state: something owning no
        bodies offers no handle, so it stays reachable through the inspector
        rather than the click map. ``EntityRod.handle`` returns ``None`` for it
        rather than a handle onto nothing."""
        index = SceneIndex(_handles(), scene=types.SimpleNamespace(by_name={}))
        self.assertIsNone(index.by_name("film"))


class TestASubjectSurvivesAReset(unittest.TestCase):
    def test_its_key_is_kind_and_name_rather_than_an_index(self) -> None:
        """Indices move when a scene recompiles and a name does not. Re-selecting
        by index after a reload is how a page shows one object's numbers under
        another one's heading."""
        self.assertEqual(Subject(kind="object", name="hammer_0").key, "object:hammer_0")

    def test_a_key_round_trips_through_the_index(self) -> None:
        index = SceneIndex(_handles(objects={"hammer_0": EntityHandle(name="hammer_0", body=1)}))
        self.assertEqual(index.by_key("object:hammer_0").name, "hammer_0")

    def test_a_label_says_what_kind_of_thing_it_is(self) -> None:
        self.assertEqual(Subject(kind="cloth", name="towel").label, "cloth · towel")


class TestAnEmptySceneIsNotAnError(unittest.TestCase):
    def setUp(self) -> None:
        self.index = SceneIndex(_handles())

    def test_it_has_no_subjects(self) -> None:
        self.assertEqual(len(self.index), 0)

    def test_it_opens_on_nothing(self) -> None:
        self.assertIsNone(self.index.first())

    def test_it_says_so(self) -> None:
        self.assertEqual(self.index.summary(), "nothing")


if __name__ == "__main__":
    unittest.main()
