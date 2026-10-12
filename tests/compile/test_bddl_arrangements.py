# SPDX-License-Identifier: Apache-2.0
"""Named physics arrangements and camera rigs, and how a document reaches one.

A document stages both by name and says nothing about what either is -- which is
the point, since a solver's contact stiffnesses are a calibrated fact about a
setup rather than something a task is in a position to state. What was missing
was somewhere for the name to resolve *to*: it indexed a dict the caller passed
in, so a shipped arrangement was rebuilt by every loader that staged it, and a
name nobody supplied was a build failure rather than a lookup.

The refusals are the same ones import profiles make, for the same reasons, and
that is deliberate rather than copied: an arrangement revised in place restages
every task that named it without changing a single document.
"""

from __future__ import annotations

import unittest

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.specs import ArrangementSpec
from robosimstudio_bench.taskbuild import arrangements
from robosimstudio_bench.taskbuild.lower import LowerError, lower
from robosimstudio_bench.taskdoc.ast import Instance, TaskDoc

TIER = "fast"
"""No GPU, no scene. Gates every commit."""


def document(**staging) -> TaskDoc:
    """The smallest document that stages something.

    Built rather than parsed, as the rest of this directory builds them: what is
    under test is what lowering does with a staged name, and going through the
    parser would put its own refusals between the assertion and the thing
    asserted.
    """
    return TaskDoc(name="t", objects=(Instance(name="x"),), staging=staging)


class _Table:
    """A registry table borrowed for one test and put back.

    The module tables are process-global and the fast tier runs in one process,
    so a test that registered into them would be visible to every test after it
    -- including the ones asserting that an unregistered name is refused.
    """

    def __init__(self, table) -> None:
        self.table = table  # a Registry; its items are the state borrowed

    def __enter__(self):
        self.saved = dict(self.table._items)
        return self.table

    def __exit__(self, *_exc) -> None:
        self.table._items.clear()
        self.table._items.update(self.saved)


class TestRegistration(unittest.TestCase):
    def test_an_unversioned_name_is_refused(self) -> None:
        """The refusal a profile makes, for the reason a profile makes it: an
        arrangement revised in place restages every task that named it, and no
        document changes to say so."""
        with _Table(arrangements.PHYSICS), self.assertRaises(ValueError) as caught:
            arrangements.register_arrangement("unversioned", ArrangementSpec())
        self.assertIn("carries no version", str(caught.exception))

    def test_a_different_arrangement_under_a_taken_name_is_refused(self) -> None:
        """Otherwise whichever module imported second would decide what every
        task staging that name actually ran under."""
        with _Table(arrangements.PHYSICS):
            arrangements.register_arrangement("x@v1", ArrangementSpec())
            with self.assertRaises(ValueError):
                arrangements.register_arrangement("x@v1", ArrangementSpec(control_hz=17.0))

    def test_an_identical_copy_is_allowed(self) -> None:
        """Two tasks each shipping the cell they stand in each declare its
        arrangement, and both say the same thing.

        Refusing this is what would force a shared directory, which is the thing
        a self-contained task directory is trying to do without. What the rule is
        actually for -- two *different* arrangements claiming one name -- is
        unchanged and tested above.
        """
        with _Table(arrangements.PHYSICS):
            first = arrangements.register_arrangement("x@v1", ArrangementSpec())
            again = arrangements.register_arrangement("x@v1", ArrangementSpec())
            self.assertIs(first, again, "the second copy must not replace the object others already hold")

    def test_camera_rigs_are_held_to_the_same_rules(self) -> None:
        with _Table(arrangements.CAMERA_RIGS):
            with self.assertRaises(ValueError):
                arrangements.register_camera_rig("unversioned_rig", ())
            arrangements.register_camera_rig("probe_rig@v1", (CameraSpec(name="front"),))
            with self.assertRaises(ValueError):
                arrangements.register_camera_rig("probe_rig@v1", ())

    def test_a_missing_name_says_what_is_registered(self) -> None:
        with _Table(arrangements.PHYSICS), self.assertRaises(KeyError) as caught:
            arrangements.get_arrangement("nobody@v1")
        self.assertIn("registered", str(caught.exception))
        self.assertIn("nobody@v1", str(caught.exception))


class TestLoweringResolvesThroughIt(unittest.TestCase):
    def test_a_registered_arrangement_needs_no_loader(self) -> None:
        """The whole point: a document staging a shipped arrangement builds with
        nothing supplied beside it."""
        with _Table(arrangements.PHYSICS):
            arrangements.register_arrangement("probe_only@v1", ArrangementSpec(control_hz=90.0))
            spec = lower(document(physics="probe_only@v1")).spec
            self.assertAlmostEqual(spec.physics.control_hz, 90.0)

    def test_a_supplied_table_wins_over_the_registry(self) -> None:
        """The injection seam stays, and stays first. A loader supplies the one
        thing a versioned registration cannot hold -- a rig aimed at a layout the
        document has no coordinates for -- and must be able to override."""
        with _Table(arrangements.PHYSICS):
            arrangements.register_arrangement("probe_only@v1", ArrangementSpec(control_hz=90.0))
            supplied = {"probe_only@v1": ArrangementSpec(control_hz=30.0)}
            spec = lower(document(physics="probe_only@v1"), physics=supplied).spec
            self.assertAlmostEqual(spec.physics.control_hz, 30.0)

    def test_a_name_in_neither_place_is_refused_and_says_so(self) -> None:
        """The message has to separate the two failures it could be: a loader
        that forgot to supply one, and a corpus that was never imported."""
        with _Table(arrangements.PHYSICS), self.assertRaises(LowerError) as caught:
            lower(document(physics="nobody@ref"))
        message = str(caught.exception)
        self.assertIn("neither supplied nor registered", message)
        self.assertIn("Registered:", message)

    def test_staging_nothing_takes_the_defaults(self) -> None:
        """A task may legitimately want default physics and no cameras."""
        spec = lower(document(scene="bench_075")).spec
        self.assertEqual(spec.cameras, ())
        self.assertEqual(spec.physics, ArrangementSpec())

    def test_a_registered_rig_resolves_too(self) -> None:
        with _Table(arrangements.CAMERA_RIGS):
            arrangements.register_camera_rig("two_view@v1", (CameraSpec(name="front"), CameraSpec(name="side")))
            spec = lower(document(cameras="two_view@v1")).spec
            self.assertEqual([c.name for c in spec.cameras], ["front", "side"])


if __name__ == "__main__":
    unittest.main()
