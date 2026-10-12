# SPDX-License-Identifier: Apache-2.0
"""One vocabulary for "what sort of thing is this", read under two names.

``BUILD_STAGES`` is the taxonomy; a task's four views -- ``statics``,
``objects``, ``deformables``, ``extras`` -- are the same stages under the names
a task author uses, and exactly one of them is renamed. The table mapping them
was a second literal for as long as it was written out, so what is pinned here
is that it is derived, and what ``restage``'s view keyword actually does.

Static: no scene, no GPU.
"""

from __future__ import annotations

import unittest

from robosimstudio.specs import TaskSpec, restage
from robosimstudio.specs.asset import BUILD_STAGES
from robosimstudio.specs.task import VIEW_OF_STAGE

TIER = "fast"
"""No GPU. Gates every commit."""


class _At:
    """A spec that says only where it is built -- ``build_stage`` duck-types."""

    def __init__(self, stage: str, name: str) -> None:
        self.stage, self.name = stage, name


class TestTheEntityViewsAreTheStagesRenamed(unittest.TestCase):
    def test_every_stage_has_one_view_and_the_rename_is_the_only_difference(self) -> None:
        self.assertEqual(tuple(TaskSpec.STAGE_OF_VIEW.values()), BUILD_STAGES, "views in stage order, one each")
        renamed = {view: stage for view, stage in TaskSpec.STAGE_OF_VIEW.items() if view != stage}
        self.assertEqual(renamed, {view: stage for stage, view in VIEW_OF_STAGE.items()})
        self.assertEqual(renamed, {"statics": "scenery"})

    def test_each_view_reads_the_stage_the_table_names(self) -> None:
        spec = TaskSpec(name="t", entities=tuple(_At(stage, stage[0]) for stage in BUILD_STAGES))
        for view, stage in TaskSpec.STAGE_OF_VIEW.items():
            self.assertEqual([s.name for s in getattr(spec, view)], [stage[0]], view)


class TestRestageReplacesAViewAndBuildsAtTheStage(unittest.TestCase):
    """Measured before it was pinned: the view keyword is not decorative.

    ``deformables=(x,)`` drops the deformables and keeps the extras;
    ``extras=(x,)`` the other way round. Where ``x`` is built comes from its own
    stage either way -- a spec with no stage is an extra whichever keyword handed
    it in.
    """

    def setUp(self) -> None:
        self.base = TaskSpec(
            name="t",
            entities=(_At("scenery", "s"), _At("objects", "o"), _At("deformables", "d"), _At("extras", "e")),
        )

    def test_the_keyword_selects_what_is_replaced(self) -> None:
        x = _At("", "x")
        self.assertEqual([s.name for s in restage(self.base, deformables=(x,)).entities], ["s", "o", "x", "e"])
        self.assertEqual([s.name for s in restage(self.base, extras=(x,)).entities], ["s", "o", "d", "x"])

    def test_the_spec_is_built_at_its_own_stage_not_the_keyword_s(self) -> None:
        x = _At("", "x")
        as_deformable = restage(self.base, deformables=(x,))
        self.assertEqual([s.name for s in as_deformable.extras], ["x", "e"], "no stage means extras")
        self.assertEqual([s.name for s in as_deformable.deformables], [], "and the deformables view is empty")
        self.assertEqual([s.name for s in as_deformable.build_order()], ["s", "o", "x", "e"])

    def test_a_field_change_rides_along_with_a_view_change(self) -> None:
        out = restage(self.base, objects=(), max_steps=7)
        self.assertEqual([s.name for s in out.entities], ["s", "d", "e"])
        self.assertEqual(out.max_steps, 7)

    def test_an_unknown_view_is_refused_rather_than_dropped(self) -> None:
        """``with_entities`` lists the views; a typo would otherwise drop a fixture silently."""
        with self.assertRaises(ValueError) as caught:
            self.base.with_entities(scenery=())
        self.assertIn("statics", str(caught.exception), "the message lists the views that exist")


if __name__ == "__main__":
    unittest.main()
