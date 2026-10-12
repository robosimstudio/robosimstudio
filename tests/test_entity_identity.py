# SPDX-License-Identifier: Apache-2.0
"""Two entity classes are "the same" by *name*, and that has a cost.

``ENTITIES`` accepts a second registration under a taken name when the two
classes look like one class shipped twice -- which is the case that has to keep
working: a task directory is self-contained on purpose, so two of them staging
the same kind of thing each carry their own copy, and refusing the second would
force the shared directory the layout is doing without.

``same_source`` decides that by comparing ``(module basename, qualname)``, and
the basename is what makes it survive a package rename. **The cost is that two
genuinely different classes that happen to agree on both are indistinguishable,
and the second one silently loses.** The refactor plan's risk register lists this
and notes that nothing tests it. This is that test.

It does not argue the rule should change. Identity would refuse the copy the
layout depends on, and a full dotted path would break on the rename the basename
exists to survive. What it pins is that the hazard is **real and bounded**: it
needs two classes agreeing on leaf module *and* qualname *and* registry key, and
when that happens the first registration is what every caller gets.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""Constructs classes; touches no model."""


def _class_named(module: str, qualname: str) -> type:
    """A fresh class wearing somebody else's module and qualname."""
    made = type(qualname, (), {})
    made.__module__ = module
    made.__qualname__ = qualname
    return made


class TestTheCopyRuleIsWhatItSaysItIs(unittest.TestCase):
    def test_a_copy_under_the_same_name_is_accepted(self) -> None:
        """The case the rule exists for: two tasks, one class, shipped twice."""
        from robosimstudio.registry import Registry, same_source

        registry: Registry[type] = Registry("entity", same=same_source)
        first = _class_named("tasks.rgbs.assets.peel.tape", "EntityTape")
        second = _class_named("tasks.film.assets.tape", "EntityTape")
        registry.register("tape", first)
        self.assertIs(
            first, registry.register("tape", second), "the registered object must be handed back, not the twin"
        )

    def test_a_package_rename_does_not_break_the_copy_rule(self) -> None:
        """Why the *basename* and not the dotted path. ``rstask.peel.x`` and
        ``tasks.peel.x`` are the same module under two roots."""
        from robosimstudio.registry import same_source

        self.assertTrue(
            same_source(
                _class_named("rstask.peel.assets.tape", "EntityTape"),
                _class_named("tasks.rgbs.assets.peel.tape", "EntityTape"),
            )
        )

    def test_two_different_classes_that_collide_are_not_told_apart(self) -> None:
        """The hazard, stated as a fact rather than left as a comment.

        Two unrelated entities, in different packages, that happen to share a
        leaf module name *and* a class name, registered under one key: the second
        is accepted as a copy and every caller gets the first. Nothing raises,
        and the scene builds the wrong thing.
        """
        from robosimstudio.registry import Registry, same_source

        registry: Registry[type] = Registry("entity", same=same_source)
        mine = _class_named("tasks.mine.assets.entity", "Widget")
        theirs = _class_named("tasks.theirs.assets.entity", "Widget")
        self.assertIsNot(mine, theirs)
        registry.register("widget", mine)
        got = registry.register("widget", theirs)
        self.assertIs(mine, got, "the collision is real: the second registration is silently discarded")

    def test_a_different_qualname_is_refused(self) -> None:
        """The bound on the hazard in one direction: the names must agree."""
        from robosimstudio.registry import Registry, same_source

        registry: Registry[type] = Registry("entity", same=same_source)
        registry.register("widget", _class_named("tasks.a.assets.entity", "Widget"))
        with self.assertRaises(ValueError):
            registry.register("widget", _class_named("tasks.b.assets.entity", "Gadget"))

    def test_a_different_leaf_module_is_refused(self) -> None:
        """And in the other. A collision needs both halves to agree."""
        from robosimstudio.registry import Registry, same_source

        registry: Registry[type] = Registry("entity", same=same_source)
        registry.register("widget", _class_named("tasks.a.assets.entity", "Widget"))
        with self.assertRaises(ValueError):
            registry.register("widget", _class_named("tasks.b.assets.parts", "Widget"))


class TestTheShippedEntitiesDoNotCollide(unittest.TestCase):
    def test_no_two_registered_entities_share_a_basename_and_qualname(self) -> None:
        """The corpus is clear of it today, which is the thing worth knowing.

        Aliases are excluded by identity: ``rod``, ``cable_rod`` and ``hang_line``
        are three keys onto one class, and one class cannot collide with itself.
        """
        from robosimstudio.scene.registry import ENTITIES

        seen: dict[tuple[str, str], type] = {}
        collisions: list[str] = []
        for name, cls in ENTITIES:
            key = (cls.__module__.rpartition(".")[2], cls.__qualname__)
            first = seen.setdefault(key, cls)
            if first is not cls:
                collisions.append(f"{name}: {cls.__module__}.{cls.__qualname__} collides with {first.__module__}")
        self.assertEqual([], collisions, "two different entity classes are indistinguishable to same_source")


if __name__ == "__main__":
    unittest.main()
