# SPDX-License-Identifier: Apache-2.0
"""``rss.sheet(held_by=...)``: a sheet a body holds from the start, as a spec and a registered entity.

No GPU: these are the seam's rules as data. No shipped example or demo holds a sheet this way
any more: the fold demo's hand holds its towel by friction (``tests/test_demo_fold_the_towel.py``).
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""No GPU. Specs and the entity registry only."""


def _edge(rest):
    """A selector: the strip along the sheet's far edge."""
    return rest[:, 1] > rest[:, 1].max() - 0.012


class TestTheHeldSheetSpec(unittest.TestCase):
    def test_held_by_is_what_makes_a_sheet_held(self) -> None:
        from robosimstudio.domains.cloth import HeldClothSpec

        import robosimstudio as rss

        free = rss.sheet(name="towel")
        self.assertNotIsInstance(free, HeldClothSpec)
        self.assertEqual("cloth", free.entity)
        held = rss.sheet(name="towel", held_by="left_link_6", particle_sets=(("held", _edge),))
        self.assertIsInstance(held, HeldClothSpec)
        self.assertIsInstance(held, rss.ClothSpec, "still a sheet, to everything that reads one")
        self.assertEqual(("held_cloth", "left_link_6"), (held.entity, held.held_by))

    def test_a_held_sheet_names_its_holder_and_its_patch(self) -> None:
        from robosimstudio.domains.cloth import HeldClothSpec

        with self.assertRaisesRegex(ValueError, "no 'held' particle set"):
            HeldClothSpec(name="towel", held_by="left_link_6", particle_sets=(("edge", _edge),))
        with self.assertRaisesRegex(ValueError, "held by nothing"):
            HeldClothSpec(name="towel", particle_sets=(("held", _edge),))


class TestOnlyAHeldSheetAddsAKernel(unittest.TestCase):
    """A substep kernel written for one world is what refuses a replicated build, so a free sheet has none."""

    def test_the_held_entity_is_registered_and_the_free_one_stays_hookless(self) -> None:
        from robosimstudio.domains.cloth import EntityCloth, EntityHeldCloth

        from robosimstudio.scene.entity import Entity
        from robosimstudio.scene.registry import get_entity

        self.assertIs(EntityHeldCloth, get_entity("held_cloth"))
        self.assertTrue(issubclass(EntityHeldCloth, EntityCloth))
        self.assertIs(Entity.contribute_hooks, EntityCloth.contribute_hooks, "a free sheet can still be replicated")
        self.assertIsNot(Entity.contribute_hooks, EntityHeldCloth.contribute_hooks)


if __name__ == "__main__":
    unittest.main()
