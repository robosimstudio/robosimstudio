# SPDX-License-Identifier: Apache-2.0
"""Reading what an entity publishes about itself.

An asset's record declares *which* states exist and what each means. Until this
existed, nothing said where the numbers came from: every consumer reached into
the entity with ``getattr`` and its own idea of the attribute's name, which is a
convention rather than a contract -- it holds right up until two suites spell one
thing differently.

That became load-bearing with a goal language. A document saying
``(<= (mate-gap assembly) 0.006)`` has to resolve ``mate-gap`` against
*something*, and "whatever attribute the entity happens to expose" is not an
answer a compiler can check.

The case that forced it: an assembly that builds its own plug and socket as
internal bodies. Neither is an object a document can name, so neither can be
measured from outside -- and a task should not have to know that.
"""

from __future__ import annotations

import unittest

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.scene.entity import Entity
from robosimstudio_bench.goals.bind import bind
from robosimstudio_bench.taskdoc.ast import Condition, Instance, Ref, TaskDoc
from tests.fixtures.snapshots import ctx

TIER = "fast"
"""No GPU. Gates every commit."""


class _Assembly(Entity):
    """An entity whose state is computed, not stored."""

    def __init__(self, gap: float) -> None:
        self.gap = gap

    @property
    def name(self) -> str:
        return "assembly"

    def published_states(self, snapshot) -> dict[str, float]:
        del snapshot
        return {"mate_gap": self.gap}


class _Silent(Entity):
    """One that publishes nothing, which is the default and the common case."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "silent"


class TestTheDefaultIsSilence(unittest.TestCase):
    def test_an_entity_publishes_nothing_unless_it_says_so(self) -> None:
        """An entity that computes a quantity nobody declared is not publishing
        it. Most entities publish nothing at all."""
        self.assertEqual(_Silent().published_states(None), {})


class TestAGoalReadsThroughIt(unittest.TestCase):
    """The whole point: a clause naming a state the entity computes."""

    def setUp(self) -> None:
        self.record = AssetRecord(
            name="an_assembly",
            summary="a plug and a socket that are not separately nameable",
            states=(
                StateSpec(
                    name="mate_gap",
                    measure="mate_gap",
                    op="<=",
                    value=0.006,
                    summary="travel remaining",
                    provenance="fixture",
                ),
            ),
        )
        self.doc = TaskDoc(name="t", objects=(Instance(name="assembly", type="an_assembly"),))

    def _leaf(self, **clause):
        return bind(
            Condition(head="mate-gap", args=(Ref(base="assembly"),), **clause),
            self.doc,
            lambda n: self.record if n == "an_assembly" else None,
        )

    def test_a_computed_state_resolves(self) -> None:
        leaf = self._leaf(op="<=", value=0.006)
        self.assertTrue(leaf(ctx(extras={"assembly": _Assembly(0.003)})).success)
        self.assertFalse(leaf(ctx(extras={"assembly": _Assembly(0.02)})).success)

    def test_the_hyphenated_name_matches_the_python_one(self) -> None:
        """BDDL writes ``mate-gap`` and Python cannot. Matching one spelling only
        is how a goal reports the single quantity its task is scored on as
        unresolved."""
        self.assertIn("mate_gap", self._leaf(op="<=", value=1.0)(ctx(extras={"assembly": _Assembly(0.0)})).metrics)

    def test_a_plain_mapping_still_works(self) -> None:
        """The structural half of the binder is meant to be testable without
        building an entity -- and an entity is exactly the thing that needs a
        scene. So a dict is accepted where an entity would go."""
        leaf = self._leaf(op="<=", value=0.006)
        self.assertTrue(leaf(ctx(extras={"assembly": {"mate_gap": 0.001}})).success)

    def test_a_declared_state_the_entity_never_publishes_is_loud(self) -> None:
        """Rather than scoring zero for a whole run, which looks exactly like a
        policy that never achieved anything."""
        from robosimstudio_bench.goals.bind import BindError

        leaf = self._leaf(op="<=", value=0.006)
        with self.assertRaises(BindError) as caught:
            leaf(ctx(extras={"assembly": _Silent()}))
        self.assertIn("mate_gap", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
