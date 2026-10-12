# SPDX-License-Identifier: Apache-2.0
"""Placement scoring: the operand form, and the smoothing that makes it usable.

Three mechanisms meet here, and each was added for this measure:

* an operand read as a ``handle`` rather than a position, because a contact test
  needs shapes and velocities and a body index together;
* an atom that ``reads_scene``, because the contact set belongs to the scene and
  not to either object;
* a declared ``smoothing`` window, because the instrument is noisy in a specific,
  measured way.

The smoothing tests are the load-bearing ones. Without the window a settled item
reads zero force on 6-10% of steps and a five-item goal flickers on about 28% of
them; with it, the goal is stable. Both directions are checked -- it must reject
the dropouts *and* stop counting an item genuinely removed.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomSpec, get_atom, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.scene.handles import EntityHandle, SceneHandles
from robosimstudio_bench.goals.bind import BindError, Operand, bind, read
from robosimstudio_bench.taskdoc.ast import Condition, Instance, Ref, TaskDoc
from tests.fixtures.snapshots import ctx

TIER = "fast"
"""No GPU. Gates every commit."""


class TestTheAtomsAreDeclaredWell(unittest.TestCase):
    """What the mint gate checks, on the three atoms this module added."""

    def test_they_validate(self) -> None:
        for name in ("contact_force", "speed", "height"):
            with self.subTest(atom=name):
                self.assertEqual(get_atom(name).problems(), [])

    def test_arenas_predicate_is_two_atoms_and_not_one(self) -> None:
        """Arena answers one boolean from two readings. A single ``resting_on``
        atom would have to threshold one and carry the other, which makes the
        name a lie about half of what it does -- and when a run fails, "not
        resting on it" is not a diagnosis where "0.4 N, wanted 1.0" and "moving
        at 0.3 m/s" are two different faults with two different fixes."""
        self.assertEqual(get_atom("contact_force").returns, ("force",))
        self.assertEqual(get_atom("speed").returns, ("speed",))

    def test_each_operand_is_read_in_the_form_its_measure_needs(self) -> None:
        """The contact measure needs shapes, velocities and a body index
        together, so it takes handles and looks them up. A speed is a property of
        one body and arrives as one."""
        self.assertEqual([a.reads for a in get_atom("contact_force").args], ["handle", "handle"])
        self.assertEqual([a.reads for a in get_atom("speed").args], ["velocity"])
        self.assertEqual([a.reads for a in get_atom("height").args], ["position"])

    def test_only_the_contact_measure_reads_the_scene(self) -> None:
        """Declared rather than inferred from the signature, so that "this
        measure needs the world" is a decision a reviewer can see -- and the line
        is where it should be: a contact belongs to the solver's contact set, a
        speed and a height belong to their body."""
        self.assertTrue(get_atom("contact_force").reads_scene)
        self.assertFalse(get_atom("speed").reads_scene)
        self.assertFalse(get_atom("height").reads_scene)

    def test_only_the_contact_measure_is_smoothed(self) -> None:
        """A position and a velocity are read straight off the state and do not
        churn. Smoothing them would add lag for nothing."""
        self.assertEqual(get_atom("contact_force").smoothing, 10)
        self.assertEqual(get_atom("speed").smoothing, 0)
        self.assertEqual(get_atom("height").smoothing, 0)

    def test_neither_arena_measure_states_a_gate(self) -> None:
        """Arena uses 0.1 N for a single-object place and 1.0 N for the
        multi-object sort. A gate baked in here would silently make one of its
        two task shapes measure the other's, so the task states the number."""
        for name in ("contact_force", "speed"):
            with self.subTest(atom=name):
                calibration = get_atom(name).calibration
                self.assertIsNotNone(calibration)
                self.assertIsNone(calibration.gate)
                self.assertTrue(calibration.positives and calibration.negatives)


class TestSmoothingRejectsContactChurn(unittest.TestCase):
    """The measured failure this window exists for.

    A settled item's contact with a tote floor is not continuous: the solver's
    contact set churns and the pair drops out for up to six consecutive steps
    while nothing has moved. A goal sampling one instant is a coin flip.
    """

    NAME = "test_churny_force"

    def setUp(self) -> None:
        if not _registered(self.NAME):
            register_atom(
                AtomSpec(
                    name=self.NAME,
                    summary="A scripted force reading, for testing the window.",
                    args=(ArgSpec(name="thing", reads="handle"),),
                    directions=("check",),
                    returns=("force",),
                    reads_scene=True,
                    smoothing=10,
                )
            )

            @handler(self.NAME, kinds=("",), substrates=("rigid",), summary="reads off a script")
            def _scripted(snapshot, thing: str) -> dict:
                del thing
                return {"force": float(snapshot.ctx.extras["script"].pop(0))}

        self.doc = TaskDoc(name="t", objects=(Instance(name="can", type="a_can"),))
        self.record = AssetRecord(name="a_can", summary="s")
        self.leaf = bind(
            Condition(head=self.NAME, args=(Ref(base="can"),), op=">=", value=1.0),
            self.doc,
            lambda n: self.record if n == "a_can" else None,
        )

    def _run(self, script: list[float]) -> list[bool]:
        state = {"script": list(script)}
        return [bool(self.leaf(ctx(extras=state)).success) for _ in script]

    def test_a_dropout_does_not_flip_the_verdict(self) -> None:
        """Six consecutive zeros in the middle of a settled contact -- the
        longest run measured -- and the goal must hold throughout."""
        script = [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0]
        self.assertEqual(self._run(script), [True] * 10)

    def test_an_item_genuinely_removed_stops_counting(self) -> None:
        """The other direction, and the reason the window is a maximum over a
        *sliding* window rather than a latch. Ten steps at 50 Hz is 0.2 s, which
        is a real delay and a small one."""
        script = [3.0] + [0.0] * 12
        verdicts = self._run(script)
        self.assertTrue(verdicts[0])
        self.assertTrue(verdicts[9], "still inside the window")
        self.assertFalse(verdicts[11], "the reading has left the window")

    def test_the_instantaneous_reading_is_still_recorded(self) -> None:
        """A run scored on a smoothed value must still record the raw one, or
        nobody can tell a smoothed pass from a genuine one."""
        state = {"script": [3.0, 0.0]}
        self.leaf(ctx(extras=state))
        result = self.leaf(ctx(extras=state))
        self.assertEqual(result.metrics["force.instant"], 0.0)
        self.assertEqual(result.metrics["force"], 3.0)

    def test_the_window_clears_between_episodes(self) -> None:
        """Otherwise the first steps of episode 2 are scored on the last steps of
        episode 1, which reads as a task that starts already satisfied and then
        settles back down -- and it would only show up in a multi-episode run."""
        state = {"script": [5.0, 0.0]}
        self.leaf(ctx(extras=state))
        self.leaf.reset(ctx())
        self.assertFalse(self.leaf(ctx(extras=state)).success, "the 5.0 N reading did not survive the reset")


class TestTheOperandFormIsHonoured(unittest.TestCase):
    """Which form a handler gets is the atom's declaration, not the object's."""

    def test_a_handle_form_yields_the_scene_name(self) -> None:
        got = read(Snapshot(ctx=ctx()), Operand(ref=Ref(base="bin"), substrate="rigid"), "handle")
        self.assertEqual(got, "bin")

    def test_a_form_nothing_reads_is_refused_by_name(self) -> None:
        """Rather than falling back to a position -- a handler expecting collider
        indices and handed three floats does not fail, it computes something."""
        with self.assertRaises(BindError) as caught:
            read(Snapshot(ctx=ctx()), Operand(ref=Ref(base="x"), substrate="rigid"), "shapes")
        self.assertIn("shapes", str(caught.exception))

    def test_the_same_object_reads_two_ways(self) -> None:
        """The whole point of putting the form on the argument. ``containment``
        and ``resting_on`` both accept a rigid body; one wants a point and the
        other wants to look the entity up itself -- so the two go to different
        readers and only one of them touches the scene."""
        handles = SceneHandles(objects={"can": EntityHandle(name="can", body=0)})
        snapshot = Snapshot(ctx=ctx(handles=handles, body_q=np.array([[1.0, 2.0, 3.0, 0, 0, 0, 1]])))
        operand = Operand(ref=Ref(base="can"), substrate="rigid")
        self.assertEqual(read(snapshot, operand, "handle"), "can")
        np.testing.assert_allclose(read(snapshot, operand, "position"), [1.0, 2.0, 3.0])


def _registered(name: str) -> bool:
    try:
        get_atom(name)
    except KeyError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
