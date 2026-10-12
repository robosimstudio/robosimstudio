# SPDX-License-Identifier: Apache-2.0
"""The task-level schedule: its order, its refusals, and who may join it.

The claim worth testing is not that a list holds ten strings. It is that a phase
is reachable by *anything* that implements it -- the restriction this schedule
exists to remove was that six of these phases could only be reached by
subclassing the entity base class, which left a success condition, a camera rig
and a metric structurally unable to run at compile time at all.

Static: a schedule is ordering. No model, no scene, no GPU.
"""

from __future__ import annotations

import unittest

from robosimstudio.core.lifecycle import PHASE_METHODS, TASK_PHASES, Step, TaskSchedule, steps_of

TIER = "fast"
"""No GPU. Gates every commit."""


class TestThePhaseOrder(unittest.TestCase):
    def test_the_order_is_the_one_the_compiler_depends_on(self):
        """Build order is load-bearing -- the compiler's own docstring says doing
        a step later or earlier "fails, usually silently". Pinned here so the
        tuple cannot be reordered by tidying."""
        self.assertEqual(
            TASK_PHASES,
            ("stage", "build", "bind", "materials", "kernels", "ready", "reset", "observe", "score", "close"),
        )

    def test_materials_sits_between_a_finalized_model_and_the_solvers(self):
        """The one window there is: after ``bind`` has a model to resolve
        against, before ``kernels`` hands anything to the solvers that copy its
        arrays."""
        order = list(TASK_PHASES)
        self.assertLess(order.index("bind"), order.index("materials"))
        self.assertLess(order.index("materials"), order.index("kernels"))

    def test_every_discovered_method_names_a_real_phase(self):
        for phase in PHASE_METHODS:
            self.assertIn(phase, TASK_PHASES)


class TestAnythingMayJoinAPhase(unittest.TestCase):
    """The restriction this exists to remove."""

    def test_a_part_that_is_not_an_entity_reaches_the_phases_it_implements(self):
        """A success condition owning no geometry: it could not previously run
        at ``ready`` at all, because the only caller looped over entities."""

        class Success:
            name = "goal"

            def on_scene_ready(self, pipeline):
                return "settled"

            def reset(self, pipeline, episode):
                return "rearmed"

        schedule = steps_of(Success())
        self.assertEqual([phase for phase, _ in schedule.names()], ["ready", "reset"])
        self.assertEqual(schedule.steps("ready")[0].fn(None), "settled")

    def test_a_part_implementing_nothing_costs_nothing(self):
        schedule = steps_of(object())
        self.assertFalse(schedule)
        self.assertEqual(len(schedule), 0)

    def test_steps_are_named_for_what_contributed_them(self):
        class Rope:
            name = "rope"

            def bind(self, model):
                pass

        self.assertEqual(steps_of(Rope()).names(), (("bind", "rope.bind"),))

    def test_a_part_with_no_name_falls_back_to_its_type(self):
        class Anon:
            def bind(self, model):
                pass

        self.assertEqual(steps_of(Anon()).names(), (("bind", "Anon.bind"),))


class TestTheScheduleRefuses(unittest.TestCase):
    def test_an_unknown_phase_is_refused_at_construction(self):
        """Not at run time, when the step would simply never fire."""
        with self.assertRaises(ValueError):
            Step(name="x", fn=lambda: None, phase="settle")

    def test_a_cadence_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            Step(name="x", fn=lambda: None, phase="observe", cadence=0)

    def test_asking_for_a_phase_that_does_not_exist_raises(self):
        with self.assertRaises(KeyError):
            TaskSchedule().steps("settle")


class TestOrderWithinAPhase(unittest.TestCase):
    def test_contribution_order_is_preserved(self):
        """Inside a phase the order is the order parts arrived, which is the
        build order the compiler already fixed -- scenery before the fixtures
        standing on it."""

        class Part:
            def __init__(self, name):
                self.name = name

            def bind(self, model):
                pass

        schedule = TaskSchedule()
        for part in (Part("bench"), Part("peg"), Part("rope")):
            schedule.extend(steps_of(part))
        self.assertEqual([s.name for s in schedule.steps("bind")], ["bench.bind", "peg.bind", "rope.bind"])


if __name__ == "__main__":
    unittest.main()
