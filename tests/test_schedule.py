# SPDX-License-Identifier: Apache-2.0
"""The substep program: one ordering, a cadence, and a list of stages you can edit.

:mod:`~robosimstudio.core.schedule` is what replaced three frozen substep
orderings and three fixed hook slots. It knows nothing about solvers -- the
composer puts something in ``solve`` and this runs it -- so everything here can
be checked on the CPU against callables that only write down that they ran.
That is the whole reason this file is fast: a scheduler tested only through a
scene is a scheduler tested only where a GPU is.

**The list surface is the half that bites.** ``pipeline.hooks.<phase>`` was a
plain Python list for as long as the pipeline had fixed slots, and callers
learned it: the studio's direct manipulation appends a dragger's ``apply`` on
click and removes it when you let go. :class:`_PhaseList` replaced that list and
shipped without ``remove``, which no fast test noticed and no library caller
uses -- it is reached only through the studio, whose tests live in the slow
tier. The failure was not the missing method either. When removal raised, the
dragger *stayed attached*, and three tests later a rod that should have been
resting had been dragged 119 mm. So the list methods are pinned here, one at a
time, where they cost a second to check.
"""

from __future__ import annotations

import unittest

import newton
import warp as wp

from robosimstudio.core.schedule import (
    HOOK_PHASES,
    PHASES,
    SUBSTEP_PHASES,
    Schedule,
    ScheduleExecutor,
    Stage,
    SubstepHooks,
    _PhaseList,
)

TIER = "fast"
"""CPU only: the stages are plain callables and no solver is attached."""


def _model() -> newton.Model:
    """One body on a free joint, finalized on the CPU.

    The executor needs a model only for its device, its states and its control;
    nothing here steps physics, and attaching a solver would put a solver's
    behaviour inside assertions about ordering.
    """
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    return builder.finalize(device="cpu")


def _recorder(log: list, label: str):
    """A stage that writes down that it ran."""

    def run(_state, _dt) -> None:
        log.append(label)

    return run


class TestAStageIsCheckedWhereItIsMade(unittest.TestCase):
    """A stage names a phase and a cadence, and both can be typos.

    Checked in ``__post_init__`` rather than at run time because the run time is
    inside a CUDA graph capture: a phase nobody dispatches would be a stage that
    silently never runs, which reads as physics rather than as a mistake.
    """

    def test_a_phase_that_is_not_one_is_refused_and_lists_them(self) -> None:
        with self.assertRaises(ValueError) as caught:
            Stage(name="x", fn=lambda s, dt: None, phase="presolve")
        self.assertIn("presolve", str(caught.exception))
        self.assertIn("pre_solve", str(caught.exception), "the message must show the spelling that works")

    def test_a_cadence_below_one_is_refused(self) -> None:
        """Zero would be a division by zero in the modulo; negative never fires."""
        for bad in (0, -1):
            with self.subTest(cadence=bad), self.assertRaises(ValueError):
                Stage(name="x", fn=lambda s, dt: None, cadence=bad)

    def test_the_phases_are_in_the_order_a_substep_runs_them(self) -> None:
        """``PHASES`` is documentation and dispatch order at once, so it is pinned.

        ``before`` and ``after`` bracket the control step; the seven between are
        one substep, and ``solve`` is followed by the state swap.
        """
        self.assertEqual(
            PHASES,
            ("before", "clear", "rigid_force", "pre_solve", "collide", "solve", "post_solve", "mirror", "after"),
        )


class TestThePhaseTuplesAreOneTable(unittest.TestCase):
    """Three tuples said what a phase is; one table says it now, and these pin the three.

    ``PHASES`` (nine), ``SUBSTEP_PHASES`` (seven) and ``SubstepHooks.PHASES``
    (six) were each a literal, and an entity author writing ``contribute_hooks``
    had to work out from three lists which one to read. The literals here are
    what those three were the day they were derived, so a change to the table
    that moves a phase between views fails by name.
    """

    def test_the_substep_phases_are_what_the_literal_was(self) -> None:
        self.assertEqual(
            SUBSTEP_PHASES,
            ("clear", "rigid_force", "pre_solve", "collide", "solve", "post_solve", "mirror"),
        )

    def test_the_hook_phases_are_what_the_literal_was(self) -> None:
        self.assertEqual(HOOK_PHASES, ("before", "rigid_force", "pre_solve", "post_solve", "mirror", "after"))
        self.assertIs(SubstepHooks.PHASES, HOOK_PHASES, "the class attribute readers use is the derived tuple")

    def test_both_subsets_keep_the_order_of_the_whole(self) -> None:
        """A view of the ordering is in that ordering; nothing here re-sorts."""
        for subset in (SUBSTEP_PHASES, HOOK_PHASES):
            self.assertEqual(tuple(p for p in PHASES if p in subset), subset)

    def test_the_two_columns_say_why_a_phase_is_left_out(self) -> None:
        """``before``/``after`` bracket the control step; the built-in three are the executor's."""
        self.assertEqual(set(PHASES) - set(SUBSTEP_PHASES), {"before", "after"})
        self.assertEqual(set(PHASES) - set(HOOK_PHASES), {"clear", "collide", "solve"})

    def test_a_hook_list_exists_for_every_hookable_phase_and_no_other(self) -> None:
        hooks = SubstepHooks()
        for phase in HOOK_PHASES:
            self.assertIsInstance(getattr(hooks, phase), _PhaseList, phase)
        for phase in set(PHASES) - set(HOOK_PHASES):
            self.assertFalse(hasattr(hooks, phase), f"{phase} is built in and must not offer a list")


class TestThePhaseListIsListShaped(unittest.TestCase):
    """Every list method a caller learned when this was a list.

    Not an abstract completeness argument: each of these is reached from the
    studio, and the one that was missing cost a slow-tier suite.
    """

    def setUp(self) -> None:
        self.schedule = Schedule()
        self.phase = _PhaseList(self.schedule, "rigid_force")

    def test_append_extend_and_length(self) -> None:
        a, b, c = (_recorder([], name) for name in "abc")
        self.phase.append(a)
        self.phase.extend([b, c])
        self.assertEqual(len(self.phase), 3)
        self.assertEqual(list(self.phase), [a, b, c])
        self.assertTrue(self.phase)

    def test_an_empty_phase_is_falsey(self) -> None:
        self.assertFalse(self.phase)
        self.assertEqual(list(self.phase), [])

    def test_indexing_and_slicing_yield_the_callables_not_the_stages(self) -> None:
        """A caller holding this list wants what it put in, not the wrapper."""
        a, b = _recorder([], "a"), _recorder([], "b")
        self.phase.extend([a, b])
        self.assertIs(self.phase[0], a)
        self.assertEqual(self.phase[:], [a, b])

    def test_deleting_a_slice_is_how_the_env_unwinds_an_episode(self) -> None:
        """``core/env.py`` trims back to a floor with ``del hooks[floor:]``."""
        self.phase.extend([_recorder([], str(i)) for i in range(4)])
        del self.phase[2:]
        self.assertEqual(len(self.phase), 2)

    def test_remove_takes_a_bound_method_back_out(self) -> None:
        """The studio's drag, which is the case that was broken.

        A bound method is a new object on every attribute access, so this is
        also the assertion that ``remove`` cannot be identity-only.
        """

        class Dragger:
            def apply(self, _state, _dt) -> None:
                pass

        one, two = Dragger(), Dragger()
        self.phase.append(one.apply)
        self.phase.append(two.apply)
        self.assertIsNot(one.apply, one.apply, "the premise: each access makes a new object")

        self.phase.remove(one.apply)
        self.assertEqual(len(self.phase), 1)
        self.assertNotIn(one.apply, self.phase)
        self.assertIn(two.apply, self.phase, "removing one dragger must not take the other's hook")

    def test_remove_takes_the_first_of_two_identical_hooks(self) -> None:
        """``list.remove``'s own rule, kept: two clicks appended two stages."""
        fn = _recorder([], "a")
        self.phase.extend([fn, fn])
        self.phase.remove(fn)
        self.assertEqual(len(self.phase), 1)

    def test_removing_something_that_is_not_there_raises(self) -> None:
        """A silent miss leaves the hook running and the caller believing it let go."""
        with self.assertRaises(ValueError):
            self.phase.remove(_recorder([], "never added"))

    def test_contains_answers_for_a_bound_method_too(self) -> None:
        class Dragger:
            def apply(self, _state, _dt) -> None:
                pass

        dragger = Dragger()
        self.assertNotIn(dragger.apply, self.phase)
        self.phase.append(dragger.apply)
        self.assertIn(dragger.apply, self.phase)


class TestTheSubstepBlockIsOneOrdering(unittest.TestCase):
    """What used to be three branches is one sequence, asserted as a sequence."""

    def _executor(self, schedule: Schedule, **kwargs) -> ScheduleExecutor:
        # setdefault, not a literal: a caller overriding ``substeps`` would
        # otherwise pass it twice and get a TypeError, which an
        # ``assertRaises`` around the constructor would happily accept as the
        # refusal it was looking for. That is what this helper did first.
        for key, value in (("control_hz", 50.0), ("substeps", 2), ("use_cuda_graph", False)):
            kwargs.setdefault(key, value)
        return ScheduleExecutor(_model(), schedule, **kwargs)

    def test_every_substep_phase_runs_once_per_substep_in_order(self) -> None:
        log: list[str] = []
        schedule = Schedule()
        substep_phases = [p for p in PHASES if p not in ("before", "after")]
        for phase in substep_phases:
            schedule.add(Stage(name=phase, fn=_recorder(log, phase), phase=phase))

        self._executor(schedule).substep_block()
        self.assertEqual(log, substep_phases * 2, "two substeps, each running the seven phases in order")

    def test_before_and_after_bracket_the_control_step_not_the_substep(self) -> None:
        """They are per control step, which is why a read-back lives in ``after``."""
        log: list[str] = []
        schedule = Schedule()
        schedule.add(Stage(name="before", fn=_recorder(log, "before"), phase="before"))
        schedule.add(Stage(name="mirror", fn=_recorder(log, "mirror"), phase="mirror"))
        schedule.add(Stage(name="after", fn=_recorder(log, "after"), phase="after"))

        self._executor(schedule).step()
        self.assertEqual(log, ["before", "mirror", "mirror", "after"])

    def test_the_solve_swaps_buffers_within_a_substep(self) -> None:
        """``post_solve`` sees the buffer the solve wrote, not the one it read.

        Asserted by watching the state object each phase is handed, because the
        obvious check -- that ``state_0`` is back where it started after two
        substeps -- passes just as well if the swap never happened at all.
        """
        seen: dict[str, list[int]] = {"pre_solve": [], "post_solve": []}

        def watch(phase: str):
            def run(state, _dt) -> None:
                seen[phase].append(id(state))

            return run

        schedule = Schedule()
        for phase in ("pre_solve", "post_solve"):
            schedule.add(Stage(name=phase, fn=watch(phase), phase=phase))

        self._executor(schedule).substep_block()
        for index in range(2):
            with self.subTest(substep=index):
                self.assertNotEqual(
                    seen["pre_solve"][index],
                    seen["post_solve"][index],
                    "post_solve must run against the other buffer",
                )
        self.assertEqual(seen["pre_solve"][0], seen["post_solve"][1], "and the two alternate")

    def test_two_substeps_leave_the_frame_in_the_buffer_the_caller_reads(self) -> None:
        """Which is what makes an even count a requirement rather than a habit."""
        executor = self._executor(Schedule())
        first, second = executor.state_0, executor.state_1
        executor.substep_block()
        self.assertIs(executor.state_0, first, "two swaps put it back")
        self.assertIs(executor.state_1, second)

    def test_an_odd_substep_count_is_refused(self) -> None:
        """Otherwise the frame's result is left in the *other* buffer, and inside
        a captured graph the swap is baked in -- so the scene appears to run at
        half rate on stale state."""
        with self.assertRaises(ValueError) as caught:
            self._executor(Schedule(), substeps=3)
        self.assertIn("even", str(caught.exception))


class TestCadence(unittest.TestCase):
    """``cadence`` is how a stage costs less than every substep."""

    def test_a_stage_runs_every_nth_substep_starting_at_the_first(self) -> None:
        log: list[str] = []
        schedule = Schedule()
        schedule.add(Stage(name="every", fn=_recorder(log, "every"), phase="post_solve"))
        schedule.add(Stage(name="third", fn=_recorder(log, "third"), phase="post_solve", cadence=3))

        ScheduleExecutor(_model(), schedule, control_hz=50.0, substeps=6, use_cuda_graph=False).substep_block()
        self.assertEqual(log.count("every"), 6)
        self.assertEqual(log.count("third"), 2, "substeps 0 and 3 of six")
        self.assertEqual(log[:2], ["every", "third"], "the first substep runs both")

    def test_a_control_phase_counts_control_steps(self) -> None:
        """``before``/``after`` are indexed by ``step_count``, not by substep."""
        log: list[str] = []
        schedule = Schedule()
        schedule.add(Stage(name="other", fn=_recorder(log, "other"), phase="after", cadence=2))
        executor = ScheduleExecutor(_model(), schedule, control_hz=50.0, substeps=2, use_cuda_graph=False)
        for _ in range(4):
            executor.step()
        self.assertEqual(len(log), 2, "control steps 0 and 2")


class TestTheCounters(unittest.TestCase):
    """What a session reports about itself, and what the studio reads."""

    def _executor(self) -> ScheduleExecutor:
        return ScheduleExecutor(_model(), Schedule(), control_hz=50.0, substeps=2, use_cuda_graph=False)

    def test_a_step_with_no_graph_counts_as_eager(self) -> None:
        executor = self._executor()
        executor.step()
        executor.step()
        self.assertEqual((executor.steps_eager, executor.steps_replayed), (2, 0))
        self.assertFalse(executor.captured)

    def test_the_clock_advances_by_one_control_period(self) -> None:
        executor = self._executor()
        executor.step()
        self.assertAlmostEqual(executor.sim_time, 1.0 / 50.0)
        self.assertEqual(executor.step_count, 1)

    def test_invalidating_asks_for_a_recapture_without_one_being_possible(self) -> None:
        """On the CPU the re-capture is a no-op, and must not be a failure.

        ``recapture_failures`` counts capture attempts that *raised*; a device
        that cannot capture at all has not failed at anything.
        """
        executor = self._executor()
        executor.invalidate_graph()
        executor.step()
        self.assertFalse(executor.captured)
        self.assertEqual(executor.recapture_failures, 0)
        self.assertEqual(executor.capture_error, "")

    def test_substeps_at_runs_finer_and_puts_the_rate_back(self) -> None:
        """A settle wants a finer timestep than the run does."""
        executor = self._executor()
        log: list[str] = []
        executor.schedule.add(Stage(name="tick", fn=_recorder(log, "tick"), phase="post_solve"))
        with executor.substeps_at(8):
            self.assertTrue(executor.settling)
            executor.substep_block()
        self.assertEqual(len(log), 8)
        self.assertEqual(executor.substeps, 2, "the rate is restored")
        self.assertFalse(executor.settling)


class TestSubstepHooksIsThePhaseListsTogether(unittest.TestCase):
    """The object entities are handed, and the one ``pipeline.hooks`` exposes."""

    def test_adding_by_phase_name_lands_in_that_phase(self) -> None:
        hooks = SubstepHooks()
        stage = hooks.add("pre_solve", _recorder([], "bond"), name="bond", cadence=2)
        self.assertEqual((stage.phase, stage.name, stage.cadence), ("pre_solve", "bond", 2))
        self.assertEqual(len(hooks.pre_solve), 1)

    def test_the_named_slots_are_the_phases_entities_contribute_to(self) -> None:
        hooks = SubstepHooks()
        for phase in ("rigid_force", "pre_solve", "post_solve"):
            with self.subTest(phase=phase):
                self.assertIsInstance(getattr(hooks, phase), _PhaseList)

    def test_empty_hooks_are_falsey_and_filled_ones_are_not(self) -> None:
        hooks = SubstepHooks()
        self.assertFalse(hooks)
        hooks.add("post_solve", _recorder([], "x"))
        self.assertTrue(hooks)


if __name__ == "__main__":
    unittest.main()
