# SPDX-License-Identifier: Apache-2.0
"""What a policy is, single-world or batched, before it drives anything.

The two policy bases are twins on purpose -- driving one environment and driving
N behind a live mask is different work -- and this module holds the part that is
*not* different, plus the boundary of it. That boundary moved once and this is
where the reason lives, so the next person to look at the twins and see
duplication finds the measurement rather than repeating it.

Split out of ``test_fleet.py`` when that module hit the size ratchet: these are
claims about a policy's contract, not about fleet mechanics.

Static: no scene, no GPU.
"""

from __future__ import annotations

import unittest

from robosimstudio_bench.solutions.fleet import FleetSolutionIdle

TIER = "fast"
"""No GPU. Gates every commit."""


def _protocol_attrs(protocol) -> set[str]:
    """A protocol's members: Python 3.12's ``__protocol_attrs__``, or typing's own helper before it."""
    import typing

    declared = getattr(protocol, "__protocol_attrs__", None)
    return set(declared if declared is not None else typing._get_protocol_attrs(protocol))


class TestBothPolicyBasesHoldWhatTheyAreGiven(unittest.TestCase):
    """The half the single-world and fleet policies genuinely share.

    They share `close()` and `name` through `SolutionCommon` and nothing else --
    in particular **not** a constructor, because ``self.env`` is world scope and
    ``self.fleet`` is driver scope and
    :meth:`TestTheWorldViewAnswersWhatTheToolsAsk.test_every_name_a_tool_reads_is_one_a_world_can_answer`
    enforces the difference by scanning for ``env.`` reads.

    Which is why this exists. Moving the constructor up and then deciding not to
    left `SolutionBase` with no ``__init__`` at all: every policy still imported,
    every fast test still passed, and the failure surfaced only when a scene
    rollout constructed a scripted policy on a GPU. A base class quietly losing
    the thing that makes it a base class is cheap to check and expensive to miss.
    """

    def test_a_single_world_policy_holds_its_env(self) -> None:
        from robosimstudio_bench.solutions.base import SolutionIdle

        policy = SolutionIdle("ENV", stride=3)
        self.assertEqual(policy.env, "ENV")
        self.assertEqual(policy.kwargs, {"stride": 3})

    def test_a_fleet_policy_holds_its_fleet_under_that_name(self) -> None:
        policy = FleetSolutionIdle("FLEET", stride=3)
        self.assertEqual(policy.fleet, "FLEET")
        self.assertEqual(policy.kwargs, {"stride": 3})
        self.assertEqual(policy.slots, [])

    def test_both_report_a_name_from_the_one_place_it_is_written(self) -> None:
        from robosimstudio_bench.solutions.base import SolutionCommon, SolutionIdle
        from robosimstudio_bench.solutions.fleet import FleetSolution

        self.assertIs(type(SolutionIdle).__mro__ and SolutionCommon in SolutionIdle.__mro__, True)
        self.assertIn(SolutionCommon, FleetSolution.__mro__)
        self.assertTrue(SolutionIdle("E").name.endswith(".SolutionIdle"))
        self.assertTrue(FleetSolutionIdle("F").name.endswith(".FleetSolutionIdle"))


class TestBothBusesAreOneContract(unittest.TestCase):
    """`SimBus` and a fleet's `WorldBus` answer one protocol by two transports.

    `SimBus` owns a queue and an owner thread it pumps; `WorldBus` is a view
    onto a service queue shared by every world. Neither is the batched form of
    the other -- so what merges is the *contract*, named once in
    `core.bus.Bus`, and not the machinery under it.

    Checked rather than asserted in a docstring: a Protocol nothing verifies is
    a comment that drifts the first time an implementation grows a method the
    other lacks.
    """

    def test_the_owning_bus_satisfies_the_contract(self) -> None:
        from robosimstudio.core.bus import Bus, SimBus

        self.assertIsInstance(SimBus(), Bus)

    def test_a_world_bus_satisfies_it_too(self) -> None:
        from robosimstudio.core.bus import Bus
        from robosimstudio_policy_agent.fleet.world_bus import WorldBus

        for member in ("call", "snapshot", "wait_for_step", "close", "closed"):
            self.assertTrue(hasattr(WorldBus, member), f"WorldBus is missing {member!r}")
        self.assertTrue(set(_protocol_attrs(Bus)) <= set(dir(WorldBus)))

    def test_the_owner_side_is_deliberately_outside_the_contract(self) -> None:
        """`pump` and `publish` drive the loop; a tool that reached for either
        would have confused consuming the simulator with running it. `SimBus`
        has them only because it is also its own owner."""
        from robosimstudio.core.bus import Bus, SimBus
        from robosimstudio_policy_agent.fleet.world_bus import WorldBus

        self.assertNotIn("pump", _protocol_attrs(Bus))
        self.assertNotIn("publish", _protocol_attrs(Bus))
        self.assertTrue(hasattr(SimBus, "pump"))
        self.assertFalse(hasattr(WorldBus, "pump"))


class TestBothFacadesAreOneContract(unittest.TestCase):
    """The env surface the graph runtime drives, single-world or batched.

    The strongest of the fleet pairs: seven methods with *identical signatures*
    in both implementations, which is what makes it a contract rather than a
    resemblance. It was already being used as one -- the connector and the robot
    tools took ``facade: Any``, defaulted to a `GapEnvFacade` and accepted an
    injected `WorldFacade` -- so only the name was missing.
    """

    def _protocol(self):
        from robosimstudio_policy_agent.facade import EnvFacade

        return EnvFacade

    def test_the_single_world_facade_satisfies_it(self) -> None:
        from robosimstudio_policy_agent.facade import GapEnvFacade

        missing = [n for n in _protocol_attrs(self._protocol()) if not hasattr(GapEnvFacade, n)]
        self.assertEqual(missing, [])

    def test_a_world_facade_satisfies_it_too(self) -> None:
        from robosimstudio_policy_agent.fleet.world_facade import WorldFacade

        missing = [n for n in _protocol_attrs(self._protocol()) if not hasattr(WorldFacade, n)]
        self.assertEqual(missing, [])

    def test_what_is_outside_the_contract_is_outside_on_purpose(self) -> None:
        """``reset`` and ``publish_step`` are the single-world facade's, because
        there the facade drives the episode; a fleet resets in waves from the
        driver. ``servo_to_pose`` and ``set_grip`` are the fleet's, reaching
        segment drivers a single ``Env`` has not got. A runtime reaching for any
        of the four would be assuming a world count it is meant not to know."""
        from robosimstudio_policy_agent.facade import GapEnvFacade
        from robosimstudio_policy_agent.fleet.world_facade import WorldFacade

        shared = set(_protocol_attrs(self._protocol()))
        for name in ("reset", "publish_step"):
            self.assertTrue(hasattr(GapEnvFacade, name))
            self.assertNotIn(name, shared, f"{name} is single-world only")
        for name in ("servo_to_pose", "set_grip"):
            self.assertTrue(hasattr(WorldFacade, name))
            self.assertNotIn(name, shared, f"{name} is fleet only")


if __name__ == "__main__":
    unittest.main()
