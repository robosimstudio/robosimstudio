# SPDX-License-Identifier: Apache-2.0
"""Every wait on the fleet's clock has to be one the barrier knows about.

A wave steps only when every live world's graph thread is parked in a fleet
wait. That makes "which waits park" a **closed set**, and the barrier's other
tests check each member of it thoroughly -- what none of them could check is
whether the set is complete.

It was not. ``robot_tools._wait`` polled ``bus.snapshot.step`` in a
``time.sleep`` loop and parked nowhere, so a graph that ramped its gripper
closed held the whole fleet: that world spun, the fleet could not step because
the barrier was waiting for it, the step it was spinning for therefore never
came, and every wait in the ramp burned its full timeout while the other worlds
sat parked on intents needing that same step. Silent -- a ramp logs nothing --
and it surfaced fifty-five minutes later as a runner cap reporting "wrote no
results rows", which reads like a broken policy rather than a deadlocked one.

So this scans for the shape rather than trusting the list: a loop that watches
the simulation clock go by while sleeping is a wait the barrier cannot see. The
fix in every case is the same -- ask the bus, whose ``wait_for_step`` parks the
world in a fleet and blocks plainly in a single world.

No GPU, no import of the modules scanned: this reads source.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

TIER = "fast"
"""Source only. Gates every commit, which is the point -- the thing it guards
against is invisible at one world and costs an hour to find at twelve."""

ROOT = pathlib.Path(__file__).resolve().parents[1]

SCANNED = ("robosimstudio_policy_agent",)
"""Where a tool that a graph can call lives. The fleet's own modules under
``robosimstudio_policy_agent/fleet`` are the ones that *implement* parking and are
scanned too -- a poll there would be the same bug one layer down."""

ALLOWED = frozenset()
"""Poll loops that are known not to wait on the fleet. Empty, and a new entry
wants a sentence saying why the loop cannot deadlock a wave."""

CLOCK_READS = ("snapshot.step", "step_count", "sim_time")
"""Attribute paths that mean "the simulation clock". Watching one go by is what
makes a loop a wait for the fleet rather than a wait for anything else."""


def _sleeps(node: ast.AST) -> bool:
    """Whether this subtree sleeps -- the tell of a poll rather than a block."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            fn = inner.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "sleep":
                return True
    return False


def _watches_the_clock(node: ast.AST) -> bool:
    """Whether this subtree reads the simulation clock."""
    src = ast.dump(node)
    return any(part in src for part in ("snapshot", "step_count", "sim_time")) and "step" in src


def _poll_loops() -> list[str]:
    found: list[str] = []
    for package in SCANNED:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.While, ast.For)):
                    continue
                if not (_sleeps(node) and _watches_the_clock(node)):
                    continue
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                if where.split(":", maxsplit=1)[0] not in ALLOWED:
                    found.append(where)
    return found


class TestEveryFleetWaitParks(unittest.TestCase):
    """The closed set is closed."""

    def test_no_tool_polls_the_simulation_clock(self):
        """A sleep loop watching the clock is a wait the barrier cannot see.

        Ask the bus instead: ``wait_for_step`` is on both bus kinds, parks the
        world in a fleet so the barrier can close, and is the plain blocking
        wait a poll approximates at one world.
        """
        found = _poll_loops()
        self.assertEqual(
            [],
            found,
            "these loops wait for the fleet's clock without parking, so a wave "
            f"cannot step while one is running: {found}. Use the bus's "
            "`wait_for_step`, which parks.",
        )

    def test_the_scan_can_see_the_shape_it_looks_for(self):
        """The guard's own guard: a test that matches nothing because its
        pattern is wrong passes exactly as loudly as one with nothing to find.

        This is the loop that was really there, reduced.
        """
        offending = ast.parse(
            "import time\n"
            "def _wait(self, steps):\n"
            "    start = int(self.bus.snapshot.step)\n"
            "    while int(self.bus.snapshot.step) - start < steps:\n"
            "        time.sleep(0.001)\n"
        )
        loops = [n for n in ast.walk(offending) if isinstance(n, ast.While)]
        self.assertEqual(1, len(loops))
        self.assertTrue(_sleeps(loops[0]))
        self.assertTrue(_watches_the_clock(loops[0]))


if __name__ == "__main__":
    unittest.main()
