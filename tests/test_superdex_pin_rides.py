# SPDX-License-Identifier: Apache-2.0
"""A SuperDex pin can ride a Newton body, re-aimed before every engine substep.

A held rod node used to be re-aimed once per control step, from the ``before``
phase -- from the hand as it stood *before* the step the arm then took -- so the
cable's end trailed the hand by a whole step's travel (5.2 mm at 0.3 m/s on the
cable demo). ``bridge.ride(pin, body, local)`` hands the pin to the tick, which
aims it from the body's pose swept through the control step the way it sweeps
the mirrors: each substep's target is the pose ``(k + 1) / substeps`` of the way
from where the body was at the last tick to where it is now (2.1 mm of trail at
the same speed, measured then). No shipped demo pins a cable any more -- the route
demo holds its cable by friction -- so this module, which holds the arithmetic
against a stand-in engine with no GPU and no ``[superdex]`` wheel, is the pin's check.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "fast"
"""A stand-in scene and constraints. No GPU, no wheel."""

BODY = 3
LOCAL = np.array([[0.0, 0.0, -0.1]])
"""The node 10 cm down the body's own z."""


class _Constraint:
    def __init__(self, position) -> None:
        self.target = np.asarray(position, dtype=np.float64)

    def set_target_position(self, position) -> None:
        self.target = np.asarray(position, dtype=np.float64)


class _Stats:
    convergence_status = "CONVERGED"


class _Scene:
    """Records every pin's target at each engine step: what the solve saw."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []
        self.constraints: list[_Constraint] = []
        self.destroyed: list[_Constraint] = []

    def create_deformable_node_position_constraint(self, *, position, **kwargs):
        self.constraints.append(_Constraint(position))
        return self.constraints[-1]

    def destroy_constraint(self, constraint) -> None:
        self.destroyed.append(constraint)

    def step(self, dt) -> None:
        self.seen.append(np.array([c.target for c in self.constraints if c not in self.destroyed]))

    def get_solver_stats(self):
        return _Stats()


class _State:
    def __init__(self, pose) -> None:
        body_q = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (BODY + 1, 1))
        body_q[BODY] = pose
        self.body_q = type("_BQ", (), {"numpy": lambda _self: body_q})()


def _pose(x: float, yaw: float = 0.0) -> np.ndarray:
    return np.array([x, 0.0, 0.5, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _bridge(substeps: int = 2):
    """A bridge with one world, no mirrors and no actors: only its pins and its tick."""
    from robosimstudio.engines.cosim import CosimRig
    from robosimstudio.engines.superdex.bridge import SuperdexBridge

    bridge = object.__new__(SuperdexBridge)
    scene = _Scene()
    bridge.scenes, bridge.num_worlds, bridge._lanes = [scene], 1, None
    bridge.substeps, bridge.mirrors, bridge.rig = substeps, [], CosimRig((), substeps=substeps)
    bridge.physics = None  # the engine module: only a mirror reaches for it, and there are none
    bridge.actors = [{"name": "rope", "start": 0, "actor": object()}]  # what pin_nodes looks the rope up in
    bridge.recipe, bridge.feedback_enabled, bridge.apply_wrenches = {}, False, None
    bridge._diverged_worlds, bridge._diverged_logged = [False], set()
    bridge._pins, bridge._pin_world, bridge._riders, bridge._rider_prev, bridge._pin_seq = {}, {}, {}, {}, 0
    return bridge, scene


def _tick(bridge, pose) -> None:
    """One control step, with the rope's read-back taken out: only the pins and the steps happen."""
    actors, bridge.actors = bridge.actors, []
    try:
        bridge.tick(_State(pose), 1.0 / 60.0)
    finally:
        bridge.actors = actors


class TestAPinRidesItsBody(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge, self.scene = _bridge(substeps=2)
        self.pin = self.bridge.pin_nodes("rope", np.array([5]), np.array([[0.0, 0.0, 0.4]]))
        self.bridge.ride(self.pin, BODY, LOCAL)

    def test_the_first_tick_aims_at_the_body_as_it_stands(self) -> None:
        """No previous pose to sweep from, as after a reset: every substep aims at now."""
        _tick(self.bridge, _pose(0.2))
        np.testing.assert_allclose(np.concatenate(self.scene.seen), [[0.2, 0.0, 0.4]] * 2, atol=1e-12)

    def test_each_substep_aims_its_share_of_the_sweep(self) -> None:
        _tick(self.bridge, _pose(0.0))
        self.scene.seen.clear()
        _tick(self.bridge, _pose(0.01))
        np.testing.assert_allclose(np.concatenate(self.scene.seen), [[0.005, 0.0, 0.4], [0.01, 0.0, 0.4]], atol=1e-12)

    def test_the_offset_turns_with_the_body(self) -> None:
        """``local`` is in the body's frame: a body rolled half a turn about x puts the node above it."""
        _tick(self.bridge, np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(self.scene.seen[-1], [[0.0, 0.0, 0.6]], atol=1e-12)

    def test_riding_again_moves_the_offset_and_unpin_ends_the_ride(self) -> None:
        self.bridge.ride(self.pin, BODY, LOCAL + np.array([0.0, 0.0, 0.05]))
        _tick(self.bridge, _pose(0.0))
        np.testing.assert_allclose(self.scene.seen[-1], [[0.0, 0.0, 0.45]], atol=1e-12)
        self.assertTrue(self.bridge.has_pin(self.pin))
        self.bridge.unpin(self.pin)
        self.assertFalse(self.bridge.has_pin(self.pin))
        self.assertEqual({}, self.bridge._riders)
        _tick(self.bridge, _pose(0.3))  # nothing left to aim, and nothing raises

    def test_a_ride_is_refused_for_a_pin_that_is_not_there(self) -> None:
        with self.assertRaises(KeyError):
            self.bridge.ride(self.pin + 1, BODY, LOCAL)
        with self.assertRaises(ValueError):
            self.bridge.ride(self.pin, BODY, np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
