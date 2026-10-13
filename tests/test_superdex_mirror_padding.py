# SPDX-License-Identifier: Apache-2.0
"""The SuperDex mirrors' contact padding: an option that pads every mirror, and nothing when unset.

A rod built without its tubular skin samples contact on its own axis, and the
engine adds a collider's ``penalty_threshold_extra_padding`` to the stand-off
only against such a point-cloud actor. Unpadded, a 5 mm cable rested with its
axis 0.93 mm over the bench, half of it inside the wood, and a clip's prong came
within 2.6 mm of the axis -- inside the prong. ``mirror_padding`` on the backend
pads every mirror, and padded by one radius less the engine's own 1 mm the same
cable rested with its axis 2.43 mm up, on its skin (measured on the SuperDex cable
demo this tree used to ship). This module holds the option itself against a
stand-in engine, so it runs with no GPU and no ``[superdex]`` wheel.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

TIER = "fast"
"""A stub model and a stand-in engine. No GPU, no wheel, no scene."""


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def numpy(self):
        return self._a


class _Model:
    """A bench (a world-fixed box), the ground plane and a fingertip capsule on body 3."""

    def __init__(self):
        from newton import GeoType

        self.shape_type = _Arr([int(GeoType.PLANE), int(GeoType.BOX), int(GeoType.CAPSULE)])
        self.shape_scale = _Arr([[0.0, 0.0, 0.0], [0.3, 0.65, 0.375], [0.005, 0.01, 0.0]])
        self.shape_transform = _Arr(np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (3, 1)))
        self.shape_body = _Arr([-1, -1, 3])
        self.shape_material_mu = _Arr([1.0, 1.0, 0.8])
        self.shape_source = [None, None, None]


class _ContactParams:
    """The engine's ``ContactParams``: its defaults, and which fields were written after construction."""

    def __init__(self, **given):
        object.__setattr__(self, "written", set())
        for key, value in {"penalty_threshold_default": 0.001, "penalty_threshold_extra_padding": 0.0, **given}.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        self.written.add(key)
        object.__setattr__(self, key, value)


class _Engine:
    """Enough of ``superdex.physics`` for :func:`build_mirror_actors`."""

    class ColliderType:
        AUTO, BOX, SDF = "auto", "box", "sdf"

    ContactParams = _ContactParams

    @staticmethod
    def TransformRT(rotation, translation):
        return (rotation, translation)

    @staticmethod
    def Quaternion(*xyzw):
        return xyzw

    @staticmethod
    def create_plane_shape(**kwargs):
        return ("plane", kwargs)

    @staticmethod
    def create_sphere_shape(**kwargs):
        return ("sphere", kwargs)

    @staticmethod
    def create_tri_mesh_shape(**kwargs):
        return ("mesh", len(kwargs["connectivity"]) // 3)


class _Scene:
    def __init__(self):
        self.actors: list[dict] = []

    def create_rigid_actor(self, **kwargs):
        self.actors.append(kwargs)
        return kwargs


RECORDS = [{"shape": 0, "body": -1}, {"shape": 1, "body": -1}, {"shape": 2, "body": 3, "two_way": True}]


def _build(**options):
    from robosimstudio.engines.superdex.mirror import build_mirror_actors

    scene = _Scene()
    mirrors = build_mirror_actors(_Engine, scene, _Model(), shapes=RECORDS, **options)
    return mirrors, [actor["contact"] for actor in scene.actors]


class TestTheMirrorPadding(unittest.TestCase):
    def test_unset_it_leaves_every_mirror_as_it_was(self):
        mirrors, contacts = _build()
        self.assertEqual(3, len(mirrors), "the plane, the bench and the fingertip")
        for contact in contacts:
            self.assertNotIn("penalty_threshold_extra_padding", contact.written, "an unset option writes nothing")
            self.assertEqual(0.0, contact.penalty_threshold_extra_padding)

    def test_set_it_pads_every_mirror_and_changes_nothing_else(self):
        _, plain = _build()
        _, padded = _build(contact_padding=0.0015)
        for before, after in zip(plain, padded, strict=True):
            self.assertEqual(0.0015, after.penalty_threshold_extra_padding)
            # The threshold itself is a different knob, and stays the engine's.
            self.assertEqual(before.penalty_threshold_default, after.penalty_threshold_default)
            for field in ("coulomb_friction_coefficient", "penalty_smoothing_half_distance", "friction_falloff_vel"):
                self.assertEqual(getattr(before, field), getattr(after, field), field)

    def test_it_composes_with_the_threshold(self):
        _, contacts = _build(contact_padding=0.0015, contact_threshold=2.0e-5)
        for contact in contacts:
            self.assertEqual(
                (2.0e-5, 0.0015), (contact.penalty_threshold_default, contact.penalty_threshold_extra_padding)
            )

    def test_a_negative_padding_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "contact_padding must not be negative"):
            _build(contact_padding=-0.001)


class TestTheBackendOption(unittest.TestCase):
    """``backend("superdex", mirror_padding=...)`` reaches the bridge, which hands it to the mirrors."""

    def test_the_solver_forwards_it_to_the_bridge(self):
        import robosimstudio.engines.superdex.bridge as bridge
        from robosimstudio.engines.superdex.solver import SolverSuperdex

        seen: dict = {}

        class _Bridge:
            def __init__(self, model, recipe, **kwargs):
                seen.update(kwargs)

        real = bridge.SuperdexBridge
        bridge.SuperdexBridge = _Bridge
        try:
            SolverSuperdex.open(object.__new__(SolverSuperdex), None, {}, substeps=2, mirror_padding=0.0015)
            self.assertEqual(0.0015, seen["mirror_padding"])
            SolverSuperdex.open(object.__new__(SolverSuperdex), None, {}, substeps=2)
            self.assertEqual(0.0, seen["mirror_padding"], "unstated, it is 0: the mirrors as they were")
        finally:
            bridge.SuperdexBridge = real

    def test_the_bridge_hands_it_to_the_mirrors(self):
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        self.assertEqual(0.0, inspect.signature(SuperdexBridge).parameters["mirror_padding"].default)
        source = inspect.getsource(SuperdexBridge.__init__)
        self.assertIn("contact_padding=self.mirror_padding", source)


if __name__ == "__main__":
    unittest.main()
