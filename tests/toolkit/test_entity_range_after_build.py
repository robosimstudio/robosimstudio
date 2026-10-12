# SPDX-License-Identifier: Apache-2.0
"""An entity's shape range covers what its own ``after_build`` appended.

``Entity.build_into`` closes the range after :meth:`build`, then runs
:meth:`after_build`, then :meth:`apply_display` -- and ``apply_display`` paints
the range. A rigid body whose spec asks for a ``collider`` replaces its mesh with
an approximation *inside* ``after_build``, and Newton keeps the mesh it replaced
as a new, visual-only shape at the end of the builder. That shape lay outside
the closed range, so the stated colour landed on the invisible collider and the
one shape a camera sees kept the build-order palette. Measured on
``sharps_disposal/tray_pair``'s spare syringe: stated ``(0.90, 0.91, 0.93)``,
rendered cyan, and SAM-3 read a different scene than the one the numbers were
taken in. No device: a builder, a fake entity and the paint step.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import newton
import numpy as np

from robosimstudio.core.render.palette import is_build_order_color
from robosimstudio.scene.entity import Entity

TIER = "fast"

STATED = (0.90, 0.91, 0.93)


class _Painted(Entity):
    """Adds one box in ``build`` and a second on the same body in ``after_build``."""

    role = "objects"

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(name="spare", color=STATED, visible=True, collide=True, finish=None))
        self.body = -1

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        self.body = builder.add_body(label="spare")
        builder.add_shape_box(self.body, hx=0.05, hy=0.005, hz=0.005)

    def after_build(self, builder: newton.ModelBuilder, episode) -> None:
        # What approximate_meshes(keep_visual_shapes=True) does: a new shape,
        # this body's, appended after the range was closed.
        builder.add_shape_box(self.body, hx=0.05, hy=0.005, hz=0.005)


class _Found(Entity):
    """A scene-resident stand-in: adds nothing, then resets its range as find_in_scene does."""

    role = "objects"

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(name="resident", color=STATED, visible=True, collide=True, finish=None))

    def after_build(self, builder: newton.ModelBuilder, episode) -> None:
        builder.add_shape_box(builder.add_body(label="somebody_else"), hx=0.1, hy=0.1, hz=0.1)
        self.range.shapes = (0, 0)


class TestTheRangeFollowsAfterBuild(unittest.TestCase):
    def test_a_shape_appended_by_after_build_is_the_entitys_and_is_painted(self):
        builder = newton.ModelBuilder()
        entity = _Painted()
        entity.build_into(builder, episode=None)
        self.assertEqual(entity.range.shapes, (0, 2))
        for s in range(2):
            with self.subTest(shape=s):
                self.assertEqual(tuple(round(float(v), 3) for v in builder.shape_color[s]), STATED)
                self.assertFalse(is_build_order_color(np.asarray(builder.shape_color[s], dtype=np.float64), s))

    def test_an_entity_that_added_nothing_does_not_claim_what_after_build_touched(self):
        builder = newton.ModelBuilder()
        builder.add_shape_box(builder.add_body(label="arena"), hx=1.0, hy=1.0, hz=0.01)
        entity = _Found()
        entity.build_into(builder, episode=None)
        # The reset stands: an empty range stays empty, and nothing was repainted.
        self.assertEqual(entity.range.shapes, (0, 0))
        for s in range(builder.shape_count):
            with self.subTest(shape=s):
                self.assertNotEqual(tuple(round(float(v), 3) for v in builder.shape_color[s]), STATED)


if __name__ == "__main__":
    unittest.main()
