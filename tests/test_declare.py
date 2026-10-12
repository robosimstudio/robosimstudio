# SPDX-License-Identifier: Apache-2.0
"""What the domain constructors promise: a domain, a contact set, and a refusal.

``Rod.from_length`` and ``Body.from_mesh``/``from_export`` exist so that an
author states what is *measured* about a thing and nothing about how the engine
files it. Two claims are worth pinning: that each supplies its domain and the
contact set its domain almost always wants, and that each refuses the mistakes
it was written to refuse -- a landmark off the rod, or one name declared twice.

Neither had a test. Both are reached by the corpus load, which proves they run
and proves nothing about what they promise: a constructor whose only coverage is
"66 assets happen to import cleanly" cannot tell you that its ``ValueError``
branches still fire.

Static: a record is data. No model, no scene, no GPU.
"""

from __future__ import annotations

import unittest

from robosimstudio.domains.rod import Rod

from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.domains.rigid import Body
from robosimstudio.domains.rigid.declare import TOUCHES

TIER = "fast"
"""No GPU. Gates every commit."""


def _touched(record) -> tuple[str, ...]:
    return tuple(rule.with_ for rule in record.sim.contacts)


class TestBody(unittest.TestCase):
    def test_it_supplies_the_domain_and_the_contact_set(self):
        """The two things all four hand-rolled copies agreed on exactly."""
        record = Body.from_mesh("h/h.obj", name="hammer")
        self.assertEqual(record.sim.domain, "rigid")
        self.assertEqual(_touched(record), TOUCHES)
        self.assertEqual(record.substrate, "rigid")

    def test_a_texture_is_the_pair_it_usually_travels_as(self):
        """Mesh-plus-diffuse is 12 of the corpus's rigid assets, so it is an
        argument rather than a ``files`` dict to assemble."""
        record = Body.from_mesh("h/h.obj", name="hammer", texture="h/d.png")
        self.assertEqual(record.files, {"mesh": "h/h.obj", "texture": "h/d.png"})

    def test_further_files_merge_over_the_geometry(self):
        """An export-backed object still names the per-object MJCF a suite reads
        for mass, and a caller may override what the constructor placed."""
        record = Body.from_export("vab", name="t", files={"mjcf": "t/t.xml", "export": "other"})
        self.assertEqual(record.files, {"export": "other", "mjcf": "t/t.xml"})

    def test_a_body_that_drops_fixtures_says_so(self):
        """Ten of them do, which is a real difference and not a default."""
        record = Body.from_mesh("h.obj", name="h", touches=("surface", "robot"))
        self.assertEqual(_touched(record), ("surface", "robot"))

    def test_it_refuses_a_body_with_no_name(self):
        """The name is what a task document refers to it by."""
        with self.assertRaises(ValueError):
            Body.from_mesh("h.obj", name="")


class TestRod(unittest.TestCase):
    def test_it_supplies_the_domain_and_derives_its_summary(self):
        record = Rod.from_length(name="rope", length=2.0, radius=0.008)
        self.assertEqual(record.sim.domain, "rod")
        self.assertIn("2 m", record.summary)

    def test_a_landmark_becomes_a_point_carrying_why_it_is_derived(self):
        """Arc length is true by construction, so the constructor states that on
        the author's behalf rather than making three assets type it."""
        record = Rod.from_length(name="rope", length=1.0, radius=0.01, points={"tip": 1.0})
        tip = record.feature("tip")
        self.assertEqual(tip.kind, "point")
        self.assertEqual(tip.payload["s"], 1.0)
        self.assertTrue(tip.provenance)

    def test_a_landmark_off_the_rod_is_refused(self):
        """Normalised, so a reference survives a change of segment count."""
        with self.assertRaises(ValueError):
            Rod.from_length(name="rope", length=1.0, radius=0.01, points={"tip": 1.4})

    def test_one_landmark_declared_twice_is_refused(self):
        """Two declarations of one landmark is the drift this exists to stop."""
        with self.assertRaises(ValueError):
            Rod.from_length(
                name="rope",
                length=1.0,
                radius=0.01,
                points={"tip": 1.0},
                features=(FeatureSpec(name="tip", kind="point", payload={"s": 0.9}),),
            )


if __name__ == "__main__":
    unittest.main()
