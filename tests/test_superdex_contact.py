# SPDX-License-Identifier: Apache-2.0
"""What a cable on SuperDex can touch, and where it comes to rest.

Each class here is one defect that read as "the physics is wrong" and was not:

* a Cosserat rod sampled contact on its own axis, so it rested **one radius
  inside** whatever it lay on (+0.91 mm of clearance for an 11 mm cable);
* a soft body was created with **no collider** (``ColliderType.NONE``), so a
  cable pressed into a soft clip passed straight through it;
* a mirror's contact threshold -- the collider's, which is the one that governs
  a pair -- was the engine's 1 mm, a stand-off between every pad and what it
  pressed;
* a rod's ``anchors`` were never read on this engine, and there was no way to
  mould a cable into another body at all;
* a body on a rigid core silently lost the SDF collider its material asked for;
* a join into a plain soft body tied each cable node to the body's nearest
  node with no rest offset, so a body whose nodes did not meet the cable's was
  yanked onto it on the first step;
* and once read, an anchor held only the end node -- a pin, where the capsule
  chain's fixed joint is a clamp -- so one spec was a cantilever on VBD and a
  pendulum here.

That a stress-free rod's teleport moves nothing else is :mod:`tests.test_superdex_teleport`.

Bands, not bits: the engine documents no determinism guarantee.
"""

from __future__ import annotations

import importlib.util
import unittest
import warnings

import numpy as np

TIER = "medium"
"""A GPU, the [superdex] extra, and a scene or two."""

BENCH_TOP = 0.75
"""bench_075's work surface height [m]."""


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _arrangement(**superdex):
    import robosimstudio as rss
    from robosimstudio.specs.physics import backend

    engine = backend("superdex", substeps=2, **superdex)
    return rss.replace(
        rss.GENERIC,
        backends=(("articulated", backend("mujoco")), ("rod", engine), ("soft", engine), ("*", backend("vbd"))),
    )


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestTheCableRestsOnItsSkin(unittest.TestCase):
    """The scene ``examples/10_rope/02_superdex.py`` builds, measured where it lands."""

    def test_an_11mm_cable_rests_11mm_above_the_bench(self):
        import robosimstudio as rss

        radius = 0.011
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="skin")
        cable = scene.add(
            rss.rod(name="cable", length=0.6, radius=radius, at=(0.0, -0.3, radius + 0.03), along=(0, 1, 0))
        )
        sim = scene.build()
        sim.run(200).sync()
        clearance = float(cable.particle_q[:, 2].min()) - BENCH_TOP
        # +0.91 mm before the skin; the engine's own 1 mm threshold and its
        # smoothing ramp put the axis a fraction of a millimetre off r.
        self.assertAlmostEqual(
            clearance, radius, delta=0.0015, msg=f"axis rests {clearance * 1e3:.2f} mm up, radius 11"
        )
        self.assertTrue(sim.state_is_finite())


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestACableCanTouchASoftBody(unittest.TestCase):
    """A cable dropped across a soft block lies on it; it used to fall through it."""

    def test_the_cable_lies_on_the_block(self):
        from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

        import robosimstudio as rss

        block = SoftSpec(
            name="block",
            size=(0.06, 0.06, 0.03),
            cells=(6, 6, 3),
            pose=(0.0, 0.0, 0.0),
            material=SoftMaterialSpec(young=2.0e5, density=200.0, collider="auto"),
            settle_frames=5,
        )
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="cable_on_block")
        scene.add(block)
        cable = scene.add(
            rss.rod(name="cable", length=0.2, radius=0.004, at=(0.0, -0.1, 0.045), along=(0, 1, 0), segments=40)
        )
        sim = scene.build()
        sim.run(120).sync()
        q = cable.particle_q
        middle = q[np.abs(q[:, 1]) < 0.01]
        # The block's top is 30 mm up; through it, the cable would reach the bench.
        self.assertGreater(float(middle[:, 2].min()) - BENCH_TOP, 0.025, "the cable's middle sank through the block")
        self.assertTrue(sim.state_is_finite())


def _hanging_cable(*, nodes: int, name: str = "cable", at: tuple[float, float] = (0.0, 0.0), **kw):
    """A 3 mm cable hung straight down from 400 mm over ``at``, ``nodes - 1`` legs of 10 mm."""
    from robosimstudio.domains.rod.spec import RodSpec

    top = np.array([at[0], at[1], 0.40])
    return RodSpec(
        name=name,
        nodes=[tuple(top + np.array([0.0, 0.0, -0.01 * i])) for i in range(nodes)],
        radius=0.003,
        density=1400.0,
        young_modulus=3.0e6,
        axial_stiffness=2.0e4,
        anchors=("start",),
        **kw,
    )


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestAnchorsAndJoins(unittest.TestCase):
    """A cable anchored at its top holds a soft weight moulded onto its end.

    The weight is cut 2 x 2 x 3 so one lattice column runs down the cable's
    axis and each joined cable node sits on a weight node -- what a node-to-node
    weld, which has no rest offset, needs. The 3 x 3 x 3 cut this test used
    before put them 8.7-16.6 mm apart, and the first step yanked the weight
    34 mm onto the cable; that join is now refused (see
    :class:`TestAJoinWhoseNodesDoNotMeet`).
    """

    @classmethod
    def setUpClass(cls):
        from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

        import robosimstudio as rss

        weight = SoftSpec(
            name="weight",
            size=(0.03, 0.03, 0.03),
            cells=(2, 2, 3),  # a node every 10 mm up x = y = 0, from 170 to 200 mm
            pose=(0.0, 0.0, 0.17),  # bottom face; the cable's last 20 mm lie inside it
            material=SoftMaterialSpec(young=1.0e6, density=300.0),
            settle_frames=0,
        )
        cable = _hanging_cable(nodes=23, joins=(("end", "weight", 0.021),))  # down to 180 mm
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="anchored")
        cls.weight = scene.add(weight)
        cls.cable = scene.add(cable)
        cls.top = np.array([0.0, 0.0, 0.40])
        cls.sim = scene.build()
        cls.built = np.asarray(cls.weight.particle_q).copy()
        cls.sim.run(90).sync()

    def test_the_anchor_holds(self):
        first = self.cable.particle_q[0] - np.array([0.6, 0.0, BENCH_TOP])
        np.testing.assert_allclose(first, self.top, atol=0.002)

    def test_the_weight_hangs_rather_than_falling(self):
        bottom = float(self.weight.particle_q[:, 2].min()) - BENCH_TOP
        # 170 mm up at rest; unjoined it falls to the bench.
        self.assertGreater(bottom, 0.12, f"the weight's bottom is {bottom * 1e3:.0f} mm up")

    def test_the_weight_hangs_where_it_was_built(self):
        # 0.004 mm in 30 steps on coincident nodes; 34 mm on the 3 x 3 x 3 cut's gaps.
        moved = float(np.linalg.norm(np.asarray(self.weight.particle_q) - self.built, axis=1).max())
        self.assertLess(moved, 5.0e-4, f"the weight moved {moved * 1e3:.2f} mm")

    def test_the_end_stays_inside_the_weight(self):
        end = self.cable.particle_q[-1]
        lo, hi = self.weight.particle_q.min(axis=0), self.weight.particle_q.max(axis=0)
        self.assertTrue(np.all(end >= lo - 0.002) and np.all(end <= hi + 0.002), f"end {end}, weight {lo}..{hi}")


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestAJoinWhoseNodesDoNotMeet(unittest.TestCase):
    """A join into a plain soft body is refused unless its nodes meet; on a rigid core it keeps its offsets.

    The 3 x 3 x 3 weight leaves the cable's last three nodes 16.6, 8.7 and
    8.7 mm from their nearest weight nodes, two of them on one node: welded
    node to node, the first step launched the weight's nodes at 21 m/s.

    The refusal's advice -- put the body on a rigid core -- is taken as
    written: ``rigid=`` and nothing else, the collider left at ``"auto"``.
    """

    def _scene(self, *, rigid: bool):
        from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

        import robosimstudio as rss

        material = (
            SoftMaterialSpec(young=1.0e9, density=1200.0) if rigid else SoftMaterialSpec(young=1.0e6, density=300.0)
        )
        weight = SoftSpec(
            name="weight",
            size=(0.03, 0.03, 0.03),
            cells=(3, 3, 3),
            pose=(0.0, 0.0, 0.175),
            material=material,
            settle_frames=0,
            rigid=(lambda rest: rest[:, 2] < rest[:, 2].max() - 1.0e-6) if rigid else None,
        )
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="join_gap")
        body = scene.add(weight)
        scene.add(_hanging_cable(nodes=21, joins=(("end", "weight", 0.021),)))
        return scene, body

    def test_it_is_refused(self):
        scene, _ = self._scene(rigid=False)
        with self.assertRaisesRegex(ValueError, r"(?s)'cable' joins 'weight'.*16\.6, 8\.66, 8\.66 mm.*rigid core"):
            scene.build()

    def test_onto_a_rigid_core_it_keeps_its_offsets(self):
        scene, body = self._scene(rigid=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim = scene.build()
        # The only other deformable is the cable welded into it, which could not touch it anyway.
        self.assertEqual([str(w.message) for w in caught if "rides a rigid core" in str(w.message)], [])
        built = np.asarray(body.particle_q).copy()
        sim.run(60).sync()
        moved = float(np.linalg.norm(np.asarray(body.particle_q) - built, axis=1).max())
        self.assertLess(moved, 5.0e-4, f"the weight moved {moved * 1e3:.2f} mm")  # 0.004 mm in 30 steps
        self.assertTrue(sim.state_is_finite())


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestTheMirrorStandOff(unittest.TestCase):
    """A sponge resting on the bench: 1 mm up at the engine's threshold, on it at 20 um."""

    def _rest(self, **knobs) -> float:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", physics=_arrangement(**knobs), name="standoff")
        sponge = scene.add(rss.sponge(name="sponge", at=(0.0, 0.0, 0.005)))
        sim = scene.build()
        sim.run(60).sync()
        return float(sponge.particle_q[:, 2].min()) - BENCH_TOP

    def test_the_threshold_is_the_stand_off(self):
        default = self._rest()
        tight = self._rest(mirror_threshold=2.0e-5, mirror_smoothing=1.0e-4)
        self.assertLess(tight, default - 0.0005, f"default {default * 1e3:.2f} mm, tight {tight * 1e3:.2f} mm")
        self.assertLess(abs(tight), 0.0004)


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestAStiffBodyOnARigidCore(unittest.TestCase):
    """A 1e9 Pa block falls onto the bench when its body rides a rigid core.

    Free, a body this stiff has rigid modes ~1e8 below its elastic ones, under
    the engine's single precision: a 2.6 g plug hung in mid-air for 1.5 s.
    """

    def test_it_falls_and_its_free_layer_stays_on_it(self):
        from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

        import robosimstudio as rss

        drop = 0.05
        brick = SoftSpec(
            name="brick",
            size=(0.03, 0.02, 0.01),
            cells=(3, 2, 2),
            pose=(0.0, 0.0, drop),
            material=SoftMaterialSpec(young=1.0e9, poisson=0.35, density=1200.0, collider="none"),
            # Everything but the top layer: that one stays deformable.
            rigid=lambda rest: rest[:, 2] < rest[:, 2].max() - 1.0e-6,
            settle_frames=0,
        )
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="core")
        body = scene.add(brick)
        sim = scene.build()
        sim.run(60).sync()
        q = body.particle_q
        bottom = float(q[:, 2].min()) - BENCH_TOP
        self.assertLess(
            abs(bottom), 0.002, f"the bottom is {bottom * 1e3:.1f} mm up; it was dropped from {drop * 1e3:.0f}"
        )
        self.assertAlmostEqual(float(np.ptp(q[:, 2])), 0.01, delta=0.001)
        self.assertTrue(sim.state_is_finite())


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestARigidCoreHasNoCollider(unittest.TestCase):
    """A body on a rigid core has no collider here: asking for an SDF is refused.

    It was dropped without a word: with ``collider="sdf"`` on a 30 mm block on a
    rigid core a cable dropped across it came to rest 4.59 mm off the bench,
    straight through it, where the same block without the core held it at
    33.02 mm. No other deformable can detect such a body; it touches one only
    where its own surface samples meet that one's collider, which a soft
    body's SDF is and a cable's is not.

    ``"auto"``, the default, asks for the collider the body can have, which on
    a rigid core is none: it builds, and warns, naming each deformable with no
    SDF -- the cable, not the plain block beside it. Refusing it too sent a
    body that a join's own refusal had put on a rigid core straight into this
    one.
    """

    def _scene(self, collider: str):
        from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

        import robosimstudio as rss

        brick = SoftSpec(
            name="brick",
            size=(0.03, 0.02, 0.01),
            cells=(3, 2, 2),
            pose=(0.0, 0.0, 0.0),
            material=SoftMaterialSpec(young=1.0e9, poisson=0.35, density=1200.0, collider=collider),
            rigid=lambda rest: rest[:, 2] < rest[:, 2].max() - 1.0e-6,
            settle_frames=0,
        )
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name=f"core_{collider}")
        scene.add(brick)
        # Other deformables in the world, which is what makes "auto" pick an SDF: a plain block takes
        # one, and the brick's samples meet it; the cable has none.
        scene.add(rss.rod(name="cable", length=0.1, radius=0.003, at=(0.0, -0.05, 0.05), along=(0, 1, 0), segments=10))
        scene.add(
            SoftSpec(
                name="block",
                size=(0.03, 0.03, 0.01),
                cells=(1, 1, 1),
                pose=(0.1, 0.0, 0.0),
                material=SoftMaterialSpec(young=1.0e5, density=300.0),
                settle_frames=0,
            )
        )
        return scene

    def test_an_sdf_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"(?s)'brick' rides a rigid core.*collider='sdf'.*'auto' or 'none'"):
            self._scene("sdf").build()

    def test_auto_builds_and_names_what_cannot_touch_it(self):
        with self.assertWarnsRegex(UserWarning, r"(?s)'brick' rides a rigid core.*\['cable'\] in its world has no SDF"):
            sim = self._scene("auto").build()
        self.assertTrue(sim.state_is_finite())

    def test_none_builds_quietly(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim = self._scene("none").build()
        self.assertEqual([str(w.message) for w in caught if "rides a rigid core" in str(w.message)], [])
        self.assertTrue(sim.state_is_finite())


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestAStressFreeShape(unittest.TestCase):
    """A cable laid out as an L and hung by its top: it keeps the L, unless it is stress-free straight.

    A rod with no rest/initial split rests in whatever shape it was laid out in: a
    cable built in an S on the bench, lifted, bowed back toward the S.
    """

    def _hang(self, stress_free) -> float:
        from robosimstudio.domains.rod.spec import RodSpec

        import robosimstudio as rss

        top = np.array([0.0, 0.0, 0.40])
        down = [top + np.array([0.0, 0.0, -0.005 * i]) for i in range(21)]  # 100 mm down
        out = [down[-1] + np.array([0.005 * i, 0.0, 0.0]) for i in range(1, 21)]  # then 100 mm across
        cable = RodSpec(
            name="cable",
            nodes=tuple(tuple(float(v) for v in p) for p in down + out),
            radius=0.003,
            density=1400.0,
            young_modulus=3.0e6,
            axial_stiffness=2.0e4,
            anchors=("start",),
            stress_free=stress_free,
        )
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="stress_free")
        rod = scene.add(cable)
        sim = scene.build()
        start = np.asarray(rod.particle_q).copy()
        sim.run(120).sync()
        q = np.asarray(rod.particle_q)
        self.assertTrue(np.allclose(start[0], q[0], atol=1e-4), "the anchor moved")
        return float(q[-1, 0] - q[0, 0])

    def test_laid_out_keeps_its_corner_straight_hangs(self):
        laid = self._hang(None)
        straight = self._hang("straight")
        # The L droops, corner and all, but stays out (60 mm with its top clamped, 51 mm when the
        # anchor was a pin); unbent, it hangs plumb (0.8 mm, 2.6 mm pinned).
        self.assertGreater(laid, 0.035, f"the laid-out L's end is {laid * 1e3:.0f} mm out")
        self.assertLess(abs(straight), 0.02, f"the stress-free straight cable's end is {straight * 1e3:.0f} mm out")

    def test_a_polyline_of_other_legs_is_refused(self):
        from robosimstudio.domains.rod.spec import RodSpec

        import robosimstudio as rss

        nodes = tuple((0.0, 0.0, 0.3 - 0.01 * i) for i in range(5))
        wrong = tuple((0.0, 0.0, 0.3 - 0.02 * i) for i in range(5))
        scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="stress_free_bad")
        scene.add(RodSpec(name="cable", nodes=nodes, radius=0.003, stress_free=wrong))
        with self.assertRaises(ValueError):
            scene.build()


def _cantilever(engine: str, **matter) -> dict:
    """A level 300 mm cable anchored at its start, 300 mm over the bench, for 120 steps.

    Returns the worst first-segment dip [deg] below level, the tip's drop [m]
    (largest and at the end) and how far the anchor node moved [m], read every
    30 steps.
    """
    from robosimstudio.domains.rod.spec import RodSpec

    import robosimstudio as rss
    from robosimstudio.specs.physics import backend

    rod = backend("superdex", substeps=2) if engine == "superdex" else backend("vbd")
    soft = rod if engine == "superdex" else backend("vbd")
    physics = rss.replace(
        rss.GENERIC,
        backends=(("articulated", backend("mujoco")), ("rod", rod), ("soft", soft), ("*", backend("vbd"))),
    )
    scene = rss.Scene(cell="bench_075", physics=physics, name=f"cantilever_{engine}")
    cable = scene.add(
        RodSpec(
            name="cable",
            start=(0.0, -0.15, 0.30),
            direction=(0.0, 1.0, 0.0),
            length=0.30,
            segments=20,
            radius=0.003,
            density=1400.0,
            anchors=("start",),
            **matter,
        )
    )
    sim = scene.build()

    def nodes() -> np.ndarray:
        if engine == "superdex":
            return np.asarray(cable.particle_q, dtype=np.float64)
        state = type("State", (), {"body_q": sim.pipeline.state_0.body_q.numpy()})()
        return np.asarray(sim.scene.by_name["cable"].centerline(state), dtype=np.float64)

    p0 = nodes()
    dips, drops, moved = [], [], []
    for _ in range(4):
        sim.run(30).sync()
        p = nodes()
        leg = p[1] - p[0]
        dips.append(float(np.degrees(np.arctan2(-leg[2], np.hypot(leg[0], leg[1])))))
        drops.append(float(p0[-1, 2] - p[-1, 2]))
        moved.append(float(np.linalg.norm(p[0] - p0[0])))
    return {
        "dip": max(dips),
        "drop_max": max(drops),
        "drop": drops[-1],
        "moved": max(moved),
        "finite": sim.state_is_finite(),
    }


def _twisted(anchor: str) -> dict:
    """A level L 300 mm over the bench anchored at ``anchor``: 100 mm along y, then 100 mm along x, 120 steps.

    The free leg's weight twists the anchored leg about its own axis, which only
    the clamp's twist hold resists. Returns the free tip's largest drop [m] and
    how far the anchor node moved [m], read every 30 steps.
    """
    from robosimstudio.domains.rod.spec import RodSpec

    import robosimstudio as rss

    held = [(0.0, -0.10 + 0.01 * i, 0.30) for i in range(11)]
    nodes = held + [(0.01 * i, 0.0, 0.30) for i in range(1, 11)]
    scene = rss.Scene(cell="bench_075", physics=_arrangement(), name=f"twisted_{anchor}")
    cable = scene.add(
        RodSpec(
            name="cable",
            nodes=tuple(nodes if anchor == "start" else nodes[::-1]),
            radius=0.003,
            density=1400.0,
            young_modulus=1.0e9,
            anchors=(anchor,),
        )
    )
    sim = scene.build()
    tip, root = (-1, 0) if anchor == "start" else (0, -1)
    p0 = np.asarray(cable.particle_q, dtype=np.float64).copy()
    drops, moved = [], []
    for _ in range(4):
        sim.run(30).sync()
        p = np.asarray(cable.particle_q, dtype=np.float64)
        drops.append(float(p0[tip, 2] - p[tip, 2]))
        moved.append(float(np.linalg.norm(p[root] - p0[root])))
    return {"drop": max(drops), "moved": max(moved), "finite": sim.state_is_finite()}


class TestAnAnchorIsAClamp(unittest.TestCase):
    """``RodSpec.anchors`` clamps its end on both engines: a level cable anchored at one end sags, it does not swing.

    Held by its end node alone, the SuperDex rod was a pendulum: its first
    segment dipped 74 degrees in 120 steps and its tip fell 288 mm, while the
    capsule chain's fixed joint held its first segment within 0.2 degrees.

    The bands differ by engine because the matter does. The Cosserat rod is
    stated at E = 1e9 (EI = 0.064 N m^2), where a clamped beam sags about
    6 mm; measured, its tip settles 5.6 mm down and its first segment stays
    level. The capsule chain is stated by a VBD bend stiffness of 1e2, which
    sags more and rings: its tip swung between 14.6 and 25.9 mm down with its
    first segment within 0.2 degrees and the anchor 0.05 mm off its mark.

    A clamp holds twist too. A level L whose free leg twists the anchored one
    about its axis keeps its tip within 0.715 mm of level on SuperDex at either
    end; with the twist left free -- or held on the last node's padding DoF --
    the free leg swings 100 mm down about the anchored one.
    """

    def _assert_clamped(self, got: dict, *, dip: float, drop: tuple[float, float], moved: float) -> None:
        self.assertTrue(got["finite"])
        self.assertLess(got["dip"], dip, f"the first segment dipped {got['dip']:.1f} deg: {got}")
        self.assertLess(got["moved"], moved, f"the anchor moved {got['moved'] * 1e3:.2f} mm: {got}")
        # Sags rather than swings: a pinned 300 mm cable's tip falls ~300 mm.
        self.assertGreater(got["drop"], drop[0], f"the tip did not sag: {got}")
        self.assertLess(got["drop_max"], drop[1], f"the tip fell {got['drop_max'] * 1e3:.0f} mm: {got}")

    def test_the_capsule_chain(self):
        self._assert_clamped(_cantilever("vbd", bend_ke=1.0e2), dip=1.0, drop=(0.002, 0.05), moved=5.0e-4)

    @unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
    def test_the_cosserat_rod(self):
        self._assert_clamped(_cantilever("superdex", young_modulus=1.0e9), dip=0.5, drop=(0.002, 0.012), moved=1.0e-4)

    @unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
    def test_the_cosserat_rod_holds_its_twist_at_either_end(self):
        for anchor in ("start", "end"):
            with self.subTest(anchor=anchor):
                got = _twisted(anchor)
                self.assertTrue(got["finite"])
                self.assertLess(got["moved"], 1.0e-4, f"the anchor moved {got['moved'] * 1e3:.2f} mm: {got}")
                self.assertLess(got["drop"], 0.002, f"the free leg swung {got['drop'] * 1e3:.1f} mm down: {got}")


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestATetFileIsInactiveToNewton(unittest.TestCase):
    """A soft body read from a file was left ACTIVE, so VBD integrated it too."""

    def test_its_particles_carry_no_active_flag(self):
        import tempfile
        from pathlib import Path

        import newton
        from robosimstudio.domains.soft import SoftSpec

        import robosimstudio as rss
        from robosimstudio.specs import AssetSpec

        v = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]], dtype=np.float64)
        t = np.array([[0, 1, 2, 3]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tet.npz"
            np.savez(path, vertices=v, tets=t)
            scene = rss.Scene(cell="bench_075", physics=_arrangement(), name="tetfile")
            body = scene.add(
                SoftSpec(name="tet", asset=AssetSpec(path=str(path)), pose=(0.0, 0.0, 0.01), settle_frames=0)
            )
            sim = scene.build()
            flags = sim.pipeline.model.particle_flags.numpy()[np.asarray(body.particles)]
        self.assertFalse(np.any(flags & int(newton.ParticleFlags.ACTIVE)))


if __name__ == "__main__":
    unittest.main()
