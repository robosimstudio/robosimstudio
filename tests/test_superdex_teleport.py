# SPDX-License-Identifier: Apache-2.0
"""A stress-free rod's teleport onto its starting polyline moves nothing else in its world.

A SuperDex rod takes no node positions, so a cable whose stress-free shape is
not where it starts is built at that shape and carried onto its polyline by two
microsecond steps of the whole scene. Every actor built before it takes those
steps too. They pushed a weight a cable was joined into, set an earlier
stress-free rod moving at 1.83 m/s, and with any contact live across them --
the rod's own self contact included -- made the engine abort the process, so
each scene here builds in a child process, where an abort fails the test
instead of ending the run.

Split from :mod:`tests.test_superdex_contact`, whose scenes and helpers it
shares. Bands, not bits: the engine documents no determinism guarantee.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_superdex_contact import _arrangement, _hanging_cable, _importable

TIER = "medium"
"""A GPU, the [superdex] extra, and a child process per scene."""


def _laid_l(name: str, x: float, **kw):
    """A 3 mm cable laid out as an L from 400 mm over ``x``: 100 mm down, then 100 mm across."""
    from robosimstudio.domains.rod.spec import RodSpec

    top = np.array([x, 0.0, 0.40])
    down = [top + np.array([0.0, 0.0, -0.005 * i]) for i in range(21)]
    out = [down[-1] + np.array([0.005 * i, 0.0, 0.0]) for i in range(1, 21)]
    return RodSpec(
        name=name,
        nodes=tuple(tuple(float(v) for v in p) for p in down + out),
        radius=0.003,
        density=1400.0,
        axial_stiffness=2.0e4,
        anchors=("start",),
        **{"young_modulus": 3.0e6, **kw},
    )


def _coil(name: str, pitch: float, **kw):
    """A 3 mm cable wound three turns round a 15 mm radius from 100 mm up, ``pitch`` [m] a turn, anchored at its foot."""
    from robosimstudio.domains.rod.spec import RodSpec

    t = np.linspace(0.0, 3 * 2 * np.pi, 120)
    nodes = np.stack([0.015 * np.cos(t), 0.015 * np.sin(t), 0.10 + pitch * t / (2 * np.pi)], axis=1)
    return RodSpec(
        name=name,
        nodes=tuple(tuple(float(v) for v in p) for p in nodes),
        radius=0.003,
        density=1400.0,
        young_modulus=3.0e6,
        axial_stiffness=2.0e4,
        anchors=("start",),
        **kw,
    )


def _run_child(test: unittest.TestCase, child: str, args: tuple, result: str) -> dict:
    """Run this module's ``child(*args)`` in a child process; return the JSON it printed after ``result``.

    Fails ``test`` -- rather than ending the run -- when the child dies (an
    engine abort is SIGILL, -4), prints no result, or is not finite after the
    ten steps it reports as ``STEPPED <finite>``.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    call = f"{child}{args!r}"
    done = subprocess.run(
        [sys.executable, "-c", f"from tests.test_superdex_teleport import {child}; {call}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith((f"{result} ", "STEPPED "))]
    test.assertEqual(
        done.returncode,
        0,
        f"{call} died with exit {done.returncode} (an engine abort is SIGILL, -4) after printing {lines}:\n"
        f"{done.stderr[-3000:]}",
    )
    found = next((ln for ln in lines if ln.startswith(f"{result} ")), None)
    test.assertIsNotNone(found, f"the child printed no result:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    test.assertIn("STEPPED True", lines, f"ten steps after the build are not finite: {lines}")
    return json.loads(found.split(" ", 1)[1])


def _self_touching(shape: str, self_contact: bool) -> None:
    """The child process of :class:`TestATeleportKeepsSelfContact`: a stress-free rod with or without self contact.

    Prints ``SELF {json}`` -- how far its farthest node moved in the ten steps
    after the build [m] -- then ``STEPPED <finite>``.
    """
    import json

    import robosimstudio as rss
    from robosimstudio.specs.physics import SimRequirements

    needs = SimRequirements(domain="rod", self_contact=self_contact)
    scene = rss.Scene(cell="bench_075", physics=_arrangement(), name=f"self_{shape}")
    if shape == "l":
        rod = scene.add(_laid_l("cable", 0.0, stress_free="straight", sim=needs))
    elif shape == "coil":
        rod = scene.add(_coil("cable", 0.008, stress_free="straight", sim=needs))
    else:
        raise ValueError(shape)
    built = scene.build()
    start = np.asarray(rod.particle_q, dtype=np.float64).copy()
    built.run(10).sync()
    moved = float(np.linalg.norm(np.asarray(rod.particle_q, dtype=np.float64) - start, axis=1).max())
    print("SELF " + json.dumps({"moved": moved}), flush=True)
    print(f"STEPPED {bool(built.state_is_finite())}", flush=True)


def _fastest(components: dict, stride: int, reach: float) -> float:
    """The fastest point of one actor in an engine state dump [m/s].

    Its node velocity slices (``stride`` DoFs a node, the first three a
    velocity), and every rigid twist anywhere in it -- a skeleton's joint, a
    link, their integrator history -- as ``|v| + |w| * reach``.
    """
    fastest = 0.0

    def walk(value) -> None:
        nonlocal fastest
        if isinstance(value, dict):
            if "vcom" in value and "omega" in value:
                spin = np.linalg.norm(np.asarray(value["omega"], dtype=np.float64)[:3]) * reach
                fastest = max(fastest, float(np.linalg.norm(np.asarray(value["vcom"], dtype=np.float64)[:3]) + spin))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for key, component in components.items():
        if "VelocityVectorMetadata" in key and "Current" in key:
            velocity = np.asarray(component["value"], dtype=np.float64).reshape(-1, stride)[:, :3]
            fastest = max(fastest, float(np.linalg.norm(velocity, axis=1).max()))
        else:
            walk(component)
    return fastest


def _built_still(scenario: str) -> None:
    """The child process of :class:`TestATeleportMovesNothingElse`: build a scene, print where everything is.

    Prints a ``BUILT_STILL {json}`` line -- per actor, how far its nodes are
    from where they were built [m] and its fastest node [m/s], read off the
    engine's whole-scene state, the one place a node velocity can be read --
    then steps ten times and prints ``STEPPED <finite>``.
    """
    import json
    import tempfile
    from pathlib import Path

    from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

    import robosimstudio as rss
    from robosimstudio.engines.superdex.bridge import SuperdexBridge

    scene = rss.Scene(cell="bench_075", physics=_arrangement(), name=f"teleport_{scenario}")
    if scenario in ("join", "core"):
        # A cable joined into a weight, then a second one, stress-free, joined beside it: the
        # first pair's contact is live inside the weight when the second rod's steps run. On a
        # rigid core ("core") the weight is a soft-skinned actor, which the teleport rests whole.
        core = scenario == "core"
        scene.add(
            SoftSpec(
                name="weight",
                size=(0.03,) * 3,
                cells=(2, 2, 3),
                pose=(0.0, 0.0, 0.17),
                material=SoftMaterialSpec(young=1.0e9 if core else 1.0e6, density=1200.0 if core else 300.0),
                rigid=(lambda rest: rest[:, 2] < rest[:, 2].max() - 1.0e-6) if core else None,
                settle_frames=0,
            )
        )
        joined = {
            "nodes": 23,
            "joins": (("end", "weight", 0.021),),
            "contact_penalty": 1.0e11,
            "contact_threshold": 5.0e-5,
        }
        scene.add(_hanging_cable(name="a", **joined))
        scene.add(_hanging_cable(name="b", at=(0.015, 0.015), stress_free="straight", **joined))
    elif scenario == "rods":
        # Two stress-free cables 200 mm apart: the first carries its own elastic stress into the
        # second one's steps.
        scene.add(_laid_l("a", 0.0, young_modulus=1.0e9, stress_free="straight"))
        scene.add(_laid_l("b", 0.2, young_modulus=1.0e9, stress_free="straight"))
    else:
        raise ValueError(scenario)
    sim = scene.build()
    bridge = next(
        host.bridge
        for host in sim.pipeline.program.hosts.values()
        if isinstance(getattr(host, "bridge", None), SuperdexBridge)
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        bridge.scene_of(0).capture_state_to_file(str(path))
        state = json.loads(path.read_text())["actors"]
    out = {}
    layout = bridge.model.particle_q.numpy().astype(np.float64)
    for entry in bridge.actors:
        built = layout[entry["start"] : entry["start"] + entry["count"]]
        displacements = np.asarray(entry["actor"].get_displacements(), dtype=np.float64)
        stride = len(displacements) // len(built)  # 4 for a rod (a twist per node), 3 for a soft body
        parts = [state[entry["name"]]] if entry["name"] in state else []
        if "skinned" in entry:
            # On a rigid core the displacements are world-scale and the motion is split between the
            # soft part and its skeleton: read the nodes through the root, every velocity under it.
            nodes = entry["read"]()
            moved = np.linalg.norm(nodes - built, axis=1).max()
            parts = [actor for name, actor in state.items() if name.startswith(f"{entry['name']}:skeleton")]
        elif stride == 4:  # a rod is built at its stress-free shape: rest plus displacements
            nodes = entry["read"]()
            moved = np.linalg.norm(nodes - built, axis=1).max()
        else:  # a soft body is built where it is: its displacements are the motion
            nodes = built
            moved = np.abs(displacements).max()
        reach = float(np.linalg.norm(nodes - nodes.mean(axis=0), axis=1).max())
        speed = max(_fastest(part["components"], stride, reach) for part in parts)
        out[entry["entity"]] = {"moved": float(moved), "speed": speed, "parts": len(parts)}
    # Printed before stepping: an abort at the first step still leaves the numbers behind.
    print("BUILT_STILL " + json.dumps(out), flush=True)
    sim.run(10).sync()
    print(f"STEPPED {bool(sim.state_is_finite())}", flush=True)


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestATeleportMovesNothingElse(unittest.TestCase):
    """A stress-free rod's teleport leaves every actor built before it where it was, and at rest.

    The teleport's two microsecond steps are whole-scene steps. Before they were
    isolated, any contact live across them -- a cable joined into a weight, a
    cable pressed into a block -- made the engine abort the process before the
    first control step ("Expected a force vector for every contact between
    this pair of entities", SIGILL), with the weight's nodes already kicked to
    0.025 m/s; and a second stress-free rod's steps left the first 3.2 um off
    its mark and moving at 1.83 m/s (E = 1e9).

    Each scene builds in a child process: a regression here is a process abort,
    which must fail this test rather than end the run. Measured now: every
    actor exactly where it was built and exactly at rest; the bands are a
    micrometre and a micrometre per second.

    A weight on a rigid core is a soft-skinned actor, which takes no node
    velocity ("SetNodeVelocitiesLocal is not supported for soft-skinned
    actors" ended the build when the reset treated it as a plain soft body),
    so the teleport rests its soft part whole and zeroes its skeleton's joint.
    Its nodes are read through its root and every velocity under its skeleton
    -- the soft part, the link, the joint, their history. That case guards
    the build, not the reset's effect: with gravity left on and no reset it
    reads exactly zero too, the 2 us of gravity moving nothing in the solve.
    """

    MOVED = 1.0e-6
    """[m]"""
    SPEED = 1.0e-6
    """[m/s]"""

    def _built(self, scenario: str) -> dict:
        return _run_child(self, "_built_still", (scenario,), "BUILT_STILL")

    def _assert_still(self, got: dict) -> None:
        for name, actor in got.items():
            with self.subTest(actor=name):
                self.assertLess(actor["moved"], self.MOVED, f"{name!r} is {actor['moved'] * 1e6:.3g} um off: {got}")
                self.assertLess(actor["speed"], self.SPEED, f"{name!r} moves at {actor['speed']:.3g} m/s: {got}")

    def test_a_joined_pair_built_before_it(self):
        self._assert_still(self._built("join"))

    def test_a_stress_free_rod_built_before_it(self):
        self._assert_still(self._built("rods"))

    def test_a_body_on_a_rigid_core_built_before_it(self):
        got = self._built("core")
        self.assertEqual(got["weight"]["parts"], 3, f"the skeleton, its soft part and its link: {got}")
        self._assert_still(got)


@unittest.skipUnless(_importable("superdex.physics"), "the [superdex] extra is not installed")
class TestATeleportKeepsSelfContact(unittest.TestCase):
    """A stress-free rod that touches itself builds, and still touches itself after.

    The teleport switched every pair between actors off but left the rod's own
    self pair on, and a rod stated with ``self_contact`` aborted the process at
    build (exit 132, the same "Expected a force vector" abort) -- an L, a U
    and a three-turn helix alike. Its self pair is now off for the two steps
    and back on after: measured, a coil of 8 mm pitch settles 2.04 mm in its
    first ten steps with self contact and 12.40 mm without, its turns passing
    through each other.
    """

    def test_an_l_with_self_contact_builds(self):
        _run_child(self, "_self_touching", ("l", True), "SELF")  # exits 0 and is finite ten steps on

    def test_a_coil_still_touches_itself(self):
        on = _run_child(self, "_self_touching", ("coil", True), "SELF")["moved"]
        off = _run_child(self, "_self_touching", ("coil", False), "SELF")["moved"]
        self.assertLess(on, 0.5 * off, f"{on * 1e3:.2f} mm with self contact, {off * 1e3:.2f} mm without")


if __name__ == "__main__":
    unittest.main()
