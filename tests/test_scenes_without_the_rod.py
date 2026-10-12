# SPDX-License-Identifier: Apache-2.0
"""Scenes build and step with the rod pack left out.

``tests/test_rod_is_a_pack.py`` asks the registries; this asks the simulator.
A tree without ``examples/10_rope/`` still has to put a robot, rigid bodies, a
post, a sheet and a sponge on a bench and step them -- and the post is the one
that used to be impossible: ``rss.post`` named an entity only the rod domain
registered, so the constructor was public and the thing it built was not.

One subprocess, with the domain roots rebuilt without the pack (the same
mechanism the registry test uses) and the task roots switched off, so nothing a
task directory imports can put the pack back. Each scene reports a number that
is only right if the thing is really there:

* the **post**: a plate dropped onto it comes to rest one post-height above a
  cube resting on the bench beside it. A post that built as nothing would let
  the plate fall to the bench.
* the **sheet** and the **sponge**: every particle finite and on the bench
  after the steps, on the solver ``GENERIC`` routes each to.
* **all of them with a robot**: one arrangement, one model, still finite.

The sponge needs the ``superdex`` wheel and is skipped by name without it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path

from tests.test_rod_is_a_pack import roots_without_the_pack

TIER = "medium"
"""A GPU and four small scenes."""

REPO = Path(__file__).resolve().parents[1]
HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None

SCENES = """
import importlib.util, json
import numpy as np
import robosimstudio as rss
from robosimstudio.specs import CollisionSpec, coupler

assert importlib.util.find_spec("robosimstudio.domains.rod") is None, "the rod pack is importable; this proves nothing"
assert not hasattr(rss, "rod")
WITH_SPONGE = __SPONGE__
out = {}


def particles(sim):
    q = np.asarray(sim.pipeline.state_0.particle_q.numpy(), dtype=float)
    assert np.isfinite(q).all() and sim.state_is_finite()
    return q


# A post 100 mm tall, a cube beside it on the bench, a plate dropped onto the post's top.
scene = rss.Scene(cell="bench_075", name="post")
scene.add(rss.post(name="post", radius=0.03, half_height=0.05, at=(0.15, 0.0, 0.05)))
cube = scene.add(rss.box(name="cube", half=(0.01,) * 3, at=(-0.10, 0.0, 0.05)))
plate = scene.add(rss.box(name="plate", half=(0.03, 0.03, 0.01), at=(0.15, 0.0, 0.13)))
sim = scene.build()
entity = sim.scene.by_name["post"]
sim.run(180).sync()
assert sim.state_is_finite()
out["post_entity"] = type(entity).__module__ + "." + type(entity).__qualname__
out["plate_over_cube"] = float(plate.position[2]) - float(cube.position[2])
bench_top = float(cube.position[2]) - 0.01

physics = rss.replace(rss.GENERIC, coupling=coupler("one_way"), collision=CollisionSpec(full_surface_contact=True))
scene = rss.Scene(cell="bench_075", name="sheet", physics=physics)
scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(12, 12), at=(0.0, 0.0, 0.05)))
sim = scene.build()
sim.run(60).sync()
q = particles(sim)
out["sheet"] = {"particles": len(q), "lowest_over_bench": float(q[:, 2].min()) - bench_top}

if WITH_SPONGE:
    scene = rss.Scene(cell="bench_075", name="sponge")
    scene.add(rss.sponge(name="sponge", at=(0.0, 0.0, 0.03)))
    sim = scene.build()
    sim.run(60).sync()
    q = particles(sim)
    out["sponge"] = {"particles": len(q), "lowest_over_bench": float(q[:, 2].min()) - bench_top}

scene = rss.Scene(cell="bench_075", name="everything")
scene.add(rss.robot("yam_bimanual"))
scene.add(rss.post(name="post", radius=0.015, half_height=0.05, at=(0.15, 0.20, 0.05)))
cube = scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.20, 0.10)))
scene.add(rss.sheet(name="towel", size=(0.16, 0.16), cells=(10, 10), at=(0.0, -0.20, 0.05)))
if WITH_SPONGE:
    scene.add(rss.sponge(name="sponge", at=(0.20, -0.05, 0.03)))
sim = scene.build()
sim.run(30).sync()
q = particles(sim)
out["everything"] = {
    "robot": sim.scene.embodiment.spec.name,
    "arms": len(sim.scene.embodiment.arms),
    "entities": sorted(sim.scene.by_name),
    "particles": len(q),
    "cube_over_bench": float(cube.position[2]) - bench_top,
}
print(json.dumps(out))
"""


class TestScenesBuildWithoutTheRodPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio import roots

        with ExitStack() as stack:
            env = {**os.environ, **roots_without_the_pack(stack), f"{roots.ROOTS_ENV}_ONLY": "1"}
            proc = subprocess.run(
                [sys.executable, "-c", SCENES.replace("__SPONGE__", repr(HAS_SUPERDEX))],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=1500,
                check=False,
            )
        if proc.returncode != 0:
            raise AssertionError(f"a scene did not build and step without the rod pack:\n{proc.stderr[-6000:]}")
        cls.got = json.loads([line for line in proc.stdout.splitlines() if line.startswith("{")][-1])

    def test_a_post_is_built_by_the_static_domain_and_holds_a_plate_up(self) -> None:
        self.assertEqual("robosimstudio.domains.static.posts.EntityPost", self.got["post_entity"])
        self.assertAlmostEqual(
            0.10, self.got["plate_over_cube"], delta=0.005, msg="the plate is not resting on a 100 mm post"
        )

    def test_a_sheet_steps_and_stays_on_the_bench(self) -> None:
        sheet = self.got["sheet"]
        self.assertEqual(13 * 13, sheet["particles"])
        self.assertGreater(sheet["lowest_over_bench"], -0.005, "the sheet went through the bench")

    @unittest.skipUnless(HAS_SUPERDEX, "the superdex wheel is not installed, so no engine serves a sponge here")
    def test_a_sponge_steps_and_stays_on_the_bench(self) -> None:
        sponge = self.got["sponge"]
        self.assertGreater(sponge["particles"], 0)
        self.assertGreater(sponge["lowest_over_bench"], -0.005, "the sponge went through the bench")

    def test_a_robot_shares_one_scene_with_all_of_them(self) -> None:
        everything = self.got["everything"]
        self.assertEqual("yam_bimanual", everything["robot"])
        self.assertEqual(2, everything["arms"])
        expected = {"cube", "post", "towel"} | ({"sponge"} if HAS_SUPERDEX else set())
        self.assertLessEqual(expected, set(everything["entities"]))
        self.assertGreater(everything["cube_over_bench"], 0.0, "the cube fell through the bench")


if __name__ == "__main__":
    unittest.main()
