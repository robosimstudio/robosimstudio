# SPDX-License-Identifier: Apache-2.0
"""The compiler's contract, and who satisfies it.

:func:`~robosimstudio.scene.compiler.compile_scene` used to take a ``TaskSpec``
and read ten of its twenty-six fields. Those ten are now
:class:`~robosimstudio.scene.request.SceneRequest`, a protocol, and the claim
this module pins is the one that makes the split worth anything:

* **``TaskSpec`` still satisfies it, unedited.** Nothing about a task changed to
  make a bare scene possible. If somebody adds a field to the protocol without
  adding it to ``TaskSpec``, or renames one on ``TaskSpec``, this fails here --
  at import speed, with no GPU -- rather than when the next person compiles a
  suite.
* **A ``BuildRequest`` satisfies it too**, and it carries no evaluator, no
  episodes, no goal and no instruction. That is the whole point: the compiler
  cannot tell the difference, so a scene does not need a benchmark.

Static: no scene, no GPU, no registries loaded.
"""

from __future__ import annotations

import typing
import unittest

from robosimstudio.scene.request import BuildRequest, SceneRequest
from robosimstudio.specs import ArrangementSpec, Binding, RegionSpec, TaskSpec

TIER = "fast"
"""No GPU. Gates every commit."""

CONTRACT = (
    "physics",
    "embodiment",
    "embodiment_pose",
    "arm_actuation",
    "gripper_actuation",
    "regions",
    "full_name",
    "statics",
    "build_order",
    "binding_for",
)
"""Every member the compiler reads. Stated here as well as in the protocol so
that a member added to one and not the other is a failure rather than a
coincidence -- ``runtime_checkable`` only checks the members it was given."""


class TestTheContractIsWhatTheCompilerReads(unittest.TestCase):
    def test_the_protocol_names_exactly_the_contract(self) -> None:
        # ``__protocol_attrs__`` is Python 3.12's; before it the same set comes from typing's own helper.
        declared = getattr(SceneRequest, "__protocol_attrs__", None)
        if declared is None:
            declared = typing._get_protocol_attrs(SceneRequest)
        declared = set(declared)
        self.assertEqual(
            sorted(declared),
            sorted(CONTRACT),
            "the protocol and this module disagree about what the compiler asks for",
        )

    def test_a_task_satisfies_it_unedited(self) -> None:
        """The load-bearing claim: no task changed to make a bare scene possible."""
        self.assertIsInstance(TaskSpec(name="t"), SceneRequest)

    def test_a_build_request_satisfies_it(self) -> None:
        self.assertIsInstance(BuildRequest(), SceneRequest)

    def test_a_build_request_carries_no_benchmark(self) -> None:
        """It is a scene, so it has no opinion about what would count as done."""
        request = BuildRequest()
        for absent in ("evaluator", "episodes", "instruction", "routine", "graphs", "max_steps"):
            self.assertFalse(
                hasattr(request, absent),
                f"a BuildRequest carries {absent!r}; that is the benchmark's, not the scene's",
            )


class TestABuildRequestIsAScene(unittest.TestCase):
    def test_the_stages_come_out_in_build_order(self) -> None:
        """The compiler's fixed order, over specs that name their own stage."""

        class _At:
            def __init__(self, stage: str, name: str) -> None:
                self.stage, self.name = stage, name

        specs = (_At("extras", "e"), _At("scenery", "s"), _At("deformables", "d"), _At("objects", "o"))
        request = BuildRequest(entities=specs)
        self.assertEqual(["s", "o", "d", "e"], [s.name for s in request.build_order()])
        self.assertEqual(("s",), tuple(s.name for s in request.statics))
        self.assertEqual(("o",), tuple(s.name for s in request.objects))
        self.assertEqual(("d",), tuple(s.name for s in request.deformables))
        self.assertEqual(("e",), tuple(s.name for s in request.extras))

    def test_the_cell_and_the_robot_become_the_pairing(self) -> None:
        request = BuildRequest(cell="bench_075", embodiment="panda")
        binding = request.binding_for()
        self.assertEqual("bench_075", binding.scene)
        self.assertEqual("panda", binding.embodiment)

    def test_an_explicit_binding_wins(self) -> None:
        """A caller that already has a pairing hands it in rather than restating it."""
        binding = Binding(scene="void", embodiment="ur5", work_arm="right")
        self.assertIs(binding, BuildRequest(binding=binding).binding_for())

    def test_asking_for_a_pairing_it_is_not_says_so(self) -> None:
        """A request is one scene. Asked to be another, it refuses by name.

        A task refuses an undeclared pairing because a robot mounted at a pose
        nobody chose builds a scene that looks plausible and reaches nothing.
        The same argument applies here, and silently building the only pairing
        there is would be the failure that argument is about.
        """
        request = BuildRequest(cell="bench_075", embodiment="panda", name="mine")
        with self.assertRaises(KeyError) as caught:
            request.binding_for(embodiment="ur5")
        self.assertIn("mine", str(caught.exception))
        self.assertIn("ur5", str(caught.exception))

    def test_it_is_hashable(self) -> None:
        """Frozen, so a compiler may cache on one -- the rule every spec here follows."""
        self.assertEqual(
            hash(BuildRequest(cell="bench_075", physics=ArrangementSpec())),
            hash(BuildRequest(cell="bench_075", physics=ArrangementSpec())),
        )

    def test_regions_are_geometry_not_scoring(self) -> None:
        """A region is a named box in a frame; who reads it is not the scene's business."""
        region = RegionSpec(name="bin", frame="table", xy_ranges=((-0.1, -0.1, 0.1, 0.1),))
        self.assertEqual((region,), BuildRequest(regions=(region,)).regions)


if __name__ == "__main__":
    unittest.main()
