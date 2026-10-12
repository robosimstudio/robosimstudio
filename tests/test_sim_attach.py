# SPDX-License-Identifier: Apache-2.0
"""``sim.attach``/``sim.detach``: the deformable hold three suites each wrote.

``latch_cloth``/``release_cloth``, ``latch_tape``/``release_tape`` and
``hold_cable``/``release_cable`` are one tool: a pinning ``ParticleDragger`` at
a hand, re-aimed every control step, over a declared particle set. Their return
shapes already agreed. The only thing that differed was which particles, and a
feature name says that.

What this guards is not the physics -- that needs a scene, and the suites'
own rollouts exercise it -- but the three properties a copy is likely to get
wrong when the seventh caller arrives:

* the tool appears only where there is something to hold, so a rigid cell does
  not offer a graph a tool that raises the moment it is believed;
* both entries are tagged privileged, because seizing particles has no
  real-robot analogue and a graph that attaches will not transfer;
* the hook is registered **and the CUDA graph invalidated**, in that order.
  Getting the second wrong is silent: the grab bookkeeping is host-side, so the
  tool reports the particles it seized while nothing holds them.

No GPU: a stub env is the whole fixture.
"""

from __future__ import annotations

import types
import unittest
import unittest.mock

from robosimstudio_policy_agent.sim_tools import ClothAttachment, register_attachment

TIER = "fast"
"""No GPU. Gates every commit."""


class _Registry:
    """Records what was claimed and under which tags."""

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def register_callable(self, name, fn, **kwargs):
        if name in self.calls:
            raise AssertionError(f"duplicate registration of {name!r}")
        self.calls[name] = {"fn": fn, **kwargs}
        return fn


class _Bus:
    """This thread owns the env, so a call is just a call."""

    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _env(*, deformables: bool) -> types.SimpleNamespace:
    thing = types.SimpleNamespace(carries_particles=True)
    return types.SimpleNamespace(scene=types.SimpleNamespace(by_name={"tape": thing} if deformables else {}))


class TestItAppearsOnlyWhereItApplies(unittest.TestCase):
    def test_a_scene_with_a_deformable_gets_both_entries(self) -> None:
        registry = _Registry()
        attachment = register_attachment(registry, _env(deformables=True), _Bus())
        self.assertIsInstance(attachment, ClothAttachment)
        self.assertEqual(sorted(registry.calls), ["sim.attach", "sim.detach"])

    def test_a_rigid_scene_gets_neither(self) -> None:
        registry = _Registry()
        self.assertIsNone(register_attachment(registry, _env(deformables=False), _Bus()))
        self.assertEqual(registry.calls, {})

    def test_both_entries_are_privileged(self) -> None:
        registry = _Registry()
        register_attachment(registry, _env(deformables=True), _Bus())
        for name, call in registry.calls.items():
            self.assertIn("privileged", tuple(call.get("tags") or ()), f"{name} is not tagged privileged")


class TestTheHookOrdering(unittest.TestCase):
    """The invalidation has to follow the append, or the pin never launches."""

    def _scene(self) -> tuple[types.SimpleNamespace, list[str]]:
        order: list[str] = []
        dragger_state = types.SimpleNamespace(grabbed=0)

        class _Dragger:
            radius = 0.0
            last_distance = 0.25
            active = False

            def __init__(self, *_a, **_k) -> None:
                pass

            def apply(self, *_a, **_k) -> None:
                pass

            def grab(self, *_a, **_k) -> int:
                dragger_state.grabbed += 1
                return 4

            def release(self) -> None:
                order.append("release")

        hooks = types.SimpleNamespace(post_solve=_Recorder(order, "append-hook"))
        pipeline = types.SimpleNamespace(hooks=hooks, invalidate_graph=lambda: order.append("invalidate"))
        arm = types.SimpleNamespace(ee_pose=lambda _state: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        env = types.SimpleNamespace(
            model=object(),
            state=object(),
            pipeline=pipeline,
            post_step_hooks=[],
            handles=types.SimpleNamespace(arms={"left": object()}, work_arm="left"),
            task=types.SimpleNamespace(deformables=[types.SimpleNamespace(name="tape")]),
            scene=types.SimpleNamespace(
                by_name={"tape": types.SimpleNamespace(particles=[0, 1, 2, 3], carries_particles=True)}
            ),
            arm=lambda _name: arm,
        )
        return env, order, _Dragger

    def test_the_graph_is_invalidated_after_the_hook_is_appended(self) -> None:
        env, order, dragger = self._scene()
        attachment = ClothAttachment(env, _Bus())
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", dragger):
            result = attachment.attach()
        self.assertEqual(order, ["append-hook", "invalidate"], "the pin's kernel is captured before it exists")
        self.assertTrue(result["attached"])
        self.assertEqual(result["n_grabbed"], 4)

    def test_a_miss_reports_the_distance_that_missed(self) -> None:
        env, _order, dragger = self._scene()
        dragger.grab = lambda self, *a, **k: 0  # type: ignore[assignment]
        attachment = ClothAttachment(env, _Bus())
        with unittest.mock.patch("robosimstudio.core.draggers.ParticleDragger", dragger):
            result = attachment.attach()
        self.assertFalse(result["attached"])
        self.assertEqual(result["n_grabbed"], 0)
        self.assertGreater(result["nearest_m"], 0.0, "a miss that does not say how far off it was is not routable")


class _Recorder(list):
    """A list that records the fact of an append, for ordering assertions."""

    def __init__(self, order: list[str], label: str) -> None:
        super().__init__()
        self._order, self._label = order, label

    def append(self, item) -> None:
        self._order.append(self._label)
        super().append(item)


if __name__ == "__main__":
    unittest.main()
