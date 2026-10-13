# SPDX-License-Identifier: Apache-2.0
"""What a policy installs during an episode must not survive it.

Its own module rather than another eighty lines of `test_core.py`, which the
layout ratchet records as oversized and may only shrink -- and which was the
right call anyway: this is one claim about one lifetime, and it earned a name.
"""

from __future__ import annotations

import types
import unittest

TIER = "fast"
"""No GPU, no scene. The claim is about lifetime, not about physics."""


class TestEpisodeHooksDoNotOutliveTheirEpisode(unittest.TestCase):
    """A pin installed by a policy must not still be holding next episode.

    A tool may install a `ParticleDragger` into `pipeline.hooks.post_solve` and
    a follower into `post_step_hooks`, and release both when it finishes. An
    episode that ends mid-tool never reaches the release, so the reset has to.

    Tested here with counters rather than a scene, because the claim is about
    *lifetime* and not about physics: hooks installed while the scene is built
    survive a reset, hooks installed after it do not, and a captured graph that
    holds a retired hook is dropped.
    """

    def _env(self):
        """An `EnvBase` with only the two hook lists and a pipeline, built without
        `__init__` -- everything else it holds is a simulator."""
        from robosimstudio.core.env import EnvBase

        env = EnvBase.__new__(EnvBase)
        env.post_step_hooks = []
        env.teleop_hooks = []
        env._episode_hook_floor = None
        env.pipeline = types.SimpleNamespace(hooks=types.SimpleNamespace(post_solve=[]), invalidated=[])
        env.pipeline.invalidate_graph = lambda *, recapture=True: env.pipeline.invalidated.append(recapture)
        return env

    def test_retiring_a_substep_hook_drops_the_captured_graph(self) -> None:
        """A captured graph replays the hooks it was captured with, so a retired
        hook would go on running inside it unless the graph is re-captured."""
        env = self._env()
        env._retire_episode_hooks()  # the floor
        env._retire_episode_hooks()  # nothing installed: the graph stays
        self.assertEqual(env.pipeline.invalidated, [])

        env.pipeline.hooks.post_solve.append(object())
        env._retire_episode_hooks()
        self.assertEqual(env.pipeline.invalidated, [True], "a removed hook must invalidate, with a re-capture")

        env.post_step_hooks.append(object())  # host-side only: no graph holds it
        env._retire_episode_hooks()
        self.assertEqual(env.pipeline.invalidated, [True])

    def test_scene_hooks_survive_and_episode_hooks_do_not(self) -> None:
        env = self._env()
        scene_hook, scene_step = object(), object()
        env.pipeline.hooks.post_solve.append(scene_hook)
        env.post_step_hooks.append(scene_step)

        env._retire_episode_hooks()  # the first reset: this is the floor

        env.pipeline.hooks.post_solve.append(object())  # a dragger, mid-episode
        env.post_step_hooks.append(object())
        env._retire_episode_hooks()  # the next reset

        self.assertEqual(env.pipeline.hooks.post_solve, [scene_hook], "the scene's own hook was dropped")
        self.assertEqual(env.post_step_hooks, [scene_step])

    def test_the_floor_is_taken_at_the_first_reset_not_at_construction(self) -> None:
        """Entities install hooks while the scene is built -- the tape's bond
        field, the gravity ramp. Those are the scene, not the episode, and a
        floor taken in `__init__` would retire them on the first reset."""
        env = self._env()
        self.assertIsNone(env._episode_hook_floor)
        env.pipeline.hooks.post_solve.extend([object(), object()])
        env._retire_episode_hooks()
        self.assertEqual(env._episode_hook_floor, (2, 0))
        self.assertEqual(len(env.pipeline.hooks.post_solve), 2, "the build's own hooks must survive")

    def test_it_is_idempotent_across_many_episodes(self) -> None:
        """Ten resets with a dragger installed each time leaves the floor exact.

        A leak that grew by one per episode would be slower to notice than the
        one this replaces and no less wrong.
        """
        env = self._env()
        env.pipeline.hooks.post_solve.append(object())
        env._retire_episode_hooks()
        for _ in range(10):
            env.pipeline.hooks.post_solve.append(object())
            env.post_step_hooks.append(object())
            env._retire_episode_hooks()
        self.assertEqual(len(env.pipeline.hooks.post_solve), 1)
        self.assertEqual(env.post_step_hooks, [])

    def test_reset_actually_retires_them(self) -> None:
        """The wiring, not the arithmetic.

        The three tests above call `_retire_episode_hooks` directly, so every one
        of them passes with the call removed from `reset()` -- which is the shape
        of mistake that let this bug exist in the first place: the pieces were
        all present and one caller did not use them. Verified by deleting the
        call, watching this test alone go red, and putting it back.
        """
        from robosimstudio.core.env import EnvBase

        env = self._env()
        env.warmup = False
        env._warmed = True
        env.step_count = 7
        env.termination_request = "stale"
        env.controllers = {}
        env.obs_providers = {}
        env.pipeline.sim_time = 3.0
        env.pipeline.step_count = 9
        env.reset_scene = lambda: None
        env.observation = lambda: {}
        env.reset_info = lambda: {}

        env._retire_episode_hooks()  # the floor
        leaked = object()
        env.pipeline.hooks.post_solve.append(leaked)

        EnvBase.reset(env)

        self.assertNotIn(leaked, env.pipeline.hooks.post_solve, "reset() must drop what the episode installed")
