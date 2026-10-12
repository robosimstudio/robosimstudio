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

    **This is the bug that had a benchmark manufacturing successes.** A peel
    routine installs a `ParticleDragger` into `pipeline.hooks.post_solve` and a
    follower into `post_step_hooks`; both lists are built once and were never
    cleared, while a successful peel ends mid-lift with the bus closed so the
    tool's own release never runs.

    Measured on `peel/handle` before the fix: `--episodes 0:3` reported
    410, 411 and **24** steps, every one at reward 1.00 -- and at step 24 the
    workflow was still in `approach`, the gripper had never closed, and six tab
    particles were already held. The same episode run alone takes 410 steps.

    Tested here with counters rather than a scene, because the claim is about
    *lifetime* and not about physics: hooks installed while the scene is built
    survive a reset, hooks installed after it do not.

    **What this does not do.** Closing the leak did not stop the false
    successes: the same three-episode run afterwards reads 412, 58, 41 steps,
    still all at reward 1.00. The leak was real and is now closed; it was not
    the mechanism. These tests are about the lifetime and claim nothing about
    the peel.
    """

    def _env(self):
        """An `EnvBase` with only the two hook lists and a pipeline, built without
        `__init__` -- everything else it holds is a simulator."""
        from robosimstudio.core.env import EnvBase

        env = EnvBase.__new__(EnvBase)
        env.post_step_hooks = []
        env.teleop_hooks = []
        env._episode_hook_floor = None
        env.pipeline = types.SimpleNamespace(hooks=types.SimpleNamespace(post_solve=[]))
        return env

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
