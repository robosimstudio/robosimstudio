# SPDX-License-Identifier: Apache-2.0
"""The environment a policy is handed: one scene, or a fleet of them.

Where the spec-driven surface lives. :class:`~robosimstudio_bench.envs.env.Env`
takes a task name, compiles it, and steps it; :class:`~robosimstudio_bench.envs.fleet.FleetEnv`
is the same contract over many worlds behind one model, which is the shape a
batched policy is written against.

Distinct from :mod:`robosimstudio.core`, which holds the *mechanisms* -- the
substep loop, the probes, the renderers -- and knows nothing about tasks. This
package is the one that does: it resolves a task, refuses an inadmissible
pairing, installs the evaluator, and answers ``observe()``.

The single/fleet pair is a shape repeated deliberately, once here and once in
:mod:`robosimstudio_bench.solutions`, so that "the batched twin of X" is always a name
you can guess.
"""

from __future__ import annotations

from .env import Env, make_env
from .fleet import FleetEnv

__all__ = ["Env", "FleetEnv", "make_env"]
