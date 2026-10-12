# SPDX-License-Identifier: Apache-2.0
"""robosimstudio.core: the task-agnostic manipulation runtime for Newton.

What makes *one hand-built Newton scene* steppable, controllable, observable and
renderable -- mechanisms only. There are no registries here, no specs beyond
:class:`~robosimstudio.core.cameras.CameraSpec`, and no knowledge of tasks,
embodiments, policies or benchmarks. Anything that writes its own
``ModelBuilder`` code can use this the way anything can use Isaac Lab without
Arena.

The pieces:

* :mod:`~robosimstudio.core.math` -- the canonical host-side quaternion and
  transform helpers, so xyzw/wxyz conversions happen in one place.
* :mod:`~robosimstudio.core.builder` -- builder fixups, above all the
  ``collapse_fixed_joints`` custom-attribute re-key that silently ungravity-
  compensates a collapsed MJCF arm.
* :mod:`~robosimstudio.core.schedule` -- the substep program as data, and the
  executor that captures it into a CUDA graph.
* :mod:`~robosimstudio.core.pipeline` -- the substep loop over a composed
  physics program, with the batched read-back.
* :mod:`~robosimstudio.core.readback` -- one device-to-host read per control
  step, and a finite check that stays on the device.
* :mod:`~robosimstudio.core.draggers` -- teleport and weld handles over a built
  model, which the studio drags with and a task's hold tools seize through.
* :mod:`~robosimstudio.core.bus` -- marshalling reads and writes across the one
  thread allowed to touch the simulation.
* :mod:`~robosimstudio.core.probes` -- contact impulses and particle state,
  which is what success tests and deformable observations are built from.
* :mod:`~robosimstudio.core.render` -- cameras, five backends, an offline
  re-render path, and streaming video.
* :mod:`~robosimstudio.core.env` -- the environment skeleton the toolkit
  layer's spec-driven ``Env`` subclasses.

Deformable support here mirrors what Newton itself treats as generic: cloth
built through the builder API, a schedule any composed solver can run, particle
observation. There is deliberately **no adhesion or bond machinery** -- that is
task physics, and it belongs to the task that needs it.
"""

from .._version import __version__
from .builder import assert_even_substeps, rekey_custom_attributes, restate_body_attribute
from .cameras import CameraSpec
from .draggers import ObjectDragger, ParticleDragger
from .env import EnvBase, StepResult
from .pipeline import ScenePipeline, SubstepHooks
from .probes import ParticleProbe
from .readback import HostSnapshot, Readback
from .render import RendererBase, VideoSet, VideoWriter, VisualPart
from .schedule import Schedule, ScheduleExecutor, Stage

__all__ = [
    "CameraSpec",
    "EnvBase",
    "HostSnapshot",
    "ObjectDragger",
    "ParticleDragger",
    "ParticleProbe",
    "Readback",
    "RendererBase",
    "ScenePipeline",
    "Schedule",
    "ScheduleExecutor",
    "Stage",
    "StepResult",
    "SubstepHooks",
    "VideoSet",
    "VideoWriter",
    "VisualPart",
    "__version__",
    "assert_even_substeps",
    "rekey_custom_attributes",
    "restate_body_attribute",
]
