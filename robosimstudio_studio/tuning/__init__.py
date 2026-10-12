# SPDX-License-Identifier: Apache-2.0
"""Tuning deformable physics with the simulation running.

Not a second page. These are the panels and the engine behind them, mounted by
:mod:`robosimstudio_studio` whenever the compiled scene has something
soft in it -- so the sliders come up beside the robot, the policy runner and the
episode score.

That placement is the design, not a convenience. A bare strip can tell you a
bending length, and that is worth having; it cannot tell you whether the shirt
folds, whether the jaws keep hold of it, or whether the graph gets through its
second crease. Those are what a material is *for*, and they only exist in a task
with the real asset and the robot in it.

What makes the loop possible is a fact about Newton rather than a UI idea: most
of a cloth's constitutive parameters are Warp arrays the solver re-reads every
substep, so they can be rewritten mid-simulation and take effect immediately --
even under CUDA graph capture, which bakes the array's pointer and not the
numbers in it. :mod:`.apply` does the writing, :mod:`.tuner` holds the working
material for every deformable in a scene, and
:mod:`robosimstudio.specs.params` is where each parameter says which of the
three tiers it belongs to.

Two supporting pieces:

* :mod:`.metrics` reports in the units cloth is actually specified in -- a
  bending length in centimetres against the band for real terry towelling, not
  a stiffness.
* :mod:`robosimstudio.assetlib.materials` gives materials names and the
  measurement behind them, so a tuned sheet can be saved and adopted by a task
  deliberately rather than by a number somebody pasted.

When a number rather than a verdict is what is wanted, the bench is the cloth
drape bench -- ASTM D1388 in the simulator, and the procedure the
recorded measurements in :mod:`robosimstudio.assetlib.materials` came from.
"""

from .apply import ClothTarget, LiveMaterial, apply_contact, locate_cloth, rebuild_fields
from .tuner import MaterialTuner

__all__ = [
    "ClothTarget",
    "LiveMaterial",
    "MaterialTuner",
    "apply_contact",
    "locate_cloth",
    "rebuild_fields",
]
