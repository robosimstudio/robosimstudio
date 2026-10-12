# SPDX-License-Identifier: Apache-2.0
"""Assemble a scene, step it, read it. The core tier's whole surface.

Three objects and one rule. :class:`Scene` is what you put things into,
:class:`Sim` is what you get when you build it, and
:class:`~robosimstudio.scene.ref.Ref` is what you got back for each thing you
added. The rule is that a *kind of thing* is composed on top of this rather than
built into it: a rod, a sheet and a rigid body reach ``Scene.add`` on exactly the
terms a kind you invent does.

Nothing here knows what a task is.
"""

from __future__ import annotations

from robosimstudio.scene.ref import Ref

from .draw import Viewer, draw, view, viewer
from .scene import Robot, Scene, robot
from .sim import Sim

__all__ = ["Ref", "Robot", "Scene", "Sim", "Viewer", "draw", "robot", "view", "viewer"]
