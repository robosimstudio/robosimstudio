# SPDX-License-Identifier: Apache-2.0
"""Lift a sponge

A foam sponge under an open Panda hand: drag the hand down onto it, press **Close** and lift.

The hand starts pointing down over the sponge. Drag its gizmo straight down until
the fingertips are a centimetre over the bench, press **Close**, and drag it up:
the pads squeeze the foam and it comes up by friction alone, nothing pinned. The
arm is Newton's and the sponge is a volumetric FEM body that SuperDex steps,
which is why three settings below are the sponge's and the engine's, each with
what it measured. Needs the ``[superdex]`` extra (Python 3.12).

**Key features:**

* :func:`robosimstudio.sponge` with a foam whose ``friction_falloff`` lets a pinch hold it
* :class:`robosimstudio.ActuationSpec` ``effort_limit`` for a grip that squeezes and stops
* :func:`robosimstudio.backend` ``("superdex", feedback_dv_max=...)`` for foam that pushes the fingers back
* :func:`robosimstudio.view` to drive the arm from the page
"""

from robosimstudio.domains.soft import FIRM_FOAM

import robosimstudio as rss


def main():
    cell = "bench_075"
    # Under the Panda's home tool centre, in the bench's work-surface frame [m]: the
    # hand starts 231 mm over the bench pointing down, so the gesture is one drag.
    at = (-0.143, -0.115)
    # 50 mm across the jaws, which open 79 mm; 35 mm tall, so pads closed with the
    # tool centre 20 mm up span 11-27 mm of it.
    size = (0.090, 0.050, 0.035)
    cells = (9, 5, 4)  # ~10 mm nodes, 300 of them
    # SuperDex uses the sponge's falloff, not the hand's, against a collider it
    # drives by pose. Below it friction grows with slip and nothing is static: at
    # the engine's 10 mm/s the held foam slid 1.4 mm in 2.5 s and 4.5 mm in 10 s
    # more; at 1 mm/s, 0.1 mm and 0.1 mm.
    foam = rss.replace(FIRM_FOAM, friction_falloff=1.0e-3)
    # The pads are 10 g bodies, and the foam's push on them is capped at what moves
    # one 0.5 m/s in a control step: 0.3 N. Under that cap the jaws closed the 50 mm
    # foam to 6 mm with the pads 2-3 mm inside it; at 20 m/s the foam holds them at
    # 44 mm, the 8 N grip below and the foam's push in balance.
    feedback_dv_max = 20.0
    grip = rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=8.0)  # [N] per finger

    physics = rss.GENERIC.with_backend("soft", rss.backend("superdex", substeps=2, feedback_dv_max=feedback_dv_max))
    scene = rss.Scene(cell=cell, physics=physics, name="sponge")
    scene.add(rss.robot("panda", gripper=grip))
    scene.add(rss.sponge(name="sponge", size=size, cells=cells, at=(*at, 0.0), material=foam))
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
