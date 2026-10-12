# SPDX-License-Identifier: Apache-2.0
"""Press a sponge

A foam sponge under a Panda arm: drag the wrist gizmo down and the top face gives way under the fingertips.

The arm is Newton's and the sponge is a volumetric FEM body that SuperDex steps.
The default arrangement already routes soft bodies there, so the script says
nothing about engines. Needs the ``[superdex]`` extra (Python 3.12).

**Key features:**

* :func:`robosimstudio.sponge` for a soft block
* :func:`robosimstudio.robot` with ``"panda"`` for a single arm
* :class:`robosimstudio.Scene` with ``cell="bench_075"`` for a bench to stand them on
* :func:`robosimstudio.view` to drive the arm from the page
"""

import robosimstudio as rss


def main():
    scene = rss.Scene(cell="bench_075", name="sponge")
    scene.add(rss.robot("panda"))
    scene.add(rss.sponge(name="sponge", at=(0.0, 0.0, 0.0)))
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
