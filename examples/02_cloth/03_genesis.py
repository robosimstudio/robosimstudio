# SPDX-License-Identifier: Apache-2.0
"""Cloth on Genesis

A sheet simulated by Genesis beside a Newton arm: drag the wrist gizmo down onto the cloth.

One entry in the physics arrangement routes cloth to Genesis's IPC solver. The
arm stays on MuJoCo, and the contact reaction flows back onto the hand.
Needs the ``[genesis]`` extra (Linux or Windows x86 with an NVIDIA GPU).

**Key features:**

* :func:`robosimstudio.backend` with ``"genesis"`` to route cloth to a foreign engine
* ``GENERIC``, the default arrangement the entry replaces
* :func:`robosimstudio.sheet`, the same spec the default solver takes
* :func:`robosimstudio.view` to drive the arm from the page
"""

import robosimstudio as rss


def main():
    # ``GENERIC`` routes cloth to VBD, which is the measured default and stays
    # the default; naming Genesis is one entry, and that is the whole edit.
    physics = rss.replace(
        rss.GENERIC,
        backends=(
            ("articulated", rss.backend("mujoco")),
            ("cloth", rss.backend("genesis", substeps=2)),
            ("*", rss.backend("vbd")),
        ),
    )
    scene = rss.Scene(cell="bench_075", name="genesis_cloth", physics=physics)
    scene.add(rss.robot("panda"))
    scene.add(rss.sheet(name="towel", size=(0.3, 0.3), cells=(16, 16), at=(-0.15, -0.15, 0.02)))
    sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
