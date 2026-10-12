# SPDX-License-Identifier: Apache-2.0
"""Cable on SuperDex

The same cable spec stepped by another engine: a Cosserat rod on SuperDex, beside arms that stay on MuJoCo.

One entry in the physics arrangement routes the ``rod`` kind to SuperDex, and
the cable stops being a chain of capsules: it becomes one centerline with a
twist at every node, and its matter is read as an elastic modulus rather than
as the capsule chain's stretch and bend stiffness. Newton still holds its
particles, so the page drags it the same way. Needs the ``[superdex]`` extra
(Python 3.12).

**Key features:**

* :func:`robosimstudio.backend` with ``"superdex"`` to name the engine
* :func:`robosimstudio.replace` on ``rss.GENERIC`` to route the ``rod`` kind to it
* :class:`robosimstudio.RodSpec` with ``young_modulus=``, the stiffness a Cosserat rod reads
* ``sim.pipeline.program.summary()`` to print which engine got what
"""

import robosimstudio as rss


def main():
    length = 0.6  # end to end [m]
    radius = 0.011  # of the cable's section [m]
    segments = 28  # elements along the centerline
    young_modulus = 5.0e6  # [Pa]; a rubber-jacketed cord is of this order

    # One entry: an exact one wins over GENERIC's fallback, so the rest stays as it was.
    physics = rss.replace(rss.GENERIC, backends=(("rod", rss.backend("superdex", substeps=2)), *rss.GENERIC.backends))

    scene = rss.Scene(cell="bench_075", physics=physics, name="cable_superdex")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(
        rss.rod(
            name="cable",
            length=length,
            radius=radius,
            segments=segments,
            at=(0.0, -length / 2, radius + 0.01),
            along=(0.0, 1.0, 0.0),
            young_modulus=young_modulus,
        )
    )
    sim = scene.build(control_mode="joint_position")
    print(sim.pipeline.program.summary())

    rss.view(sim)


if __name__ == "__main__":
    main()
