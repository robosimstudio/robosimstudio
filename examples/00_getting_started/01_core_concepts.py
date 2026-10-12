# SPDX-License-Identifier: Apache-2.0
"""Core concepts

A scene you declare, a sim that runs it, and a ref on each thing you added.

A :class:`robosimstudio.Scene` is a list of specs in a named place and simulates
nothing. :meth:`robosimstudio.Scene.build` compiles it into a
:class:`robosimstudio.Sim`, which owns the state and steps it. Each ``add``
returns a :class:`robosimstudio.Ref`, a name that reads the live state.

Every ``at=`` is measured from the middle of the place's work surface, so the
same two specs are built here on a bench and again on a bare floor.

**Key features:**

* :meth:`robosimstudio.Scene.add` returning a :class:`robosimstudio.Ref` for each thing
* The work-surface frame: one ``at=`` in two places, ``"bench_075"`` and ``"void"``
* :meth:`robosimstudio.Sim.run`, :meth:`robosimstudio.Sim.sync` and :meth:`robosimstudio.Sim.reset`
* ``sim["cube"]`` and :meth:`robosimstudio.Sim.names` to find a thing by its name
* :attr:`robosimstudio.Ref.position`, read from the simulation each time it is asked
"""

import robosimstudio as rss


def main() -> None:
    release = (0.0, 0.0, 0.2)  # 200 mm over the middle of the work surface, wherever that is

    sims = {}
    for cell in ("bench_075", "void"):  # a bench 750 mm high, then a bare floor
        # A Scene is a declaration. Nothing is simulated until build().
        scene = rss.Scene(cell=cell, name=cell)
        cube = scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=release))
        ball = scene.add(rss.sphere(name="ball", radius=0.03, at=(0.0, 0.12, 0.2)))
        scene.add(rss.robot("panda"))

        # build() compiles it into a Sim, and the refs from add() now read that Sim.
        sim = scene.build()
        surface = rss.get_cell(cell).surface_z
        print(f"{cell}: the work surface is at z={surface:.3f} m and the scene holds {sim.names()}")
        print(f"  released  cube z={cube.position[2]:.3f}  ball z={ball.position[2]:.3f}")

        sim.run(150).sync()  # step, then wait for the GPU before reading
        print(f"  at rest   cube z={cube.position[2]:.3f}  ball z={ball.position[2]:.3f}  t={sim.time:.2f} s")

        # A name finds the same handle, and finds the place's own furniture too.
        print(f"  by name   sim['cube'] is cube: {sim['cube'] is cube}")

        sim.reset()  # everything back where the scene declared it
        print(f"  reset     cube z={cube.position[2]:.3f}  t={sim.time:.2f} s")
        sims[cell] = sim

    rss.view(sims["bench_075"])


if __name__ == "__main__":
    main()
