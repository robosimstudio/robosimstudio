# SPDX-License-Identifier: Apache-2.0
"""Hello world

The smallest RoboSimStudio program: a bench, a cube released above it and a robot arm.

``build()`` turns the scene into a simulation, 150 steps let the cube land, and
the last line opens the studio in a browser: click the cube to drag it, or drag
the gizmo on the hand to drive the arm.

**Key features:**

* :class:`robosimstudio.Scene` and :meth:`robosimstudio.Scene.add` to declare what stands where
* :func:`robosimstudio.box` and :func:`robosimstudio.robot` for a free body and a registered robot
* :meth:`robosimstudio.Sim.run` to step and :func:`robosimstudio.view` to open the browser studio
"""

import robosimstudio as rss


def main() -> None:
    scene = rss.Scene(cell="bench_075")  # a bench 750 mm high
    # ``at`` is measured from the middle of the bench top: 200 mm above it.
    cube = scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.0, 0.2)))
    scene.add(rss.robot("panda"))

    sim = scene.build()
    sim.run(150).sync()  # 150 control steps are 2.5 s; sync before reading

    height = cube.position[2]  # read from the simulation now, not remembered
    print(f"the cube came to rest at z={height:.3f} m, on a bench whose top is at 0.750 m")

    rss.view(sim)


if __name__ == "__main__":
    main()
