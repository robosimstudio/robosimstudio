# SPDX-License-Identifier: Apache-2.0
"""Hello world

The smallest RoboSimStudio program: a bench, a robot arm, and a cube that lands under its hand.

``build()`` turns the scene into a simulation, 150 steps let the cube land, and
the last line opens the studio in a browser. Drag the gizmo on the hand straight
down onto the cube, press **Close** and drag it up: the fingers lift the cube by
friction, which ``rss.MUJOCO_500HZ`` is there for. Or click the cube to drag it.

**Key features:**

* :class:`robosimstudio.Scene` and :meth:`robosimstudio.Scene.add` to declare what stands where
* :func:`robosimstudio.box` and :func:`robosimstudio.robot` for a free body and a registered robot
* :data:`robosimstudio.MUJOCO_500HZ` for physics a hand can pick a rigid body up in
* :meth:`robosimstudio.Sim.run` to step and :func:`robosimstudio.view` to open the browser studio
"""

import robosimstudio as rss


def main() -> None:
    scene = rss.Scene(cell="bench_075", physics=rss.MUJOCO_500HZ)  # a bench 750 mm high
    # ``at`` is measured from the middle of the bench top: 100 mm above the
    # spot under the Panda's hand, which starts 231 mm over the bench, pointing down.
    cube = scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(-0.143, -0.115, 0.1)))
    scene.add(rss.robot("panda"))

    sim = scene.build()
    sim.run(150).sync()  # 150 control steps are 3 s at 50 Hz; sync before reading

    height = cube.position[2]  # read from the simulation now, not remembered
    print(f"the cube came to rest at z={height:.3f} m, on a bench whose top is at 0.750 m")

    rss.view(sim)


if __name__ == "__main__":
    main()
