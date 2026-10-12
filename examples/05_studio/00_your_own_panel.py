# SPDX-License-Identifier: Apache-2.0
"""Your own panel

A folder of your own on the studio page: a slider, a button that tosses the cube, and a readout of its height.

``page=`` on :func:`robosimstudio.view` is called once with the live page after
its own panels are built, and ``studio.gui`` is viser's GUI API, so anything
written for a bare ``ViserServer`` drops in. One rule: a viser callback runs on
the server's thread and may only assign; the calls that touch the simulation go
in ``studio.each_step``, which runs on the main thread once per control step.

**Key features:**

* ``page=`` on :func:`robosimstudio.view` for controls of your own beside the studio's
* ``studio.gui.add_folder``, ``add_slider``, ``add_button`` and ``add_number``, viser's own
* ``studio.each_step`` for the work that touches the simulation
* :attr:`robosimstudio.Ref.bodies` and :attr:`robosimstudio.Ref.position` to reach the cube's state
"""

import robosimstudio as rss


def main() -> None:
    bench = rss.get_cell("bench_075").surface_z

    scene = rss.Scene(cell="bench_075", name="your_own_panel")
    cube = scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.0, 0.2)))
    scene.add(rss.robot("panda"))
    sim = scene.build()

    def page(studio):
        wanted = {"toss": False}  # written on viser's thread, read on the main thread
        with studio.gui.add_folder("Cube"):
            speed = studio.gui.add_slider("toss speed (m/s)", min=0.5, max=3.0, step=0.1, initial_value=1.5)
            button = studio.gui.add_button("Toss the cube")
            height = studio.gui.add_number("height above the bench (mm)", 0.0, disabled=True)
        button.on_click(lambda _: wanted.update(toss=True))

        def each_step() -> None:
            if wanted["toss"]:
                wanted["toss"] = False
                body = int(cube.bodies[0])
                # Both halves of the double buffer, so the next step reads it whichever it starts from.
                for state in (sim.pipeline.state_0, sim.pipeline.state_1):
                    velocity = state.body_qd.numpy()
                    velocity[body] = (0.0, 0.0, speed.value, 0.0, 0.0, 0.0)  # linear, then angular
                    state.body_qd.assign(velocity)
            height.value = round(1000.0 * (cube.position[2] - bench), 1)

        studio.each_step(each_step)

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
