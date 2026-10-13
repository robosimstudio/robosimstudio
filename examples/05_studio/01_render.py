# SPDX-License-Identifier: Apache-2.0
"""Render a still and a video

Headless: one path-traced still of a scene, then a short video of it running, with no page open.

:meth:`robosimstudio.Sim.render` photographs the scene as it stands through a
camera you declare: the room, the lights and the materials, path traced by
``ovrtx`` on a GPU with RT cores. On one without them, or with ``ovrtx`` left
out, the same call draws the colliders alone and says why once;
``ROBOSIMSTUDIO_RENDERER`` picks either by name. :meth:`robosimstudio.Sim.record`
steps the scene and writes what the same camera sees to an mp4.

**Key features:**

* :class:`robosimstudio.CameraSpec` to declare a camera: where it stands, what it looks at, its field of view and size
* :meth:`robosimstudio.Sim.render` for one frame, as ``{"rgb": uint8 [H, W, 3]}``
* :meth:`robosimstudio.Sim.record` for a video of the scene running
"""

import tempfile
from pathlib import Path

import imageio.v3 as iio

import robosimstudio as rss


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="robosimstudio_render_"))

    scene = rss.Scene(cell="bench_075", name="render")
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.0, 0.2)))
    scene.add(rss.robot("panda"))
    sim = scene.build()
    sim.run(150).sync()  # the cube lands first

    # A three-quarter view of the bench, in world coordinates: the bench top is at z = 0.75 m.
    camera = rss.CameraSpec(
        name="view",
        width=960,
        height=540,
        fov_y=45.0,
        eye=(-0.4, -0.9, 1.5),
        look_at=(0.6, 0.0, 0.8),
        up=(0.0, 0.0, 1.0),
        near=0.01,
        far=8.0,
    )

    rgb = sim.render(camera)["rgb"]
    iio.imwrite(out / "still.png", rgb)
    print(f"still: {rgb.shape[1]}x{rgb.shape[0]} written to {out / 'still.png'}")

    sim.record(str(out / "clip"), steps=120, camera=camera, fps=30, every=2)
    print(f"video: {out / 'clip' / 'view.mp4'}")


if __name__ == "__main__":
    main()
