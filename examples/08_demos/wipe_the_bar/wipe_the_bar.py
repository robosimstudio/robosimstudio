# SPDX-License-Identifier: Apache-2.0
"""Wipe the bar

Two YAMs, a sponge on the mat and a chrome grab bar with a marker stripe on it: press **Wipe the bar** and they clean it.

The scene is ``bar.py`` and the routine is ``routine.py``. The jaws grip the
sponge by contact, and once the grip settles a ``post_step_hooks`` latch pins the
foam to the hand until **Open**, because friction alone let it slide out in the
wipe (README). Needs the ``[superdex]`` extra (Python 3.12).
"""

import time

import bar
import routine

import robosimstudio as rss


def page(sim):
    """``page=``: the press, where it lands, and how much of the stripe is clean."""
    stripe = sim["stripe"].entity

    def build(studio) -> None:
        studio.aim_camera(bar.EYE, bar.LOOK)  # from behind the arms, over the bar and the sponge
        with studio.gui.add_folder("Sponge"):
            reading = studio.gui.add_markdown("")
            picture = studio.gui.add_image(bar.contact_map(stripe), format="png")
        last = {"drawn": 0.0}

        def each_step() -> None:
            bar.latch(sim)  # back in the hooks if a reset took it
            if time.monotonic() - last["drawn"] > 0.3:
                last["drawn"] = time.monotonic()
                reading.content = (
                    f"**{stripe.press:.1f} N** · peak {stripe.peak / 1000:.1f} kPa · {stripe.cleaned:.0%} clean"
                )
                picture.image = bar.contact_map(stripe)

        studio.each_step(each_step)

    return build


def main():
    scene = bar.scene()
    sim = scene.build(control_mode="joint_position")
    bar.latch(sim)  # pins the sponge to the right hand once its jaws grip it, until Open

    rss.view(sim, page=page(sim), routines={"Wipe the bar": routine.routine})


if __name__ == "__main__":
    main()
