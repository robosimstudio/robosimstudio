# SPDX-License-Identifier: Apache-2.0
"""Wipe the bar

Two YAMs, a sponge on the mat and a chrome grab bar with a marker stripe on it: press **Wipe the bar** and they clean it.

The twin of a real cell; its scene, every distance in it measured off the clip,
is ``bar.py`` beside this file, and the job is ``routine.py``. From the home
pose the left hand steadies the bar, and the right takes the sponge off the mat,
presses it onto the bar, strokes the stripe off, puts it back, and both go home,
dragging the gizmos and pressing Grip as a person would. The grasp is contact;
the hold after it is a **latch**: once the closed jaws have settled on the
foam, the foam under the pads is pinned to the hand until **Open**, because
friction alone let it slide out in the wipe (README). Needs the ``[superdex]``
extra (Python 3.12).

**Key features:**

* ``routines=`` on :func:`robosimstudio.view` for a Run button that plays the job through the arm panels
* :class:`robosimstudio.ActuationSpec` ``effort_limit`` and ``joint_damping`` for jaws that close on foam and stop
* ``post_step_hooks`` for a latch that engages on a grasp the contact made, and lets go on Open
* :func:`robosimstudio.register_entity` for the welded bar and for a stripe that wears off
"""

import time

import bar
import routine

import robosimstudio as rss


def page(sim):
    """``page=``: the press, where it lands, and how much of the stripe is clean."""
    stripe = sim["stripe"].entity

    def build(studio) -> None:
        def aim(client) -> None:  # from behind the arms, over the bar and the sponge
            client.camera.position, client.camera.look_at = bar.EYE, bar.LOOK

        studio.server.on_client_connect(aim)
        for client in studio.server.get_clients().values():
            aim(client)
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
    scene = bar.scene()  # two YAMs at home, the mat, the blocks, the bar, the sponge and the stripe
    sim = scene.build(control_mode="joint_position")
    bar.latch(sim)  # the right jaws, closed on the sponge, latch it to the hand until they open

    rss.view(sim, page=page(sim), routines={"Wipe the bar": routine.routine})


if __name__ == "__main__":
    main()
