# Rope

Cables and ropes: a chain of capsules by default, or a Cosserat rod on
SuperDex with one line of physics. How a cable is built, part by part:
[`rod/`](rod/), the implementation `rss.rod` builds. The cable demo,
[route the cable](../../demo/route_the_cable/README.md), is this kind under two
YAM hands that hold it by friction alone.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Drag a cable

![Drag a cable](media/00_cable.webp)

A cable lying on the bench under an open hand: drag the hand down onto it, close it and lift, or click any segment and drag it.

The right hand starts pointing down over the capsule nearest the cable's middle.
Drag its gizmo straight down until the fingertips are just over the bench, press
**Close**, and drag it up slowly: the pads hold the cable by friction alone, so a
grip that slips is a grip that slips. That takes static friction, which the
default contact law does not have: `contact_matching="sticky"` keeps each
contact's anchor from step to step. A click on any segment puts a handle on it
instead, and `at` places the cable's first end, not its middle.

**Key features:**

* `rss.rod` for a cable, a rope or a cord, with `scene_friction` for the pads' grip
* `rss.CollisionSpec` `contact_matching="sticky"` for friction that holds still
* `rss.ArmSpec` `home_q` for a hand that starts over the cable
* `rss.view` to drag it and drive the arms from the page

```bash
uv run examples/10_rope/00_cable.py
```

Source: [`00_cable.py`](00_cable.py)

## Cable between two posts

![Cable between two posts](media/01_between_posts.webp)

A cable clamped by both ends on top of two posts: hook it with a gripper or drag its middle, and it springs back.

`anchors` clamps a cable's ends to the world where they were built: the end's
position, its direction and its twist are held, so a cable clamped by one end
is a cantilever and by both a span that sags under its own weight. The posts
are scenery to rest the ends on, and the cable is built just clear of their
tops so that nothing pushes against a clamp.

**Key features:**

* `rss.rod` with `anchors=("start", "end")` to clamp both ends
* `rss.post` for something welded in place to string it between
* `rss.robot` with `"yam_bimanual"` for two arms to hook it with
* `rss.view` to pull the span and watch it recover

```bash
uv run examples/10_rope/01_between_posts.py
```

Source: [`01_between_posts.py`](01_between_posts.py)

## Cable on SuperDex

![Cable on SuperDex](media/02_superdex.webp)

The same cable spec stepped by another engine: a Cosserat rod on SuperDex, beside arms that stay on MuJoCo.

One entry in the physics arrangement routes the `rod` kind to SuperDex, and
the cable stops being a chain of capsules: it becomes one centerline with a
twist at every node, and its matter is read as an elastic modulus rather than
as the capsule chain's stretch and bend stiffness. Newton still holds its
particles, so the page drags it the same way. Needs the `[superdex]` extra
(Python 3.12).

**Key features:**

* `rss.backend` with `"superdex"` to name the engine
* `rss.replace` on `rss.GENERIC` to route the `rod` kind to it
* `rss.RodSpec` with `young_modulus=`, the stiffness a Cosserat rod reads
* `sim.pipeline.program.summary()` to print which engine got what

```bash
uv run --extra superdex examples/10_rope/02_superdex.py
```

Source: [`02_superdex.py`](02_superdex.py)

<!-- examples:end -->
