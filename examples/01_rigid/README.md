# Rigid bodies

Boxes, spheres, capsules and bodies loaded from a file.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Drag rigid bodies

![Drag rigid bodies](media/00_drag_things.webp)

A cube under an open hand, a ball and a capsule on a bench: drag the hand down onto the cube, close it and lift.

The right hand starts pointing down over the cube. Drag its gizmo straight down,
press **Close**, and drag it up: the fingers hold the cube by friction alone, so
a grip that slips is a grip that slips. That needs `rss.MUJOCO_500HZ`, which
puts the arm and every free body in one MuJoCo island; the default physics lets
a hand push a body but not pinch one. A click on any body puts a handle on it
instead, and the body stays fully dynamic under it.

**Key features:**

* `rss.Scene` and `rss.Scene.add` to put things in a place
* `rss.box`, `rss.sphere` and `rss.capsule` for shapes with no asset file
* `rss.MUJOCO_500HZ` for physics a hand can pick a rigid body up in
* `rss.ArmSpec` `home_q` for a hand that starts over the cube
* `rss.view` to open the studio page on it

```bash
uv run examples/01_rigid/00_drag_things.py
```

Source: [`00_drag_things.py`](00_drag_things.py)

<!-- examples:end -->
