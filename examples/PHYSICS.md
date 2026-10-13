# Physics and engines

Which engine simulates each kind of thing, and how to change
it.

## The defaults

A scene uses `rss.GENERIC` unless you pass
`physics=`. It gives each kind a solver:

| Kind | Made by | Engine | Solver |
| --- | --- | --- | --- |
| Robot | `rss.robot` | Newton | `mujoco` (MuJoCo Warp) |
| Rigid body | `rss.box`, `sphere`, `capsule`, `cylinder`, `body` | Newton | `vbd` (a hand pushes it but cannot pick it up: see [below](#letting-a-gripper-hold-a-rigid-body)) |
| Cloth | `rss.sheet` | Newton | `vbd` |
| Soft body | `rss.sponge` | SuperDex | `superdex` (FEM) |
| Rope | `rss.rod` | Newton | `vbd` (a chain of capsules; `superdex` makes it a Cosserat rod, see [`10_rope/`](10_rope/README.md)) |
| Fixed thing | `rss.slab`, `rss.post`, the bench | None | Fixed to the world |

`summary()` prints what a built scene runs on:

```python
import robosimstudio as rss

scene = rss.Scene(cell="bench_075")
scene.add(rss.robot("panda"))
scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0.2, 0.2)))
scene.add(rss.sheet(name="towel", size=(0.3, 0.3), cells=(30, 30), at=(-0.15, -0.15, 0.1)))
sim = scene.build()

print(sim.pipeline.program.summary())
# physics: proxy over 2 island(s)
#   island mujoco: mujoco x1 -> ['embodiment'] (16 bodies, ...)
#   island vbd: vbd x1 -> ['cube', 'towel'] (1 bodies, 1 joints, 961 particles, ...)
```

An island is the group of things one solver steps. The robot is listed as `embodiment`.

## Rates

`rss.replace` returns a copy of the defaults with
some fields changed:

```python
physics = rss.replace(rss.GENERIC, substeps=4)

scene = rss.Scene(cell="bench_075", physics=physics)
scene.add(rss.robot("panda"))
scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
sim = scene.build()
```

| Field | Default | Meaning |
| --- | --- | --- |
| `control_hz` | 60 | Control steps per second |
| `substeps` | 16 | Solver steps per control step. Must be even. |
| `gravity` | -9.81 | Along z, in m/s² |

Sixteen substeps is a cautious default for cloth. A scene of rigid bodies steps
faster with fewer: the one above runs at 107 steps/s with 16 substeps and at
426 steps/s with 4, on one NVIDIA RTX PRO 6000 Blackwell GPU.

## Switching a solver

`backends` pairs a kind with a solver. The kind names are `articulated` (a
robot), `rigid`, `cloth` and `soft`. `"*"` stands for every kind not named.

```python
physics = rss.replace(
    rss.GENERIC,
    backends=(
        ("articulated", rss.backend("mujoco")),
        ("cloth", rss.backend("superdex", substeps=2)),  # cloth moves to SuperDex
        ("*", rss.backend("vbd")),
    ),
)

scene = rss.Scene(cell="bench_075", physics=physics)
scene.add(rss.robot("yam_bimanual"))
scene.add(rss.sheet(name="towel", size=(0.3, 0.3), cells=(20, 20), at=(-0.15, -0.15, 0.06)))
sim = scene.build()
```

The sheet's spec does not change. `rss.backend`
takes one of these names:

| Solver | Engine | Simulates | Needs |
| --- | --- | --- | --- |
| `mujoco` | Newton | Robots, rigid bodies | |
| `vbd` | Newton | Rigid bodies, cloth, soft bodies | |
| `xpbd` | Newton | Rigid bodies, cloth, soft bodies | |
| `featherstone` | Newton | Robots | |
| `kinematic` | Built in | A robot that follows its joint targets exactly | |
| `superdex` | SuperDex | Cloth, soft bodies | The `[superdex]` extra |
| `genesis` | Genesis | Cloth, soft bodies | The `[genesis]` extra |

`substeps=2` steps SuperDex or Genesis twice per control step. The shipped
examples use that value.

## Letting a gripper hold a rigid body

With the defaults a hand pushes a box but cannot hold it: the box is on VBD and
the fingers on MuJoCo, and the proxy coupler between them passes pushes, not a
pinch. Closed on a 50 mm cube, a Panda lets it slide out or flings it.
`rss.MUJOCO_500HZ` puts the arm and every free rigid body in one MuJoCo
island, so the fingers stall on the faces and friction carries what they
closed on:

```python
scene = rss.Scene(cell="bench_0728", physics=rss.MUJOCO_500HZ)
scene.add(rss.robot("yam_bimanual"))
scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, -0.20, 0.025)))
sim = scene.build(control_mode="joint_position")
rss.view(sim)    # a hand closed on the cube lifts it
```

Nothing holds the cube to the hand but the fingers: a grip that slips, slips.
Dragged down onto a 50 mm cube through the page, closed and lifted 150 mm, the
cube came up 145 mm with 0.1 mm of slip in the YAM's hand
([`01_rigid/00_drag_things.py`](01_rigid/00_drag_things.py)) and 150 mm with
0.3 mm in the Panda's ([`00_getting_started/`](00_getting_started/README.md));
**Open** drops it.

The arm and the bodies are stepped as joint coordinates here, so a body's
velocity is its free joint's `joint_qd` (linear, then angular). Writing
`body_qd`, which this island does not read, moves nothing:
[`05_studio/00_your_own_panel.py`](05_studio/00_your_own_panel.py) tosses its
cube the first way.

| Field | Value |
| --- | --- |
| `control_hz` | 50 |
| `substeps` | 10 (500 Hz) |
| `backends` | `"*"` on `mujoco` (`nconmax=512`, `njmax=512`, `enable_multiccd=True`) |
| `collision` | `contact_ke=6e4`, `contact_kd=500`, `rigid_gap=2 mm` |

It is the tool-sorting benchmark's arrangement, which tasks name
`mujoco_500hz@v1`. It is for rigid bodies: a sheet or a capsule-chain rope has
no MuJoCo solver and stays on `GENERIC`. A sponge keeps its own engine:
`rss.MUJOCO_500HZ.with_backend("soft", rss.backend("superdex", substeps=2))`.

## Letting a gripper pinch cloth

With the defaults a hand pushes a sheet but cannot hold it between its fingers.
Two settings change that:

```python
physics = rss.replace(
    rss.GENERIC,
    coupling=rss.coupler("one_way"),
    collision=rss.CollisionSpec(full_surface_contact=True),
)

scene = rss.Scene(cell="bench_0728", physics=physics)
scene.add(rss.robot("yam_bimanual"))
scene.add(rss.sheet(name="towel", size=(0.35, 0.35), cells=(30, 30), at=(-0.175, -0.20, 0.02)))
sim = scene.build(control_mode="joint_position")
rss.view(sim)    # fingertips down to the bench at an edge, Close, lift
```

- `rss.coupler` chooses how the robot's solver and
  the cloth's solver exchange forces.
- `full_surface_contact` adds contacts on the cloth's edges and faces. Without
  it, a finger pad narrower than a cloth cell finds no vertex to hold.

Then the pads hold the sheet by friction alone; nothing is pinned. A YAM hand
put its fingertips on the bench at a lying terry towel's edge, closed and lifted
100 mm: the edge rose 96 mm, and 6 of the 7 particles within 20 mm of the tool
centre stayed in the hand, in 3 of 3 runs
([`02_cloth/00_towel.py`](02_cloth/00_towel.py)). The fold demo,
[`demo/fold_the_towel`](../demo/fold_the_towel/README.md), folds a towel in half
the same way, from the home pose.

- **On a flat sheet the fingertips must reach the bench**, or the pads close
  over the edge rather than around it. A tool centre 20 mm up held nothing.
- **Under `one_way` the arm feels nothing from the cloth**, so a shut hand
  closes right through it. On a thin membrane close to about its contact
  thickness instead: the Panda in
  [`02_cloth/04_your_own_cloth.py`](02_cloth/04_your_own_cloth.py) holds a
  draped hem at **Grip amount** 0.95 (pads 5 mm apart on a 4 mm sheet) and
  lets it slip at **Close**.
- The coupler hands the cloth solver the arm's previous pose and its new one,
  and friction acts on the pad's motion between them. (Before this release it
  handed over the new pose twice, every finger looked still, and no setting of
  friction made a pinch hold.)

| Coupler | What it does |
| --- | --- |
| `proxy` | The default. The robot pushes things and they push back. |
| `one_way` | The robot leads. Cloth collides with it and does not push back. |
| `admm` | Registered. No shipped example uses it. |

## Letting a gripper hold a rope

A capsule-chain rod held between fingers needs static friction, which the
default contact law does not have: a pinched 22 mm cable crept through the pads
at 15 mm/s and fell out of a 0.15 m/s lift. Anchored contacts hold it:

```python
physics = rss.replace(
    rss.GENERIC.with_backend("*", rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)),
    collision=rss.CollisionSpec(contact_matching="sticky"),
)
cable = rss.rod(name="cable", length=0.6, radius=0.011, segments=28,
                allow_grasp_latch=False, scene_friction=True)
```

- `scene_friction=True` gives the pads the rod's `mu_gripper` (60) and the bench
  0.02; `allow_grasp_latch=False` says nothing else holds it.
- With them a 120 mm lift held the cable to 0.25 mm in the fingers, in 3 of 3
  runs ([`10_rope/00_cable.py`](10_rope/00_cable.py)), and two arms route a
  600 mm cable through two spools into a port from home
  ([`demo/route_the_cable`](../demo/route_the_cable/README.md)).
- **Drag the gizmo, don't flick it.** On a rope scene the arm has no
  rate-limited IK, so a jump of the cursor is a fast move of the hand, and a
  0.4 m/s jerk drops the cable.

## Letting a gripper hold a sponge or a SuperDex cable

SuperDex's friction has no static part: below a falloff speed it grows with
slip, so a held thing creeps. Against a hand, **the falloff that counts is the
deformable's own** -- the hand's colliders are static actors to the engine,
which then uses the colliding body's value:

```python
foam = rss.replace(FIRM_FOAM, friction_falloff=1.0e-3)          # from robosimstudio.domains.soft
grip = rss.ActuationSpec(target_ke=1500.0, target_kd=60.0, effort_limit=8.0)   # [N] per finger
physics = rss.GENERIC.with_backend("soft", rss.backend("superdex", substeps=2, feedback_dv_max=20.0))
```

- **Falloff.** A lifted sponge slid 1.4 mm in 2.5 s and 4.5 mm in the next 10 s
  at the engine's 10 mm/s, and 0.1 mm and 0.1 mm at 1 mm/s. A SuperDex cable
  takes `RodSpec(friction_falloff=1e-3)` the same way: 0.5-1.3 mm of slip over
  a lift, against 3.2-4.0 mm. `rss.backend("superdex", mirror_friction_falloff=...)`
  sets the hand's value, which no contact reads, and measured identical.
- **A capped grip.** At 8 N per finger the Panda's jaws stop at 44 mm on a
  50 mm firm sponge; uncapped they cut it to 34 mm.
- **Feedback on, and allowed to push.** The foam's push on a 10 g pad is capped
  at what moves it `feedback_dv_max` in a control step. At the default 0.5 m/s
  that is 0.3 N, and the jaws closed the foam to 6 mm; at 20 m/s the foam holds
  them where the grip and its push balance.

With these the Panda lifts a sponge 99 mm on **Close**, 1.7 mm of slip, in 3 of
3 runs ([`03_soft/00_sponge.py`](03_soft/00_sponge.py)). A YAM hand holds it in a
plain lift as well, but not through a pressed wipe: the sponge twisted out of
the jaws on the second or third stroke in every run we made, which is why the
wipe demo latches it once the grip has closed
([`demo/wipe_the_bar`](../demo/wipe_the_bar/README.md)).

## Extras

| Engine | Install | Notes |
| --- | --- | --- |
| SuperDex | `uv sync --extra superdex` | It solves on the CPU. |
| Genesis | `uv sync --extra genesis` | We recommend a virtual environment of its own. |

From a checkout, `uv run --extra <name> <script>` installs the extra and runs
the script. With pip, into another environment:
`pip install "robosimstudio[<name>] @ git+https://github.com/robosimstudio/robosimstudio"`.

A scene that needs a missing engine raises an `ImportError` at `build()`, and
the message names the extra. A scene with a sponge needs SuperDex.

## Limits

- `full_surface_contact=True` works with `one_way` only. With `proxy` the build
  raises `NotImplementedError`.
- `full_surface_contact=True` needs a signed-distance field on the robot's
  colliders. `yam_bimanual` builds as is. For the other robots, pass
  `rss.CollisionSpec(full_surface_contact=True, force_sdf=True)`.
- Under `one_way` the robot feels no force from the cloth.
- An engine may not read every field of a material. The build warns and names
  the fields it ignores.
- Under `admm` a sheet fell through the bench in our test. We do not recommend
  it for cloth.


## Performance

What costs time and what to do about it. The numbers are from
one NVIDIA RTX PRO 6000 Blackwell GPU.

### The first run

The first run compiles Warp kernels and caches them under `~/.cache/warp`. The
hello world took 83 s with an empty cache and 12 s with a warm one. Later runs
on the same machine start warm.

### Substeps

`rss.GENERIC` runs 16 substeps per control step,
a cautious value chosen for cloth. A scene of rigid bodies and a robot does well
with 4:

```python
import robosimstudio as rss

scene = rss.Scene(cell="bench_075", physics=rss.replace(rss.GENERIC, substeps=4))
scene.add(rss.robot("panda"))
scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
sim = scene.build()
```

A Panda and a cube step at 107 steps/s with 16 substeps and at 426 steps/s
with 4.

### Cloth

- The cost grows with `cells`. Two arms and a 30-cell sheet step at 47
  steps/s; a 60-cell sheet steps at 20 to 28. Start coarse.
- The coupling that lets a gripper pinch cloth costs some speed: the same
  30-cell sheet steps at 38 steps/s under it. See [Rates](#rates) above.
- At 60 control steps per second, a scene that steps slower than that plays
  slower than real time on the page.

### Soft bodies

SuperDex is a CPU solver. The default sponge, 440 particles beside a Panda,
steps at 150 steps/s.

### The page

- `render_every=4` draws one frame per four control steps. More steps per
  second, fewer frames.
- Leave `share` at its default, `False`: `share=True` asks for a public URL,
  which blocks until the tunnel answers.
- The live camera view renders in the loop. It is capped at a share of wall
  clock (`live_load`, 0.3 by default) and `StudioConfig(live=False)` turns it
  off.

### Reading state

- `sim.run(150).sync()`: step, wait for the GPU once, then read.
- A `Ref` caches nothing. Every `cube.position` reads
  the live state.
- `sim.read()` returns every pose in one read. Use it when you need many.

### Building

- A build takes 6 to 7 s for a Panda and a cube with a warm cache. The live
  page rebuilds two arms and a box in 7 to 8 s.
- Run one process per GPU, and pick the GPU with `CUDA_VISIBLE_DEVICES`.

**See also:**

- the docstrings of `rss.GENERIC`, `rss.ArrangementSpec` and
  `rss.CollisionSpec`
- `examples/02_cloth/00_towel.py` for the pinch
- `examples/02_cloth/02_engines.py` for one sheet on three engines
- `examples/03_soft/00_sponge.py` for a soft body on SuperDex
- [Extending](04_extending/README.md) to add a kind of your own
