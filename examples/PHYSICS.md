# Physics and engines

Which engine simulates each kind of thing, and how to change
it.

## The defaults

A scene uses `rss.GENERIC` unless you pass
`physics=`. It gives each kind a solver:

| Kind | Made by | Engine | Solver |
| --- | --- | --- | --- |
| Robot | `rss.robot` | Newton | `mujoco` (MuJoCo Warp) |
| Rigid body | `rss.box`, `sphere`, `capsule`, `cylinder`, `body` | Newton | `vbd` |
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

## Letting a gripper pinch cloth

With the defaults a hand pushes a sheet but cannot hold it between its fingers.
Two settings change that:

```python
physics = rss.replace(
    rss.GENERIC,
    coupling=rss.coupler("one_way"),
    collision=rss.CollisionSpec(full_surface_contact=True),
)

scene = rss.Scene(cell="void", physics=physics)
scene.add(rss.robot("yam_bimanual"))
scene.add(rss.sheet(name="towel", size=(0.35, 0.35), cells=(30, 30), at=(-0.175, -0.175, 0.25)))
sim = scene.build(control_mode="joint_position")
rss.view(sim)    # close a gripper on an edge of the towel
```

- `rss.coupler` chooses how the robot's solver and
  the cloth's solver exchange forces.
- `full_surface_contact` adds contacts on the cloth's edges and faces. Without
  it, a finger pad narrower than a cloth cell finds no vertex to hold.

| Coupler | What it does |
| --- | --- |
| `proxy` | The default. The robot pushes things and they push back. |
| `one_way` | The robot leads. Cloth collides with it and does not push back. |
| `admm` | Registered. No shipped example uses it. |

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
