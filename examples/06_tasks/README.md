# Tasks

A task says what is on the bench and when an episode succeeds. This page also
covers how to run one.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Your first task

![Your first task](media/00_your_first_task.webp)

A task is a short document: what is on the bench, and when it counts as done.

`bench.task` reads the document and `bench.Env` builds it, robot and all.
The studio opens with the goal beside the scene. Drag the cube into the square
and the goal turns true.

```bash
uv run examples/06_tasks/00_your_first_task.py
```

Source: [`00_your_first_task.py`](00_your_first_task.py)

## Goals and rewards

![Goals and rewards](media/01_goals_and_rewards.webp)

A goal is one clause built from measures; a reward says how close an episode came.

The cube must be in the square, still and clear of the hand for one second.
`once` latches success the first time that holds, and `(fraction)` pays one
share per clause that holds now. `env.step` returns both for one second, then
the studio shows each clause as you drag.

```bash
uv run examples/06_tasks/01_goals_and_rewards.py
```

Source: [`01_goals_and_rewards.py`](01_goals_and_rewards.py)

## Episodes and samplers

![Episodes and samplers](media/02_episodes.webp)

A task has numbered episodes, and a sampler decides how each one starts.

`register_sampler` names a function a document can cite. It returns where the
cube starts in episode *k*, drawn from a generator seeded for that episode, so
`env.reset(k)` builds the same start every time.

```bash
uv run examples/06_tasks/02_episodes.py
```

Source: [`02_episodes.py`](02_episodes.py)

## Sweep rice, scored

![Sweep rice, scored](media/03_granular_scored.webp)

The same rice, pusher and tray as a task document, with a goal on the share of rice in the tray: sweep, and the goal turns true at half.

The heap is a `Heap.from_spec` record the document places under its own name,
the tray walls are `Body.from_spec` fixtures, and the physics is the granular
arrangement under a task-language name. The region is the tray's inside;
`(:z-range -0.005 0.03)` is the walls' height, starting 5 mm under the bench
top because the bottom layer of a pushed heap rests at the top to within a
rounding, and a region's default window is 150 mm, which would count rice
carried over the tray. `tests/test_sweep_rice` drives the hand through the
sweep and the goal turns true, at 93%; 16.5 ms per control step on an RTX PRO
6000, GPU 3, graph-captured.

```bash
uv run examples/06_tasks/03_granular_scored.py
```

Source: [`03_granular_scored.py`](03_granular_scored.py)

## Scrub a patch, scored

![Scrub a patch, scored](media/04_brush_scored.webp)

The same brush, handle and square as a task document, with a goal on the share of the square the bristle tips have scrubbed: turn the hand to point down, close on the handle, scrub, and the goal turns true at half.

The brush is a `Brush.from_spec` record and its handle a `Body.from_spec`
one, placed by the document under the names the brush's `handle` and the
goal use, and the physics is the brush arrangement under a task-language name.
The region is the square, and `scrubbed` reads its 5 mm cells off the
lattice the brush keeps on the device; a reset clears it. `tests/test_scrub_bench`
drives the hand through the same three strokes from the robot's own home and
the goal turns true, at 1.0 (22.3 ms per control step, GPU 3).

```bash
uv run examples/06_tasks/04_brush_scored.py
```

Source: [`04_brush_scored.py`](04_brush_scored.py)

## Pick up a plate, scored

![Pick up a plate, scored](media/05_suction_scored.webp)

The same plate, cup and square as a task document, with a goal on the plate resting in the square: carry it over, untick **Vacuum**, lift the hand away, and the goal turns true.

The plate is a `Body.from_spec` record the document places under its own
name, the robot is `yam_bimanual_suction` and the physics the one MuJoCo
island under its task-language name, `mujoco_500hz@v1`. The square's height
window, `(:z-range 0.0 0.01)`, counts the plate's centre only within 10 mm of
the bench, so a plate carried over the square does not count, and
`hands-clear` asks every tool centre to be 40 mm from the plate's, so a plate
still held down on the square does not either. `tests/test_suction_pick`
drives the hand through the pick and the goal turns true.

```bash
uv run examples/06_tasks/05_suction_scored.py
```

Source: [`05_suction_scored.py`](05_suction_scored.py)

## Open a bag, scored

![Open a bag, scored](media/06_bag_scored.webp)

The same bag and hand as a task document, with a goal on how far the mouth stands open: turn the right hand to point down, open the bag as in `03_materials/06_bag`, and the goal turns true at 30 mm.

The bag is a `Bag.from_spec` record the document places under its own name, the physics is
`rss.BAG_60HZ` under a task-language name, and the staging states the stiffer finger servo the
pinch needs. The hands start at the YAM's home, pointing ahead, so turn the right gizmo down
first. `tests/test_open_a_bag_scored` drives the right hand through the page and `evaluate`
says success, the mouth 75 mm open; 12 ms per control step at rest through the page on an
RTX PRO 6000, GPU 3. Needs the `[superdex]` extra (Python 3.12).

```bash
uv run --extra superdex examples/06_tasks/06_bag_scored.py
```

Source: [`06_bag_scored.py`](06_bag_scored.py)

## Hook a band, scored

![Hook a band, scored](media/07_band_scored.webp)

The same pegboard and band as a task document, with a goal that the band encircles both pegs: hook it from the upper peg onto the lower, and the goal turns true.

The pegs are the example's posts declared with a `rod` feature, their two
ends, so `encircles` holds the band to each peg's length: hanging in front of a
peg's tip, round only the line through it, does not count. The embodiment is the
YAM with `03_materials/04_band`'s right-arm home, the hand in front of the band, and
`tests/test_band_on_pegs` drives the hook there: the goal turns true. Needs `[superdex]`.

```bash
uv run --extra superdex examples/06_tasks/07_band_scored.py
```

Source: [`07_band_scored.py`](07_band_scored.py)

## Peel a label, scored

![Peel a label, scored](media/08_tape_scored.webp)

The same box and label as a task document, with a goal on the share of the label's bonds that have let go: turn the right hand to point down, take the tab and peel as in `03_materials/08_tape`, and the goal turns true at 80%.

The label is a `Tape.from_spec` record and the box a `Body.from_spec` one, placed by the
document under the names the label's `on` and the goal use, and the physics is
`rss.TAPE_60HZ` under a task-language name. The staging states the finger servo of
`03_materials/08_tape`: stiffer than the YAM's own, damped, and capped at 12 N a finger. The cap
is what Close's first push meets: 1500 times the 47.5 mm to go is about 71 N, held to 12 N as on the
material page; the pads then stall on the tab 3.8 mm short of shut and hold it with about 5.8 N
through the peel (`g2_close.py`, `g2_grip.py`). The hands start at the YAM's home, pointing
ahead, so turn the right gizmo down first. `tests/test_peel_a_label_scored` drives the right
hand through the page and `evaluate` says success.

```bash
uv run examples/06_tasks/08_tape_scored.py
```

Source: [`08_tape_scored.py`](08_tape_scored.py)

## Cut tofu, scored

![Cut tofu, scored](media/09_cutting_scored.webp)

The same knife, rest, board and tofu as a task document, with a goal that the tofu is in two pieces: take the knife and push it down through the tofu as in `03_materials/10_cutting`, and the goal turns true.

The tofu is a `Cuttable.from_spec` record, the knife a `Cuttable.knife` one carrying its edge as a
`blade` feature, and the rest and the board are fixtures. The embodiment is the YAM with
`03_materials/10_cutting`'s right-arm home, the hand over the handle, and the physics `rss.CUTTING_50HZ`
under a task-language name. Before the hand comes the goal is unmet; `tests/test_cut_tofu_scored`
drives the cut through the page and `evaluate` reads `pieces` 2 and says success.

```bash
uv run examples/06_tasks/09_cutting_scored.py
```

Source: [`09_cutting_scored.py`](09_cutting_scored.py)

## Pour water, scored

![Pour water, scored](media/10_water_scored.webp)

The same cup, water and glass as a task document, with a goal on the water's level in the glass: pour as in `03_materials/14_water`, and the goal turns true at the 25 mm line.

The water is a `Water.from_spec` record the document places under its own name, the cup a
`Body.from_spec` object, the glass's walls and its painted line `Body.from_spec` fixtures, and
the physics `rss.WATER_60HZ` under a task-language name. The region is the glass's inside, from
the bench top to its 55 mm rim, and `fill_level` reads the height the water stands there. The
hands start at the YAM's home, pointing ahead. `tests/test_pour_water` drives the same pour
through the right arm's panel and `evaluate` says success, the water 29.5 mm deep in both of two
drives, 99.3 and 99.7% of it in the glass; 25 ms per control step on an RTX PRO 6000, GPUs 1 and 3.
The `[genesis]` extra lives in a virtual environment of its own (`examples/03_materials/README.md`).

```bash
uv run examples/06_tasks/10_water_scored.py
```

Source: [`10_water_scored.py`](10_water_scored.py)

<!-- examples:end -->

## The document

An `Env` is a `Sim`, so `rss.view` and anything else that takes a `Sim` takes it.

| Section | What it says |
| --- | --- |
| `:objects` | What moves: `name - type`, where a type is a registered asset |
| `:fixtures` | What is welded in place, in the same form |
| `:regions` | Boxes `(x0 y0 x1 y1)` in a `:target`'s frame, or on the bench top when none is named; `:z-range` for height |
| `:staging` | The cell, the robot, the physics bundle, cameras, episodes, `:max-steps` |
| `:init` | What must hold at the start, checked once the scene settles |
| `:goal` | One clause: when the episode succeeds |
| `:reward` | `(fraction)` (the default) or `(progress MEASURE (:gate g))` |
| `:observe` | Measures recorded every step that score nothing |
| `:artifacts` | `(:graph LABEL "DIR")`: the graphs that solve it, a folder under the task's home |

## Goals

Comparisons put a measure on the left and a number on the right:
`(>= (containment cube square) 1.0)`.

| Kind | Words |
| --- | --- |
| Logic | `and`, `or`, `not`, `worst-of` |
| Time | `once` (latch the first success), `hold-for SECONDS`, `after`, `peak`, `integral` |
| Measures | `containment`, `speed`, `height`, `on`, `contact-force`, `hands-clear`, `gripped`, `alignment`, `tcp-distance` |

`once` latches success; the reward keeps changing after it. `(fraction)` pays
one share per clause that holds now.

## Objects and physics

A document names an object type: `Body.from_spec` makes one from any rigid or
slab constructor, and `register_asset` registers it. Physics is a named bundle,
never raw numbers:

| Bundle | For |
| --- | --- |
| `mujoco_500hz@v1` | Picking things up: the arm and every free body in one MuJoCo island, 50 Hz over 500 Hz |
| `rigid_mujoco@v1` | Pushing things: free bodies on VBD beside a MuJoCo arm, which pushes but cannot pinch |
| `cloth_vbd@v1` | A sheet beside an arm |

`bench.register_arrangement` registers your own bundle; `bench.register_camera_rig`
registers cameras for `(:cameras NAME)`.

## Running many episodes

`bench.rollout` scores one episode. A `JobSpec` runs many and writes one results
row per episode (see [Policies](../07_policies/README.md#many-worlds)). From a
shell, a task root holds one folder per suite, each with `.task` files and a
`records.py` that registers their types and samplers. `--task-root` adds a root:

```bash
uv run --extra gap scripts/bench_run.py --task-root my_tasks --task push/cube \
    --solve graph --graph pick --episodes 0:16 --num-worlds 16
```

The benchmark's own task suites are not in this release yet.

A scene assembled by hand, with no document, uses `rss.layouts` for the same
kind of spatial draw: see [Spatial layouts](../00_getting_started/README.md#spatial-layouts).

**See also:**

- [Policies](../07_policies/README.md) for the graphs that solve a task
- [Studio](../05_studio/README.md) for the page an `Env` opens on
