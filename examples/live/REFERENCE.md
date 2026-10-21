# live: reference

The `scene.py` contract, every `./ctl` command, and what each reading means.
Setup and the prompts are in [`README.md`](README.md).

## How it works

`./ctl up` serves `scene.py` in one process and one browser tab, and rebuilds
the scene every time the file is saved.

- The tab never reloads, and the camera stays where you left it.
- A file that does not build leaves the previous scene up. `./ctl status`
  prints the traceback.
- A save also re-imports the helper modules beside `scene.py` (a module or a
  package in its directory) and takes back what the file and its helpers
  registered (an entity class, a cell, a gripper, a scenery), so an edit to a
  helper lands with the next save: no new name per save, no restart.
- The page works in `scene.py`'s directory: a file the scene or `eval` reads
  or writes by a relative path is beside `scene.py`.
- A rebuild of the starter scene, with or without a box added, takes about 6
  to 7 s; about 4.8 s of it is CoACD decomposing the YAM fingertip colliders.
  The first build in a process also compiles kernels and takes about 10 s.
- The page listens on all interfaces (`0.0.0.0`, viser's default), so on a
  shared network firewall its port or use an SSH tunnel.

## The contract for `scene.py`

The one requirement: when the file has run, `scene` is an `rss.Scene`.

```python
import robosimstudio as rss

scene = rss.Scene(cell="void", name="live")
scene.add(rss.robot("yam_i2rt"))  # a scene needs something that moves before it will build
```

- The server builds it with `scene.build(control_mode="joint_position")`. Bind
  `sim` yourself and the server takes that instead.
- A `BUILD = {...}` dict is passed to `build` as keyword arguments.
- `rss.view`, `rss.draw` and `rss.viewer` do nothing and return None while the
  file runs, so a script pasted from the examples works unchanged unless it
  drives the viewer `rss.viewer` returns.
- Poses are in the work-surface frame. See [Frames and units](#frames-and-units).
- Bind `routine` (a callable taking the built `sim` and returning a
  generator that yields once per control step), or `routines`, a dict of
  name to such callables, and the page grows a **Run** button per routine
  with a **Reset** beside them. A click plays the routine on the main thread
  as `./ctl play` does, and `status` lists them under `routines`. A routine
  that declares a `session` parameter is handed the page's.

## `./ctl`

| Command | Does |
| --- | --- |
| `up [--port N] [--scene F] [--share]` | Start a server on the file, wait for the page, and print its URL (the next free port if N is busy). With a page already up, names it; refuses if that page is on another scene file or lacks a requested `--share`. |
| `status [--wait]` | What is loaded, the names in it and where they are, the last error, the page's address, how the last routine ended (`last_script`), and `speed`: simulated seconds per wall second. While the page is building or rendering it answers at once with `busy` and for how long. `--wait` returns once the page is free and every save is built. |
| `reload` | Rebuild now. Saving the file does this too. |
| `snapshot [--camera N] [--beside REF] [--eye x,y,z --look-at x,y,z] [--size WxH] [--note TEXT]` | A path-traced PNG of the current state, with a reference picture beside it. A snapshot draws the room, the mat and the textures; the page draws only the bench and what you added. |
| `ref frames CLIP [--at 0:closed,4.5:lid_open,end] [--every S]` | Stills out of a clip at those seconds, a contact sheet with the seconds burnt in, and `ref/timeline.json` anchoring them. Needs no page. |
| `compare [--ref PHOTO] [--camera N] --key NAME=HOW ... [--note TEXT]` | The scene against the photograph: the pair at full size, per keyed thing its size, offset and overlap, and a look sheet. |
| `compare --take NAME --key ...` | A recorded take against the clip's timeline: each anchor, and what moved in each. |
| `check [--camera N] --key ... [--object NAME ...]` | The gate: builds, finite state, camera at the clip's aspect, every named thing present, penetration and drift at rest under their bars. |
| `log [--last N]` | Every snapshot, compare and check so far, with its score and the `scene.py` it measured. |
| `play [NAME] [--record TAKE --camera N] [--wait]` | Play a routine the scene file binds, as its button on the page does. `--wait` returns when it ends, printing the page's log as it goes, and exits 1 with the traceback if it failed. |
| `show PATH [--label TEXT]`, `show --clear` | A picture on the page's board for the person watching. |
| `run N` | Step N control steps, then hold. Refused while a routine plays; ends with an error if the scene is rebuilt or reset during the run. |
| `pause`, `resume` | Pause or resume stepping. |
| `reset` | Reset the scene. It also stops a playing routine and finishes its take; Pause is left as it was. |
| `eval "CODE" [--record NAME --every K --camera N]` | Run Python on the sim thread in `scene.py`'s namespace (`sim`, `scene`, `np`, `rss`, and every name the file bound). What it binds, routines and `play` see until the next save starts the namespace over. A generator plays on the page step by step; `--record` films it; `--wait` waits for it as `play --wait` does. |
| `catalog [robots\|grippers\|cells\|assets]` | The names the registries know. |
| `ref show PATH`, `ref hide` | A picture in the page's sidebar. |
| `quit` | Stop the server. |

`--json` after any command prints the reply raw.

Output files (numbering continues from the files already in `run/`, across
restarts):

- `run/snap_NNN.png`; with `--beside`, also `run/snap_NNN_vs_ref.png`
- `run/cmp_NNN_beside.png`, `run/cmp_NNN_look.png`, `run/cmp_NNN.json`; for a
  take, `run/cmp_NNN_strip.png`
- takes: `run/NAME.mp4`, `run/NAME.json` (every frame's step, time and
  positions), `run/NAME_seg.npz`
- `run/log.jsonl` (every reading), `run/serve.log` (the server's output)
- from `ref frames`: `ref/tSSS.SS.jpg`, `ref/contact.jpg`, `ref/timeline.json`

## Acting

`eval` blocks the page while it runs, unless it returns a generator. A generator
is played one `next()` per loop pass, so a routine that steps the sim between
`yield`s is drawn step by step:

```bash
./ctl eval "$(printf 'def go():\n    for _ in range(120):\n        sim.run(1)\n        yield\n')"
./ctl eval "go()"
```

Rebuilds land running. A routine (from `eval`, `play` or **Run**) plays
whatever **Pause** says, one `next()` per pass, by the studio's rule:

- a routine that takes the page's `session` drives through the arm panels
  (`robosimstudio_studio.routine`'s `reach`, `grip` and `wait`), which stay
  live: a pass whose `next()` only yielded is stepped by the page, one that
  stepped the scene itself (`sim.run(30)` to settle) is not, so `reach`, settle,
  `reach` again works;
- any other routine drives the scene itself (`sim.run(1)` between yields): the
  page never steps it, and the arm panels are suspended while it plays, so
  they do not write the joints and the gripper under it, and given back as
  found when it ends.

A generator `eval` returns is the second kind; to drive through the page,
bind the routine in `scene.py` with a `session` parameter and `play` it. A
routine of the second kind that yields 300 times running without stepping is
stopped, with a line saying why. A routine that takes `session` and writes
joints itself holds the panels off with `session.arms[...].suspend()`. The
page returns to its previous state when the routine ends.

## Readings

`compare` measures each keyed thing in the render against the photograph:
size ratio, centre offset in pixels, and overlap (IoU). Re-run it after each
edit to see whether the edit helped. The method and the finished-twin
criteria are in the
[`real-to-sim` skill](../../.claude/skills/real-to-sim/SKILL.md).

### Keys

The render's segmentation names each thing, a cloth, a rope or a heap
included; the photograph needs a `--key` saying how to find it:

| key | finds |
| --- | --- |
| `case=red` | the largest patch of that colour (`red orange yellow green cyan blue purple white black grey wood`) |
| `socket=@1240,410` | the patch of the colour at that pixel of the photograph |
| `case=red@980,640` | the patch of that colour containing that pixel |
| `case=#d01a10~0.1` | a hex colour, with a tolerance |

The name is the thing's scene name; pixels are in the photograph's own
resolution.

### What `compare` says

```
case: sim is 0.70x the photo's size (w 0.77x, h 0.60x), centre 5 px left, 78 px high (6.1% of the frame), IoU 0.31
socket: sim is 0.81x the photo's size (w 0.44x, h 0.87x), centre 122 px left, 20 px low (9.7% of the frame), IoU 0.00
```

`score` is the mean IoU.

`snapshot --beside ref/<still>` and `compare` (as `cmp_NNN_beside.png`) put
the render beside the still at full size; open it first after each save. The
**look sheet** (`cmp_NNN_look.png`) shows the pair; the photograph with each
thing outlined twice (the photo's outline yellow, the render's cyan); each
keyed thing cropped from both pictures at one scale; and both pictures with
the things painted out, which shows what stands behind them. Write down what
the photograph has that the render lacks, per thing and behind them (a lid
seam, latch tabs, a label, the table top, the couch); that list is the next
edit.

### The gate

`check` prints PASS/FAIL per test with the measured number:

```
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       phone 1280x720, fov 37 deg; clip 1920x1080
PASS  objects      all of case, socket, extension are in the scene
FAIL  penetration  worst overlap 1.99 mm between socket and table
PASS  rest         nothing moved more than 0.4 mm in 60 steps
```

The rest check reads a rigid thing's drift off its centre and a deformable's
off its particle that moved most, so a sagging cloth or a sinking rope fails it;
a named thing with nothing to read is listed as not measured. It steps the
scene, then resets it to its start: a pose set with
`eval` or reached with `run N` is not kept. `check` is refused while a routine
plays. `--steps`, `--settle-mm` and `--penetration-mm` set the bars. The bars
are defaults and `check` blocks nothing.

### The take

`--record NAME` films a routine as the page plays it (the video, every
frame's segmentation at quarter size, every thing's position).
`compare --take NAME --key ...` holds it against the clip's
`ref/timeline.json`: each anchor's still beside the take's frame at the same
point of the action, the keyed things' IoU there, and whether, and by how
much, each thing moved in the clip and in the take. The clip and the take
need not run the same length. An anchor is matched by proportion, or pinned
with `"sim_t"` or `"sim_step"` in the timeline. A seed is a pixel of one
still, so key the things that move by colour. The strip has the clip's
anchors above and the take's frames below.

When a take ends, the page runs this compare itself with the last
`compare`'s keys, logs it like any other reading, and puts the strip on the
board under a one-line verdict: `2 of 3 did what the clip did; lid stayed
(clip moved)`. Without a timeline or a prior `compare --key`, it prints one
line saying which is missing and does nothing else.

### The board

The `Live` folder at the top of the sidebar is a board for the person
watching: `./ctl show <png> --label "<what it is>"` adds a picture (the same
label replaces it), and `--clear` removes them all. `compare` posts its pair
and look sheet, and a take its strip under its verdict, by themselves. The
board survives a rebuild. Below it are the **Run** and **Reset** buttons.

### The log

`compare`, `check` and `snapshot --note` append to `run/log.jsonl` with the
hash of the `scene.py` they measured; `./ctl log` prints it. Use `--note` to
say what each edit changed.

## Frames and units

### Units and axes

- Metres, seconds, radians, kilograms and newtons.
- `+z` is up. Gravity points along `-z`.
- A position is `(x, y, z)`. A pose is `(x, y, z, qx, qy, qz, qw)`: the
  quaternion is scalar-last, as Newton stores it.

### The work-surface frame

Every `at` you give a constructor is in the cell's work-surface frame. Its
origin is the middle of the work surface, at its top, and its axes are the
world's.

```python
import robosimstudio as rss

scene = rss.Scene(cell="bench_075")
cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
sim = scene.build()
sim.run(150).sync()
print(cube.position)    # (0.600, 0.000, 0.775)

cell = rss.get_cell("bench_075")
print(cell.center, cell.surface_z)    # (0.6, 0.0, 0.0) 0.75
```

The cube was placed 200 mm above the middle of the bench. It reads in the world
frame: `bench_075` is a 750 mm bench centred at world `x = 0.6`, and the cube
rests with its centre 25 mm above the top. The same spec moves to another bench
unchanged.

| Cell | Work surface |
| --- | --- |
| `void` | The floor. `z` is a height above the ground. |
| `bench_075` | A bench 750 mm high, 595 mm by 1300 mm. |
| `bench_0728` | A bench 728 mm high, 600 mm by 800 mm. |

`rss.list_cells` lists every cell and
`rss.get_cell` returns one. `center` and
`surface_z` are the frame's origin in the world.

What `at` places:

- Rigid bodies, `slab` and `post`: the centre.
- `sheet`: the origin corner. Subtract half the size to centre a sheet.
- `sponge`: the centre of the bottom face.
- `robot`: the base. `rss.robot("panda", at=(-0.5, 0, 0))` puts it there, and a
  pose `(x, y, z, qx, qy, qz, qw)` also turns it. Left out, a registered robot
  stands at its own mount: at the edge of the work surface, facing `+x`, so the
  work is in front of it.

### Where the world frame shows

- Reads: `ref.position`, `ref.pose`, `ref.velocity` and `sim.read()`.
- Cameras: `rss.CameraSpec` `eye` and `look_at`, and
  `./ctl snapshot --eye --look-at`.

### Time

- `sim.run(n)` advances `n` control steps. Under `rss.GENERIC` a control step
  is 1/60 s, with 16 solver substeps.
- `sim.time` is the simulated time in seconds since the build.
- `sim.sync()` waits for the GPU. Call it before reading.

### Grippers

`sim.step(action, mode="joint_position")` takes, for each arm, its joint
targets and then one gripper value. The gripper value is `0` for open and `1`
for shut. It is not a width in metres.

### Names

Every thing has a unique `name`. `scene.add` refuses a duplicate, and
`sim["cube"]` looks one up.
