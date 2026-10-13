# Real to sim with an agent: the reference

How a coding agent rebuilds a real bench from a photo on the live page: the
contract for `scene.py`, every `./ctl` command, and what each reading means.
[`README.md`](README.md) has the prompts to start one.

## How it works

One process, one browser tab, one file. `./ctl up` opens the studio on
`scene.py` and rebuilds the scene every time the file is saved.

- The tab never reloads. The camera stays where you left it.
- A file that does not build leaves the previous scene up. `./ctl status`
  prints the traceback.
- A save also re-imports the helper modules beside `scene.py` (a module or a
  package in its directory), so an edit to a helper lands with the next save.
- `./ctl` is the command line onto the same process: status, snapshots,
  stepping, and Python against the live scene.

The agent edits the file and looks at the result through snapshots. You watch
each edit land on the page.


## Start it

```bash
cd examples/live
./ctl up                    # start the server and print its URL (8080, or the next free port)
./ctl up --share            # the same, plus a public URL
./ctl quit                  # stop it
```

Then start the agent in the same directory:

```bash
claude --permission-mode acceptEdits        # or: codex -a on-request -s workspace-write
```

On a remote GPU box, pick a card with `CUDA_VISIBLE_DEVICES`, then use
`--share` or an SSH tunnel (`ssh -L 8080:localhost:8080 <box>`). The page
listens on all interfaces (`0.0.0.0`, viser's default), so on a shared network
firewall its port or use the tunnel. The agent reads the procedure from `AGENTS.md` at
the repository root.

## The contract for `scene.py`

When the file has run, `scene` is an `rss.Scene`. That is all.

```python
import robosimstudio as rss

scene = rss.Scene(cell="void", name="live")
scene.add(rss.robot("yam_i2rt"))  # a scene needs something that moves before it will build
```

- The server builds it with `scene.build(control_mode="joint_position")`. Bind
  `sim` yourself and the server takes that instead.
- A `BUILD = {...}` dict is passed to `build` as keyword arguments.
- `rss.view`, `rss.draw` and `rss.viewer` do nothing while the file runs, so a
  script pasted from the examples works unchanged.
- Poses are in the work-surface frame. See [Frames and units](#frames-and-units).
- Bind `routine` (a callable taking the built `sim` and returning a
  generator that yields once per control step), or `routines`, a dict of
  name to such callables, and the page grows a **Run** button per routine
  with a **Reset** beside them. `./ctl play` presses the same button. A
  routine that declares a `session` parameter is handed the page's.

## `./ctl`

| Command | Does |
| --- | --- |
| `up [--port N] [--scene F] [--share]` | Start a server on the file and wait for the page; it prints the port actually bound (the next one up when N is busy). With a page already up it names that page, and refuses when that page is on another scene file or lacks a requested `--share`. |
| `status` | What is loaded, the names in it and where they are, the last error, the page's address. |
| `reload` | Rebuild now. Saving the file does this too. |
| `snapshot [--camera N] [--beside REF] [--eye x,y,z --look-at x,y,z] [--size WxH] [--note TEXT]` | A path-traced PNG of the current state, with a reference picture beside it. |
| `ref frames CLIP [--at 0:closed,4.5:lid_open,end] [--every S]` | Stills out of a clip at those seconds, a contact sheet with the seconds burnt in, and `ref/timeline.json` anchoring them. |
| `compare [--ref PHOTO] [--camera N] --key NAME=HOW ... [--note TEXT]` | The scene against the photograph: the pair at full size, per keyed thing its size, offset and overlap, and a look sheet for the eye. |
| `compare --take NAME --key ...` | A recorded take against the clip's timeline: each anchor, and what moved in each. |
| `check [--camera N] --key ... [--object NAME ...]` | What a photograph cannot tell: builds, finite, camera at the clip's aspect, every named thing present, nothing sinking, nothing drifting. |
| `log [--last N]` | Every snapshot, compare and check so far, with its score and the `scene.py` it measured. |
| `play [NAME] [--record TAKE --camera N]` | Play a routine the scene file binds, as its button on the page does. |
| `show PATH [--label TEXT]`, `show --clear` | A picture on the page's board for the person watching; `compare` puts its pair, its look sheet and a take's strip there itself. |
| `run N` | Step N control steps, then hold. Refused while a routine plays; ends with an error if the scene is rebuilt or reset during the run. |
| `pause`, `resume` | As named. |
| `reset` | Reset the scene. It also stops a playing routine and finishes its take; Pause is left as it was. |
| `eval "CODE" [--record NAME --every K --camera N]` | Python on the simulator's thread, in `scene.py`'s own namespace: `sim`, `scene`, `np`, `rss` and every name the file bound. What it binds, routines and `play` see until the next save starts the namespace over. A generator plays on the page; `--record` films it. |
| `catalog [robots\|grippers\|cells\|assets]` | The names the registries know. |
| `ref show PATH`, `ref hide` | A picture in the page's sidebar. |
| `quit` | Stop the server. |

`--json` after any command prints the reply raw. Snapshots go to
`run/snap_NNN.png`, and the pair to `run/snap_NNN_vs_ref.png`; a comparison to
`run/cmp_NNN_beside.png` and `run/cmp_NNN_look.png` with its numbers in
`run/cmp_NNN.json`; every reading to `run/log.jsonl`.

## The loop

There is no prescribed order. The page rebuilds on every save, and the
readings below say how far the scene is from the clip; an agent builds,
reads, and edits until a finished twin stands:

- the clip's first frame agrees, to the eye and in numbers: the render beside
  the still (`snapshot --beside`), looked at; each thing the clip shows in
  the scene, at its size and place, through a camera where the clip's was,
  in its detail (the thing up close on the look sheet), with what stands
  behind it (the painted-out pair on the look sheet) (`compare`);
- the gate passes, or the report says which line fails and why that is right
  for this scene (`check`);
- a recorded take moved what the clip moved, by about as much, and its
  before and after match the clip's first and last frames (`eval --record`,
  `compare --take`);
- where the reference cannot settle a number, the report says which number
  and what was assumed.

The readings are instruments, not rules: `check` blocks nothing, and its bars
are defaults. What they replace is judging a side-by-side by eye.
`examples/live/README.md` has a prompt that asks for this, with a clip, a
photo, or neither.

## Acting

`eval` blocks the page while it runs, unless it returns a generator. A generator
is played one `next()` per loop pass, so a routine that steps the sim between
`yield`s is drawn step by step:

```bash
./ctl eval "$(printf 'def go():\n    for _ in range(120):\n        sim.run(1)\n        yield\n')"
./ctl eval "go()"
```

Every rebuild lands running: drag a thing, drive an arm, tick **Pause**.
`up --hold` lands every rebuild paused instead. While a routine plays (from
`eval`, `play` or a **Run** button) it steps the scene itself, so the page's own
stepping is held whatever **Pause** says, and resumes as it was when the
routine ends.

## Notes

- A snapshot is path traced and draws the room, the mat and the textures; the
  page draws the bench and what you added.
- A rebuild of the starter scene with one box added took 7 to 8 s on our
  machine. The first build in a process compiles kernels and takes longer.



## Numbers beside the picture

A side-by-side says the case looks a bit small and somewhat high. `compare`
says it is 0.70x the photograph's size and 78 px too high, with an overlap
(IoU) of 0.31, and the next `compare` says whether the edit helped. The loop
closes on a reading rather than an impression -- what 4DCodeBench found
separates the coding agents that reconstruct a video from the ones that
produce something that resembles it.

### Keys

The render knows what every pixel is: the collider ray tracer's segmentation
names each thing. The photograph does not, so each thing is *keyed*, and the
key says how to find it there:

| key | finds |
| --- | --- |
| `case=red` | the largest patch of that colour (`red orange yellow green cyan blue purple white black grey wood`) |
| `socket=@1240,410` | the patch of the colour at that pixel of the photograph |
| `case=red@980,640` | the patch of that colour containing that pixel |
| `case=#d01a10~0.1` | a hex colour, with a tolerance |

The name is the thing's name in the scene. Pixels are the photograph's own,
the one the agent looked at: a model that can see gives a pixel as readily
as a word, and a pixel is exact.

### What `compare` says

```
case: sim is 0.70x the photo's size (w 0.77x, h 0.60x), centre 5 px left, 78 px high (6.1% of the frame), IoU 0.31
socket: sim is 0.81x the photo's size (w 0.44x, h 0.87x), centre 122 px left, 20 px low (9.7% of the frame), IoU 0.00
```

Per keyed thing: its size against the photograph's, where its centre sits,
the overlap. `score` is the mean overlap. That is all the numbers say, and
on the socket-case clip one camera correction read off them took the case
from an overlap of 0.40 to 0.82.

What the numbers cannot say, the eye does. `snapshot --beside ref/<still>`
puts the render beside the still at full size, and `compare` writes the same
pair as `cmp_NNN_beside.png`: the first thing to open after a save, and the
whole instrument the first twins were built with. Beside it, the **look
sheet** (`cmp_NNN_look.png`) shows:
the pair; the photograph with each thing outlined twice (the photo's outline
yellow, the render's cyan); each keyed thing cut from both pictures at one
scale; and both pictures with the things painted out, which is what stands
behind them. The sheet is for the agent's own eye. A model that can see is
asked to open it and write down what the photograph has that the render
lacks, per thing and behind them -- a lid seam, latch tabs, a label, the
table top, the couch -- and that list is the next edit. Readings that tried
to put a number on detail and background (edge agreement, a DINOv2
similarity, a depth fit) were measured on this clip and cut: none of them
moved an edit that the size, offset and overlap lines had not already
moved, and none told a red box from a case with latches the way one look at
the sheet does.

### The gate

`check` asks what a photograph cannot, and answers each with the number it
passed or failed on:

```
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       phone 1280x720, fov 37 deg; clip 1920x1080
PASS  objects      all of case, socket, extension are in the scene
FAIL  penetration  worst overlap 1.99 mm between socket and table
PASS  rest         nothing moved more than 0.4 mm in 60 steps
```

The rest check steps the scene and puts it back afterwards. `--steps`,
`--settle-mm` and `--penetration-mm` set the bars. On the socket-case twin
the overlap line found the socket two millimetres into the table, which no
picture showed.

### The take

`eval "t.play()" --record take1 --camera phone` films the routine as the page
plays it -- the video, every frame's segmentation at quarter size, every
thing's position -- and `compare --take take1 --key ...` holds it against the
clip's `ref/timeline.json`: each anchor's still beside the take's frame at the
same point of the action, the keyed things' overlap there, and whether each
thing moved in the clip and moved in the take, by how much. The clip and the
take need not run the same length -- a robot is slower than a person. An
anchor is matched by proportion, or pinned with `"sim_t"` or `"sim_step"` in
the timeline when you know when the take reaches it. A seed is a pixel of one
still, so key the things that move by colour for a take. The strip
it writes has the clip's anchors above and the take's frames below; the
agent is asked to look at it and say, for the clip and for the take, what
the first state is, what the last is, what happened between and what touched
what, and to treat a difference as unfinished work.

The agent need not remember to ask. When a take ends, the page holds it
against the clip itself, with the keys of the last `compare`, and puts the
strip on the board under a one-line verdict: `2 of 3 did what the clip did;
lid stayed (clip moved)`. The reading is logged like any other. A session
in which the agent recorded five takes of a routine that never opened the
lid, and compared none of them, is what this is for: the verdict is in
front of the agent and the person watching after every take, and a thing
the clip moved that the take did not is named. Without a timeline or
without a prior `compare --key`, the page prints one line saying which is
missing and does nothing else.

### The page as the agent's board

The person watches the page while the agent works, and what the agent is
looking at is worth their seeing. The `Live` folder at the top of the sidebar
is the agent's board: `./ctl show <png> --label "<what it is>"` puts a
picture there, a label replaces the picture that had it, `--clear` takes
them all off, and `compare` puts its pair and its look sheet there itself,
and a take its strip under its verdict, so every comparison the agent makes
appears beside the scene as it is made, with the per-thing reading the
person would otherwise never see. The board survives a rebuild.

Snapshots and comparisons are numbered from the files in `run/`, not from
the process: a page restarted three times in a session once wrote
`cmp_001` four times over, and the log's early rows pointed at pictures
that no longer held those readings.

Below the board stand the buttons: one **Run** per routine the scene file
binds (`routine`, or `routines` by name) and a **Reset**. A click plays the
routine on the main thread exactly as `./ctl play` does, so the person can
run what the agent built without typing, and run it again after dragging
something. The earlier page had such a button only for a shipped task's
policy; this one is the scene file's own.

### The log

Every `compare`, `check` and `snapshot --note` appends to `run/log.jsonl`
with the hash of the `scene.py` it measured, and `./ctl log` prints them:
before and after an edit are two lines apart. Say what each save changed in
`--note`; the log is the account of how the twin got there.

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
