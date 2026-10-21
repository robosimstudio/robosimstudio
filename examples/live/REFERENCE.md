# live: reference

The `scene.py` contract, every `./ctl` command, and what each reading means.
How to work and what to deliver are in [`AGENTS.md`](AGENTS.md); setup and the
prompts in [`README.md`](README.md).

## How it works

`./ctl up` serves `scene.py` in one process and one browser tab, and rebuilds
the scene every time the file is saved. A twin lives in a folder of its own,
made by `./ctl new` (`twins/<name>/`: `scene.py`, `ref/`, `REPORT.md`, and a
`ctl` that pins `run/` in the folder); from there `./ctl up` serves that
folder's `scene.py`, and everything below -- `ref/`, `run/`, the helpers --
is that folder's. `twins/` is not committed.

- The tab never reloads, and the camera stays where you left it.
- A file that does not build leaves the previous scene up. `./ctl status`
  prints the traceback.
- A save also re-imports the helper modules beside `scene.py` (a module or a
  package in its directory) and takes back what the file and its helpers
  registered (an entity class, a cell, a gripper, a scenery), so an edit to a
  helper lands with the next save: no new name per save, no restart.
- The page works in `scene.py`'s directory: a file the scene or `eval` reads
  or writes by a relative path is beside `scene.py`.
- A rebuild of a starter scene -- one YAM here, two on a bench in a twin's
  folder -- takes about 6 to 8 s, most of it CoACD decomposing the YAM
  fingertip colliders. The first build in a process also compiles kernels and
  takes about 10 s.
  Materials cost more, measured on twins: a heap or a liquid on the MPM about
  20 to 30 s, cloth leaves on rope stalks 30 to 50 s, a SuperDex rope about
  50 s, a SuperDex bag 60 to 90 s. `./ctl status` says `busy` meanwhile, and
  `./ctl status --wait` returns when it is done.
- The first path-traced picture in a process (a snapshot, a compare, a take's
  first frame) compiles the renderer's shaders and can take minutes; later
  ones take seconds.
- The page listens on all interfaces (`0.0.0.0`, viser's default), so on a
  shared network firewall its port or use an SSH tunnel.

## The contract for `scene.py`

The one requirement: when the file has run, `scene` is an `rss.Scene`.

```python
import robosimstudio as rss

scene = rss.Scene(cell="bench_075", name="lid")
scene.add(rss.robot("yam_bimanual"))  # a scene needs something that moves before it will build
```

- The server builds it with `scene.build(control_mode="joint_position")`. Bind
  `sim` yourself and the server takes that instead.
- A `BUILD = {...}` dict is passed to `build` as keyword arguments.
- `rss.view`, `rss.draw` and `rss.viewer` do nothing and return None while the
  file runs, so a script pasted from the examples works unchanged unless it
  drives the viewer `rss.viewer` returns.
- Poses are in the work-surface frame. See [Frames and units](#frames-and-units).
- Bind `routine` (a callable taking the built `sim`, and the page's `session`
  if it declares one, returning a generator), or `routines`, a dict of name to
  such callables, and the page grows a **Run** button per routine with a
  **Reset** beside them. A click plays the routine on the main thread as
  `./ctl play` does, and `status` lists them under `routines`. How a routine
  is stepped is in [Acting](#acting).

## `./ctl`

| Command | Does |
| --- | --- |
| `new NAME [--ref FILE ...] [--dir D]` | Make a twin's folder, `twins/NAME/` beside the run directory (or `D`): a starter `scene.py`, the files copied into `ref/`, `REPORT.md` with its four headings, and the folder's own `ctl`. Needs no page. |
| `up [--port N] [--scene F] [--share]` | Start a server on the file (default: the `scene.py` beside the run directory), wait for the page, and print its URL (the next free port if N is busy). With a page already up, names it; refuses if that page is on another scene file or lacks a requested `--share`. |
| `status [--wait [--wait-limit S]]` | What is loaded, the names in it and where they are, the last error, the page's address, how the last routine ended (`last_script`), and `speed`: simulated seconds per wall second. While the page is building or rendering it answers at once with `busy` and for how long. `--wait` returns once the page is not building or rendering and every save is built (a routine may still be playing: `play --wait` waits on one), giving up after `--wait-limit` seconds (3600). |
| `reload` | Rebuild now. Saving the file does this too. |
| `snapshot [--camera N] [--beside REF] [--eye x,y,z --look-at x,y,z] [--size WxH] [--note TEXT]` | A path-traced PNG of the current state, with a reference picture beside it. A snapshot draws the room, the mat and the textures; the page draws only the bench and what you added. |
| `ref frames CLIP [--at 0:closed,4.5:lid_open,end] [--every S] [--out D] [--width W]` | Stills out of a clip at those seconds, a contact sheet with the seconds burnt in, and `ref/timeline.json` anchoring them, in `ref/` beside the scene file or `--out`, at the clip's width or `--width`. Needs no page. |
| `compare [--ref PHOTO] [--camera N] [--key NAME=HOW ...] [--note TEXT]` | The scene beside the photograph: the pair at full size and a look sheet; per keyed thing, its size, offset and overlap. |
| `compare --take NAME [--key ...]` | A recorded take beside the clip's timeline: the clip's anchors above, the take's frames at the same point below, and what moved in the take. |
| `check [--camera N] [--key ...] [--object NAME ...]` | The gate: builds, finite state, camera at the clip's aspect, every named thing present, penetration and drift at rest under their bars. |
| `log [--last N]` | Every compare, check, take and `snapshot --note` so far: what it left (PASS/FAIL, the picture: a snapshot's pair when it made one) and the `scene.py` it measured. |
| `play [NAME] [--record TAKE --camera N --every K] [--wait [--wait-limit S]]` | Play a routine the scene file binds, as its button on the page does; `--record` films a frame every `K` control steps (4). `--wait` returns when it ends and its take is written, printing the page's log as it goes, exits 1 with the traceback if it failed, and gives up after `--wait-limit` seconds (3600). |
| `show PATH [--label TEXT]`, `show --clear` | A picture on the page's board for the person watching. |
| `run N` | Step N control steps, then hold. Refused while a routine plays; ends with an error if the scene is rebuilt or reset during the run. |
| `pause`, `resume` | Pause or resume stepping. |
| `reset` | Reset the scene. It also stops a playing routine and finishes its take; Pause is left as it was. |
| `eval "CODE"` or `eval -` (stdin) `[--record NAME --every K --camera N] [--wait [--wait-limit S]]` | Run Python on the sim thread in `scene.py`'s namespace (`sim`, `scene`, `session`, `np`, `rss`, `live` -- the page's server -- and every name the file bound). What it binds, routines and `play` see until the next save starts the namespace over. A generator plays on the page step by step; `--record` films it; `--wait` waits for it as `play --wait` does. |
| `catalog [robots\|grippers\|cells\|assets]` | The names the registries know. |
| `ref show PATH`, `ref hide` | A picture in the page's sidebar. |
| `quit` | Stop the server. |

`--json` after any command prints the reply raw.

Output files, in the `run/` and `ref/` beside the scene file (numbering
continues from the files already in `run/`, across restarts):

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
./ctl eval - <<'PY'
def go():
    for _ in range(120):
        sim.run(1)
        yield
PY
./ctl eval "go()" --wait
```

`eval -` reads the code from stdin, so a heredoc carries several lines without
quoting; what it defines stays in the namespace until the next save.

Rebuilds land running. A routine (from `eval`, `play` or **Run**) plays
whatever **Pause** says, one `next()` per pass, by its kind (unlike the
studio's own Run buttons, which pause with the page and step every routine):

- a routine that takes the page's `session` drives through the arm panels
  (`robosimstudio_studio.routine`'s `reach`, `grip` and `wait`), which stay
  live: a pass whose `next()` only yielded is stepped by the page, one that
  stepped the scene itself (`sim.run(30)` to settle) is not, so `reach`, settle,
  `reach` again works;
- any other routine drives the scene itself (`sim.run(1)` between yields): the
  page never steps it, and the arm panels are suspended while it plays, so
  they do not write the joints and the gripper under it.

When a routine ends, however it ends, the arm panels are put back as they
were when it started, one it suspended itself included.

A generator `eval` returns is the second kind; to drive through the page,
bind the routine in `scene.py` with a `session` parameter and `play` it. A
routine of the second kind that yields 300 times running without stepping is
stopped, with a line saying why. A routine that takes `session` and writes
joints itself holds the panels off with `session.arms[...].suspend()`. The
page returns to its previous state when the routine ends, and stays paused
through a `reset` while paused.

## Readings

`compare` puts the render beside the photograph and writes the look sheet;
for each keyed thing it also measures size ratio, centre offset in pixels, and
overlap (IoU). Keys are optional.

### Keys

The render's segmentation names each thing, a cloth, a rope or a heap
included; the photograph needs a `--key` saying how to find it:

| key | finds |
| --- | --- |
| `case=red` | the largest patch of that colour (`red orange yellow green cyan blue purple white black grey wood`) |
| `socket=@1240,410` | the patch of the colour at that pixel of the photograph |
| `case=red@980,640` | the patch of that colour containing that pixel |
| `case=#d01a10~0.1` | a hex colour, with a tolerance |
| `gasket=mask:ref/gasket.png` | a mask drawn some other way: every pixel of the PNG that is not black |

The name is the thing's scene name; pixels are in the photograph's own
resolution. A colour key on a black, many-coloured, shiny or thin thing finds
a fragment of it, which its yellow outline on the look sheet shows; a mask
finds what was drawn.

### What `compare` says

```
case: sim is 0.70x the photo's size (w 0.77x, h 0.60x), centre 5 px left, 78 px high (6.1% of the frame), IoU 0.31
socket: sim is 0.81x the photo's size (w 0.44x, h 0.87x), centre 122 px left, 20 px low (9.7% of the frame), IoU 0.00
```

`snapshot --beside ref/<still>` and `compare` (as `cmp_NNN_beside.png`) put
the render beside the still at full size, render on the left, photo on the
right, as every pair here; open it first after each save. The **look sheet**
(`cmp_NNN_look.png`) shows the pair; with keys, the photograph with each thing
outlined twice (the photo's outline yellow, the render's cyan) and each keyed
thing cropped from both pictures at one scale; and both pictures with the
things painted out, which shows what stands behind them.

The render's outlines come off the collider ray tracer, the picture off the
path tracer: a thing drawn with a skin of its own (grains over a heap's
particles, a mesh over a simple collider) is outlined by what collides.

### The gate

`check` prints PASS/FAIL per test with the measured number:

```
PASS  builds       the last rebuild landed
PASS  finite       every number in the state is finite
PASS  camera       phone 1280x720, fov 37 deg; clip 1920x1080
PASS  objects      all of case, socket, extension are in the scene
FAIL  penetration  worst overlap 1.99 mm between socket and table (bar 1 mm)
PASS  rest         nothing moved more than 0.40 mm in 60 steps (bar 2 mm)
PASS  finite_after still finite after 60 steps
```

The rest check reads a rigid thing's drift off its centre; a cloth's or a
rope's off its particle that moved most, so a sag or a sink fails it; and a
heap's or a pool's off the 95th percentile of its particles, so one grain that
never settles does not. A named thing with nothing to read is listed as not
measured. It steps the scene, then resets it to its start: a pose set with
`eval` or reached with `run N` is not kept. `check` is refused while a routine
plays.

`--steps` (60), `--settle-mm` (2) and `--penetration-mm` (1) set the bars; the
penetration and rest lines print the bar they were held to. The camera line
holds the camera's aspect to the clip's in `ref/timeline.json`, within 2%;
against a photo, `compare` says when it had to stretch the photo to the
camera's aspect. A line given nothing to look at (`objects` with no `--key`
or `--object`) is a SKIP. The bars are defaults, and `check` blocks nothing.

### The take

`--record NAME` films a routine as the page plays it (the video, every
frame's segmentation at half size each way, every thing's position). When it ends,
the page puts it beside the clip's `ref/timeline.json` by itself, as
`compare --take NAME` does: a strip (`cmp_NNN_strip.png`) with the clip's
anchors above and the take's frames at the same point of the action below,
posted to the board and logged, and one line on what moved in the take, read
off its recorded positions (a jointed thing's moving bodies and a
deformable's particles included). With no `ref/timeline.json` there is no
strip: the line is printed and logged with the take.

```
in the take: case_lid went 118 mm and ended 4 mm from its start; case went 3 mm; 5 other things stayed within 2 mm
```

Nothing says whether the take did what the clip did: that is read off the
strip. Each thing is read at the centre of mass of each of its bodies, or a
sample of a deformable's particles: "went" is the most any of them went,
"ended" the most any of them ended from where it started, so a key turning in
a case whose base stays put, or a lid swinging on its hinge, is seen.

The clip and the take need not run the same length. An anchor is matched by
proportion, or pinned with `"sim_t"` or `"sim_step"` in the timeline; one
pinned past the take's end is held against its last frame, and a line says
so. Keys -- `--key`, else the last still `compare`'s -- only outline the
things: a colour key in both rows; a seed or a mask, which keys one still, in
the take's row alone.

### The board

The `Live` folder at the top of the sidebar is a board for the person
watching: `./ctl show <png> --label "<what it is>"` adds a picture (the same
label replaces it), and `--clear` removes them all. `compare` posts its pair
and look sheet, and a take its strip, by themselves. The
board survives a rebuild. Below it are the **Run** and **Reset** buttons.

### The log

`compare`, `check`, a take and `snapshot --note` append to `run/log.jsonl`
with the hash of the `scene.py` they measured and the picture they left;
`./ctl log` prints it. `--note` says what each edit changed.

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
