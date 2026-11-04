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
  prints the traceback. One that builds but will not go into the page leaves
  a bare scene (one YAM) standing in: `status` says `serving: stand-in`. If
  even that will not go in after a rebuild, no scene is in the page
  (`serving: nothing`; at start-up the server exits instead, and `up` says so):
  `snapshot`, `compare`, `check`, `eval`, `play`, `run` and `reset` are
  refused until a save goes in, or `./ctl quit` and `./ctl up`. Until a save
  goes in, the replies of those that act on the scene carry `stale`, saying
  which scene they acted on; while a save is not built yet, they carry
  `pending`.
- The scene's own code is `scene.py` and every `.py` in its folder, two
  folders down -- but not under `run/`, `ref/`, `assets/`, `twins/`, a hidden
  folder or one named `robosimstudio*`, and not a module beside `scene.py`
  named `robosimstudio*.py`, which is taken for the library's. A save to any
  of it rebuilds (the first 500 files are watched), and a rebuild imports it
  afresh: a helper module goes there, and a scratch script in `run/`. A
  rebuild re-imports the helper modules beside `scene.py` (a module or a
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
- A command the page is too busy to start before `ctl` stops waiting for
  its reply (30 s for most; `reload`, `quit`, `reset`, `pause` and `resume`
  5.5 minutes, `catalog` 2, `snapshot`, `compare`, `check` and `ref` 10,
  `play` and `eval` 15) is not run, and the reply says so and what to do:
  after a build or a render, `./ctl status --wait`, then send it again; after
  one of a routine's steps, send it again, and it goes in between two steps
  (`status --wait` does not wait out a routine). One the page started runs
  to its end; if that is after `ctl` stopped waiting, `ctl` says it timed out
  and what the page is busy with.
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
  file runs. An example builds its scene inside `main()`, which the page does
  not call (the file runs as `scene`, so a `__main__` guard does not fire):
  copy the whole file into `scene.py`, move the body of its `main()` to
  module level and drop the guard; its imports and helpers above `main()`
  stay, and its `rss.view` and `rss.draw` calls are harmless (one that drives
  the viewer `rss.viewer` returns is not).
- Poses are in the work-surface frame. See [Frames and units](#frames-and-units).
- The light is `rss.Scene(..., render=...)`; left out, it is the house room.
  Which fields count depends on the renderer that draws every `snapshot`,
  `compare` and take: the path tracer where the GPU has RT cores and ovrtx is
  installed, else the tiled tracer, unless the page was started with
  `up --renderer` (*Renderers*, below; `status` says which draws). Blender
  under `environment="lab"` reads the fields as the path tracer does, and
  otherwise as the tiled tracer does.
  - The path tracer, under `environment="lab"`, lights the scene with one
    soft panel overhead and a neutral dome. Start from the lab room and
    change what it reads:

    ```python
    from dataclasses import replace
    from robosimstudio.core.render.room import LAB

    scene = rss.Scene(..., render=replace(LAB, key_intensity=1.6, ambient=(1.2, 1.2, 1.2)))
    ```

    `key_intensity` scales the panel and `ambient` the dome, by its mean (the
    dome stays neutral); `quality` is `"preview"` (6 passes a frame) or
    `"final"` (12, less flicker in a take). It reads no `light_direction`,
    `fill_lights`, `shadows` or `background`, and draws nothing from
    `environment_map` (setting one only dims the dome).
  - `scenery` is the room it draws around the bench. `LAB`'s is this lab's:
    a gridded mat over the bench top, two white posts at its far corners, a
    black apron along its near edge, and slats and glazing 60 mm past its far
    edge -- a camera further out than them sees the back of a wall, black.
    `""` draws none, and past the bench is then dark. Either way, a wall, a
    window or a backdrop the clip shows is a thing in the scene (a slab, a
    pane, a photo on a slab).
  - The tiled tracer reads `light_direction`, `fill_lights`, `shadows` and
    `background` instead, and draws no panel, dome or scenery.

  A `RenderSpec` given whole keeps none of the room -- each field it leaves
  out is the renderer's own default -- hence `replace(LAB, ...)`. A
  direction, a warmth or a shadow the renderer cannot make goes in the
  report. The page draws its own way: judge the light on the pair.
- A camera with a `background` -- `rss.CameraSpec(name="cam", ..., background="ref/plate_cam.png")`,
  a clean plate taken through that camera (`./ctl ref plate`), read beside
  `scene.py` -- is drawn over that plate in every `snapshot`, `compare`,
  `check` and take through it: what can move (the robot's moving links, the
  things it handles, every deformable) is rendered, and every pixel where the
  render hit nothing or a shape fixed to the world (the bench, a fixture, a
  robot's base, a wall) is the plate's -- so a thing welded to the world that
  the scene moves by rewriting its weld counts as static: give it a joint. The background is then the clip's own
  pixels, and what is left is the camera and what moves. The static shapes
  still collide: a collider not under its picture in the plate shows as a
  thing resting on air or sunk into the table. What moves casts a soft
  shadow onto the plate (the collider tracer draws the camera with and
  without the moving things), from overhead unless `render=` states a
  `light_direction`: aim that the way the clip's shadows fall. On the page,
  the path-traced viewport locked to that camera (its **Lock to** list)
  draws it over the plate too, for the price of a collider pass and a few
  image filters a frame; the free view, the studio's video export and its
  Cameras panel draw the scene alone. A camera on a body (a wrist) is drawn
  without its plate, as is a camera whose plate is not found (beside
  `scene.py` or among the asset roots); each says so once. Anything the path
  tracer draws that has no collider or particle in the model (a co-simulated
  actor with no Newton body) is covered by the plate.
- Bind `routine` (a callable taking the built `sim`, and the page's `session`
  if it declares one, returning a generator), or `routines`, a dict of name to
  such callables, and the page grows a **Run** button per routine with a
  **Reset** beside them (a `routines` entry named `routine` is shadowed by a
  bound `routine`). A click plays the routine on the main thread as
  `./ctl play` does, and `status` lists them under `routines`. How a routine
  is stepped is in [Acting](#acting).

## `./ctl`

| Command | Does |
| --- | --- |
| `new NAME [--ref FILE ...] [--dir D]` | Make a twin's folder, `twins/NAME/` beside the run directory (or `D`): a starter `scene.py`, the files copied into `ref/`, `REPORT.md` with its four headings, and the folder's own `ctl`. Needs no page. Refused from a twin's own folder, and for a `D` inside another scene's own code (where that page would take the twin for its helpers). |
| `up [--port N] [--scene F] [--share] [--renderer R]` | Start a server on the file (default: the `scene.py` beside the run directory), wait for the page, and print its URL (the next free port if N is busy). `--renderer` names what draws every picture: `rtx`, `blender`, `gl` or `tiled` (*Renderers*). With a page already up, names it; refuses if that page is on another scene file, lacks a requested `--share`, or draws with another renderer. |
| `status [--wait [--wait-limit S]]` | What is loaded, the names in it and where they are, the last error, the page's address, how the last routine ended (`last_script`), and `speed`: simulated seconds per wall second. While the page is building or rendering it answers at once with `busy` and for how long; during a routine's step, within 5 s. `--wait` returns once the page is not building or rendering and every save is built (a routine may still be playing: `play --wait` waits on one), giving up after `--wait-limit` seconds (3600). |
| `reload` | Rebuild now. Saving the file does this too. |
| `snapshot [--camera N] [--beside REF] [--eye x,y,z --look-at x,y,z] [--size WxH] [--note TEXT]` | A PNG of the current state from the page's renderer (path-traced by default), with a reference picture beside it. A snapshot draws the room, the mat and the textures; the page draws only the bench and what you added. |
| `ref frames CLIP [--at 0:closed,4.5:lid_open,end] [--every S] [--out D] [--width W]` | Stills out of a clip at those seconds, a contact sheet with the seconds burnt in, and `ref/timeline.json` anchoring them, in `ref/` beside the scene file or `--out`, at the clip's width or `--width`. Needs no page. |
| `ref plate CLIP [--window 0:5,36:52] [--count N] [--paint MASK] [--name F] [--out D]` | A clean plate of a fixed camera's clip: the per-pixel median of `N` (60) frames drawn from those seconds (default: all of it), so whatever passes through a pixel for under half of them is gone, and `ref/plate_<clip>_spread.png`, the share of frames each pixel differs from the plate in (white: half, where a ghost can be). `--paint` fills the white of a mask the clip's size from around it (the nearest pixel outside, blurred across the hole, so a wide hole comes out smeared: fill a large one from another window instead): a thing that stayed put all along. Needs no page. A camera takes it as its `background` (*The contract*). |
| `camera fit --point X,Y,Z=U,V ... --ref STILL [--name N] [--fov DEG] [--free-center]` | The pinhole that puts world points (metres, the frame `CameraSpec` uses) on their pixels in the still: six or more, spread over the picture and in depth. Prints the `rss.CameraSpec(...)` to paste, its field of view, and each point's reprojection error in pixels (a point read off wrong is the one far above the rest); `run/camfit_<name>.png` marks each clicked pixel (yellow) and where the fit puts its point (cyan). `--fov` holds the field of view, `--free-center` solves the principal point too (ten or more points). Needs no page. |
| `compare [--ref PHOTO] [--camera N] [--key NAME=HOW ...] [--note TEXT]` | The scene beside the photograph: the pair at full size and a look sheet; per keyed thing, its size, offset and overlap; and how far the whole picture's colours are from the photo's, over the frame and behind the keyed things. |
| `compare --take NAME [--key ...]` | A recorded take beside the clip's timeline: the clip's anchors above, the take's frames at the same point below, and what moved in the take. |
| `check [--camera N] [--key ...] [--object NAME ...]` | The gate: builds, finite state, camera at the clip's aspect, every named thing present, penetration and drift at rest under their bars. |
| `log [--last N]` | Every compare, check, take and `snapshot --note` so far: what it left (PASS/FAIL, the picture: a snapshot's pair when it made one) and the `scene.py` it measured. |
| `play [NAME] [--record TAKE --camera N --every K] [--wait [--wait-limit S]]` | Play a routine the scene file binds, as its button on the page does; `--record` films a frame every `K` control steps (4). `--wait` returns when it ends and its take is written, printing the page's log as it goes, exits 1 with the traceback if it failed, and gives up after `--wait-limit` seconds (3600). |
| `physics [NAME] [--only frozen,open,...] [--settle-mm MM] [--note TEXT]` | Replay a routine seven ways (written, frozen, open, built, heavy, drop, plain), each from a reset scene, and say whether its physics is real (*The physics replays*): as written, with the robot frozen, with the fingers held open, with the robot as built, with the handled things twice as heavy, with the fingers opened early in the carry (`drop`), and played with nothing hooked in (`plain`). Writes `run/physics.json` and a log line; takes as many times the routine's length as it plays replays. |
| `show PATH [--label TEXT]`, `show --clear` | A picture on the page's board for the person watching. |
| `run N [--no-wait] [--wait-limit S]` | Step N (at least 1) control steps, then hold, waiting until they are taken however slowly the scene steps (`--wait-limit`, 3600 s). Refused while a routine plays. A run cut short is an error saying how, at what step: paused, a routine started, a reset (also `check`'s), or a loop pass that failed end it held; a save that rebuilt the scene, `resume` or another run end it and leave the page to that. `status`'s `last_run` says how the last one ended. |
| `pause`, `resume` | Pause or resume stepping. |
| `reset` | Reset the scene. It also stops a playing routine and finishes its take; Pause is left as it was, unless a `run N` was under way, which ends held. |
| `eval "CODE"` or `eval -` (stdin) `[--record NAME --every K --camera N] [--wait [--wait-limit S]]` | Run Python on the sim thread in `scene.py`'s namespace (`sim`, `scene`, `session`, `np`, `rss`, `live` -- the page's own `LiveServer` --, `server` -- its viser server --, and every name the file bound). What it binds, routines and `play` see until the next save starts the namespace over. Its value is its last line's, when that line is an expression; a generator plays on the page step by step; `--record` films it; `--wait` waits for it as `play --wait` does. |
| `catalog [robots\|grippers\|cells\|assets]` | The names the registries know. |
| `ref show PATH`, `ref hide` | A picture in the page's sidebar. |
| `quit` | Stop the server, once a build, a render or a routine's step under way ends (it waits 5.5 minutes for that; past them it is not run, and says so). |

`--json` after any command prints the reply raw.

Output files, in the `run/` and `ref/` beside the scene file (numbering
continues from the files already in `run/`, across restarts):

- `run/snap_NNN.png`; with `--beside`, also `run/snap_NNN_vs_ref.png`
- `run/cmp_NNN_beside.png`, `run/cmp_NNN_look.png`, `run/cmp_NNN.json`; for a
  take, `run/cmp_NNN_strip.png`
- takes: `run/NAME.mp4`, `run/NAME.json` (every frame's step, time and
  positions), `run/NAME_seg.npz`
- `run/physics.json` (the replays' readings and the lines)
- `run/log.jsonl` (every reading), `run/serve.log` (the server's output)
- from `ref frames`: `ref/tSSS.SS.jpg`, `ref/contact.jpg`, `ref/timeline.json`
- from `ref plate`: `ref/plate_<clip>.png`, `ref/plate_<clip>_spread.png`; from
  `camera fit --ref`: `run/camfit_<name>.png`

## Renderers

Every `snapshot`, `compare` and take is drawn by one renderer, named by
`./ctl up --renderer` (or `ROBOSIMSTUDIO_RENDERER`); `status` says which and
why, and each reading's reply and log line carry its name. A camera over a
clean plate composites under all four, and `compare`'s per-thing readings
(size, place, IoU) come off the colliders whichever draws, so only the
`picture` line changes with the renderer.

| Renderer | Draws | Runs where | A 960x540 still |
| --- | --- | --- | --- |
| `rtx` (default) | the path tracer: materials, the `lab` room's light, the scenery | an NVIDIA GPU with RT cores | under a second; the first in a process compiles shaders |
| `blender` | Cycles: the finishes and the `lab` light, as `rtx` does; no scenery | anywhere Blender 5 is installed (`ROBOSIMSTUDIO_BLENDER`, `blender` on the PATH, or Blender.app); Metal on a Mac | about a second on a GPU or a many-core CPU |
| `gl` | flat colour and a sun, from the shapes as drawn; cloth, ropes and grains from their particles (a liquid as its grains) | anywhere: EGL on a GPU, software GL on a CPU, CGL on a Mac | milliseconds, and about 5 µs more a grain |
| `tiled` | flat colour, from the colliders | anywhere Warp runs | milliseconds on a GPU, a second or two on a CPU |

On a machine with no NVIDIA GPU -- a Mac -- the page steps on the CPU
(MuJoCo's C library) and `rtx` cannot run: start it with `--renderer gl` to
work fast, or `--renderer blender` for pictures that read like the path
tracer's. Do not set the variable in `scene.py`: it outlives the save.

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
  `reach` again works. `reach` raises when the hand does not get within its
  `tolerance` (6 mm) in `settle` seconds;
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
page returns to its previous state when the routine ends -- held, if the
routine cut a `run N` short -- and stays paused through a `reset` while
paused.

## Pitfalls twins have met

Each of these cost a twin a quarter of an hour to find:

- **A downloaded arm is not driven the page's way.** An arm from MuJoCo
  Menagerie (the UR5e among them) brings its own actuators, and under the
  page's joint targets the hand sags and stops short. Twins replaced them with
  plain position servos -- `<position kp="2000" kv="400">` on a UR5e's
  shoulder and elbow, `kp="500" kv="100"` on its wrist -- before tuning
  anything else.
- **Linkage fingers travel as they close.** A Robotiq 2F-85's pads swing up
  and in on their four-bar: the grasp height read at open is not the one at
  shut. Read the pads' colliders at the closing value you command.
- **The contact gap is 2 mm.** Under `rss.GENERIC` rigid shapes start
  touching 2 mm apart: a thing a few millimetres thick is held by the gap,
  not its faces. A thin part wants a smaller gap -- the scene's
  `rss.CollisionSpec(rigid_gap=...)`, or an asset's `contact_gap`
  (`examples/08_demos/wipe_the_bar/bar.py` sets both kinds of knob).
- **Many collider pieces overflow the contact buffers.** A thing built of
  many convex parts (spokes, a rim of segments) in a pile of contacts runs out
  of room: raise `rss.backend("mujoco", nconmax=..., njmax=...)` and
  `CollisionSpec(rigid_contact_max=...)` as `bar.py` does.
- **IK picks an elbow per call.** Two targets solved apart can land on two
  branches and swing the arm through the bench between them: seed each solve
  from the joints of the last one.
- **The first path-traced picture compiles shaders** for minutes, in each new
  process; `./ctl status` says so.

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
picture: colour difference 14.2 (mean CIE76 dE) over the frame, 11.8 behind the keyed things; PSNR 17.9 dB, 19.3 dB behind them (identical pixels read dE 0)
```

The `picture` line is the whole frame's pixels against the photo's: a dE of
about 2 is the least a person sees side by side, 10 a plainly different
colour. Behind the keyed things a camera drawn over its plate reads near 0
where the camera is right.

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

### The physics replays

`physics` plays the routine again as written and six ways changed, each from
a reset scene and each as long as the written one, and prints a line per
finding, PASS and FAIL as `check` says them and NOTE for what the report has
to explain:

```
PASS  contact  nothing ties a thing to the robot, no motor on a thing's joints, nothing welded
PASS  written  finished; moved flower_1, flower_2, flower_3
FAIL  frozen   flower_1 moved 18.6 mm with the robot frozen: it starts inside something, or something other than the robot pushes it
PASS  open     with the fingers held open the task did not happen (RuntimeError: Flower 1 did not remain in the fingers)
FAIL  built    with every friction at most 1 and the arm's torque at 27/10 Nm the routine stopped: RuntimeError: Flower 3 is outside the pitcher
NOTE  heavy    with the handled things 2 times as heavy the routine stopped: RuntimeError: Flower 3 did not remain in the fingers (the grasp's margin)
NOTE  grip     the fingers pressed flower_1 0.7 N (p95 0.7) by hand 2; ...; the solver used friction 2 to 4 at the finger contacts (above any real pad: the built line is the one to trust)
```

- **frozen** holds every target the robot has (arms, fingers, a torso or a
  rail) where it started. A rigid thing that moves more than 5 mm is pushed
  out of something it starts inside (a finger, the table) or by something
  other than the robot; one that moves 2 to 5 mm (a key in a tight socket, a
  bottle loose in a tray) and a deformable still settling (a heap, a liquid,
  a draped cloth) are NOTEs. A thing tied to the robot or flagged kinematic
  fails on the `contact` line whatever it moves, and `frozen` names it when it
  moves past 2 mm (a part on its own joint is read like any other). `settle`
  reads the start: a thing that starts more than 1 mm inside something and
  moves past 2 mm with the robot frozen fails there, even where `frozen`
  calls that move a note. One
  the `contact` line names for a latch, an equality row or a motor in the
  written run is read like any other, with that hold named as what may move
  it. `--settle-mm` sets the 2 mm, and the
  5 mm moves with it (2.5 times).
- **open** holds the fingers fully open and moves the arms as written. A
  pinch fails; a task that still happens is carried by a support, a hook or
  two arms squeezing, and the report says which. What happened is read off
  where the things end, so a routine that says it finished while the thing
  never moved counts as the task not done.
- **built** puts every contact friction at 1.0 or under and caps the arm's
  joint torque at the real arm's, named by the robot or its own joints (a
  YAM's 27 Nm and 10 Nm; a UR3's, UR5's, UR10's or UR16's; a Panda's 87 Nm and
  12 Nm; for another robot, or a Universal Robots arm of no stated size,
  friction only), and where the twin set its own gripper servo it puts back
  the registered hand's (the YAM's 50 N clamp). It
  passes when the routine finishes and the things end where the written run
  left them (within 50 mm and 30°): a grip that only holds at a friction no
  rubber pad has fails here, even when the routine's own checks miss it.
  When `built` lowered nothing (the scene was already as built, so it only
  played the routine again) and nothing in the scene keeps a friction built
  does not cap (a deformable, or a thing another solver steps), a different
  end or a routine's stop is a
  NOTE: the outcome is not repeatable, and the report says so. When it lowered
  something, a different end is a NOTE only where `plain`, which changes
  nothing, carried the same thing (at least a third as far as written) and
  missed it by the same measure (position, shape or turn) past the
  tolerance, `built` missed it by no more than `plain` did plus one
  tolerance, and `built` ended no farther from `plain`'s end than `plain`'s
  is from the written one: the two are samples of the scene's spread, and a
  slip outside it fails. Where `plain`'s own grasp failed, the FAIL says the
  grasp is marginal at any friction. A thing it lost on the way or a thing it
  knocked over that the written run left alone still fails, and so does a
  routine that stops.
  When the written replay did not finish, there is no end to hold it to, and
  `built`, `heavy` and `open` are NOTEs that say so.
- A replay the page cuts (a reset, a save, Stop) or whose own machinery
  breaks gives no report: `./ctl physics` says which replay did not run to a
  reading, and is run again.
- **heavy** is `built` with every handled thing twice its mass: how much margin
  the grasp has.
- **drop** plays it as written and opens the fingers early in the written
  run's carry (where a thing it lifted first reached 90% of its height). What
  the fingers carried falls; a thing that still ends where the written run
  left it fails, unless the robot's other parts still bear its weight (two
  hands squeezing, a hook), which is a NOTE. It catches a weld or a pose
  written every step even while the fingers touch the thing.
- **plain** plays it with nothing hooked in -- no cap, no hold, nothing on
  any hook list -- and reads only at the routine's own yields (and once when
  it returns), cut a little past the written run's length. It fails when a
  thing flies with nothing touching it: a routine that carries it another way
  when it is not replayed. A different end, or a routine that does not finish,
  is a NOTE: the sim is not deterministic, and a loose thing lands elsewhere
  run to run. A routine that never yields while it steps is read only at its
  start and end.
- **model** reads the scene's own numbers: gravity off the Earth's by more
  than 2% fails; a handled thing lighter than expanded polystyrene or heavier
  than lead (its mass over its colliders' volume), and a collider nothing
  draws touching a handled thing (an invisible shelf or lip), are NOTEs to
  explain. A handled thing's drawn shapes that do not collide (a leaf, a
  label) reaching more than 10 mm outside its colliders fail -- they pass
  through what they meet -- and 3 to 10 mm is a NOTE.
- **robot** compares the robot the scene was built with to the registered one
  of its name. A copy of its file (in the twin's folder, edited or not), or
  arm or gripper servo gains and force limits of the twin's own (`arm=`,
  `gripper=`), fail: use the registered robot, placed with `at=` and
  `arms_at=`. A robot built from a file nothing registered is a NOTE: its
  limits are the file's own and are not checked.
- **settle** reads how the scene's own settle, before the first step, moved
  each thing from where it was put, and how deep each sits in something at
  the reset: deeper than a soft contact sinks (5 mm), or more than 1 mm deep
  and pushed out by the frozen run, fails; a shallower rest that stays, a thing
  that starts higher than it was put, or one pushed aside without rising, is a
  NOTE (say which); fallen into place is fine.
- **contact** reads the model and the written replay. A joint between a thing
  and the robot (either way round), an equality row between them that is on at
  any step of the routine (a suction cup's seal is a NOTE instead), a thing flagged
  kinematic or of infinite mass while free to move, a hook put on a substep
  phase while the routine played (a dragger, a pin), or a stiff joint on a
  thing whose target moved while it played (a motor) fails it. A thing welded
  to the world or to another thing, or a hook the routine put on the step
  lists, is a NOTE for the report to explain. A stiff joint held at its target
  is the thing's own elasticity (a rod's bending), a hinge or a slide to the
  world or to furniture is a mechanism, and damping alone is not a motor: all
  pass.
- **flight** is a single free body with nothing within the contact margin
  that does not fall at g: something holds it that is not a contact. Ten
  steps or more in a row fail; fewer are a NOTE (a contact that comes and goes between
  reads). It is not read in a scene with a deformable, whose contacts are not
  in MuJoCo's list.
- **grip** is the median force each hand's fingers put on what they moved,
  and the friction MuJoCo mixed at those contacts (the larger of the two
  surfaces'): a hand whose own pads carry 2 or 4 grips with that, not with the
  number written for the thing.
  It fails when the fingers' median force while touching is more than 1.25
  times what the real hand squeezes with both its jaws (a Robotiq 2F-85's
  235 N a jaw, a Franka hand's 140 N, a YAM's 50 N): a squeeze no real hand
  gives is the arm pressing, not the hand holding.
  The default arrangement gives free rigid things a VBD island coupled to the
  robot, whose contacts never reach MuJoCo's list: for those things the
  replay runs a collision pass of its own after every step, so their flight
  is judged and the fingers touching them is seen (`box (stepped by
  SolverVBD): touched by hand 2, its force not read`). Put the handled things
  in the robot's MuJoCo island (`backends=(("*", rss.backend("mujoco",
  ...)),)`) to have their grip force read. A jointed part of such a thing
  that sags on that solver's joints at the reset is a NOTE in `frozen`, as is
  a part whose stiff joint is its elasticity (a stem bending under its own
  weight).

Every replay starts at the reset with the page held (it is put back as it
was when `physics` ends): a routine that only works once the page has run a
while, letting what starts inside something pop out and settle, fails
`written`, and `frozen` says what moved. A replay holds the robot's joint
targets in a pre-step hook, after the routine's commands land and before the
physics, so it works whichever way the routine was written. A replay other
than `written` is cut a little past the written run's length (10%, at least
25 steps) when its routine is still going: one waiting for a pose the frozen
arm never reaches is read as not done. A routine's own checks stopping it is
a reading, not an error: the line quotes them. A deformable is read by its own particles; `built`
and `heavy` change neither its friction nor its mass, and `drop` follows only a rigid thing's carry,
so where what moved is deformable `heavy` and `drop` are NOTEs that say so. `physics` resets the
scene when it ends; it is refused while a routine plays.

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
with the hash of the files the scene they measured was built from --
`scene.py` and its helpers, as the build read them (`bare` for the stand-in)
-- and the picture they left;
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
