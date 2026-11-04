# Real to sim on the live page

You are rebuilding a real bench, scene or clip as a simulation. What to
deliver is fixed below; how to get there is yours. Codex reads this file by
itself; Claude Code through `CLAUDE.md`. Every command and reading is in
[`REFERENCE.md`](REFERENCE.md), and `./ctl <command> --help` gives its flags.

## The setup

- `scene.py` in the twin's folder (`twins/<name>/`, made by `./ctl new <name>
  --ref <clip>`; if you were started in one, it is made) is rebuilt by the open
  page every time it is saved. `./ctl` is the command line onto that page:
  `./ctl status --wait` after a save, `./ctl up` if no page answers.
- `./ctl eval` runs Python in the page's process (`sim`, `scene`, `rss`, and
  everything `scene.py` bound), on the page's GPU: use it for probes, sweeps,
  measurements, a second scene built headless to try a parameter -- whatever
  you need. Several lines go in a heredoc: `./ctl eval - <<'PY' ... PY`.
- The instruments: `./ctl ref frames` (stills out of a clip, a contact sheet,
  a timeline), `./ctl snapshot --camera C --beside ref/<still>` (the render
  beside the photo), `./ctl compare` (per keyed thing its size, offset and
  overlap, a look sheet, and how far the frame's colours are from the
  photo's), `./ctl check` (finite, drift, overlap), `./ctl play NAME --record
  TAKE --wait` (film a routine; it is put beside the clip), `./ctl physics`
  (the routine replayed with the robot frozen, the fingers open, the robot
  as built and the things heavier), `./ctl log`. `./ctl check --camera C
  --clip ref/<C's clip>` holds a camera on a wrist to the hand its clip shows.
  `./ctl camera fit` solves a camera from six or more points whose world
  place you know and whose pixel you read off a still; `./ctl ref plate`
  makes a clean plate of a fixed camera's clip, which a camera takes as its
  `background` so that every picture through it draws what moves over the
  clip's own pixels (REFERENCE). Open the pictures they write and look at
  them; use the numbers too. Use any other tool you like: your own scripts
  in the twin's `run/`, OpenCV, numpy, a model you download.
- The API: `examples/00_getting_started/README.md`, each example folder's
  README, `robosimstudio_materials/` (cloth, rope/rod, soft, bag, paper, tape,
  granular, liquid, dough, ...), and `./ctl catalog` for robots, grippers,
  cells and assets.
- Pick the GPU with `CUDA_VISIBLE_DEVICES=N ./ctl up`; the page then calls it
  `cuda:0`.

## What a finished twin is

1. **The picture matches.** Through a camera where the clip's camera was
   (its pose and field of view, calibrated, not eyeballed), the first frame
   of the render matches the clip's: the things at their size and place, and
   the background -- the table, the room, what stands behind -- identical
   (a fixed camera over its clean plate: `compare`'s `picture` line reads
   near 0 behind the things). The picture you judge is the path tracer's,
   the renderer the take films with: `up --renderer gl` is for fast work on
   geometry, not for the look you report.
2. **The physics is real.** The robot moves things only through its fingers'
   contact: no pose written onto a thing, no weld, attachment, latch, spring
   or dragger, no motor on a thing's own joints, no keyframed or scripted
   motion of a thing, no collider added to the fingers. The robot is the
   registered one as it ships: its own file (no copy of it in the twin's
   folder) and its own servo gains and force limits (no `arm=` or `gripper=`
   of the twin's), placed with `at=` and, for each arm of a two-arm robot,
   `arms_at=`. A thing keeps its real size and features: a grasp that will
   not hold is fixed in the approach or the contact point, not by enlarging
   the thing or stiffening the robot, and anything changed about a thing is
   a departure the report names. Every visible part of a handled thing
   collides (no leaf or label drawn that passes through what it meets).
   Deformable things are simulated as the material they are and behave like
   the clip (how they bend, sag and slide). `./ctl physics` says whether it holds:
   with the robot frozen nothing moves (a few millimetres is a note to
   explain); with the fingers held open the task
   does not happen (or the report says what carries the thing: a support, a
   hook, two arms squeezing); with the robot as built -- every friction at
   most 1, the real arm's torque -- the routine still finishes (REFERENCE
   says when a difference is only a note: a scene whose end differs run to
   run).
3. **The motion matches.** A `routine` bound in `scene.py` (a Run button on
   the page) does what the clip shows: the same first state, the same last
   state, the same contacts in between -- the clip's grasp, which surface of
   each finger touches which surface of the thing (a pinch on a rim is one
   finger inside it and one outside, not a squeeze across the whole thing),
   not another grasp that happens to hold -- at the clip's pace: each phase
   starts within a second or two of the clip's on the take's strip, and the
   arm takes the path the clip's does, not a detour of its own.

## What has mattered on every twin so far

- **Calibrate every camera first, then hold it.** Points whose place you know
  or can measure -- each robot base's centre, the bench's corners, a mat's
  grid crossings, a fixture's corners, spread over the picture and in depth
  -- go through `./ctl camera fit`; a few pixels of error is a good fit. Then
  the camera stays: when a thing does not line up, the thing moves. A
  calibration that came with the clip is a claim: check it the same way.
  Appearance work over a wrong camera hides wrong geometry.
- **The background is the clip's own pixels.** For a camera that does not
  move, a clean plate (`./ctl ref plate`, with windows in which what moves is
  elsewhere; paint out what never left) given as the camera's `background`.
  A plate that still holds the robot or the thing it handles matches the
  first frame perfectly and leaves a ghost the moment they move: look for one
  on the take's strip, not only on the first frame.
  Every static thing still has to be there as a collider under its picture
  (the bench, a box, a tray, a fixture); its look comes from the plate, and
  the shadows of what moves fall on it from overhead, or from `render=`'s
  `light_direction` when the clip's light comes from elsewhere. A plate is
  the clip's own pixels throughout: a region painted in comes from around it
  (`ref plate --paint`), never a flat fill, and it holds no robot or thing
  that will not be drawn over it. The clip's pixels go behind a fixed camera
  and nowhere else: no plate, frame or straightened photo of the clip laid on
  geometry (a table top, a wall, a plane of the room) -- it carries one
  camera's glare and parallax into every other view. A camera that moves (on
  a wrist) cannot use a plate: what it sees is modelled -- the table, the
  room, the furniture -- with materials of its own (a wood, a paint, a
  fabric) made to look like the plates, and only a printed face (a label, a
  tag, a mat's grid) cut out of a still.
- **The physics is measured, not assumed.** Decide what each thing is --
  rigid, jointed, or one of the materials (a rubber loop is a closed rod, a
  band; a sheet is cloth; a sponge is soft) -- and probe it before filming:
  the scene sits three seconds without creeping, the grasp holds at the force
  the routine uses, and a deformable's stiffness is fitted against a moment
  in the clip where it bends or sags. A mass, a friction or a stiffness is a
  guess until a probe says otherwise; say what the probe read. The hand's
  own pads count too: MuJoCo grips with the larger of the two surfaces'
  friction, so `physics`'s grip line, not the number written for the thing,
  says what held it. Run `./ctl physics` before the report and quote its
  lines.
- **What moves is drawn by the render, so it has to look like the clip.**
  Over a plate the robot's links, its hands and the things it handles are
  the only rendered pixels: give them the clip's colours and finish (an arm
  can read silver, grey or tan under the clip's light; a rubber part is
  black), and check them in the composite, not in the free view. The robot's
  look is its spec's `color`, `finish` and `appearance` (`replace(...)` of the
  registered one), never a copy of its file.
- **A camera on a wrist is calibrated against the hand it carries.** In its
  clip the robot's own palm and fingers sit in the same pixels in every
  frame, so they say where the camera is on the wrist -- its offset, its roll
  (a wrist camera is often mounted turned 180 degrees: the fingers at the top
  of one arm's picture and the bottom of the other's), its field of view --
  and the view follows the arm's joints: read each arm's first pose off the
  fixed cameras and the wrist views together. `./ctl check --camera <wrist
  camera> --clip ref/<its clip>` draws the robot over the clip's hand and
  holds the camera to the robot's own camera on that arm (the registered
  file's; the line prints it): start there, fit its roll and a few
  millimetres, and get the `hand` line to PASS, the cyan outline on the
  yellow, before the routine. Moving the camera or opening its field of view
  until the hand lines up leaves the rest of the view wrong. A wrist view
  that does not match is the camera or the arm, never something to explain
  away.
- **Read the reference before writing the routine**: every camera, its phases
  with their seconds, and the moments of contact -- which surface of the
  gripper touches which surface of the thing, and what the other hand holds.
- **A routine that keeps failing at the same point is fixed in the scene**
  (the mechanism, the friction, the size, the robot's start pose), not
  retried.

REFERENCE's *Pitfalls twins have met* lists what each cost a twin a quarter
of an hour (a downloaded arm's actuators, linkage fingers, the 2 mm contact
gap, contact buffers, IK elbows): read it before the first grasp.

## The report

In the twin's `REPORT.md`, and in short in your last message: what still
differs from the clip (things, background, light, motion); every shortcut or
departure from the clip and why; the readings and pictures you judged by file
name; each number the clip could not settle and what you assumed.

## Keep in mind

- Everything the twin needs lives in its folder (`scene.py`, helper modules,
  `assets/`, `ref/`, `REPORT.md`); scratch goes in its `run/`. It is what the
  person zips and hands on, so leave it in the state you report. Nothing goes
  in the library, `tests/`, the repository root or `/tmp`: a twin is not a
  library change, and the test tiers are not for it.
- Long commands block until they are done, so never sleep and poll:
  `./ctl status --wait` after a save, `./ctl play NAME --wait` and
  `./ctl eval "..." --wait` for a routine. Probe through `./ctl eval` (the
  page's process and GPU) rather than a second process of your own.
- Under `CUDA_VISIBLE_DEVICES=N ./ctl up` the page sees that one card as
  `cuda:0`: leave `ROBOSIMSTUDIO_DEVICE` and `BUILD = {"device": ...}` out, or
  make them `cuda:0`.
- The page rebuilds on save; restarting it is not needed for an edit to land.
  A save that does not build leaves the previous scene up and `./ctl status`
  shows the traceback.
- The person may be dragging things on the page; `./ctl reset` before judging
  a reading if the scene looks moved.
