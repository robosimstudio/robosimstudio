# Real to sim on the live page

The method for any agent rebuilding a real bench, scene or clip in simulation.
Codex reads this file by itself when started in this folder or below it, and
Claude Code through `CLAUDE.md`; the repository's `AGENTS.md` and the
`real-to-sim` skill point here. This file says how to work and what to
deliver; [`REFERENCE.md`](REFERENCE.md) says what every command, flag, file
and reading is, and how the page behaves.

`scene.py` runs on every save, and the `rss.Scene` it binds is built and
swapped into the open page without a reload. `./ctl` is the command line onto
that process. Do not run the scene yourself: edit the file and read what the
page and its readings show. The API is in
`examples/00_getting_started/README.md` and each example folder's README, the
frames and units in REFERENCE, the names in `./ctl catalog`. Read those rather
than guessing a constructor or a name.

## Working the page

- **The twin is its own folder.** If you were started in a twin's folder
  (`twins/<name>/`, with its `REPORT.md`), it is made: work in it. Otherwise,
  before the first edit, from this folder:
  `./ctl new <name> --ref <clip or photo> [--ref ...]` makes `twins/<name>/`
  with a starter `scene.py`, the references copied into `ref/`, an empty
  `REPORT.md` and the folder's own `ctl`. Work from it: `cd twins/<name>`, and
  every `./ctl` below is that folder's, its readings landing in its `run/`.
  Everything the twin needs goes in that folder -- `scene.py`, its helper
  modules beside it or in a folder of them at most two down (not under
  `run/`, `ref/`, `assets/`, `twins/`, a hidden folder or one named
  `robosimstudio*`, and not named `robosimstudio*.py` itself: REFERENCE,
  *How it works*), `assets/`, `ref/` -- and the report in its `REPORT.md`; a
  scratch script goes in its `run/`. Nothing goes in this folder's own files,
  the library, `tests/`, the repository root or `/tmp`: the twin's folder is
  what the person zips and hands on.
- `./ctl status --wait` first: a page is up, or `./ctl up` starts one on the
  folder's `scene.py` (tell the user its URL). Picking up from an earlier
  session: `./ctl log`, then open the last look sheet it names, before the
  next edit.
- Open every picture a command writes with your image viewer (`view_image`
  in Codex): the pair, the look sheet, the strip. The numbers come second.
- Commands that take long block until they are done, so never sleep and poll:
  `./ctl status --wait` after a save (a bag or a rope rebuilds in about a
  minute), `./ctl play NAME --wait` and `./ctl eval "..." --wait` for a routine
  (its output streamed; exit 1 with the traceback if it failed). A command
  that times out says what the page is busy with.
- Several lines of Python go in a heredoc: `./ctl eval - <<'PY' ... PY`.
  `--json` after any command prints the raw reply.
- Pick the GPU with `CUDA_VISIBLE_DEVICES=N ./ctl up`: the page then sees
  that card alone, as `cuda:0`, so leave out `ROBOSIMSTUDIO_DEVICE` and any
  `BUILD = {"device": ...}` (or make them `cuda:0`; `cuda:N` names a card it
  cannot see). Naming the card with those alone only picks where the scene
  builds, and the page still holds about 600 MiB on every other card,
  someone else's included. The first path-traced picture in a process
  compiles its shaders for minutes: `./ctl status` says so, and it is not a
  hang.
- A twin is not a library change: the test tiers and `tests/` are not for
  it.

## The tools

- `./ctl ref frames <clip> --at 0:start,<t>:<event>,end:end`: stills named by
  their second, a contact sheet, and `ref/timeline.json` (the moments you
  chose). `./ctl ref show ref/<still>` puts one in the page's sidebar.
- `./ctl snapshot --camera <name> --beside ref/<still>`: the render beside the
  still, full size. `--eye x,y,z --look-at x,y,z` looks from anywhere else.
- `./ctl compare --camera <name> [--key <thing>=<how> ...] --note "<why>"`: the
  pair (`cmp_NNN_beside.png`) and a look sheet (`cmp_NNN_look.png`: photo
  outlines yellow, render cyan, each keyed thing up close, both pictures with
  the things painted out); per keyed thing its size ratio, offset and overlap.
  On the look sheet, list per thing what the photo has that the render lacks
  (a lid seam, latch tabs, a label, a chamfer, a colour, a texture), what
  stands behind it (table top, floor, wall, furniture, a window) and how it
  is lit (where the light comes from, how bright and how warm, which way the
  shadows fall, what shines): that list is your next edit. Key what keys
  cleanly, typically the thing the robot handles; a colour key that outlines
  a fragment measures nothing, and a mask (`gasket=mask:ref/gasket.png`) is
  the way round it.
- `./ctl check --camera <name> [--key ...]`: PASS/FAIL with its number for
  builds, finite, camera aspect, things present, worst overlap, and drift
  when left alone. Advisory: explain any FAIL you accept, with the number.
- Bind `routine` or `routines = {"open lid": fn, ...}` in `scene.py` for a
  **Run** button per routine; `./ctl play [name] --record <take> --wait` presses
  it and films it. Write a routine one of two ways (REFERENCE, *Acting*):
  taking the page's `session` and driving through the arm panels with
  `robosimstudio_studio.routine`'s `reach`, `grip` and `wait`; or taking only
  `sim` and stepping the scene itself (`sim.run(1)` between yields), the arm
  panels held off meanwhile. A generator `eval` returns is the second kind.
  When a take ends it is put beside the clip: the strip (clip anchors above,
  take frames below) goes on the board, and one line says what moved in the
  take and how far.
- `./ctl show <png> --label "<what>"` posts a picture to the board for the
  person watching. `./ctl log` lists every reading with the picture it left.

## How to go about it

There is no prescribed order of saves or of commands. Four things have mattered
on every twin so far:

- **Read the reference before writing the routine.** Watch the whole clip
  (every camera, if there are several), mark its phases with their seconds
  in `ref frames --at`, and crop the moments of contact: which surface of the
  gripper touches which surface of the thing, and what the other hand holds.
- **Build what the robot handles first:** its shape, how it behaves under the
  hand, and the surfaces it touches. Decide for each thing whether it is
  rigid, jointed, or one of the materials in `robosimstudio_materials`
  (cloth, rope, soft, bag, paper, tape, granular, liquid, dough, ...), and say
  which. The rest of the bench and the room still has to be there before you
  report.
- **A size you were given is a claim.** Measure it in the reference; when the
  two disagree, report both and follow the measurement unless told otherwise.
  A photo hides depth: look at the scene from the side before you finish.
- **When a reading stops moving, test the instrument** before editing the
  scene again: the key's outline on the look sheet, the camera, the robot's
  own model (a hand too long, fingertips that travel as they close). When the
  next edit depends on something only the person can settle, ask in one line.

## What a finished twin is

- The first frame agrees: each thing the clip shows is in the scene at its
  size and place, through a camera where the clip's was, and to your own eye
  on the pair and the look sheet it looks like the clip: the things in their
  detail and what stands behind them, with their colours and textures, under
  light as close to the clip's as the renderer can make it -- as bright, a
  window or a lamp the clip shows built in -- through the scene's `render=`
  (REFERENCE, *The contract for `scene.py`*). Where the light still differs
  (its direction, its warmth, the shadows it casts) the report says so.
- The robot moves things only through its fingers' contact: no pose written
  onto a thing, no latch, weld, attachment, dragger or spring, no motor on the
  thing's own joints, no finger colliders added. If the gripper needs
  purchase, enlarge a real feature and report it as an assumption.
- The take does what the clip does, judged on the strip: the first state, the
  last state, what happened between, what touched what. A routine that keeps
  failing at the same point is fixed in the scene (the mechanism, the
  friction, the size), not retried.
- The gate passes, or the report says which line fails and why that is right.

## The report

In the twin's `REPORT.md`, under its four headings, and in short in your last
message:

- What still differs from the clip: things, background, light, motion.
- Every shortcut that remains (anything the contact rule above forbids, a
  scripted motion standing in for physics) and every way the scene or the
  robot was moved away from the clip to make the routine work, first.
- The gate lines, the look sheet and the strip you judged by their file names
  (`cmp_012_look.png`), and each number the reference could not settle with
  what was assumed.

## Rules of the page

How the page behaves is in REFERENCE (*How it works*, *The contract for
`scene.py`*). What changes how you work:

- Save and read; never restart the page to make an edit land. A save that
  does not build leaves the scene before it up, and every command that acts
  on the scene (`snapshot`, `compare`, `check`, `eval`, `play`, `run`,
  `reset`) says so (`stale`) until a save builds; one sent before a save is
  built says that (`pending`).
- An example from `examples/` goes in whole, with the body of its `main()`
  moved to module level (REFERENCE, *The contract for `scene.py`*).
- What the person drags on the page is what your next reading sees; if the
  scene looks moved, `./ctl reset` before judging it.
- The twin's folder is your result: leave it in the state you report.
  `twins/` and `run/` are not committed.
