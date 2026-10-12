---
name: real-to-sim
description: Rebuild a real bench, scene or reference clip in simulation on the live page. Use when asked to real-to-sim a photo or video, twin a real cell, match a reference picture, or build a scene interactively and watch it appear. The page rebuilds examples/live/scene.py on every save; ./ctl measures the scene against the photograph, gates it, and films what the robot does.
---

# Real to sim on the live page

The live page is the studio held open over one file. `examples/live/scene.py`
is executed every time it is saved, the `rss.Scene` it binds is built and
swapped into the open browser, and `./ctl` (run from `examples/live/`) is the
command line onto that process. You never run the scene yourself: you edit
the file, and the page and its readings show what it built.

Reference: `examples/live/REFERENCE.md` (the contract for `scene.py`,
every `ctl` command, what each reading means, the frame and units). The API:
`examples/00_getting_started/README.md` and each example folder's README. The
catalog: `./ctl catalog`. Read those rather than guessing a constructor or a
name.

## What you have

- `./ctl status` / `./ctl up`: a page is up, or start one and tell the user
  its URL. A failed rebuild leaves the previous scene up with the traceback
  in `last_error`.
- `./ctl ref frames <clip> --at 0:start,<t>:<event>,end:end`: stills out of
  the clip named by their second, a contact sheet of the whole clip, and
  `ref/timeline.json` listing the moments you chose. Look at them with your
  image tool. `./ctl ref show ref/<still>` puts one in the page's sidebar.
- `./ctl snapshot --camera <name> --beside ref/<still>`: the render and the
  still side by side, full size. **Open it and look.** What is there, how
  big, where it sits, what it looks like, what stands behind it: your eye
  answers all of these at once, and the first twins were built on this
  alone. Judge every save by it before reaching for a number.
- `./ctl compare --camera <name> --key <thing>=<how> ... --note "<why>"`: the
  same pair at full size (`cmp_NNN_beside.png`) plus the scene against the
  still in numbers. Per keyed thing, how many times the photo's size
  the render has it, how many pixels off its centre sits, the overlap. And
  a **look sheet** (`cmp_NNN_look.png`): the pair, the photo with every
  thing outlined twice (the photo's outline yellow, the render's cyan),
  each thing up close at one scale, and the two pictures with the things
  painted out. **Open it with your image tool every time and write down,
  per thing, what the photo has that the render lacks** -- a lid seam,
  latch tabs, a label, a chamfer, a colour, a texture -- and what stands
  behind the things: the table top, the floor, the wall, the furniture.
  That list is the next edit. The numbers say how far; the sheet says what.
  A key is a colour (`case=red`), a pixel of the photograph the thing is at
  (`socket=@1240,410`), or both (`case=red@980,640`); you have looked at the
  photograph, so you know the pixel.
- `./ctl check --camera <name> --key ...`: what a photograph cannot tell:
  builds, finite, camera at the clip's aspect, keyed things present, worst
  overlap, drift when left alone. One PASS or FAIL per line with its number.
  It blocks nothing; its bars are defaults, and a FAIL you disagree with is
  something to say in the report, with the number.
- `./ctl eval "<generator>" --record <take> --camera <name>`: a routine
  plays on the page and is filmed; then `./ctl compare --take <take> --key ...`
  holds the film against the clip's timeline: each anchor beside the frame
  at the same point of the action, and whether each thing moved in the clip
  and in the take. It writes a strip (`cmp_NNN_strip.png`), the clip's
  anchors above the take's frames. **Look at it and answer, for the clip
  and for the take: what is the first state, what is the last, what
  happened between, what touched what.** Where the two answers differ, the
  take is not done. Every take is held against the clip the moment it
  ends, with the keys of your last `compare`: the strip goes on the board
  under its verdict (`2 of 3 did what the clip did; lid stayed (clip
  moved)`) and the page's log prints it. A take whose verdict names a thing
  that stayed is not a take to fix the robot on; it is a before and after
  that still differ, and the next edit is whatever closes that.
- `./ctl log`: every reading so far, with the hash of the scene code it
  measured.
- **The page is yours to fill.** The person is watching it while you work,
  and the `Live` folder at the top of its sidebar is your board:
  `./ctl show <png> --label "<what it is>"` puts a picture there (a label
  replaces the picture that had it; `--clear` takes them all off), and
  `compare` puts its pair and its look sheet there itself, and a take its
  strip. Show what you are looking at as you look at it: a still from the
  clip, a close-up, the snapshot you are judging.
- **A button for what the robot does.** Bind `routine` in `scene.py` (a
  callable taking the built `sim` and returning a generator that yields once
  per control step), or `routines = {"tidy": ..., "open lid": ...}` for
  several, and the page grows a **Run** button per routine and a **Reset**
  beside them; `./ctl play [name] [--record <take>]` presses the same button
  from the command line. `status` lists them under `routines`. A routine
  that declares a `session` parameter is handed the page's, so it can hold
  the arm panels off while it drives a gripper.
- `./ctl eval`, `run`, `pause`, `resume`, `reset`, `catalog`: as
  named; python against the live scene, stepping, pictures, the registries.

## What a finished twin is

- The first frame agrees: each thing the clip shows is in the scene, at its
  size and place, through a camera where the clip's was -- and it looks like
  the clip's, to your own eye on the pair and the look sheet: the things in their detail
  (a case with its latches, seam and label, not a red box), and what stands
  behind them (the table top, the floor, the wall, the furniture the clip
  shows), with its colours and textures.
- The gate passes, or the report says which line fails and why that is
  right for this scene.
- The take moved what the clip moved, by about as much; the before and the
  after match the clip's first and last frames.
- Where the reference cannot settle a number, the report says which number
  and what was assumed.

How you get there is yours: what to build first, when to compare, how many
saves. Report readings, not impressions: the `./ctl log` lines, the gate,
and the take comparison.

Two things end a twin early and are not finished work: the first frame
placed to the numbers but not looking like the clip (the look sheet's list
of what the photo has that the render lacks still standing), and a routine
debugged take after take while the take's verdict keeps naming a thing the
clip moved and the take did not. When the last look sheet or the last take
on the board says the gap is open, the gap is the task; the mechanism, the
grain or the room that closes it is the next save, before the next take.

## Rules of the page

- Poses are in the cell's work-surface frame, as everywhere in the core API;
  `snapshot --eye/--look-at` are world metres.
- `rss.view`, `rss.draw` and `rss.viewer` are no-ops while the file runs, so
  a script from `examples/` pastes in unchanged.
- A snapshot is path traced and draws the scenery (a room, a mat, a texture);
  the page draws the bench and what you added.
- A rebuild costs about 6 s (measured; half of it is CoACD on the
  fingertips). `eval` blocks the page for as long as it runs unless it
  yields.
- `run/` holds every snapshot, comparison, take and the log, and is not
  committed; `scene.py` is your result, so leave it in the state you are
  reporting.
- What the user drags on the page is what your next reading sees; if the
  scene looks moved, `./ctl reset` before judging it.
