# For the agent working in this repository

RoboSimStudio is a Newton-native simulation toolkit: a core API that puts
things in a place and steps them (`robosimstudio/`), a browser page over any
scene (`robosimstudio_studio/`), tasks and their scoring
(`robosimstudio_bench/`), policies as graphs of robot tools
(`robosimstudio_policy_agent/`), and a seam built for you
(`robosimstudio_real2sim_agent/`): a page that rebuilds a `scene.py` every time
you save it, and a command line onto that page.

## Read first

1. [`README.md`](README.md): what is here and how to run it.
2. [`examples/00_getting_started/`](examples/00_getting_started/) (`Scene`,
   `Sim`, `Ref`, the constructors) and the docstrings of the `rss.` names. Poses
   are in the cell's work-surface frame: `at=(0, 0, 0.2)` is 200 mm over the
   middle of the bench, whatever bench (frames and units:
   [`examples/live/REFERENCE.md`](examples/live/REFERENCE.md#frames-and-units)).
3. `examples/`: numbered scripts, shortest first. `examples/02_cloth/cloth.py`,
   `examples/03_soft/soft.py` and `examples/10_rope/rod/` are the shipped cloth,
   soft-body and rope implementations, kept beside the scripts that run them and
   written to be read.

## Checks

- `uv run python scripts/testrun.py --tier fast`: no GPU, gates every commit.
- `uv run python scripts/testrun.py --tier medium`: needs the GPU; builds and
  steps scenes.
- `uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format . --check`.
- `examples/live/run/` is evidence, not source; it is not committed. A commit
  message is a lowercase `area:` prefix and one declarative sentence
  (`studio: the room, and lights that match the render`).

## Real to sim: a real bench, scene or clip, rebuilt on the live page

When asked to rebuild a real scene from a photo or video, match a reference,
twin a real cell, or build a scene interactively, use the live page rather
than writing a script and running it blind. The page is the studio held open
over one file: you edit the file, and the page and your snapshots show what it
built. From [`examples/live/`](examples/live/README.md):

```bash
./ctl status                          # a page is up (names, positions, last error), or it exits 2
./ctl up                              # if not: start one; prints the URL (--share for a public one)
```

What `./ctl` can tell you, from a clip or a photo:

```bash
./ctl ref frames <clip> --at 0:start,4.5:lid_open,end:end   # stills named by their second, a contact sheet, ref/timeline.json
./ctl snapshot --camera <cam> --beside ref/<still>   # the render beside the still, full size: open it and look
./ctl compare --camera <cam> --key case=red --key socket=@1240,410 --note "<why>"
                                      # the scene against a still: size, offset, overlap per thing, and a look sheet
                                      # (each thing up close, the pictures with the things painted out) for your own eyes
./ctl check --camera <cam> --key ...  # builds, finite, camera aspect, things present, overlap, drift: PASS/FAIL with the number
./ctl eval "t.play()" --record take1 --camera <cam>     # film a routine as the page plays it
./ctl compare --take take1 --key ...  # the film against the clip's timeline: each anchor, and what moved in each
                                      # (done by itself when a take ends, with the last compare's keys; the strip and verdict go on the board)
./ctl log                             # every reading so far, with the scene code it measured
./ctl show ref/t005.00.jpg --label "lid open"   # a picture on the page for the person watching (compare shows its pair and look sheet itself)
./ctl play tidy --record take1        # the Run button on the page, from here: scene.py binds routine or routines={name: fn}
./ctl catalog robots cells            # the names rss.robot() and Scene(cell=) accept
./ctl eval "sim['<name>'].position"   # python against the live scene
```

A finished twin has the clip's first frame agreeing in those numbers and to
your eye on the pair and the look sheet -- the things in their detail, and what stands
behind them -- the gate passing (or the report saying which line fails and
why that is right), and a take that moved what the clip moved. How to get there -- what to build
first, when to compare, how many saves -- is yours. Report readings, not
impressions. A first frame placed to the numbers but not looking like the clip, or a
routine debugged take after take while each take's verdict names a thing the clip moved
and the take did not, is the gap still open, and closing it comes before the next take.
The contract for `scene.py`, every command and every reading are in
[`examples/live/REFERENCE.md`](examples/live/REFERENCE.md). A snapshot is path traced
and draws the room, the mat and the textures; the page draws the bench and what
you added. For Claude Code the same is the `real-to-sim` skill in
[`.claude/skills/`](.claude/skills/real-to-sim/SKILL.md).
