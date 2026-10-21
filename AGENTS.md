# For the agent working in this repository

RoboSimStudio is a Newton-native simulation toolkit. Packages:

- `robosimstudio/`: the core API; builds and steps scenes.
- `robosimstudio_materials/`: the shipped kinds of deformable thing -- cloth,
  soft bodies, the rope, paper, bags, the brush, tape, cutting, the granular, liquid and
  dough materials on the implicit MPM, and water on Genesis's SPH -- one module or
  package each; it imports only `robosimstudio`.
- `robosimstudio_studio/`: a browser page over any scene.
- `robosimstudio_bench/`: tasks and their scoring.
- `robosimstudio_policy_agent/`: policies as graphs of robot tools.
- `robosimstudio_real2sim_agent/`: the live page that rebuilds a `scene.py`
  on every save, and the `./ctl` command line onto it.

## Read first

1. [`README.md`](README.md): what is here and how to run it.
2. [`examples/00_getting_started/README.md`](examples/00_getting_started/README.md)
   (`Scene`, `Sim`, `Ref`, the constructors) and each example folder's README. Poses
   are in the cell's work-surface frame: `at=(0, 0, 0.2)` is 200 mm over the
   middle of the bench, whatever bench (frames and units:
   [`examples/live/REFERENCE.md`](examples/live/REFERENCE.md#frames-and-units)).
3. `examples/`: numbered scripts, shortest first. `robosimstudio_materials/cloth.py`,
   `robosimstudio_materials/soft.py`, `robosimstudio_materials/rod/`,
   `robosimstudio_materials/paper.py`, `robosimstudio_materials/bag.py`,
   `robosimstudio_materials/brush.py`, `robosimstudio_materials/tape.py`,
   `robosimstudio_materials/cutting/` (over a vendored walk-on-spheres cage cutter),
   `robosimstudio_materials/{granular,liquid,dough}.py` (over the shared
   `robosimstudio_materials/mpm/`) and `robosimstudio_materials/water.py` are the
   shipped cloth, soft-body, rope, paper, bag, brush, tape, cutting, particle-material and
   water implementations; read them to see how each works.

## Checks

These gate a change to the library. A twin built in `examples/live/` is not one:
do not run the tiers for it, and do not add its tests to `tests/`.

- `uv run python scripts/testrun.py --tier fast`: no GPU, gates every commit.
- `uv run --extra cpu --no-group rtx python scripts/testrun.py --tier cpu`: no
  GPU; builds and steps scenes on a CPU (MuJoCo's C library, SuperDex).
- `uv run --extra gap python scripts/testrun.py --tier medium`: needs the GPU;
  builds and steps scenes.
- `uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format . --check`.
- `examples/live/run/` is not committed. A commit message is a lowercase
  `area:` prefix and one declarative sentence
  (`studio: the room, and lights that match the render`).

## Real to sim: a real bench, scene or clip, rebuilt on the live page

When asked to rebuild a real scene from a photo or video, match a reference,
twin a real cell, or build a scene interactively, use the live page in
[`examples/live/`](examples/live/README.md) instead of writing a script and
running it blind: edit `scene.py`, and the page rebuilds it on every save.
**Read [`examples/live/AGENTS.md`](examples/live/AGENTS.md) before the first
edit**: the method, what a finished twin is (among other things: the robot
moves things only through its fingers' contact), and what the report says.
Every command and reading is in
[`examples/live/REFERENCE.md`](examples/live/REFERENCE.md). From
`examples/live/`:

```bash
./ctl status --wait                   # the page, once it is free: names, positions, last error, how the last routine ended
./ctl up                              # if there is none: start one; prints the URL (--share for a public one)
./ctl ref frames <clip> --at 0:start,4.5:lid_open,end:end   # stills named by their second, a contact sheet, ref/timeline.json
./ctl snapshot --camera <cam> --beside ref/<still>          # the render beside the still, full size
./ctl compare --camera <cam> [--key case=red] --note "<why>" # the pair and a look sheet; size, offset, overlap per keyed thing
./ctl check --camera <cam>            # the gate: PASS/FAIL with the number
./ctl play tidy --record take1 --wait # the page's Run button, filmed; returns when it ends, saying how
./ctl log                             # every reading so far, with the picture it left and the scene code it measured
./ctl catalog robots cells            # the names rss.robot() and Scene(cell=) accept
./ctl eval - <<'PY'                   # python against the live scene, several lines from stdin
print(sim['<name>'].position)
PY
```

A snapshot is path traced and draws the room, the mat and the textures; the
page draws the bench and what you added.
