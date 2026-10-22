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
- `examples/live/run/` and `examples/live/twins/` are not committed. A commit message is a lowercase
  `area:` prefix and one declarative sentence
  (`studio: the room, and lights that match the render`).

## Real to sim: a real bench, scene or clip, rebuilt on the live page

When asked to rebuild a real scene from a photo or video, match a reference,
twin a real cell, or build a scene interactively, work on the live page in
[`examples/live/`](examples/live/README.md), in a folder of the twin's own that
`./ctl new` makes there: edit its `scene.py`, and the open page rebuilds it on
every save. **Read
[`examples/live/AGENTS.md`](examples/live/AGENTS.md) whole before the first
edit**: it is the method -- how to work the page, what a finished twin is
(the robot moves things only through its fingers' contact), and what the report
says. Every command, output file and reading is in
[`examples/live/REFERENCE.md`](examples/live/REFERENCE.md), and
`./ctl <command> --help` gives a command's flags.
