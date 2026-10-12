# Examples

Each script builds a scene, and all but the headless render example open the
studio on it in your browser. Run one by path, from the repository root:

```bash
uv run examples/02_cloth/00_towel.py
```

With the package installed, `python examples/02_cloth/00_towel.py` does the
same. No script takes a flag: every parameter is a local at the top of
`main()`, so change a number and run it again. A script that needs an optional
engine says so in its docstring.

Folders and scripts are numbered in reading order, and each script runs on its
own.

| Folder | What is in it |
| --- | --- |
| `00_getting_started/` | The first scripts to run. |
| `01_rigid/` | Boxes, balls and capsules beside two arms: click one and drag it. |
| `02_cloth/` | A towel two grippers can pinch, fabrics side by side, the same sheet on another engine, and a cloth implementation of your own. |
| `03_soft/` | A sponge under an arm, a soft body read from a tet mesh file, and a soft-body implementation of your own. |
| `04_extending/` | A robot of your own, and a new kind of thing in one file. |
| `05_studio/` | The browser page: a panel of your own, and a still and a video rendered with no page open. |
| `06_teleop/` | Driving the simulated arm from a real leader arm, with force feedback. See [`06_teleop/README.md`](06_teleop/README.md). |
| `07_tasks/` | A task as a short document, its goals and rewards, and its episodes. |
| `08_policies/` | A policy as a graph: pick and place, check and retry, through a camera, and in many worlds at once. |
| `09_demos/` | The live demos from the project page: wipe a grab bar with a sponge, route a cable into clips, fold a towel. |
| `10_rope/` | A cable to drag across the bench, a cable clamped between two posts, and the same cable on SuperDex. |
| `live/` | A page that rebuilds `scene.py` on every save, for a coding agent to work in. See [`live/README.md`](live/README.md). |

[`02_cloth/cloth.py`](02_cloth/cloth.py), [`03_soft/soft.py`](03_soft/soft.py)
and [`10_rope/rod/`](10_rope/rod/) are not scripts to run: they are the
implementations `rss.sheet`, `rss.sponge` and `rss.rod` build, kept beside the
examples so they can be read and copied.
