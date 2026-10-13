# Examples

Each script builds a scene, and all but the headless render example open the
studio on it in your browser. Run one by path, from the repository root:

```bash
uv run examples/02_cloth/00_towel.py
```

With the package installed, `python examples/02_cloth/00_towel.py` does the
same. Apart from the teleoperation scripts in `06_teleop/`, which take the
leader arm's port and calibration as flags, no script takes a flag: every
parameter is a local at the top of `main()`, so change a number and run it
again. A script that needs an optional
engine says so in its docstring.

Folders and scripts are numbered in reading order, and each script runs on its
own. Each folder's README shows every script in it running, says what it shows,
and gives the line that runs it; the guide for that topic follows the examples.
Which engine simulates what, and how to change it, is in [`PHYSICS.md`](PHYSICS.md).

| Folder | What is in it |
| --- | --- |
| [`00_getting_started/`](00_getting_started/README.md) | The first scripts to run: a cube lands under a Panda's hand, and you drag the hand down, close it and lift. |
| [`01_rigid/`](01_rigid/README.md) | Boxes, balls and capsules beside two arms: drag a hand down onto the cube, close it and lift, or click any of them and drag it. |
| [`02_cloth/`](02_cloth/README.md) | A towel a hand lifts by its edge, fabrics side by side, the same sheet on another engine, and a cloth implementation of your own. |
| [`03_soft/`](03_soft/README.md) | A sponge a hand lifts on a Close, a soft body read from a tet mesh file, and a soft-body implementation of your own. |
| [`04_extending/`](04_extending/README.md) | A robot of your own that lifts the cube under it, and a new kind of thing in one file. |
| [`05_studio/`](05_studio/README.md) | The browser page: a panel of your own that tosses the cube, and a still and a video rendered with no page open. |
| [`06_teleop/`](06_teleop/README.md) | Driving the simulated arm from a real leader arm, whose jaws hold a cube by friction, with force feedback. |
| [`07_tasks/`](07_tasks/README.md) | A task as a short document, its goals and rewards, and its episodes; the page shows the goal turn true as you carry the cube into the square. |
| [`08_policies/`](08_policies/README.md) | A policy as a graph: pick and place (press Run on the page), check and retry, through a camera, and in many worlds at once. |
| [`10_rope/`](10_rope/README.md) | A cable a hand lifts by friction, a cable clamped between two posts, and the same cable on SuperDex. |
| [`live/`](live/README.md) | A page that rebuilds `scene.py` on every save, for a coding agent to work in. |

The three whole jobs from the project page -- wipe a grab bar, route a cable,
fold a towel -- are not here but in [`demo/`](../demo/README.md), each in a folder
of its own with a **Run** button that plays the job from the home pose.

[`02_cloth/cloth.py`](02_cloth/cloth.py), [`03_soft/soft.py`](03_soft/soft.py)
and [`10_rope/rod/`](10_rope/rod/) are not scripts to run: they are the
implementations `rss.sheet`, `rss.sponge` and `rss.rod` build, kept beside the
examples so they can be read and copied.
