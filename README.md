# <img alt="" src=".github/media/logo.png" height="44" align="top" /> RoboSimStudio

<p align="left">
  <a href="https://github.com/robosimstudio/robosimstudio/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/robosimstudio/robosimstudio/actions/workflows/tests.yml/badge.svg" /></a>
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue" />
</p>

<!-- TODO(release): a row of links here: "Tech report" (arXiv) and "Live demos". -->

RoboSimStudio is an open interactive toolkit that accelerates human and agentic
physics-based Real2Sim of Robot Manipulation Tasks.

<p align="center">
  <img alt="Forty-eight manipulation tasks built with RoboSimStudio, playing side by side" src=".github/media/splash.webp" width="100%" />
</p>

Features include:

- Rigid bodies, cloth, FEM soft bodies and cables in one scene, in a few lines of Python.
- Several physics engines behind one interface (Newton, SuperDex, Genesis).
- A browser studio: click anything and drag it, drive each arm by its gizmo,
  tune materials live, record.
- Tuned robot models (i2rt YAM, Franka Panda), and your own robot as one declaration.
- Real leader-arm teleoperation, with force feedback.
- Tasks as short documents: what is on the bench, when it counts as done, and a
  harness that scores many episodes at once.
- Policies as graphs of robot tools and short scripts, with or without the
  simulator's own answers.
- A live page that rebuilds a scene file on every save, made for coding agents
  that turn a photo into a simulation.

The goal is a toolkit that is (1) short for a simple scene, and (2) open at
every layer: a robot, a material, or a whole kind of physics is registered from
outside the library, not patched into it.

Every feature has a short runnable script under [`examples/`](examples/).

## Quick start

You need an NVIDIA GPU.

```bash
git clone https://github.com/robosimstudio/robosimstudio.git && cd robosimstudio
uv run examples/00_getting_started/00_hello_world.py          # Installs on the first run, then opens the studio.
uv run --extra superdex examples/03_soft/00_sponge.py         # Plus FEM soft bodies (Python 3.12).
uv run --extra gap examples/08_policies/00_pick_and_place.py  # Plus the graph runtime: a policy picks a cube.
```

Every script under `examples/` runs the same way.

## Demos

The live demos from the project page run from [`examples/09_demos/`](examples/09_demos/):
wipe a grab bar with a sponge, route a cable into clips, and fold a towel, each
by dragging the robot's hand in the studio.

## Citation

<!-- TODO(release): authors and arXiv identifier. -->
```bibtex
@misc{robosimstudio2026,
  title = {RoboSimStudio: An Interactive Agentic Simulation Toolkit},
  year  = {2026},
}
```

## Acknowledgements

RoboSimStudio is built on [Newton](https://github.com/newton-physics/newton),
[Warp](https://github.com/NVIDIA/warp) and
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), and its studio
on [viser](https://github.com/viser-project/viser). It draws on
[Isaac Lab](https://github.com/isaac-sim/IsaacLab),
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena),
[PyRoKi](https://github.com/chungmin99/pyroki),
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis),
[SuperDex](https://github.com/facebookresearch/project_superdex) and
[LeRobot](https://github.com/huggingface/lerobot). Robot models come from
[i2rt](https://github.com/i2rt-robotics/i2rt) and
[robosuite](https://github.com/ARISE-Initiative/robosuite); see `NOTICE`.
Thanks to their authors.
