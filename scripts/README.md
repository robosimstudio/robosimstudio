# scripts/

Every script here runs by path: `uv run python scripts/<name>.py`. Each answers
`--help`.

## Run and test

| script | what it does |
| --- | --- |
| `live.py` | Open the live page on a `scene.py` and rebuild it on every save. `python -m robosimstudio_real2sim_agent up` is the same thing, detached. |
| `testrun.py` | Run one tier of the test suite: `--tier fast` needs no GPU, `--tier medium` builds and steps scenes. |
| `example_readmes.py` | Write each example folder's README sections from its scripts' docstrings; `--check` says whether they are up to date. |
| `bench_run.py` | Run a policy over a task's episodes and write a run directory: one results row per episode and the success rate. `--task-root` adds a folder of your tasks, `--num-worlds` runs them side by side. |

## Hardware

| script | what it does |
| --- | --- |
| `calibrate_leader_arms.py` | Calibrate a Feetech leader arm against the simulated arm's joint convention, in the browser. `--zero-only` re-zeros a calibrated arm in one capture. |
| `check_leader_signs.py` | Show which leader joints read backwards: a live, read-only readout of every joint's signed delta. |
| `analyze_leader_trace.py` | Read a force-feedback trace (`examples/06_teleop/01_force_feedback.py --trace`) and say what stalled. |

`calibration/yam_leader.json` is a sample calibration for two arms that are not
yours; `udev/99-ch340-serial.rules` gives each USB-serial adapter a fixed device
path. See [`examples/06_teleop/README.md`](../examples/06_teleop/README.md).
