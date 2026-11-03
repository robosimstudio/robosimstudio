"""Execute Hanson's published pick/drop defaults without modifying either source tree."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RSS = ROOT / "robosimstudio"
NEWTON = ROOT / "newton"
OUT = Path(__file__).resolve().parent / "run"
OUT.mkdir(exist_ok=True)
expected = {RSS: "d50f86f51b31a1fbc42844ef78a4b851eb693159", NEWTON: "3d3f9e4c13fd0e60e7e53a9870946c0983d1d14c"}
for repo, sha in expected.items():
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert actual == sha, (repo, actual)
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD"], check=True)

sys.path.insert(0, str(RSS / "examples/10_heterogeneous"))
started = time.perf_counter()
import newton
import robosimstudio
import warp
import clawhauser_pick_drop as task
assert Path(newton.__file__).resolve().is_relative_to(NEWTON)
assert Path(robosimstudio.__file__).resolve().is_relative_to(RSS)
metadata = {
    "rss_revision": expected[RSS], "newton_revision": expected[NEWTON],
    "rss_import": robosimstudio.__file__, "newton_import": newton.__file__,
    "warp_version": warp.__version__, "python": sys.version,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "tracked_source_diff": "empty in both repositories before run",
    "task_source_sha256": hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest(),
    "run_defaults": {name: param.default for name, param in inspect.signature(task.run).parameters.items() if param.default is not inspect.Parameter.empty},
    "duration_s": task.DURATION, "control_hz": task.CONTROL_HZ,
    "substeps": task.SUBSTEPS, "frames": round(task.DURATION * task.CONTROL_HZ),
    "overrides": {"output_directory": str(OUT), "render_every": 1},
    "note": "All physical, solver, trajectory and scoring defaults are Hanson's unchanged published source. Existing diagnostics will be assessed separately against the same pickup/carry criteria as the prior run.",
}
(OUT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
result = task.run(OUT, steps=round(task.DURATION * task.CONTROL_HZ), render_every=1)
(OUT / "process_timing.json").write_text(json.dumps({"whole_process_seconds": time.perf_counter() - started}, indent=2) + "\n")
for repo in expected:
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD"], check=True)
