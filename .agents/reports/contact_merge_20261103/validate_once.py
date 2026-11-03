"""One merged-scene validation; no retries or parameter search."""
import importlib.metadata
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
sys.path.insert(0, str(RSS / "examples/10_heterogeneous"))
started = time.perf_counter()
import newton
import robosimstudio
import warp
import clawhauser_pick_drop as task

assert Path(newton.__file__).resolve().is_relative_to(NEWTON)
assert Path(robosimstudio.__file__).resolve().is_relative_to(RSS)

def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

metadata = {
    "rss_revision": revision(RSS), "newton_revision": revision(NEWTON),
    "rss_import": robosimstudio.__file__, "newton_import": newton.__file__,
    "warp_version": warp.__version__, "python": sys.version,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "validation": "One 300-frame standalone pickup, X carry, open and settle episode",
    "command": "python -u outputs/contact_merge_20261103/validate_once.py",
}
(OUT / "revisions.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
result = task.run(OUT, steps=300, render_every=1, coarse=False)
result["whole_process_seconds"] = time.perf_counter() - started
(OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
sys.exit(0 if result["success"] else 2)
