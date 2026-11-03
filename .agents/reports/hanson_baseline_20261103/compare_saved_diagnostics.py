"""Compare existing recordings only; this script cannot run the simulator."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = Path("/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/contact-merge-20261103/robosimstudio/.agents/reports/contact_merge_20261103")
NEW = HERE / "run"
import gzip
with gzip.open(OLD / "diagnostics.json.gz", "rt") as stream:
    old_rows = json.load(stream)
new_rows = json.loads((NEW / "diagnostics.json").read_text())

def measurements(rows):
    final = rows[-1]
    contacts = [row for row in rows if row["finger_contacts"] > 0]
    transit = [row for row in rows if 2.7 <= row["time_s"] <= 5.3]
    carry_start = next(row for row in rows if row["time_s"] >= 3.5)
    carry_end = next(row for row in rows if row["time_s"] >= 4.7)
    tail = [row for row in rows if row["time_s"] >= final["time_s"] - 0.5]
    speed = max(math.dist(a["body_centre_m"], b["body_centre_m"]) / (b["time_s"] - a["time_s"]) for a,b in zip(tail,tail[1:]))
    first = lambda row: {"frame":row["frame"], "time_s":row["time_s"]} if row is not None else None
    moved = carry_end["body_centre_m"][0] - carry_start["body_centre_m"][0]
    min_clearance = min(row["minimum_z_m"] - 0.75 for row in transit)
    retained = bool(transit) and all(row["finger_contacts"] > 0 for row in transit)
    support_min = min(row["minimum_z_m"] - 0.75 for row in tail)
    support_max = max(row["minimum_z_m"] - 0.75 for row in tail)
    return {
        "frames": len(rows), "duration_s": final["time_s"],
        "first_finger_contact":first(contacts[0] if contacts else None),
        "last_finger_contact":first(contacts[-1] if contacts else None),
        "peak_whole_object_clearance_m": max(row["minimum_z_m"] - 0.75 for row in rows),
        "minimum_transit_clearance_m": min_clearance,
        "held_until_open": retained,
        "carry_phase_moved_x_m": moved,
        "final_half_second_contacts_max": max(row["finger_contacts"] for row in tail),
        "final_half_second_clearance_range_m": [support_min, support_max],
        "final_half_second_peak_centroid_speed_m_s": speed,
        "max_frame_inverted_tets": max(row["inverted_tetrahedra"] for row in rows),
        "peak_self_contact_pairs": max(row["self_contact_pairs"] for row in rows),
        "checks_from_existing_diagnostics": {
            "held_until_open": retained,
            "transit_clearance": min_clearance >= 0.020,
            "carry_distance": abs(moved - 0.080) <= 0.020,
            "released": max(row["finger_contacts"] for row in tail) == 0,
            "table_support": support_min >= -0.0005 and support_max <= 0.002,
            "settled_speed": speed <= 0.005,
        },
    }
comparison = {
    "candidate_first_10_s": measurements(old_rows),
    "hanson_first_10_s": measurements([row for row in new_rows if row["time_s"] <= 10.0]),
    "hanson_full_20_s": measurements(new_rows),
    "candidate_native_result": json.loads((OLD / "result.json").read_text()),
    "hanson_native_result": json.loads((NEW / "result.json").read_text()),
    "limits": "Postprocessing existing recordings only. No new simulation or score changes. Hanson native result includes whole-20s substep quality and maximum recorded penetration, but no saved peak jaw effort. Bundle comparison cannot isolate an individual causal change.",
}
(HERE / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps({key:value for key,value in comparison.items() if "native" not in key}, indent=2))
