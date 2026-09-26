"""Merge the shards of probe_pfode_reached_sigma.py and print the answer.

The probe answers one question: when the PF-ODE arm exhausts its 1000-evaluation
cap, at what noise level does the terminal structure sit?  See
scripts/probe_pfode_reached_sigma.py for why the question exists and why the
terminal structure is a partially annealed one rather than a sigma_min sample.

The sharded runs write one JSON per shard (--out), each covering the tasks where
index % N == k; this merges them, checks for duplicate tasks across shards (they
are disjoint by construction -- a collision would mean two shards ran the same
task and the merged statistics would double-count it), and reports per
(sigma_max, arm) cell:
    n            runs in the cell (compositions x seeds x shards)
    capped       share that exhausted the cap
    reached_med  median sigma reached over ALL runs (completed ones contribute
                 sigma_min, so this is the schedule progress of the cell)
    capped_med   median sigma reached over the capped runs only -- the number
                 the paper quotes for a truncated trajectory
    frac_done    median fraction of the sigma range completed
    wall_med     median wall time per run

Output: results/phase1/analysis/pfode_reached_sigma.json (+ stdout table).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_DIR = os.path.join(_ROOT, "results", "interim", "pfode_probe")
OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "pfode_reached_sigma.json")
SIGMA_MIN = 0.01


def main() -> None:
    paths = sorted(glob.glob(os.path.join(PROBE_DIR, "shard*.json")))
    if not paths:
        raise SystemExit(f"no shard files under {PROBE_DIR}")
    rows, seen = [], {}
    for p in paths:
        for r in json.load(open(p)):
            key = (r["sigma_max"], r["arm"], r["composition"], r["seed"], r["cand"])
            if key in seen:
                raise SystemExit(f"duplicate task {key} in {p} and {seen[key]} -- "
                                 "two shards ran the same task; merge is unsafe")
            seen[key] = p
            rows.append(r)
    print(f"merged {len(rows)} runs from {len(paths)} shards: "
          f"{[os.path.basename(p) for p in paths]}\n")

    cells = []
    print(f"{'sigma':>5s} {'arm':5s} {'n':>4s} {'capped':>7s} {'reached_med':>11s} "
          f"{'capped_med':>10s} {'capped_q10':>10s} {'frac_done':>9s} {'wall_med':>8s}")
    for sig in sorted({r["sigma_max"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            sel = [r for r in rows if r["sigma_max"] == sig and r["arm"] == arm]
            if not sel:
                continue
            reach = np.array([r["sigma_final"] for r in sel], float)
            capd = np.array([not r["converged"] for r in sel])
            capped_med = float(np.median(reach[capd])) if capd.any() else float("nan")
            frac = (sig - capped_med) / (sig - SIGMA_MIN) if capd.any() else 1.0
            cells.append({
                "sigma_max": sig, "arm": arm, "n": len(sel),
                "capped_share": float(capd.mean()),
                "reached_median": float(np.median(reach)),
                "reached_capped_median": capped_med,
                "reached_capped_q10": (float(np.percentile(reach[capd], 10))
                                       if capd.any() else None),
                "reached_completed_median": (float(np.median(reach[~capd]))
                                             if (~capd).any() else None),
                "schedule_fraction_completed_median": float(frac),
                "wall_median_s": float(np.median([r["wall_time_s"] for r in sel])),
                "n_valid": int(sum(1 for r in sel if r["valid"])),
            })
            print(f"{sig:5g} {arm:5s} {len(sel):4d} {capd.mean()*100:6.1f}% "
                  f"{np.median(reach):11.3f} {capped_med:10.3f} "
                  f"{(np.percentile(reach[capd],10) if capd.any() else float('nan')):10.3f} "
                  f"{frac*100:8.0f}% "
                  f"{np.median([r['wall_time_s'] for r in sel]):7.0f}s")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"cells": cells, "runs": rows,
                   "source_shards": [os.path.basename(p) for p in paths]}, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, _ROOT)}")
    print("\nreading: 'capped_med' is the noise level a truncated trajectory's "
          "terminal structure belongs to; 'frac_done' is the share of the "
          "sigma_max -> sigma_min range the median capped trajectory covered.")


if __name__ == "__main__":
    main()
