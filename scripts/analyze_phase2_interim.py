#!/usr/bin/env python
"""Interim Phase-2 readout -- the same math as `run_phase2.py --analyze`,
redirected out of the canonical summary directory and tagged with completeness.

Why this exists instead of just calling `--analyze`:

`run_phase2.analyze()` writes `results/phase2/summary/phase2_esen.csv` and
`phase2_<dataset>_esen.json` with **no completeness gate** -- `_load_tasks()`
accepts every file that passes `_task_ok()` (same protocol + config) and the
writer runs as soon as one group has a single task.  Run mid-fleet that leaves
canonical-looking artifacts built from a partial grid, which is exactly what the
project's analysis-layer rule forbids.  So:

  * `rp.SUMMARY` is monkeypatched to a dated directory under `results/interim/`,
    which is outside `results/phase2/` entirely, so no glob over the run tree
    (progress counts, per-config cost measurements) can ever pick these files up;
  * every row carries `n_tasks` / `n_expected` and a `complete` flag, so a
    partial row physically cannot be quoted as a final number.

The per-cell and per-group math is *not* reimplemented -- it is the tested
`analyze_cell` / `cell_metrics` / `hull_summary` path, unchanged.

Usage:  python scripts/analyze_phase2_interim.py [--nnp esen] [--tag latest]
        python scripts/analyze_phase2_interim.py --samplers ald --tag latest_ald

`--samplers` restricts which arms are analysed.  A group costs ~90-125 s (the
expensive part is match_coverage structure matching and the hull summaries, not
the JSON read), so the ALD-only sweep is 18 groups instead of 27.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import run_phase2 as rp  # noqa: E402

# (json key, csv/markdown header) pulled from analyze_cell's aggregate block.
SUN_COLS = [
    ("sun_rate", "SUN"),
    ("validity_rate", "valid"),
    ("stable_rate", "stable"),
    ("novel_rate", "novel"),
    ("unique_rate", "unique"),
]
DYN_COLS = [
    ("ood_rate", "OOD"),
    ("induced_ood_rate", "indOOD"),
    ("start_ood_rate", "startOOD"),
    ("collapse_rate", "collapse"),
    ("capped_rate", "capped"),
]
GEOM_COLS = [
    ("amsd", "AMSD"),
    ("match_rate", "match"),
    ("coverage", "cov"),
    ("r_angle_kl", "rKL"),
]


def _m(v):
    """Render a _mean_std block as `mean` (or `n/a`)."""
    if v is None:
        return "n/a"
    mean, std = v.get("mean"), v.get("std")
    if mean is None:
        return "n/a"
    return f"{mean:.3f}" if std is None else f"{mean:.3f}±{std:.3f}"


def _m3(v):
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        v = v.get("mean")
    return "n/a" if v is None else f"{v:.3f}"


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--nnp", default="esen", choices=["mace", "esen"])
    ap_.add_argument("--tag", default="latest")
    ap_.add_argument("--samplers", default="ald,pfode",
                     help="comma-separated arms to analyse (ald, pfode)")
    args = ap_.parse_args()
    samplers = tuple(s.strip() for s in args.samplers.split(",") if s.strip())

    out_dir = REPO / "results" / "interim" / f"phase2_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Redirect every write analyze() makes.  It resolves the module global at
    # call time, so this is sufficient -- nothing else in the module writes.
    canonical = rp.SUMMARY
    assert out_dir != canonical and canonical not in out_dir.parents, \
        f"refusing to write into the canonical summary dir ({canonical})"
    rp.SUMMARY = out_dir
    print(f"[interim] canonical summary left untouched: {canonical}")
    print(f"[interim] writing to {out_dir}")

    print(f"[interim] arms: {samplers}")
    rp.analyze(nnp=args.nnp, samplers=samplers)

    # n_expected is the full grid for one (dataset, config, sampler) group.
    n_expected = len(rp.SEEDS) * rp.N_CELLS

    rows = []
    for ds in rp.DATASETS:
        blob = json.loads((out_dir / f"phase2_{ds}_{args.nnp}.json").read_text())
        for key, grp in blob["configs"].items():
            agg = grp["aggregate"]
            row = {
                "dataset": ds,
                "config": grp["config"],
                "sampler": grp["sampler"],
                "n_tasks": agg["n_tasks"],
                "n_expected": n_expected,
                "complete": "yes" if agg["n_tasks"] >= n_expected else "NO",
                "n_traj": agg["n_traj"],
            }
            for k, h in SUN_COLS + GEOM_COLS:
                row[h] = _m(agg.get(k))
            d = agg.get("dynamics", {})
            for k, h in DYN_COLS:
                row[h] = _m3(d.get(k))
            row["e_hull_med_phys"] = _m(agg.get("e_hull_median_over_cells"))
            # Not a top-level key: the unphysical share lives inside the
            # e_hull_valid block.  agg.get(...) here returned None for every
            # row and _m3 rendered it as 0.0, i.e. an all-zero column that
            # reads as "no unphysical hull values anywhere" -- false on
            # Perov-5, where the share is 44-65% under every arm.
            row["frac_unphys"] = _m3(
                (agg.get("e_hull_valid") or {}).get("frac_unphysical"))
            row["nfe"] = _m3(d.get("nfe_mean"))
            rows.append(row)

    # Dataset-major, then config order as declared, then sampler -- matches the
    # task-table ordering so rows line up with how the fleet actually fills in.
    cfg_order = {c: i for i, c in enumerate(rp.PHASE2_SUBSET)}
    rows.sort(key=lambda r: (rp.DATASETS.index(r["dataset"]),
                             cfg_order.get(r["config"], 99),
                             0 if r["sampler"] == "ald" else 1))

    csv_path = out_dir / f"phase2_interim_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[interim] -> {csv_path}")

    headers = ["dataset", "config", "sampler", "n_tasks", "n_expected", "complete",
               "SUN", "valid", "stable", "novel", "unique", "AMSD", "match", "cov",
               "rKL", "OOD", "indOOD", "startOOD", "collapse", "capped",
               "e_hull_med_phys", "frac_unphys", "nfe"]
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    md_path = out_dir / f"phase2_interim_{args.tag}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[interim] -> {md_path}")

    n_done = sum(r["n_tasks"] for r in rows)
    n_full = sum(r["n_expected"] for r in rows)
    print(f"\n[interim] grid coverage: {n_done}/{n_full} tasks "
          f"({100 * n_done / n_full:.1f}%) across {len(rows)} rows")
    print("[interim] rows marked complete=NO are PARTIAL -- do not quote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
