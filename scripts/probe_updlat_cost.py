#!/usr/bin/env python
"""Price the stress-driven lattice update (Phase-2 budget input, plan step 4).

Phase 2 turns on ``update_lattice`` for five of its six configurations, so every
one of their ALD steps gains a stress evaluation.  The Phase-1 fleet measured
its costs with the cell frozen and cannot price that, and the preflight runs
that *did* use upd_lat never persisted ``wall_time_s`` -- so the budget script's
upd_lat factor was an estimate.  This probe measures it instead.

Method: run the same cell with the same config twice, once with
``update_lattice=True`` (C1) and once with it forced off (C5), on the same
device in the same process, and compare the per-candidate ``wall_time_s`` the
runner itself records.  C1 and C5 differ in exactly that one field
(``run_phase2.CONFIGS``), so the ratio is the lattice channel's cost and
nothing else.

Cells are the ones the upd_lat failure mode actually lives on (the preflight's
SrTiO3/ZnS/FeNi3), plus the two extreme cell sizes for a size trend, because
the factor is a *ratio* of two runs at the same size and only becomes a
multiplier in the budget if it holds across sizes.

Output: results/phase2/summary/phase2_updlat_cost.json
Usage:  python scripts/probe_updlat_cost.py [--device cuda:3] [--n-cand 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_phase1 as rp                                            # noqa: E402
import run_phase2 as r2                                            # noqa: E402

OUT = REPO / "results" / "phase2" / "summary" / "phase2_updlat_cost.json"
CELLS = ["SrTiO3", "ZnS", "FeNi3", "AlCu", "MoS2"]      # 2 atoms .. 16 atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnp", default="mace", choices=["mace", "esen"])
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--n-cand", type=int, default=3)
    ap.add_argument("--dataset", default="mp_20")
    args = ap.parse_args()

    r2.N_CAND = args.n_cand          # run_task reads the module-level grid knob
    ctx = r2.context(args.dataset, args.nnp, need_index=False)
    calc = rp.load_calculator(args.nnp, args.device)

    rows = []
    for cell_id in CELLS:
        cell = next((c for c in ctx.cells if c.id == cell_id), None)
        if cell is None:
            print(f"skip {cell_id}: not among {args.dataset} cells")
            continue
        for cfg_name in ("C1", "C5"):        # upd_lat True / False, else equal
            cfg = r2.CONFIGS[cfg_name]
            task = {"dataset": args.dataset, "config": cfg_name, "sampler": "ald",
                    "cell": cell_id, "seed": 42, "nnp": args.nnp}
            payload = r2.run_task(task, ctx, calc)
            w = np.array([c["wall_time_s"] for c in payload["candidates"]])
            n_at = payload["n_atoms"]
            rows.append({"cell": cell_id, "config": cfg_name, "n_atoms": n_at,
                         "upd_lat": cfg.update_lattice, "n_cand": args.n_cand,
                         "wall_med": float(np.median(w)),
                         "wall_mean": float(w.mean())})
            print(f"{cell_id:8s} {cfg_name} upd_lat={str(cfg.update_lattice):5s} "
                  f"n_atoms={n_at:3d}  med {np.median(w):7.2f}s  mean {w.mean():7.2f}s",
                  flush=True)

    ratios = []
    print("\nupd_lat cost ratio (C1 / C5), same cell same process:")
    for cell_id in CELLS:
        a = next((r for r in rows if r["cell"] == cell_id and r["config"] == "C1"), None)
        b = next((r for r in rows if r["cell"] == cell_id and r["config"] == "C5"), None)
        if not a or not b:
            continue
        ratio = a["wall_med"] / b["wall_med"]
        ratios.append(ratio)
        print(f"  {cell_id:8s} {a['wall_med']:7.2f} / {b['wall_med']:7.2f} = {ratio:5.3f}"
              f"   (s/atom {a['wall_med']/a['n_atoms']:.2f} vs {b['wall_med']/b['n_atoms']:.2f})")

    factor = float(np.median(ratios)) if ratios else None
    print(f"\nmedian factor = {factor:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"nnp": args.nnp, "dataset": args.dataset, "n_cand": args.n_cand,
         "rows": rows, "ratios": {r["cell"]: float(x) for r, x in zip(rows[::2], ratios)},
         "factor_median": factor}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
