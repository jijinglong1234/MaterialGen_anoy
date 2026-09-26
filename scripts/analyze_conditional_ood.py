"""Conditional failure rate P(indiced OOD | clean start) for the sigma scan.

Motivation (a reviewer minor issue).  The published induced rate is

    induced_k / n_total            (numerator over a clean-start subset,
                                    denominator over ALL trajectories)

so the curve mixes two changes as sigma grows: the per-trajectory risk of a
clean-start trajectory, and the shrinking share of the pool that is still at
risk.  At sigma = 5.0 the scan's start-OOD floor reaches 43.1% pooled, i.e.
almost half the denominator was never exposed.  The conditional rate

    induced_k / n_clean            (n_clean = n_total - n_start_ood)

holds the population fixed and is the quantity a per-trajectory failure
probability should be.

Both are computed here from the same trajectories, with the same OOD mask
(0b010011) and the same cells as the frozen pooled table, by importing the
loading path of make_percomp_sigma_tables, so the published column is exactly
reproducible and any difference is the denominator and nothing else.

Output: results/phase1/analysis/conditional_ood.json (+ stdout table).
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)          # analyze_phase1 imports run_phase1 -> materialgen

from make_percomp_sigma_tables import (  # noqa: E402
    BASE, COLS, OOD_MASK, SIGMAS, TEX, induced_k, load_cell, n_per_comp,
)
from analyze_phase1 import fit_sigmoid  # noqa: E402

OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "conditional_ood.json")


def start_ood_k(cands: list) -> int:
    """Trajectories whose step-0 state already violates the OOD mask."""
    return sum(bool(c["ood_bitmask"][0] & OOD_MASK)
               for c in cands if c["ood_bitmask"])


def cell_stats(sig: float, arm: str, smp: str) -> dict:
    comps = load_cell(sig, arm, smp)
    n_cell = n_per_comp(comps)
    n_total = n_clean = n_start = n_ind = 0
    for cands in comps.values():
        k_start = start_ood_k(cands)
        k_ind = induced_k(cands)
        n_total += len(cands)
        n_start += k_start
        n_ind += k_ind
        n_clean += len(cands) - k_start
    assert n_ind <= n_clean, f"{sig}/{arm}/{smp}: induced > clean-start"
    return {
        "sigma": sig, "arm": arm, "sampler": smp, "n_total": n_total,
        "n_start_ood": n_start, "n_clean": n_clean, "n_induced": n_ind,
        "start_ood_rate": n_start / n_total,
        "published_induced_rate": n_ind / n_total,
        "conditional_induced_rate": n_ind / n_clean if n_clean else None,
    }


def main() -> None:
    rows = [cell_stats(sig, arm, smp) for sig in SIGMAS for arm, smp in COLS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"mask": bin(OOD_MASK), "cells": rows}, fh, indent=1)

    print(f"{'sigma':>5}  {'arm':6} {'smp':6} {'n_clean':>7} {'startOOD':>9} "
          f"{'published':>10} {'conditional':>12}")
    for r in rows:
        cond = r["conditional_induced_rate"]
        print(f"{r['sigma']:5.1f}  {r['arm']:6} {r['sampler']:6} "
              f"{r['n_clean']:7d} {r['start_ood_rate']*100:8.1f}% "
              f"{r['published_induced_rate']*100:9.2f}% "
              f"{cond*100:11.2f}%" if cond is not None
              else f"{r['sigma']:5.1f}  {r['arm']:6} {r['sampler']:6} "
                   f"{r['n_clean']:7d} {r['start_ood_rate']*100:8.1f}% "
                   f"{r['published_induced_rate']*100:9.2f}%          n/a")

    pooled = {}
    for arm, smp in COLS:
        sel = [r for r in rows if (r["arm"], r["sampler"]) == (arm, smp)]
        pub = sum(r["n_induced"] for r in sel) / sum(r["n_total"] for r in sel)
        con = sum(r["n_induced"] for r in sel) / sum(r["n_clean"] for r in sel)
        pooled[f"{arm}|{smp}"] = {"published": pub, "conditional": con,
                                  "n_total": sum(r["n_total"] for r in sel),
                                  "n_clean": sum(r["n_clean"] for r in sel)}
    print("\npooled over the ten noise levels:")
    for k, v in pooled.items():
        print(f"  {k:12s} published {v['published']*100:6.2f}%   "
              f"conditional {v['conditional']*100:6.2f}%   "
              f"(n_clean {v['n_clean']}/{v['n_total']})")

    # Paired sigmoid fits.  fit_sigmoid's amplitude form handles a curve that
    # saturates below 0.5, which both conventions do for every arm here, and
    # the two fits differ only in the denominator of the ordinate -- so the
    # paired sigma_c / amplitude split the convention effect from the physics.
    print("\nsigmoid fits, same estimator on both curves:")
    print(f"  {'arm|smp':12s} {'sigma_c':>9} {'tau':>7} {'amp':>7}   |"
          f" {'sigma_c':>9} {'tau':>7} {'amp':>7}")
    fits = {}
    for arm, smp in COLS:
        sel = {r["sigma"]: r for r in rows
               if (r["arm"], r["sampler"]) == (arm, smp)}
        pub = [sel[s]["published_induced_rate"] for s in SIGMAS]
        con = [sel[s]["conditional_induced_rate"] for s in SIGMAS]
        fp, fc = fit_sigmoid(SIGMAS, pub), fit_sigmoid(SIGMAS, con)
        fits[f"{arm}|{smp}"] = {"published": fp, "conditional": fc}
        def cells(f):
            return (f"{f['sigma_c']:9.3f} {f['tau']:7.3f} {f['amplitude']:7.3f}"
                    if f["fit_ok"] else f"{'no fit':>25}")
        print(f"  {arm + '|' + smp:12s} {cells(fp)}   |{cells(fc)}")
    with open(OUT, "w") as fh:
        json.dump({"mask": bin(OOD_MASK), "cells": rows, "pooled": pooled,
                   "sigmoid": fits}, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, _ROOT)}")


if __name__ == "__main__":
    main()
