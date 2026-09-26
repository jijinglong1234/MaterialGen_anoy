"""Threshold-free arm ranking for the sigma scan.

Motivation (reviewer issue 7).  Every headline metric of the sigma scan --
induced OOD, terminal validity -- is defined by a distance threshold
(d_min < 0.5 A, plus the force and volume bits of OOD_MASK 0b010011).  Layer 3
enforces that same distance by construction, so part of its win on those
metrics is the definition returning.  The reviewer asks for evidence that does
not share the definition.

Statistics of the sampled distribution, per (sigma, arm, sampler) cell, pooled
over all trajectories of the cell:

  dmin_med      median of the pooled per-step d_min distribution
  dmin_min_med  median over trajectories of min_t d_min
  fmax_med      median over final structures of max |F_NNP|
  fmax_p90      90th percentile of the same, kept because the unprotected
                force distribution is heavy-tailed and a median alone would
                hide the tail that the OOD mask's force bit reacts to

None of the four is a threshold, and none can be satisfied by a constraint
recording its own satisfaction -- with one caveat that the reporting keeps
visible: dmin_min_med is a lower-tail distance, and L3's whole action is to
raise it, so for L3 arms that statistic is partly construction.  dmin_med and
the two force statistics are not: a reflection at r_min = 0.5 A does not move
the median nearest-neighbour distance, nor the forces of the final structure,
unless the sampled distribution itself has changed.

What is compared is the arm *ordering*, since that is what the paper claims
(L1 and L1-L3 beat bare).  Per sigma the six columns are ranked by each
statistic and by the published induced rate; concordant pairs are counted.
Cells whose induced rates are exactly tied (every arm at 0.00%, i.e. the two
lowest noise levels) carry no ordering and are excluded rather than scored by
an alphabetical accident.

Output: results/phase1/analysis/threshold_free.json (+ stdout table).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

from make_percomp_sigma_tables import COLS, SIGMAS, induced_k, load_cell  # noqa: E402

OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "threshold_free.json")

# key -> True if larger is better (distance statistics) / lower is better (force)
STATS = [("dmin_med", True), ("dmin_min_med", True),
         ("fmax_med", False), ("fmax_p90", False)]


def label(arm: str, smp: str) -> str:
    """Unambiguous column name: arm[:2] collides (l1 vs l1l4 on ALD)."""
    return {"bare": "Bare", "l1": "L1", "l1l4": "L1-3"}[arm] + \
        ("-ALD" if smp == "ald" else "-ODE")


def cell_stats(sig: float, arm: str, smp: str) -> dict:
    """Pooled d_min / |F| statistics for one cell; no threshold anywhere."""
    comps = load_cell(sig, arm, smp)
    steps: list[float] = []          # every per-step d_min of every trajectory
    traj_min: list[float] = []       # each trajectory's closest approach
    fmax: list[float] = []           # each final structure's max |F_NNP|
    n_ind = 0
    for cands in comps.values():
        for c in cands:
            h = c["d_min_hist"]
            if not h:
                continue
            steps.extend(h)
            traj_min.append(min(h))
            f = c["final"].get("max_f")
            if f is not None and np.isfinite(f):
                fmax.append(float(f))
        n_ind += induced_k(cands)
    s = np.asarray(steps, float)
    return {
        "sigma": sig, "arm": arm, "sampler": smp, "col": label(arm, smp),
        "n_traj": len(traj_min), "n_steps": int(s.size),
        "dmin_med": float(np.median(s)),
        "dmin_min_med": float(np.median(traj_min)),
        "fmax_med": float(np.median(fmax)) if fmax else None,
        "fmax_p90": float(np.percentile(fmax, 90)) if fmax else None,
        "induced_rate": n_ind / len(traj_min) if traj_min else None,
    }


def main() -> None:
    rows = [cell_stats(sig, arm, smp) for sig in SIGMAS for arm, smp in COLS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def fmt(r, key):
        if key.startswith("dmin"):
            return f"{r[key]:.3f}"
        if key == "induced_rate":
            return f"{r[key]*100:.1f}%"
        return f"{r[key]:.1f}"

    print("per cell, best arm first (distances: larger is better; forces and "
          "induced rate: smaller is better)")
    agree = {k: [0, 0] for k, _ in STATS}
    per_sigma = {}
    for sig in SIGMAS:
        sel = [r for r in rows if r["sigma"] == sig]
        ind = sorted(sel, key=lambda r: r["induced_rate"])
        tied = len({round(r["induced_rate"], 6) for r in sel}) == 1
        print(f"\nsigma = {sig:g}" + ("   [all induced rates tied: no ordering]"
                                      if tied else ""))
        print("  induced_rate  " + "  ".join(f"{r['col']}:{fmt(r,'induced_rate')}"
                                             for r in ind))
        for key, higher_better in STATS:
            srt = sorted([r for r in sel if r[key] is not None],
                         key=lambda r: -r[key] if higher_better else r[key])
            print(f"  {key:13s} " + "  ".join(f"{r['col']}:{fmt(r,key)}" for r in srt))
            if tied:
                continue
            pos = {r["col"]: i for i, r in enumerate(srt)}
            val = {r["col"]: r[key] for r in srt}
            ind_order = [r["col"] for r in ind]
            # Within-sampler pairs are the paper's claim (L1 and L1-L3 beat
            # bare on a given sampler); cross-sampler pairs are a different
            # question -- whether ALD or PF-ODE samples better geometry -- and
            # are scored separately below so the two cannot be conflated.
            pairs = conc = wpairs = wconc = 0
            for i in range(len(ind_order)):
                for j in range(i + 1, len(ind_order)):
                    a, b = ind_order[i], ind_order[j]
                    # A tie in the statistic carries no ordering: skipping it
                    # is what keeps "the medians are all equal" from reading as
                    # disagreement with the induced-rate order.
                    if abs(val[a] - val[b]) <= 1e-9 * max(1.0, abs(val[a])):
                        continue
                    pairs += 1
                    conc += pos[a] < pos[b]
                    if a[-3:] == b[-3:]:        # same -ALD / -ODE suffix
                        wpairs += 1
                        wconc += pos[a] < pos[b]
            per_sigma.setdefault(f"{sig:g}", {})[key] = {
                "concordant": conc, "pairs": pairs,
                "within_sampler": {"concordant": wconc, "pairs": wpairs}}
            agree[key][0] += wconc
            agree[key][1] += wpairs

    print("\nordering agreement with the induced-rate order, same sampler only "
          "(the paper's claim), over the cells whose induced rates are not all "
          "tied (sigma >= 0.2) and the pairs the statistic does not tie:")
    for key, _hb in STATS:
        c, p = agree[key]
        print(f"  {key:13s} {c:3d}/{p} = {c/p*100:5.1f}%" if p
              else f"  {key:13s} n/a")
    print("  (a statistic that merely re-derived the OOD definition would score "
          "100%; independence shows up as a high but not perfect score)")
    cross = {k: [0, 0] for k, _ in STATS}
    for sig, d in per_sigma.items():
        del sig
        for k, v in d.items():
            cross[k][0] += v["concordant"] - v["within_sampler"]["concordant"]
            cross[k][1] += v["pairs"] - v["within_sampler"]["pairs"]
    print("\nordering agreement across samplers -- a different question, "
          "(whether the ODE path samples better geometry than ALD):")
    for key, _hb in STATS:
        c, p = cross[key]
        print(f"  {key:13s} {c:3d}/{p} = {c/p*100:5.1f}%" if p
              else f"  {key:13s} n/a")

    with open(OUT, "w") as fh:
        json.dump({"cells": rows, "per_sigma": per_sigma,
                   "agreement": {k: {"concordant": v[0], "pairs": v[1]}
                                 for k, v in agree.items()}}, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, _ROOT)}")


if __name__ == "__main__":
    main()
