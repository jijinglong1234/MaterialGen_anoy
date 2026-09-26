"""Step-size control for sub-exp 1: is the 200-NFE failure the score, or the grid?

Sub-exp 1 runs ALD at K=100 noise levels x M=2 steps = 200 NFE.  A reviewer's
first objection to "the bare score drives structures below 0.5 A" is that the
discretization is simply too coarse: at alpha_0 = 1e-3 and 200 steps, one step
can cross the whole OOD basin, so the sampler could be *jumping over* a good
score's own minimum.  `subexp1_fine` answers that directly: same sigma
schedule, same SDE time, 4x finer steps (M=8, alpha_0/4, 800 NFE).  Refining
the integrator at fixed SDE time converges to the same continuous-time law, so

  * the induced rate is *unchanged* within CI  -> the failure is in the score;
  * the induced rate *falls* materially       -> 200 NFE was too coarse, and
    the paper's absolute rates are an artifact of the discretization.

The two arms are compared at matched sigma (the fine tree covers a subset of
the grid), per-trajectory, using the paper's own convention: a trajectory is
induced-OOD if it starts clean and has an OOD event at step >= 1, normalized by
ALL trajectories.  That is `analyze_subexp1.agg`, imported rather than
reimplemented so the two cannot drift.

Uncertainty is a cluster bootstrap over *trajectories* (the independent unit;
the 10 candidates of one (composition, seed) share an initial cell).  A cell
that has no files yet is reported as pending instead of crashing, so this
doubles as the progress monitor for the fleet run.

Usage:
  python scripts/analyze_step_control.py [--boot 400] [--out .../step_control.json]
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from analyze_subexp1 import agg, _load_cell, SIGMAS  # noqa: E402

STD_BASE = os.path.join(_ROOT, "results", "phase1", "subexp1")
FINE_BASE = os.path.join(_ROOT, "results", "phase1", "subexp1_fine")
FINE_PREFIX = "m4_"
# run_phase1.FINE_ARMS, duplicated here on purpose: a missing sigma in the fine
# tree is a *task that has not landed*, which this script must report rather
# than silently average over.
FINE_ARMS = [("bare", [0.3, 0.5, 1.0, 2.0, 5.0]),
             ("l1l2l3", [0.5, 1.0, 2.0])]
RNG = np.random.default_rng(0)


def cell_cands(prefix, sigma, prot):
    """Candidates of one cell, or [] if the tree does not have it yet."""
    tag = f"{prefix}s{sigma:g}_{prot}_ald"
    base = FINE_BASE if prefix else STD_BASE
    if not os.path.isdir(os.path.join(base, tag)):
        return []
    return _load_cell(tag, base)


def boot_delta(a, b, n):
    """Cluster bootstrap of agg(a).induced - agg(b).induced.

    Trajectories are resampled with replacement inside each cell
    independently, which is what the 400-replicate CIs elsewhere in this
    analysis do (`analyze_sigma_c_robust.py`).
    """
    if not a or not b:
        return np.nan, np.nan, np.nan
    d = np.empty(n)
    for i in range(n):
        ra = [a[j] for j in RNG.integers(0, len(a), len(a))]
        rb = [b[j] for j in RNG.integers(0, len(b), len(b))]
        d[i] = agg(ra)["induced"] - agg(rb)["induced"]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float((d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "results", "phase1", "analysis", "step_control.json"))
    args = ap.parse_args()

    rows, pending = [], 0
    print(f"{'arm':>7} {'sigma':>6} {'n_std':>6} {'n_fine':>7} "
          f"{'induced std':>12} {'induced fine':>13} {'delta [95% CI]':>26}")
    for prot, sigmas in FINE_ARMS:
        for sigma in sigmas:
            std = cell_cands("", sigma, prot)
            fine = cell_cands(FINE_PREFIX, sigma, prot)
            if not std or not fine:
                missing = "std" if not std else "fine"
                print(f"{prot:>7} {sigma:>6g} {len(std):>6} {len(fine):>7}   "
                      f"pending ({missing})")
                pending += 1
                continue
            i_std, i_fine = agg(std)["induced"], agg(fine)["induced"]
            lo, hi, p_gt = boot_delta(fine, std, args.boot)
            rows.append(dict(prot=prot, sigma=sigma, n_std=len(std),
                             n_fine=len(fine), induced_std=i_std,
                             induced_fine=i_fine, delta=i_fine - i_std,
                             ci=[lo, hi], p_delta_gt0=p_gt))
            print(f"{prot:>7} {sigma:>6g} {len(std):>6} {len(fine):>7} "
                  f"{i_std:>11.1%} {i_fine:>12.1%} "
                  f"{i_fine - i_std:>+8.1%} [{lo:+.1%},{hi:+.1%}]")

    if not rows:
        sys.exit(f"\nno matched cells yet ({pending} pending) -- "
                 f"the fine fleet has not landed")

    # The verdict is per arm, over the sigmas where both step sizes exist: a
    # consistent sign outside CI across the whole matched range is what makes
    # the "score, not grid" reading safe.
    print()
    for prot, _ in FINE_ARMS:
        rs = [r for r in rows if r["prot"] == prot]
        if not rs:
            continue
        outside = [r for r in rs if r["ci"][0] > 0 or r["ci"][1] < 0]
        sign = {np.sign(r["delta"]) for r in rs}
        verdict = ("identical within CI -- the failure is in the score, not the "
                   "discretization" if not outside else
                   "DIFFERS -- 200 NFE is too coarse at "
                   + ", ".join(f"{r['sigma']:g}" for r in outside))
        print(f"{prot:>7}: {len(rs)} matched sigma(s), {len(outside)} with a "
              f"CI excluding 0, delta signs {sorted(sign)} -> {verdict}")

    with open(args.out, "w") as fh:
        json.dump(dict(n_boot=args.boot, grid=[r["sigma"] for r in rows],
                       rows=rows, pending=pending), fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
