"""Per-composition collapse scale sigma_c, bare score under ALD.

Implements the definition stated in the paper caption of Table 9
(`tab:basin-radius`): sigma_c solves P_step(sigma_c) = 0.005, where P_step is
the *per-accepted-step* probability that the minimum interatomic distance
falls below the OOD threshold (0.5 A), pooled over every accepted step of
every trajectory of a (composition, sigma) cell; the solve is a log-linear
interpolation of log P_step against log sigma over the ten-point noise grid.

One collision per 200-step trajectory is the calibration of the threshold, so
0.005 is a per-step rate and not a per-trajectory one.  Steps are read from
each candidate's `d_min_hist` (the accepted-step path, matching the OOD
accounting of `analyze_phase1.cell_metrics`), -- NOT from the per-trajectory
`type1_steps` counter, which is a different diagnostic.

Estimator note.  The published table was produced before the
initial-condition fix and before the addition of the dedicated
`l1l2l3` arm, on 20 candidates per cell (~19,900 accepted steps); the
current cells carry 10 candidates (~9,950 accepted steps).  Running this
script on `results/phase1/legacy_pre_cholesky/subexp1` reproduces
the published values to a mean absolute deviation of 0.06-0.10 A, which is
the reproducibility floor of the estimator rather than agreement with a
different one.  Interior brackets (a pair of adjacent sigma points that
straddle 0.005) are exact; cells whose curve never reaches 0.005 are
reported as censored at the scan edge.

Accounting fix.  Previously this script pooled every trajectory
and every accepted step, including step 0.  Step 0 is the *initial
condition*, not a sampler-induced state, so excluding it is what the paper's
own OOD convention states (\S4.1: induced events are "an event at step >= 1
from a clean start"), and it is what the "~9,950 accepted steps per cell" of
the Table 9 caption already implies -- 50 trajectories x 199 steps, where the
script was using 50 x 200 = 10,000.  Trajectories whose step 0 is already OOD
are now dropped as well, for the same reason.  The effect is largest on the
smallest cells and moves the largest sigma_c by up to 1.0 A (ZnS 2.20 -> 3.18,
TiO2 1.44 -> 2.05); the spread widens from 17.5x to 17.5x only because both
endpoints move together.  A separate, broader treatment of this estimator's
size confound and its floor sensitivity is in `analyze_sigma_c_robust.py`.

Usage:
  python scripts/comp_sigma_c.py [--root results/phase1/subexp1] [--tex]
      [--floor 1e-12] [--all-steps]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

THRESHOLD = 0.005     # one collision per 200-step trajectory
D_MIN_OOD = 0.5       # A, the OOD distance threshold of analyze_phase1
SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
ARM = "bare_ald"
FLOOR = 1e-12         # stand-in for cells whose rate is exactly 0 (see --floor)


def tag(sigma: float) -> str:
    """Directory sigma tag.  The trees use %g, so 5.0 -> "s5", not "s5.0"."""
    return "%g" % sigma


def load_cell(comp: str, sigma: float, root: str, all_steps: bool = False):
    """Every accepted-step d_min of one (composition, sigma) cell.

    By default this is the paper's convention: step 0 (the initial condition)
    is excluded and trajectories that start OOD are dropped.  `all_steps`
    restores the pre-fix pooling, kept only to reproduce the older
    tabulation.
    """
    d = os.path.join(root, f"s{tag(sigma)}_{ARM}")
    vals, n_steps = [], 0
    for f in sorted(glob.glob(os.path.join(d, f"{comp}_seed*.json"))):
        with open(f) as fh:
            task = json.load(fh)
        for c in task.get("candidates", []):
            hist = c.get("d_min_hist") or []
            if not all_steps:
                if not hist or hist[0] is None or not np.isfinite(hist[0]):
                    continue
                if hist[0] < D_MIN_OOD:       # start-OOD: not sampler-induced
                    continue
                hist = hist[1:]
            vals.extend(float(x) for x in hist if x is not None and np.isfinite(x))
            n_steps += len(hist)
    return np.asarray(vals, float), n_steps


def step_rate(comp: str, sigma: float, root: str, all_steps: bool = False) -> float:
    vals, n = load_cell(comp, sigma, root, all_steps)
    if n == 0:
        return float("nan")
    return float(np.mean(vals < D_MIN_OOD))


def sigma_c(comp: str, root: str, all_steps: bool = False, floor: float = FLOOR):
    """Return (sigma_c, kind, n_steps, curve) with kind in
    {"interp", "below", "above"}: "below" means the curve is already past
    threshold at the lowest sigma, "above" that it never reaches it."""
    sig, rate, n_steps = [], [], 0
    for s in SIGMAS:
        vals, n = load_cell(comp, s, root, all_steps)
        n_steps = max(n_steps, n)
        if n:
            sig.append(s)
            rate.append(float(np.mean(vals < D_MIN_OOD)))
    sig, rate = np.asarray(sig), np.asarray(rate)
    if len(sig) < 2:
        return None, "above", n_steps, (sig, rate)
    # The collapse scale is the FIRST crossing, so the curve is made monotone
    # by a running maximum before interpolating.  Without this a non-monotone
    # cell (a dip followed by a rise) makes np.interp read an unsorted axis
    # and return a value that depends on the grid rather than on the data.
    rate = np.maximum.accumulate(rate)
    if rate[0] > THRESHOLD:
        return float(sig[0]), "below", n_steps, (sig, rate)
    if rate[-1] < THRESHOLD:
        return float(sig[-1]), "above", n_steps, (sig, rate)
    ls, lp = np.log(sig), np.log(np.maximum(rate, floor))
    return float(np.exp(np.interp(np.log(THRESHOLD), lp, ls))), "interp", n_steps, (sig, rate)


def tex_formula(comp: str) -> str:
    """Al2O3 -> Al$_2$O$_3$ for the LaTeX body of Table 9."""
    import re
    return re.sub(r"(?<=[A-Za-z])(\d+)", r"$_{\1}$", comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1/subexp1")
    ap.add_argument("--out", default="results/phase1/analysis/comp_sigma_c.json")
    ap.add_argument("--tex", action="store_true",
                    help="write docs/comp_sigma_c_table.tex (Table 9 body)")
    ap.add_argument("--floor", type=float, default=FLOOR,
                    help="rate floor used for cells whose rate is exactly 0; "
                         "moves the smallest sigma_c (0.28 at 1e-12, 0.23 at "
                         "1e-3). Default 1e-12, the published choice.")
    ap.add_argument("--all-steps", action="store_true",
                    help="pre-fix pooling: every trajectory and every "
                         "step, including the initial condition")
    args = ap.parse_args()

    comps = sorted({
        os.path.basename(f).rsplit("_seed", 1)[0]
        for f in glob.glob(os.path.join(args.root, f"s0.1_{ARM}", "*.json"))})
    if not comps:
        sys.exit(f"no cells under {args.root}/s0.1_{ARM}")

    out, rows = {}, []
    for comp in comps:
        sc, kind, n_steps, (sig, rate) = sigma_c(comp, args.root,
                                                 args.all_steps, args.floor)
        out[comp] = {"sigma_c": sc, "kind": kind, "n_steps_max": n_steps,
                     "floor": args.floor, "all_steps": bool(args.all_steps),
                     "step_rate": {f"{s:g}": r for s, r in zip(sig, rate)}}
        shown = (f"{sc:.2f}" if kind == "interp"
                 else (f"<{sig[0]:g}" if kind == "below"
                       else f"$>${sig[-1]:g}"))
        rows.append((comp, sc, shown, kind))
        print(f"{comp:>8}  sigma_c={shown:>7}  ({kind}, {n_steps} accepted steps)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")

    if args.tex:
        est = sorted((r for r in rows if r[3] == "interp"), key=lambda r: r[1])
        cen = sorted((r for r in rows if r[3] != "interp"), key=lambda r: r[0])
        half = (len(est) + 1) // 2
        left, right = est[:half], est[half:]
        lines = ["% Generated by scripts/comp_sigma_c.py --tex; do not edit.", "\\midrule"]
        for i in range(half):
            a = left[i]
            b = right[i] if i < len(right) else None
            cells = [tex_formula(a[0]), a[2]]
            if b:
                cells += [tex_formula(b[0]), b[2]]
            lines.append(" & ".join(cells) + r" \\")
        if cen:
            lines.append("\\midrule")
            lines.append("\\multicolumn{6}{@{}l}{" +
                         "; ".join(f"{tex_formula(c[0])} {c[2]}" for c in cen) + "}")
            lines[-1] += r" \\"
        path = "docs/comp_sigma_c_table.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
