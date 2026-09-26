"""Regenerate the pooled sigma-failure table (paper Table tab:sigma-failure).

Pools results/phase1/subexp1 over the 20 MP-20 test compositions and the 5
seeds of the post-fix re-run, using the frozen
definition (OOD mask 0b010011; induced = start-clean trajectories with any
OOD event at steps >= 1, denominator = ALL trajectories).  The metric is
computed by analyze_phase1.cell_metrics so reader and paper cannot drift.

Arm mapping: the ALD protected column is now the dedicated
L1--L3 arm (run_phase1.SIGMA_SCAN_ALD_ONLY = ["l1l2l3"]); the PF-ODE
protected column stays on the l1l4 code name, which on the ODE path IS
L1 + L3 (PFODEConfig carries no L2 density noise and no L4 monitor --
see run_phase1.make_pfode_config, where make_pfode_config(s,"l1l2l3") ==
make_pfode_config(s,"l1l4") field by field).

Usage:  python scripts/regen_sigma_pooled.py [--tex] [--allow-partial]
  --tex  emit the LaTeX body of tab:sigma-failure for splicing into the paper
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_phase1 as ap                                       # noqa: E402
import make_percomp_sigma_tables as mp                            # noqa: E402

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
# (column label, directory arm name, sampler, group)
COLS = [
    ("Bare ALD",   "bare",   "ald"),
    ("Bare ODE",   "bare",   "pfode"),
    ("L1 ALD",     "l1",     "ald"),
    ("L1 ODE",     "l1",     "pfode"),
    ("L1-L3 ALD",  "l1l2l3", "ald"),
    ("L1+L3 ODE",  "l1l4",   "pfode"),
]
N_FILES_PER_CELL = 20 * 5          # 20 compositions x 5 seeds


def load_cell(sig, arm, smp, allow_partial):
    d = os.path.join("results", "phase1", "subexp1", f"s{sig:g}_{arm}_{smp}")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if len(files) != N_FILES_PER_CELL and not allow_partial:
        raise SystemExit(
            f"cell s{sig:g}_{arm}_{smp}: {len(files)}/{N_FILES_PER_CELL} task "
            f"files -- re-run incomplete.  Pass --allow-partial to compute "
            f"anyway (values then rest on a short denominator).")
    cands, comps_seen = [], {}
    for p in files:
        j = json.load(open(p))
        if j.get("init_cell_gen") != ap.INIT_CELL_GEN:
            continue
        c = ap.cands_of(j)
        cands.extend(c)
        comps_seen.setdefault(j["composition"], 0)
        comps_seen[j["composition"]] += len(c)
    return cands, len(files), comps_seen


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tex", action="store_true")
    ap_.add_argument("--allow-partial", action="store_true")
    args = ap_.parse_args()

    cells, short = {}, []
    for sig in SIGMAS:
        for name, arm, smp in COLS:
            cands, nf, per_comp = load_cell(sig, arm, smp, args.allow_partial)
            m = ap.cell_metrics(cands)
            cells[(sig, name)] = m
            if nf != N_FILES_PER_CELL:
                short.append((sig, name, nf, m["n_traj"]))

    print(f"{'sigma':>5} " + " ".join(f"{n:>12}" for n, _, _ in COLS))
    print("  induced OOD rate (%), denominator = all trajectories")
    for sig in SIGMAS:
        print(f"{sig:>5} " + " ".join(
            f"{cells[(sig, n)]['induced_ood_rate'] * 100:>12.2f}"
            for n, _, _ in COLS))

    print("\n  start-OOD floor (%)")
    for sig in SIGMAS:
        print(f"{sig:>5} " + " ".join(
            f"{cells[(sig, n)]['start_ood_rate'] * 100:>12.2f}"
            for n, _, _ in COLS))

    print("\n  terminal validity (%)")
    for sig in SIGMAS:
        print(f"{sig:>5} " + " ".join(
            f"{cells[(sig, n)]['validity'] * 100:>12.2f}"
            for n, _, _ in COLS))

    print("\n  E_hull median of valid structures (meV/atom)")
    for sig in SIGMAS:
        print(f"{sig:>5} " + " ".join(
            f"{cells[(sig, n)]['e_hull_med_meV']:>12.1f}"
            for n, _, _ in COLS))

    print("\n  n_traj per cell")
    for sig in SIGMAS:
        print(f"{sig:>5} " + " ".join(
            f"{cells[(sig, n)]['n_traj']:>12}" for n, _, _ in COLS))

    print("\nper-arm ranges over the scan:")
    for n, _, _ in COLS:
        ind = [cells[(s, n)]["induced_ood_rate"] * 100 for s in SIGMAS]
        val = [cells[(s, n)]["validity"] * 100 for s in SIGMAS]
        start = [cells[(s, n)]["start_ood_rate"] * 100 for s in SIGMAS]
        eh = [cells[(s, n)]["e_hull_med_meV"] for s in SIGMAS]
        print(f"  {n:>10}: induced {min(ind):.2f}-{max(ind):.2f}  "
              f"validity {min(val):.1f}-{max(val):.1f}  "
              f"start-OOD {min(start):.1f}-{max(start):.1f}  "
              f"E_hull {min(eh):.0f}-{max(eh):.0f} meV/atom")

    # fit_sigmoid needs a curve that crosses 0.5, so it is applied to the
    # pooled OOD rate and to validity -- exactly as analyze_subexp1 does --
    # never to the induced rate (which saturates well below 0.5 for the
    # protected arms and is then not identifiable by construction).
    print("\nsigmoid fits (G1 statistical definition, pooled OOD rate):")
    for n, _, _ in COLS:
        y = [cells[(s, n)]["ood_rate"] for s in SIGMAS]
        fit = ap.fit_sigmoid(SIGMAS, y)
        print(f"  {n:>10}: sigma_c={fit['sigma_c']} tau={fit['tau']} "
              f"amp={fit['amplitude']} fit_ok={fit['fit_ok']} "
              f"max_ood_rate={max(y):.3f}")
    print("\nsigmoid fits (validity):")
    for n, _, _ in COLS:
        y = [cells[(s, n)]["validity"] for s in SIGMAS]
        fit = ap.fit_sigmoid(SIGMAS, y)
        print(f"  {n:>10}: sigma_c={fit['sigma_c']} tau={fit['tau']} "
              f"amp={fit['amplitude']} fit_ok={fit['fit_ok']}")

    if short:
        print("\nWARNING -- incomplete cells (values rest on a short denominator):")
        for sig, n, nf, ntr in short:
            print(f"  s{sig:g} {n}: {nf}/100 files, {ntr} trajectories")
        print("Re-run this script after the fleet finishes (plan item 4).")

    if args.tex:
        print("\n% ---- tab:sigma-failure body ----")
        for sig in SIGMAS:
            vals = " & ".join(f"{cells[(sig, n)]['induced_ood_rate'] * 100:.2f}"
                              for n, _, _ in COLS)
            print(f"    {sig:g} & {vals} \\\\")
        print("\n% terminal validity ranges:")
        for n, _, _ in COLS:
            val = [cells[(s, n)]["validity"] * 100 for s in SIGMAS]
            print(f"%   {n:>10}: {min(val):.1f}--{max(val):.1f}")


if __name__ == "__main__":
    main()
