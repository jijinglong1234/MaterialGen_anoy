"""How far down the noise schedule the truncated PF-ODE runs actually get.

Motivation (reviewer issue: budget truncation limits the sampler comparison).
The sigma scan reports PF-ODE's induced OOD rate at each sigma_max next to the
fraction of trajectories that exhaust the 1000-evaluation cap (92.7-98.9% at
sigma_max >= 1.5).  A capped trajectory is a *partially annealed* one: the
solver integrates in sigma from sigma_max down to sigma_min, and when the cap
is hit the RHS stops moving the state while sigma keeps decreasing, so the
terminal structure belongs to the noise level of the last real evaluation --
which the task files did not record.  Without it the capped OOD rate cannot be
read: 1.7% induced at sigma_max = 5 is a different claim if the trajectories
reached 2.0 A than if they reached 0.3 A.

This probe measures it.  It re-runs a subsample of the PF-ODE ladder
configurations through the production path (run_phase1.run_candidate, so the
initial conditions are the same make_initial draws from the same
seed/candidate indices), and records PFODEResult.sigma_final.

This is a *probe*, not a re-run: it covers a fraction of the cells with a
fraction of the candidates, and it exists to answer one question about the
truncation.  The scan itself is unchanged.

Usage:
    python scripts/probe_pfode_reached_sigma.py --device cuda:0
    python scripts/probe_pfode_reached_sigma.py --shard 1/4     # fleet-style

Output: results/phase1/analysis/pfode_reached_sigma.json (+ stdout table).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import run_phase1 as rp  # noqa: E402

OUT = os.path.join(_ROOT, "results", "phase1", "analysis",
                   "pfode_reached_sigma.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--arms", nargs="+", default=["bare", "l1l4"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--cands", type=int, nargs="+", default=[0])
    ap.add_argument("--comps", nargs="+", default=None,
                    help="default: all 20 MP-20 test compositions")
    ap.add_argument("--nnp", default="mace", choices=["mace", "esen"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None, help="k/N")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    comps = args.comps or list(rp.COMPOSITIONS_20)
    tasks = [(sig, arm, comp, seed, cand)
             for sig in args.sigmas for arm in args.arms
             for comp in comps for seed in args.seeds for cand in args.cands]
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % n == k]

    calc = rp.load_calculator(args.nnp, args.device)
    refs = rp.load_reference_structures(sorted({t[2] for t in tasks}))
    hull = rp.HullEvaluator(rp.load_hull_table())

    rows = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Resume: the CUDA fault that kills a shard takes the process, not the
    # output -- flush() leaves a valid JSON of every task finished so far
    # (shard0 lost 26 tasks to an "illegal memory access" inside MACE).  Re-
    # running from scratch would repay for trajectories already computed, so
    # skip the keys already on disk.
    done_keys = set()
    if os.path.exists(args.out):
        try:
            rows = list(json.load(open(args.out)))
            for r in rows:
                done_keys.add((r["sigma_max"], r["arm"], r["composition"],
                               r["seed"], r["cand"]))
            print(f"resuming: {len(done_keys)} task(s) already in {args.out}, "
                  f"{len(tasks) - len(done_keys)} to go", flush=True)
        except (ValueError, KeyError, TypeError):
            rows, done_keys = [], set()

    def flush():
        """Write after every task.

        The first run of this probe lost three of four shards to a kill at
        52-57 of 60 tasks: the results were held in memory and written once at
        the end, so an interruption near the finish discarded nearly all of the
        work.  Writing per task costs nothing next to a 50-100 s trajectory and
        makes a kill cost at most one task.  The file is valid JSON at every
        point, so the merge can read a shard that was still running when the
        previous one finished.
        """
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rows, fh, indent=1)
        os.replace(tmp, args.out)

    for i, (sig, arm, comp, seed, cand) in enumerate(tasks):
        if (sig, arm, comp, seed, cand) in done_keys:
            continue
        score = rp.make_score(calc, arm, None, label=rp.nnp_label(args.nnp))
        cfg = rp.make_pfode_config(sig, arm)
        sampler = rp.make_pfode_sampler(cfg)
        c = rp.run_candidate(score, sampler, refs[comp], sig, seed, cand, hull)
        rows.append({
            "sigma_max": sig, "arm": arm, "composition": comp, "seed": seed,
            "cand": cand, "nfe": c["nfe"], "cap": cfg.max_nfe,
            "converged": bool(c["nfe"] < cfg.max_nfe),
            "sigma_final": c["sigma_final"], "valid": c["valid"],
            "wall_time_s": round(c["wall_time_s"], 1),
        })
        print(f"[{i+1}/{len(tasks)}] s{sig:g} {arm} {comp} seed{seed} "
              f"nfe={c['nfe']} reached sigma={c['sigma_final']:.3f} "
              f"({c['wall_time_s']:.0f}s)", flush=True)
        flush()

    if not rows:
        return
    flush()

    print("\nper cell: n, share capped, sigma reached (median over all runs | "
          "median over capped runs only)")
    for sig in args.sigmas:
        for arm in args.arms:
            sel = [r for r in rows if r["sigma_max"] == sig and r["arm"] == arm]
            if not sel:
                continue
            reached = np.array([r["sigma_final"] for r in sel], float)
            capd = np.array([not r["converged"] for r in sel])
            med_cap = np.median(reached[capd]) if capd.any() else float("nan")
            print(f"  s{sig:<4g} {arm:5s} n={len(sel):3d} capped={capd.mean()*100:5.1f}%"
                  f"  reached: med {np.median(reached):.3f}  q10 {np.percentile(reached,10):.3f}"
                  f"  q90 {np.percentile(reached,90):.3f}"
                  f"  | capped-only med {med_cap:.3f}"
                  f"  wall {np.median([r['wall_time_s'] for r in sel]):.0f}s")
    print(f"\nwrote {os.path.relpath(args.out, _ROOT)}")


if __name__ == "__main__":
    main()
