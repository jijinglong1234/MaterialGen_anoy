"""Which PF-ODE trajectories carry the OOD events, and what they cost (D-4).

Reviewer issue: "budget truncation limits the sampler comparison -- the lower
collision rate of PF-ODE at high noise should be read together with the noise
level actually reached, the completed paths, and the runtime."

Three readings, all from the frozen scan tree plus the reached-sigma probe:

1. SPLIT.  Every PF-ODE cell, split by whether the run exhausted the 1000-
   evaluation cap (nfe >= cap, the paper's "Cap" column).  For each group:
   trajectories, induced-OOD rate (published and conditional), OOD share of
   the recorded steps, and the terminal d_min.  This answers where the OOD
   events come from: if the capped group carries them all, the arm's rate is a
   statement about truncated trajectories.

2. REACHED.  How far down the sigma schedule the truncated runs got
   (results/phase1/analysis/pfode_reached_sigma.json, from
   scripts/probe_pfode_reached_sigma.py).  A capped run's terminal structure
   belongs to that noise level, not to sigma_min.

3. COST.  Wall time per run and per unit of schedule progress, so the
   comparison is priced in the same currency the reduction in OOD rate buys.

What this does NOT do: it does not re-run the scan, and it cannot tell a run
that integrated smoothly to sigma_min from one that terminated early because
solve_ivp failed -- the frozen scan files do not store the sampler's converged
flag (run_phase1.run_candidate now stores it, so the
composition-only cells do carry it).  Both appear here as "completed" (nfe <
cap); note the distinction whenever a short run is also a collapsed one, which
happens at sigma_max = 5 (100% conditional OOD in the completed group).

Output: results/phase1/analysis/pfode_truncation.json (+ stdout tables).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import make_percomp_sigma_tables as mps  # noqa: E402
from make_percomp_sigma_tables import OOD_MASK  # noqa: E402

REACHED = os.path.join(_ROOT, "results", "phase1", "analysis", "pfode_reached_sigma.json")
OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "pfode_truncation.json")
CAP = 1000


def group_stats(sel: list) -> dict | None:
    if not sel:
        return None
    n = len(sel)
    clean = [c for c in sel if not (c["ood_bitmask"][0] & OOD_MASK)]
    k = sum(1 for c in clean if any(m & OOD_MASK for m in c["ood_bitmask"][1:]))
    steps = sum(len(c["ood_bitmask"]) for c in sel)
    ood_steps = sum(sum(1 for m in c["ood_bitmask"] if m & OOD_MASK) for c in sel)
    # Bit decomposition of the step-level OOD: the distance bit is what the
    # boundary claim is about; the force bit can fire on a structurally sound
    # but highly strained step (relevant at large sigma).
    return {
        "n": n,
        "induced_published": k / n,
        "induced_conditional": k / len(clean) if clean else None,
        "ood_step_share": ood_steps / steps if steps else None,
        "bit_dmin_share": sum(sum(1 for m in c["ood_bitmask"] if m & 0b000001)
                              for c in sel) / steps if steps else None,
        "bit_fmax_share": sum(sum(1 for m in c["ood_bitmask"] if m & 0b000010)
                              for c in sel) / steps if steps else None,
        "bit_vol_share": sum(sum(1 for m in c["ood_bitmask"] if m & 0b010000)
                             for c in sel) / steps if steps else None,
        "steps_median": float(np.median([len(c["ood_bitmask"]) for c in sel])),
        "d_min_final_median": float(np.median([c["final"]["d_min"] for c in sel])),
        "wall_median_s": float(np.median([c["wall_time_s"] for c in sel])),
        "valid_share": float(np.mean([c["valid"] for c in sel])),
    }


def main() -> None:
    rows: dict = {}
    sigmas = sorted({float(os.path.basename(d).split("_")[0][1:])
                     for d in os.listdir(mps.BASE) if d.startswith("s")})
    print("PF-ODE scan cells split by whether the run exhausted the "
          f"{CAP}-evaluation cap.\n")
    print(f"{'sig':>5s} {'arm':5s} {'group':>10s} {'n':>5s} {'ind.pub':>8s} "
          f"{'ind.cond':>9s} {'ood/steps':>10s} {'dmin:dist':>10s} "
          f"{'force':>7s} {'vol':>5s} {'d_min':>6s} {'valid':>6s} {'wall':>6s}")
    for sig in sigmas:
        for arm in ["bare", "l1", "l1l4"]:
            try:
                comps = mps.load_cell(sig, arm, "pfode")
            except Exception:
                continue
            cands = [c for lst in comps.values() for c in lst]
            groups = {name: group_stats([c for c in cands
                                         if (c["nfe"] >= CAP) == capped])
                      for name, capped in (("capped", True), ("completed", False))}
            rows[f"s{sig:g}_{arm}"] = {k: v for k, v in groups.items() if v}
            for name in ("capped", "completed"):
                g = groups[name]
                if not g:
                    continue
                print(f"{sig:5g} {arm:5s} {name:>10s} {g['n']:5d} "
                      f"{g['induced_published']*100:7.2f}% "
                      f"{(g['induced_conditional'] or 0)*100:8.2f}% "
                      f"{g['ood_step_share']*100:9.2f}% "
                      f"{g['bit_dmin_share']*100:9.2f}% "
                      f"{g['bit_fmax_share']*100:6.2f}% "
                      f"{g['bit_vol_share']*100:4.2f}% "
                      f"{g['d_min_final_median']:6.2f} "
                      f"{g['valid_share']*100:5.1f}% "
                      f"{g['wall_median_s']:5.0f}s")
        print()

    # Where do the induced events live?  Pooled over the cells where the
    # question is meaningful (the ones with both groups present).
    print("induced events by group, pooled over cells with both groups present:")
    tot = {"capped": [0, 0], "completed": [0, 0]}     # events, denominator
    for key, g in rows.items():
        if len(g) < 2:
            continue
        for name in tot:
            gg = g.get(name)
            if gg:
                tot[name][0] += gg["induced_published"] * gg["n"]
                tot[name][1] += gg["n"]
    for name, (k, n) in tot.items():
        print(f"  {name:10s} {int(round(k)):4d} induced / {n:5d} runs "
              f"= {k/n*100:5.2f}%")
    share = tot["capped"][0] / max(tot["capped"][0] + tot["completed"][0], 1e-9)
    print(f"  -> the capped group carries {share*100:.1f}% of all induced events")

    if os.path.exists(REACHED):
        probe = json.load(open(REACHED))
        print("\nreached sigma of the capped runs (probe):")
        print(f"{'sigma':>5s} {'arm':5s} {'capped':>7s} {'reached_med':>11s} "
              f"{'schedule_done':>13s} {'wall_med':>8s}")
        for c in probe["cells"]:
            print(f"{c['sigma_max']:5g} {c['arm']:5s} {c['capped_share']*100:6.1f}% "
                  f"{c['reached_capped_median']:11.3f} "
                  f"{c['schedule_fraction_completed_median']*100:12.0f}% "
                  f"{c['wall_median_s']:7.0f}s")
        rows["probe"] = probe["cells"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, _ROOT)}")


if __name__ == "__main__":
    main()
