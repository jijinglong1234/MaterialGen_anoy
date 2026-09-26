"""Pre-registered verify: L1' items 2-3 — deep wall + probe calibration.

Design (plan, L1' items 2-3): replace the Pauli exponential tail
below 0.6*r_wall with a C^1-matched power-law floor (item 3), and calibrate
the wall entry force per element pair so that
F_Pauli(r_wall) >= 2.5 * F_NNP(r_wall) (item 2; probe table
data/pauli/probe_calibration.json).  L1' item 1 (wall-relative drift cap) is
already adopted fleet-wide (verify #3) and stays ON in both L1 arms.

Arms (only the potential form differs between L1 and L1'):
  bare : no protection (ALD only — dwell/collision baselines for G2)
  L1   : PauliRepulsion(deep_wall=False, calibration={})  — pre-L1' form
  L1'  : PauliRepulsion(deep_wall=True) + auto-loaded probe calibration

Pre-registered gates (ALD, sigma in {1.0, 2.0}, pooled over the 5 comps):
  G1 induced : induced(L1') <= induced(L1) + 0.02            [per sigma]
  G2 safety  : dwell(L1') < dwell(bare) AND deep(L1') < deep(bare)   [per sigma]
  G3 quality : penalty(L1') <= penalty(L1), where
               penalty = median(E_hull) - median(E_hull(bare))
  G4 pfode   : cap-hit(L1') <= cap-hit(L1) - 0.02            [per sigma]

Metrics (sampler-induced, steps >= 1, pooled over trajectories):
  induced : fraction of start-clean trajectories with any OOD step
            (start-OOD = the initial record trips the OOD mask)
  dwell   : fraction of steps with d_min < 1.3 A (wall-dwell diagnostic)
  deep    : fraction of steps with d_min < 0.5 A (deep-collision)
  cap-hit : PF-ODE fraction of trajectories hitting max_nfe = 1000

Verdict path (pre-registered):
  G1-G4 pass -> adopt L1' for the full run (make_score already wired)
  G4 fails   -> keep the deep wall for ALD only (make_score gets a flag)
  G1/G3 fail -> revert the deep wall; keep calibration unless G3 blames it
  G2 fails   -> re-open the L1' design (floor power / fraction / offset)

Usage: python scripts/verify_l1prime.py [--device cuda:1] [--cands 40]
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import run_phase1 as rp
from materialgen.core.pauli import PauliRepulsion
from materialgen.core.score_function import AugmentedScore, NNPScore, PauliScore

COMPS = rp.MINI_COMPOSITIONS            # SrTiO3, ZnS, LiF, GaN, FeNi3
SIGMAS = [1.0, 2.0]
SEEDS = [42, 123]
OOD_MASK = 0b010011                     # d_min<0.5 | max_f>500 | Type III
DWELL_DMIN = 1.3
G1_TOL, G4_MARGIN = 0.02, 0.02
OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "l1prime_verify.json")


def make_scores(calc, label):
    nnp = NNPScore(calc, beta=rp.BETA_300K, compute_stress=False, label=label)
    l1 = AugmentedScore(
        nnp, PauliScore(pauli=PauliRepulsion(deep_wall=False, calibration={}),
                        beta=rp.BETA_300K))
    l1p = rp.make_score(calc, "l1", label=label)   # deep_wall + calibration
    bare = nnp
    return {"bare": bare, "L1": l1, "L1'": l1p}


def cell_metrics(cands, sampler_name):
    """Pooled metrics over candidates of one (comp, sigma, arm) cell."""
    n = len(cands)
    clean = [c for c in cands
             if not (c["ood_bitmask"][0] & OOD_MASK)]      # start-OOD floor
    induced = sum(any(m & OOD_MASK for m in c["ood_bitmask"][1:])
                  for c in clean) / max(len(clean), 1)
    hists = [h[1:] for h in [c["d_min_hist"] for c in cands] if len(h) > 1]
    steps = np.concatenate(hists) if hists else np.array([])
    dwell = float((steps < DWELL_DMIN).mean()) if len(steps) else float("nan")
    deep = float((steps < 0.5).mean()) if len(steps) else float("nan")
    eh = [c["final"]["e_hull"] for c in cands if c["final"]["e_hull"] is not None]
    out = {"induced": round(induced, 4),
           "dwell": round(dwell, 4), "deep": round(deep, 4),
           "e_hull_med": None if not eh else round(float(np.median(eh)), 2),
           "n_traj": n, "n_start_clean": len(clean)}
    if sampler_name == "pfode":
        out["cap_hit"] = round(sum(c["nfe"] >= rp.PF_MAX_NFE for c in cands) / n, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cands", type=int, default=40)
    ap.add_argument("--cells", default="",
                    help="comma-separated 'sigma|sampler|arm' keys to run "
                         "(default: all)")
    ap.add_argument("--resume", default="",
                    help="path to a partial l1prime JSON whose cells are "
                         "carried into the final output")
    args = ap.parse_args()
    only = set(c.strip() for c in args.cells.split(",") if c.strip()) or None
    resume_cells = {}
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            resume_cells = json.load(f).get("cells", {})

    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    refs = rp.load_reference_structures(COMPS)
    hull = rp.HullEvaluator(rp.load_hull_table())
    for comp in COMPS:
        ref = refs[comp]
        n_atoms = len(ref["numbers"])
        ref_c = rp.CrystalStructure.from_frac_coords(
            ref["numbers"], ref["frac_coords"], ref["lattice"])
        hull.calibrate(comp, rp.calc_ref_energy(calc, ref_c), n_atoms,
                       float(ref["metadata"].get("formation_energy_per_atom", 0.0)))

    scores = make_scores(calc, rp.nnp_label("mace"))
    cells = {}                     # "sigma|sampler|arm" -> metrics
    for sigma in SIGMAS:
        for sampler_name in ("ald", "pfode"):
            arms = ["bare", "L1", "L1'"] if sampler_name == "ald" else ["L1", "L1'"]
            for arm in arms:
                key = f"{sigma}|{sampler_name}|{arm}"
                if only is not None and key not in only:
                    continue
                cands = []
                for comp in COMPS:
                    if sampler_name == "ald":
                        prot = "bare" if arm == "bare" else "l1"
                        smp = rp.make_ald_sampler(rp.make_ald_config(sigma, prot), prot)
                    else:
                        smp = rp.make_pfode_sampler(rp.make_pfode_config(sigma, "l1"))
                    for seed in SEEDS:
                        for cand in range(args.cands):
                            c = rp.run_candidate(scores[arm], smp, refs[comp],
                                                 sigma, seed, cand, hull)
                            cands.append(c)
                cells[key] = cell_metrics(cands, sampler_name)
                print(f"{key}: {cells[key]}", flush=True)
                # incremental save: completed cells survive a kill/restart
                with open(OUT, "w") as f:
                    json.dump({"meta": {"partial": True, "done": sorted(cells)},
                               "cells": cells}, f, indent=1)

    # ---- gates (cells may come from --resume) ----
    all_cells = {**resume_cells, **cells}

    def cell(sigma, sampler, arm):
        return all_cells.get(f"{sigma}|{sampler}|{arm}")

    verdicts = {}
    for sigma in SIGMAS:
        b, l1, l1p = (cell(sigma, "ald", a) for a in ("bare", "L1", "L1'"))
        p1, p1p = cell(sigma, "pfode", "L1"), cell(sigma, "pfode", "L1'")
        missing = [k for k, v in (("ald", b), ("L1", l1), ("L1'", l1p),
                                  ("pfode_L1", p1), ("pfode_L1'", p1p)) if v is None]
        if missing:
            verdicts[str(sigma)] = {"missing_cells": missing}
            print(f"sigma={sigma}: MISSING {missing}", flush=True)
            continue
        g1 = l1p["induced"] <= l1["induced"] + G1_TOL
        g2 = l1p["dwell"] < b["dwell"] and l1p["deep"] < b["deep"]
        pen_l1 = (l1["e_hull_med"] or 0) - (b["e_hull_med"] or 0)
        pen_l1p = (l1p["e_hull_med"] or 0) - (b["e_hull_med"] or 0)
        g3 = pen_l1p <= pen_l1
        g4 = p1p["cap_hit"] <= p1["cap_hit"] - G4_MARGIN
        verdicts[str(sigma)] = {"G1_induced": g1, "G2_safety": g2,
                                "G3_quality": g3, "G4_pfode_nfe": g4,
                                "penalty_L1": round(pen_l1, 2),
                                "penalty_L1p": round(pen_l1p, 2)}
        print(f"sigma={sigma}: G1={g1} G2={g2} G3={g3} G4={g4}", flush=True)

    payload = {"meta": {"cands": args.cands,
                        "comps": COMPS, "seeds": SEEDS, "sigmas": SIGMAS,
                        "note": "L1' = deep_wall + probe calibration; cap ON in "
                                "both L1 arms; asymmetric-cap build ("
                                "verify #9 fix) for all ALD non-bare cells; "
                                "1.0|ald|bare carried over from the pre-fix run "
                                "(bare has no cap, unaffected by the fix)",
                        "resumed_from": args.resume or None},
               "cells": all_cells, "verdicts": verdicts}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
