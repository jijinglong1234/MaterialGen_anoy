"""Pre-registered verify #7: A2 start-protocol fix — d_min filtering.

Evidence (verify #1): at sigma=5, 70-80% of P_step comes from start-OOD
trajectories (P_start 0.14/0.06/0.02 on SrTiO3/FeNi3/LiF, pooled); clean
trajectories collide at ~3% (early big steps). Fix candidate: re-sample the
make_initial perturbation until d_min >= threshold (max 10 tries, fallback to
last sample — the protocol fix is applied at the start, NOT inside the step
loop, so trajectory dynamics are unchanged).

Variants at sigma_max = 5.0 (collision-heavy regime, max signal), bare, 200 NFE:
  base  : current make_initial (sigma_max Gaussian, no filter)
  f04   : filter d_min >= 0.4 A   (rejection re-sample)
  f05   : filter d_min >= 0.5 A
  f06   : filter d_min >= 0.6 A

Pre-registered gates:
  G1 (benefit)  : sigma=5, P_step(filter) <= 0.5 * P_step(base), per comp;
                  require SrTiO3 + at least one other comp to pass.
  G2 (low-sigma untouched): sigma=1 rejection rate < 5% for all comps at the
                  chosen threshold -> low-sigma conditions keep their
                  distribution, preserving mini-validation comparability.
  G3 (quality)  : sigma=5 E_final(filter) <= E_final(base) + 0.02 eV/atom.

Verdict: the best threshold of {0.4, 0.5, 0.6} passing G1+G2+G3 is adopted in
make_initial; if none passes, close the filtering variant (keep protocol).

Usage: python scripts/verify_ald_start.py [--device cuda:0] [--cands 10]
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
from verify_ald_budget import run_track

COMP = ["SrTiO3", "FeNi3", "LiF"]
SIGMA = 5.0            # collision-heavy regime (max signal)
SIGMA_LOW = 1.0        # rejection-rate check (low-sigma parity)
SEEDS = [42, 123, 999, 2024, 7777]   # mini-aligned (sigma=5 bad starts are heavy-tailed)
P_STEP_STAT = "mean"   # pooled per-trajectory mean, matching verify #1 (0.10425).
                       # median underestimates the heavy-tailed P_step distribution
                       # (my first run: median 0.0225 vs verify #1 mean 0.104).
THRESH = {"f04": 0.4, "f05": 0.5, "f06": 0.6}
MAX_TRIES = 10


def make_initial_f(refs, comp, sigma, rng, thresh):
    """make_initial + d_min rejection filter; returns (crystal, n_tries)."""
    c = None
    for t in range(1, MAX_TRIES + 1):
        c = rp.make_initial(refs[comp], sigma, rng)
        if c.get_minimum_distance() >= thresh:
            return c, t
    return c, MAX_TRIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cands", type=int, default=10)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    refs = rp.load_reference_structures(COMP)
    cfg5 = rp.make_ald_config(SIGMA, "bare")

    res = {}

    # ---- G2: low-sigma rejection rates (no NNP needed) ----
    print("=== G2: rejection rate at sigma=1.0 (2000 starts/comp) ===")
    for comp in COMP:
        for th, tval in THRESH.items():
            n_rej = 0
            n = 2000
            rng = np.random.RandomState(7)
            for _ in range(n):
                c, tries = make_initial_f(refs, comp, SIGMA_LOW, rng, tval)
                n_rej += tries - 1
            rate = n_rej / n
            res.setdefault(f"rej1|{comp}|{th}", rate)
            print(f"{comp:7} {th}: rej_rate={rate:.4f} "
                  f"({'PASS' if rate < 0.05 else 'FAIL'})", flush=True)

    # ---- G1+G3: sigma=5 trajectories ----
    print("\n=== G1/G3: P_step & E at sigma=5.0 ===")
    for name in ["base", "f04", "f05", "f06"]:
        for comp in COMP:
            P, E, rej = [], [], 0
            for seed in SEEDS:
                for cand in range(args.cands):
                    rng0 = np.random.RandomState(seed * 100 + cand)
                    if name == "base":
                        initial = rp.make_initial(refs[comp], SIGMA, rng0)
                    else:
                        initial, tries = make_initial_f(refs, comp, SIGMA, rng0,
                                                        THRESH[name])
                        rej += tries - 1
                    score = rp.make_score(calc, "bare")
                    p_frac, E_arr, dmin, sdev, L = run_track(
                        score, cfg5, initial.copy(), np.random.RandomState(seed),
                        cfg5.sigma_min)
                    P.append(np.mean(dmin < 0.5))
                    E.append(E_arr[-1] / initial.num_atoms)
            key = f"{name}|{comp}"
            res[key] = dict(P_step=float(np.mean(P)),
                            P_step_med=float(np.median(P)),
                            E_final=float(np.median(E)),
                            rej_rate=rej / (len(SEEDS) * args.cands))
            print(f"{name:5} {comp:7} | P_step(mean)={np.mean(P):7.4f} "
                  f"med={np.median(P):7.4f} E_final={np.median(E):8.3f} "
                  f"rej={rej / (len(SEEDS) * args.cands):.3f}",
                  flush=True)

    # ---- gate table ----
    print("\n=== gates ===")
    g1 = {}
    for th in THRESH:
        ok = {c: res[f"{th}|{c}"]["P_step"] <= 0.5 * res[f"base|{c}"]["P_step"]
              for c in COMP}
        g1[th] = ok
        print(f"{th}: G1 {ok} (need SrTiO3 + >=1 more)")
    print("G2:", {th: all(res[f"rej1|{c}|{th}"] < 0.05 for c in COMP) for th in THRESH})
    print("G3:", {th: all(res[f"{th}|{c}"]["E_final"] <=
                          res[f"base|{c}"]["E_final"] + 0.02 for c in COMP)
                  for th in THRESH})

    with open("results/phase1/analysis/ald_start_filter.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\nwrote results/phase1/analysis/ald_start_filter.json")


if __name__ == "__main__":
    main()
