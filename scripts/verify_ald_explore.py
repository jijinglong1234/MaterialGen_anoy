"""Verify pack #4: ALD exploration fix — pre-registered P1/P2 test.

Conditions (SrTiO3, FeNi3, LiF  x  sigma_max = 1.0  x  2 seeds  x  20 candidates):
  c0  current protocol      : alpha_noise = alpha_0 = 1e-3, anneal to sigma_min = 0.01
  c1  P2 only               : alpha_noise = 1e-2,          anneal to sigma_min = 0.01
  c2  P2 + P1               : alpha_noise = 1e-2,          anneal to sigma_floor = 0.1,
                              then fixed-sigma Langevin with the shared-alpha step
                              (drift and noise both use alpha_0 * sigma_floor/sigma_max
                              -> Boltzmann stationary distribution preserved)
  c3  P2 strong             : alpha_noise = 1e-1,          anneal to sigma_min = 0.01
                              (tradeoff-curve point; collision risk expected higher)

The drift step is untouched in all variants (alpha_0 = 1e-3, per-atom cap 3*sigma_k),
so drift-driven collisions are controlled for; only the noise scale changes.

Pre-registered gates (per composition):
  G1  total displacement >= 5x c0        (exploration fixed)
  G2  P_step (d_min<0.5 fraction) < 2x c0 (collision cost bounded)
  G3  drift-cap binding rate not inflated (defense structure intact)
  G4  terminal-phase energy stationary   (shared-alpha tail preserves equilibrium)

Displacement convention: total = sum over steps of sum over atoms of |dr_cart|,
same convention as the PF-ODE comparison (ALD ~1 A vs PF-ODE 19-51 A at sigma=1).

Usage: python scripts/verify_ald_explore.py [--device cuda:0]
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

COMP = ["SrTiO3", "FeNi3", "LiF"]
SIGMA = 1.0
SEEDS = [42, 123]
N_CAND = 20
CONDS = {
    "c0": dict(alpha_noise=1e-3, sigma_floor=0.01),
    "c1": dict(alpha_noise=1e-2, sigma_floor=0.01),
    "c2": dict(alpha_noise=1e-2, sigma_floor=0.10),
    "c3": dict(alpha_noise=1e-1, sigma_floor=0.01),
}


def run_variant(score, cfg, crystal, rng, n_steps, alpha_noise, sigma_floor):
    """ALD step loop with noise-drift decoupling (P2) and sigma-floor tail (P1).

    Returns per-step records: d_min, per-atom displacement sum, cap binding, E.
    """
    crystal = crystal.copy()
    sigmas = np.geomspace(cfg.sigma_max, sigma_floor, cfg.K + 1)  # = geometric_schedule
    rec = []
    for k in range(cfg.K):
        sigma_k = min(sigmas[k], cfg.sigma_max)
        alpha_k = cfg.alpha * (sigma_k / cfg.sigma_max)          # drift step (unchanged)
        floor_phase = sigma_k <= sigma_floor + 1e-9
        a_noise = alpha_k if floor_phase else alpha_noise * (sigma_k / cfg.sigma_max)
        for _ in range(cfg.M):
            score_result = score.compute(crystal)
            drift = alpha_k * score_result.frac_score
            drift_cart = drift @ crystal.lattice
            norms = np.linalg.norm(drift_cart, axis=1)
            cap = cfg.drift_cap * sigma_k
            scale = np.minimum(1.0, cap / np.maximum(norms, 1e-12))
            capped = drift_cart * scale[:, None] @ np.linalg.inv(crystal.lattice)
            eps = rng.normal(0, 1, (crystal.num_atoms, 3))
            eps = np.linalg.solve(crystal.lattice.T, eps.T).T
            noise = np.sqrt(2.0 * a_noise) * sigma_k * eps
            step = capped + noise
            step_cart = step @ crystal.lattice
            new_frac = (crystal.frac_coords + step) % 1.0
            crystal = rp.CrystalStructure.from_frac_coords(
                crystal.atomic_numbers, new_frac, crystal.lattice,
                dict(crystal.properties))
            rec.append(dict(
                d_min=float(crystal.get_minimum_distance()),
                disp=float(np.sum(np.linalg.norm(step_cart, axis=1))),
                cap_bound=bool(np.any(scale < 1.0)),
                energy=float(score_result.energy) if score_result.energy is not None else None,
                sigma_k=float(sigma_k)))
            if len(rec) >= n_steps:
                return rec
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    refs = rp.load_reference_structures(COMP)
    cfg = rp.make_ald_config(SIGMA, "bare")

    out = {}
    for name, cpar in CONDS.items():
        for comp in COMP:
            recs = []
            for seed in SEEDS:
                for cand in range(N_CAND):
                    initial = rp.make_initial(refs[comp], SIGMA,
                                              np.random.RandomState(seed * 100 + cand))
                    score = rp.make_score(calc, "bare")
                    recs += run_variant(score, cfg, initial,
                                        np.random.RandomState(seed),
                                        args.steps, cpar["alpha_noise"], cpar["sigma_floor"])
            r = recs
            dm = np.array([x["d_min"] for x in r])
            disp = np.array([x["disp"] for x in r])
            capb = np.mean([x["cap_bound"] for x in r])
            es = [x["energy"] for x in r if x["energy"] is not None]
            # terminal-phase energy stationarity: linear slope over last 50 steps (eV/step)
            tail = es[-50:] if len(es) >= 50 else es
            t = np.arange(len(tail))
            slope = float(np.polyfit(t, tail, 1)[0]) if len(tail) >= 3 else float("nan")
            key = f"{name}|{comp}"
            out[key] = dict(
                total_disp=float(np.sum(disp)),          # sum over steps of per-atom sum
                per_nfe=float(np.mean(disp)),
                P_step=float(np.mean(dm < 0.5)),
                d_min_med=float(np.median(dm)),
                cap_bind=float(capb),
                E_tail_slope=slope,
                E_tail_med=float(np.median(tail)),
                n_steps=len(r))
            print(f"{name} {comp:8} | total={np.sum(disp):8.1f} A per_nfe={np.mean(disp):.4f} "
                  f"P_step={np.mean(dm<0.5):7.4f} cap_bind={capb:5.1%} "
                  f"E_tail_slope={slope:+.2e} E_tail_med={np.median(tail):8.2f}", flush=True)

    with open("results/phase1/analysis/ald_explore.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/phase1/analysis/ald_explore.json")


if __name__ == "__main__":
    main()
