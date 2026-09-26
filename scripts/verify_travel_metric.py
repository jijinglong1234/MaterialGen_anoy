"""Verify #5: unified travel-metric measurement, ALD vs PF-ODE.

The exploration-collapse claim ("ALD total ~1 A vs PF-ODE 19-51 A, 26x gap")
rests on a median extrapolation (verify #2) that is wrong for the heavy-tailed
step distribution (cap-bound steps contribute most of the path length).
This script re-measures BOTH samplers with the PF-ODE script's own convention
and adds the end-to-end displacement (the true exploration coverage):

  path_len   = sum over steps of sum over axes of ||(frac diff @ L)_axis||_2
               over atoms  (exactly the pfode script convention)
  e2e        = median over atoms of |minimage(r_final - r_initial) @ L|
  disp_sum   = sum over steps of sum over atoms of |dr_cart| (my #4 convention)

Conditions: SrTiO3/FeNi3/LiF x sigma_max=1.0, bare, 2 seeds x 5 cands.
ALD variants: c0 (current) and c2 (P2+P1 fix candidate).

Usage: python scripts/verify_travel_metric.py [--device cuda:0]
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
N_CAND = 5


def per_step_cart(p_frac, L):
    """Per-step cartesian displacement array (S, N, 3), min-image corrected.

    The raw frac difference can jump by ~1.0 when an atom wraps the periodic
    cell (% 1.0 in the step loop), which would inflate path length and step
    max by ~|L| (~3.9 A) per wrap event.  min-image: d_frac -= round(d_frac).
    """
    d_frac = np.diff(p_frac, axis=0)
    d_frac -= np.round(d_frac)
    return d_frac @ L


def metrics(p_frac, L):
    d = per_step_cart(p_frac, L)
    axis_norm = np.linalg.norm(d, axis=1)                 # (S, 3): per-axis atom norm
    path_len = float(axis_norm.sum())                     # pfode-script total (min-image)
    disp_sum = float(np.sum(np.linalg.norm(d, axis=2)))   # sum |dr| over atoms
    step_max = float(np.max(np.linalg.norm(d, axis=2)))   # max per-atom step
    # end-to-end, min-image corrected
    delta = p_frac[-1] - p_frac[0]
    delta -= np.round(delta)
    e2e = np.linalg.norm(delta @ L, axis=1)
    return dict(path_len=path_len, disp_sum=disp_sum, step_max=step_max,
                e2e_med=float(np.median(e2e)), e2e_max=float(np.max(e2e)))


def run_ald(score, cfg, crystal, rng, n_steps, alpha_noise, sigma_floor):
    """Same step loop as verify_ald_explore; returns frac-coordinate path."""
    crystal = crystal.copy()
    sigmas = np.geomspace(cfg.sigma_max, sigma_floor, cfg.K + 1)
    fracs = [crystal.frac_coords.copy()]
    for k in range(cfg.K):
        sigma_k = min(sigmas[k], cfg.sigma_max)
        alpha_k = cfg.alpha * (sigma_k / cfg.sigma_max)
        floor_phase = sigma_k <= sigma_floor + 1e-9
        a_noise = alpha_k if floor_phase else alpha_noise * (sigma_k / cfg.sigma_max)
        for _ in range(cfg.M):
            sr = score.compute(crystal)
            drift = alpha_k * sr.frac_score
            drift_cart = drift @ crystal.lattice
            norms = np.linalg.norm(drift_cart, axis=1)
            cap = cfg.drift_cap * sigma_k
            scale = np.minimum(1.0, cap / np.maximum(norms, 1e-12))
            capped = drift_cart * scale[:, None] @ np.linalg.inv(crystal.lattice)
            eps = rng.normal(0, 1, (crystal.num_atoms, 3))
            eps = np.linalg.solve(crystal.lattice.T, eps.T).T
            noise = np.sqrt(2.0 * a_noise) * sigma_k * eps
            new_frac = (crystal.frac_coords + capped + noise) % 1.0
            crystal = rp.CrystalStructure.from_frac_coords(
                crystal.atomic_numbers, new_frac, crystal.lattice,
                dict(crystal.properties))
            fracs.append(crystal.frac_coords.copy())
            if len(fracs) - 1 >= n_steps:
                return np.array(fracs), crystal
    return np.array(fracs), crystal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    refs = rp.load_reference_structures(COMP)
    cfg = rp.make_ald_config(SIGMA, "bare")
    pcfg = rp.make_pfode_config(SIGMA, "bare")

    out = {}
    for comp in COMP:
        for seed in SEEDS:
            for cand in range(N_CAND):
                rng0 = np.random.RandomState(seed * 100 + cand)
                initial = rp.make_initial(refs[comp], SIGMA, rng0)
                # --- PF-ODE (bare, ivp) ---
                pf = rp.make_pfode_sampler(pcfg)
                res = pf.sample(rp.make_score(calc, "bare"), initial.copy())
                p_frac = np.array([s.frac_coords for s in res.path])
                m = metrics(p_frac, res.path[-1].lattice)
                out.setdefault(f"pfode|{comp}", []).append(
                    dict(nfe=res.nfe, **m))
                # --- ALD c0 and c2 ---
                for name, (an, sf) in (("c0", (1e-3, 0.01)), ("c2", (1e-2, 0.10))):
                    score = rp.make_score(calc, "bare")
                    p_frac, crys = run_ald(score, cfg, initial.copy(),
                                           np.random.RandomState(seed), 200, an, sf)
                    m = metrics(p_frac, crys.lattice)
                    out.setdefault(f"ald-{name}|{comp}", []).append(dict(nfe=200, **m))

    res = {}
    for k, v in out.items():
        a = np.array([[x["path_len"], x["disp_sum"], x["e2e_med"], x["e2e_max"],
                       x["step_max"], x["nfe"]] for x in v])
        res[k] = dict(path_len_med=float(np.median(a[:, 0])),
                      path_len_max=float(np.max(a[:, 0])),
                      disp_sum_med=float(np.median(a[:, 1])),
                      e2e_med=float(np.median(a[:, 2])),
                      e2e_q75=float(np.percentile(a[:, 2], 75)),
                      step_max=float(np.max(a[:, 4])),
                      nfe_med=float(np.median(a[:, 5])),
                      n_traj=len(v))
        print(f"{k:14} | path_len_med={res[k]['path_len_med']:8.1f} "
              f"e2e_med={res[k]['e2e_med']:6.2f} e2e_q75={res[k]['e2e_q75']:6.2f} "
              f"step_max={res[k]['step_max']:6.2f} nfe={res[k]['nfe_med']:.0f}", flush=True)

    with open("results/phase1/analysis/travel_metric.json", "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote results/phase1/analysis/travel_metric.json")


if __name__ == "__main__":
    main()
