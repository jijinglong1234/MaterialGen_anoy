"""Verify pack #2: measure actual ALD three-scale balance.

Re-runs bare ALD trajectories (SrTiO3/LiF/GaN x sigma in {1,5} x 2 seeds x 3
candidates) with per-step logging of the drift (pre-cap), cap scale, noise
displacement, and NNP forces — to test the analytic claim that the ALD is
drift/cap-dominated (noise subdominant) at sigma <= 2 and to measure the
cap binding rate.

Usage: python scripts/verify_ald_scales.py [--device cuda:0] [--steps 200]
"""
import argparse
import json
import numpy as np
import sys

sys.path.insert(0, "scripts")
import run_phase1 as rp

COMP = ["SrTiO3", "LiF", "GaN"]
SIGMAS = [1.0, 5.0]
SEEDS = [42, 123]
N_CAND = 3
BETA = 38.68


def run_logged(score, cfg, crystal, rng, n_steps):
    """ALD step loop with logging (mirrors ald.py sample(); no L2/L3/L4)."""
    crystal = crystal.copy()
    sigmas = np.geomspace(cfg.sigma_max, cfg.sigma_min, cfg.K + 1)
    rec = []
    sigma_current = cfg.sigma_max
    for k in range(cfg.K):
        sigma_k = min(sigmas[k], sigma_current)
        alpha_k = cfg.alpha * (sigma_k / cfg.sigma_max)
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
            noise = np.sqrt(2.0 * alpha_k) * sigma_k * eps
            step = capped + noise
            new_frac = (crystal.frac_coords + step) % 1.0
            crystal = rp.CrystalStructure.from_frac_coords(
                crystal.atomic_numbers, new_frac, crystal.lattice,
                dict(crystal.properties))
            rec.append(dict(
                fmax=float(np.max(np.abs(score_result.forces))),
                fmed=float(np.median(np.abs(score_result.forces))),
                drift_max=float(np.max(norms)),
                noise_max=float(np.max(np.linalg.norm(noise @ crystal.lattice, axis=1))),
                step_max=float(np.max(np.linalg.norm(step @ crystal.lattice, axis=1))),
                cap_bound=bool(np.any(scale < 1.0)),
                scale_min=float(np.min(scale)),
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

    out = {}
    for sigma in SIGMAS:
        cfg = rp.make_ald_config(sigma, "bare")
        for comp in COMP:
            recs = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                for cand in range(N_CAND):
                    initial = rp.make_initial(refs[comp], sigma,
                                              np.random.RandomState(seed * 100 + cand))
                    score = rp.make_score(calc, "bare")
                    recs += run_logged(score, cfg, initial, rng, args.steps)
            r = recs
            f = np.array([x["fmax"] for x in r])
            drift = np.array([x["drift_max"] for x in r])
            noise = np.array([x["noise_max"] for x in r])
            step = np.array([x["step_max"] for x in r])
            n_cap = np.mean([x["cap_bound"] for x in r])
            out[f"{comp}|{sigma}"] = dict(
                f_med=float(np.median(f)), f_q75=float(np.percentile(f, 75)),
                drift_med=float(np.median(drift)), noise_med=float(np.median(noise)),
                drift_noise_ratio=float(np.median(drift) / max(np.median(noise), 1e-12)),
                cap_bind_rate=float(n_cap), step_med=float(np.median(step)),
                n_steps=len(r))
            print(f"{comp:8} sigma={sigma:4.1f} | |F|_max med={np.median(f):7.2f} q75={np.percentile(f,75):7.2f} "
                  f"| drift_med={np.median(drift):.4f} noise_med={np.median(noise):.4f} "
                  f"ratio={np.median(drift)/max(np.median(noise),1e-12):6.2f} "
                  f"| cap_bind={n_cap:5.1%} step_med={np.median(step):.3f}", flush=True)
    with open("results/phase1/analysis/ald_scales.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
