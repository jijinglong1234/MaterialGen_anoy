"""Pre-registered verify #6: ALD displacement-budget reallocation.

Evidence (verify pack #2 + travel-metric audit):
  - ~70% of NFE sits at sigma_k < 0.1, where noise_std = sqrt(2*alpha_0/sigma_max)
    * sigma_k^1.5 < 1.4e-3 A and drift/noise ~ 0.1-0.5: those steps neither
    explore nor refine (verify #2: late phase is pure noise jitter, ~0.005 A/step).
  - noise amplification does NOT create coverage (P2 gate failed: e2e +2-4%,
    path +3.3x; travel audit: path 283 A vs e2e 2.63 A -> the path body is an
    isotropic random walk). Real coverage comes from the early drift phase;
    the sigma<0.1 tail is dead weight.

Two budget-reallocation candidates (sigma_max = 1.0, bare, K=100, M=2, alpha=1e-3):
  early-stop (t1/t2): run the base schedule, stop at the first step with
                      sigma_k <= stop (0.3 -> ~54 NFE; 0.1 -> 100 NFE).
  stretch (d1)      : sigma_min = 0.1 instead of 0.01, same 200 NFE -> the
                      high-sigma phase gets 2x step density.
  NOTE: stretch is NOT the rejected c2 (no noise-drift decoupling, no P1 tail;
  the standard shared-alpha step is kept throughout). The rejected P1/P2 aimed
  at fixing an "exploration collapse" that the travel audit showed does not
  exist; this verification targets NFE efficiency with quality held fixed.

Key design: early-stop trajectories are exact sub-sequences of base (identical
sigma schedule and RNG stream until the stop), so t1/t2 cost zero extra compute
and pair exactly per (comp, seed, cand).

Pre-registered gates (per composition; medians over 120 paired trajectories):
  G1 refinement intact : dE    = E(idx_stop) - E(end)         <= +0.01 eV/atom
  G2 coverage intact   : de2e  = e2e(end) - e2e(idx_stop)     <= +0.20 A
  G3 stretch not worse : E_final(stretch) <= E_final(base) + 0.01 eV/atom
                         and e2e(stretch) >= 0.8 * e2e(base)
  G4 stretch safe      : P_step(stretch)  <= P_step(base) + 0.01

Verdict path (pre-registered):
  t2 passes G1+G2 on all comps -> adopt sigma_stop = 0.1 early stop
     (Phase-2 ALD default: 100 NFE, or 2x candidates/seed at 200 NFE)
  t1 also passes                -> adopt sigma_stop = 0.3 (~54 NFE)
  stretch passes G3+G4 AND E_final(stretch) <= E_final(t2) + 0.01
                                -> adopt sigma_min = 0.1 stretched schedule
  none                          -> keep c0; close the budget-reallocation line

Usage: python scripts/verify_ald_budget.py [--device cuda:0] [--cands 20]
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
STOPS = (0.3, 0.1)          # early-stop thresholds, sigma_k <= stop
STRETCH_MIN = 0.10          # stretched schedule sigma_min


def run_track(score, cfg, crystal, rng, sigma_min):
    """Same step loop as verify_ald_explore (shared-alpha step); records the
    full frac path plus per-step energy / d_min / sigma.

    Returns (p_frac, E, dmin, sdev, L): frac path (S+1, N, 3), per-step energy
    (S,) in eV (total), per-step d_min (S,), per-step sigma (S,), lattice matrix.
    """
    crystal = crystal.copy()
    sigmas = np.geomspace(cfg.sigma_max, sigma_min, cfg.K + 1)
    L = crystal.lattice.copy()
    fracs = [crystal.frac_coords.copy()]
    E, dmin, sdev = [], [], []
    for k in range(cfg.K):
        sigma_k = min(sigmas[k], cfg.sigma_max)
        alpha_k = cfg.alpha * (sigma_k / cfg.sigma_max)
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
            noise = np.sqrt(2.0 * alpha_k) * sigma_k * eps
            new_frac = (crystal.frac_coords + capped + noise) % 1.0
            crystal = rp.CrystalStructure.from_frac_coords(
                crystal.atomic_numbers, new_frac, crystal.lattice,
                dict(crystal.properties))
            fracs.append(crystal.frac_coords.copy())
            E.append(float(sr.energy) if sr.energy is not None else float("nan"))
            dmin.append(float(crystal.get_minimum_distance()))
            sdev.append(float(sigma_k))
    return np.array(fracs), np.array(E), np.array(dmin), np.array(sdev), L


def e2e_med(p_frac, L, i):
    """Median over atoms of min-image |r_i - r_0| @ L (end-to-end displacement)."""
    delta = p_frac[i] - p_frac[0]
    delta -= np.round(delta)
    return float(np.median(np.linalg.norm(delta @ L, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cands", type=int, default=20)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")
    refs = rp.load_reference_structures(COMP)
    cfg = rp.make_ald_config(SIGMA, "bare")

    # ---- run base (c0) and stretch (d1) trajectories ----
    traj = {"base": [], "stretch": []}
    for comp in COMP:
        for seed in SEEDS:
            for cand in range(args.cands):
                initial = rp.make_initial(refs[comp], SIGMA,
                                          np.random.RandomState(seed * 100 + cand))
                for name, s_min in (("base", cfg.sigma_min), ("stretch", STRETCH_MIN)):
                    score = rp.make_score(calc, "bare")
                    p_frac, E, dmin, sdev, L = run_track(
                        score, cfg, initial.copy(),
                        np.random.RandomState(seed), s_min)
                    traj[name].append(dict(comp=comp, seed=seed, cand=cand,
                                           p_frac=p_frac, E=E, dmin=dmin,
                                           sdev=sdev, L=L,
                                           n_atoms=initial.num_atoms))
                    print(f"{name:7} {comp:7} seed={seed} cand={cand:2d} "
                          f"E_end={E[-1]/initial.num_atoms:8.3f} eV/atom", flush=True)

    # ---- early-stop sub-sequence analysis on base trajectories ----
    def early_stop(t, stop):
        idx = int(np.argmax(t["sdev"] <= stop))
        if t["sdev"][idx] > stop:            # never reached (safety)
            idx = len(t["sdev"]) - 1
        n_atoms = t["n_atoms"]
        dE = (t["E"][idx] - t["E"][-1]) / n_atoms            # eV/atom, >= 0 if still descending
        de2e = e2e_med(t["p_frac"], t["L"], -1) - e2e_med(t["p_frac"], t["L"], idx)
        return dict(idx=idx, nfe=idx, dE=dE, de2e=de2e,
                    E_stop=t["E"][idx] / n_atoms,
                    dmin_stop=float(t["dmin"][idx]))

    def summary(rows):
        a = np.array(rows)
        return dict(n=len(a), med=float(np.median(a)), q25=float(np.percentile(a, 25)),
                    q75=float(np.percentile(a, 75)))

    res = {}
    print("\n=== early-stop (sub-sequences of base) ===")
    for comp in COMP:
        for stop in STOPS:
            dE = [early_stop(t, stop)["dE"] for t in traj["base"] if t["comp"] == comp]
            de2e = [early_stop(t, stop)["de2e"] for t in traj["base"] if t["comp"] == comp]
            nfe = np.median([early_stop(t, stop)["nfe"] for t in traj["base"] if t["comp"] == comp])
            key = f"stop{stop}|{comp}"
            res[key] = dict(nfe_med=int(nfe), dE=summary(dE), de2e=summary(de2e))
            print(f"stop={stop} {comp:7} | nfe~{int(nfe):3d} "
                  f"dE_med={np.median(dE):+.4f} eV/at (G1 <= +0.01: {'PASS' if np.median(dE)<=0.01 else 'FAIL'}) "
                  f"de2e_med={np.median(de2e):+.3f} A (G2 <= +0.20: {'PASS' if np.median(de2e)<=0.20 else 'FAIL'})")

    print("\n=== stretch vs base (same 200 NFE) ===")
    for comp in COMP:
        b = [t for t in traj["base"] if t["comp"] == comp]
        s = [t for t in traj["stretch"] if t["comp"] == comp]
        E_b = np.median([t["E"][-1] / t["n_atoms"] for t in b])
        E_s = np.median([t["E"][-1] / t["n_atoms"] for t in s])
        e2e_b = np.median([e2e_med(t["p_frac"], t["L"], -1) for t in b])
        e2e_s = np.median([e2e_med(t["p_frac"], t["L"], -1) for t in s])
        P_b = np.mean([np.mean(t["dmin"] < 0.5) for t in b])
        P_s = np.mean([np.mean(t["dmin"] < 0.5) for t in s])
        g3 = (E_s <= E_b + 0.01) and (e2e_s >= 0.8 * e2e_b)
        g4 = P_s <= P_b + 0.01
        key = f"stretch|{comp}"
        res[key] = dict(E_final_base=E_b, E_final_stretch=E_s,
                        e2e_base=e2e_b, e2e_stretch=e2e_s,
                        P_step_base=P_b, P_step_stretch=P_s)
        print(f"{comp:7} | E base={E_b:8.3f} stretch={E_s:8.3f} (G3: {'PASS' if g3 else 'FAIL'}) "
              f"e2e base={e2e_b:5.2f} stretch={e2e_s:5.2f} "
              f"P_step base={P_b:.4f} stretch={P_s:.4f} (G4: {'PASS' if g4 else 'FAIL'})")

    with open("results/phase1/analysis/ald_budget.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\nwrote results/phase1/analysis/ald_budget.json")


if __name__ == "__main__":
    main()
