"""Post-hoc probe: NNP pair potential along the shortest bond.

Measures, per composition, the MACE-MP-0 energy/force along the nearest-
neighbor bond coordinate d (single pair compressed, others fixed):

  * k_eff  — harmonic curvature of E(d) at equilibrium (the "restoring
             drift" that suppresses the naive geometric collision model)
  * F_short — learned pair force at d = 0.5..0.75 * r_NN (is there any
             learned repulsion, or does the NNP push the pair in?)
  * E(d) curve — used to compute the equilibrium tail P(d < 0.5) under the
             Langevin target exp(-beta*E), the confinement model's
             prediction for the collision rate at small sigma.

Test: does n_pairs * Phi((0.5 - r_NN)*sqrt(beta*k_eff)/sqrt(2)) match the
empirical per-step collision rate at sigma = 0.5? Does the measured k_eff
ordering match the ordering implied by the empirical sigma_c?

Usage:  python scripts/probe_pair_potential.py [--device cuda:0] [--comps SrTiO3 ZnS LiF GaN FeNi3]
"""
import argparse
import math
import pickle
import sys
import numpy as np
import torch

RECHECK_COMPS = ["SrTiO3", "ZnS", "LiF", "GaN", "FeNi3"]
PROCESSED = "results/reference/mp_20/mace_test.pkl"
BETA = 1.0 / 0.02585  # eV^-1 at 300 K
D_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
          0.98, 1.00, 1.02, 1.05]  # x r_NN


def load_best(comps):
    with open(PROCESSED, "rb") as f:
        records = pickle.load(f)
    best = {}
    for r in records:
        meta = r["metadata"]
        formula = meta.get("pretty_formula")
        if formula not in comps:
            continue
        ehull = float(meta.get("e_above_hull", np.inf))
        if formula not in best or ehull < best[formula][1]:
            best[formula] = (r, ehull)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--comps", nargs="+", default=RECHECK_COMPS)
    ap.add_argument("--out", default="results/phase1/analysis/pair_potential.json")
    args = ap.parse_args()

    from mace.calculators import mace_mp
    from ase import Atoms
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")

    best = load_best(args.comps)
    out = {}
    for comp in args.comps:
        r = best[comp][0]
        atoms = Atoms(numbers=r["numbers"], positions=r["frac_coords"] @ r["lattice"],
                      cell=r["lattice"], pbc=True)
        atoms.calc = calc
        pos0 = atoms.positions.copy()
        L = r["lattice"]
        # shortest pair via min-image
        dist, idx = atoms.get_all_distances(mic=True), None
        n = len(atoms)
        d0, pair = np.inf, None
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] < d0:
                    d0, pair = dist[i, j], (i, j)
        i, j = pair
        u = (pos0[j] - pos0[i]) / d0  # unit bond vector (direct)
        u = u - L @ np.round(np.linalg.solve(L, u))  # min-image direction
        u = u / np.linalg.norm(u)

        Es, Fs = [], []
        for x in D_GRID:
            atoms.positions = pos0.copy()
            d = d0 * x
            atoms.positions[j] = pos0[j] + (d - d0) * u
            atoms.wrap()
            E = atoms.get_potential_energy()
            F = atoms.get_forces()
            # force on j along u (positive = pushing apart)
            Fpair = np.dot(F[j], u)
            Es.append(E)
            Fs.append(Fpair)
        Es, Fs = np.array(Es), np.array(Fs)

        # harmonic curvature near equilibrium (fit E = c0 + c1*d + c2*d^2 on 0.95-1.05 r_NN)
        m = np.array([k for k, x in enumerate(D_GRID) if 0.95 <= x <= 1.05])
        c = np.polyfit(np.array(D_GRID)[m], Es[m], 2)
        k_eff = 2 * c[0]  # eV/A^2
        # equilibrium tail probability at sigma=0.5, per pair (1D harmonic marginal)
        P_tail_pair = 0.5 * (1.0 + math.erf((0.5 - d0) * np.sqrt(BETA * k_eff / 2)))
        out[comp] = dict(r_nn=float(d0), k_eff=float(k_eff),
                         E_min=float(np.min(Es)), E_max=float(np.max(Es)),
                         d_grid=[float(x) for x in D_GRID],
                         E=[float(e) for e in Es], F_pair=[float(f) for f in Fs],
                         F_at_half=float(Fs[D_GRID.index(0.50)]),
                         F_at_075=float(Fs[D_GRID.index(0.75)]),
                         P_tail_sigma05=float(P_tail_pair),
                         n_pairs_shortest=1)
        print(f"{comp:8} r_NN={d0:6.3f} A  k_eff={k_eff:8.3f} eV/A^2  "
              f"F(0.50r)={Fs[D_GRID.index(0.50)]:+9.2f} eV/A  "
              f"F(0.75r)={Fs[D_GRID.index(0.75)]:+9.2f}  E_min={np.min(Es):8.2f} eV  "
              f"P_tail@0.5={P_tail_pair:.3e}", flush=True)

    import json
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
