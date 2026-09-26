"""L1' item 2: probe-driven Pauli wall entry-force calibration.

For every element pair present in the fleet's 20 MP-20 compositions, measure
the NNP's learned pair force at the wall radius r_wall = 0.75*(R_i + R_j) by
compressing the closest such pair along its min-image bond direction.  A
two-point differencing against r_wall + 0.1 Angstrom removes the background
force the other atoms exert on the moved atom, isolating the pair's own
restoring/inward force F_pair = |(F(r_wall) - F(r_wall+0.1)) . u|.

Calibration: the Pauli prefactor A is scaled per pair so the wall's entry
force meets F_Pauli(r_wall) >= 2.5 * F_pair (the L1' design criterion;
scale = max(1, 2.5*F_pair / F_pauli) — the ZBL fit is kept wherever it
already dominates).  Pairs without probe data keep the ZBL fit (the design's
no-probe default).

Output: data/pauli/probe_calibration.json, auto-loaded by PauliRepulsion.

Usage: python scripts/calibrate_pauli_wall.py [--device cuda:1] [--comps ...]
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
from materialgen.core.pauli import PauliRepulsion, pair_key

OUT_PATH = os.path.join(_ROOT, "data", "pauli", "probe_calibration.json")
OFFSET = 0.1      # Angstrom: differencing window above r_wall
CRITERION = 2.5   # F_Pauli(r_wall) >= CRITERION * F_pair


def closest_pair(atoms, Zi, Zj):
    """(i, j, u, d0) for the min-image-closest pair with elements (Zi, Zj);
    u is the unit vector from i to j under the minimum-image convention."""
    numbers = atoms.get_atomic_numbers()
    pos = atoms.positions
    L = np.asarray(atoms.cell[:])
    d0, best, u_best = np.inf, None, None
    for i in range(len(atoms)):
        if numbers[i] != Zi:
            continue
        for j in range(len(atoms)):
            if i == j or numbers[j] != Zj:
                continue
            du = pos[j] - pos[i]
            du = du - L.T @ np.round(np.linalg.solve(L.T, du))  # min-image
            d = np.linalg.norm(du)
            if d < d0:
                d0, best, u_best = d, (i, j), du / d
    return best[0], best[1], u_best, d0


def pair_force(atoms, calc, i, j, u, d_target):
    """Move atom j to distance d_target from i along u; return F[j] . u.

    Positive = pushing j away from i (repulsive); negative = pulling in."""
    pos = atoms.positions.copy()
    pos[j] = pos[i] + d_target * u
    atoms.positions = pos
    atoms.wrap()
    F = atoms.get_forces()
    return float(np.dot(F[j], u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--comps", nargs="+", default=rp.COMPOSITIONS_20)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    from ase import Atoms
    calc = mace_mp(model="medium", device=args.device, default_dtype="float32")

    refs = rp.load_reference_structures(args.comps)
    radii = dict(PauliRepulsion().radii)          # Cordero 2008

    # F_pair per (pair-key, composition) -> max over comps per pair.
    f_pair: dict = {}
    for comp in args.comps:
        r = refs[comp]
        atoms = Atoms(numbers=r["numbers"], positions=r["frac_coords"] @ r["lattice"],
                      cell=r["lattice"], pbc=True)
        atoms.calc = calc
        numbers = atoms.get_atomic_numbers()
        seen = set()
        for Zi in sorted(set(numbers.tolist())):
            for Zj in sorted(set(numbers.tolist())):
                if Zi >= Zj:
                    continue
                key = pair_key(Zi, Zj)
                if key in seen:
                    continue
                seen.add(key)
                i, j, u, d0 = closest_pair(atoms, Zi, Zj)
                r_wall = 0.75 * (radii[Zi] + radii[Zj])
                # Two-point differencing isolates the pair force from the
                # background exerted by the other (fixed) atoms.
                f1 = pair_force(atoms, calc, i, j, u, r_wall)
                f2 = pair_force(atoms, calc, i, j, u, r_wall + OFFSET)
                f = abs(f1 - f2)
                f_pair.setdefault(key, []).append((comp, f))
                print(f"{comp:8} {Zi:3}-{Zj:<3} r_wall={r_wall:5.2f} A  "
                      f"F_pair={f:7.2f} eV/A  ({f1:+.1f}/{f2:+.1f})", flush=True)

    # Wall entry force of the current (unscaled) potential per pair.
    pauli = PauliRepulsion(deep_wall=False)        # reference: ZBL-fitted form
    box = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
    out = {}
    for key, entries in f_pair.items():
        f_max = max(f for _, f in entries)          # conservative over comps
        Zi, Zj = (int(z) for z in key.split("-"))
        r_wall = 0.75 * (radii[Zi] + radii[Zj])
        atoms2 = Atoms(numbers=[Zi, Zj], positions=[[0, 0, 0], [r_wall, 0, 0]],
                       cell=box, pbc=True)
        _, forces, _ = pauli.energy_forces_stress(atoms2)
        f_pauli = abs(float(forces[1, 0]))          # along +x, at theta=0.5
        scale = max(1.0, CRITERION * f_max / f_pauli if f_pauli > 0 else 1.0)
        out[key] = {
            "F_nnp_wall": round(f_max, 4),
            "F_pauli_wall_unscaled": round(f_pauli, 4),
            "scale": round(scale, 4),
            "comps": [c for c, _ in entries],
        }
        print(f"{key:8} F_pair_max={f_max:7.2f}  F_pauli={f_pauli:7.2f}  "
              f"scale={scale:7.2f}")

    payload = {
        "meta": {
            "model": "mace-mp-0-medium",
            "device": args.device,
            "criterion": f"F_Pauli(r_wall) >= {CRITERION} * |F_NNP(r_wall)|",
            "r_wall": "0.75 * (R_i + R_j)",
            "differencing_offset_A": OFFSET,
            "note": "pairs absent from this table keep the ZBL-fitted A (no-probe default)",
        },
        "pairs": out,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {OUT_PATH} ({len(out)} pairs)")


if __name__ == "__main__":
    main()
