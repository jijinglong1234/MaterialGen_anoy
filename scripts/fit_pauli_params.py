"""
fit_pauli_params.py — generate the Pauli repulsion parameter table.

Derives (A, B) for every element pair from the ZBL universal screening
potential via the closed-form fit in materialgen.core.pauli.fit_exp_params,
and writes:
    data/pauli/pauli_params.json   — {"pairs": {"Zi-Zj": {"A", "B"}}}
    data/pauli/covalent_radii.json — Cordero et al. (2008) radii used

Also prints a validation table for the representative element pairs listed
in the paper outline (Table: pauli-params), for order-of-magnitude comparison
(A ~ 1e3-1e5 eV, B ~ 3-6 Angstrom^-1).

Usage:
    python scripts/fit_pauli_params.py [--out data/pauli]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from materialgen.core.pauli import fit_exp_params, pair_key, zbl_energy, zbl_force
from materialgen.utils.constants import COVALENT_RADII

# Representative pairs from the paper outline (section 6.0.4, Table pauli-params)
_REPRESENTATIVE = [
    (6, 6, "C-C"), (14, 14, "Si-Si"), (8, 8, "O-O"), (14, 8, "Si-O"),
    (3, 8, "Li-O"), (3, 16, "Li-S"), (22, 8, "Ti-O"), (56, 8, "Ba-O"),
    (20, 8, "Ca-O"), (38, 8, "Sr-O"), (11, 17, "Na-Cl"), (40, 8, "Zr-O"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "pauli"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    elements = sorted(COVALENT_RADII.keys())
    pairs = {}
    for a_i, Zi in enumerate(elements):
        for Zj in elements[a_i:]:
            A, B = fit_exp_params(Zi, Zj)
            pairs[pair_key(Zi, Zj)] = {"A": A, "B": B}

    params_file = out_dir / "pauli_params.json"
    with open(params_file, "w") as f:
        json.dump(
            {
                "description": "Pauli repulsion A*exp(-B*r) fitted to ZBL at r* = 0.75*(R_i + R_j) - 0.1 A",
                "units": {"A": "eV", "B": "Angstrom^-1"},
                "pairs": pairs,
            },
            f, indent=1,
        )

    radii_file = out_dir / "covalent_radii.json"
    with open(radii_file, "w") as f:
        json.dump(
            {
                "description": "Covalent radii (single bond), Cordero et al., Dalton Trans. 2008",
                "units": "Angstrom",
                "radii": {str(z): r for z, r in COVALENT_RADII.items()},
            },
            f, indent=1,
        )

    print(f"Wrote {len(pairs)} element pairs -> {params_file}")
    print(f"Wrote {len(elements)} radii -> {radii_file}")

    # ---- Validation table ----
    print("\nRepresentative pairs (fit vs ZBL consistency at r* = R_sum - 0.1 A):")
    print(f"{'pair':>8} {'A (eV)':>12} {'B (A^-1)':>9} {'r* (A)':>7} "
          f"{'U_ZBL(r*)':>10} {'U_exp(r*)':>10} {'F_ZBL(r*)':>10} {'F_exp(r*)':>10}")
    for Zi, Zj, name in _REPRESENTATIVE:
        A, B = fit_exp_params(Zi, Zj)
        r_star = COVALENT_RADII[Zi] + COVALENT_RADII[Zj] - 0.1
        u_zbl = zbl_energy(Zi, Zj, r_star)
        f_zbl = zbl_force(Zi, Zj, r_star)
        u_exp = A * pow(2.718281828459045, -B * r_star)
        f_exp = A * B * pow(2.718281828459045, -B * r_star)
        print(f"{name:>8} {A:>12.3e} {B:>9.3f} {r_star:>7.3f} "
              f"{u_zbl:>10.3f} {u_exp:>10.3f} {f_zbl:>10.2f} {f_exp:>10.2f}")


if __name__ == "__main__":
    main()
