"""Empirical lower edge of the training distribution of interatomic distances.

Companion to Fig. ``fig:training-gap`` (docs/paper/fig_training_gap_brief.md).
Reads the MP-20 CIF corpus, forms every minimum-image pair distance, normalises it
by the sum of single-bond covalent radii (Cordero et al.),

    u = d_ij / (R_i^cov + R_j^cov)

and reports the lower edge of the distribution together with a histogram.

Establishes: the deepest pair in the whole corpus sits at u = 0.709 and no pair
lies below u = 0.70, while the completeness boundary of the paper is at
u = f_typ = 0.75 with only 0.014% of pairs below it.

Structures containing elements absent from the covalent-radii table (the 58-71
lanthanides) are skipped and counted; the caption of the figure must carry the
"elements with tabulated covalent radii" caveat accordingly.

Measured with pymatgen 2023.7.17:
    train   : 17,286 structures, 189,164 atoms, 1,186,442 pairs, min u = 0.709
    val+test: 11,403 structures, 124,171 atoms,   773,062 pairs, min u = 0.717

Usage:  python scripts/analyze_training_distance_edge.py [--out /tmp/pairhist.json]
"""
import argparse
import csv
import json

import numpy as np
from pymatgen.core import Structure

CIFS = ["data/mp_20/train.csv", "data/mp_20/val.csv", "data/mp_20/test.csv"]
RADII = "data/pauli/covalent_radii.json"
GRID = np.arange(0.30, 6.0001, 0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/pairhist_mp20.json")
    ap.add_argument("--max-atoms", type=int, default=200)
    args = ap.parse_args()

    rcov = {int(k): float(v)
            for k, v in json.load(open(RADII))["radii"].items()}

    us, mins = [], []
    hist = np.zeros(len(GRID) - 1, dtype=np.int64)
    n_struct = n_atoms = n_skip = 0

    for fn in CIFS:
        with open(fn, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s = Structure.from_str(row["cif"], fmt="cif")
                except Exception:
                    continue
                n = len(s)
                if n < 2 or n > args.max_atoms:
                    continue
                z = np.array([sp.Z for sp in s.species])
                if any(int(zz) not in rcov for zz in z):
                    n_skip += 1
                    continue
                r = np.array([rcov[int(zz)] for zz in z])
                iu = np.triu_indices(n, 1)
                d = s.lattice.get_all_distances(s.frac_coords, s.frac_coords)[iu]
                u = d / (r[iu[0]] + r[iu[1]])
                us.append(u.astype(np.float32))
                mins.append(float(u.min()))
                hist += np.histogram(u, bins=GRID)[0]
                n_atoms += n
                n_struct += 1
        print(f"  after {fn}: {n_struct} structures, {n_atoms} atoms", flush=True)

    u = np.concatenate(us)
    mins = np.array(mins)
    out = {
        "n_struct": n_struct, "n_atoms": n_atoms, "n_pairs": int(u.size),
        "n_skipped_missing_radius": n_skip,
        "min_u": float(u.min()), "u_median": float(np.median(u)),
        "count_below": {str(t): int((u < t).sum())
                        for t in (0.5, 0.6, 0.70, 0.75, 0.8, 0.85, 0.9, 1.0)},
        "struct_frac_min_below": {str(t): float((mins < t).mean())
                                  for t in (0.75, 0.8, 0.9, 1.0)},
        "per_structure_min_u_quantiles": {str(q): float(np.quantile(mins, q))
                                          for q in (0.0, 0.001, 0.01, 0.05, 0.25, 0.5)},
        "grid": GRID.tolist(), "hist": hist.tolist(),
    }
    json.dump(out, open(args.out, "w"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "hist")},
                     indent=1))


if __name__ == "__main__":
    main()
