"""
Phase 0.3 — precompute NNP reference energies/forces on the test sets.

For each dataset (mp_20, carbon_24, perov_5) test split, evaluate the raw
(as-published) structures with one NNP backend and store per-structure
energy / forces / stress alongside metadata. Reference values are used by
later phases for E_hull, match-rate and trajectory diagnostics.

Outputs:
    results/reference/<dataset>/<nnp>_test.pkl
        list of dicts {numbers, frac_coords, lattice, metadata,
                       energy, forces, stress, wall_time_s}

Usage:
    python scripts/precompute_reference.py --nnp mace --datasets mp_20          # main env
    python scripts/precompute_reference.py --nnp esen --datasets mp_20          # esen env
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from materialgen.core.crystal import CrystalStructure
from materialgen.data.datasets import _DATA_ROOT, load_cdvae_csv

OUT = Path(__file__).resolve().parents[1] / "results" / "reference"


def load_calculator(nnp: str, device: str):
    if nnp == "esen":
        from materialgen.nnp.esen import load_esen_calculator

        return load_esen_calculator(device=device)
    if nnp == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium", device=device, default_dtype="float32")
    if nnp == "chgnet":
        from chgnet.model import CHGNet, CHGNetCalculator

        return CHGNetCalculator(CHGNet.load(use_device=device))
    raise ValueError(nnp)


def run(nnp: str, dataset: str, device: str, limit: int | None = None):
    out_dir = OUT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{nnp}_test.pkl"
    done = []
    if out_path.exists():
        with open(out_path, "rb") as f:
            done = pickle.load(f)

    ds = load_cdvae_csv(dataset, "test", max_structures=limit)
    calc = load_calculator(nnp, device)

    results = list(done)
    for i, (struct, meta) in enumerate(zip(ds.structures, ds.metadata)):
        if i < len(done):
            continue
        atoms = struct.ase_atoms
        atoms.calc = calc
        t0 = time.time()
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces())
            try:
                stress = np.asarray(atoms.get_stress())
            except Exception:
                stress = None
            rec = {
                "numbers": np.asarray(struct.atomic_numbers),
                "frac_coords": np.asarray(struct.frac_coords),
                "lattice": np.asarray(struct.lattice),
                "metadata": meta,
                "energy": energy,
                "forces": forces,
                "stress": stress,
                "wall_time_s": time.time() - t0,
            }
        except Exception as exc:
            rec = {"metadata": meta, "error": str(exc),
                   "numbers": np.asarray(struct.atomic_numbers),
                   "frac_coords": np.asarray(struct.frac_coords),
                   "lattice": np.asarray(struct.lattice)}
        results.append(rec)
        if (i + 1) % 200 == 0:
            with open(out_path, "wb") as f:
                pickle.dump(results, f)
            print(f"{dataset}/{nnp}: {i+1}/{len(ds)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    n_err = sum(1 for r in results if "error" in r)
    print(f"DONE {dataset}/{nnp}: {len(results)} structures, {n_err} errors -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnp", required=True, choices=["esen", "mace", "chgnet"])
    parser.add_argument("--datasets", nargs="+",
                        default=["mp_20", "carbon_24", "perov_5"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for dataset in args.datasets:
        run(args.nnp, dataset, args.device, args.limit)


if __name__ == "__main__":
    main()
