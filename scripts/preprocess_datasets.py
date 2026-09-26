"""
Phase 0.3 — dataset preprocessing.

For each CDVAE benchmark (mp_20, carbon_24, perov_5) and each split:
    1. parse CIF -> CrystalStructure
    2. primitive-cell standardization (symprec=0.1)
    3. supercell expansion until every perpendicular cell height >= 10 Angstrom
       (minimum-image convention unambiguous beyond 10 A)

Outputs (per dataset/split) under data/processed/<dataset>/:
    <split>.pkl          list of dicts {numbers, frac_coords, lattice, metadata}
    <split>_stats.json   aggregate statistics

Dependencies:
    numpy, pandas, pymatgen, ase
    materialgen.data.datasets
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from materialgen.data.datasets import (
    _DATA_ROOT,
    load_cdvae_csv,
    preprocess_structure,
)

OUT_ROOT = _DATA_ROOT / "processed"


def process_split(name: str, split: str) -> dict:
    ds = load_cdvae_csv(name, split)
    records = []
    n_fail = 0
    for struct, meta in zip(ds.structures, ds.metadata):
        rec = {
            "numbers": np.asarray(struct.atomic_numbers),
            "frac_coords": np.asarray(struct.frac_coords),
            "lattice": np.asarray(struct.lattice),
            "metadata": meta,
        }
        try:
            proc = preprocess_structure(struct)
            rec["supercell"] = {
                "numbers": np.asarray(proc.atomic_numbers),
                "frac_coords": np.asarray(proc.frac_coords),
                "lattice": np.asarray(proc.lattice),
            }
        except Exception:
            n_fail += 1
            rec["supercell"] = None
        records.append(rec)

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}.pkl", "wb") as f:
        pickle.dump(records, f)

    nats = np.array([len(r["numbers"]) for r in records])
    stats = {
        "dataset": name,
        "split": split,
        "num_structures": len(records),
        "preprocess_failures": n_fail,
        "atoms_min": int(nats.min()),
        "atoms_max": int(nats.max()),
        "atoms_mean": float(nats.mean()),
        "chemical_systems": ds.chemical_systems,
    }
    (out_dir / f"{split}_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["mp_20", "carbon_24", "perov_5"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    for name in args.datasets:
        for split in args.splits:
            stats = process_split(name, split)
            print(f"{name}/{split}: {stats['num_structures']} structures, "
                  f"failures={stats['preprocess_failures']}")


if __name__ == "__main__":
    main()
