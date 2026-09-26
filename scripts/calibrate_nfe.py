"""
Phase 0.4 — NFE speed calibration.

Times one NFE (energy + forces evaluation) for each NNP backend at
N = 20 / 40 / 100 atoms, float32, on a single A100. Results are merged
into results/calibration/nfe_timing.json; the §6.1 scan budget is decided
from the measured eSEN rate (plan: eSEN <= 0.1 s/NFE -> keep eSEN for the
large sigma scan, else downgrade large scans to MACE).

Usage:
    python scripts/calibrate_nfe.py --nnp mace    # in the materialgen env
    python scripts/calibrate_nfe.py --nnp chgnet  # in the materialgen env
    python scripts/calibrate_nfe.py --nnp esen    # in the materialgen-esen env

Dependencies:
    numpy, ase, torch; backend-specific packages (mace-torch / chgnet / fairchem-core)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results" / "calibration" / "nfe_timing.json"

SIZES = [20, 40, 100]
WARMUP = 3
REPEAT = 10


_REPS = {20: (5, 2, 1), 40: (5, 2, 2), 100: (5, 5, 2)}


def make_structure(n_target: int, seed: int = 0):
    """Rattled NaCl rocksalt supercell with exactly n_target atoms.

    ase bulk("NaCl", "rocksalt") returns the 2-atom primitive cell.
    """
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64) * _REPS[n_target]
    assert len(atoms) == n_target
    rng = np.random.RandomState(seed)
    atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    return atoms


def load_calculator(nnp: str):
    if nnp == "esen":
        from materialgen.nnp.esen import load_esen_calculator

        return load_esen_calculator(device="cuda")
    if nnp == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium", device="cuda", default_dtype="float32")
    if nnp == "chgnet":
        from chgnet.model import CHGNet, CHGNetCalculator

        return CHGNetCalculator(CHGNet.load(use_device="cuda"))
    raise ValueError(nnp)


def time_nfe(nnp: str) -> dict:
    import torch

    calc = load_calculator(nnp)
    out = {}
    for n in SIZES:
        atoms = make_structure(n)
        atoms.calc = calc
        base_pos = atoms.get_positions()
        rng = np.random.RandomState(1)

        def one_nfe(k):
            # perturb positions so ASE cannot serve cached results
            atoms.set_positions(base_pos + rng.normal(0, 1e-4, base_pos.shape))
            atoms.get_potential_energy()
            atoms.get_forces()

        for k in range(WARMUP):
            one_nfe(k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(REPEAT):
            one_nfe(k)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / REPEAT
        out[str(len(atoms))] = {
            "seconds_per_nfe": dt,
            "n_atoms": len(atoms),
            "repeat": REPEAT,
        }
        print(f"  {nnp:6s} N={len(atoms):4d}: {dt*1000:8.1f} ms/NFE")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnp", required=True, choices=["esen", "mace", "chgnet"])
    args = parser.parse_args()

    import torch

    print(f"Calibrating {args.nnp} on {torch.cuda.get_device_name(0)}")
    timings = time_nfe(args.nnp)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    record = {}
    if RESULTS.exists():
        record = json.loads(RESULTS.read_text())
    record[args.nnp] = {
        "device": torch.cuda.get_device_name(0),
        "dtype": "float32",
        "timings": timings,
    }
    RESULTS.write_text(json.dumps(record, indent=2))
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
