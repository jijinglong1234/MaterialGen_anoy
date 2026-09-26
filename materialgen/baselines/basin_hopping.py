"""
Basin Hopping baseline (paper section 6.0.6).

Functionality:
    Manual Basin Hopping loop matching the paper protocol:
    - Perturbation: random displacement of ALL atoms, sigma_disp = 0.3 Angstrom.
    - Local relaxation: FIRE (max 200 steps, fmax = 0.05 eV/Angstrom).
    - Acceptance: Metropolis at T_BH = 1000 K.
    - Tracks the lowest-energy relaxed structure.
    - NFE accounting: accumulated FIRE steps per hop.

Dependencies:
    numpy, ase (FIRE)
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.crystal import CrystalStructure
from ..utils.constants import KB


@dataclass
class BasinHoppingResult:
    """Result of a Basin Hopping run."""
    best_structure: CrystalStructure
    best_energy: float
    energies: List[float] = field(default_factory=list)   # accepted energies
    n_hops: int = 0
    n_accepted: int = 0
    nfe: int = 0                                          # total FIRE steps


def run_basin_hopping(
    initial: CrystalStructure,
    calculator,
    n_hops: int = 100,
    T_bh: float = 1000.0,
    sigma_disp: float = 0.3,
    relax_fmax: float = 0.05,
    relax_max_steps: int = 200,
    seed: Optional[int] = None,
) -> BasinHoppingResult:
    """Run Basin Hopping on the given calculator."""
    from ase.optimize import FIRE

    rng = np.random.RandomState(seed)

    def _relax(crystal: CrystalStructure):
        atoms = crystal.ase_atoms
        atoms.calc = calculator
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=relax_fmax, steps=relax_max_steps)
        return atoms, atoms.get_potential_energy(), opt.get_number_of_steps()

    current_atoms, current_e, nfe0 = _relax(initial)
    nfe = nfe0
    best_atoms, best_energy = current_atoms.copy(), current_e
    energies: List[float] = [current_e]
    n_accepted = 0

    for _ in range(n_hops):
        trial = CrystalStructure(current_atoms).add_noise_to_cart(sigma_disp, seed=rng.randint(2**31))
        trial_atoms, trial_e, nfe_trial = _relax(trial)
        nfe += nfe_trial

        dE = trial_e - current_e
        if dE <= 0 or rng.rand() < np.exp(-dE / (KB * T_bh)):
            current_atoms, current_e = trial_atoms, trial_e
            n_accepted += 1
            energies.append(current_e)
            if current_e < best_energy:
                best_atoms, best_energy = trial_atoms.copy(), trial_e

    return BasinHoppingResult(
        best_structure=CrystalStructure(best_atoms),
        best_energy=float(best_energy),
        energies=energies,
        n_hops=n_hops,
        n_accepted=n_accepted,
        nfe=int(nfe),
    )
