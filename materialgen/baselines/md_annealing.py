"""
MD simulated annealing baseline (paper section 6.0.6).

Functionality:
    Runs Langevin MD with a linear cooling schedule on a given ASE
    calculator (any NNP backend; LJ works for testing):

        T(t): T_max -> T_min linear over n_steps, dt = 1 fs, gamma = 0.01 fs^-1

    Protocol matching the paper:
    - Initial structure: random positions in a 1.5x dilated cell (helper
      ``make_random_initial``) to avoid initial overlaps.
    - The trajectory is segmented (temperature updated per segment).
    - The lowest-energy snapshot is selected and FIRE-relaxed.
    - NFE accounting: nfe_md = MD steps; nfe_relax reported separately.

Dependencies:
    numpy, ase (Langevin MD, FIRE)
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.crystal import CrystalStructure


@dataclass
class MDAnnealResult:
    """Result of one simulated-annealing trajectory."""
    best_structure: CrystalStructure
    best_energy: float
    final_energy: float
    energy_trace: List[float] = field(default_factory=list)
    nfe_md: int = 0
    nfe_relax: int = 0


def make_random_initial(
    atomic_numbers: np.ndarray,
    lattice: np.ndarray,
    dilation: float = 1.5,
    seed: Optional[int] = None,
) -> CrystalStructure:
    """Random fractional positions in a dilated cell (paper protocol)."""
    rng = np.random.RandomState(seed)
    n = len(atomic_numbers)
    frac = rng.rand(n, 3)
    cell = np.array(lattice) * (dilation ** (1.0 / 3.0))
    return CrystalStructure.from_frac_coords(atomic_numbers, frac, cell)


def run_simulated_annealing(
    initial: CrystalStructure,
    calculator,
    T_max: float = 2000.0,
    T_min: float = 100.0,
    n_steps: int = 10000,
    dt_fs: float = 1.0,
    friction: float = 0.01,
    n_segments: int = 20,
    log_interval: int = 50,
    relax: bool = True,
    relax_fmax: float = 0.05,
    relax_max_steps: int = 500,
    seed: Optional[int] = None,
) -> MDAnnealResult:
    """
    Run Langevin-MD simulated annealing on the given calculator.

    Returns the FIRE-relaxed lowest-energy snapshot (if relax=True).
    """
    from ase import units
    from ase.md.langevin import Langevin

    atoms = initial.ase_atoms
    atoms.calc = calculator
    atoms.set_momenta(
        np.random.RandomState(seed).normal(0, 1, (len(atoms), 3))
    )

    energies: List[float] = []
    best_energy = np.inf
    best_atoms = None

    steps_per_seg = max(n_steps // n_segments, 1)
    for seg in range(n_segments):
        T_seg = T_max + (T_min - T_max) * seg / max(n_segments - 1, 1)
        dyn = Langevin(
            atoms, dt_fs * units.fs,
            temperature_K=T_seg, friction=friction,
        )

        def _record(atoms=atoms):
            nonlocal best_energy, best_atoms
            try:
                e = atoms.get_potential_energy()
            except Exception:
                return
            energies.append(e)
            if e < best_energy:
                best_energy = e
                best_atoms = atoms.copy()

        dyn.attach(_record, interval=log_interval)
        dyn.run(steps_per_seg)

    if best_atoms is None:
        best_atoms = atoms.copy()
        best_energy = atoms.get_potential_energy()

    nfe_relax = 0
    if relax:
        from ase.optimize import FIRE
        best_atoms.calc = calculator
        opt = FIRE(best_atoms, logfile=None)
        opt.run(fmax=relax_fmax, steps=relax_max_steps)
        nfe_relax = opt.get_number_of_steps()
        best_energy = best_atoms.get_potential_energy()

    return MDAnnealResult(
        best_structure=CrystalStructure(best_atoms),
        best_energy=float(best_energy),
        final_energy=float(energies[-1]) if energies else float(best_energy),
        energy_trace=energies,
        nfe_md=int(n_steps),
        nfe_relax=int(nfe_relax),
    )
