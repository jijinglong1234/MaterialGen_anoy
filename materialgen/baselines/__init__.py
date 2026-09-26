"""
Baseline samplers and learned-model interfaces.

Physics baselines (paper section 6.0.6):
    run_simulated_annealing — Langevin-MD cooling on any ASE calculator
    run_basin_hopping — Metropolis + FIRE relaxation loop

Learned-model score interfaces (DiffCSP / MatterGen / CDVAE) are
design-stage; see the module docstrings.
"""

from .md_annealing import run_simulated_annealing, make_random_initial, MDAnnealResult
from .basin_hopping import run_basin_hopping, BasinHoppingResult

__all__ = [
    "run_simulated_annealing", "make_random_initial", "MDAnnealResult",
    "run_basin_hopping", "BasinHoppingResult",
]
