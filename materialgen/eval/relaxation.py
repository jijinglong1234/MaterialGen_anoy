"""
Structure relaxation pipeline for generated crystal structures.

Relaxation is where the NNP surface is *used as a minimizer* rather than a
score: the metric chain reports E_form/E_hull of the relaxed structure, and
Delta E = E_relaxed - E_relaxed(original CIF) is the energy-improvement metric
of paper section 6.0.7.

    relax_structure(structure, calculator, fmax=0.05, max_steps=500,
                    cell_relax=False, optimizer="FIRE") -> CrystalStructure
        Minimizes the potential energy on the calculator's surface.  The result
        carries its diagnostics in ``properties``: energy (eV), fmax (eV/A),
        converged (bool), n_steps (int), optimizer.

    batch_relax(structures, calculator, max_workers=4, **kwargs) -> list[CrystalStructure]
        Serial by default; set max_workers > 1 to relax in a process pool.  An
        ASE calculator is not picklable in general, so a worker factory
        (``calculator_factory``) is required for parallel runs.

    RelaxationConfig
        Dataclass of the parameters above.

Not implemented: "dft" and "two_stage" relaxers (Appendix DFT pipeline is a
separate Phase-5 activity, not part of the generation benchmark).

Dependencies:
    numpy, ase (optimize, filters)
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..core.crystal import CrystalStructure

_FMAX_DEFAULT = 0.05   # eV/A, paper section 6.0.7 (Delta E definition)


@dataclass
class RelaxationConfig:
    """Relaxation parameters (paper section 6.0.7 / Appendix DFT pipeline)."""

    optimizer: str = "FIRE"
    fmax: float = _FMAX_DEFAULT
    max_steps: int = 500
    cell_relax: bool = False
    pressure: float = 0.0          # GPa, used when cell_relax is True


def _resolve_config(config: Optional[RelaxationConfig], kwargs: dict) -> RelaxationConfig:
    """A config wins over keyword overrides; both may be omitted."""
    if config is not None and kwargs:
        raise ValueError("pass either a RelaxationConfig or keyword parameters, not both")
    if config is not None:
        return config
    return RelaxationConfig(**kwargs) if kwargs else RelaxationConfig()


def _make_optimizer(atoms, config: RelaxationConfig):
    from ase import optimize

    name = config.optimizer.upper()
    if name == "FIRE":
        cls = optimize.FIRE
    elif name in ("BFGS", "LBFGS", "CG"):
        cls = getattr(optimize, name)
    else:
        raise ValueError(f"unsupported optimizer {config.optimizer!r}")

    target = atoms
    if config.cell_relax:
        from ase.filters import ExpCellFilter

        target = ExpCellFilter(atoms, scalar_pressure=config.pressure * 0.1)  # GPa -> eV/A^3
    return cls(target, logfile=None)


def relax_structure(structure, calculator, config: Optional[RelaxationConfig] = None,
                    **kwargs) -> CrystalStructure:
    """Relax on the calculator's potential energy surface.

    Returns a CrystalStructure whose ``properties`` carry ``energy``, ``fmax``,
    ``converged``, ``n_steps`` and ``optimizer``.  A structure that fails to
    converge is returned at its best-so-far geometry with ``converged=False``
    -- per the statistical protocol, it is not dropped.
    """
    config = _resolve_config(config, kwargs)
    atoms = structure.ase_atoms
    atoms.calc = calculator
    opt = _make_optimizer(atoms, config)

    converged = False
    n_steps = 0
    try:
        for _ in range(config.max_steps):
            opt.step()
            n_steps += 1
            forces = atoms.get_forces()
            if np.linalg.norm(forces, axis=1).max() < config.fmax:
                converged = True
                break
    except Exception as exc:
        print(f"WARN relaxation aborted after {n_steps} steps: {exc}")

    try:
        energy = float(atoms.get_potential_energy())
        fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    except Exception:
        energy, fmax = float("nan"), float("nan")
    finally:
        atoms.calc = None

    out = CrystalStructure(atoms, dict(getattr(structure, "properties", {}) or {}))
    out.properties.update({"energy": energy, "fmax": fmax, "converged": converged,
                           "n_steps": n_steps, "optimizer": config.optimizer,
                           "relax_fmax_target": config.fmax})
    return out


def batch_relax(structures: Sequence, calculator=None, max_workers: int = 4,
                calculator_factory: Optional[Callable[[], object]] = None,
                config: Optional[RelaxationConfig] = None, **kwargs
                ) -> list[CrystalStructure]:
    """Relax a list of structures.  ``max_workers > 1`` needs a factory.

    A live ASE calculator cannot be shared across processes (and MACE/eSEN
    calculators are not reliably picklable), so parallel mode builds one
    calculator per worker through ``calculator_factory``.
    """
    config = _resolve_config(config, kwargs)
    if max_workers <= 1 or calculator_factory is None:
        if calculator is None and calculator_factory is not None:
            calculator = calculator_factory()
        if calculator is None:
            raise ValueError("relax_structure needs a calculator or a calculator_factory")
        return [relax_structure(s, calculator, config) for s in structures]

    from multiprocessing import Pool

    def _one(s):
        return relax_structure(s, calculator_factory(), config)

    with Pool(max_workers) as pool:
        return pool.map(_one, list(structures))


def compute_relaxation_trajectory(structure, calculator, config: Optional[RelaxationConfig] = None,
                                  **kwargs) -> dict:
    """Energy/force trace of one relaxation (diagnostics only)."""
    config = _resolve_config(config, kwargs)
    atoms = structure.ase_atoms
    atoms.calc = calculator
    opt = _make_optimizer(atoms, config)
    energies, fmaxes = [], []
    try:
        for _ in range(config.max_steps):
            opt.step()
            energies.append(float(atoms.get_potential_energy()))
            fmaxes.append(float(np.linalg.norm(atoms.get_forces(), axis=1).max()))
            if fmaxes[-1] < config.fmax:
                break
    finally:
        atoms.calc = None
    return {"energies": energies, "max_forces": fmaxes,
            "steps_to_convergence": (len(fmaxes) if fmaxes and fmaxes[-1] < config.fmax else None)}
