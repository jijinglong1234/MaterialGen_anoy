"""
Validity metrics for generated crystal structures (paper section 6.0.7).

Aligned with the validity criteria of the DiffCSP (Jiao et al., 2023) and
MatterGen (Jain et al., 2024) literature:

    A generated structure is VALID iff
        (a) correct elemental composition, and
        (b) no atom overlap: d_min >= 0.5 Å (minimum-image convention).

Notes:
    - MatterGen additionally requires charge neutrality; we omit it because
      oxidation-state assignment for arbitrary generated structures is
      unreliable, and DiffCSP does not use it.
    - Composition correctness is vacuous for fixed-composition generators
      (all of ours fix the composition at initialization), but the check is
      kept for Phase 2 benchmark use and for any variable-composition mode.
    - Non-finite energies/forces/positions count as validity failures,
      consistent with the statistical protocol (§6.0.9): diverged
      trajectories contribute zero to Match Rate and Coverage.
    - The stricter short-range criteria (max|F_NNP| > 500 eV/Å, volume
      explosion) belong to the OOD spike rate of §6.1, NOT to validity —
      the two diagnostics are deliberately not interchangeable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.crystal import CrystalStructure

D_MIN_VALID = 0.5  # Å — no atom overlap (DiffCSP / MatterGen validity)


def validity_reasons(
    crystal: CrystalStructure,
    composition: Optional[str] = None,
) -> list[str]:
    """List of validity-failure reasons; empty list == valid.

    Accepts a pymatgen ``Structure`` as well as a ``CrystalStructure``: the
    third-party adapters hand back pymatgen objects, and the checks below use
    ``num_atoms`` / ``chemical_formula`` / ``get_minimum_distance``, of which a
    pymatgen Structure has none -- without the coercion every such structure
    would be reported invalid with an "exception: AttributeError" reason.
    """
    from ..utils.crystal_io import from_pymatgen

    if not isinstance(crystal, CrystalStructure):
        crystal = from_pymatgen(crystal)
    reasons: list[str] = []

    # (a) Composition correctness (vacuous for fixed-composition samplers).
    if composition is not None:
        from pymatgen.core import Composition as PmgComposition

        try:
            got = PmgComposition(crystal.chemical_formula)
            want = PmgComposition(composition)
            if got.reduced_formula != want.reduced_formula:
                reasons.append(
                    f"composition mismatch: {got.reduced_formula} != {want.reduced_formula}"
                )
        except Exception:
            reasons.append(f"composition unparseable: {crystal.chemical_formula!r}")

    # (b) No atom overlap: d_min >= 0.5 Å.
    if crystal.num_atoms >= 2:
        d_min = crystal.get_minimum_distance()
        if not np.isfinite(d_min):
            reasons.append("non-finite d_min (NaN/inf positions)")
        elif d_min < D_MIN_VALID:
            reasons.append(f"atom overlap (d_min={d_min:.3f} Å < {D_MIN_VALID:.1f} Å)")
    else:
        reasons.append("degenerate structure (<2 atoms)")

    return reasons


def is_valid(
    crystal: CrystalStructure,
    composition: Optional[str] = None,
) -> bool:
    """True iff the structure is valid per the DiffCSP/MatterGen definition."""
    return len(validity_reasons(crystal, composition)) == 0
