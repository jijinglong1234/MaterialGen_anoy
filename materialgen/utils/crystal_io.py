"""
Crystal structure I/O and format conversion.

Implemented here: the CrystalStructure <-> pymatgen.Structure converters used
by the evaluation metric chain and the third-party output adapters, plus CIF
and JSON read/write and directory batch reads.

Not implemented (no current consumer; add when a caller appears):
    POSCAR / extxyz read+write.

Dependencies:
    numpy, ase, pymatgen (Structure, CifParser, CifWriter)
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..core.crystal import CrystalStructure


# ---------------------------------------------------------------------------
# pymatgen <-> CrystalStructure
# ---------------------------------------------------------------------------

def to_pymatgen(crystal):
    """CrystalStructure -> pymatgen.core.Structure (no symmetry reduction).

    A pymatgen Structure is passed through unchanged, so every function that
    takes "a structure" can accept either flavour -- third-party generators
    (DiffCSP, CDVAE) hand back pymatgen objects, and the metric chain must not
    silently degrade on them (a bare isinstance guard here is what keeps
    ``detect_spacegroup`` from returning None for such input).
    """
    from pymatgen.core import Lattice, Structure

    if isinstance(crystal, Structure):
        return crystal
    if not isinstance(crystal, CrystalStructure):
        # Third-party callers hand over lists/slices/arrays by mistake; say so
        # here rather than failing on a missing `.lattice` attribute.
        raise TypeError(f"expected a CrystalStructure or pymatgen Structure, "
                        f"got {type(crystal).__name__}")
    return Structure(
        Lattice(np.asarray(crystal.lattice)),
        [int(z) for z in crystal.atomic_numbers],
        np.asarray(crystal.frac_coords),
    )


def from_pymatgen(structure, properties: Optional[dict] = None) -> CrystalStructure:
    """pymatgen.core.Structure -> CrystalStructure (CrystalStructure passes through)."""
    if isinstance(structure, CrystalStructure):
        return structure
    return CrystalStructure.from_frac_coords(
        np.array([site.specie.Z for site in structure]),
        np.array(structure.frac_coords),
        np.array(structure.lattice.matrix),
        properties=properties,
    )


def structure_key(crystal: CrystalStructure) -> tuple:
    """(species, positions) hashable key, for de-duplicating ensembles."""
    return (tuple(int(z) for z in crystal.atomic_numbers),
            tuple(np.round(np.asarray(crystal.frac_coords).ravel() % 1.0, 6)))


# ---------------------------------------------------------------------------
# CIF
# ---------------------------------------------------------------------------

def read_cif(filepath: str | Path, primitive: bool = False) -> CrystalStructure:
    """Read the first structure of a CIF file."""
    from pymatgen.io.cif import CifParser

    structures = CifParser(str(filepath)).parse_structures(primitive=primitive)
    if not structures:
        raise ValueError(f"no structure parsed from {filepath}")
    return from_pymatgen(structures[0])


def write_cif(crystal: CrystalStructure, filepath: str | Path, symprec: float = 0.1) -> Path:
    """Write a CIF (P1, explicit atoms) and return the path."""
    from pymatgen.io.cif import CifWriter

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    CifWriter(to_pymatgen(crystal), symprec=symprec).write_file(str(path))
    return path


# ---------------------------------------------------------------------------
# JSON (CrystalStructure.to_dict / from_dict)
# ---------------------------------------------------------------------------

def read_json(filepath: str | Path) -> CrystalStructure:
    import json

    return CrystalStructure.from_dict(json.loads(Path(filepath).read_text()))


def write_json(crystal: CrystalStructure, filepath: str | Path) -> Path:
    import json

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(crystal.to_dict()))
    return path


# ---------------------------------------------------------------------------
# Directory batch
# ---------------------------------------------------------------------------

def batch_read(directory: str | Path, pattern: str = "*.cif") -> list[CrystalStructure]:
    """Read every matching file in a directory, in sorted-name order."""
    out = []
    for p in sorted(Path(directory).glob(pattern)):
        try:
            out.append(read_cif(p))
        except Exception as exc:  # a malformed CIF must not abort a whole cohort
            print(f"WARN {p}: {exc}")
    return out


def as_pymatgen(structures: Iterable) -> list:
    """Coerce a mixed iterable of CrystalStructure / pymatgen Structure to the latter."""
    from pymatgen.core import Structure

    out = []
    for s in structures:
        out.append(s if isinstance(s, Structure) else to_pymatgen(s))
    return out
