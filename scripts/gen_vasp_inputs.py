"""
Phase 0.3 — VASP input generator (two template sets).

Templates:
    pbe_u_mp    pymatgen MPRelaxSet  — GGA-PBE + U (MP2020-compatible),
                ENCUT 520 eV, k-point density 1000 per Angstrom^-3 of
                reciprocal cell (pymatgen "reciprocal_density" = kppvol,
                NOT per reciprocal atom; actual grid depends on cell size).
    r2scan_omat pymatgen MPScanRelaxSet — r2SCAN (aligned with the OMat24
                reference protocol), ENCUT 700 eV, same k-point rule.

Both use NSW=0 (single point; Tier-1 of Phase 5) unless --relax is given
(ISIF=3, NSW=150 for Tier-2 full relaxation).

POTCARs are proprietary: the generator writes the element-ordered spec and
concatenates from $VASP_POT_PBE if available, otherwise leaves a POTCAR.spec
note listing the required potentials.

Usage:
    python scripts/gen_vasp_inputs.py --structure data/processed/... --out vasp_runs/t1/xxx --set pbe_u_mp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

KPPA = 1000  # pymatgen "reciprocal_density": k-points per Angstrom^-3 of
             # reciprocal cell volume (kppvol), NOT kppra. See vasp_templates/README.md.


def make_input_set(name: str, structure, relax: bool):
    user_incar = {}
    if name == "pbe_u_mp":
        from pymatgen.io.vasp.sets import MPRelaxSet

        user_incar["ENCUT"] = 520
        cls = MPRelaxSet
    elif name == "r2scan_omat":
        from pymatgen.io.vasp.sets import MPScanRelaxSet

        user_incar["ENCUT"] = 700
        cls = MPScanRelaxSet
    else:
        raise ValueError(name)

    if not relax:
        user_incar.update({"NSW": 0, "IBRION": -1, "ISIF": 2, "LCHARG": False})
    else:
        user_incar.update({"NSW": 150, "IBRION": 2, "ISIF": 3})

    return cls(structure,
               user_incar_settings=user_incar,
               user_kpoints_settings={"reciprocal_density": KPPA})


def write_inputs(structure, out_dir: Path, set_name: str, relax: bool = False):
    vset = make_input_set(set_name, structure, relax)
    out_dir.mkdir(parents=True, exist_ok=True)
    vset.write_input(str(out_dir), potcar_spec=True)

    # concatenate real POTCARs if the library is present
    pot_root = os.environ.get("VASP_POT_PBE")
    spec = (out_dir / "POTCAR.spec").read_text().split()
    if pot_root:
        parts = []
        for sym in spec:
            pot_path = Path(pot_root) / sym / "POTCAR"
            if not pot_path.exists():
                raise FileNotFoundError(f"missing POTCAR: {pot_path}")
            parts.append(pot_path.read_text())
        (out_dir / "POTCAR").write_text("".join(parts))
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True, help="CIF file of the structure")
    parser.add_argument("--out", required=True, help="output run directory")
    parser.add_argument("--set", default="pbe_u_mp",
                        choices=["pbe_u_mp", "r2scan_omat"])
    parser.add_argument("--relax", action="store_true",
                        help="Tier-2 full relaxation (ISIF=3); default single point")
    args = parser.parse_args()

    from pymatgen.core import Structure

    structure = Structure.from_file(args.cif)
    write_inputs(structure, Path(args.out), args.set, args.relax)
    print(f"wrote {args.set} inputs to {args.out}")


if __name__ == "__main__":
    main()
