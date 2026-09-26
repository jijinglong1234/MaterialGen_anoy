"""
Phase 0.3 / Phase 2 — per-chemical-system convex hull entries, all three datasets.

Builds, for every chemical system of a benchmark dataset, the formation-energy
entries needed to reconstruct a pymatgen PhaseDiagram and evaluate
E_above_hull of generated structures.

Output (one pair per dataset):
    data/hulls/<key>_hull_entries.json   { "<system>": {"elements": [...],
                                          "entries": [{"formula", "natoms",
                                                       "e_form_per_atom"}, ...]}, ...}
    data/hulls/<key>_hull_meta.json      provenance + the energy-zero convention
                                          needed to turn a DFT reference energy
                                          into a formation energy (see below)

    key: mp_20 -> "mp20", perov_5 -> "perov5", carbon_24 -> "carbon24"

System key convention: sorted element symbols joined by "-" (e.g. "O-Sr-Ti" for
SrTiO3), matching run_phase1.HullEvaluator and materialgen.eval.stability.

Energy-zero convention (per dataset, recorded in the sidecar meta file)
------------------------------------------------------------------------
The hull is expressed in formation energies referenced to the elemental ground
states, so elemental endmembers are added implicitly at energy 0 when the
diagram is rebuilt (materialgen.eval.stability.HullTable.phase_diagram).

  mp_20      e_form = metadata["formation_energy_per_atom"]  (eV/atom, already
             referenced to MP elemental ground states -- used as-is).
  perov_5    e_form = heat_all / n_atoms, heat_all being per 5-atom cell.
             heat_all is a ~0-anchored stability quantity (mean +1.481 eV/cell,
             0.36% below zero), so the elemental endmembers at 0 are its own
             reference level.  NOT heat_all - heat_ref: heat_ref varies within
             a composition and tracks heat_all (corr 0.971) -- a per-structure
             label, not a hull level.  See build_perov5 for the full evidence.
             Validated: 0 of 18,928 entries fall below the hull built on
             heat_all.
  carbon_24  single element C.  The only phases are the dataset's own carbon
             allotropes, so the hull is the lowest-energy structure and
             e_form = energy_per_atom - min(energy_per_atom over train+val+test)
             (the energy zero is written to the meta file).  Validated:
             10.5% of the dataset sits within 100 meV/atom of the ground state
             (diamond at 10 GPa).

Scope note (deliberate deviation from the paper text): paper_outline_v0.4.tex
and the submission both state that Perov-5 hulls come from "all known phases in
the Materials Project".  That is not achievable and not meaningful here --
Perov-5 spans 8,268 exotic systems (Co-N-O-Tl, Ti-Os-N-O-F, ...) of which MP
contains essentially none, and no MP API access exists for this project.  The
hull is therefore built from the phases the dataset itself provides, per
chemical system; it measures stability relative to all *known phases of the
same chemical system in the same dataset*.  For MP-20 that is exactly the
paper's rule (MP-20's own entries are MP phases).

How degenerate these hulls are (measured)
------------------------------------------------------------------------
A hull rebuilt from one dataset's phases is only as competitive as that
dataset's composition coverage.  Phases per system, and systems holding a
single composition:

    mp_20      23,371 systems, 1.59 phases/system (max 33); 70.2% single-
               composition -> for those, E_hull reduces to the calibrated
               formation energy (element simplex at 0 is the only competitor).
    perov_5     8,268 systems, 1.17 phases/system (max 2); 83.3% single-
               composition -> E_hull is effectively heat_all / n_atoms.
    carbon_24       1 system (elemental C), 1 phase.

So "E_hull < 100 meV/atom" means "on the dataset's own formation-energy scale,
close to or below the best phase the dataset knows for that composition" --
which is what the S.U.N. definition needs, but the *absolute* threshold is
per-dataset, and MP-20 vs Perov-5 rates are not like-for-like.

Do not use MP-20's ``e_above_hull`` CSV column as ground truth
------------------------------------------------------------------------
The column disagrees with a hull rebuilt from the same CSV's
``formation_energy_per_atom`` values for ~38% of MP-20 phases: our value is
*lower*, by up to 80 meV/atom (median disagreement 0; the tail is ~50-80
meV/atom).  It is one-sided, and it concentrates on the elements whose
energies the MP2020 compatibility scheme corrects (Mn -0.014, W -0.014,
Mo -0.012, Li -0.009, Cr -0.008, Fe -0.007 median) while agreeing exactly for
Ge/Si/Mg/Al/Ni (0.000) -- i.e. the column and the formation energies are on two
different MP energy conventions.  The metric chain therefore never reads it:
reference-anchor selection ranks same-composition records by their own
formation energy (which is convention-free), see
materialgen.data.registry.DatasetSpec.calib.

Dependencies:
    pandas, pymatgen (Composition)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = DATA / "hulls"

# dataset dir -> output key
KEYS = {"mp_20": "mp20", "perov_5": "perov5", "carbon_24": "carbon24"}


def _write(key: str, systems: dict, meta: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    entries_path = OUT_DIR / f"{key}_hull_entries.json"
    meta_path = OUT_DIR / f"{key}_hull_meta.json"
    entries_path.write_text(json.dumps(systems, indent=1))
    meta_path.write_text(json.dumps(meta, indent=1))
    n_phases = sum(len(v["entries"]) for v in systems.values())
    print(f"  {key}: {len(systems)} systems, {n_phases} phases -> {entries_path}")


# ---------------------------------------------------------------------------
# MP-20
# ---------------------------------------------------------------------------

def build_mp20() -> None:
    """MP-20: formation energies are given per atom and already referenced to
    the MP elemental ground states, so the CSV column is used verbatim."""
    import pandas as pd
    from pymatgen.core import Composition

    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(DATA / "mp_20" / f"{split}.csv",
                         usecols=["material_id", "pretty_formula",
                                  "formation_energy_per_atom", "elements"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    hulls: dict = {}
    for _, row in df.iterrows():
        comp = Composition(row["pretty_formula"])
        system = "-".join(sorted(el.symbol for el in comp.elements))
        natoms = comp.num_atoms
        rec = hulls.setdefault(system, {"elements": sorted(el.symbol for el in comp.elements),
                                        "entries": {}})
        # keep the lowest-formation duplicate formula
        key = comp.reduced_formula
        e_form = float(row["formation_energy_per_atom"])
        prev = rec["entries"].get(key)
        cand = {"formula": key, "natoms": natoms, "e_form_per_atom": e_form}
        if prev is None or e_form < prev["e_form_per_atom"]:
            rec["entries"][key] = cand

    out = {sys: {"elements": v["elements"], "entries": list(v["entries"].values())}
           for sys, v in hulls.items()}
    _write("mp20", out, {
        "dataset": "mp_20",
        "splits": ["train", "val", "test"],
        "energy_zero": "elemental ground states (MP)", "e_form_units": "eV/atom",
        "e_form_from_metadata": "formation_energy_per_atom",
        "scope": "all MP-20 phases of the chemical system",
    })


# ---------------------------------------------------------------------------
# Perov-5
# ---------------------------------------------------------------------------

def build_perov5(verify: bool = True) -> None:
    """Perov-5: heat_all (per 5-atom cell, eV) / n_atoms.

    Measured properties of this dataset (all 18,928 rows):

    * Every cell is exactly 5 atoms -- ``heat_all`` is per *cell*, not per
      atom, so ``heat_all / 5`` is the per-atom quantity.  (Per atom the range
      would be -0.13..+1.03 eV; as written +5.16 eV/atom would be impossible.)
    * 9,646 canonical compositions for 18,928 rows: ~2 structures per
      composition, differing by up to 3.64 eV/cell.  These are hypothetical
      screening candidates, not a set of DFT-relaxed ground states.
    * ``heat_all`` is a ~0-anchored stability quantity (mean +1.481 eV/cell,
      0.36% below zero), which is why the elemental endmembers at 0 are the
      right reference level for it.
    * ``heat_ref`` is NOT a composition-level chemical potential: it varies
      within a composition (constant for only 3.8% of compositions) and tracks
      ``heat_all`` almost exactly (corr 0.971, mean difference -0.083 eV/cell).
      CDVAE and DiffCSP both use ``heat_ref`` as their perov_5 property-model
      target.  We use ``heat_all`` -- the difference is not a hull level.

    Consequence for the metric chain: because 83.3% of the 8,268 systems hold a
    single composition, the rebuilt diagram's only stable entries are the
    elemental ones and ``E_hull`` reduces to ``heat_all / 5`` -- i.e. Perov-5's
    E_hull is the dataset's own heat-of-formation scale, not an
    energy-above-MP-hull.  It is therefore NOT directly comparable with MP-20's
    E_hull (see the module docstring's energy-zero section).

    ``elements`` come from parsing the formula column, which reproduces the
    reference pickles' element sets exactly (verified on the test split by
    ``verify=True``).
    """
    import pandas as pd
    from pymatgen.core import Composition

    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(DATA / "perov_5" / f"{split}.csv")
        df = df[["material_id", "formula", "heat_all", "heat_ref"]]
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    hulls: dict = {}
    for _, row in df.iterrows():
        comp = Composition(row["formula"])
        system = "-".join(sorted(el.symbol for el in comp.elements))
        rec = hulls.setdefault(system, {"elements": sorted(el.symbol for el in comp.elements),
                                        "entries": {}})
        key = comp.reduced_formula
        e_form = float(row["heat_all"]) / comp.num_atoms
        prev = rec["entries"].get(key)
        cand = {"formula": key, "natoms": comp.num_atoms, "e_form_per_atom": e_form}
        if prev is None or e_form < prev["e_form_per_atom"]:
            rec["entries"][key] = cand

    out = {sys: {"elements": v["elements"], "entries": list(v["entries"].values())}
           for sys, v in hulls.items()}

    if verify:
        _verify_perov5(out, df)

    _write("perov5", out, {
        "dataset": "perov_5",
        "splits": ["train", "val", "test"],
        "energy_zero": ("elemental endmembers at 0, which coincide with the "
                        "zero of heat_all itself (a ~0-anchored quantity)"),
        "e_form_units": "eV/atom",
        "e_form_from_metadata": "heat_all / n_atoms (heat_all is per 5-atom cell)",
        "unused_column": ("heat_ref varies within a composition (constant for "
                          "3.8% of compositions) and tracks heat_all (corr "
                          "0.971); it is a per-structure label, not a hull level"),
        "scope": ("all Perov-5 phases of the chemical system -- MP-derived hulls "
                  "are unavailable for its 8,268 exotic systems"),
        "comparability": ("83.3% of systems hold a single composition, so E_hull "
                          "reduces to heat_all / n_atoms: the dataset's own "
                          "heat-of-formation scale, NOT comparable with MP-20's "
                          "energy-above-MP-hull"),
    })


def _verify_perov5(out: dict, df) -> None:
    """Cross-check the built table against the reference pickles (test split)
    and against the hull's own self-consistency (no entry below the hull)."""
    import pickle
    import warnings

    warnings.filterwarnings("ignore")
    from ase.data import chemical_symbols
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    ref_path = Path(__file__).resolve().parents[1] / "results" / "reference" / "perov_5" / "mace_test.pkl"
    if ref_path.exists():
        with open(ref_path, "rb") as f:
            recs = pickle.load(f)
        by_id = {r["metadata"]["material_id"]: r for r in recs}
        csv_test = df[df.split == "test"].set_index("material_id")
        n = mismatch = 0
        for mid, rec in by_id.items():
            if mid not in csv_test.index:
                continue
            n += 1
            syms = tuple(sorted(chemical_symbols[int(z)] for z in rec["numbers"]))
            comp = Composition(csv_test.loc[mid, "formula"])
            csv_syms = tuple(sorted(symbol for symbol, amt in comp.get_el_amt_dict().items()
                                    for _ in range(int(amt))))
            if syms != csv_syms or abs(rec["metadata"]["heat_all"] - csv_test.loc[mid, "heat_all"]) > 1e-9:
                mismatch += 1
        print(f"  verify: {n} test-split records cross-checked against reference "
              f"pickles, {mismatch} mismatches")
        assert mismatch == 0, "perov_5 hull entries disagree with the reference pickles"

    # self-consistency: rebuild each diagram from the table alone
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    below = total = 0
    for sys_name, v in out.items():
        ents = [ComputedEntry(Composition(e["formula"]),
                              e["e_form_per_atom"] * Composition(e["formula"]).num_atoms)
                for e in v["entries"]]
        ents += [ComputedEntry(Composition(el), 0.0) for el in v["elements"]]
        try:
            pd_ = PhaseDiagram(ents)
        except Exception:
            continue
        for e in v["entries"]:
            comp = Composition(e["formula"])
            total += 1
            if pd_.get_e_above_hull(ComputedEntry(
                    comp, e["e_form_per_atom"] * comp.num_atoms)) < -1e-6:
                below += 1
    print(f"  verify: {total} phases rebuilt, {below} below the hull (must be 0)")
    assert below == 0, "hull table is not self-consistent (entries below the hull)"


# ---------------------------------------------------------------------------
# Carbon-24
# ---------------------------------------------------------------------------

def build_carbon24() -> None:
    """Single-element hull: the ground state is the lowest-energy carbon
    allotrope in the dataset (diamond at 10 GPa), and E_hull of a structure is
    its energy per atom minus that value."""
    import pandas as pd

    frames = [pd.read_csv(DATA / "carbon_24" / f"{split}.csv",
                          usecols=["material_id", "energy_per_atom"])
              for split in ["train", "val", "test"]]
    df = pd.concat(frames, ignore_index=True)
    e_min = float(df["energy_per_atom"].min())
    frac_within = float(((df["energy_per_atom"] - e_min) < 0.1).mean())

    out = {"C": {"elements": ["C"],
                 "entries": [{"formula": "C", "natoms": 1, "e_form_per_atom": 0.0}]}}
    print(f"  ground state: {e_min:.6f} eV/atom; spread "
          f"{df['energy_per_atom'].max() - e_min:.4f} eV/atom; "
          f"{frac_within:.1%} of the dataset within 100 meV/atom")
    _write("carbon24", out, {
        "dataset": "carbon_24",
        "splits": ["train", "val", "test"],
        "energy_zero": {"kind": "elemental_carbon_ground_state",
                        "value_per_atom": e_min,
                        "source": "min energy_per_atom over train+val+test"},
        "e_form_units": "eV/atom",
        "e_form_from_metadata": "energy_per_atom - energy_zero.value_per_atom",
        "scope": "elemental carbon phases (dataset allotropes)",
    })


# ---------------------------------------------------------------------------

BUILDERS = {"mp_20": build_mp20, "perov_5": build_perov5, "carbon_24": build_carbon24}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="all",
                        choices=["all", *BUILDERS],
                        help="which dataset's hull to build (default: all)")
    args = parser.parse_args()
    names = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"[{name}]")
        BUILDERS[name]()


if __name__ == "__main__":
    main()
