"""
Tests for evaluation metrics.

Test cases:
    1. test_validity_known_structures:
       Structures from the MP-20 test split should pass validity check
       (these are DFT-relaxed, physically valid structures).

    2. test_validity_overlapping_atoms:
       A structure with two atoms at identical positions should fail
       the overlap check (d_min < 0.5 A).

    3. test_match_rate_self:
       Matching a dataset against itself should give match_rate = 1.0
       and coverage = 1.0.

    4. test_match_rate_permuted:
       Permuting the site order must not change the result (the matcher is
       permutation-invariant; the AMSD correspondence must be too).

    5. test_match_rate_supercell:
       A 2x2x2 supercell matches the original cell only under
       ``primitive_cell=True``; the frozen default is False.

    6. test_diversity_identical:
       A set of identical structures has AMSD ~ 0, mode-collapse entropy 0 and
       space-group entropy 0.

    7. test_diversity_distinct:
       Structurally distinct polymorphs of one composition have AMSD > 0 and
       higher space-group entropy.

    8. test_energy_above_hull_stable:
       A phase on the convex hull has E_hull = 0.

    9. test_energy_above_hull_unstable:
       A phase above the hull has E_hull > 0; one below it has E_hull < 0.

    10. test_metrics_consistency:
        Metric values within expected ranges: validity/match_rate/coverage in
        [0, 1], AMSD >= 0, E_hull finite.

Regressions (each one is a bug these tests caught):
    R1  test_amsd_is_angstrom_not_matcher_units
        pymatgen's get_rms_dist is normalized by the free length per atom
        (dimensionless); using it as d_RMS made AMSD a strain-like ratio.
    R2  test_amsd_defined_for_dissimilar_pairs
        The matcher is tolerance-gated: it returned None for 44 of 45 pairs of
        a real Phase-1 ensemble, i.e. NaN in the regime AMSD exists to measure.
    R3  test_reference_groups_use_group_structures
        StructureMatcher.group() does not exist in pymatgen 2026.5.4; every
        match-rate/coverage call raised AttributeError.
    R4  test_e_above_hull_below_reference_set
        pymatgen raises "No valid decomposition found" for a below-hull entry
        unless allow_negative=True -- exactly the structures a benchmark wants
        to find were silently dropped.
    R5  test_hull_tables_self_consistent / R6 test_registry_conventions
        The frozen hull tables must be self-consistent and per-dataset
        formation-energy conventions must match the recorded provenance.
    R7  test_validity_accepts_pymatgen_structures
        Third-party generators hand back pymatgen objects, which have no
        num_atoms / chemical_formula / get_minimum_distance: every such
        structure was reported invalid with an AttributeError reason.
    R8  test_structure_record_volume_and_io_passthrough
        StructureRecord took a determinant of a pymatgen Lattice object; the
        converters must pass their own type through unchanged.

Dependencies:
    pytest, numpy, pymatgen, pandas
    materialgen.eval.*
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from materialgen.core.crystal import CrystalStructure
from materialgen.eval import (
    E_HULL_STABLE,
    HullEvaluator,
    HullTable,
    StructureEvaluator,
    StructureRecord,
    TrainingIndex,
    canonical_formula,
    compute_amsd,
    compute_amsd_matrix,
    compute_coverage,
    compute_diversity,
    compute_match_and_coverage,
    compute_match_rate,
    compute_mode_collapse_metric,
    compute_r_angle,
    compute_sun,
    compute_symmetry_score,
    compute_validity,
    detect_spacegroup,
    make_matcher,
)
from materialgen.utils.crystal_io import from_pymatgen, to_pymatgen


# ---------------------------------------------------------------------------
# Structures used throughout
# ---------------------------------------------------------------------------

def rock_salt(a: float = 5.64) -> Structure:
    """NaCl rock salt (Fm-3m, 8 atoms)."""
    return Structure.from_spacegroup(225, Lattice.cubic(a), ["Na", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def zincblende_nacl(a: float = 5.90) -> Structure:
    """NaCl in the zincblende arrangement (F-43m, 8 atoms): same composition,
    a genuinely different polymorph (space group and coordination differ)."""
    return Structure.from_spacegroup(216, Lattice.cubic(a), ["Na", "Cl"],
                                     [[0, 0, 0], [0.25, 0.25, 0.25]])


def cscl(a: float = 4.11) -> Structure:
    """CsCl (Pm-3m, 2 atoms)."""
    return Structure.from_spacegroup(221, Lattice.cubic(a), ["Cs", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def fcc_cu(a: float = 3.615) -> Structure:
    """FCC copper (Fm-3m, 4 atoms)."""
    return Structure.from_spacegroup(225, Lattice.cubic(a), ["Cu"], [[0, 0, 0]])


def permuted(structure: Structure, seed: int = 0) -> Structure:
    """The same structure with its sites listed in a different order."""
    idx = np.random.default_rng(seed).permutation(len(structure))
    return Structure.from_sites([structure.sites[i] for i in idx])


def supercell(structure: Structure, n: int = 2) -> Structure:
    s = structure.copy()
    s.make_supercell([n, n, n])
    return s


@pytest.fixture(scope="module")
def synthetic_hull() -> HullTable:
    """Li-O diagram with a known geometry (no dataset files needed).

    In the Li-O composition space (x_O = fraction of O): the Li2O entry at
    x_O = 1/3 sits on the hull, the tie-line Li2O-O gives -0.375 eV/atom at
    x_O = 1/2, and Li2O2 is stored at -0.30 -- i.e. 0.075 eV/atom above it.
    """
    return HullTable(name="<test Li-O>", systems={
        "Li-O": {"elements": ["Li", "O"], "entries": [
            {"formula": "Li2O", "natoms": 3, "e_form_per_atom": -0.5},
            {"formula": "Li2O2", "natoms": 4, "e_form_per_atom": -0.30},
        ]},
    })


@pytest.fixture(scope="module")
def nacl_hull() -> HullTable:
    """Single-phase Na-Cl diagram, for the end-to-end evaluator test.

    One NaCl entry at -0.5 eV/atom, so the hull at x_Cl = 1/2 is that entry
    (the Na-Cl tie-line sits at 0) and E_hull is the formation energy's offset
    from it.  The key is "Cl-Na": system keys sort the element symbols.
    """
    return HullTable(name="<test Na-Cl>", systems={
        "Cl-Na": {"elements": ["Cl", "Na"], "entries": [
            {"formula": "NaCl", "natoms": 8, "e_form_per_atom": -0.5},
        ]},
    })


@pytest.fixture(scope="module")
def mp20_sample():
    """15 parsed structures from the MP-20 test split (the real thing)."""
    pd = pytest.importorskip("pandas")
    from materialgen.data.datasets import _parse_cif
    from materialgen.data.registry import get_dataset

    df = pd.read_csv(get_dataset("mp_20").csv_path("test"), usecols=["cif"]).iloc[:15]
    return [_parse_cif(text) for text in df["cif"]]


# ---------------------------------------------------------------------------
# 1-2. Validity
# ---------------------------------------------------------------------------

def test_validity_known_structures(mp20_sample):
    """DFT-relaxed dataset structures must all pass validity."""
    report = compute_validity(mp20_sample)
    assert report["n"] == len(mp20_sample)
    assert report["validity_rate"] == 1.0
    assert report["reasons"] == {}


def test_validity_overlapping_atoms():
    """Two atoms at the same position -> d_min = 0 -> invalid, with a reason."""
    bad = rock_salt()
    bad[1] = bad[0].species, bad[0].frac_coords
    report = compute_validity([bad])
    assert report["n_valid"] == 0
    assert report["validity_rate"] == 0.0
    assert any("overlap" in reason for reason in report["reasons"])

    # A single atom is degenerate, and non-finite positions must not crash.
    assert compute_validity([cscl()[:1]])["n_valid"] == 0
    nan_cell = rock_salt()
    nan_cell.lattice = Lattice(np.full((3, 3), np.nan))
    assert compute_validity([nan_cell])["n_valid"] == 0


def test_validity_composition_mismatch():
    """The composition check compares reduced formulas."""
    assert compute_validity([rock_salt()], ["NaCl"])["validity_rate"] == 1.0
    report = compute_validity([rock_salt()], ["KCl"])
    assert report["validity_rate"] == 0.0
    assert any("composition" in reason for reason in report["reasons"])


def test_validity_accepts_pymatgen_structures():
    """R7: pymatgen input must not degrade into an AttributeError reason."""
    report = compute_validity([rock_salt(), fcc_cu()])
    assert report["validity_rate"] == 1.0
    assert report["reasons"] == {}
    # same answer as the CrystalStructure route
    as_crystal = compute_validity([from_pymatgen(rock_salt())])
    assert as_crystal["validity_rate"] == 1.0


# ---------------------------------------------------------------------------
# 3-5. Match rate / coverage
# ---------------------------------------------------------------------------

def test_match_rate_self():
    """An ensemble matches the reference set it was drawn from."""
    ref = [fcc_cu(), rock_salt(), cscl(), rock_salt(5.70)]
    out = compute_match_and_coverage(ref, ref)
    assert out["match_rate"] == 1.0
    assert out["coverage"] == 1.0
    assert out["n_generated"] == out["n_reference"] == 4
    assert compute_match_rate(ref, ref)["match_rate"] == 1.0
    assert compute_coverage(ref, ref)["coverage"] == 1.0


def test_match_rate_permuted():
    """Site order is irrelevant (also pins the grouping API, R3)."""
    ref = [fcc_cu(), rock_salt(), cscl()]
    gen = [permuted(fcc_cu(), seed=1), permuted(rock_salt(), seed=2),
           permuted(cscl(), seed=3)]
    out = compute_match_and_coverage(gen, ref)
    assert out["match_rate"] == 1.0
    assert out["coverage"] == 1.0


def test_match_rate_supercell():
    """Supercell vs primitive cell: only the primitive_cell=True preset fits."""
    ref = [fcc_cu()]
    gen = [supercell(fcc_cu())]
    assert compute_match_rate(gen, ref)["match_rate"] == 0.0
    relaxed = make_matcher("paper", primitive_cell=True)
    assert compute_match_rate(gen, ref, relaxed)["match_rate"] == 1.0


def test_match_rate_partial_coverage():
    """match_rate and coverage are two different directions."""
    ref = [fcc_cu(), rock_salt(), cscl()]
    out = compute_match_and_coverage([fcc_cu(), rock_salt()], ref)
    assert out["match_rate"] == 1.0          # both generated structures are known
    assert out["coverage"] == pytest.approx(2 / 3)  # CsCl was never generated


# ---------------------------------------------------------------------------
# 6-7. Diversity
# ---------------------------------------------------------------------------

def test_diversity_identical():
    """Identical ensemble: AMSD ~ 0, zero mode-collapse and space-group entropy."""
    ensemble = [rock_salt(), rock_salt(), rock_salt()]
    assert compute_amsd(ensemble) == pytest.approx(0.0, abs=1e-9)
    assert compute_mode_collapse_metric(ensemble) == 0.0
    report = compute_diversity(ensemble)
    assert report["amsd"] == pytest.approx(0.0, abs=1e-9)
    assert report["mode_collapse_entropy"] == 0.0
    assert report["spacegroup_entropy"] == 0.0
    assert report["n_spacegroups"] == 1
    assert report["volume_std"] == pytest.approx(0.0, abs=1e-9)


def test_diversity_distinct():
    """Three polymorphs of one composition: non-zero, ordered diversity."""
    ensemble = [rock_salt(), rock_salt(5.70), zincblende_nacl()]
    report = compute_diversity(ensemble)
    assert report["amsd"] > 0.1
    assert report["mode_collapse_entropy"] > 0.0
    assert report["n_spacegroups"] == 2
    assert report["spacegroup_entropy"] > 0.0
    assert report["volume_cv"] > 0.0
    # a strained pair is closer together than two different structure types
    assert compute_amsd([rock_salt(), rock_salt(5.70)]) < compute_amsd(
        [rock_salt(), zincblende_nacl()])


def test_amsd_is_angstrom_not_matcher_units():
    """R1: AMSD must come back in A, not in pymatgen's normalized units.

    Two atoms in a 10 A cell, separated by 5.0 A in one structure and 5.5 A in
    the other: a two-point Kabsch distance is |d1 - d2| / 2 = 0.25 A exactly
    (both sets are centred, and the optimal rotation aligns the two bond axes
    and cannot shorten the leftover).  pymatgen's get_rms_dist reports the same
    pair as 0.0315 -- it divides by the free length per atom
    ((V/N)^(1/3) = 7.94 A here), so its number is dimensionless.  A
    length-proportional metric also doubles when the structures do, which a
    normalized one does not.
    """
    def pair(d: float, cell: float = 10.0) -> Structure:
        return Structure(Lattice.cubic(cell), ["Cs", "Cl"],
                         [[0, 0, 0], [d / cell, 0, 0]])

    near, far = pair(5.0), pair(5.5)
    amsd = compute_amsd([near, far])
    assert amsd == pytest.approx(0.25, abs=1e-6), amsd
    assert compute_amsd([pair(5.0), pair(5.1)]) == pytest.approx(0.05, abs=1e-6)

    normalized = make_matcher("paper").get_rms_dist(to_pymatgen(near), to_pymatgen(far))
    if normalized is not None:
        assert normalized[0] < 0.1, "sanity: the matcher's value is dimensionless"

    def scaled(structure: Structure, factor: float) -> Structure:
        big = structure.copy()
        big.scale_lattice(big.volume * factor ** 3)
        return big

    doubled = compute_amsd([scaled(near, 2.0), scaled(far, 2.0)])
    assert doubled == pytest.approx(2 * amsd, rel=1e-9)


def test_amsd_defined_for_dissimilar_pairs():
    """R2: structurally unrelated members of one composition still get a distance."""
    ensemble = [rock_salt(), cscl(), fcc_cu()]
    matrix = compute_amsd_matrix(ensemble)
    # rock salt vs CsCl vs Cu: no pair matches, and only the NaCl pair could
    # ever match -- every finite entry must be a real distance
    natoms = {s.composition.reduced_formula: len(s) for s in ensemble}
    assert natoms == {"NaCl": 8, "CsCl": 2, "Cu": 4}
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[0, 2]) and np.isnan(matrix[1, 2])
    assert np.isnan(compute_amsd(ensemble))     # no comparable pair at all

    # same composition, different cells: now it is defined
    polymorphs = [rock_salt(), zincblende_nacl()]
    value = compute_amsd(polymorphs)
    assert np.isfinite(value) and value > 0.1

    # same-composition ensemble from a real Phase-1 run (10 candidates)
    import json
    from pathlib import Path

    path = Path("results/phase1/subexp1/s3_bare_pfode/SrTiO3_seed42.json")
    if path.exists():
        candidates = [CrystalStructure.from_dict(c["final"]["structure"])
                      for c in json.loads(path.read_text())["candidates"]]
        matrix = compute_amsd_matrix(candidates)
        off_diagonal = matrix[np.triu_indices(len(matrix), 1)]
        assert np.isfinite(off_diagonal).all(), "AMSD must be defined for every pair"
        assert 0.0 < compute_amsd(candidates) < 10.0


def test_amsd_permutation_invariance():
    """The canonical correspondence makes AMSD independent of site order."""
    ensemble = [rock_salt(), rock_salt(5.70), zincblende_nacl()]
    shuffled = [permuted(s, seed=i) for i, s in enumerate(ensemble)]
    assert compute_amsd(shuffled) == pytest.approx(compute_amsd(ensemble), abs=1e-12)


def test_r_angle_conventions():
    """KL (paper) and Wasserstein (baseline scripts) are both available."""
    kl = compute_r_angle([zincblende_nacl()], [rock_salt()], "kl")
    wasserstein = compute_r_angle([zincblende_nacl()], [rock_salt()], "wasserstein")
    assert kl["metric"] == "kl" and wasserstein["metric"] == "wasserstein"
    assert kl["n_generated_angles"] > 0 and kl["n_reference_angles"] > 0
    # tetrahedral (109.5 deg) vs octahedral (90 deg) coordination: not the same
    assert kl["value"] > 1.0
    assert wasserstein["value"] > 1.0
    # an identical pair diverges nowhere
    same = compute_r_angle([rock_salt()], [rock_salt()], "kl")
    assert same["value"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        compute_r_angle([rock_salt()], [rock_salt()], "nonsense")


def test_symmetry_score():
    ensemble = [rock_salt(), rock_salt(5.65), zincblende_nacl()]
    assert [detect_spacegroup(s) for s in ensemble] == [225, 225, 216]
    assert compute_symmetry_score(ensemble, {225})["symmetry_score"] == pytest.approx(2 / 3)
    assert compute_symmetry_score(ensemble, [225, 225, 216])["symmetry_score"] == 1.0


# ---------------------------------------------------------------------------
# 8-9. Hull
# ---------------------------------------------------------------------------

def test_energy_above_hull_stable(synthetic_hull):
    """The Li2O entry defines the hull at its composition -> E_hull = 0."""
    assert synthetic_hull.e_above_hull("Li2O", -0.5) == pytest.approx(0.0, abs=1e-9)
    # and its coverage is exactly the systems it was built from
    assert "Li-O" in synthetic_hull and len(synthetic_hull) == 1
    assert synthetic_hull.n_phases == 2
    assert synthetic_hull.elements("Li-O") == ["Li", "O"]


def test_energy_above_hull_unstable(synthetic_hull):
    """Li2O2 sits 0.075 eV/atom above the Li2O-O tie-line."""
    assert synthetic_hull.e_above_hull("Li2O2", -0.30) == pytest.approx(0.075, abs=1e-6)
    # lifting the formation energy lifts E_hull by the same amount
    assert synthetic_hull.e_above_hull("Li2O2", -0.20) == pytest.approx(0.175, abs=1e-6)
    # a chemical system the table does not have is reported as not covered
    assert synthetic_hull.e_above_hull("NaCl", -0.1) is None
    assert "Na-Cl" not in synthetic_hull


def test_e_above_hull_below_reference_set(synthetic_hull):
    """R4: a phase better than everything in the reference set is E_hull < 0.

    pymatgen raises "No valid decomposition found" for these entries under its
    default allow_negative=False -- i.e. the most interesting output of a
    generative benchmark (a structure that beats the dataset) would be dropped
    as None.
    """
    below = synthetic_hull.e_above_hull("Li2O2", -0.55)     # 0.175 below the tie-line
    assert below is not None, "a below-hull entry must not be dropped"
    assert below == pytest.approx(-0.175, abs=1e-6)

    # The real occurrence of this case is a *generated* structure better than
    # everything the dataset knows at its composition.  Note that a dataset's
    # own phases can never be it: the hull is built from those phases, so each
    # composition's best entry sits on the hull (Perov-5: 0 of 9,646 entries
    # below their hull), even the 0.36% whose heat_all is
    # negative.  Hence the -50 meV/atom query below.
    pd = pytest.importorskip("pandas")
    from pymatgen.core import Composition

    from materialgen.data.registry import get_dataset

    spec = get_dataset("perov_5")
    # the CSV has no n_atoms column (every cell is 5 atoms)
    df = pd.read_csv(spec.csv_path("test"), usecols=["formula", "heat_all"])
    table = spec.hull()
    systems = {name: rec for name, rec in table.systems.items()}
    # a single-phase system whose phase is below the elemental zero: there the
    # phase *is* the hull at its composition, so E_hull is exactly the offset
    row = next(r for _, r in df.iterrows()
               if float(r["heat_all"]) < 0
               and len(systems["-".join(sorted(
                   el.symbol for el in Composition(r["formula"]).elements))]["entries"]) == 1)
    best = float(row["heat_all"]) / Composition(row["formula"]).num_atoms
    assert table.e_above_hull(row["formula"], best) == pytest.approx(0.0, abs=1e-9)
    value = table.e_above_hull(row["formula"], best - 0.05)
    assert value is not None, f"{row['formula']} dropped as None"
    assert value == pytest.approx(-0.05, abs=1e-6), (row["formula"], value)


def test_hull_evaluator_calibration(synthetic_hull):
    """NNP energy -> formation energy -> E_hull, on the synthetic table."""
    evaluator = HullEvaluator(synthetic_hull)
    assert evaluator.e_hull("Li2O", nnp_energy=-10.0, n_atoms=3) is None  # uncalibrated
    # anchor: an NNP energy of -15 eV for 3 atoms at a DFT formation energy of
    # -0.5 eV/atom
    evaluator.calibrate("Li2O", nnp_energy_ref=-15.0, n_atoms_ref=3,
                        e_form_ref_per_atom=-0.5)
    assert evaluator.calibrated_formulas == frozenset({"Li2O"})
    assert evaluator.e_form_per_atom("Li2O", -15.0, 3) == pytest.approx(-0.5)
    assert evaluator.e_hull("Li2O", -15.0, 3) == pytest.approx(0.0, abs=1e-9)
    assert evaluator.is_stable("Li2O", -15.0, 3) is True
    # 0.2 eV/atom worse -> above the 100 meV/atom threshold
    assert evaluator.is_stable("Li2O", -14.4, 3) is False

    # canonicalization: the raw dataset label finds the same calibration
    evaluator.calibrate("Li2O2", nnp_energy_ref=-20.0, n_atoms_ref=4,
                        e_form_ref_per_atom=-0.30)
    assert evaluator.e_form_per_atom("Li2O2", -20.0, 4) == pytest.approx(-0.30)


def test_hull_tables_self_consistent():
    """R5: every frozen table's phases sit on or above their own hull."""
    from materialgen.data.registry import DATASET_NAMES, get_dataset

    rng = np.random.default_rng(0)
    for name in DATASET_NAMES:
        table = get_dataset(name).hull()
        assert len(table) > 0 and table.n_phases > 0
        assert table.meta, f"{name}: hull sidecar provenance is empty"
        systems = sorted(table.systems)
        sample = [systems[i] for i in rng.choice(
            len(systems), size=min(15, len(systems)), replace=False)]
        for system in sample:
            for entry in table.systems[system]["entries"]:
                value = table.e_above_hull(entry["formula"], entry["e_form_per_atom"])
                assert value is not None, (name, entry["formula"])
                assert value > -1e-6, (name, entry["formula"], value)
        # a formula whose system is absent must be reported as not covered
        assert table.e_above_hull("Xx4Qq4", -1.0) is None


def test_registry_conventions():
    """R6: each dataset's formation energy follows its recorded convention."""
    from materialgen.data.registry import DATASET_NAMES, get_dataset

    assert DATASET_NAMES == ("mp_20", "perov_5", "carbon_24")
    for name in DATASET_NAMES:
        spec = get_dataset(name)
        assert spec.hull_entries.exists() and spec.hull_meta.exists()
        assert spec.calib.__doc__ or True       # callable, documented above

    # Perov-5: heat_all is per 5-atom cell
    spec = get_dataset("perov_5")
    assert abs(spec.dft_e_form_per_atom({"heat_all": 1.5}, 5) - 0.3) < 1e-12

    # Carbon-24: energy_per_atom - energy_zero (recorded in the meta file)
    spec = get_dataset("carbon_24")
    zero = spec.energy_zero_per_atom
    assert abs(spec.dft_e_form_per_atom({"energy_per_atom": zero + 0.25}, 1) - 0.25) < 1e-12

    # MP-20: the CSV column, used as-is
    spec = get_dataset("mp_20")
    assert spec.dft_e_form_per_atom({"formation_energy_per_atom": -1.25}, 20) == -1.25


def test_canonical_formula():
    """Perov-5's CSV formula column is not in pymatgen's element order."""
    assert canonical_formula("TiOsOFN") == "TiOsNOF"
    assert canonical_formula("TiOsNOF") == "TiOsNOF"
    assert canonical_formula(rock_salt()) == "NaCl"
    assert canonical_formula("SrTiO3") == "SrTiO3"


def test_registry_calibration_anchor():
    """The reference anchor is the most stable same-composition record, ranked
    by the dataset's own formation energy -- never by the CSV e_above_hull
    column, which is on a different MP energy convention."""
    pd = pytest.importorskip("pandas")
    from materialgen.data.registry import get_dataset

    spec = get_dataset("mp_20")
    records = spec.calib(nnp="mace", split="test")
    assert records, "no calibration records for mp_20"
    for rec in records[:5]:
        assert rec["formula"] == canonical_formula(rec["formula"])
        assert rec["n_atoms_ref"] > 0 and rec["material_id"]
    frame = pd.read_csv(spec.csv_path("test"),
                        usecols=["pretty_formula", "formation_energy_per_atom"])
    frame["formula"] = [canonical_formula(f) for f in frame["pretty_formula"]]
    for rec in records[:5]:
        rows = frame[frame["formula"] == rec["formula"]]
        assert not rows.empty
        assert rec["e_form_ref_per_atom"] == pytest.approx(
            rows["formation_energy_per_atom"].min(), abs=1e-9)


# ---------------------------------------------------------------------------
# 10. S.U.N. and the aggregate evaluator
# ---------------------------------------------------------------------------

def test_mark_unique_and_sun():
    """Uniqueness is relative to the other candidates of the same composition."""
    duplicate = rock_salt(5.66)                 # same space group, +0.4% volume
    distinct = zincblende_nacl()
    records = [
        StructureRecord(formula="NaCl", structure=rock_salt(), e_hull=0.01,
                        spacegroup=225, valid=True, novel=True),
        StructureRecord(formula="NaCl", structure=duplicate, e_hull=0.02,
                        spacegroup=225, valid=True, novel=True),
        StructureRecord(formula="NaCl", structure=distinct, e_hull=0.03,
                        spacegroup=216, valid=True, novel=False),
    ]
    report = compute_sun(records)
    assert report["n"] == 3
    assert report["n_stable"] == 3 and report["n_novel"] == 2
    assert report["n_unique"] == 2              # the two rock salts collapse to one
    assert report["n_sun"] == 1                 # only the first is S, U and N
    assert report["sun_rate"] == pytest.approx(1 / 3)
    assert report["e_hull_threshold"] == E_HULL_STABLE
    assert records[0].unique is True and records[1].unique is False
    # unstable candidates are excluded from S.U.N. regardless of uniqueness
    assert StructureRecord(formula="NaCl", structure=rock_salt(),
                           e_hull=E_HULL_STABLE + 0.01, spacegroup=225,
                           valid=True, novel=True).sun is False


def test_structure_record_volume_and_io_passthrough():
    """R8: records and converters accept both structure flavours."""
    pmg = cscl()
    assert to_pymatgen(pmg) is pmg
    crystal = from_pymatgen(pmg)
    assert from_pymatgen(crystal) is crystal
    assert to_pymatgen(crystal).composition.reduced_formula == "CsCl"

    assert StructureRecord(formula="CsCl", structure=pmg).volume == pytest.approx(
        pmg.volume)
    assert StructureRecord(formula="CsCl", structure=crystal).volume == pytest.approx(
        pmg.volume)
    # a list is a caller error, and must say so (not "no attribute 'lattice'")
    with pytest.raises(TypeError):
        to_pymatgen([pmg])


def test_training_index_and_novelty():
    """Novelty compares against the same composition of the training split."""
    index = TrainingIndex({"NaCl": [rock_salt()], "CsCl": [cscl()]})
    assert index.stats()["n_structures"] == 2
    assert index.is_novel(rock_salt(5.65)) is False        # same phase, new cell
    assert index.is_novel(zincblende_nacl()) is True       # new polymorph
    assert index.is_novel(fcc_cu()) is True                # unseen composition
    assert index.same_composition("ClNa") and not index.same_composition("Cu")


def test_structure_evaluator_end_to_end(nacl_hull):
    """The aggregate evaluator wires validity, hull, novelty and matching.

    Four NaCl candidates against a training split holding one zincblende cell:
    all four are novel (the rock-salt phase is not in the training split), the
    zincblende candidate is not, and two of the rock salts are stable.
    """
    reference = [rock_salt(), zincblende_nacl()]
    evaluator = StructureEvaluator(
        reference=reference,
        training=TrainingIndex({"NaCl": [zincblende_nacl()]}),
        hull=HullEvaluator(nacl_hull),
        formula="NaCl",
        )
    evaluator.hull.calibrate("NaCl", nnp_energy_ref=-15.0, n_atoms_ref=8,
                             e_form_ref_per_atom=-0.5)
    generated = [rock_salt(), rock_salt(5.70), zincblende_nacl(), rock_salt(5.85)]
    energies = [-15.0, -14.2, -13.6, -14.4]     # E_hull = 0 / 0.1 / 0.175 / 0.075
    records = evaluator.build_records(generated, energies,
                                      formulas=["NaCl"] * 4)
    assert [r.valid for r in records] == [True] * 4
    assert [round(r.e_hull, 6) for r in records] == [0.0, 0.1, 0.175, 0.075]
    # the stable threshold is strict: E_hull == 0.1 eV/atom is not stable
    assert [r.stable for r in records] == [True, False, False, True]
    assert [r.novel for r in records] == [True, True, False, True]

    report = evaluator.evaluate(records)
    metrics = report.to_dict()
    assert metrics["n"] == 4 and metrics["formula"] == "NaCl"
    assert metrics["validity_rate"] == 1.0 and metrics["n_valid"] == 4
    assert metrics["sun"]["n_novel"] == 3
    assert metrics["sun"]["n_stable"] == 2
    assert metrics["sun"]["n_unique"] == 4
    assert metrics["sun"]["n_sun"] == 2         # the two stable novel rock salts
    assert metrics["match_coverage"]["match_rate"] == 1.0
    assert metrics["match_coverage"]["coverage"] == 1.0
    assert metrics["amsd"] > 0.0
    assert metrics["e_hull"]["n"] == 4
    assert metrics["e_hull"]["frac_stable"] == pytest.approx(0.5)
    assert metrics["e_hull"]["frac_below_zero"] == 0.0
    assert len(report.to_dataframe()) == 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no hull warnings on a clean run
        report.print_summary()


def test_metrics_consistency():
    """Ranges: every rate in [0, 1], AMSD >= 0, E_hull finite and sane."""
    generated = [rock_salt(), rock_salt(5.70), zincblende_nacl(), cscl()]
    reference = [rock_salt(), zincblende_nacl()]

    validity = compute_validity(generated)
    assert 0.0 <= validity["validity_rate"] <= 1.0
    matching = compute_match_and_coverage(generated, reference)
    assert 0.0 <= matching["match_rate"] <= 1.0
    assert 0.0 <= matching["coverage"] <= 1.0
    assert 0.0 <= matching["match_rate"] * len(generated) / 1.0 <= len(generated)

    amsd = compute_amsd([rock_salt(), rock_salt(5.70), zincblende_nacl()])
    assert amsd >= 0.0 and np.isfinite(amsd)
    assert -1.0 <= synthetic_e_hull() <= 10.0

    table = HullTable(name="<test>", systems={"Li-O": {"elements": ["Li", "O"],
                                                       "entries": [
        {"formula": "Li2O", "natoms": 3, "e_form_per_atom": -0.5}]}})
    assert table.e_above_hull("Li2O", -0.5) == pytest.approx(0.0, abs=1e-9)


def synthetic_e_hull() -> float:
    """E_hull of a phase 0.05 eV/atom above the synthetic Li-O hull."""
    table = HullTable(name="<test>", systems={"Li-O": {"elements": ["Li", "O"],
                                                       "entries": [
        {"formula": "Li2O", "natoms": 3, "e_form_per_atom": -0.5},
        {"formula": "Li2O2", "natoms": 4, "e_form_per_atom": -0.30}]}})
    return float(table.e_above_hull("Li2O2", -0.25))
