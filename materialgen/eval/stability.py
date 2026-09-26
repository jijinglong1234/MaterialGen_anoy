"""
Thermodynamic stability evaluation for generated crystal structures.

Implements the E_hull term of the S.U.N. criterion (paper section 6.0.7) on top
of the per-dataset hull tables built by ``scripts/build_hulls.py``:

    Stable  <=>  E_hull^NNP < 100 meV/atom

Pipeline (mirrors run_phase1.HullEvaluator, which this module supersedes once
Phase 1 is frozen):

    E_form^NNP(x) = E_NNP(x)/N - [ E_NNP(ref)/N_ref - E_form^DFT(ref) ]

The bracket is a per-composition calibration offset, computed once from a
reference structure whose DFT formation energy is known (``calibrate``).  It
converts the NNP total energy scale into the DFT formation-energy scale of the
hull table, assuming the NNP's per-atom error is a constant shift -- the same
proxy convention used throughout Phase 1/2.  ``e_hull`` then queries the
pymatgen PhaseDiagram of the structure's chemical system, rebuilt from the hull
table (elemental endmembers at 0).

Units: E_hull is returned in **eV/atom** (pymatgen's native convention for
``get_e_above_hull``).  Older files stored E_hull/n_fu and
carry no ``e_hull_units`` marker; see the ERRATUM in docs/experiment_plan_v1.md.

Dataset scope of the hull tables (frozen; see scripts/build_hulls.py):
    mp_20      per chemical system, all MP-20 phases (MP formation energies)
    perov_5    per chemical system, all Perov-5 phases (heat_all as E_form)
    carbon_24  single element; the hull is the lowest-energy carbon allotrope

Dependencies:
    numpy, pymatgen (PhaseDiagram, Composition, ComputedEntry)
    materialgen.utils.constants
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

E_HULL_UNITS = "eV/atom"
#: S.U.N. "Stable" threshold (paper section 6.0.7), eV/atom.
E_HULL_STABLE = 0.1
#: Lower edge of the stability band, eV/atom.
#:
#: The published criterion ("E_hull < 100 meV/atom") is one-sided, and that is
#: exploitable by the thing it is meant to measure: an NNP extrapolation
#: blow-up has an enormous *negative* E_hull and is therefore counted as a
#: stability success.  The Carbon-24 PF-ODE pilot made this concrete -- a
#: trajectory with max|F| = 611 eV/A and d_min = 0.83 A lands at E/N =
#: -67.6 eV/atom in a single-element system whose reference sits at -9.06
#: eV/atom, i.e. E_hull = -58.5 eV/atom, and the one-sided criterion calls it
#: stable.  The band is therefore two-sided: a structure is Stable iff
#:
#:     -E_HULL_FLOOR <= E_hull < E_HULL_STABLE
#:
#: The floor is a tolerance, not physics: an NNP may legitimately place a
#: structure a few tens of meV/atom below the DFT hull it was calibrated
#: against (that is its own error scale), but it cannot beat it by more than
#: half the stability window.  The guard is empirically inert on the frozen
#: Phase-1 corpus -- 0 of 331,470 candidates across results/phase1 (including
#: the pre-Cholesky archive) have E_hull < -50 meV/atom -- so no published
#: Phase-1 number moves; it only removes the failure mode where a sampler is
#: *rewarded* for diverging.  Reported alongside every rate as
#: `unphysical_rate` so the affected fraction is never hidden.
E_HULL_FLOOR = 0.05


def is_physical_e_hull(e_hull: Optional[float],
                       floor: float = E_HULL_FLOOR) -> Optional[bool]:
    """True iff E_hull is inside the physical band's lower edge (see above)."""
    if e_hull is None:
        return None
    return bool(e_hull >= -floor)

HULL_DIR = Path(__file__).resolve().parents[2] / "data" / "hulls"


def canonical_formula(formula_or_structure) -> str:
    """Composition key: pymatgen's reduced formula.

    Both sides of the calibration funnel through this.  The reason is
    dataset-specific: Perov-5's CSV ``formula`` column is *not* in pymatgen's
    electronegativity order -- 3014 of its 3785 test rows differ, e.g.
    ``'TiOsOFN'`` canonicalizes to ``'TiOsNOF'`` -- while a generator labels
    its output with whatever pymatgen produces.  Keying the calibration on a
    raw label would therefore miss ~80% of Perov-5.  For MP-20 the two agree
    on 9045/9046 test rows (the outlier is ``'HeSiO2'``, whose He has no
    Pauling electronegativity), so Phase 1's MP-20 keys are unaffected.
    """
    from pymatgen.core import Composition

    comp = getattr(formula_or_structure, "composition", None)
    if comp is not None:
        return comp.reduced_formula
    if isinstance(formula_or_structure, (list, tuple, set)):
        return Composition("".join(str(x) for x in formula_or_structure)).reduced_formula
    return Composition(str(formula_or_structure)).reduced_formula


def system_key(formula_or_elements) -> str:
    """Chemical-system key: sorted element symbols joined by '-'.

    'SrTiO3' -> 'O-Sr-Ti'.  Matches run_phase1.HullEvaluator and the keys of
    data/hulls/*_hull_entries.json.
    """
    from pymatgen.core import Composition, Element

    if isinstance(formula_or_elements, (list, tuple, set)):
        symbols = [el.symbol if isinstance(el, Element) else str(el)
                   for el in formula_or_elements]
    else:
        symbols = [el.symbol for el in Composition(formula_or_elements).elements]
    return "-".join(sorted(symbols))


# ---------------------------------------------------------------------------
# Hull table
# ---------------------------------------------------------------------------

@dataclass
class HullTable:
    """Formation-energy entries per chemical system, plus provenance metadata.

    ``systems`` maps a system key ("O-Sr-Ti") to
    ``{"elements": [symbols], "entries": [{"formula", "natoms", "e_form_per_atom"}]}``
    with formation energies referenced to the elemental ground states (so
    elemental endmembers are added at 0 when the diagram is rebuilt).
    """

    name: str
    systems: dict
    meta: dict = field(default_factory=dict)
    source: Optional[Path] = None
    _diagrams: dict = field(default_factory=dict, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, name: Optional[str] = None) -> "HullTable":
        """Load ``<path>`` and its sidecar ``<stem-without-_entries>_meta.json``."""
        path = Path(path)
        systems = json.loads(path.read_text())
        meta_path = Path(str(path).replace("_entries.json", "_meta.json"))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(name=name or path.stem, systems=systems, meta=meta, source=path)

    @classmethod
    def for_dataset(cls, dataset: str) -> "HullTable":
        """Load the frozen hull table of a benchmark dataset (name or key)."""
        from ..data.registry import DATASETS

        spec = DATASETS[dataset]
        return cls.load(spec.hull_entries, name=spec.key)

    # -- queries ------------------------------------------------------------

    def __contains__(self, system: str) -> bool:
        return system in self.systems

    def __len__(self) -> int:
        return len(self.systems)

    @property
    def n_phases(self) -> int:
        return sum(len(v["entries"]) for v in self.systems.values())

    def elements(self, system: str) -> list[str]:
        return list(self.systems[system]["elements"])

    def phase_diagram(self, system: str):
        """Cached pymatgen PhaseDiagram of one chemical system."""
        if system in self._diagrams:
            return self._diagrams[system]
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        rec = self.systems[system]
        entries = []
        for ent in rec["entries"]:
            comp = Composition(ent["formula"])
            entries.append(ComputedEntry(
                comp, float(ent["e_form_per_atom"]) * comp.num_atoms))
        for el in rec["elements"]:
            entries.append(ComputedEntry(Composition(el), 0.0))
        diagram = PhaseDiagram(entries)
        self._diagrams[system] = diagram
        return diagram

    def e_above_hull(self, formula: str, e_form_per_atom: float) -> Optional[float]:
        """E_hull in eV/atom of a phase of the given formation energy, or None
        if its chemical system is not covered by this table.

        The entry energy is the *total* energy for one formula unit
        (``e_form_per_atom * comp.num_atoms``); pymatgen renormalizes by
        ``comp.num_atoms`` internally, so the per-atom comparison -- and only
        that -- is what the hull uses.

        ``allow_negative=True`` is load-bearing.  pymatgen computes
        ``E_hull = E_entry - E_hull_at_composition`` and, with its default
        ``allow_negative=False``, *raises* "No valid decomposition found" when
        that value is negative -- the decomposition having been found.  A
        negative E_hull is the interesting case here: the query is more stable
        than everything in the reference set at its composition, i.e. a
        generated structure that beats the dataset (the dataset's own phases
        cannot be it -- the hull is built from them, so each composition's best
        phase sits *on* the hull at exactly 0.0).  Without the flag those
        structures come back as None and are silently dropped from every
        distribution -- a bias against exactly the result the benchmark is
        looking for.
        """
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        system = system_key(formula)
        if system not in self.systems:
            return None
        comp = Composition(formula)
        entry = ComputedEntry(comp, float(e_form_per_atom) * comp.num_atoms)
        try:
            return float(self.phase_diagram(system).get_e_above_hull(
                entry, allow_negative=True))
        except (ValueError, KeyError, RuntimeError) as exc:
            # A real coverage gap (e.g. the composition sits outside the
            # diagram's simplex), unlike the not-covered case above: say so
            # instead of returning None in silence.
            warnings.warn(f"E_hull unavailable for {formula} ({system}): {exc}",
                          stacklevel=2)
            return None


def load_hull(path_or_dataset: str | Path, name: Optional[str] = None) -> HullTable:
    """Load a hull table from a path, or by dataset name ("mp_20"/"perov_5"/...)."""
    p = Path(path_or_dataset)
    if p.exists():
        return HullTable.load(p, name=name)
    return HullTable.for_dataset(str(path_or_dataset))


# ---------------------------------------------------------------------------
# NNP energy -> formation energy -> E_hull
# ---------------------------------------------------------------------------

class HullEvaluator:
    """Formation energies and E_hull on a dataset's frozen hull.

    Same contract as ``run_phase1.HullEvaluator`` (including the eV/atom
    units fix), extended with the explicit formation-energy accessor
    the formation-energy metric needs.
    """

    def __init__(self, table: HullTable):
        self.table = table
        self._ref_calib: dict[str, float] = {}       # formula -> eV/atom offset
        self._ref_nnp_energy: dict[str, float] = {}

    @classmethod
    def for_dataset(cls, dataset: str) -> "HullEvaluator":
        return cls(HullTable.for_dataset(dataset))

    # -- calibration --------------------------------------------------------

    def calibrate(self, formula: str, nnp_energy_ref: float, n_atoms_ref: int,
                  e_form_ref_per_atom: float) -> None:
        """Anchor the NNP energy scale to the DFT formation-energy scale.

        ``e_form_ref_per_atom`` is the DFT formation energy of the reference
        structure (per atom, eV), from the dataset metadata: MP-20's
        ``formation_energy_per_atom``, Perov-5's ``heat_all / n_atoms``, or
        Carbon-24's ``energy_per_atom - energy_zero``.

        ``formula`` is canonicalized (see ``canonical_formula``), so callers may
        pass a dataset's raw label.
        """
        formula = canonical_formula(formula)
        self._ref_calib[formula] = nnp_energy_ref / n_atoms_ref - e_form_ref_per_atom
        self._ref_nnp_energy[formula] = nnp_energy_ref

    @property
    def calibrated_formulas(self) -> frozenset[str]:
        return frozenset(self._ref_calib)

    def e_form_per_atom(self, formula: str, nnp_energy: float,
                        n_atoms: int) -> Optional[float]:
        """NNP energy -> DFT-scale formation energy per atom (eV/atom)."""
        formula = canonical_formula(formula)
        if formula not in self._ref_calib:
            return None
        return float(nnp_energy) / n_atoms - self._ref_calib[formula]

    def e_hull(self, formula: str, nnp_energy: float,
               n_atoms: int) -> Optional[float]:
        """E_above_hull in eV/atom of an NNP-relaxed structure, or None."""
        e_form = self.e_form_per_atom(formula, nnp_energy, n_atoms)
        if e_form is None:
            return None
        return self.table.e_above_hull(formula, e_form)

    def is_stable(self, formula: str, nnp_energy: float, n_atoms: int,
                  threshold: float = E_HULL_STABLE,
                  floor: float = E_HULL_FLOOR) -> Optional[bool]:
        """S.U.N. 'Stable' predicate.

        Two-sided now: ``-floor <= E_hull < threshold`` (default
        50 meV/atom below the hull to 100 meV/atom above it).  See
        E_HULL_FLOOR for why the published one-sided rule cannot be used on
        datasets where the NNP can diverge.
        """
        e_hull = self.e_hull(formula, nnp_energy, n_atoms)
        if e_hull is None:
            return None
        return bool(-floor <= e_hull < threshold)


# ---------------------------------------------------------------------------
# Standalone helpers (paper section 6.0.7)
# ---------------------------------------------------------------------------

def build_phase_diagram(hull_table: HullTable | dict, system: str):
    """PhaseDiagram of one chemical system from a HullTable (or raw systems dict)."""
    if isinstance(hull_table, HullTable):
        return hull_table.phase_diagram(system)
    return HullTable(name="<dict>", systems=hull_table).phase_diagram(system)


def compute_formation_energy(formula: str, energy: float, n_atoms: int,
                             element_refs: dict[str, float] | None = None,
                             evaluator: HullEvaluator | None = None) -> Optional[float]:
    """Formation energy per atom (eV/atom).

    Two equivalent routes: through a calibrated ``HullEvaluator`` (NNP scale ->
    DFT scale, the pipeline actually used), or directly against a dict of
    per-element chemical potentials ``element_refs`` in the same energy scale
    as ``energy``.
    """
    if evaluator is not None:
        return evaluator.e_form_per_atom(formula, energy, n_atoms)
    if element_refs is None:
        return None
    from pymatgen.core import Composition

    comp = Composition(formula)
    mu = sum(float(element_refs[el.symbol]) * amt
             for el, amt in comp.items())
    return (float(energy) - mu) / n_atoms


def compute_energy_above_hull(formula: str, energy: float, n_atoms: int,
                              hull: HullTable | HullEvaluator,
                              evaluator: HullEvaluator | None = None) -> Optional[float]:
    """E_hull in eV/atom of a structure given its total NNP energy.

    ``hull`` may be a HullTable (then ``evaluator`` must carry the calibration)
    or a calibrated HullEvaluator.
    """
    if isinstance(hull, HullEvaluator):
        return hull.e_hull(formula, energy, n_atoms)
    if isinstance(evaluator, HullEvaluator):
        return evaluator.e_hull(formula, energy, n_atoms)
    raise TypeError("compute_energy_above_hull needs a calibrated HullEvaluator")


def hull_summary(e_hulls: Iterable[Optional[float]]) -> dict:
    """Distribution of E_hull values over valid structures (paper reporting).

    ``frac_stable`` uses the two-sided band of ``is_stable``; ``frac_below_zero``
    and ``n_unphysical`` are reported separately so that a blow-up population is
    visible rather than merely capped.  ``median_physical`` is the median over
    the physical subset -- the number to quote whenever ``n_unphysical > 0``,
    since an unphysical value cannot be averaged with real ones.
    """
    vals = np.array([v for v in e_hulls if v is not None and np.isfinite(v)])
    if vals.size == 0:
        return {"n": 0, "units": E_HULL_UNITS}
    phys = vals[vals >= -E_HULL_FLOOR]
    return {
        "n": int(vals.size),
        "units": E_HULL_UNITS,
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "frac_stable": float(((-E_HULL_FLOOR <= vals)
                              & (vals < E_HULL_STABLE)).mean()),
        "frac_below_zero": float((vals < -1e-6).mean()),
        "n_unphysical": int(vals.size - phys.size),
        "frac_unphysical": float(1.0 - phys.size / vals.size),
        "median_physical": float(np.median(phys)) if phys.size else None,
        "e_hull_floor": E_HULL_FLOOR,
    }
