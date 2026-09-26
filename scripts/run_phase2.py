"""
Phase 2 — the multi-dataset generation-quality benchmark (paper §6.1, App. C).

Grid: three datasets x 20 cells x five seeds x N candidates per cell, taken
over both samplers -- except that PF-ODE runs only PFODE_CONFIGS:

    samplers        ALD (all six configs), PF-ODE (PFODE_CONFIGS only)
    configs         C1, C3, C5, C8, C9, C11 of App. C (C10 available, not in
                    the streamlined subset); PF-ODE runs C5, C8, C11
    datasets        MP-20, Perov-5, Carbon-24 (CDVAE splits)
    cells           20 starting structures per dataset (see "Cell axes")
    seeds           run_phase1.SEEDS = [42, 123, 999, 2024, 7777]
    candidates      run_phase1.N_CAND = 10 (see "Candidate count")

Every cell reuses the frozen Phase-1 protocol code (import run_phase1 as rp):
the same noise schedule, cartesian-vs-fractional noise convention, layer
implementations, drift caps, PF-ODE tolerances and NFE cap, and the same
per-candidate payload shape.  Phase 2 differs in exactly three respects:
lattice updates are on for five of the six configurations, the hull/calibration
come from each dataset's own frozen table (data/hulls/<key>_hull_entries.json),
and the full metric chain of materialgen.eval runs on every cell.

Configurations (docs/paper_outline_v0.4.tex Table tab:param-configs; the
tabulated ALD alpha column is an *ordering scale*, not the implemented step --
every configuration runs the same normalized-score step alpha_0 = 1e-3 with the
3*sigma_k per-atom drift cap, so Config carries no alpha=1e-3 knob):

    C1  baseline              sigma 0.5  beta 38.68  K 100  M 2  al 0.001  upd_lat  bare
    C3  exploratory           sigma 1.5  beta 11.60  K 100  M 2  al 0.002  upd_lat  bare
    C5  fixed_cell            sigma 0.5  beta 38.68  K 100  M 2  al 0.001  no upd   bare
    C8  protected_baseline    = C1 + Layers 1-3                              l1l2l3
    C9  protected_exploratory = C3 + Layers 1-3                              l1l2l3
    C10 protected_fixed_cell  = C5 + Layers 1-3                              l1l2l3
    C11 pauli_only            = C1 + Layer 1 only                            l1

The layer names are the fleet's exact-subset names (run_phase1.LEVEL_SUBSETS):
"l1l2l3" = Layers 1-3, the paper's protected stack, and the optional Layer 4
SafetyMonitor is *not* part of any configuration here.  On the PF-ODE arm L2
and L4 do not exist (no density-noise field, no acceptance semantics), so
"l1l2l3" resolves to L1+L3 there -- exactly the paper's "C1 + Layers 1-3" for a
deterministic sampler.  beta enters through the score (NNPScore(beta=...)), and
the Pauli term takes the *same* beta: the wall's A/B calibration is a force
calibration, so beta weighs the wall and the NNP force equally instead of
silently changing their ratio at 1000 K.

Cell axes (20 per dataset; a cell is one starting structure, noised by
sigma_max in the frozen make_initial):

    MP-20       the 20 chemistry-class-stratified compositions of the frozen
                Phase-1 axis (run_phase1.COMPOSITIONS_20); each cell is that
                composition's lowest-formation-energy test structure.
    Perov-5     20 compositions present in *both* splits, stratified by anion
                class -- see perov5_axis().  The dual-split restriction is what
                makes Novelty non-trivial (otherwise every Perov-5 generation
                is novel for free) and Match rate meaningful.
    Carbon-24   one composition ("C"), variable cell: 20 density strata of the
                test split's volume-per-atom distribution.  lattice_mode is
                "fixed", so update_lattice is forced False for every
                configuration on this dataset (lattice_forced_off=True in the
                payload); C1 == C5 and C8 == C10 there by construction, and the
                analysis reports the column anyway so the identity is visible.

Calibration: every cell is anchored on its *own* reference record -- the
structure the sampler is initialised from, whose DFT formation energy is in the
dataset metadata -- so the E_hull scale of a cell is set locally rather than
through one dataset-wide offset.  On MP-20 and Perov-5 that is identical to the
registry's per-formula anchor (one cell = one composition, verified at context
build); on Carbon-24 it is what keeps the 20 density strata of the single
composition "C" from being scored through the NNP error at one density.

Pooling scope: the paper pools candidates per *test composition*; here the
uniqueness pool is per **cell** (POOL_SCOPE), which is the same thing on MP-20
and Perov-5 (one cell = one composition) and the documented deviation on
Carbon-24 (20 cells of the same composition -- pooling all 1000 structures
under "C" would measure the density strata against each other instead of
measuring the sampler).  The task pass therefore stores only seed-local,
order-independent fields (validity, novelty, E_hull, space group, volume); the
`unique` flag is deliberately *not* written per task, because its value depends
on the pool it is computed in.

Candidate count: N_CAND = run_phase1.N_CAND = 10, the frozen Phase-1 protocol
after the Cholesky rerun halved it for budget.  The paper draft says
"20 candidates per test composition per seed pooled over five seeds to 100 per
composition"; at 10 the pool is 50 per cell.  This is a step-4 (budget) knob --
override with --n-cand 20 to reproduce the draft's wording.  Downstream code
must read n_candidates/n_seeds from the payload and never hardcode either.

Stability is two-sided: Stable = -50 meV/atom <= E_hull < 100
meV/atom, not the published one-sided "E_hull < 100 meV/atom"
(materialgen.eval.stability.E_HULL_FLOOR documents this at length).  The
trigger was the Carbon-24 PF-ODE pilot: with the lattice frozen and a
deterministic sampler there is no acceptance test, so a trajectory can drive a
6-atom cell into the NNP's extrapolation regime -- max|F| = 611 eV/A,
d_min = 0.83 A, NNP E/N = -67.6 eV/atom against a reference at -9.06, i.e.
E_hull = -58.5 eV/atom, which the one-sided rule counts as a *stability
success* and which drags the cell's E_hull median to -31.7 eV/atom.  The
two-sided band scores that as a sampler failure, and every rate is reported
next to its `unphysical_rate` so the affected fraction is always visible.  The
guard is inert on the frozen Phase-1 corpus (0 of 331,470 candidates below -50
meV/atom), so no published number moves.

Output:
    results/phase2/<dataset>/<config>_<sampler>/<cell>_seed<seed>.json
        one per task: {**task, config_spec, cell, ref, candidates, records}
    results/phase2/summary/phase2_<dataset>_<nnp>.json    --analyze
    results/phase2/summary/phase2_<nnp>.csv               --analyze
    data/hulls/<key>_dft_test_stats.json                  --dft-baseline

CPU budget: a `--device cpu` run bounds torch's intra-op pool with
``set_cpu_threads`` (--threads, default DEFAULT_THREADS).  Two effects make an
unbounded CPU run useless for planning: torch's default (one thread per core)
oversubscribes the moment a second process shares the host, and these cells are
2-16 atoms, so intra-op parallelism is pure overhead.  Measured on a 10-atom
MP-20 cell: 0.11 s/NFE at 4-16 threads vs 2.4-3.1 s/NFE at 64 threads on the
*same* structure -- a 6-25x swing.  The second effect is geometric: a reference
cell in a non-standard setting (MP-20's SrTiO3 is the 120-degree rhombohedral
primitive cell) makes the neighbour search 2-5x slower per call than the same
atoms in an orthogonal cell of equal volume.  Both are wall clock only -- the
numbers a task writes are unaffected.

Usage:
    python scripts/run_phase2.py --list
    python scripts/run_phase2.py --nnp mace --shard 0/10
    python scripts/run_phase2.py --nnp mace --pilot
    python scripts/run_phase2.py --nnp mace --analyze
    python scripts/run_phase2.py --dft-baseline

Long runs must be detached (setsid nohup) — see the fleet notes in
docs/experiment_plan_v1.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_phase1 as ap                      # frozen metric definitions
import run_phase1 as rp                          # frozen protocol helpers

from materialgen.core.crystal import CrystalStructure
from materialgen.core.pauli import PauliRepulsion
from materialgen.core.score_function import AugmentedScore, NNPScore, PauliScore
from materialgen.data.registry import DATASET_NAMES, get_dataset
from materialgen.eval.metrics import (StructureEvaluator, compute_amsd,
                                      compute_match_and_coverage, compute_r_angle)
from materialgen.eval.stability import canonical_formula
from materialgen.utils.crystal_io import to_pymatgen

OUT = REPO / "results" / "phase2"
SUMMARY = OUT / "summary"
PROCESSED = REPO / "data" / "processed"
HULLS = REPO / "data" / "hulls"

DATASETS = list(DATASET_NAMES)                   # mp_20, perov_5, carbon_24
N_CELLS = 20
N_CAND = rp.N_CAND                               # 10 -- see "Candidate count"
SEEDS = tuple(rp.SEEDS)                          # 42, 123, 999, 2024, 7777
SAMPLERS = ("ald", "pfode")
POOL_SCOPE = "cell"


# ---------------------------------------------------------------------------
# Configuration table (App. C)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """One row of the App. C parameter table, in the *implemented* convention.

    ``layers`` is a run_phase1.LEVEL_SUBSETS key, not a layer count, so the
    fleet's single resolver (rp.level_subset) stays the only place that maps a
    name to active layers.  ``alpha_lat`` is ALDConfig.alpha_lattice, the
    lattice step; it is unused when update_lattice is False, and ignored
    entirely on the PF-ODE arm (which carries no step size at all).
    """

    name: str
    label: str
    sigma_max: float
    beta: float                 # 1/(k_B T) in 1/eV: 38.68 = 300 K, 11.60 = 1000 K
    K: int                      # ALD noise levels (NFE = K*M)
    M: int                      # Langevin steps per level
    alpha_lat: float            # lattice update step (ALDConfig.alpha_lattice)
    update_lattice: bool
    layers: str                 # run_phase1.LEVEL_SUBSETS key
    purpose: str

    @property
    def nfe(self) -> int:
        return self.K * self.M


CONFIGS = {
    "C1": Config("C1", "baseline", 0.5, 38.68, 100, 2, 0.001, True, "bare",
                 "standard reference point"),
    "C3": Config("C3", "exploratory", 1.5, 11.60, 100, 2, 0.002, True, "bare",
                 "high-noise, high-temperature exploration (completeness boundary)"),
    "C5": Config("C5", "fixed_cell", 0.5, 38.68, 100, 2, 0.001, False, "bare",
                 "primary ablation: are stress-driven lattice updates usable?"),
    "C8": Config("C8", "protected_baseline", 0.5, 38.68, 100, 2, 0.001, True, "l1l2l3",
                 "protective stack at the baseline (H1b)"),
    "C9": Config("C9", "protected_exploratory", 1.5, 11.60, 100, 2, 0.002, True, "l1l2l3",
                 "protective stack at high sigma_max"),
    "C10": Config("C10", "protected_fixed_cell", 0.5, 38.68, 100, 2, 0.001, False, "l1l2l3",
                  "protective stack on the fixed-cell setting"),
    "C11": Config("C11", "pauli_only", 0.5, 38.68, 100, 2, 0.001, True, "l1",
                  "isolate the augmented-score (L1) contribution"),
}
# The streamlined benchmark subset of the draft: C10 is tabulated but excluded.
PHASE2_SUBSET = ("C1", "C3", "C5", "C8", "C9", "C11")

#: Configs whose PF-ODE arm is in the frozen grid; the rest are ALD-only.
#:
#: freeze (docs/phase2_frozen_config.md sections 5-6): the full
#: cartesian grid costs 543 sequential GPU-h, 2.5x the plan's stale ~220
#: estimate, and 71% of it is the PF-ODE arm.  The plan's own "do not cut"
#: clause names PF-ODE x {C5, C8, C11} as the core comparison -- so the ODE arm
#: is exactly those three and the sigma-dependence configs (C1/C3/C9) stay
#: ALD-only; their sigma curves already exist from the Phase-1 10-point scan.
#: Dropping those two ODE arms is what brings the grid to 320 GPU-h.
PFODE_CONFIGS = ("C5", "C8", "C11")


def config_sha1(cfg: Config) -> str:
    """Provenance hash: a task file is only reusable for the identical config."""
    blob = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def config_meta(cfg: Config) -> dict:
    meta = asdict(cfg)
    meta.update({"nfe": cfg.nfe, "layers_subset": sorted(rp.level_subset(cfg.layers)),
                 "alpha_lat": cfg.alpha_lat})
    return meta


# ---------------------------------------------------------------------------
# Cell axes
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One starting structure: the thing a task's trajectories are noised from."""

    id: str                     # unique within the dataset (file name stem)
    formula: str                # canonical reduced formula (hull + record key)
    label: str                  # the dataset's own label for it
    ref_index: int              # index into spec.load_reference(nnp, "test")
    meta: dict = field(default_factory=dict)


def _record_crystal(rec: dict) -> CrystalStructure:
    return CrystalStructure.from_frac_coords(
        rec["numbers"], rec["frac_coords"], rec["lattice"])


def _anchor_table(spec, refs: list[dict]) -> dict:
    """canonical formula -> (index, DFT E_form) of the lowest-E_form record.

    The selection rule must stay identical to registry.DatasetSpec.calib
    (anchor="most_stable"), which is what spec.hull_evaluator() calibrates
    from: the same record has to provide both the initial cell and the
    calibration offset, or every E_hull would carry a silent bias.
    """
    best: dict[str, tuple] = {}
    for i, rec in enumerate(refs):
        meta = rec["metadata"]
        key = canonical_formula(spec.formula(meta))
        e_form = spec.dft_e_form_per_atom(meta, len(rec["numbers"]))
        if key not in best or e_form < best[key][1]:
            best[key] = (i, e_form)
    return best


def mp20_cells(spec, refs: list[dict], n_cells: int = N_CELLS) -> list[Cell]:
    """The frozen Phase-1 composition axis: 20 hand-stratified compositions.

    ``n_cells`` truncates the axis *in COMPOSITIONS_20 order* (smoke/pilot runs
    only; the benchmark runs all 20).  It is honoured here for the same reason
    the other two axes honour it: a smoke run that asks for 2 cells and
    silently gets 20 costs 10x the intended wall clock and writes 20 cells of
    files that then look like real grid output.
    """
    anchors = _anchor_table(spec, refs)
    cells = []
    for comp in rp.COMPOSITIONS_20[:n_cells]:
        key = canonical_formula(comp)
        if key not in anchors:
            raise KeyError(f"MP-20 axis composition {comp} ({key}) has no test record")
        i = anchors[key][0]
        cells.append(Cell(id=key, formula=key, label=spec.formula(refs[i]["metadata"]),
                          ref_index=i))
    return cells


#: Perov-5 anion classes kept as strata: the six largest by pool size.  The
#: A/B sites vary freely in this dataset; the anion sublattice is what its
#: chemistry is organized by (ABX3), so it plays the role MP-20's oxide /
#: sulfide / fluoride / nitride / intermetallic split plays there.
PEROV5_CLASSES = ("NO", "FNO", "N", "FO", "O", "OS")
PEROV5_ANIONS = frozenset("O N F S Cl Br I Se Te C P".split())
PEROV5_AXIS_CACHE = PROCESSED / "perov_5" / "phase2_axis.json"


def _anion_class(formula: str) -> str:
    from pymatgen.core import Composition

    return "".join(sorted(el.symbol for el in Composition(formula).elements
                          if el.symbol in PEROV5_ANIONS))


def _spread(items: list, k: int) -> list:
    """k items spread evenly over ``items`` by rank (deterministic, in order)."""
    if k >= len(items):
        return list(items)
    idx = [int(round(x)) for x in np.linspace(0, len(items) - 1, k)]
    return [items[i] for i in sorted(set(idx))]


def perov5_axis(spec, n_cells: int = N_CELLS, use_cache: bool = True) -> list[str]:
    """Dual-split Perov-5 compositions, stratified by anion class.

    Pool: compositions appearing in *both* the train and the test CSV
    (2,214 of 3,409 test compositions currently; every one of them has
    exactly one train and one test record, so there is no frequency gradient to
    rank by).  Keyed on canonical formulas -- the CSVs' own ``formula`` column
    orders the same composition differently in the two splits ('CoTlON2' vs
    'TiOsOFN'), so the raw strings have a literal zero intersection.

    Within each class the compositions are ordered by their best test
    formation energy and ``k`` are taken evenly spaced over that rank order, so
    each class contributes both its stable and its marginal members instead of
    a cherry-picked tail.  20 = 6 classes x 3 + 2 extra to the two largest.

    The result is NNP-independent (split membership + DFT metadata only) and
    cached under data/processed/perov_5/phase2_axis.json with its rule.
    """
    if use_cache and PEROV5_AXIS_CACHE.exists():
        blob = json.loads(PEROV5_AXIS_CACHE.read_text())
        if blob.get("rule") == _axis_rule() and len(blob.get("cells", [])) == n_cells:
            return list(blob["cells"])

    import pandas as pd

    train = pd.read_csv(spec.csv_dir / "train.csv")
    test = pd.read_csv(spec.csv_dir / "test.csv")
    in_train = {canonical_formula(f) for f in train["formula"]}

    # best (lowest) DFT formation energy per dual-split composition, from the
    # preprocessed test split -- pure metadata arithmetic, no NNP involved.
    best: dict[str, float] = {}
    for rec in spec.processed("test"):
        meta = rec["metadata"]
        key = canonical_formula(spec.formula(meta))
        if key not in in_train:
            continue
        e_form = spec.dft_e_form_per_atom(meta, len(rec["numbers"]))
        if key not in best or e_form < best[key]:
            best[key] = e_form

    by_class: dict[str, list[str]] = {}
    for key in best:
        by_class.setdefault(_anion_class(key), []).append(key)
    for key in by_class:
        by_class[key].sort(key=lambda k: (best[k], k))

    classes = [c for c in PEROV5_CLASSES if c in by_class]
    missing = [c for c in PEROV5_CLASSES if c not in by_class]
    if missing:
        raise KeyError(f"Perov-5 axis: expected anion classes {missing} absent")
    sizes = {c: len(by_class[c]) for c in classes}
    k_per_class = {c: n_cells // len(classes) for c in classes}
    for c in sorted(classes, key=lambda c: (-sizes[c], c))[:n_cells % len(classes)]:
        k_per_class[c] += 1

    cells: list[str] = []
    for c in classes:
        for key in _spread(by_class[c], k_per_class[c]):
            if key not in cells:
                cells.append(key)
    if len(cells) != n_cells:
        # _spread de-duplicates; top up deterministically from the largest class
        pool = [k for c in classes for k in by_class[c] if k not in cells]
        cells.extend(pool[:n_cells - len(cells)])
    cells = cells[:n_cells]

    PEROV5_AXIS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PEROV5_AXIS_CACHE.write_text(json.dumps({
        "rule": _axis_rule(), "cells": cells, "n_pool": len(best),
        "class_sizes": sizes, "k_per_class": k_per_class,
        "class_e_form_range": {c: [round(min(best[k] for k in by_class[c]), 4),
                                   round(max(best[k] for k in by_class[c]), 4)]
                               for c in classes},
        "cell_e_form": {k: round(best[k], 4) for k in cells},
        "source": "train.csv + test.csv formula columns, DFT E_form from test.pkl",
    }, indent=1))
    return cells


def _axis_rule() -> str:
    return ("perov5: dual-split (train & test) canonical compositions, "
            "anion-class stratified, evenly spaced by DFT formation-energy rank")


def perov5_cells(spec, refs: list[dict], n_cells: int = N_CELLS) -> list[Cell]:
    anchors = _anchor_table(spec, refs)
    cells = []
    for key in perov5_axis(spec, n_cells=n_cells):
        if key not in anchors:
            raise KeyError(f"Perov-5 axis composition {key} has no test record")
        i = anchors[key][0]
        cells.append(Cell(id=key, formula=key, label=spec.formula(refs[i]["metadata"]),
                          ref_index=i,
                          meta={"e_form_ref_per_atom":
                                round(spec.dft_e_form_per_atom(
                                    refs[i]["metadata"], len(refs[i]["numbers"])), 4)}))
    return cells


def carbon24_cells(spec, refs: list[dict], n_cells: int = N_CELLS) -> list[Cell]:
    """20 density strata of the test split's volume-per-atom distribution.

    Carbon-24 has a single composition, so its axis cannot be a composition
    list: the dataset's own degree of freedom is the cell (6-24 atoms,
    volume/atom 5.43-8.92 A^3 with 1,837 distinct values in the test split),
    and the released data does not realize the "24 densities" reading either.
    Each stratum is the test record nearest the stratum's median volume/atom,
    ties broken by (atom count, index) so the axis is reproducible.
    """
    vol_per_atom = np.array([float(np.linalg.det(r["lattice"])) / len(r["numbers"])
                             for r in refs])
    edges = np.quantile(vol_per_atom, np.linspace(0.0, 1.0, n_cells + 1))
    cells, used = [], set()
    for j in range(n_cells):
        lo, hi = edges[j], edges[j + 1]
        # half-open stratum; the last one closes so the maximum is included
        inside = [i for i, v in enumerate(vol_per_atom)
                  if (lo <= v < hi) or (j == n_cells - 1 and v == hi)]
        if not inside:                                   # degenerate stratum
            inside = [min(range(len(refs)),
                          key=lambda i: abs(vol_per_atom[i] - 0.5 * (lo + hi)))]
        target = float(np.median(vol_per_atom[inside]))
        i = min(inside, key=lambda i: (abs(vol_per_atom[i] - target),
                                       len(refs[i]["numbers"]), i))
        while i in used and len(used) < len(refs):       # never emit a duplicate
            inside = [k for k in inside if k not in used] or inside
            i = min(inside, key=lambda k: (abs(vol_per_atom[k] - target),
                                           len(refs[k]["numbers"]), k))
        used.add(i)
        cells.append(Cell(id=f"C_rho{vol_per_atom[i]:.3f}", formula="C", label="C",
                          ref_index=i,
                          meta={"vol_per_atom": round(float(vol_per_atom[i]), 3),
                                "n_atoms": len(refs[i]["numbers"]),
                                "stratum": [round(float(lo), 3), round(float(hi), 3)]}))
    cells.sort(key=lambda c: c.meta["vol_per_atom"])
    return cells


def cells_for(spec, refs: list[dict], n_cells: int = N_CELLS) -> list[Cell]:
    if spec.key == "mp20":
        return mp20_cells(spec, refs, n_cells)
    if spec.key == "perov5":
        return perov5_cells(spec, refs, n_cells)
    if spec.key == "carbon24":
        return carbon24_cells(spec, refs, n_cells)
    raise ValueError(f"no Phase-2 cell axis for dataset {spec.name!r}")


# ---------------------------------------------------------------------------
# Per-dataset context
# ---------------------------------------------------------------------------

class DatasetContext:
    """Everything one dataset contributes to a task: cells, hull, index, refs."""

    def __init__(self, dataset: str, nnp: str, n_cells: int = N_CELLS,
                 need_index: bool = True):
        self.spec = get_dataset(dataset)
        self.name = self.spec.name
        self.nnp = nnp
        self.refs = self.spec.load_reference(nnp, "test")
        self.cells = cells_for(self.spec, self.refs, n_cells)
        self.hull = self.spec.hull_evaluator(nnp, "test")
        self.index = self.spec.training_index("train") if need_index else None
        self.cell_calib: dict[str, dict] = {}
        self._ref_views: dict[str, dict] = {}
        self._verify_calibration()
        self._reference_groups: dict[str, list] = {}
        self._reference_index: dict[str, object] = {}

    def _verify_calibration(self) -> None:
        """Anchor every cell on its own reference record, and check the rule.

        The calibration offset (E_NNP(ref)/N_ref - E_form^DFT(ref)) is taken
        from the *cell's own* reference structure -- the same structure the
        sampler is initialised from, whose DFT formation energy is in the
        dataset metadata.  That is one uniform rule across the three datasets:

            MP-20 / Perov-5   the cell IS its composition's most-stable test
                              record, so the per-cell offset is by construction
                              the registry's per-formula anchor (verified below;
                              a divergence here is a cell-selection bug);
            Carbon-24         the 20 cells are density strata of the *same*
                              composition, so a per-formula anchor would score
                              every stratum through the NNP error at one
                              density.  Measured: the carbon
                              offsets span 169 meV/atom across the 20 strata --
                              more than the 100 meV/atom stability threshold --
                              so a shared anchor would tilt the carbon E_hull
                              columns by stratum instead of measuring them.
                              Per-cell anchoring removes that tilt (each cell's
                              own anchor scores exactly its DFT E_form).

        Hence the Carbon-24 check is not equality but finiteness plus a report;
        the spread is a measured property of a single-element hull, not a bug.
        """
        calib = {r["formula_canonical"]: r for r in self.spec.calib(self.nnp, "test")}
        self.calib = calib
        deltas = []
        for cell in self.cells:
            rec = self.refs[cell.ref_index]
            meta, n_atoms = rec["metadata"], len(rec["numbers"])
            anchor = calib.get(cell.formula)
            if anchor is None:
                raise KeyError(f"{self.name}: no calibration anchor for "
                               f"{cell.formula!r}")
            cc = {
                "formula": cell.formula,
                "e_ref": float(rec["energy"]), "n_atoms_ref": n_atoms,
                "e_form_ref_per_atom": float(self.spec.dft_e_form_per_atom(meta, n_atoms)),
                "material_id": meta.get("material_id"),
            }
            if not all(np.isfinite([cc["e_ref"], cc["e_form_ref_per_atom"]])):
                raise ValueError(f"{self.name}/{cell.id}: non-finite anchor "
                                 f"{cc['e_ref']}, {cc['e_form_ref_per_atom']}")
            self.cell_calib[cell.id] = cc
            # per-atom NNP energy gap to the registry's formula anchor
            delta = abs(cc["e_ref"] / n_atoms
                        - anchor["nnp_energy_ref"] / anchor["n_atoms_ref"])
            deltas.append((cell.id, delta * 1000.0))
        worst = max(d for _, d in deltas)
        if self.spec.key == "carbon24":
            off = [c["e_ref"] / c["n_atoms_ref"] - c["e_form_ref_per_atom"]
                   for c in self.cell_calib.values()]
            print(f"[ctx] {self.name}: per-cell offset spans "
                  f"{max(off) - min(off):.3f} eV/atom over the density strata "
                  f"(NNP per-atom energy spans {worst:.3f}); anchoring per cell",
                  flush=True)
        elif worst > 1e-6:
            raise ValueError(
                f"{self.name}/{max(deltas, key=lambda x: x[1])[0]}: cell anchor is not "
                f"the registry's most-stable anchor (gap {worst:.4f} meV/atom) -- "
                "cell selection and hull calibration have diverged")

    # -- accessors ----------------------------------------------------------

    def cell_ref(self, cell: Cell) -> dict:
        """The cell's reference record, with an MP-20-schema formula alias.

        run_phase1.run_candidate resolves the E_hull query formula as
        ``final.properties.get("formula") or ref_record["metadata"]["pretty_formula"]``
        -- the MP-20 schema, which Perov-5 (``formula``) and Carbon-24 (no
        formula key at all) do not follow.  Rather than edit the frozen Phase-1
        candidate protocol, each cell hands it a copy of its reference record
        whose metadata carries ``pretty_formula`` as an alias for the dataset's
        own label; the hull evaluator canonicalizes it either way.
        """
        if cell.id not in self._ref_views:
            rec = dict(self.refs[cell.ref_index])
            rec["metadata"] = {**rec["metadata"], "pretty_formula": cell.label}
            self._ref_views[cell.id] = rec
        return self._ref_views[cell.id]

    def calibrate_cell(self, cell: Cell) -> dict:
        """Point the hull evaluator at this cell's own anchor; return it.

        HullEvaluator keeps one offset per formula, so this must run before
        every task of a cell that shares its formula with another (Carbon-24).
        """
        cc = self.cell_calib[cell.id]
        self.hull.calibrate(cc["formula"], cc["e_ref"], cc["n_atoms_ref"],
                            cc["e_form_ref_per_atom"])
        return cc

    def reference_structures(self, cell: Cell) -> list:
        """All test-split structures of the cell's composition (metric reference)."""
        if cell.formula not in self._reference_groups:
            self._reference_groups[cell.formula] = [
                to_pymatgen(_record_crystal(r)) for r in self.refs
                if canonical_formula(self.spec.formula(r["metadata"])) == cell.formula]
        return self._reference_groups[cell.formula]

    def reference_index(self, cell: Cell):
        """ReferenceIndex for the cell (built once, reused across configs).

        Load-bearing on Carbon-24, where a cell's reference is every test
        structure of composition "C": the plain helpers run a quadratic
        ``group_structures`` scan over it, once per config per sampler.
        """
        from materialgen.eval.metrics import ReferenceIndex

        if cell.id not in self._reference_index:
            self._reference_index[cell.id] = ReferenceIndex(
                self.reference_structures(cell))
        return self._reference_index[cell.id]


_CONTEXTS: dict[tuple, DatasetContext] = {}


def context(dataset: str, nnp: str, n_cells: int = N_CELLS,
            need_index: bool = True) -> DatasetContext:
    key = (dataset, nnp, n_cells, need_index)
    if key not in _CONTEXTS:
        t0 = time.time()
        _CONTEXTS[key] = DatasetContext(dataset, nnp, n_cells, need_index)
        ctx = _CONTEXTS[key]
        print(f"[ctx] {dataset}/{nnp}: {len(ctx.cells)} cells, "
              f"{len(ctx.hull.calibrated_formulas)} calibrated formulas, "
              f"{time.time() - t0:.1f}s", flush=True)
    return _CONTEXTS[key]


# ---------------------------------------------------------------------------
# Score / sampler construction (Phase-2 overrides of the frozen builders)
# ---------------------------------------------------------------------------

def effective_update_lattice(cfg: Config, spec) -> bool:
    """Dataset lattice_mode wins: Carbon-24 is fixed-cell in every config."""
    return bool(cfg.update_lattice and spec.lattice_mode == "variable")


#: Intra-op threads for a CPU NNP run.  torch defaults to one thread per *core*
#: (16 here, 64 on the 128-core host), which oversubscribes as soon as two
#: runners share the box -- and these cells are 2-16 atoms, far too small for
#: intra-op parallelism to pay.  Measured on a 10-atom MP-20 cell
#: with the host at load ~170 (four dmlm trainings, eleven VASP jobs, the
#: Phase-1 fleet): 0.11 s/call at 4-16 threads vs 2.4-3.1 s/call at 64.  A
#: 6-25x wall-clock swing on the *same* structure, so an unbounded run makes
#: every budget number meaningless.  See "CPU budget" in the module docstring.
DEFAULT_THREADS = 8


def set_cpu_threads(n: int) -> int:
    """Bound torch/BLAS intra-op threads for the CPU path; returns what was set.

    The env vars are set with ``setdefault`` so an explicit ``OMP_NUM_THREADS``
    in the caller's environment still wins, and they are set *before* torch is
    imported (``load_calculator`` imports it lazily) because OpenMP reads them
    at library load.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(n, 4)))
    except RuntimeError:
        pass        # already started parallel work; intra-op bound still applies
    return n


def make_score_p2(calc, cfg: Config, update_lattice: bool, nnp: str) -> object:
    """Score of one configuration: NNP (+ L1 wall when the subset has layer 1).

    compute_stress is enabled only where the lattice can move -- the stress
    tensor is what drives the cell update, and asking for it costs a second
    backward-style pass on every evaluation.
    """
    nnp_score = NNPScore(calc, beta=cfg.beta, compute_stress=update_lattice,
                         label=rp.nnp_label(nnp))
    if 1 not in rp.level_subset(cfg.layers):
        return nnp_score
    pauli = PauliScore(pauli=PauliRepulsion(params=None, deep_wall=False),
                       beta=cfg.beta)
    return AugmentedScore(nnp_score, pauli)


def make_sampler_p2(cfg: Config, update_lattice: bool, sampler_name: str):
    """Sampler of one configuration, built on the frozen rp.make_* builders.

    Reusing rp.make_ald_config keeps every layer flag (cap v3, the probe-driven
    L1' calibration, the L2 density knobs, L3) byte-identical to the fleet's;
    only the configuration-table fields are overwritten here.  PF-ODE has no
    K/M/alpha, so only its lattice switch differs from rp.make_pfode_config.
    """
    if sampler_name == "ald":
        ald_cfg = rp.make_ald_config(cfg.sigma_max, cfg.layers)
        ald_cfg.K, ald_cfg.M = cfg.K, cfg.M
        ald_cfg.alpha_lattice = cfg.alpha_lat
        ald_cfg.update_lattice = update_lattice
        return rp.make_ald_sampler(ald_cfg, cfg.layers)
    if sampler_name == "pfode":
        pf_cfg = rp.make_pfode_config(cfg.sigma_max, cfg.layers)
        pf_cfg.update_lattice = update_lattice
        return rp.make_pfode_sampler(pf_cfg)
    raise ValueError(f"unknown sampler {sampler_name!r}")


# ---------------------------------------------------------------------------
# Task grid
# ---------------------------------------------------------------------------

def configs_for(sampler: str, configs=PHASE2_SUBSET) -> tuple:
    """The configs whose ``sampler`` arm is in the frozen grid.

    ALD runs the whole subset; PF-ODE runs PFODE_CONFIGS only.
    """
    if sampler == "pfode":
        return tuple(c for c in configs if c in PFODE_CONFIGS)
    return tuple(configs)


def build_tasks(nnp: str, datasets=DATASETS, configs=PHASE2_SUBSET,
                samplers=SAMPLERS, seeds=SEEDS, n_cells: int = N_CELLS):
    """[(task, out_path)] in a deterministic order (stable across shards).

    Task construction touches the registry for cell ids only (no calculator),
    so listing the grid never loads a model.
    """
    tasks = []
    for dataset in datasets:
        spec = get_dataset(dataset)
        cells = cells_for(spec, spec.load_reference(nnp, "test"), n_cells)
        for sampler in samplers:
            for cfg_name in configs_for(sampler, configs):
                for cell in cells:
                    for seed in seeds:
                        task = {"dataset": dataset, "nnp": nnp, "config": cfg_name,
                                "sampler": sampler, "cell": cell.id, "seed": seed}
                        tasks.append((task, OUT / dataset / f"{cfg_name}_{sampler}"
                                      / f"{cell.id}_seed{seed}.json"))
    return tasks


#: CUDA failures that poison the whole context rather than one call.  Once one
#: of these fires, every later CUDA op in the process fails with the same
#: message, so the only useful response is to restart the process.
STICKY_CUDA_PATTERNS = (
    "illegal memory access",
    "device-side assert",
    "an illegal instruction was encountered",
    "unspecified launch failure",
    "out of memory",          # torch's own text: "CUDA out of memory. ..."
)


def _is_sticky_cuda_error(exc: BaseException) -> bool:
    """True if ``exc`` means the CUDA context is dead, not just this call."""
    msg = str(exc).lower()
    return "cuda" in msg and any(p.lower() in msg for p in STICKY_CUDA_PATTERNS)


def _task_ok(path: Path, cfg: Config, n_cand: Optional[int] = None) -> bool:
    """A finished file is reusable only if it came from the same protocol+config.

    ``n_cand`` is checked when given: the candidate count is a grid parameter
    (--n-cand), so a 3-candidate file left by a smoke run must not be accepted
    as "done" by a 10-candidate fleet run -- the analysis pools candidates per
    cell, and a mixed pool silently weights that cell differently.
    """
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return False
    if blob.get("init_cell_gen") != rp.INIT_CELL_GEN:
        return False
    if blob.get("config_sha1") != config_sha1(cfg):
        return False
    return n_cand is None or blob.get("n_candidates") == n_cand


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def record_dict(rec) -> dict:
    """Seed-local, order-independent fields of one StructureRecord.

    ``unique`` is intentionally absent: it is defined against a pool, so it is
    computed in the analysis pass over the pooled cell (see POOL_SCOPE).
    ``stable`` is absent for the same reason ``physical`` is present: both are
    pure functions of ``e_hull`` (two-sided band, see E_HULL_FLOOR), so storing
    ``e_hull`` is enough and the analysis re-derives them from disk.
    """
    return {"formula": rec.formula, "energy": rec.energy, "e_form": rec.e_form,
            "e_hull": rec.e_hull, "spacegroup": rec.spacegroup,
            "volume": rec.volume, "valid": rec.valid, "novel": rec.novel,
            "physical": rec.physical}


def run_task(task: dict, ctx: DatasetContext, calc) -> dict:
    cfg = CONFIGS[task["config"]]
    cell = next(c for c in ctx.cells if c.id == task["cell"])
    ref_record = ctx.cell_ref(cell)
    upd_lat = effective_update_lattice(cfg, ctx.spec)
    calib = ctx.calibrate_cell(cell)            # hull anchored on this cell

    score = make_score_p2(calc, cfg, upd_lat, task["nnp"])
    sampler = make_sampler_p2(cfg, upd_lat, task["sampler"])

    candidates = []
    for cand in range(N_CAND):
        c = rp.run_candidate(score, sampler, ref_record, cfg.sigma_max,
                             task["seed"], cand, ctx.hull)
        c["cell"] = cell.id
        candidates.append(c)

    # --- metric chain over this seed's ensemble ---------------------------
    # The structures are round-tripped through to_dict/from_dict so the
    # analysis pass reconstructs exactly what is scored here.
    structures, energies = [], []
    for c in candidates:
        structures.append(CrystalStructure.from_dict(c["final"]["structure"]))
        energies.append(c["final"]["energy"])
    formulas = [cell.formula] * len(structures)
    records = StructureEvaluator(training=ctx.index, hull=ctx.hull,
                                 formula=cell.formula).build_records(
        structures, energies, formulas)
    for c, rec in zip(candidates, records):
        c["valid_chain"] = rec.valid            # metric-chain validity
        c["novel"] = rec.novel

    n_mismatch = sum(1 for c in candidates if bool(c["valid"]) != bool(c["valid_chain"]))
    return {
        **task,
        "n_atoms": len(ref_record["numbers"]),
        "n_candidates": N_CAND,
        "e_hull_units": "eV/atom",              # units convention
        "init_cell_gen": rp.INIT_CELL_GEN,      # initial-cell generation marker
        "config_sha1": config_sha1(cfg),
        "config_spec": config_meta(cfg),
        "effective_update_lattice": upd_lat,
        "lattice_forced_off": bool(cfg.update_lattice and not upd_lat),
        "cell": asdict(cell),
        # The anchor this task's E_hull scale is built on: the cell's own
        # reference record, not the dataset's formula anchor (they coincide on
        # MP-20/Perov-5; on Carbon-24 this is the per-stratum offset).
        "ref": {"index": cell.ref_index, "label": cell.label,
                "formula": calib["formula"], "material_id": calib["material_id"],
                "e_ref": calib["e_ref"], "n_atoms_ref": calib["n_atoms_ref"],
                "e_form_ref_per_atom": calib["e_form_ref_per_atom"],
                "calib_offset_per_atom": calib["e_ref"] / calib["n_atoms_ref"]
                - calib["e_form_ref_per_atom"]},
        "n_validity_definition_mismatch": n_mismatch,
        "candidates": candidates,
        "records": [record_dict(r) for r in records],
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _load_tasks(dataset: str, config: str, sampler: str) -> list[dict]:
    cfg = CONFIGS[config]
    d = OUT / dataset / f"{config}_{sampler}"
    if not d.exists():
        return []
    out = []
    for path in sorted(d.glob("*.json")):
        if _task_ok(path, cfg):
            try:
                out.append(json.loads(path.read_text()))
            except Exception as exc:
                print(f"  WARN unreadable {path}: {exc}")
    return out


def _mean_std(vals: list) -> dict:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "n": len(vals), "values": [float(v) for v in vals]}


def _pooled_records(cands: list[dict], records: list[dict], formula: str) -> list:
    """StructureRecord objects of a pooled cell, with uniqueness marked.

    Rebuilt from the stored payloads rather than from live objects: the
    analysis pass runs on whatever is on disk, and this is the only way the
    reported numbers are guaranteed to be the ones the files support.
    """
    from materialgen.eval.metrics import StructureRecord, mark_unique

    out = []
    for c, r in zip(cands, records):
        rec = StructureRecord(formula=formula,
                              structure=CrystalStructure.from_dict(c["final"]["structure"]),
                              energy=r["energy"], e_hull=r["e_hull"],
                              e_form=r["e_form"], spacegroup=r["spacegroup"],
                              volume=r["volume"], valid=r["valid"], novel=r["novel"])
        out.append(rec)
    mark_unique(out)
    return out


def analyze_cell(tasks: list[dict], ctx: DatasetContext, cell: Cell) -> dict:
    """Every metric of one cell, pooled over its seeds (POOL_SCOPE)."""
    from materialgen.eval.metrics import compute_sun
    from materialgen.eval.stability import E_HULL_FLOOR, hull_summary

    tasks = sorted(tasks, key=lambda t: t["seed"])
    cands = [c for t in tasks for c in ap.cands_of(t)]
    records = [r for t in tasks for r in t["records"]]
    formula = tasks[0]["cell"]["formula"]
    pool = _pooled_records(cands, records, formula)
    sun = compute_sun(pool)

    structures = [to_pymatgen(p.structure) for p in pool]
    valid_structs = [to_pymatgen(p.structure) for p in pool if p.valid]
    reference = ctx.reference_structures(cell)

    # E_hull summary: raw and physical-only.  The two differ only where the
    # sampler drove the NNP out of its training distribution (see the
    # E_HULL_FLOOR note); `median_physical` is the one to quote whenever
    # n_unphysical > 0, since an extrapolation cannot be averaged with real
    # energies.  Both are stored so the difference is auditable per cell.
    eh_valid = [p.e_hull for p in pool if p.valid and p.e_hull is not None]
    unphys = [p for p in pool if p.physical is False]

    out = {"n": len(pool), "n_seeds": len(tasks), "seed_list": [t["seed"] for t in tasks],
           "sun": sun, "dynamics": ap.cell_metrics(cands),
           "e_hull_valid": hull_summary(eh_valid),
           "e_hull_valid_physical": hull_summary(
               [p.e_hull for p in pool if p.valid and p.physical]),
           "unphysical": {
               "n": len(unphys),
               "rate": len(unphys) / len(pool) if pool else 0.0,
               "e_hull_min": min((p.e_hull for p in unphys), default=None),
               "e_hull_median": (float(np.median([p.e_hull for p in unphys]))
                                 if unphys else None),
               "e_hull_floor": E_HULL_FLOOR,
           },
           "n_validity_definition_mismatch": sum(
               t.get("n_validity_definition_mismatch", 0) for t in tasks)}

    if reference:
        # Two scopes, so the third-party adapter step (step 2 of the plan) can
        # align with whatever the official DiffCSP/CDVAE script does without a
        # re-run: the validity-filtered (primary) and the unfiltered one.
        # ReferenceIndex returns exactly what compute_match_and_coverage /
        # compute_r_angle do (same keys, same numbers) while bucketing by
        # composition and site count -- see the class docstring for why that
        # is exact, and why Carbon-24 cannot be analysed without it.
        ref_index = ctx.reference_index(cell)
        out["match_coverage"] = ref_index.match_coverage(valid_structs, formula=formula)
        out["match_coverage_all"] = ref_index.match_coverage(structures, formula=formula)
        out["r_angle_kl"] = ref_index.r_angle(structures, "kl")
        out["r_angle_wasserstein"] = ref_index.r_angle(structures, "wasserstein")
    if len(valid_structs) >= 2:
        out["amsd"] = compute_amsd(valid_structs)

    # Per-seed scatter (error bars): validity/E_hull/OOD are seed-local, so
    # they can be reported per seed as-is.  n_sun is not -- it uses the seed's
    # own candidate pool, which is a *smaller* uniqueness pool than the
    # reported one, so it is labelled as such rather than compared head to head.
    per_seed = {}
    for t in tasks:
        c_t = ap.cands_of(t)
        m = ap.cell_metrics(c_t)
        seed_pool = _pooled_records(c_t, t["records"], formula)
        s = compute_sun(seed_pool)
        per_seed[str(t["seed"])] = {
            "n": m["n_traj"], "validity": m["validity"],
            "ood_rate": m["ood_rate"], "induced_ood_rate": m["induced_ood_rate"],
            "e_hull_med_meV": m["e_hull_med_meV"],
            "sun_rate_within_seed": s["sun_rate"], "n_sun_within_seed": s["n_sun"],
        }
    out["per_seed"] = per_seed
    return out


def analyze(nnp: str, datasets=DATASETS, configs=PHASE2_SUBSET, samplers=SAMPLERS,
            n_cells: int = N_CELLS) -> dict:
    from materialgen.eval.stability import hull_summary

    SUMMARY.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        ctx = context(dataset, nnp, n_cells=n_cells, need_index=True)
        dft = _dft_stats(ctx.spec)
        per_dataset = {"dataset": dataset, "nnp": nnp, "n_cells": n_cells,
                       "n_seeds": len(SEEDS), "n_candidates": N_CAND,
                       "pool_scope": POOL_SCOPE,
                       "dataset_stats": {"lattice_mode": ctx.spec.lattice_mode,
                                         "dft_test_ceiling": dft},
                       "configs": {}}
        for config in configs:
            for sampler in samplers:
                tasks = _load_tasks(dataset, config, sampler)
                if not tasks:
                    continue
                # A cell's pool is the concatenation of its tasks' candidates,
                # so a task left over from a different --n-cand run would
                # silently weight that cell differently.  Report it (never drop
                # it): the fix is a --force rerun of the odd files.
                counts = collections.Counter(t.get("n_candidates") for t in tasks)
                if len(counts) > 1:
                    print(f"  WARN {dataset}/{config}/{sampler}: mixed candidate "
                          f"counts {dict(counts)} -- re-run the odd files with "
                          "--force before quoting this row", flush=True)
                t0 = time.time()
                by_cell: dict[str, list[dict]] = {}
                for t in tasks:
                    by_cell.setdefault(t["cell"]["id"], []).append(t)
                cells = {c.id: c for c in ctx.cells}
                per_cell = {}
                used = []                       # the tasks that entered the row
                for cid, ts in sorted(by_cell.items()):
                    if cid not in cells:
                        print(f"  WARN {dataset}/{config}/{sampler}: cell {cid} "
                              "not in the current axis -- skipped")
                        continue
                    per_cell[cid] = analyze_cell(ts, ctx, cells[cid])
                    used.extend(ts)

                # Pool the OOD/dynamics side over the analysed cells only.  A
                # task can survive on disk after its cell leaves the axis (a
                # truncated --n-cells smoke leaves 20 cells of files behind a
                # 2-cell axis; an axis edit does the same to a finished grid),
                # and those trajectories must not re-enter the pooled rates
                # while their per-cell rows are being dropped.
                if not per_cell:
                    print(f"  WARN {dataset}/{config}/{sampler}: {len(tasks)} "
                          "task file(s) on disk, none on the current axis -- "
                          "row skipped (delete them or widen --n-cells)",
                          flush=True)
                    continue
                all_cands = [c for t in used for c in ap.cands_of(t)]
                e_hulls = [r["e_hull"] for t in used for r in t["records"]
                           if r.get("valid") and r.get("e_hull") is not None]
                agg = {
                    "n_tasks": len(used), "n_traj": len(all_cands),
                    "n_cells": len(per_cell),
                    "n_tasks_on_disk": len(tasks),
                    "sun_rate": _mean_std([v["sun"]["sun_rate"] for v in per_cell.values()]),
                    "validity_rate": _mean_std([v["sun"]["validity_rate"] for v in per_cell.values()]),
                    "stable_rate": _mean_std([v["sun"]["stable_rate"] for v in per_cell.values()]),
                    "novel_rate": _mean_std([v["sun"]["novel_rate"] for v in per_cell.values()]),
                    "unique_rate": _mean_std([v["sun"]["unique_rate"] for v in per_cell.values()]),
                    "amsd": _mean_std([v.get("amsd") for v in per_cell.values()]),
                    "match_rate": _mean_std([v.get("match_coverage", {}).get("match_rate")
                                             for v in per_cell.values()]),
                    "coverage": _mean_std([v.get("match_coverage", {}).get("coverage")
                                           for v in per_cell.values()]),
                    "r_angle_kl": _mean_std([v.get("r_angle_kl", {}).get("value")
                                             for v in per_cell.values()]),
                    "dynamics": ap.cell_metrics(all_cands),
                    # Unphysical fraction (NNP extrapolation, E_hull below the
                    # two-sided band): the sampler-failure rate that the
                    # one-sided stability rule would have scored as success.
                    "unphysical_rate": _mean_std(
                        [v["sun"]["unphysical_rate"] for v in per_cell.values()]),
                    "e_hull_valid": hull_summary(e_hulls),   # pooled over cells
                    "e_hull_median_over_cells": _mean_std(
                        [v["e_hull_valid_physical"].get("median_physical")
                         for v in per_cell.values()]),
                }
                key = f"{config}_{sampler}"
                per_dataset["configs"][key] = {
                    "config": config, "sampler": sampler,
                    "config_spec": used[0]["config_spec"],
                    "effective_update_lattice": used[0]["effective_update_lattice"],
                    "lattice_forced_off": used[0]["lattice_forced_off"],
                    "candidate_counts": {str(k): v for k, v in sorted(
                        counts.items(), key=lambda kv: (kv[0] is None, kv[0]))},
                    "aggregate": agg, "per_cell": per_cell,
                }
                rows.append(_csv_row(dataset, nnp, config, sampler, agg, len(all_cands)))
                print(f"  {dataset:9s} {key:10s} cells={len(per_cell):2d} "
                      f"traj={len(all_cands):5d} SUN={_fmt(agg['sun_rate']['mean'])} "
                      f"valid={_fmt(agg['validity_rate']['mean'])} "
                      f"match={_fmt(agg['match_rate']['mean'])} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        path = SUMMARY / f"phase2_{dataset}_{nnp}.json"
        path.write_text(json.dumps(per_dataset, indent=1))
        print(f"  -> {path}", flush=True)

    if rows:
        import csv

        csv_path = SUMMARY / f"phase2_{nnp}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  -> {csv_path}", flush=True)
    return {"rows": rows}


def _fmt(v):
    return "  n/a " if v is None else f"{v:.3f}"


def _csv_row(dataset: str, nnp: str, config: str, sampler: str,
             agg: dict, n_traj: int) -> dict:
    out = {"dataset": dataset, "nnp": nnp, "config": config, "sampler": sampler,
           "n_traj": n_traj, "n_cells": agg["n_cells"]}
    for k in ("sun_rate", "validity_rate", "stable_rate", "novel_rate", "unique_rate",
              "amsd", "match_rate", "coverage", "r_angle_kl", "unphysical_rate"):
        out[k] = agg[k]["mean"]
        out[f"{k}_std"] = agg[k]["std"]
    d = agg["dynamics"]
    for k in ("ood_rate", "induced_ood_rate", "start_ood_rate", "validity",
              "collapse_rate", "e_hull_med_meV", "e_hull_mean_meV", "nfe_mean",
              "capped_rate"):
        out[k] = d.get(k)
    # E_hull median over cells, physical-only (identical to the raw one unless
    # unphysical_rate > 0, where the raw median is not a stability number).
    out["e_hull_med_phys_meV"] = (
        agg["e_hull_median_over_cells"]["mean"] * 1000
        if agg["e_hull_median_over_cells"]["mean"] is not None else None)
    out["e_hull_frac_unphysical"] = agg["e_hull_valid"].get("frac_unphysical")
    return out


def _dft_stats(spec) -> dict:
    """The dataset's own DFT-side E_hull distribution (the S.U.N. ceiling).

    Cached: it is NNP-independent and costs ~20 min on MP-20 (one pymatgen
    PhaseDiagram per chemical system).  Generated by --dft-baseline.
    """
    path = HULLS / f"{spec.key}_dft_test_stats.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": f"missing {path.name} -- run: python scripts/run_phase2.py --dft-baseline"}


def dft_baseline(datasets=DATASETS, split: str = "test") -> None:
    """Cache each dataset's own DFT-side E_hull distribution and hull sanity."""
    for dataset in datasets:
        spec = get_dataset(dataset)
        t0 = time.time()
        vals = np.array([v for v in spec.dft_e_hull_all(split) if v is not None], float)
        rec = {
            "split": split, "n": int(vals.size),
            "frac_below_hull": float((vals < -1e-6).mean()),
            "min": float(vals.min()), "p10": float(np.percentile(vals, 10)),
            "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max()),
            "frac_stable": float((vals < 0.1).mean()),
            "units": "eV/atom",
            "note": ("the dataset's own phases scored on its own frozen hull -- "
                     "each composition's best phase sits ON the hull (E_hull=0), "
                     "so frac_stable is the ceiling any generator is measured "
                     "against, not a quality claim about the dataset"),
        }
        HULLS.joinpath(f"{spec.key}_dft_test_stats.json").write_text(
            json.dumps(rec, indent=1))
        print(f"{dataset:9s} n={rec['n']:5d} median {rec['median']:.4f} "
              f"max {rec['max']:.4f} frac_stable {rec['frac_stable']*100:.1f}% "
              f"negatives {rec['frac_below_hull']*100:.2f}% ({time.time()-t0:.0f}s)",
              flush=True)


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

def compute_sun_of(payload: dict) -> dict:
    from materialgen.eval.metrics import compute_sun

    return compute_sun(_pooled_records(payload["candidates"], payload["records"],
                                       payload["cell"]["formula"]))


def pilot(nnp: str, device: str, dataset: str = "mp_20") -> None:
    """One small task per arm on one dataset: exercises the whole chain.

    Uses the module N_CAND (--n-cand 3 for a fast pass): the point is that the
    score/sampler/hull/metric plumbing runs end to end on a real dataset, not
    that the ensemble is large.
    """
    ctx = context(dataset, nnp, need_index=True)
    calc = rp.load_calculator(nnp, device)
    arms = [("C1", "ald"), ("C8", "ald"), ("C1", "pfode")]
    for config, sampler in arms:
        task = {"dataset": dataset, "nnp": nnp, "config": config,
                "sampler": sampler, "cell": ctx.cells[0].id, "seed": SEEDS[0]}
        t0 = time.time()
        payload = run_task(task, ctx, calc)
        sun = compute_sun_of(payload)
        d = ap.cell_metrics(payload["candidates"])
        eh = d["e_hull_med_meV"]
        print(f"PILOT {dataset} {config}/{sampler} cell={payload['cell']['id']} "
              f"({payload['n_atoms']} atoms, upd_lat={payload['effective_update_lattice']}) "
              f"{time.time()-t0:.1f}s", flush=True)
        print(f"  validity {d['validity']:.2f} ood {d['ood_rate']:.2f} "
              f"(induced {d['induced_ood_rate']:.2f}) e_hull_med "
              f"{'n/a' if eh is None else f'{eh/1000:.4f}'} eV/atom "
              f"| SUN pooled {sun['sun_rate']:.2f} "
              f"(S {sun['stable_rate']:.2f} U {sun['unique_rate']:.2f} "
              f"N {sun['novel_rate']:.2f})", flush=True)
        if sun.get("n_unphysical"):
            print(f"  !! unphysical (NNP extrapolation) {sun['n_unphysical']}/"
                  f"{sun['n']} = {sun['unphysical_rate']:.2f}, "
                  f"E_hull median {sun['unphysical_e_hull_median']:.3f} eV/atom "
                  f"(min {sun['unphysical_e_hull_min']:.3f}) -- excluded from "
                  f"Stable", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # The CLI rebinds the module-level grid constants (the same pattern
    # run_phase1 uses for --mini): build_tasks and run_task read them, so a
    # resume with different --n-cand/--seeds must not silently mix grids.
    global N_CAND, SEEDS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--nnp", default="mace", choices=["mace", "esen"])
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--configs", default=",".join(PHASE2_SUBSET))
    p.add_argument("--samplers", default=",".join(SAMPLERS))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--n-cells", type=int, default=N_CELLS)
    p.add_argument("--n-cand", type=int, default=N_CAND,
                   help="candidates per (cell, seed); Phase-1 frozen value is 10")
    p.add_argument("--shard", default=None, help="k/N: run tasks where index %% N == k")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help="intra-op threads for --device cpu (see DEFAULT_THREADS; "
                        "0 = leave torch's default alone)")
    p.add_argument("--force", action="store_true", help="re-run finished tasks")
    p.add_argument("--list", action="store_true", help="print the task grid and exit")
    p.add_argument("--pilot", action="store_true", help="3 sanity tasks, print metrics")
    p.add_argument("--pilot-dataset", default="mp_20")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--dft-baseline", action="store_true")
    args = p.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    configs = [c for c in args.configs.split(",") if c]
    samplers = [s for s in args.samplers.split(",") if s]
    SEEDS = tuple(int(s) for s in args.seeds.split(","))
    N_CAND = args.n_cand
    for c in configs:
        if c not in CONFIGS:
            p.error(f"unknown config {c!r}; known: {sorted(CONFIGS)}")

    if args.dft_baseline:
        dft_baseline(datasets)
        return
    if args.list:
        tasks = build_tasks(args.nnp, datasets, configs, samplers, SEEDS, args.n_cells)
        done = sum(1 for t, path in tasks
                   if _task_ok(path, CONFIGS[t["config"]], N_CAND))
        # Not a plain product: PF-ODE runs a subset of the configs, so spell
        # out the per-sampler arm sizes rather than multiplying them together.
        arms = " + ".join(f"{len(configs_for(s, configs))} configs x '{s}'"
                          for s in samplers)
        print(f"grid: {len(tasks)} tasks = {len(datasets)} datasets x "
              f"({arms}) x {args.n_cells} cells x {len(SEEDS)} seeds; "
              f"{N_CAND} candidates each = {len(tasks)*N_CAND} trajectories")
        print(f"  datasets {datasets}\n  configs {configs}\n  samplers {samplers}")
        print(f"  pfode configs {PFODE_CONFIGS} (the rest are ALD-only)")
        print(f"  finished on disk: {done}/{len(tasks)}")
        return
    if args.pilot:
        pilot(args.nnp, args.device, args.pilot_dataset)
        return
    if args.analyze:
        analyze(args.nnp, datasets, configs, samplers, args.n_cells)
        return

    tasks = build_tasks(args.nnp, datasets, configs, samplers, SEEDS, args.n_cells)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % n == k]

    # Contexts for every dataset that will actually run (cells/hull/index load
    # once); the calculator is shared across all of them.
    needed = sorted({t[0]["dataset"] for t in tasks})
    ctxs = {d: context(d, args.nnp, n_cells=args.n_cells, need_index=True)
            for d in needed}
    if args.device == "cpu" and args.threads:
        print(f"[cpu] intra-op threads = {set_cpu_threads(args.threads)} "
              f"(see DEFAULT_THREADS)", flush=True)
    calc = rp.load_calculator(args.nnp, args.device)

    n_done = n_fail = n_skip = 0
    t_start = time.time()
    for task, path in tasks:
        cfg = CONFIGS[task["config"]]
        if path.exists() and not args.force:
            if _task_ok(path, cfg, N_CAND):
                n_skip += 1
                continue
            print(f"STALE {path}: wrong config_sha1 or init_cell_gen, re-running",
                  flush=True)
        try:
            payload = run_task(task, ctxs[task["dataset"]], calc)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            n_done += 1
            if n_done % 10 == 0:
                dt = time.time() - t_start
                print(f"[{n_done} done, {n_fail} fail, {n_skip} skip] "
                      f"{dt/max(n_done,1):.0f}s/task", flush=True)
        except Exception as exc:
            n_fail += 1
            print(f"FAIL {task}: {type(exc).__name__}: {exc}", flush=True)
            if _is_sticky_cuda_error(exc):
                # The CUDA context is dead, not just this task: every later
                # task in this process would raise the same error, so carrying
                # on burns the shard's remaining budget producing identical
                # FAIL lines.  Exit non-zero and let the supervisor restart
                # this shard -- completed tasks are on disk and are skipped by
                # the resume check above, so a restart costs one task.
                #
                # Seen: eSEN (fairchem 1.10) hit
                # "CUDA error: an illegal memory access was encountered" in
                # OCPCalculator.predict on an ALD trajectory, killing whole
                # preflight shards.  Primary-NNP-only risk: MACE has not shown
                # it at fleet scale (Phase 1 subexp1 was MACE).
                print(f"ABORT shard: sticky CUDA error, {n_done} tasks are on "
                      f"disk and will be skipped on restart", flush=True)
                return 3
    print(f"finished: {n_done} run, {n_skip} skipped, {n_fail} failed, "
          f"{time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
