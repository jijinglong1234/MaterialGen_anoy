"""
Phase 1 — completeness boundary verification (experiment_plan_v1.md Phase 1, paper §6.1).

Sub-experiments (all independent tasks; one JSON per (condition, composition, seed)):

  --subexp 1   σ-failure curves: 10 σ_max × {bare, L1, L1-L4} × {ALD, PF-ODE}
               × 20 MP-20 compositions × 20 candidates/seed × 5 seeds.
               ALD: K=100, M=2 (NFE=200). PF-ODE: DOPRI5(ivp) rtol=1e-4, cap 5000 NFE.
  --subexp 2   doc-12 layer grid at σ_max=1.0, ALD (12 cells: bare + the 11
               non-empty layer subsets of the full-factorial-minus-mid-pairs
               design; run on the post-4a12ce4 fixed-cell code).
  --subexp sm  SafetyMonitor parameter sub-scan: p ∈ {1,2,3} × d_safe ∈ {0.7,1.0,1.2},
               full L1-L4 ALD σ=1.0 on 5 compositions.
  --subexp 5   Pauli sensitivity (A8): 5 element pairs × A ∈ {1e2..1e6} × B ∈ {1,3,10}
               (+ ZBL reference), ALD L1 σ=1.0 on the composition containing the pair.
  --subexp rc  eSEN recheck: M0 downgrade plan — eSEN re-verifies the key conclusions
               (bare validity collapse, L1 OOD reduction, PF-ODE robustness) on the
               5-composition subset. ALD {bare, L1, L1-L4} + PF-ODE bare at
               σ ∈ {0.5, 1.0, 2.0, 5.0}.

Protocol conventions (paper §6.1):
  - OOD event: d_min < 0.5 Å or max_i ||F_NNP,i|| > 500 eV/Å or
    |det L|/|det L0| > 10 (Type III lattice explosion, paper §6.1; fires only
    under stress-driven lattice updates — structurally absent in the
    fixed-cell protocol) at any visited step.
    The force criterion judges the NNP-only force: the Pauli wall legitimately
    exceeds 500 eV/Å at short range (physical repulsion, not extrapolation),
    so augmented scores expose nnp_forces in metadata and it is stripped from
    all OOD/validity gates.
  - Failure modes per step: Type I atom overlap (any pair below 0.75×(covalent
    sum) — the plain covalent sum over-reports normal ionic bonds, e.g. Ti-O
    1.95 Å < 2.14 Å), Type II electronic collapse (U < -50 eV/atom), Type III
    lattice explosion (|det L|/|det L0| > 10; structurally absent under the
    fixed-cell protocol).
  - validity of a trajectory = final state satisfies the DiffCSP/MatterGen-
    aligned validity definition (paper §6.0.7): composition (fixed in Phase 1)
    + d_min ≥ 0.5 Å, with non-finite energy/forces counting as failures.
    max|F_NNP| ≤ 500 is an OOD criterion, not a validity criterion.
  - Initial state per candidate: ref + σ_max·ε with ε ~ N(0,1) per cartesian
    coordinate (σ in Å, isotropic), wrapped mod 1. Cartesian rather than
    fractional noise: on a short-axis lattice (≈4.5 Å) fractional σ=0.1
    displaces atoms ~0.48 Å and compresses bonds to d_min≈0.9 Å, saturating
    the σ-failure curve at the smallest σ. Per-candidate RNG derived from
    (seed, cand_idx) so all workers reproduce the same trajectories.
  - PF-ODE "full four-layer" reduces to L1 + L3 (reflecting boundary): L2 (noise
    modulation) and L4 (step rejection) are undefined for a deterministic ODE;
    this adaptation is documented in the paper's implementation notes.
  - Fixed cell (update_lattice=False) everywhere — paper C5/C10 practice.

E_hull of a generated structure (proxy): E_form = E_NNP/N − (E_NNP(ref)/N_ref −
e_form_MP(ref)), then pymatgen PhaseDiagram on the per-system hull entries
(data/hulls/mp20_hull_entries.json) → e_above_hull.

Worker model mirrors run_baselines.py: --shard k/N skips finished tasks; per-task
JSON under results/phase1/<subexp>/<condition>/<comp>_seed<s>.json. Launch
logs/run_phase1_mace.sh (10 workers, 2/card) in the `materialgen` env and
logs/run_phase1_esen.sh (10 workers, 2/card) in the `materialgen-esen` env.

Usage:
    python scripts/run_phase1.py --subexp 1 --nnp mace --pilot     # 1 sanity task
    python scripts/run_phase1.py --subexp 1 --nnp mace --shard 0/10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from materialgen.core.crystal import CrystalStructure
from materialgen.core.pauli import PauliRepulsion
from materialgen.core.reflection import ReflectionOperator
from materialgen.core.safety_monitor import MonitorThresholds, SafetyMonitor
from materialgen.core.score_function import AugmentedScore, NNPScore, PauliScore, ScoreFunction, ScoreResult
from materialgen.samplers.ald import ALDConfig, ALDSampler
from materialgen.samplers.pf_ode import PFODEConfig, PFODESampler
from materialgen.utils.constants import BETA_300K

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed" / "mp_20" / "test.pkl"
HULL_PATH = REPO / "data" / "hulls" / "mp20_hull_entries.json"
OUT = REPO / "results" / "phase1"
# Composition-only prior (subexp "comp"): the test split is the
# evaluation target and must never feed the prior.  Train is the model's own
# data; val is the fallback for the four test compositions that have no train
# structure (CdS, VN, AlCo, AlFe3) -- still not the test target.
TRAIN_PROCESSED = REPO / "data" / "processed" / "mp_20" / "train.pkl"
VAL_PROCESSED = REPO / "data" / "processed" / "mp_20" / "val.pkl"

# ---------------------------------------------------------------------------
# Protocol constants (paper §6.1)
# ---------------------------------------------------------------------------
SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
OOD_DMIN = 0.5          # Å
OOD_FMAX = 500.0        # eV/Å
TYPE2_E_PER_ATOM = -50.0  # eV/atom
TYPE3_VOL_RATIO = 10.0
# Lattice collapse (paper section 4.2): Type I (d_min below covalent sum)
# coincident with the cell shrunk below this volume ratio.  The collective
# mode requires stress-driven lattice updates; structurally zero under the
# fixed-cell protocol (Phase 1), fires only in upd_lat runs (Phase 2 C1/C3/
# C8/C9/C11).
COLLAPSE_VOL_RATIO = 0.5
ALD_K, ALD_M = 100, 2
# Base Langevin step alpha_0 of the production schedule (alpha_k = alpha_0 *
# sigma_k/sigma_max).  Factored out so the step-size control can
# divide it: make_ald_config(step_mult=m) runs M*m steps per noise level at
# alpha_0/m, which leaves the SDE time per level (steps x alpha_k) and the
# accumulated noise variance (steps x 2*alpha_k*sigma_k^2) both invariant --
# i.e. an exact temporal refinement of the same discretized process.
ALD_ALPHA = 1e-3
PF_RTOL, PF_ATOL = 1e-4, 1e-6
# PF-ODE NFE cap, 5000 -> 1000 (a mini-run finding): the raw
# Boltzmann ODE (paper §4.4.B) becomes stiff when a trajectory crosses the
# Pauli wall — the exponential repulsion collapses the RK45 step size and
# burns NFE (measured: ~1/3 of l1 candidates hit the cap at ~5 min each vs
# ~8s normally; the same candidate is 86 NFE with bare score). The cap
# bounds this cost; capped trajectories are flagged non-converged with their
# final state still evaluated on its own merits (validity/OOD unchanged).
# Fast trajectories use <= ~100 NFE, so the cap never binds them.
PF_MAX_NFE = 1000
# Candidates per (composition, seed).
# protocol change, 20 -> 10: the noise_lattice cholesky fix
# (materialgen/core/crystal.py) invalidated every Phase-1 trajectory -- they
# all ran in a sigma-independently distorted initial cell -- so the fleet was
# re-run from scratch.  Halving the candidate count kept the re-run at ~1200
# GPU-h serial instead of ~2400 (the archived files' own wall_time_s sum to
# 2414 GPU-h at 20 candidates, 76% of it in the PF-ODE arms, where 52-58% of
# trajectories hit the PF_MAX_NFE cap).  Everything else stays frozen: sigma
# grid, the 3 protections x 2 samplers, 20 compositions x 5 seeds, NFE 200 /
# cap 1000, OOD mask, induced-rate definition.  Consequence: a cell is now
# 100 files x 10 candidates = 1000 trajectories, not 2000 -- downstream code
# must read n_candidates/n_traj from the payload and never hardcode 20/2000.
N_CAND = 10
SEEDS = [42, 123, 999, 2024, 7777]

# Provenance marker stamped into every payload and required by
# scripts/analyze_phase1.py before a file may enter an analysis pass.  Bump
# this on any change that makes older trajectories non-comparable.
INIT_CELL_GEN = "cholesky-fix"

COMPOSITIONS_20 = [
    # oxides
    "SrTiO3", "TiO2", "Al2O3", "BaTiO3", "LiCoO2", "ZrO2",
    # sulfides
    "ZnS", "MoS2", "CdS", "Ag2S",
    # fluorides
    "LiF", "AlF3", "CoF3",
    # nitrides
    "GaN", "Ca3N2", "VN",
    # intermetallics
    "FeNi3", "AlCu", "AlCo", "AlFe3",
]
RECHECK_COMPS = ["SrTiO3", "ZnS", "LiF", "GaN", "FeNi3"]  # one per chemistry class

# Mini (validation) scale — small-scale Phase 1 to verify method correctness
# on GPU0 before the full fleet restarts. Shrinks only the
# statistically redundant axes; every scientific dimension of §6.1 is kept:
#   σ      10 → 5  {0.1, 0.5, 1.0, 2.0, 5.0}: log-spaced across the validity
#                   transition (sigmoid fit needs ≥4 points; the collapse at
#                   low σ and saturation at high σ stay observable)
#   comps  20 → 5  RECHECK_COMPS (one per chemistry class)
#   seeds   5 → 2  {42, 123} (per-seed error bars still computable)
#   cand   20 → 10 per (composition, seed)
#   protections × samplers: full 3 × 2 = 6 — the core of the σ-failure design.
#   scope: --subexp 1 and 2 only (sm/5/rc run in the full fleet).
# NFE: subexp1 5·3·2·5·2·10·200 = 600K + subexp2 5·5·2·10·200 = 100K ≈ 700K
# (~3-4 GPU-h at MACE-MP-0 on 2-5-atom cells).
MINI_SIGMAS = [0.1, 0.5, 1.0, 2.0, 5.0]
MINI_SEEDS = [42, 123]
MINI_N_CAND = 10
MINI_COMPOSITIONS = RECHECK_COMPS
SM_P = [1.0, 2.0, 3.0]
SM_DSAFE = [0.7, 1.0, 1.2]
PAULI_GRID = [(a, b) for a in [1e2, 1e3, 1e4, 1e5, 1e6] for b in [1.0, 3.0, 10.0]]
# (pair tag, composition containing the pair, atomic numbers)
SUBEXP5_PAIRS = {
    "cc": ("CaC2", (6, 6)),
    "sio": ("SiO2", (14, 8)),
    "tio": ("TiO2", (22, 8)),
    "lio": ("LiCoO2", (3, 8)),
    "bao": ("BaTiO3", (56, 8)),
}
PROTECTIONS = ["bare", "l1", "l1l4"]
SAMPLERS = ["ald", "pfode"]
# L1-L3 = {1,2,3} is the paper's headline configuration, so the
# sigma scan carries its own arm.  Deliberately NOT appended to PROTECTIONS:
# that list also drives the eSEN rc matrix (build_tasks("rc")), which is
# mid-flight against a fixed 400-task target.  ALD-only: the PF-ODE path has
# no L2 (no density noise field) and no L4 (no acceptance semantics), so a
# pfode "l1l2l3" arm would be a byte-identical rerun of the existing pfode
# "l1l4" arm -- both carry L1+L3, which analyze_phase1.prot_label already
# renders as "L1+L3 (refl.)".
# "l3" added at the reviewer's request.  At sigma=1.0 the doc-12
# layer grid shows the reflecting constraint alone is nearly sufficient
# (32.8% -> 0.10% induced), but that is a single noise level, and the scan's
# only protected arm bundles L1+L3 -- so the scan cannot say whether the
# analytic wall contributes anything once the constraint is present, which is
# the central claim of the physics decomposition.  ALD-only for the same
# reason as l1l2l3.
SIGMA_SCAN_ALD_ONLY = ["l1l2l3", "l3"]

# Step-size control grid (see build_tasks("fine")).  Bare covers
# the rise (0.3, 0.5) and the plateau (1.0, 2.0, 5.0); the protected stack is
# run at the three middle levels only, since the adversarial question there is
# whether protection still holds once the bare arm's advantage is removed
# rather than whether it varies along the curve.
FINE_STEP_MULT = 4
FINE_ARMS = [("bare", [0.3, 0.5, 1.0, 2.0, 5.0]),
             ("l1l2l3", [0.5, 1.0, 2.0])]


# ---------------------------------------------------------------------------
# Layer-subset resolution (doc-12 grid)
# ---------------------------------------------------------------------------
# Level names encode the active protection subset as digits after 'l'
# (1 = L1' Pauli wall + drift caps, 2 = L2 density noise, 3 = L3 reflection,
# 4 = L4 SafetyMonitor).  Legacy exception: "l1l4" means the FULL stack
# {1,2,3,4} (prefix-ladder naming from the 5-condition era, kept for all
# pre-existing call sites), NOT the mid-pair {1,4}.  The doc-12 grid
# (docs/experiment_plan_v1.md §Phase 1 Sub-exp 2) = the full factorial
# minus the four mid-pairs {1,4},{2,3},{2,4},{3,4} = 11 layer subsets + bare.
# Cells flagged for the pre-4a12ce4 ratchet-era data are archived in
# results/phase1/subexp2_ratchetera/; subexp2/* below are the
# rerun on the fixed-cell code (allow_lattice_scale=False).
LEVEL_SUBSETS = {
    "bare": frozenset(),               # no protection
    "l1": frozenset({1}),              # L1' (cap v3 + probe calibration)
    "l1l2": frozenset({1, 2}),
    "l1l2l3": frozenset({1, 2, 3}),
    "l1l4": frozenset({1, 2, 3, 4}),   # legacy name = FULL stack
    # doc-12 supplement cells
    "l2": frozenset({2}),
    "l3": frozenset({3}),
    "l4": frozenset({4}),
    "l1l3": frozenset({1, 3}),
    "l1l2l4": frozenset({1, 2, 4}),
    "l1l3l4": frozenset({1, 3, 4}),
    "l2l3l4": frozenset({2, 3, 4}),
}
SUBEXP2_LEVELS_DOC12 = [
    "bare", "l1", "l1l2", "l1l2l3", "l1l4",   # legacy prefix ladder (kept adjacent)
    "l2", "l3", "l4", "l1l3", "l1l2l4", "l1l3l4", "l2l3l4",
]


def level_subset(protection: str) -> frozenset:
    try:
        return LEVEL_SUBSETS[protection]
    except KeyError:
        raise ValueError(f"unknown protection level {protection!r}") from None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_structures(comps: list[str]) -> dict:
    """Lowest-e_above_hull processed record per composition (deterministic)."""
    import pickle

    with open(PROCESSED, "rb") as f:
        records = pickle.load(f)
    best: dict = {}
    for r in records:
        meta = r["metadata"]
        formula = meta.get("pretty_formula")
        if formula not in comps:
            continue
        ehull = float(meta.get("e_above_hull", np.inf))
        key = (formula, ehull)
        if formula not in best or ehull < best[formula][1]:
            best[formula] = (r, ehull)
    missing = set(comps) - set(best)
    if missing:
        raise RuntimeError(f"No reference structure for {sorted(missing)}")
    return {f: r for f, (r, _) in best.items()}


def load_hull_table():
    return json.loads(HULL_PATH.read_text())


# ---------------------------------------------------------------------------
# Per-NFE diagnostics (uniform for both samplers)
# ---------------------------------------------------------------------------

def structure_diagnostics(crystal: CrystalStructure) -> tuple[float, float]:
    """
    (d_min, covalent_deficit) for a crystal.

    covalent_deficit = min_ij (d_ij - 0.75·(R_i + R_j)) over neighbor pairs
    within the max covalent radius sum; positive when no pair breaches the
    Type-I boundary. The 0.75 factor (protocol fix) prevents
    over-reporting normal ionic bonds: Ti-O 1.95 Å is a standard bond length
    yet lies below the covalent sum 2.14 Å, which made Type I fire on every
    step of every trajectory. One neighbor-list call with cutoff =
    max(2.0, max_radius_sum + 0.3).
    """
    from ase.neighborlist import neighbor_list

    from materialgen.utils.constants import COVALENT_RADII

    atoms = crystal.ase_atoms
    numbers = atoms.get_atomic_numbers()
    n = len(atoms)
    if n < 2:
        return float("inf"), float("inf")
    max_rsum = max(COVALENT_RADII.get(int(z), 1.0) for z in numbers) * 2.0
    cutoff = max(2.0, max_rsum + 0.3)
    i_idx, j_idx, d = neighbor_list("ijd", atoms, cutoff)
    if len(d) == 0:
        return float("inf"), float("inf")
    d_min = float(d.min())
    deficits = np.array([
        d[k] - 0.75 * (COVALENT_RADII.get(int(numbers[i_idx[k]]), 1.0)
                       + COVALENT_RADII.get(int(numbers[j_idx[k]]), 1.0))
        for k in range(len(d))
    ])
    return d_min, float(deficits.min())


class DiagnosticScore(ScoreFunction):
    """Wraps a score; appends one diagnostics record per NFE.

    records entries: {"step": int, "d_min": float, "cov_deficit": float,
                      "max_f": float|None, "energy": float|None,
                      "pauli_active_pairs": int|None}
    """

    def __init__(self, score: ScoreFunction):
        self.score = score
        self.records: list[dict] = []

    def compute(self, crystal: CrystalStructure, t: float | None = None) -> ScoreResult:
        r = self.score.compute(crystal, t)
        d_min, cov_def = structure_diagnostics(crystal)
        # OOD force check uses the NNP-only force (fix): the Pauli
        # wall legitimately produces |F| > 500 eV/Å at short range — that is
        # physical repulsion doing its job, not NNP extrapolation. Augmented
        # scores expose nnp_forces in metadata; bare NNP has no decomposition.
        md = r.metadata or {}
        f_nnp = md.get("nnp_forces", r.forces)
        max_f = None
        if f_nnp is not None and len(f_nnp):
            max_f = float(np.linalg.norm(f_nnp, axis=1).max())
        max_f_total = None
        if r.forces is not None and len(r.forces):
            max_f_total = float(np.linalg.norm(r.forces, axis=1).max())
        self.records.append({
            "step": len(self.records),
            "d_min": d_min,
            "cov_deficit": cov_def,
            "max_f": max_f,
            "max_f_total": max_f_total,
            "energy": r.energy,
            "volume": crystal.volume,
            "pauli_active_pairs": md.get("pauli_active_pairs"),
        })
        return r


# ---------------------------------------------------------------------------
# Score / sampler construction
# ---------------------------------------------------------------------------

def make_score(calc, protection: str, pauli_params: dict | None = None,
               label: str = "MACE-MP-0") -> ScoreFunction:
    nnp = NNPScore(calc, beta=BETA_300K, compute_stress=False, label=label)
    if 1 not in level_subset(protection):
        return nnp
    # L1': the probe-driven entry-force calibration table
    # (item 2) is auto-loaded from data/pauli/probe_calibration.json when
    # present (ZBL fit elsewhere).  The deep-wall power-law floor (item 3)
    # was REMOVED: the L1' verification cell found zero effect
    # on PF-ODE NFE (G4 fail: cap-hit -0.5pt at sigma=1, 0pt at sigma=2) and
    # no ALD-side difference (L1' ~= L1 on every metric) — the RK45 NFE burn
    # happens at the wall-ENTRY tanh transition, not in the deep tail.
    pauli = PauliRepulsion(params=pauli_params, deep_wall=False)
    return AugmentedScore(nnp, PauliScore(pauli=pauli, beta=BETA_300K))


def make_ald_config(sigma_max: float, protection: str,
                    step_mult: int = 1) -> ALDConfig:
    # Layer flags resolved from the subset digits: every name is
    # now an exact layer subset — "1" = L1' wall caps, "2" = L2 density noise,
    # "3" = L3 reflection, "4" = L4 SafetyMonitor (attached in
    # make_ald_sampler).  Identity check: for the legacy ladder names this
    # equals the pre-fix flag logic (caps for every non-bare name, L2+L3
    # for l1l4 only, subexp-2 overrides folded in below).
    subset = level_subset(protection)
    return ALDConfig(
        sigma_max=sigma_max, sigma_min=0.01, K=ALD_K, M=ALD_M * step_mult,
        alpha=ALD_ALPHA / step_mult,
        # alpha re-calibrated: with the sigma-normalized score the
        # step is alpha_k * beta*F; 0.1 overran the noise scale ~100x and
        # pushed first steps ~3 A through the Pauli wall. 1e-3 keeps the
        # drift below ~0.15 A/step at sigma_max <= 0.5 (measured MACE force
        # scales) while remaining mobile over K*M = 200 steps.
        update_lattice=False,                          # fixed cell (paper C5/C10)
        use_density_noise=2 in subset,                 # L2
        use_reflection=3 in subset,                    # L3
        use_wall_cap=1 in subset,                      # L1' (verify #3):
                                                       # every L1-containing condition
        use_wall_cap_v3=1 in subset,                   # L1' cap v3 (adopted,
                                                       # verify cell
                                                       # G1-G4 all pass): bounded
                                                       # repulsive escape kills the
                                                       # wall energy pump — L1-only
                                                       # ALD induced 0.24 vs bare
                                                       # 0.39 (asym cap: 0.57);
                                                       # dwell/deep/E_hull all beat
                                                       # bare at sigma=1
        seed=None,                                     # set per candidate
    )


def make_ald_sampler(cfg: ALDConfig, protection: str, sm_params: dict | None = None) -> ALDSampler:
    subset = level_subset(protection)
    monitor = None
    if 4 in subset:
        th = MonitorThresholds()
        if sm_params:
            th.d_min_abs = sm_params["d_safe"]
        monitor = SafetyMonitor(thresholds=th, p=(sm_params or {}).get("p", 2.0),
                                gamma_base=0.5)
    return ALDSampler(cfg, safety_monitor=monitor)


def make_pfode_config(sigma_max: float, protection: str) -> PFODEConfig:
    # This function resolves only the layers that live in the *sampler*: L3
    # (reflection) here, and nothing else.  L1 is NOT missing from the ODE
    # path -- it rides in the score, which make_score() builds from the same
    # protection name and the caller hands to PFODESampler just as it does to
    # ALDSampler, so a Pauli-augmented velocity field is what the ODE
    # integrates.  L2 (density-adaptive noise) and L4 (accept/reject) have no
    # ODE counterpart at all.  Net: on the ODE path "l1l2l3" and "l1l4" are
    # the same condition, L1+L3 -- a pfode arm for either name would be a
    # byte-identical rerun of the other (doc-12 freeze).  Resolve
    # from the subset digits rather than a name list, so every name maps
    # correctly (the old ("l1l4", "l1l3") tuple silently dropped the
    # reflection for "l1l2l3").
    use_ref = 3 in level_subset(protection)
    return PFODEConfig(
        sigma_max=sigma_max, sigma_min=0.01, solver="ivp",
        rtol=PF_RTOL, atol=PF_ATOL, max_nfe=PF_MAX_NFE,
        use_reflection=use_ref,
    )


def make_pfode_sampler(cfg: PFODEConfig) -> PFODESampler:
    return PFODESampler(cfg, reflection=ReflectionOperator() if cfg.use_reflection else None)


# ---------------------------------------------------------------------------
# E_hull (proxy) via per-system PhaseDiagram
# ---------------------------------------------------------------------------

class HullEvaluator:
    """E_above_hull proxy: pymatgen PhaseDiagram per chemical system.

    Formation energies calibrated to MP elemental references through the
    reference record: E_form(x) = E_NNP(x)/N - (E_NNP(ref)/N_ref - e_form_MP(ref)).
    """

    def __init__(self, hull_table: dict):
        self.hull_table = hull_table
        self._diagrams: dict = {}
        self._ref_calib: dict = {}   # formula -> e_form offset (eV/atom)
        self._ref_nnp_energy: dict = {}

    def calibrate(self, formula: str, nnp_energy_ref: float, n_atoms_ref: int,
                  e_form_mp_ref: float):
        self._ref_calib[formula] = nnp_energy_ref / n_atoms_ref - e_form_mp_ref
        self._ref_nnp_energy[formula] = nnp_energy_ref

    def _diagram(self, system: str):
        from pymatgen.core import Composition, Element
        from pymatgen.entries.computed_entries import ComputedEntry
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        if system in self._diagrams:
            return self._diagrams[system]
        entries = []
        for ent in self.hull_table[system]["entries"]:
            comp = Composition(ent["formula"])
            entries.append(ComputedEntry(
                comp, float(ent["e_form_per_atom"]) * comp.num_atoms))
        for el in self.hull_table[system]["elements"]:
            entries.append(ComputedEntry(Composition(el), 0.0))
        pd = PhaseDiagram(entries)
        self._diagrams[system] = pd
        return pd

    def e_hull(self, formula: str, nnp_energy: float, n_atoms: int) -> float | None:
        """E_above_hull in eV/atom, or None if the system is not covered.

        units fix: pymatgen's get_e_above_hull already returns a
        per-atom value (phase_diagram.py: "The energy is given per atom"), so
        the previous extra ``/ comp.num_atoms`` -- comp being the *reduced*
        formula, e.g. Composition("Al2O3").num_atoms = 5 -- silently divided
        the result again by the formula-unit atom count (2-5, composition
        dependent).  Files written earlier therefore store
        E_hull /n_fu and carry no "e_hull_units" marker; the analysis layer
        rescales them back (analyze_phase1.cands_of).
        """
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        comp = Composition(formula)
        system = "-".join(sorted(el.symbol for el in comp.elements))
        if system not in self.hull_table or formula not in self._ref_calib:
            return None
        e_form = nnp_energy / n_atoms - self._ref_calib[formula]
        try:
            pd = self._diagram(system)
            # entry energy = total for ONE formula unit (e_form is per atom);
            # pymatgen renormalizes by comp.num_atoms internally, so the
            # per-atom comparison -- and only that -- is what the hull uses.
            entry = ComputedEntry(comp, e_form * comp.num_atoms)
            return float(pd.get_e_above_hull(entry))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Candidate protocol
# ---------------------------------------------------------------------------

def load_composition_cells(comps: list[str], refs: dict | None = None) -> dict:
    """Composition -> a typical cell of that composition, from train (or val).

    The reference prior (load_reference_structures) hands the sampler the
    lowest-e_above_hull *test* structure -- both its coordinates and its cell.
    That is the prior the reviewer objects to: an arm named "σ-failure of a
    boundary" should not start from the answer.  This loader supplies the
    weaker prior a generator actually has: the composition, plus a cell taken
    from a train structure of that composition.  Never the test split.

    Two tiers, recorded per composition in `source`:

      "train" / "val"        exact stoichiometry in that split, lowest
                             formation_energy_per_atom (the thermodynamically
                             favoured polymorph, deterministic on ties)
      "train-system(X)"      no structure of this stoichiometry exists in
                             either split, but the element system does (VN,
                             AlCo, AlFe3 have one each, X = the system's
                             lowest-formation-energy train cell)

    The cell shape and the per-atom volume come from the train/val record; the
    *number of formula units* is taken from the reference, so that this arm and
    the reference arm sample the same number of atoms and their rates are
    comparable.  Both are handled by one uniform scaling of the record's
    lattice, which leaves angles and V/n invariant:
        L_cell = L_record * (n_ref / n_record)^(1/3)

    `refs` supplies that atom count and the atom list.  Using the test
    reference for them is deliberate and is not a structural leak: `numbers`
    is the composition itself (which atoms, how many -- the quantity the arm
    is built around), while the coordinates and the cell are exactly what this
    loader withholds.
    """
    import pickle

    refs = refs if refs is not None else load_reference_structures(list(comps))
    want = set(comps)
    systems = {c: "-".join(_formula_elements(c)) for c in comps}
    sys_of_interest = set(systems.values())
    exact: dict = {}           # formula -> (e_form, split, record)
    system: dict = {}          # element system -> (e_form, split, record)
    for split, path in (("train", TRAIN_PROCESSED), ("val", VAL_PROCESSED)):
        with open(path, "rb") as fh:
            records = pickle.load(fh)
        for r in records:
            formula = r["metadata"].get("pretty_formula")
            if formula is None:
                continue
            e = float(r["metadata"].get("formation_energy_per_atom", np.inf))
            if formula in want:
                if formula not in exact or e < exact[formula][0]:
                    exact[formula] = (e, split, r)
                continue
            try:
                syskey = "-".join(_formula_elements(formula))
            except Exception:
                continue
            if syskey in sys_of_interest and (
                    syskey not in system or e < system[syskey][0]):
                system[syskey] = (e, split, r)

    out = {}
    for formula in comps:
        if formula in exact:
            e, split, r = exact[formula]
            source = split
        else:
            syskey = systems[formula]
            if syskey not in system:
                raise RuntimeError(f"No train/val cell for {formula} "
                                   f"(system {syskey})")
            e, split, r = system[syskey]
            source = f"{split}-system({r['metadata']['pretty_formula']})"
        n_target = len(refs[formula]["numbers"])
        n_record = len(np.asarray(r["numbers"]))
        out[formula] = {
            "numbers": np.asarray(refs[formula]["numbers"]),
            "lattice": np.asarray(r["lattice"], dtype=float) * (n_target / n_record) ** (1.0 / 3.0),
            "source": source,
            "e_form_per_atom": e,
            "n_atoms": n_target,
            "n_record": n_record,
        }
    return out


def _formula_elements(formula: str):
    from pymatgen.core import Composition
    return sorted(str(el) for el in Composition(formula).elements)


def _target_systems(comps) -> set:
    return {"-".join(_formula_elements(c)) for c in comps}


def make_initial_comp_only(cell_record: dict, sigma_max: float,
                           rng: np.random.RandomState) -> CrystalStructure:
    """Composition-only prior: uniform fractional coords in a composition cell.

    The reference prior's start is ref + σ·ε -- a σ-sized displacement from a
    structure that is already the answer.  Here the coordinates carry no
    structure at all: iid U[0,1)^3 on the composition's cell (which comes from
    a train/val structure of the same composition, never the test one).  The
    lattice jitter (log-Gram, sigma_latt = 0.1·σ_max) is kept identical to
    make_initial so that the only difference between the two arms is where the
    coordinates came from.

    Consequence to read the rates against: this start is not guaranteed
    in-distribution.  At small σ it can already be OOD (a uniform draw
    routinely places a pair below 0.5 Å), where the reference arm's start
    never is.  The published induced-OOD definition (start-clean -> OOD, see
    make_percomp_sigma_tables.induced_k) handles exactly this case: such a
    trajectory is neither induced nor allowed to inflate the count, and its
    exclusion is what the conditional denominator in
    results/phase1/analysis/conditional_ood.json quantifies.
    """
    n = len(cell_record["numbers"])
    frac = rng.random_sample((n, 3))
    crystal = CrystalStructure.from_frac_coords(
        cell_record["numbers"], frac, cell_record["lattice"])
    return crystal.noise_lattice(0.1 * sigma_max, rng=rng)


def make_initial(ref_record: dict, sigma_max: float, rng: np.random.RandomState) -> CrystalStructure:
    """ref + σ_max·ε in cartesian coords (isotropic; σ in Å), frac wrapped mod 1.

    Cartesian rather than fractional noise (protocol fix): on a
    short-axis lattice, fractional noise displaces atoms by ~σ·|axis| Å, so a
    pair moving toward each other compresses a bond by ~2σ·|axis|. At σ=0.1 on
    a 4.55 Å axis this already produced d_min≈0.9 Å and NNP force blow-ups
    (>1e3 eV/Å), saturating the σ-failure curve at the smallest σ. Cartesian
    noise makes σ an isotropic displacement (Å), matching the R_basin theory.

    Lattice noise: MatterGen-style multiplicative noise in
    log-Gram space, sigma_latt = 0.1·σ_max (paper section 6.0.5), applied to
    the reference lattice. The reference structure's lattice serves as the
    composition's typical cell (the lowest-e_above_hull test structure;
    equivalent to the training-set mean used by the paper), keeping atoms
    and cell mutually consistent. Log-Gram noise is always positive-definite
    and strain-isotropic — unlike per-vector Gaussian noise, which can flip
    a lattice vector. Under the fixed-cell protocol the noisy lattice is
    then held constant for the trajectory.
    """
    cart = ref_record["frac_coords"] @ ref_record["lattice"]
    cart_noisy = cart + sigma_max * rng.normal(0.0, 1.0, cart.shape)
    frac = np.linalg.solve(ref_record["lattice"].T, cart_noisy.T).T % 1.0
    crystal = CrystalStructure.from_frac_coords(ref_record["numbers"], frac,
                                                ref_record["lattice"])
    return crystal.noise_lattice(0.1 * sigma_max, rng=rng)


def classify_steps(records: list[dict], n_atoms: int, v0: float) -> tuple:
    """
    Per-step OOD/failure flags from diagnostics records.

    Returns (bitmask list, aggregates): bit0 d_min<0.5, bit1 max_f>500,
    bit2 Type I, bit3 Type II, bit4 Type III, bit5 non-finite,
    bit6 lattice collapse (paper section 4.2: Type I under a shrunken cell,
    vol/v0 < COLLAPSE_VOL_RATIO — the collective mode distinct from a local
    pair clash; structurally zero under the fixed-cell protocol, fires only
    in stress-driven lattice-update runs).
    Type III (volume ratio > 10) is structurally absent under the fixed-cell
    protocol; the check is kept for completeness.
    """
    masks = []
    n_ood, n_t1, n_t2, n_t3, n_nf, n_collapse = 0, 0, 0, 0, 0, 0
    for rec in records:
        d_min, max_f, energy = rec["d_min"], rec["max_f"], rec["energy"]
        mask = 0
        if not (np.isfinite(d_min) and (max_f is None or np.isfinite(max_f))
                and (energy is None or np.isfinite(energy))):
            mask |= 0b100000
            n_nf += 1
            masks.append(mask)
            n_ood += 1
            continue
        if d_min < OOD_DMIN:
            mask |= 0b000001
        if max_f is not None and max_f > OOD_FMAX:
            mask |= 0b000010
        if rec["cov_deficit"] < 0.0:
            mask |= 0b000100
            n_t1 += 1
            # Lattice collapse: Type I coincident with the cell shrunk below
            # COLLAPSE_VOL_RATIO of the reference volume (paper section 4.2).
            if v0 > 0 and rec.get("volume", 0.0) / v0 < COLLAPSE_VOL_RATIO:
                mask |= 0b1000000
                n_collapse += 1
        if energy is not None and energy / n_atoms < TYPE2_E_PER_ATOM:
            mask |= 0b001000
            n_t2 += 1
        if v0 > 0 and rec.get("volume", 0.0) / v0 > TYPE3_VOL_RATIO:
            mask |= 0b010000
            n_t3 += 1
        masks.append(mask)
        # OOD spike = bits 0,1 (d_min, max_f) OR bit4 (Type III volume
        # explosion) — paper §6.1 definition (Type III added).
        if mask & 0b010011:
            n_ood += 1
    return masks, {"n_ood_steps": n_ood, "type1_steps": n_t1,
                   "type2_steps": n_t2, "type3_steps": n_t3, "n_nan_steps": n_nf,
                   "collapse_steps": n_collapse}


def run_candidate(score, sampler, ref_record: dict, sigma_max: float,
                  seed: int, cand: int, hull: HullEvaluator,
                  prior: str = "reference",
                  cell_record: dict | None = None) -> dict:
    """One trajectory; returns the per-candidate payload.

    prior="reference" (default, every arm) starts from the
    test reference structure; prior="composition" (subexp "comp") starts from
    uniform coordinates on a train/val cell of the same composition and needs
    cell_record.  ref_record still supplies the E_hull formula and the
    calibration reference -- see run_task.
    """
    rng = np.random.RandomState(seed * 1000 + cand)
    if prior == "composition":
        if cell_record is None:
            raise ValueError("prior='composition' requires cell_record")
        initial = make_initial_comp_only(cell_record, sigma_max, rng)
    else:
        initial = make_initial(ref_record, sigma_max, rng)

    # per-candidate sampler seed (sampler reads cfg.seed at each sample() call)
    sampler_cfg = getattr(sampler, "config", None)
    if sampler_cfg is not None:
        sampler_cfg.seed = seed * 1000 + cand + 500000

    diag = DiagnosticScore(score)
    t0 = time.time()
    res = sampler.sample(diag, initial)
    nfe, n_rej = res.nfe, getattr(res, "n_rejections", 0)
    # Drift-cap diagnostics: present only in files written later; absent keys
    # read as None in the analysis, never as 0.
    cap_steps = getattr(res, "cap_steps", None)
    cap_atoms = getattr(res, "cap_atoms", None)
    # Noise level the trajectory actually reached — PF-ODE only;
    # an ALD run completes its schedule by construction, so the field stays
    # None there.  Absent in files written earlier, like cap_steps.
    sigma_final = getattr(res, "sigma_final", None)
    # The sampler's own convergence verdict.  For PF-ODE this is
    # stricter than "did not exhaust the cap": the ivp branch also clears it
    # when solve_ivp reports failure or the lattice left the positive-definite
    # domain, so a short run that ended in a collapsed state is not converged
    # even though it used fewer than max_nfe evaluations.  Analysis that only
    # sees "nfe >= cap" (the paper's Cap column) cannot tell those apart, which
    # is why the flag is stored rather than derived.
    converged = getattr(res, "converged", None)

    # PF-ODE ivp: per-NFE diagnostics include RK45 intermediate stage points,
    # which are extrapolation states, not trajectory states — they produce
    # spurious OOD spikes (σ=0.1 showed 18% artifact OOD on
    # near-equilibrium SrTiO3 starts; the accepted-step path is clean).
    # Rebuild diagnostics on the accepted trajectory (res.path) instead.
    if getattr(res, "path", None):
        diag.records = []
        for st in res.path:
            diag.compute(st)

    # final-state evaluation (post-trajectory state, explicit NFE)
    final = res.structure
    r_final = score.compute(final)
    d_min_f, cov_def_f = structure_diagnostics(final)
    # NNP-only force (Pauli wall force excluded) — OOD spike accounting only
    f_nnp_f = (r_final.metadata or {}).get("nnp_forces", r_final.forces)
    max_f_f = float(np.linalg.norm(f_nnp_f, axis=1).max()) if f_nnp_f is not None and len(f_nnp_f) else None
    e_final = r_final.energy
    finite_ok = np.isfinite(d_min_f) and (max_f_f is None or np.isfinite(max_f_f)) \
        and (e_final is None or np.isfinite(e_final))
    # Validity per the DiffCSP/MatterGen-aligned definition (paper §6.0.7):
    # composition (fixed in Phase 1, satisfied by construction) + d_min ≥ 0.5 Å;
    # non-finite values count as failures. max|F_NNP| > 500 is an OOD
    # criterion, not a validity criterion (paper §6.1).
    valid = bool(finite_ok and d_min_f >= OOD_DMIN)

    # Type III / lattice-collapse volume checks use the INITIAL cell |det L0|
    # as reference (paper §6.1 OOD metric, constraint manifold Eq. 4.2) — not
    # the final volume, which would mask cumulative inflation.
    masks, agg = classify_steps(diag.records, final.num_atoms, initial.volume)
    e_hull = None
    if e_final is not None and np.isfinite(e_final):
        e_hull = hull.e_hull(final.properties.get("formula") or ref_record["metadata"]["pretty_formula"],
                             e_final, final.num_atoms)

    d_min_hist = [round(float(r["d_min"]), 3) for r in diag.records]
    return {
        "seed": seed, "cand": cand,
        # Which prior the start came from ("reference"/"composition") and, for
        # the composition arm, the split the cell was taken from + the cell
        # volume: the arm's whole point is that the start is not the target,
        # so the payload has to say what it was instead.
        "prior": prior,
        "init_cell_source": (cell_record or {}).get("source"),
        "init_volume": round(float(initial.volume), 4),
        "init_d_min": d_min_hist[0] if d_min_hist else None,
        "nfe": nfe, "n_rejections": n_rej,
        "cap_steps": cap_steps, "cap_atoms": cap_atoms,
        "sigma_final": sigma_final, "converged": converged,
        "valid": valid,
        "ood_steps": agg["n_ood_steps"],
        "type1_steps": agg["type1_steps"], "type2_steps": agg["type2_steps"],
        "type3_steps": agg["type3_steps"], "nan_steps": agg["n_nan_steps"],
        "collapse_steps": agg["collapse_steps"],
        "ood_bitmask": masks,
        "d_min_hist": d_min_hist,
        "final": {"d_min": round(d_min_f, 3),
                  "max_f": None if max_f_f is None else round(float(max_f_f), 2),
                  "energy": None if e_final is None else float(e_final),
                  "e_hull": None if e_hull is None else round(e_hull, 4),
                  "structure": final.to_dict()},
        "wall_time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

def build_tasks(subexp: str, nnp: str):
    """List of (task_dict, out_path)."""
    tasks = []
    if subexp == "1":
        comps = COMPOSITIONS_20
        for sigma in SIGMAS:
            for prot in PROTECTIONS + SIGMA_SCAN_ALD_ONLY:
                for sampler in SAMPLERS:
                    if sampler == "pfode" and prot in SIGMA_SCAN_ALD_ONLY:
                        continue      # ALD-only arm, see SIGMA_SCAN_ALD_ONLY
                    for comp in comps:
                        for seed in SEEDS:
                            tag = f"s{sigma:g}_{prot}_{sampler}"
                            tasks.append(({"subexp": "1", "nnp": nnp, "sigma": sigma,
                                           "protection": prot, "sampler": sampler,
                                           "composition": comp, "seed": seed},
                                          OUT / "subexp1" / tag / f"{comp}_seed{seed}.json"))
    elif subexp == "2":
        levels = SUBEXP2_LEVELS_DOC12
        for level in levels:
            for comp in COMPOSITIONS_20:
                for seed in SEEDS:
                    tasks.append(({"subexp": "2", "nnp": nnp, "sigma": 1.0,
                                   "level": level, "composition": comp, "seed": seed},
                                  OUT / "subexp2" / level / f"{comp}_seed{seed}.json"))
    elif subexp == "sm":
        for p in SM_P:
            for d_safe in SM_DSAFE:
                for comp in RECHECK_COMPS:
                    for seed in SEEDS:
                        tasks.append(({"subexp": "sm", "nnp": nnp, "sigma": 1.0,
                                       "p": p, "d_safe": d_safe,
                                       "composition": comp, "seed": seed},
                                      OUT / "smscan" / f"p{p:g}_d{d_safe:g}"
                                      / f"{comp}_seed{seed}.json"))
    elif subexp == "fine":
        # Step-size control (reviewer request): is the bare
        # score's failure at 200 NFE a property of the score or of the
        # discretization?  Same sigma schedule, same SDE time, 4x finer steps
        # (see ALD_ALPHA).  Kept in its own tree so no existing tag, shard
        # index, or analysis glob can pick these up as scan output.
        for prot, sigmas in FINE_ARMS:
            for sigma in sigmas:
                for comp in COMPOSITIONS_20:
                    for seed in SEEDS:
                        tag = f"m{FINE_STEP_MULT}_s{sigma:g}_{prot}_ald"
                        tasks.append(({"subexp": "fine", "nnp": nnp,
                                       "sigma": sigma, "protection": prot,
                                       "sampler": "ald", "step_mult": FINE_STEP_MULT,
                                       "composition": comp, "seed": seed},
                                      OUT / "subexp1_fine" / tag
                                      / f"{comp}_seed{seed}.json"))
    elif subexp == "5":
        for pair, (comp, (z1, z2)) in SUBEXP5_PAIRS.items():
            conditions = [(None, "zbl")] + [(a, b, f"a{a:g}_b{b:g}") for (a, b) in PAULI_GRID]
            for cond in conditions:
                for seed in SEEDS:
                    task = {"subexp": "5", "nnp": nnp, "sigma": 1.0, "pair": pair,
                            "composition": comp, "seed": seed}
                    if len(cond) == 3:
                        a, b, ctag = cond
                        task.update({"A": a, "B": b})
                    else:
                        _, ctag = cond
                    tasks.append((task,
                                  OUT / "subexp5" / pair / ctag / f"{comp}_seed{seed}.json"))
    elif subexp == "comp":
        # E-1 composition-only prior (reviewer: the initial state
        # hands the sampler the test structure's coordinates and cell).  Same
        # sigma grid, arms, sampler set and seeds as the scan (so --mini
        # shrinks it the same way), but every trajectory starts from uniform
        # coordinates on a train/val cell of the composition -- see
        # make_initial_comp_only.  Its own tree: no existing tag or analysis
        # glob may pick these up as scan output.
        for sigma in SIGMAS:
            for prot in PROTECTIONS:
                for sampler in SAMPLERS:
                    for comp in COMPOSITIONS_20:
                        for seed in SEEDS:
                            tag = f"c_s{sigma:g}_{prot}_{sampler}"
                            tasks.append(({"subexp": "comp", "nnp": nnp,
                                           "sigma": sigma, "protection": prot,
                                           "sampler": sampler, "prior": "composition",
                                           "composition": comp, "seed": seed},
                                          OUT / "componly" / tag
                                          / f"{comp}_seed{seed}.json"))
    elif subexp == "rc":
        sigmas_rc = [0.5, 1.0, 2.0, 5.0]
        for sigma in sigmas_rc:
            for prot in PROTECTIONS:
                for comp in RECHECK_COMPS:
                    for seed in SEEDS:
                        tasks.append(({"subexp": "rc", "nnp": nnp, "sigma": sigma,
                                       "protection": prot, "sampler": "ald",
                                       "composition": comp, "seed": seed},
                                      OUT / "recheck" / f"s{sigma:g}_{prot}_ald"
                                      / f"{comp}_seed{seed}.json"))
            for comp in RECHECK_COMPS:
                for seed in SEEDS:
                    tasks.append(({"subexp": "rc", "nnp": nnp, "sigma": sigma,
                                   "protection": "bare", "sampler": "pfode",
                                   "composition": comp, "seed": seed},
                                  OUT / "recheck" / f"s{sigma:g}_bare_pfode"
                                  / f"{comp}_seed{seed}.json"))
    else:
        raise ValueError(subexp)
    return tasks


def run_task(task: dict, calc, refs: dict, hull: HullEvaluator) -> dict:
    comp = task["composition"]
    ref_record = refs[comp]
    n_atoms = len(ref_record["numbers"])

    # E_hull calibration: NNP energy of the reference (primitive) structure.
    # Kept on the test reference for every prior, including the composition-only
    # arm: it is one scalar per composition (the NNP-vs-DFT offset of a fixed
    # structure) that shifts every E_hull of that composition by a constant, so
    # holding it fixed is what keeps the arms' E_hull comparable.  It reaches
    # the analysis, never the sampler.
    ref_crystal = CrystalStructure.from_frac_coords(
        ref_record["numbers"], ref_record["frac_coords"], ref_record["lattice"])
    e_ref = calc_ref_energy(calc, ref_crystal)
    hull.calibrate(comp, e_ref, n_atoms,
                   float(ref_record["metadata"].get("formation_energy_per_atom", 0.0)))

    pauli_params = None
    if task["subexp"] == "5":
        pair = task["pair"]
        if task.get("A") is not None:
            (z1, z2) = SUBEXP5_PAIRS[pair][1]
            from materialgen.core.pauli import pair_key
            pauli_params = {pair_key(z1, z2): {"A": task["A"], "B": task["B"]}}

    # Resolve the effective protection level (doc-12 grid: each subexp-2
    # level name is an exact layer subset — see LEVEL_SUBSETS above).
    if task["subexp"] == "2":
        prot = task["level"]
    elif task["subexp"] == "sm":
        prot = "l1l4"                            # full stack, monitor parameters scanned
    elif task["subexp"] == "5":
        prot = "l1"                              # Pauli-only (L1)
    else:
        prot = task["protection"]                # subexp 1 / rc: bare / l1 / l1l4

    sm_params = None
    if task["subexp"] == "sm":
        sm_params = {"p": task["p"], "d_safe": task["d_safe"]}
    score = make_score(calc, prot, pauli_params, label=nnp_label(task["nnp"]))

    sigma_max = task["sigma"]
    sampler_name = task.get("sampler", "ald")
    if sampler_name == "ald":
        cfg = make_ald_config(sigma_max, prot, task.get("step_mult", 1))
        sampler = make_ald_sampler(cfg, prot, sm_params)
    else:
        cfg = make_pfode_config(sigma_max, prot)
        sampler = make_pfode_sampler(cfg)

    prior = task.get("prior", "reference")
    cell_record = None
    if prior == "composition":
        cell_record = load_composition_cells([comp])[comp]
        # n_atoms comes from the *test* reference (the fixed composition, so
        # the same atom count); the cell and the coordinates do not.
        assert len(cell_record["numbers"]) == n_atoms, (
            f"{comp}: composition cell has {len(cell_record['numbers'])} atoms, "
            f"reference has {n_atoms}")

    candidates = []
    for cand in range(N_CAND):
        c = run_candidate(score, sampler, ref_record, sigma_max,
                          task["seed"], cand, hull, prior=prior,
                          cell_record=cell_record)
        c["composition"] = comp
        c["formula"] = comp
        candidates.append(c)

    payload = {**task, "n_atoms": n_atoms, "n_candidates": N_CAND,
               "prior_cell_source": (cell_record or {}).get("source"),
               "e_hull_units": "eV/atom",   # pre-fix files lack this
               # (2nd protocol bug): noise_lattice's cholesky .T made
               # every make_initial cell distorted by 45-114% in a sigma-
               # INDEPENDENT way.  Files without this marker were generated with
               # the unfixed noise_lattice and CANNOT be corrected in the
               # analysis layer (the trajectories themselves ran in the wrong
               # cell) -- do not pool them with marked files.
               "init_cell_gen": INIT_CELL_GEN,
               "candidates": candidates}
    return payload


_REF_ENERGY_CACHE: dict = {}


def calc_ref_energy(calc, crystal: CrystalStructure) -> float:
    key = id(calc)
    cache = _REF_ENERGY_CACHE.setdefault(key, {})
    ckey = (tuple(crystal.atomic_numbers), crystal.frac_coords.tobytes())
    if ckey not in cache:
        atoms = crystal.ase_atoms
        atoms.calc = calc
        try:
            cache[ckey] = float(atoms.get_potential_energy())
        finally:
            atoms.calc = None
    return cache[ckey]


def nnp_label(nnp: str) -> str:
    return "eSEN-30M-MPTrj" if nnp == "esen" else "MACE-MP-0"


def load_calculator(nnp: str, device: str):
    if nnp == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", device=device, default_dtype="float32")
    if nnp == "esen":
        from materialgen.nnp.esen import load_esen_calculator
        return load_esen_calculator(device=device)
    raise ValueError(nnp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", required=True,
                        choices=["1", "2", "5", "sm", "rc", "fine", "comp"])
    parser.add_argument("--nnp", default="mace", choices=["mace", "esen"])
    parser.add_argument("--shard", default=None, help="k/N: run tasks where index % N == k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mini", action="store_true",
                        help="small-scale validation batch: "
                             "5 σ × 5 comps × 2 seeds × 10 candidates, "
                             "subexp 1/2 only — verifies method correctness "
                             "on one GPU before the full fleet restarts")
    parser.add_argument("--samplers", nargs="+", default=None,
                        choices=["ald", "pfode"],
                        help="restrict to these samplers (only for subexps whose "
                             "tasks name one: 1, rc, fine, comp).  Needed because "
                             "a comp-arm pilot that includes PF-ODE costs 5-10x "
                             "an ALD-only one, and the two can be run separately.")
    parser.add_argument("--pilot", action="store_true",
                        help="1 sanity task: bare ALD sigma=1.0 SrTiO3 seed 42, print metrics")
    args = parser.parse_args()

    if args.mini:
        if args.subexp not in ("1", "2", "comp"):
            parser.error("--mini supports --subexp 1, 2 and comp only "
                         "(sm/5/rc run in the full fleet)")
        global SIGMAS, SEEDS, N_CAND, COMPOSITIONS_20
        SIGMAS, SEEDS, N_CAND, COMPOSITIONS_20 = (
            MINI_SIGMAS, MINI_SEEDS, MINI_N_CAND, MINI_COMPOSITIONS)
        print(f"MINI mode: sigmas={MINI_SIGMAS} comps={MINI_COMPOSITIONS} "
              f"seeds={MINI_SEEDS} cand={MINI_N_CAND}", flush=True)

    if args.pilot:
        calc = load_calculator(args.nnp, args.device)
        refs = load_reference_structures(["SrTiO3"])
        hull = HullEvaluator(load_hull_table())
        task = {"subexp": "1", "nnp": args.nnp, "sigma": 1.0, "protection": "bare",
                "sampler": "ald", "composition": "SrTiO3", "seed": 42}
        t0 = time.time()
        payload = run_task(task, calc, refs, hull)
        dt = time.time() - t0
        cands = payload["candidates"]
        n_ood = sum(1 for c in cands if c["ood_steps"] > 0)
        n_valid = sum(1 for c in cands if c["valid"])
        print(f"PILOT bare ALD sigma=1.0 SrTiO3: {dt:.1f}s wall, "
              f"{payload['n_atoms']} atoms, {len(cands)} candidates")
        print(f"  OOD rate: {n_ood}/{len(cands)} = {n_ood/len(cands)*100:.1f}%  "
              f"(expect >50%) | valid: {n_valid}/{len(cands)}")
        nfe = sum(c["nfe"] for c in cands)
        print(f"  {nfe} NFE total, {dt/max(nfe,1)*1000:.1f} ms/NFE")
        eh = [c["final"]["e_hull"] for c in cands if c["final"]["e_hull"] is not None]
        if eh:
            print(f"  E_hull: n={len(eh)}, mean={np.mean(eh)*1000:.1f} meV/atom")
        return

    tasks = build_tasks(args.subexp, args.nnp)
    if args.samplers:
        # subexp 2/5/sm tasks carry no "sampler" key (2 and sm are ALD by
        # construction, 5 is ALD-only), so filtering them would silently empty
        # the queue -- refuse instead of running zero tasks.
        unnamed = sum(1 for t, _ in tasks if "sampler" not in t)
        if unnamed:
            parser.error(f"--samplers cannot be applied to --subexp "
                         f"{args.subexp}: {unnamed} of its tasks do not name a "
                         f"sampler")
        before = len(tasks)
        tasks = [t for t in tasks if t[0]["sampler"] in args.samplers]
        print(f"--samplers {args.samplers}: {before} -> {len(tasks)} tasks",
              flush=True)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % n == k]

    calc = load_calculator(args.nnp, args.device)
    refs = load_reference_structures(sorted({t[0]["composition"] for t in tasks}))
    hull = HullEvaluator(load_hull_table())

    n_done, n_fail, n_skip = 0, 0, 0
    for task, path in tasks:
        if path.exists():
            # a finished file is only a valid skip if it carries
            # the current cell-init generation marker.  A pre-fix file left in
            # the live tree (or one written before this protocol change) must
            # be re-run, not silently kept -- that is exactly how the first
            # bug's data would re-enter the analysis.
            try:
                ok = (json.loads(path.read_text()).get("init_cell_gen")
                      == INIT_CELL_GEN)
            except Exception:
                ok = False
            if ok:
                n_skip += 1
                continue
            print(f"STALE {path}: missing init_cell_gen={INIT_CELL_GEN!r}, "
                  "re-running", flush=True)
        try:
            payload = run_task(task, calc, refs, hull)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            n_done += 1
        except Exception as exc:
            n_fail += 1
            print(f"FAILED {task}: {exc}", flush=True)
    print(f"worker done: {n_done} new tasks, {n_skip} already current, "
          f"{n_fail} failures", flush=True)


if __name__ == "__main__":
    main()
