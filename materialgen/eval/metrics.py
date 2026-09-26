"""
Core evaluation metrics for generated crystal structures (paper section 6.0.7).

Implemented here (all structure-level, dataset-agnostic -- the per-dataset
references and hulls are injected by the caller or by StructureEvaluator):

    compute_validity(structures, compositions=None) -> dict
        DiffCSP/MatterGen-aligned validity (delegates to
        materialgen.analysis.metrics): composition match + d_min >= 0.5 A.
        Returns {n, n_valid, validity_rate, reasons: {reason: count}}.

    compute_match_rate(generated, reference, matcher=None, preset="paper") -> dict
        Fraction of generated structures matching *some* ground-truth test
        structure (pymatgen StructureMatcher).  Returns
        {n, matched, match_rate}.

    compute_coverage(generated, reference, ...) -> dict
        Recall of the test set: fraction of ground-truth structures matched by
        at least one generated structure.  Returns {n_ref, covered, coverage}.

    compute_amsd(structures, ...) -> float
        Average Minimum Self-Distance: mean over i of min_{j != i} d_RMS(x_i, x_j),
        with d_RMS the periodic-aware optimal-rotation/translation RMSD.  Defined
        for every pair of an ensemble: the matcher's permutation-optimized value
        when the pair matches, the species-sorted Kabsch distance otherwise
        (_pair_rmsd_min_image).

    compute_r_angle(generated, reference, metric="kl") -> dict
        Bond-angle distribution divergence.  Angles are triplets i-j-k with both
        bonds below 3.0 A, binned into 36 bins of 5 deg over [0, 180].  The paper
        specifies KL(P_gen || P_ref); "wasserstein" is available because the
        CDVAE/DiffCSP baseline scripts report R-angle as a Wasserstein distance,
        and the head-to-head table must use the baselines' own convention.

    compute_symmetry_score(generated, expected=None) -> dict
        Fraction of generated structures whose detected space group matches the
        expected one (per-structure list, or a set of accepted numbers).

    compute_sun(records, ...) -> dict
        S.U.N. rate -- Stable (E_hull < 100 meV/atom), Unique (no other generated
        structure of the same composition with the same space group, volume
        within 10%, and E_hull within 20 meV/atom), Novel (no match to the
        training set).

Tolerances.  Two presets are defined: "paper" (ltol=0.2, stol=0.3,
angle_tol=5.0) is the paper's own convention for Match Rate / Coverage /
Novelty; "baseline" (ltol=0.3, stol=0.5, angle_tol=10.0) reproduces the
official CDVAE and DiffCSP scripts/compute_metrics.py, so baseline numbers can
be reproduced or compared on identical footing.  Always report which one a
number came from.

StructureEvaluator aggregates the suite over an ensemble.  See tests/test_metrics.py.

Dependencies:
    numpy, scipy, pymatgen (StructureMatcher, SpacegroupAnalyzer), ase
    materialgen.core.crystal.CrystalStructure
    materialgen.analysis.metrics (validity definition)
    materialgen.utils.crystal_io
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .stability import (E_HULL_FLOOR, E_HULL_STABLE, E_HULL_UNITS,
                        is_physical_e_hull)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

MATCH_PRESETS = {
    # paper section 6.0.7 (Match Rate, Coverage, S.U.N. Novelty)
    "paper": dict(ltol=0.2, stol=0.3, angle_tol=5.0),
    # third_party/{cdvae,DiffCSP}/scripts/compute_metrics.py (RecEval default)
    "baseline": dict(ltol=0.3, stol=0.5, angle_tol=10.0),
}

R_ANGLE_CUTOFF = 3.0      # A, bond cutoff for the angle triplets
R_ANGLE_BINS = 36         # 5 deg each over [0, 180]
AMSD_FALLBACK_SHIFTS = 1  # +-1 cell in each direction for the Kabsch fallback


def make_matcher(preset: str = "paper", **overrides):
    """pymatgen StructureMatcher for a named tolerance preset.

    Defaults: ``primitive_cell=False`` (generated cells are already the target
    cell, and the reference structures are conventional cells; reducing both
    would compare standardized cells neither side produced) and ``scale=False``.
    Both are overridable through ``overrides`` -- a supercell-vs-primitive
    comparison needs ``primitive_cell=True``.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    if preset not in MATCH_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; use one of {list(MATCH_PRESETS)}")
    opts = {"primitive_cell": False, "scale": False}
    opts.update(MATCH_PRESETS[preset])
    opts.update(overrides)
    return StructureMatcher(**opts)


def _as_pmg(structures: Iterable) -> list:
    """Coerce a mixed iterable of CrystalStructure / pymatgen Structure."""
    from ..utils.crystal_io import as_pymatgen

    return as_pymatgen(structures)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def compute_validity(structures: Sequence, compositions: Optional[Sequence] = None) -> dict:
    """Structural validity: composition match (when given) + d_min >= 0.5 A.

    Diverged trajectories (non-finite positions) count as invalid, per the
    statistical protocol of paper section 6.0.9.
    """
    from ..analysis.metrics import validity_reasons

    comps = list(compositions) if compositions is not None else [None] * len(structures)
    if len(comps) != len(structures):
        raise ValueError("compositions must be None or align with structures")

    counts: collections.Counter = collections.Counter()
    n_valid = 0
    for s, c in zip(structures, comps):
        try:
            reasons = validity_reasons(s, c)
        except Exception as exc:
            reasons = [f"exception: {type(exc).__name__}"]
        if reasons:
            for r in reasons:
                counts[r.split("(")[0].strip()] += 1
        else:
            n_valid += 1
    n = len(list(structures))
    return {"n": n, "n_valid": n_valid,
            "validity_rate": (n_valid / n) if n else float("nan"),
            "reasons": dict(counts)}


# ---------------------------------------------------------------------------
# Match rate / coverage
# ---------------------------------------------------------------------------

def _reference_groups(reference: Sequence, matcher) -> list:
    """One representative per structural equivalence class of the reference set."""
    return [group[0] for group in matcher.group_structures(list(reference))]


def compute_match_rate(generated: Sequence, reference: Sequence,
                       matcher=None, preset: str = "paper",
                       formula: Optional[str] = None) -> dict:
    """Fraction of generated structures matching some reference structure."""
    matcher = matcher or make_matcher(preset)
    gen = _as_pmg(generated)
    ref = _as_pmg(reference)
    reps = _reference_groups(ref, matcher)
    matched = sum(1 for g in gen if any(matcher.fit(g, r) for r in reps))
    n = len(gen)
    out = {"n": n, "n_reference": len(ref), "matched": matched,
           "match_rate": (matched / n) if n else float("nan"), "preset": preset}
    if formula is not None:
        out["formula"] = formula
    return out


def compute_coverage(generated: Sequence, reference: Sequence,
                     matcher=None, preset: str = "paper",
                     formula: Optional[str] = None) -> dict:
    """Fraction of reference structures matched by >= 1 generated structure."""
    matcher = matcher or make_matcher(preset)
    gen = _as_pmg(generated)
    ref = _as_pmg(reference)
    reps = _reference_groups(gen, matcher)
    covered = sum(1 for r in ref if any(matcher.fit(r, g) for g in reps))
    n = len(ref)
    out = {"n_reference": n, "n_generated": len(gen), "covered": covered,
           "coverage": (covered / n) if n else float("nan"), "preset": preset}
    if formula is not None:
        out["formula"] = formula
    return out


def compute_match_and_coverage(generated: Sequence, reference: Sequence,
                               matcher=None, preset: str = "paper",
                               formula: Optional[str] = None) -> dict:
    """Both directions in one pass over the reference equivalence classes."""
    matcher = matcher or make_matcher(preset)
    gen = _as_pmg(generated)
    ref = _as_pmg(reference)
    gen_reps = _reference_groups(gen, matcher)
    ref_reps = _reference_groups(ref, matcher)
    matched = sum(1 for g in gen if any(matcher.fit(g, r) for r in ref_reps))
    covered = sum(1 for r in ref if any(matcher.fit(r, g) for g in gen_reps))
    return {
        "formula": formula, "preset": preset,
        "n_generated": len(gen), "n_reference": len(ref),
        "matched": matched, "covered": covered,
        "match_rate": (matched / len(gen)) if gen else float("nan"),
        "coverage": (covered / len(ref)) if ref else float("nan"),
    }


# ---------------------------------------------------------------------------
# AMSD
# ---------------------------------------------------------------------------

class ReferenceIndex:
    """A reference ensemble pre-partitioned for exact same-composition matching.

    ``StructureMatcher.fit`` (paper preset: ``subset=False``,
    ``attempt_supercell=False``) returns False for any pair whose reduced
    formula or site count differ -- it builds the species mask and bails before
    any lattice or coordinate work.  Partitioning the reference once by
    ``(reduced formula, n_sites)`` therefore cannot change any match, and it
    removes the case that makes the plain helpers unusable at Phase-2 scale:

        MP-20 / Perov-5   a cell's reference is a handful of structures of one
                          composition -- no difference;
        Carbon-24         a cell's reference is *every* test structure of
                          composition "C" (thousands, all the same formula), so
                          ``group_structures`` over it is a quadratic scan of
                          millions of fits, run twice per cell per config.

    Bucketing by site count cuts that by the number of distinct site counts,
    and only the buckets a candidate pool actually touches are ever grouped.
    Coverage keeps the full-reference denominator, so leaving untouched
    buckets alone changes nothing but the work.

    The reference-side quantities that do not depend on the generated
    ensemble (equivalence-class representatives, pooled bond angles) are
    cached, so an analysis pass over many configurations pays for them once
    per cell instead of once per config.
    """

    def __init__(self, reference: Sequence, matcher=None, preset: str = "paper"):
        self.reference = _as_pmg(reference)
        self.preset = preset
        self.matcher = matcher or make_matcher(preset)
        self.buckets: dict[tuple[str, int], list] = collections.defaultdict(list)
        for s in self.reference:
            self.buckets[_sites_key(s)].append(s)
        self._ref_reps: dict[tuple[str, int], list] = {}
        self._ref_angles: dict[float, np.ndarray] = {}

    # -- reference-side caches ---------------------------------------------

    def _reps(self, key: tuple[str, int]) -> list:
        """One representative per structural equivalence class of a bucket."""
        if key not in self._ref_reps:
            self._ref_reps[key] = [g[0] for g in
                                   self.matcher.group_structures(self.buckets[key])]
        return self._ref_reps[key]

    def reference_angles(self, cutoff: float = R_ANGLE_CUTOFF) -> np.ndarray:
        """Pooled reference bond angles, computed once per cell and cached.

        On Carbon-24 this is the expensive half of the analysis, not the
        matching: a cell's reference is 2030 all-carbon structures whose pooled
        angles are 1.4e7 values, measured at 207 s per cell (MACE
        host under load).  Caching it on the index -- rather than recomputing
        per configuration -- is what makes the 20-cell x 6-config Carbon-24
        analysis pass a ~70 min job instead of a ~7 h one; the second
        configuration on the same cell drops from 546 s to 48 s.
        """
        if cutoff not in self._ref_angles:
            self._ref_angles[cutoff] = _pooled_angles(self.reference, cutoff)
        return self._ref_angles[cutoff]

    # -- metrics ------------------------------------------------------------

    def match_coverage(self, generated: Sequence,
                       formula: Optional[str] = None) -> dict:
        """Same result and keys as ``compute_match_and_coverage``."""
        gen = _as_pmg(generated)
        gen_buckets: dict[tuple[str, int], list] = collections.defaultdict(list)
        for s in gen:
            gen_buckets[_sites_key(s)].append(s)

        matched = 0
        for key, gs in gen_buckets.items():
            if key not in self.buckets:
                continue
            reps = self._reps(key)
            matched += sum(1 for g in gs if any(self.matcher.fit(g, r) for r in reps))

        covered = 0
        for key, refs in self.buckets.items():
            gs = gen_buckets.get(key)
            if not gs:
                continue
            g_reps = [g[0] for g in self.matcher.group_structures(gs)]
            covered += sum(1 for r in refs if any(self.matcher.fit(r, g) for g in g_reps))

        n_gen, n_ref = len(gen), len(self.reference)
        return {
            "formula": formula, "preset": self.preset,
            "n_generated": n_gen, "n_reference": n_ref,
            "matched": matched, "covered": covered,
            "match_rate": (matched / n_gen) if n_gen else float("nan"),
            "coverage": (covered / n_ref) if n_ref else float("nan"),
            "n_buckets_touched": len(set(gen_buckets) & set(self.buckets)),
        }

    def r_angle(self, generated: Sequence, metric: str = "kl",
                cutoff: float = R_ANGLE_CUTOFF, bins: int = R_ANGLE_BINS) -> dict:
        """Same result and keys as ``compute_r_angle`` (reference side cached)."""
        gen_angles = _pooled_angles(generated, cutoff)
        ref_angles = self.reference_angles(cutoff)
        p = angle_histogram(gen_angles, bins)
        q = angle_histogram(ref_angles, bins)
        return {"metric": metric, "value": _angle_divergence(p, q, metric, bins),
                "bins": bins, "cutoff": cutoff,
                "n_generated_angles": int(gen_angles.size),
                "n_reference_angles": int(ref_angles.size),
                "p_generated": p.tolist(), "p_reference": q.tolist()}


def _sites_key(structure) -> tuple[str, int]:
    """(reduced formula, site count): the necessary conditions of ``fit``."""
    return (structure.composition.reduced_formula, len(structure))


def _kabsch_rmsd(a_cart: np.ndarray, b_cart: np.ndarray) -> float:
    """Optimal-rotation/translation RMSD between two centred Cartesian sets."""
    a = a_cart - a_cart.mean(axis=0)
    b = b_cart - b_cart.mean(axis=0)
    cov = a.T @ b
    try:
        u, _, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return float(np.sqrt(((a - b) ** 2).sum() / len(a)))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    diff = a @ rot - b
    return float(np.sqrt((diff ** 2).sum() / len(a)))


def _canonical_sites(structure) -> tuple[np.ndarray, list]:
    """Cartesian coordinates in a permutation-invariant canonical order.

    Sites are wrapped into the unit cell and sorted by (Z, fractional
    coordinates), so two structures that differ only in the order their sites
    were listed yield the same point set and the same correspondence.  A
    species-only sort would not: it is stable, so it keeps the *input* order
    within a species and a permuted copy would be compared atom-to-wrong-atom.
    """
    s = structure.to_unit_cell() if hasattr(structure, "to_unit_cell") else structure
    frac = np.asarray(s.frac_coords) % 1.0
    order = sorted(range(len(s)), key=lambda i: (s[i].specie.Z, *frac[i]))
    return (np.asarray(s.cart_coords, dtype=float)[order],
            [int(s[i].specie.Z) for i in order])


def _pair_rmsd_min_image(a, b, matcher=None) -> Optional[float]:
    """Periodic-aware RMSD between two pymatgen Structures, in A.

    This is AMSD's ``d_RMS`` (paper section 6.0.7): the minimum-image
    optimal-rotation/translation (Kabsch) distance between the two Cartesian
    point sets, over the canonical atoms correspondence of
    ``_canonical_sites``.

    Two deliberate choices, both load-bearing:

    * The value is in **A**, not pymatgen's normalized unit.
      ``StructureMatcher.get_rms_dist`` divides by the free length per atom,
      ``(V/N)^(1/3)`` -- a dimensionless number that cannot be compared with,
      or averaged against, a length.  Using it as d_RMS would silently make
      AMSD a strain-like ratio (and would make an ensemble's AMSD depend on
      its cell size).  The matcher's value is therefore not used for the
      distance at all.
    * The distance is defined for **every** pair of same-composition,
      same-atom-count structures.  The matcher is tolerance-gated -- it returns
      None for any pair it cannot fit, i.e. exactly the far-apart pairs a
      diversity statistic is made of -- so an AMSD built on it is NaN or
      truncated in the regime it exists to measure.

    Periodic images: candidate translations of ``-1..1`` cells of each lattice
    are applied to ``b`` and the best one is kept (both are tried only when the
    cells differ; centring means one cell suffices at these cell sizes).  The
    rotation is a pure point-cloud Kabsch, so a pair whose cells differ is
    compared by shape rather than by cell agreement -- which is what
    distinguishes two polymorphs, the case AMSD has to score.

    Returns None when there is no correspondence to speak of: different
    composition, atom count, or species multiset.
    """
    if a.composition.reduced_formula != b.composition.reduced_formula:
        return None
    if len(a) != len(b):
        return None
    ca, za = _canonical_sites(a)
    cb, zb = _canonical_sites(b)
    if za != zb:
        return None
    lattices = [a.lattice.matrix]
    if not np.allclose(a.lattice.matrix, b.lattice.matrix, atol=1e-3):
        lattices.append(b.lattice.matrix)
    shifts = range(-AMSD_FALLBACK_SHIFTS, AMSD_FALLBACK_SHIFTS + 1)
    best = math.inf
    for lat in lattices:
        for i in shifts:
            for j in shifts:
                for k in shifts:
                    best = min(best, _kabsch_rmsd(ca, cb + np.array([i, j, k]) @ lat))
    return float(best)


def compute_amsd(structures: Sequence) -> float:
    """Average Minimum Self-Distance (paper section 6.0.7), in A.

    AMSD = mean_i min_{j != i} d_RMS(x_i, x_j), computed on the ensemble of K
    structures generated for one composition.  Higher = more diverse.  The
    distance is purely geometric -- see _pair_rmsd_min_image for why no
    StructureMatcher tolerance enters it -- so AMSD is defined for any ensemble
    of the same composition and atom count, however dissimilar its members.
    """
    if len(structures) < 2:
        return float("nan")
    pmg = _as_pmg(structures)
    n = len(pmg)
    mins = []
    for i in range(n):
        best = math.inf
        for j in range(n):
            if i == j:
                continue
            d = _pair_rmsd_min_image(pmg[i], pmg[j])
            if d is not None:
                best = min(best, d)
        if math.isfinite(best):
            mins.append(best)
    if not mins:
        return float("nan")
    return float(np.mean(mins))


def compute_amsd_matrix(structures: Sequence) -> np.ndarray:
    """Full symmetric pairwise RMSD matrix in A (NaN where undefined)."""
    pmg = _as_pmg(structures)
    n = len(pmg)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            d = _pair_rmsd_min_image(pmg[i], pmg[j])
            if d is not None:
                m[i, j] = m[j, i] = d
    return m


# ---------------------------------------------------------------------------
# R-Angle
# ---------------------------------------------------------------------------

def bond_angles(structure, cutoff: float = R_ANGLE_CUTOFF) -> np.ndarray:
    """All i-j-k angles (deg) with d_ij < cutoff and d_jk < cutoff."""
    from ase.neighborlist import neighbor_list
    from ..utils.crystal_io import to_pymatgen

    pmg = structure if hasattr(structure, "lattice") and not hasattr(structure, "ase_atoms") \
        else to_pymatgen(structure)
    atoms = pmg.to_ase_atoms() if hasattr(pmg, "to_ase_atoms") else pmg
    i_idx, j_idx, d_ij, D = neighbor_list("ijdD", atoms, cutoff)
    if len(i_idx) == 0:
        return np.empty(0)

    # neighbour lists: j -> [(neighbour, vector from j to neighbour)]
    nbrs: dict[int, list[tuple[int, np.ndarray]]] = collections.defaultdict(list)
    for k in range(len(i_idx)):
        i, j = int(i_idx[k]), int(j_idx[k])
        v = D[k]                       # vector from j to i (ase convention: D_k = pos_i - pos_j)
        nbrs[j].append((i, v))
        nbrs[i].append((j, -v))

    angles = []
    for j, lst in nbrs.items():
        m = len(lst)
        for a in range(m):
            for b in range(a + 1, m):
                va, vb = lst[a][1], lst[b][1]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na < 1e-8 or nb < 1e-8:
                    continue
                cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
                angles.append(math.degrees(math.acos(cos)))
    return np.array(angles)


def angle_histogram(angles: np.ndarray, bins: int = R_ANGLE_BINS) -> np.ndarray:
    """Normalized histogram over [0, 180] with `bins` equal bins."""
    h, _ = np.histogram(angles, bins=bins, range=(0.0, 180.0))
    total = h.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return h / total


def _pooled_angles(structures: Sequence, cutoff: float) -> np.ndarray:
    return np.concatenate([bond_angles(s, cutoff) for s in structures]) \
        if len(structures) else np.empty(0)


def _angle_divergence(p: np.ndarray, q: np.ndarray, metric: str, bins: int) -> float:
    """Divergence between two normalized angle histograms.

    "kl" is the paper definition (natural log); "wasserstein" is the baseline
    convention in the CDVAE/DiffCSP scripts.
    """
    if metric == "kl":
        eps = 1e-12
        pp = (p + eps) / (p.sum() + eps * bins)
        qq = (q + eps) / (q.sum() + eps * bins)
        return float(np.sum(pp * np.log(pp / qq)))
    if metric == "wasserstein":
        from scipy.stats import wasserstein_distance

        centers = (np.arange(bins) + 0.5) * (180.0 / bins)
        return float(wasserstein_distance(centers, centers, p, q))
    raise ValueError(f"unknown r-angle metric {metric!r}")


def compute_r_angle(generated: Sequence, reference: Sequence,
                    metric: str = "kl", cutoff: float = R_ANGLE_CUTOFF,
                    bins: int = R_ANGLE_BINS) -> dict:
    """Bond-angle distribution divergence between generated and reference sets.

    metric: "kl" (paper definition, natural log) or "wasserstein" (baseline
    convention in the CDVAE/DiffCSP scripts).
    """
    gen_angles = _pooled_angles(generated, cutoff)
    ref_angles = _pooled_angles(reference, cutoff)
    p = angle_histogram(gen_angles, bins)
    q = angle_histogram(ref_angles, bins)
    value = _angle_divergence(p, q, metric, bins)
    return {"metric": metric, "value": value, "bins": bins, "cutoff": cutoff,
            "n_generated_angles": int(gen_angles.size),
            "n_reference_angles": int(ref_angles.size),
            "p_generated": p.tolist(), "p_reference": q.tolist()}


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def detect_spacegroup(structure, symprec: float = 0.1) -> Optional[int]:
    """Space-group number (spglib via pymatgen), or None if it cannot be found."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from ..utils.crystal_io import to_pymatgen

    try:
        return int(SpacegroupAnalyzer(to_pymatgen(structure),
                                      symprec=symprec).get_space_group_number())
    except Exception:
        return None


def compute_symmetry_score(generated: Sequence, expected: Optional[Sequence] = None,
                           symprec: float = 0.1) -> dict:
    """Fraction of generated structures with the expected space group.

    `expected` is either a per-structure sequence of space-group numbers, or a
    collection of accepted numbers applied to every structure.
    """
    numbers = [detect_spacegroup(s, symprec) for s in generated]
    if expected is None:
        return {"n": len(numbers), "spacegroups": numbers,
                "symmetry_score": float("nan")}
    if isinstance(expected, (set, frozenset, tuple)) and all(
            isinstance(e, (int, np.integer)) for e in expected):
        accepted = set(int(e) for e in expected)
        hit = sum(1 for z in numbers if z is not None and z in accepted)
    else:
        exp = list(expected)
        if len(exp) != len(numbers):
            raise ValueError("expected must be a per-structure sequence or a set of numbers")
        hit = sum(1 for z, e in zip(numbers, exp) if z is not None and z == e)
    return {"n": len(numbers), "n_matched": hit, "spacegroups": numbers,
            "symmetry_score": (hit / len(numbers)) if numbers else float("nan")}


# ---------------------------------------------------------------------------
# S.U.N.
# ---------------------------------------------------------------------------

@dataclass
class StructureRecord:
    """One evaluated generated structure; the unit the S.U.N. criterion acts on."""

    formula: str
    structure: Any
    energy: Optional[float] = None          # NNP total energy, eV
    e_hull: Optional[float] = None          # eV/atom
    e_form: Optional[float] = None          # eV/atom
    spacegroup: Optional[int] = None
    volume: Optional[float] = None
    valid: Optional[bool] = None
    novel: Optional[bool] = None
    unique: Optional[bool] = None
    stable: Optional[bool] = None
    physical: Optional[bool] = None          # E_hull not below the two-sided band
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.volume is None and self.structure is not None:
            # via crystal_io: a pymatgen Structure's `.lattice` is a Lattice
            # object, so it has no matrix to take a determinant of.
            from ..utils.crystal_io import to_pymatgen

            self.volume = float(to_pymatgen(self.structure).volume)
        if self.physical is None and self.e_hull is not None:
            self.physical = is_physical_e_hull(self.e_hull)
        if self.stable is None and self.e_hull is not None:
            # Two-sided (see E_HULL_FLOOR): a blow-up must not read as stable.
            self.stable = bool(-E_HULL_FLOOR <= self.e_hull < E_HULL_STABLE)

    @property
    def sun(self) -> bool:
        return bool(self.valid) and bool(self.stable) \
            and bool(self.unique) and bool(self.novel)


def mark_unique(records: Sequence[StructureRecord], volume_rtol: float = 0.10,
                e_hull_tol: float = 0.02) -> None:
    """In-place: mark records that duplicate an earlier (lower-E_hull) one.

    Paper section 6.0.7: a generated structure is not unique if another
    generated structure of the same composition shares its space group, has a
    volume within ``volume_rtol``, and an E_hull within ``e_hull_tol``
    (eV/atom).  Records are processed in ascending E_hull so the best member of
    each duplicate cluster is the one kept -- physical records first, so an
    NNP extrapolation (E_hull below the two-sided band, see E_HULL_FLOOR) can
    never be the representative a real structure is compared against.  On a
    dataset without extrapolations that tie-break is unreachable, which is why
    it cannot move any Phase-1 number.
    """
    by_formula: dict[str, list[StructureRecord]] = collections.defaultdict(list)
    for r in records:
        by_formula[r.formula].append(r)

    for group in by_formula.values():
        ordered = sorted(
            group,
            key=lambda r: (r.e_hull is None, r.physical is False,
                           r.e_hull if r.e_hull is not None else 0.0),
        )
        kept: list[StructureRecord] = []
        for r in ordered:
            dup = False
            if r.spacegroup is not None and r.e_hull is not None and r.volume:
                for k in kept:
                    if k.spacegroup != r.spacegroup or k.e_hull is None or not k.volume:
                        continue
                    vol_close = abs(k.volume - r.volume) / max(k.volume, r.volume) < volume_rtol
                    if vol_close and abs(k.e_hull - r.e_hull) < e_hull_tol:
                        dup = True
                        break
            r.unique = not dup
            if not dup:
                kept.append(r)


def compute_sun(records: Sequence[StructureRecord], mark: bool = True,
                volume_rtol: float = 0.10, e_hull_tol: float = 0.02) -> dict:
    """S.U.N. rate: fraction of generated structures that are S, U and N.

    ``mark=True`` computes uniqueness in place (see mark_unique); set it False
    only if the records already carry a ``unique`` flag.
    """
    if mark:
        mark_unique(records, volume_rtol=volume_rtol, e_hull_tol=e_hull_tol)
    n = len(records)
    if n == 0:
        return {"n": 0, "sun_rate": float("nan")}

    def frac(pred) -> float:
        known = [r for r in records if pred(r) is not None]
        if not known:
            return float("nan")
        return sum(1 for r in known if pred(r)) / len(known)

    n_sun = sum(1 for r in records if r.sun)
    # `physical is False` (not `not physical`): an uncalibrated record has
    # physical=None and is not an extrapolation.
    n_unphys = sum(1 for r in records if r.physical is False)
    unphys = [r.e_hull for r in records
              if r.physical is False and r.e_hull is not None]
    return {
        "n": n,
        "n_valid": sum(1 for r in records if r.valid),
        "n_stable": sum(1 for r in records if r.stable),
        "n_unique": sum(1 for r in records if r.unique),
        "n_novel": sum(1 for r in records if r.novel),
        "n_sun": n_sun,
        "sun_rate": n_sun / n,
        "validity_rate": frac(lambda r: r.valid),
        "stable_rate": frac(lambda r: r.stable),
        "unique_rate": frac(lambda r: r.unique),
        "novel_rate": frac(lambda r: r.novel),
        # E_hull below the band's lower edge: an NNP extrapolation, not a
        # result.  Reported (never silently dropped) so the affected fraction
        # and its magnitude are visible next to the rate it would inflate.
        "n_unphysical": n_unphys,
        "unphysical_rate": n_unphys / n,
        "unphysical_e_hull_min": float(min(unphys)) if unphys else None,
        "unphysical_e_hull_median": float(np.median(unphys)) if unphys else None,
        "e_hull_units": E_HULL_UNITS,
        "e_hull_threshold": E_HULL_STABLE,
        "e_hull_floor": E_HULL_FLOOR,
        "volume_rtol": volume_rtol, "e_hull_tol": e_hull_tol,
    }


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    metrics: dict
    per_structure: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.metrics)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.per_structure)

    def print_summary(self) -> None:
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k:<22s} {v:.4f}")
            elif isinstance(v, dict):
                continue
            else:
                print(f"  {k:<22s} {v}")


class StructureEvaluator:
    """Full metric suite over one ensemble of generated structures.

    ``reference``   structures of the same composition from the dataset's test
                    split (for match rate / coverage / R-angle reference).
    ``training``    NoveltyIndex over the dataset's train split (for Novelty).
    ``hull``        calibrated HullEvaluator (for E_form / E_hull / Stable).
    ``formula``     target composition; required for the hull lookup.

    Every reference/hull argument is optional -- the evaluator computes what it
    has inputs for and reports NaN for the rest, so the same object serves the
    hull-carrying datasets and the third-party adapters.
    """

    def __init__(self, reference: Optional[Sequence] = None,
                 training=None, hull=None, matcher=None,
                 preset: str = "paper", formula: Optional[str] = None):
        self.reference = list(reference) if reference is not None else None
        self.training = training
        self.hull = hull
        self.preset = preset
        self.matcher = matcher or make_matcher(preset)
        self.formula = formula

    # -- per-structure enrichment ------------------------------------------

    def build_records(self, generated: Sequence, energies: Optional[Sequence] = None,
                      formulas: Optional[Sequence] = None,
                      relax: bool = False, calculator=None) -> list[StructureRecord]:
        """Evaluate one ensemble into StructureRecords (validation, hull, novelty)."""
        from ..analysis.metrics import validity_reasons
        from ..utils.crystal_io import to_pymatgen

        n = len(generated)
        energies = list(energies) if energies is not None else [None] * n
        formulas = list(formulas) if formulas is not None else [self.formula] * n
        records = []
        for s, e, f in zip(generated, energies, formulas):
            if relax and calculator is not None:
                from .relaxation import relax_structure

                s = relax_structure(s, calculator=calculator)
            pmg = to_pymatgen(s)
            rec = StructureRecord(formula=f, structure=pmg,
                                  energy=e if e is None else float(e),
                                  volume=float(pmg.volume))
            rec.valid = not validity_reasons(s, f)
            rec.spacegroup = detect_spacegroup(s)
            if self.hull is not None and f is not None and rec.energy is not None:
                rec.e_form = self.hull.e_form_per_atom(f, rec.energy, pmg.num_sites)
                rec.e_hull = self.hull.e_hull(f, rec.energy, pmg.num_sites)
                rec.physical = is_physical_e_hull(rec.e_hull)
                # Two-sided (see E_HULL_FLOOR): an NNP blow-up has a huge
                # negative E_hull and must not be counted as Stable.
                rec.stable = (None if rec.e_hull is None
                              else bool(-E_HULL_FLOOR <= rec.e_hull < E_HULL_STABLE))
            if self.training is not None:
                rec.novel = self.training.is_novel(pmg)
            records.append(rec)
        return records

    # -- full report --------------------------------------------------------

    def evaluate(self, records: Sequence[StructureRecord]) -> EvaluationReport:
        """Metric suite over already-built records (see build_records)."""
        metrics: dict = {"n": len(records), "preset": self.preset,
                         "formula": self.formula}
        structures = [r.structure for r in records]

        valid = [r.structure for r in records if r.valid]
        validity = compute_validity(structures, [r.formula for r in records])
        metrics["validity_rate"] = validity["validity_rate"]
        metrics["n_valid"] = validity["n_valid"]
        metrics["sun"] = compute_sun(records)

        if self.reference:
            metrics["match_coverage"] = compute_match_and_coverage(
                structures, self.reference, self.matcher, self.preset, self.formula)
            metrics["r_angle_kl"] = compute_r_angle(structures, self.reference, "kl")
            metrics["r_angle_wasserstein"] = compute_r_angle(
                structures, self.reference, "wasserstein")
        if len(valid) >= 2:
            metrics["amsd"] = compute_amsd(valid)

        e_hulls = [r.e_hull for r in records if r.e_hull is not None]
        if e_hulls:
            from .stability import hull_summary

            metrics["e_hull"] = hull_summary(e_hulls)
        metrics["per_composition"] = collections.Counter(
            r.formula for r in records).most_common()
        return EvaluationReport(metrics=metrics, per_structure=[
            {"formula": r.formula, "energy": r.energy, "e_form": r.e_form,
             "e_hull": r.e_hull, "spacegroup": r.spacegroup, "volume": r.volume,
             "valid": r.valid, "stable": r.stable, "physical": r.physical,
             "unique": r.unique, "novel": r.novel, "sun": r.sun} for r in records])
