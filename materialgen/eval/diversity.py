"""
Structural diversity and novelty metrics for generated crystal ensembles.

    TrainingIndex
        Formula-keyed index of a dataset's training split, answering the S.U.N.
        Novelty question (paper section 6.0.7): a generated structure is novel
        if it matches no training structure of the same composition under the
        StructureMatcher tolerances.  Built lazily per reduced formula, pickled
        next to the dataset's processed files so a fleet of workers pays the
        CIF-parsing cost once.

    compute_novelty(generated, index) -> dict
        {n, n_novel, novelty_rate} -- index may be a TrainingIndex or any
        sequence of structures (then the same-composition filter is applied by
        reduced formula).

    compute_diversity(structures) -> dict
        AMSD (see metrics.compute_amsd), space-group entropy, lattice-parameter
        variance, and the mode-collapse metric.

    compute_mode_collapse_metric(structures) -> float
        Entropy of the pairwise-distance histogram; low values mean the
        ensemble is dominated by near-duplicates.

Not implemented: SOAP/ACSF fingerprint diversity (dscribe is not installed in
either project environment).  The stub's `cluster_structures` is likewise
deferred -- no caller needs clustering yet.

Dependencies:
    numpy, pymatgen, ase
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

import collections
import pickle
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .metrics import (compute_amsd, compute_amsd_matrix, detect_spacegroup,
                      make_matcher, _as_pmg)

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class TrainingIndex:
    """Training-split structures grouped by reduced formula, for Novelty."""

    def __init__(self, groups: Optional[dict] = None, name: str = "<memory>",
                 split: str = "train", matcher=None, preset: str = "paper"):
        self.groups: dict[str, list] = groups or {}
        self.name = name
        self.split = split
        self.preset = preset
        self.matcher = matcher or make_matcher(preset)

    # -- construction -------------------------------------------------------

    @staticmethod
    def cache_path(dataset: str, split: str = "train") -> Path:
        return INDEX_DIR / dataset / f"novelty_index_{split}.pkl"

    @classmethod
    def from_dataset(cls, dataset: str, split: str = "train",
                     use_cache: bool = True, preset: str = "paper",
                     max_structures: Optional[int] = None) -> "TrainingIndex":
        """Build (or load) the index from a dataset CSV split."""
        from ..data.registry import DATASETS

        spec = DATASETS[dataset]
        name = spec.name
        cache = cls.cache_path(name, split)
        if use_cache and cache.exists() and max_structures is None:
            with open(cache, "rb") as f:
                return cls(pickle.load(f), name=name, split=split, preset=preset)

        import pandas as pd
        from ..utils.crystal_io import from_pymatgen

        df = pd.read_csv(spec.csv_dir / f"{split}.csv")
        if max_structures is not None:
            df = df.iloc[:max_structures]
        groups: dict[str, list] = collections.defaultdict(list)
        for _, row in df.iterrows():
            from ..data.datasets import _parse_cif

            crystal = _parse_cif(row["cif"])
            rec = from_pymatgen(_as_pmg([crystal])[0])
            groups[_reduced_formula(rec)].append(_as_pmg([rec])[0])
        index = cls(dict(groups), name=name, split=split, preset=preset)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(index.groups, f)
        return index

    # -- queries ------------------------------------------------------------

    def same_composition(self, formula: str) -> list:
        return self.groups.get(_reduced_formula_str(formula), [])

    def is_novel(self, structure) -> Optional[bool]:
        """True iff no training structure of the same composition matches.

        Candidate prefilter: with pymatgen's default
        ``subset=False`` and ``attempt_supercell=False``, ``StructureMatcher.fit``
        returns False for any pair with a different number of sites -- it builds
        the species mask of shape (max(n1, n2), min(n1, n2)) and returns None
        from ``_strict_match`` as soon as the two axis lengths differ, before any
        lattice or coordinate work.  Skipping those pairs here is therefore exact
        (not an approximation) and removes the dominant cost on datasets whose
        training split has a single composition: Carbon-24 has 6,091 train
        structures all of composition "C" and 19 possible site counts, so a
        candidate is compared against ~1/19 of the pool instead of all of it.
        Guarded so the filter can never fire under a configuration where the
        equivalence stops holding.
        """
        from ..utils.crystal_io import to_pymatgen

        pmg = structure if hasattr(structure, "sites") else to_pymatgen(structure)
        pool = self.groups.get(pmg.composition.reduced_formula)
        if not pool:
            return True
        if not (getattr(self.matcher, "subset", False)
                or getattr(self.matcher, "attempt_supercell", False)):
            n = len(pmg)
            pool = [ref for ref in pool if len(ref) == n]
            if not pool:
                return True
        return not any(self.matcher.fit(pmg, ref) for ref in pool)

    def novel_mask(self, structures: Sequence) -> list:
        return [self.is_novel(s) for s in structures]

    def stats(self) -> dict:
        return {"name": self.name, "split": self.split,
                "n_formulas": len(self.groups),
                "n_structures": sum(len(v) for v in self.groups.values())}


def _reduced_formula(crystal) -> str:
    from ..utils.crystal_io import to_pymatgen

    return to_pymatgen(crystal).composition.reduced_formula


def _reduced_formula_str(formula: str) -> str:
    from pymatgen.core import Composition

    return Composition(formula).reduced_formula


def compute_novelty(generated: Sequence, index) -> dict:
    """Novelty rate of an ensemble against a training split or structure list."""
    if not isinstance(index, TrainingIndex):
        pool = index
        index = TrainingIndex()
        for s in pool:
            index.groups.setdefault(_reduced_formula(s), []).append(s)
    flags = index.novel_mask(generated)
    n = len(flags)
    known = [f for f in flags if f is not None]
    return {"n": n, "n_novel": sum(1 for f in known if f),
            "novelty_rate": (sum(1 for f in known if f) / len(known)) if known else float("nan"),
            "index": index.stats()}


def compute_mode_collapse_metric(structures: Sequence) -> float:
    """Entropy of the pairwise-RMSD histogram; ~0 means near-duplicate ensemble."""
    if len(structures) < 2:
        return float("nan")
    m = compute_amsd_matrix(structures)
    vals = m[np.triu_indices(len(m), k=1)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    lo, hi = float(vals.min()), float(vals.max())
    # Degenerate spread: an ensemble of copies gives one distance, and by
    # symmetry even distinct structures can give pairwise distances that differ
    # only in float noise (e.g. three cubic cells: 1e-16 of spread).  Either way
    # the distribution is a single spike, i.e. zero entropy, and np.histogram
    # cannot build 20 distinct bins over such a range.
    if hi - lo <= 1e-9 * max(1.0, abs(lo), abs(hi)):
        return 0.0
    h, _ = np.histogram(vals, bins=20, range=(lo, hi))
    p = h / h.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_diversity(structures: Sequence) -> dict:
    """Diversity summary of one composition's ensemble.

    Every metric here is geometric (AMSD, space-group entropy, lattice/volume
    spread), so no StructureMatcher tolerance enters -- see
    metrics._pair_rmsd_min_image.
    """
    from ..utils.crystal_io import to_pymatgen

    pmg = [to_pymatgen(s) for s in structures]
    n = len(pmg)
    if n == 0:
        return {"n": 0}
    sgs = [detect_spacegroup(s) for s in pmg]
    counts = collections.Counter(z for z in sgs if z is not None)
    total = sum(counts.values())
    entropy = float(-sum((c / total) * np.log(c / total) for c in counts.values())) if total else float("nan")
    lengths = np.array([s.lattice.abc for s in pmg])          # (n, 3)
    volumes = np.array([s.volume for s in pmg])
    return {
        "n": n,
        "amsd": compute_amsd(pmg),
        "mode_collapse_entropy": compute_mode_collapse_metric(pmg),
        "spacegroup_entropy": entropy,
        "n_spacegroups": len(counts),
        "spacegroup_histogram": dict(sorted(counts.items())),
        "lattice_length_std": lengths.std(axis=0).tolist(),
        "volume_mean": float(volumes.mean()),
        "volume_std": float(volumes.std()),
        "volume_cv": float(volumes.std() / volumes.mean()) if volumes.mean() else float("nan"),
    }
