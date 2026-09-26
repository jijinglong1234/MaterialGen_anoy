"""
Third-party generative baselines: official output loaders.

The learned baselines (DiffCSP, CDVAE) are trained and sampled in their own
repositories.  What we consume is the artifact their official eval script
writes -- ``eval_gen*.pt`` -- and this module turns it into the same
``CrystalStructure`` list our own samplers produce, so the Phase-2 metric
chain scores every method identically.  That is what makes Table 2 a
like-for-like comparison rather than a transcription of numbers from papers
that use different stability definitions.

Two official layouts exist (verified against the checkpoints under
``checkpoints/diffcsp`` and ``third_party/checkpoints``)::

    DiffCSP  <ds>_gen/eval_gen_official.pt
        frac_coords (N_total, 3)     concatenated over samples
        num_atoms   (n,)             per-sample atom counts
        atom_types  (N_total, 100)   *logits* over the MP-20 vocabulary;
                                     Z = argmax + 1
        lengths, angles (n, 3)
    CDVAE    cdvae_mp20/eval_gen.pt
        frac_coords (1, N_total, 3)  dim 0 is a dummy batch axis; the atom
                                     axis is the concatenation of every
                                     sample's atoms (no padding)
        num_atoms   (1, n)           per-sample counts, same order
        atom_types  (1, N_total)     integer Z already
        lengths, angles (1, n, 3)
    DiffCSP  <ds>_csp/eval_diff*.pt          (composition-conditioned arm)
        frac_coords (n_evals, N_total, 3)
        num_atoms   (n_evals, n_struct)
        atom_types  (n_evals, N_total)   conditioned ground-truth Z
        lengths, angles (n_evals, n_struct, 3)
        input_data_batch                 the conditioned test structures

The two rank-3 layouts differ in what dim 0 means (CDVAE: batch; DiffCSP-CSP:
eval), so the CSP artifact is identified by its ``input_data_batch`` rather
than by shape.

Both are converted with pymatgen's
``Lattice.from_parameters(a, b, c, alpha, beta, gamma)`` -- the call
DiffCSP's own eval script makes -- so a lattice-convention mismatch would
show up as a validity/coverage collapse rather than as a silent difference.

Note on the atom-type logits: DiffCSP's generation script writes the raw
(unsupervised) output of its atom-type head, not argmaxed indices.  The
mapping index -> Z is ``argmax + 1`` (index 0 is H), which is the mapping the
published MP-20 numbers were computed with; ``load_generated`` applies it and
records the observed Z range in the provenance so a vocabulary shift cannot
pass unnoticed.

Dependencies:
    torch (to read the .pt artifacts), numpy, pymatgen
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..core.crystal import CrystalStructure

#: DiffCSP MP-20 atom-type vocabulary size (argmax+1 -> Z).
DIFFCSP_VOCAB = 100


def _lattice_matrix(lengths: Sequence[float], angles: Sequence[float]) -> np.ndarray:
    from pymatgen.core import Lattice

    a, b, c = (float(x) for x in lengths)
    al, be, ga = (float(x) for x in angles)
    return np.asarray(Lattice.from_parameters(a, b, c, al, be, ga).matrix, float)


def _structure(numbers, frac, lengths, angles) -> CrystalStructure:
    return CrystalStructure.from_frac_coords(
        np.asarray(numbers, dtype=int),
        np.asarray(frac, dtype=float),
        _lattice_matrix(lengths, angles),
    )


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def detect_format(blob: dict) -> str:
    """'diffcsp' | 'cdvae' | 'diffcsp_csp' from the artifact's own layout.

    The two rank-3 layouts (CDVAE's generation dump and DiffCSP's CSP dump)
    are structurally similar but index their dim 0 differently -- batch vs
    eval -- so rank alone is not enough: the CSP dump is identified by the
    ``input_data_batch`` it carries (the conditioned compositions).
    """
    if "input_data_batch" in blob:
        return "diffcsp_csp"
    for key in ("frac_coords", "num_atoms", "atom_types", "lengths", "angles"):
        if key not in blob:
            raise KeyError(f"eval artifact has no {key!r}: keys={sorted(blob)}")
    frac, na, at = (blob[k] for k in ("frac_coords", "num_atoms", "atom_types"))
    if na.ndim == 1 and frac.ndim == 2 and at.ndim == 2:
        return "diffcsp"
    if na.ndim == 2 and frac.ndim == 3 and at.ndim == 2:
        return "cdvae"
    raise ValueError(
        f"unrecognised eval layout: frac_coords{tuple(frac.shape)} "
        f"num_atoms{tuple(na.shape)} atom_types{tuple(at.shape)}")


def _from_diffcsp(blob: dict) -> tuple[list, np.ndarray]:
    frac = _to_numpy(blob["frac_coords"])
    counts = _to_numpy(blob["num_atoms"]).astype(int)
    logits = _to_numpy(blob["atom_types"])
    lengths = _to_numpy(blob["lengths"])
    angles = _to_numpy(blob["angles"])
    if logits.shape[1] != DIFFCSP_VOCAB:
        raise ValueError(f"atom-type vocabulary {logits.shape[1]} != {DIFFCSP_VOCAB}: "
                         "the argmax+1 -> Z mapping would be wrong")
    # argmax over the logit axis; +1 because the vocabulary is 1-based (H = 1)
    z = logits.argmax(axis=1).astype(int) + 1
    out, offset = [], 0
    for i, n in enumerate(counts):
        out.append(_structure(z[offset:offset + n], frac[offset:offset + n],
                              lengths[i], angles[i]))
        offset += int(n)
    if offset != len(z):
        raise ValueError(f"num_atoms sums to {offset} but atom_types has {len(z)} rows")
    return out, z


def _formula_from_z(z: np.ndarray) -> str:
    """Reduced formula of a Z list (counted, not deduplicated: a dict
    comprehension over Z keys would collapse repeats and report SiO for SiO2)."""
    import collections as _collections

    from pymatgen.core import Composition
    from pymatgen.core.periodic_table import Element

    counts = _collections.Counter(int(zz) for zz in z)
    return Composition({Element.from_Z(k): v for k, v in counts.items()}).reduced_formula


def _from_diffcsp_csp(blob: dict) -> tuple[list, np.ndarray]:
    """Composition-conditioned arm: ``evaluate.py`` output (``eval_diff*.pt``).

    ``evaluate.py`` stacks over ``num_evals`` on dim 0 and concatenates over
    loader batches on dim 1, so a sample is ``(eval, offset : offset+n)``.
    Unlike the generation arm the atom types are *conditioned* -- they are the
    ground-truth integers from the input batch, not logits -- which is what
    makes this the arm that can be scored on a named test composition.
    """
    frac = _to_numpy(blob["frac_coords"])
    counts = _to_numpy(blob["num_atoms"]).astype(int)
    z = _to_numpy(blob["atom_types"]).astype(int)
    lengths = _to_numpy(blob["lengths"])
    angles = _to_numpy(blob["angles"])
    out, zs, formulas = [], [], []
    for e in range(counts.shape[0]):
        if int(counts[e].sum()) != frac.shape[1]:
            raise ValueError(f"eval {e}: num_atoms sums to {int(counts[e].sum())} "
                             f"but frac_coords holds {frac.shape[1]} rows")
        offset = 0
        for j, n in enumerate(counts[e]):
            zi = z[e][offset:offset + n]
            if zi.min() < 1:
                raise ValueError(f"CSP sample {e}/{j} carries Z={zi.min()} "
                                 "(unconditioned or unmapped atom types)")
            out.append(_structure(zi, frac[e][offset:offset + n],
                                  lengths[e][j], angles[e][j]))
            zs.append(zi)
            formulas.append(_formula_from_z(zi))
            offset += int(n)
    return out, np.concatenate(zs) if zs else np.zeros(0, int), formulas


def _from_cdvae(blob: dict) -> tuple[list, np.ndarray]:
    """``eval_gen.pt`` from CDVAE's ``scripts/evaluate.py``.

    CDVAE's ``generation()`` returns
    ``torch.cat([d[k] for d in all_crystals]).unsqueeze(0)``: the outer axis is
    a *dummy* batch axis (always 1) and every sample's atoms are concatenated
    into one flat axis across all sampling batches -- there is no padding, and
    dim 1 of ``num_atoms`` is the per-sample count of that same flat axis
    (verified against the vendored code and the 1000-structure artifact:
    208 = sum(num_atoms) = frac_coords.shape[1]).
    """
    frac = _to_numpy(blob["frac_coords"])
    counts = _to_numpy(blob["num_atoms"]).astype(int).reshape(-1)
    z = _to_numpy(blob["atom_types"]).astype(int).reshape(-1)
    lengths = _to_numpy(blob["lengths"]).reshape(-1, 3)
    angles = _to_numpy(blob["angles"]).reshape(-1, 3)
    if int(counts.sum()) != frac.shape[1]:
        raise ValueError(f"num_atoms sums to {int(counts.sum())} but frac_coords "
                         f"holds {frac.shape[1]} rows")
    if len(z) != frac.shape[1]:
        raise ValueError(f"atom_types has {len(z)} rows, frac_coords {frac.shape[1]}")
    out, zs, offset = [], [], 0
    for j, n in enumerate(counts):
        zi = z[offset:offset + n]
        if zi.min() < 1:
            raise ValueError(f"CDVAE sample {j} carries Z={zi.min()} "
                             "(padding or an unmapped vocabulary)")
        out.append(_structure(zi, frac[0][offset:offset + n],
                              lengths[j], angles[j]))
        zs.append(zi)
        offset += int(n)
    return out, np.concatenate(zs) if zs else np.zeros(0, int)


_LOADERS = {"diffcsp": _from_diffcsp, "cdvae": _from_cdvae,
            "diffcsp_csp": _from_diffcsp_csp}


def load_generated(path: str | Path, fmt: str = "auto", limit: Optional[int] = None,
                   map_location: str = "cpu"):
    """Read an official DiffCSP/CDVAE artifact into (structures, provenance).

    Returns the ensemble as ``CrystalStructure`` objects plus a JSON-safe
    provenance dict.  For the composition-conditioned arm the ground-truth
    formula of each sample is attached as ``provenance["conditioned_formulas"]``
    (aligned with ``structures``) -- that is the composition its hull must be
    calibrated from.

    ``limit`` truncates the ensemble before any conversion (generation arm
    only, for pilots); the provenance records that it did.
    """
    import torch

    path = Path(path)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"{path}: expected a dict, got {type(blob).__name__}")
    fmt = detect_format(blob) if fmt == "auto" else fmt
    if fmt not in _LOADERS:
        raise ValueError(f"unknown format {fmt!r}; use one of {sorted(_LOADERS)}")

    if limit is not None and fmt != "diffcsp":
        raise ValueError("limit is implemented for the generation arm only")
    if limit is not None and fmt == "diffcsp":
        # truncated by sample count: keep the first `limit` atom rows of each block
        counts = _to_numpy(blob["num_atoms"]).astype(int)[:limit]
        keep = int(counts.sum())
        blob = dict(blob)
        for k in ("frac_coords", "atom_types"):
            blob[k] = blob[k][:keep]
        for k in ("lengths", "angles", "num_atoms"):
            blob[k] = blob[k][:limit]

    got = _LOADERS[fmt](blob)
    structures, z = got[0], got[1]
    conditioned = got[2] if len(got) > 2 else None
    setting = {
        "path": str(path),
        "format": fmt,
        "n_structures": len(structures),
        "n_atoms_total": int(sum(len(s) for s in structures)),
        "z_min": int(z.min()) if len(z) else None,
        "z_max": int(z.max()) if len(z) else None,
        "truncated_to": limit,
        "conditioned_formulas": conditioned,
    }
    ev = blob.get("eval_setting")
    if ev is not None:
        setting["eval_setting"] = vars(ev) if hasattr(ev, "__dict__") else str(ev)
    return structures, setting
