"""Generation-quality metrics for the composition-only arm (E-1).

scripts/analyze_componly.py answers the *safety* half of E-1 (induced OOD,
validity, E_hull).  This script answers the half the A5 protocol bullet
actually asks for: with the target structure withheld, what survives of the
match rate, coverage, AMSD and S.U.N. numbers?

Those are the quantities \S\ref{sec:limitations} warns about -- "the reported
match rates and coverage measure recovery of a known structure's neighbourhood
rather than structure generation from composition" -- so the point of this
comparison is not a new benchmark but the size of the gap between the two
priors on the *same* test compositions, seeds, candidate count, potential and
metric suite.

Protocol, matched to run_phase2's per-cell block so the numbers are comparable
with the Phase-2 tables:
  match rate / coverage   metrics.ReferenceIndex(cell).match_coverage(...)
  AMSD                    metrics.compute_amsd(valid structures)
  S.U.N.                  metrics.compute_sun over StructureRecords built with
                          the stored E_hull (stable), the stored validity, and
                          novelty from the cached MP-20 train index
Both arms are restricted to the same (composition, seed) set, so the two
columns differ only in the prior.  Match/coverage/AMSD/S.U.N. use the
validity-filtered pool (valid structures only), as in Phase 2, and the
unfiltered match rate is reported alongside.

Output: results/phase1/analysis/componly_metrics.json (+ stdout table).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import make_percomp_sigma_tables as mps  # noqa: E402
from materialgen.eval.diversity import TrainingIndex  # noqa: E402
from materialgen.eval.metrics import (ReferenceIndex, StructureRecord,  # noqa: E402
                                      compute_amsd, compute_sun)
from materialgen.utils.crystal_io import to_pymatgen  # noqa: E402

BASE = os.path.join(_ROOT, "results", "phase1", "componly")
TEST_PKL = os.path.join(_ROOT, "data", "processed", "mp_20", "test.pkl")
OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "componly_metrics.json")


def load_test_reference():
    """Test-split structures per composition -- the evaluation target.

    Evaluation-side only: the sampler never sees these (that is the whole
    point of the arm), but match rate and coverage are defined against them.
    The pickle holds raw arrays (numbers/frac_coords/lattice), so each record
    is rebuilt as a pymatgen Structure -- what ReferenceIndex wants.
    """
    import pickle

    from materialgen.core.crystal import CrystalStructure
    with open(TEST_PKL, "rb") as fh:
        records = pickle.load(fh)
    out: dict[str, list] = {}
    for r in records:
        formula = r["metadata"].get("pretty_formula")
        if formula is None:
            continue
        cs = CrystalStructure.from_frac_coords(
            np.asarray(r["numbers"]), np.asarray(r["frac_coords"]),
            np.asarray(r["lattice"], dtype=float))
        out.setdefault(formula, []).append(to_pymatgen(cs))
    return out


def arm_candidates(sig: float, arm: str, base: str, restrict=None) -> list:
    """All candidates of one cell; `restrict` limits to a (comp, seed) set."""
    files = sorted(glob.glob(os.path.join(base, f"*{arm}*", "*.json")))
    cands = []
    for p in files:
        j = json.load(open(p))
        key = (j["composition"], j["seed"])
        if restrict is not None and key not in restrict:
            continue
        cands.extend(j["candidates"])
    return cands


def records_for(cands: list, comp: str, index: TrainingIndex) -> list:
    """StructureRecords from stored payloads (+ novelty from the train index).

    The stored E_hull is used as-is rather than recomputed: it was produced by
    the same calibrated HullEvaluator as the reference arm's, which is what
    makes the two comparable (see run_phase1.run_task's calibration note).
    """
    from materialgen.core.crystal import CrystalStructure

    recs = []
    for c in cands:
        st = c["final"]["structure"]
        try:
            # task files store CrystalStructure.to_dict(), not the object
            pmg = to_pymatgen(CrystalStructure.from_dict(st))
        except Exception:
            continue
        rec = StructureRecord(formula=comp, structure=pmg,
                              energy=c["final"].get("energy"),
                              e_hull=c["final"].get("e_hull"),
                              valid=bool(c["valid"]))
        rec.novel = index.is_novel(pmg)
        recs.append(rec)
    return recs


def cell_metrics(recs: list, ref_index: ReferenceIndex, comp: str) -> dict:
    structures = [r.structure for r in recs]
    valid = [r.structure for r in recs if r.valid]
    sun = compute_sun(recs)          # marks uniqueness in place
    out = {"n": len(recs), "n_valid": len(valid),
           "validity": len(valid) / len(recs) if recs else None,
           "sun": sun["sun_rate"],
           "n_stable": sun.get("n_stable"), "n_novel": sun.get("n_novel"),
           "match_coverage": ref_index.match_coverage(valid, formula=comp) if valid else None,
           "match_coverage_all": ref_index.match_coverage(structures, formula=comp),
           "amsd": compute_amsd(valid) if len(valid) >= 2 else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["bare", "l1", "l1l4"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cells = sorted(glob.glob(os.path.join(BASE, "c_s*_*_ald")))
    if not cells:
        raise SystemExit(f"no composition-only cells under {BASE}")
    sigmas = args.sigmas or sorted({float(os.path.basename(d).split("_")[1][1:])
                                    for d in cells})

    test_ref = load_test_reference()
    index = TrainingIndex.from_dataset("mp_20", "train")
    print(f"train index: {index.stats() if hasattr(index, 'stats') else ''}")
    # One ReferenceIndex over the test split of every composition the arm uses.
    comps_used = set()
    for d in cells:
        for p in glob.glob(os.path.join(d, "*.json")):
            comps_used.add(json.load(open(p))["composition"])
    ref_structs = [r for c in sorted(comps_used) for r in test_ref.get(c, [])]
    ref_index = ReferenceIndex(ref_structs)
    print(f"reference index: {len(ref_structs)} test structures over "
          f"{len(comps_used)} compositions\n")

    rows = {}
    print(f"{'sigma':>5s} {'arm':5s} {'prior':11s} {'n':>5s} {'valid':>6s} "
          f"{'match':>6s} {'cover':>6s} {'AMSD':>6s} {'SUN':>6s} {'novel':>6s}")
    for sig in sigmas:
        for arm in args.arms:
            comp_dir = os.path.join(BASE, f"c_s{sig:g}_{arm}_ald")
            comp_files = sorted(glob.glob(os.path.join(comp_dir, "*.json")))
            if not comp_files:
                continue
            comp_cell: dict = {}
            restrict = set()
            for p in comp_files:
                j = json.load(open(p))
                restrict.add((j["composition"], j["seed"]))
                comp_cell.setdefault(j["composition"], []).extend(j["candidates"])
            ref_dir = os.path.join(mps.BASE, f"s{sig:g}_{mps.DIR_ARM[(arm, 'ald')]}_ald")
            ref_cell: dict = {}
            for p in sorted(glob.glob(os.path.join(ref_dir, "*.json"))):
                j = json.load(open(p))
                if (j["composition"], j["seed"]) not in restrict:
                    continue
                ref_cell.setdefault(j["composition"], []).extend(j["candidates"])

            for prior, cell in (("composition", comp_cell), ("reference", ref_cell)):
                if not cell:
                    continue
                per = {c: cell_metrics(records_for(cl, c, index), ref_index, c)
                       for c, cl in cell.items()}
                m = {"n": sum(v["n"] for v in per.values()),
                     "n_valid": sum(v["n_valid"] for v in per.values()),
                     "per_composition": per}
                # Micro-average the rates over compositions (each contributes
                # equally, as in the Phase-2 per-cell tables).
                for key in ("validity", "sun"):
                    vals = [v[key] for v in per.values() if v.get(key) is not None]
                    m[key] = float(np.mean(vals)) if vals else None
                for key in ("match_rate", "coverage"):
                    vals = [(v["match_coverage"] or {}).get(key) for v in per.values()
                            if v.get("match_coverage")]
                    m[key] = float(np.mean(vals)) if vals else None
                for key in ("amsd",):
                    vals = [v[key] for v in per.values() if v.get(key) is not None]
                    m[key] = float(np.mean(vals)) if vals else None
                m["n_novel"] = sum(v.get("n_novel") or 0 for v in per.values())
                m["n_stable"] = sum(v.get("n_stable") or 0 for v in per.values())
                rows[f"s{sig:g}_{arm}_{prior}"] = {"sigma": sig, "arm": arm,
                                                   "prior": prior, **m}
                print(f"{sig:5g} {arm:5s} {prior:11s} {m['n']:5d} "
                      f"{m['validity']*100:5.1f}% "
                      f"{(m['match_rate'] or float('nan'))*100:5.1f}% "
                      f"{(m['coverage'] or float('nan'))*100:5.1f}% "
                      f"{(m['amsd'] or float('nan')):6.3f} "
                      f"{(m['sun'] or float('nan'))*100:5.1f}% "
                      f"{m['n_novel']/max(m['n'], 1)*100:5.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print(f"\nwrote {os.path.relpath(args.out, _ROOT)}")


if __name__ == "__main__":
    main()
