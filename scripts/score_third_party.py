"""
Phase 2 — score a third-party generative baseline with our metric chain.

Paper protocol (Table 2): the head-to-head against learned baselines is run
"on the identical test-set compositions and the identical metric suite" and on
standard generation-quality metrics (validity, match rate, coverage, S.U.N.,
AMSD, R-Angle KL) -- not on NFE-to-E_hull, which is reserved for the
same-PES efficiency comparison.  This script is the metric-chain half of that:
it consumes an official DiffCSP/CDVAE artifact (see
``materialgen.eval.third_party``) and scores every generated structure exactly
as the Phase-2 runner scores our own candidates:

    validity        composition + d_min >= 0.5 A (validity_reasons)
    E_form/E_hull   one single-point NNP energy on the dataset's
                    registry-calibrated hull -- no relaxation, the same
                    treatment a candidate's final step gets in run_phase1
    Novelty         TrainingIndex over the dataset's train split
    Uniqueness      within the ensemble, per composition (mark_unique)
    Match/coverage  vs the test split, restricted to same-composition groups
                    (exact prefilter: StructureMatcher.fit cannot match two
                    structures whose compositions differ)
    AMSD, R-angle   the same helpers the runner uses

Composition coverage caveat.  DiffCSP's generation arm is *unconditional* --
it draws compositions from the training histogram (``train_dist`` in
``generation.py``), not from the test split -- so most of its samples carry no
calibrated anchor and get ``e_hull = None``.  Those records are excluded from
``stable_rate`` (which conditions on a known composition) and counted as
not-Stable in ``*_all``; the fraction is reported as ``frac_hull_known``.
The composition-conditioned CSP arm (``eval_diff*.pt`` from
``third_party/DiffCSP/scripts/evaluate.py``, layout ``diffcsp_csp``) has no
such gap, which is why it is the arm that satisfies the paper's
"identical test-set compositions".

Usage:
    python scripts/score_third_party.py --input <artifact.pt> --dataset mp_20 \
        --nnp mace --name diffcsp_mp_gen [--limit 100] [--device cpu]

Output: results/phase2/third_party/<name>.json plus a row appended to
results/phase2/summary/phase2_third_party.csv (same columns as the Phase-2
summary where they exist).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np

from materialgen.data.registry import get_dataset
from materialgen.eval.metrics import (ReferenceIndex, StructureEvaluator,
                                      compute_amsd, compute_sun,
                                      compute_validity)
from materialgen.eval.stability import canonical_formula, hull_summary
from materialgen.eval.third_party import load_generated
from materialgen.utils.crystal_io import to_pymatgen

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase2" / "third_party"
CSV_PATH = REPO / "results" / "phase2" / "summary" / "phase2_third_party.csv"


def group_by_formula(structures) -> dict:
    groups: dict[str, list] = collections.defaultdict(list)
    for s in structures:
        groups[canonical_formula(to_pymatgen(s).composition)].append(s)
    return dict(groups)


def grouped_match_coverage(generated, reference, preset="paper", index=None) -> dict:
    """match/coverage against a full test split.

    Thin wrapper over ``ReferenceIndex.match_coverage`` (same exact
    (composition, site-count) prefilter), which is what the Phase-2 runner
    uses too -- so the baseline row and our own rows come from one
    implementation.  The extra keys describe how much of the reference the
    ensemble could possibly match: an unconditional baseline reproduces few
    of the test compositions, and a reference in a composition no sample
    reproduced cannot be covered by construction.
    """
    index = index or ReferenceIndex(reference, preset=preset)
    out = index.match_coverage(generated)
    gen_comps = set(group_by_formula(generated))
    ref_comps = set(group_by_formula(reference))
    out["n_formulas_shared"] = len(gen_comps & ref_comps)
    out["n_formulas_generated"] = len(gen_comps)
    out["generated_in_reference_composition"] = sum(
        len(v) for k, v in group_by_formula(generated).items() if k in ref_comps)
    out["note"] = ("per-(composition, site count) groups; reference structures "
                   "in compositions no sample reproduced cannot be covered")
    return out


def grouped_amsd(structures, min_members: int = 2) -> dict:
    """AMSD per composition, averaged over the compositions that can have one.

    AMSD is defined only between same-composition structures (see
    ``_pair_rmsd_min_image``), and an unconditional baseline generates mostly
    one sample per composition -- so the ensemble-level AMSD of such a run is
    NaN by construction, not by failure.  Reporting the per-composition mean
    is the analog of the Phase-2 protocol (AMSD within a cell's pool, then the
    mean over cells) and states how many compositions actually contributed.
    """
    groups = group_by_formula(structures)
    vals = {}
    for formula, group in groups.items():
        if len(group) < min_members:
            continue
        v = compute_amsd(group)
        if np.isfinite(v):
            vals[formula] = v
    return {
        "mean": float(np.mean(list(vals.values()))) if vals else None,
        "n_groups_used": len(vals),
        "n_groups": len(groups),
        "n_structures": len(structures),
        "min_members": min_members,
    }


def _unique_rate_multi(records) -> Optional[float]:
    """Uniqueness restricted to compositions that got more than one sample."""
    counts = collections.Counter(r.formula for r in records)
    multi = [r for r in records if counts[r.formula] > 1]
    if not multi:
        return None
    return sum(1 for r in multi if r.unique) / len(multi)


def energies_of(structures, calc, log_every: int = 100) -> list:
    out = []
    t0 = time.time()
    for i, s in enumerate(structures):
        atoms = s.ase_atoms
        atoms.calc = calc
        out.append(float(atoms.get_potential_energy()))
        if log_every and (i + 1) % log_every == 0:
            print(f"    energy {i + 1}/{len(structures)} "
                  f"({(time.time() - t0) / (i + 1):.2f} s/structure)", flush=True)
    return out


def main() -> None:
    import run_phase2 as p2

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="official eval artifact (.pt)")
    ap.add_argument("--dataset", required=True, choices=["mp_20", "perov_5", "carbon_24"])
    ap.add_argument("--nnp", default="mace", choices=["mace", "esen"])
    ap.add_argument("--name", required=True, help="row name, e.g. diffcsp_mp_gen")
    ap.add_argument("--fmt", default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N samples (generation arm only)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads for --device cpu; defaults to the "
                         "runner's DEFAULT_THREADS (see its CPU-budget note)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t_start = time.time()
    print(f"[load] {args.input}", flush=True)
    structures, prov = load_generated(args.input, fmt=args.fmt, limit=args.limit)
    conditioned = prov.get("conditioned_formulas")
    formulas = conditioned or [canonical_formula(to_pymatgen(s).composition)
                               for s in structures]
    print(f"  {prov['n_structures']} structures, fmt={prov['format']}, "
          f"Z in [{prov['z_min']}, {prov['z_max']}], "
          f"{'conditioned' if conditioned else 'unconditional'}", flush=True)

    spec = get_dataset(args.dataset)
    hull = spec.hull_evaluator(args.nnp, "test")
    index = spec.training_index("train")
    refs = [p2._record_crystal(r) for r in spec.load_reference(args.nnp, "test")]
    print(f"[ctx] {spec.name}: hull systems={len(spec.hull())}, "
          f"calibrated formulas={len(hull.calibrated_formulas)}, "
          f"train index={index.stats()['n_formulas']} formulas, "
          f"{len(refs)} reference structures", flush=True)

    print(f"[energy] {args.nnp} on {args.device}", flush=True)
    import run_phase1 as rp1

    if args.device == "cpu":
        n_thr = p2.DEFAULT_THREADS if args.threads is None else args.threads
        if n_thr:
            print(f"[cpu] intra-op threads = {p2.set_cpu_threads(n_thr)}",
                  flush=True)
    calc = rp1.load_calculator(args.nnp, args.device)
    energies = energies_of(structures, calc)

    ev = StructureEvaluator(reference=None, training=index, hull=hull)
    records = ev.build_records(structures, energies=energies, formulas=formulas)

    n_hull_known = sum(1 for r in records if r.e_hull is not None)
    n_unphys = sum(1 for r in records if r.physical is False)
    n = len(records)
    sun = compute_sun(records)
    validity = compute_validity(structures, formulas)
    valid_structures = [r.structure for r in records if r.valid]
    metrics = {
        "name": args.name,
        "dataset": args.dataset, "nnp": args.nnp,
        "provenance": prov,
        "n": n,
        "validity": validity,
        "sun": sun,
        "stable_rate_all": sum(1 for r in records if r.stable) / n if n else None,
        "novel_rate_all": sum(1 for r in records if r.novel) / n if n else None,
        "unique_rate_all": sum(1 for r in records if r.unique) / n if n else None,
        "frac_hull_known": n_hull_known / n if n else None,
        "n_hull_unknown": n - n_hull_known,
        "n_unphysical": n_unphys,
        "amsd": grouped_amsd(valid_structures),
        # Uniqueness is per composition inside mark_unique, so on an
        # unconditional ensemble (one sample per composition) it is close to
        # vacuous -- reported split out so the reader can see that.
        "unique_rate_multi": _unique_rate_multi(records),
        "per_composition_top": collections.Counter(formulas).most_common(10),
        "n_formulas_generated": len(set(formulas)),
    }
    hull_stats = hull_summary([r.e_hull for r in records if r.e_hull is not None])
    metrics["e_hull"] = hull_stats
    # Per-structure rows: the energies are the only NNP-dependent part, so
    # keeping them lets a later eSEN pass be a re-scoring rather than a rerun.
    metrics["per_structure"] = [
        {"formula": r.formula, "n_atoms": len(r.structure), "energy": r.energy,
         "e_form": r.e_form, "e_hull": r.e_hull, "valid": r.valid,
         "stable": r.stable, "physical": r.physical, "unique": r.unique,
         "novel": r.novel, "sun": r.sun, "spacegroup": r.spacegroup,
         "volume": r.volume} for r in records]

    print(f"[match] vs {len(refs)} test structures", flush=True)
    ref_index = ReferenceIndex(refs)
    metrics["match_coverage"] = grouped_match_coverage(
        structures, refs, index=ref_index)
    print(f"[r-angle] vs test split", flush=True)
    metrics["r_angle_kl"] = ref_index.r_angle(structures, "kl")

    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.json"
    path.write_text(json.dumps(metrics, indent=1, default=str))

    row = {
        "name": args.name, "dataset": args.dataset, "nnp": args.nnp,
        "n": n, "n_valid": sun["n_valid"],
        "validity_rate": validity["validity_rate"],
        "stable_rate": sun["stable_rate"], "stable_rate_all": metrics["stable_rate_all"],
        "unique_rate": sun["unique_rate"], "novel_rate": sun["novel_rate"],
        "sun_rate": sun["sun_rate"],
        "frac_hull_known": metrics["frac_hull_known"],
        "unphysical_rate": sun["unphysical_rate"],
        "match_rate": metrics["match_coverage"]["match_rate"],
        "coverage": metrics["match_coverage"]["coverage"],
        "amsd": metrics["amsd"]["mean"],
        "unique_rate_multi": metrics["unique_rate_multi"],
        "r_angle_kl": metrics["r_angle_kl"]["value"],
        "e_hull_med_phys_meV": (None if hull_stats.get("median_physical") is None
                                else hull_stats["median_physical"] * 1000),
    }
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            existing = [r for r in csv.DictReader(f) if r["name"] != args.name]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(existing + [row])

    print(f"\n  {args.name}: valid={validity['validity_rate']:.3f} "
          f"stable={sun['stable_rate']:.3f} (known comp) "
          f"unique={sun['unique_rate']:.3f} novel={sun['novel_rate']:.3f} "
          f"SUN={sun['sun_rate']:.3f}")
    amsd = metrics["amsd"]["mean"]
    amsd_txt = "n/a" if amsd is None else f"{amsd:.3f}"
    print(f"  match={metrics['match_coverage']['match_rate']:.3f} "
          f"coverage={metrics['match_coverage']['coverage']:.3f} "
          f"AMSd={amsd_txt} ({metrics['amsd']['n_groups_used']} compositions) "
          f"hull-known={metrics['frac_hull_known']:.3f}")
    print(f"  -> {path}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
