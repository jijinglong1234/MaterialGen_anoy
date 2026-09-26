"""E-1: does the sigma-failure picture survive a composition-only prior?

The reviewer's objection: `make_initial` takes the *test* reference structure
and perturbs it, so every arm starts from the answer's coordinates and cell.
An arm named "sigma-failure of a boundary" should not do that.

run_phase1 --subexp comp re-runs the scan with prior="composition":
    coordinates   iid U[0,1)^3              (no test coordinates at all)
    cell          a train/val structure of the same composition, uniformly
                  scaled to the reference's atom count so the two arms sample
                  the same number of atoms (see
                  run_phase1.load_composition_cells)
    everything else identical: sigma grid, arms, sampler, seeds, OOD mask,
    induced-OOD definition, E_hull protocol.

This script reports, per cell, the composition-only rate next to the reference
arm's rate from the frozen scan tree, plus the two quantities a composition-
only prior changes and that the published induced rate does not show:

  start-OOD share   fraction of trajectories whose very first recorded state
                    is already OOD.  Zero by construction for the reference
                    arm at small sigma (the start is the reference + sigma*eps)
                    but not here: uniform coordinates routinely place a pair
                    below 0.5 A.  Those trajectories are excluded from the
                    induced numerator (induced_k requires a clean start), so
                    the published denominator n_total dilutes this arm's rate.
  conditional rate  induced / n_start_clean -- the same numerator over the
                    trajectories that could actually be induced, i.e. the
                    composition-only counterpart of
                    results/phase1/analysis/conditional_ood.json.

The claim under test is the arm *ordering* within a sampler (protected < bare),
not the level: a weaker prior is expected to raise every arm's rate.  What
must survive is the ordering and the monotonicity in sigma.

Output: results/phase1/analysis/componly.json (+ stdout tables).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import make_percomp_sigma_tables as mps  # noqa: E402
from make_percomp_sigma_tables import OOD_MASK, SIGMAS, TEX, induced_k  # noqa: E402

BASE = os.path.join(_ROOT, "results", "phase1", "componly")
REF_BASE = mps.BASE
OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "componly.json")
E_HULL_FLOOR = 0.05       # eV/atom, run_phase1 / paper section 6.1 band
E_HULL_STABLE = 0.1
ARMS = ["bare", "l1", "l1l4"]
SAMP = ["ald", "pfode"]


def load_cell(sig: float, arm: str, smp: str) -> dict:
    """Composition-only cell: composition -> candidates.

    Unlike the frozen tables this tolerates a partial tree (the arm may be a
    pilot), so completeness is reported rather than asserted -- but the
    per-composition counts must still be uniform, otherwise a half-written
    cell would silently enter the pooled numbers with a smaller denominator.
    """
    d = os.path.join(BASE, f"c_s{sig:g}_{arm}_{smp}")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    comps: dict = {}
    for p in files:
        j = json.load(open(p))
        comps.setdefault(j["composition"], []).extend(j["candidates"])
    if not comps:
        return {}
    ns = {len(c) for c in comps.values()}
    assert len(ns) == 1, f"uneven per-composition counts in {d}: {sorted(ns)}"
    return comps


def cell_stats(comps: dict) -> dict | None:
    if not comps:
        return None
    cands = [c for lst in comps.values() for c in lst]
    n = len(cands)
    # A trajectory whose first recorded state is already OOD cannot be
    # "induced": induced_k skips it.  It is counted here so the dilution is
    # visible instead of silent.
    n_start_ood = sum(
        1 for c in cands if c["ood_bitmask"] and (c["ood_bitmask"][0] & OOD_MASK))
    n_clean = n - n_start_ood
    k = induced_k(cands)
    valid = [c["valid"] for c in cands]
    eh = [c["final"]["e_hull"] for c in cands if c["final"]["e_hull"] is not None]
    dm0 = [c["init_d_min"] for c in cands if c.get("init_d_min") is not None]
    dmf = [c["final"]["d_min"] for c in cands]
    return {
        "n": n, "n_comps": len(comps),
        "n_start_ood": n_start_ood,
        "start_ood_share": n_start_ood / n,
        "n_clean": n_clean,
        "induced_k": k,
        "induced_published": k / n,
        "induced_conditional": (k / n_clean) if n_clean else None,
        "valid_share": float(np.mean(valid)),
        "d_min_init_med": float(np.median(dm0)) if dm0 else None,
        "d_min_final_med": float(np.median(dmf)),
        "e_hull_med": float(np.median(eh)) if eh else None,
        "e_hull_stable_share": float(np.mean([e < E_HULL_STABLE for e in eh])) if eh else None,
        "e_hull_floor_share": float(np.mean([e < E_HULL_FLOOR for e in eh])) if eh else None,
    }


def cell_keys(sig: float, arm: str, smp: str) -> set:
    """The (composition, seed) pairs the composition-only arm actually ran."""
    d = os.path.join(BASE, f"c_s{sig:g}_{arm}_{smp}")
    keys = set()
    for p in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(p))
        keys.add((j["composition"], j["seed"]))
    return keys


def ref_cell_stats(sig: float, arm: str, smp: str, restrict: set | None = None) -> dict | None:
    """Same statistics for the reference arm from the frozen scan tree.

    Subset to the (composition, seed) pairs the composition-only arm ran.  The
    frozen cell holds 20 compositions x 5 seeds; a mini pilot arm holds 5 x 2.
    Comparing the two pools whole would compare different chemistry mixes as
    well as different priors -- the identical confound the arm-vs-arm claim
    exists to avoid -- so the reference side is restricted here, as
    eval_componly_metrics.py already does.  The frozen cell's own file count is
    deliberately not asserted: a subset is the point.
    """
    d = os.path.join(mps.BASE, f"s{sig:g}_{mps.DIR_ARM[(arm, smp)]}_{smp}")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        return None
    cands = []
    for p in files:
        j = json.load(open(p))
        if restrict is not None and (j["composition"], j["seed"]) not in restrict:
            continue
        cands.extend(j["candidates"])
    if not cands:
        return None
    n = len(cands)
    n_start_ood = sum(
        1 for c in cands if c["ood_bitmask"] and (c["ood_bitmask"][0] & OOD_MASK))
    k = induced_k(cands)
    return {"n": n, "n_start_ood": n_start_ood, "induced_k": k,
            "induced_published": k / n,
            "induced_conditional": k / (n - n_start_ood) if n > n_start_ood else None}


def main() -> None:
    rows = {}
    print("composition-only prior (c_*) vs reference prior (frozen scan), "
          "same sigma / arm / sampler\n")
    print(f"{'sigma':>5s} {'arm':5s} {'smp':6s} | {'comp-only':>34s} | {'reference':>30s}")
    print(f"{'':5s} {'':5s} {'':6s} | {'n':>5s} {'startOOD':>8s} {'ind.pub':>8s} "
          f"{'ind.cond':>8s} {'valid':>6s} | {'n':>5s} {'startOOD':>8s} {'ind.pub':>8s} "
          f"{'ind.cond':>8s}")
    for sig in SIGMAS:
        for arm in ARMS:
            for smp in SAMP:
                cs = cell_stats(load_cell(sig, arm, smp))
                if cs is None:
                    continue
                # Reference side restricted to the arm's own (comp, seed) keys,
                # so the two columns differ in the prior and nothing else.
                rs = ref_cell_stats(sig, arm, smp, restrict=cell_keys(sig, arm, smp))
                rows[f"s{sig:g}_{arm}_{smp}"] = {"comp_only": cs, "reference": rs,
                                                 "sigma": sig, "arm": arm,
                                                 "sampler": smp}
                rr = (f" | {rs['n']:5d} {rs['n_start_ood']/rs['n']*100:7.1f}% "
                      f"{rs['induced_published']*100:7.2f}% "
                      f"{(rs['induced_conditional'] or 0)*100:7.2f}%") if rs else " | (n/a)"
                print(f"{sig:5g} {arm:5s} {smp:6s} | {cs['n']:5d} "
                      f"{cs['start_ood_share']*100:7.1f}% "
                      f"{cs['induced_published']*100:7.2f}% "
                      f"{(cs['induced_conditional'] or 0)*100:7.2f}% "
                      f"{cs['valid_share']*100:5.1f}%{rr}")

    # The claim: within a sampler, the protected arms sit below bare, and the
    # ordering matches the reference arm's.  Compared on the conditional rate,
    # which is the one that does not mix in the start-quality difference.
    print("\narm ordering within a sampler (conditional induced rate, "
          "composition-only | reference), cells with a complete arm triple:")
    flips = []
    for sig in SIGMAS:
        for smp in SAMP:
            cs = {a: rows.get(f"s{sig:g}_{a}_{smp}", {}).get("comp_only") for a in ARMS}
            if any(v is None for v in cs.values()):
                continue
            def key(a, src):
                d = cs[a] if src == "comp" else rows[f"s{sig:g}_{a}_{smp}"]["reference"]
                v = d["induced_conditional"]
                return v if v is not None else float("nan")
            oc = sorted(ARMS, key=lambda a: key(a, "comp"))
            orf = sorted(ARMS, key=lambda a: key(a, "ref"))
            same = oc == orf
            if not same:
                flips.append((sig, smp, oc, orf))
            print(f"  s{sig:<4g} {smp:6s} comp-only {', '.join(oc):22s} "
                  f"ref {', '.join(orf):22s} {'same' if same else 'DIFFERENT'}")
    if flips:
        print(f"\n  {len(flips)} cell(s) where the composition-only ordering "
              f"differs from the reference arm's: {flips}")
    else:
        print("\n  ordering agrees with the reference arm everywhere.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, _ROOT)}")


if __name__ == "__main__":
    main()
