#!/usr/bin/env python
"""F2 preflight verdict: which lattice working point does the frozen stack need?

The F2 round exists to settle one question before Phase 2 launches
(full rationale in ``preflight_phase2.py``, "F2" block): the F round
concluded "lat_vol_max=2.0 + alpha_lat=1e-4 + L3 clamp", but only the L3 clamp
ever reached the code -- ``run_phase2.CONFIGS`` carries ``alpha_lat=1e-3`` (the
ALDConfig default) and never sets ``lattice_vol_max_ratio``, so the runner
inflates with no wall at all.  Either the clamp alone holds the cell (CONFIGS
are already correct) or the config mitigation is load-bearing (CONFIGS must be
amended).  F2 separates those by running the same six cells three ways:

  F2-a  vol wall off, alpha 1e-3, l1l2l3   == what CONFIGS does today
  F2-b  vol wall 2.0,  alpha 1e-4, l1l2l3  == the F-round conclusion
  F2-c  vol wall 2.0,  alpha 1e-4, l1l4    == the L4 contrast on the upd_lat channel

Verdict rule (decided before the run, from the failure the F round was chasing):
  * F2-a holds   -> volP90 ~ 1, latDrift ~ 0, E_hull ~ the fixed-cell baseline
                    => the clamp is sufficient, CONFIGS ship unamended.
  * F2-a inflates -> volP90 >> 1 or E_hull in the 10^3-10^4 meV range
                    => the mitigation is load-bearing, CONFIGS need the amendment.

The historical contamination this is checked against (pre-clamp,
``compare_mace_esen`` MACE side): vol_max p90 = 6.9-10.0x, E_hull 466-10412 meV.

Reads every ``preflight_shard_F2_*_<nnp>.json`` under the preflight dir and
prints one table per NNP plus the cross-NNP verdict.  Shard files are the
merge unit -- the 3-way sharding is an execution detail, so rows are pooled
and keyed by their cell label.

Usage:  python scripts/analyze_f2.py [--nnp mace esen] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO / "results" / "phase2" / "preflight"

#: The verdict thresholds.  ``vol`` and ``drift`` are the two channels the F
#: round's ratchet moved; ``e_hull`` is the collateral the mitigation bought.
VOL_HOLDS = 1.2           # a held cell sits at ~1.00-1.09; the ratchet was 6.9+
DRIFT_HOLDS = 0.05        # lat_drift_med; the ratchet ran orders of magnitude up
EHULL_HOLDS = 1500.0      # meV; the ratchet's MACE side read 466-10412

ARMS = ("F2-a", "F2-b", "F2-c")
ARM_MEANING = {
    "F2-a": "wall off, alpha 1e-3, l1l2l3  (== run_phase2.CONFIGS today)",
    "F2-b": "wall 2.0,  alpha 1e-4, l1l2l3  (== the F-round conclusion)",
    "F2-c": "wall 2.0,  alpha 1e-4, l1l4    (L4 contrast, upd_lat channel)",
}


def load_rows(nnp: str) -> dict:
    """Pool every F2 shard file for one NNP into {cell_label: metrics}.

    MACE shards keep the legacy unsuffixed names (``preflight_phase2.py``
    writes ``_esen`` only for the non-primary backend), so the MACE match is
    "everything not tagged with another NNP".  The ``--shard 0/18`` smoke
    shard is excluded: it re-runs a cell of the real sharding, and pooling it
    would double-count that cell.
    """
    rows: dict = {}
    for f in sorted(glob.glob(str(PREFLIGHT / "preflight_shard_F2_*.json"))):
        name = Path(f).name
        if name.endswith("_18.json") or name.endswith("_18_esen.json"):
            continue                       # smoke shard
        tagged = name[len("preflight_shard_F2_"):-len(".json")]
        is_esen = tagged.endswith("_esen")
        if (nnp == "esen") != is_esen:
            continue
        d = json.loads(Path(f).read_text())
        for label, m in d["rows"].items():
            if not label.startswith(ARMS):
                continue
            rows[label] = m
    return rows


def cell_key(label: str) -> tuple:
    """Sort/display key: (arm, composition, sigma, seed)."""
    p = label.split("|")
    return (p[0], p[3], p[4], p[5])


def arm_stats(rows: dict, arm: str) -> dict:
    ms = [m for k, m in rows.items() if k.startswith(arm + "|")]
    if not ms:
        return {}
    return {
        "n_cells": len(ms),
        "worst_vol_p90": max(m["vol_max_ratio_p90"] for m in ms),
        "worst_lat_drift": max(m["lat_drift_med"] for m in ms),
        "worst_e_hull": max(m["e_hull_med_meV"] or 0.0 for m in ms),
        "min_validity": min(m["validity"] for m in ms),
        "max_induced": max(m["induced_ood_rate"] for m in ms),
        "max_collapse": max(m["collapse_rate"] for m in ms),
    }


def print_table(rows: dict, nnp: str) -> None:
    hdr = (f"{'arm':6s} {'comp':8s} {'sigma':>5s} {'seed':>5s} {'ood':>7s} {'ind':>7s} "
           f"{'valid':>7s} {'coll':>6s} {'E_hull':>8s} {'latDrift':>9s} {'volP90':>8s}")
    print(f"\n=== F2 / {nnp.upper()} ({len(rows)} cells) ===")
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(rows, key=cell_key):
        m = rows[k]
        arm, sig, comp, seed = cell_key(k)
        eh = m["e_hull_med_meV"]
        print(f"{arm:6s} {comp:8s} {sig:>5s} {seed.replace('seed', ''):>5s} "
              f"{m['ood_rate']*100:6.1f}% {m['induced_ood_rate']*100:6.1f}% "
              f"{m['validity']*100:6.1f}% {m['collapse_rate']*100:5.2f}% "
              f"{eh if eh is not None else float('nan'):8.0f} "
              f"{m['lat_drift_med']:9.4f} {m['vol_max_ratio_p90']:8.4f}")

    print(f"\n--- per-arm worst case ({nnp.upper()}) ---")
    for arm in ARMS:
        s = arm_stats(rows, arm)
        if not s:
            continue
        holds = (s["worst_vol_p90"] < VOL_HOLDS and
                 s["worst_lat_drift"] < DRIFT_HOLDS and
                 s["worst_e_hull"] < EHULL_HOLDS)
        print(f"{arm} n={s['n_cells']}  volP90={s['worst_vol_p90']:.4f}  "
              f"drift={s['worst_lat_drift']:.4f}  E_hull={s['worst_e_hull']:.0f} meV  "
              f"valid>={s['min_validity']*100:.1f}%  ind<={s['max_induced']*100:.1f}%  "
              f"-> {'HOLDS' if holds else 'INFLATES'}")
        print(f"        {ARM_MEANING[arm]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnp", nargs="+", default=["mace", "esen"])
    ap.add_argument("--json", default=str(PREFLIGHT / "f2_verdict.json"))
    args = ap.parse_args()

    out = {}
    for nnp in args.nnp:
        rows = load_rows(nnp)
        if not rows:
            print(f"\n=== F2 / {nnp.upper()} : no shard files yet ===")
            continue
        print_table(rows, nnp)
        out[nnp] = {"rows": rows,
                    "arms": {a: arm_stats(rows, a) for a in ARMS if arm_stats(rows, a)}}

    if not out:
        print("\nnothing to analyse")
        return

    # ---- verdict ---------------------------------------------------------
    # The decision turns on F2-a alone: if the no-wall arm holds on every NNP
    # that ran it, the CONFIGS need no amendment.  F2-b/c inform the trade-off
    # but cannot rescue an inflating F2-a (they are the more conservative arms,
    # and shipping them means raising E_hull for a cell that was never at risk).
    print("\n=== VERDICT ===")
    verdict = {}
    for nnp, d in out.items():
        a = d["arms"].get("F2-a")
        if not a:
            continue
        holds = (a["worst_vol_p90"] < VOL_HOLDS and
                 a["worst_lat_drift"] < DRIFT_HOLDS and
                 a["worst_e_hull"] < EHULL_HOLDS)
        verdict[nnp] = "holds" if holds else "inflates"
        print(f"  {nnp.upper():5s} F2-a {verdict[nnp]:8s} "
              f"(volP90 {a['worst_vol_p90']:.3f}, drift {a['worst_lat_drift']:.4f}, "
              f"E_hull {a['worst_e_hull']:.0f} meV)")
    if verdict:
        if all(v == "holds" for v in verdict.values()):
            print("\n  -> The L3 clamp alone holds the cell.  run_phase2.CONFIGS "
                  "ship UNAMENDED\n     (alpha_lat=1e-3, no lattice_vol_max_ratio).")
        else:
            print("\n  -> The no-wall arm inflates on at least one NNP.  CONFIGS "
                  "MUST be amended to\n     lat_vol_max=2.0 + alpha_lat=1e-4 "
                  "before the full Phase 2 run\n     (Config gains a lat_vol_max "
                  "field; make_sampler_p2 applies it).")

    Path(args.json).write_text(json.dumps(
        {"thresholds": {"vol": VOL_HOLDS, "drift": DRIFT_HOLDS,
                        "e_hull": EHULL_HOLDS},
         "verdict": verdict, **out}, indent=1, default=str))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
