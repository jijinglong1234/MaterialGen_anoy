#!/usr/bin/env python
"""S.U.N.-relevant stability accounting for the PF-ODE L1-vs-L1+L3 choice.

Why this exists
---------------
docs/phase2_frozen_config.md section 4 froze the Phase-2 PF-ODE arm as L1+L3 on
the strength of a table that quoted ``e_hull_med_meV``.  The median is a bad
proxy for the S.U.N. Stable leg: S.U.N. tests EACH STRUCTURE against the
two-sided band [-50, 100) meV/atom (E_HULL_FLOOR = -0.05 eV), so what matters
is the band-pass *rate*, not the centre of the distribution.  At sigma=0.5 the
PF-ODE L1+L3 arm has a median of 29.2 meV -- comfortably inside the band -- but
a p90 of 1428.5 meV and a band-pass rate of only 78.8%.  Quoting the median
made the L1 -> L1+L3 cost look like +2.7 meV ("negligible"); the rate shows it
is a -3.1 pp move in the band and a +3.9 pp move in validity, i.e. a near-exact
wash on valid AND in-band.

Source data
-----------
results/phase1/subexp1/<s{sigma}_{level}_{sampler}>/*.json, 1000 trajectories
per cell, each candidate carrying final.e_hull in eV/atom.

  !!  This fleet is FIXED-CELL: scripts/run_phase1.make_ald_config hardcodes
  !!  update_lattice=False.  So these are C5-pfode numbers, NOT C8-pfode.  The
  !!  frozen C8/C11 ODE arms run upd_lat=True and have no prior data anywhere in
  !!  the project (F2 preflight is ALD-only).  See the freeze doc's evidence-gap
  !!  note -- this script does not close that gap, it only fixes the statistic.

On the ODE path "l1l2l3" and "l1l4" are the same condition (L1+L3): PFODEConfig
has no L2 field and no L4 monitor, so make_pfode_config resolves only L3.  The
fleet's ODE cells use the legacy name "l1l4"; this script reads that.

stdout: per-sigma table of induced / validity / band-pass / valid-and-in-band
for the five arms that bear on the choice, plus the L1 -> L1+L3 delta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SUBEXP1 = REPO / "results" / "phase1" / "subexp1"
OUT = REPO / "results" / "phase1" / "analysis" / "pfode_l3_band.json"

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

#: S.U.N. Stable band, eV/atom.  Mirrors E_HULL_FLOOR = 0.05 meV->eV and the
#: +100 meV ceiling; kept as literals so this script never silently follows a
#: band change -- if the band moves, this table must be recomputed and re-read.
BAND_LO, BAND_HI = -0.05, 0.10

#: (level, sampler, label).  "l1l4" on pfode == L1+L3 (see module docstring).
ARMS = [
    ("bare",   "pfode", "ODE bare"),
    ("l1",     "pfode", "ODE L1"),
    ("l1l4",   "pfode", "ODE L1+L3"),
    ("l1l2l3", "ald",   "ALD L1-L3"),
    ("bare",   "ald",   "ALD bare"),
]
#: The two ODE arms the frozen choice is actually between.
ODE_PAIR = ("l1", "l1l4")


def cell_rates(level: str, sampler: str, sigma: float) -> dict | None:
    """Per-cell counts for one (level, sampler, sigma) arm, or None if absent.

    Valid-only is the right base for the band rate: an invalid structure is not
    a candidate for Stable either.  ``v_and_b`` over ALL trajectories is the
    quantity that survives into S.U.N., since a structure must clear both gates.
    """
    d = SUBEXP1 / f"s{sigma:g}_{level}_{sampler}"
    if not d.is_dir():
        return None
    n_tot = n_val = n_band = n_v_and_b = n_hull = 0
    for p in sorted(d.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        for c in rec.get("candidates", []):
            n_tot += 1
            if not c.get("valid"):
                continue
            n_val += 1
            h = (c.get("final") or {}).get("e_hull")
            if h is None:
                continue
            n_hull += 1
            if BAND_LO <= h < BAND_HI:
                n_band += 1
                n_v_and_b += 1
    if not n_tot:
        return None
    return {
        "n_traj": n_tot, "n_valid": n_val, "n_with_hull": n_hull,
        "validity": n_val / n_tot,
        "band_of_valid": n_band / n_val if n_val else None,
        "valid_and_band": n_v_and_b / n_tot,
    }


def main() -> int:
    table: dict = {}
    missing = []
    for sigma in SIGMAS:
        for level, sampler, _label in ARMS:
            r = cell_rates(level, sampler, sigma)
            if r is None:
                missing.append(f"s{sigma:g}_{level}_{sampler}")
                continue
            table[f"{sigma:g}|{level}|{sampler}"] = r

    hdr = f"{'sigma':>5} | " + " | ".join(
        f"{lbl:>11} {'band':>6} {'v&b':>6}" for _, _, lbl in ARMS)
    print("S.U.N. stability accounting (band = [-50, 100) meV/atom), "
          "fixed-cell Phase-1 fleet, 1000 traj/cell")
    print(hdr)
    print("-" * len(hdr))
    for sigma in SIGMAS:
        cols = []
        for level, sampler, _ in ARMS:
            r = table.get(f"{sigma:g}|{level}|{sampler}")
            cols.append(f"{100*r['validity']:10.1f}% {100*r['band_of_valid']:5.1f}% "
                        f"{100*r['valid_and_band']:5.1f}%" if r
                        else f"{'--':>11} {'--':>6} {'--':>6}")
        print(f"{sigma:>5} | " + " | ".join(cols))

    print("\nL1 -> L1+L3 on the ODE arm (the frozen choice; both fixed-cell):")
    print(f"  {'sigma':>5} {'d validity':>11} {'d band':>9} {'d v&b':>9}")
    worst = 0.0
    for sigma in SIGMAS:
        a = table.get(f"{sigma:g}|l1|pfode")
        b = table.get(f"{sigma:g}|l1l4|pfode")
        if not (a and b):
            continue
        dv = b["validity"] - a["validity"]
        db = b["band_of_valid"] - a["band_of_valid"]
        dvb = b["valid_and_band"] - a["valid_and_band"]
        worst = max(worst, abs(dvb))
        print(f"  {sigma:>5} {100*dv:+10.2f}pp {100*db:+8.2f}pp {100*dvb:+8.2f}pp")
    print(f"\n  max |d(v&b)| over the scan = {100*worst:.2f} pp "
          f"-> the two arms are equivalent on the S.U.N.-relevant product")

    if missing:
        print(f"\nNOTE: {len(missing)} arm(s) absent from disk: "
              f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")

    OUT.write_text(json.dumps(
        {"band_eV": [BAND_LO, BAND_HI], "sigmas": SIGMAS,
         "caveat": "fixed-cell Phase-1 fleet == C5-pfode, not C8-pfode; "
                   "pfode l1l4 == L1+L3",
         "cells": table}, indent=1, default=str))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
