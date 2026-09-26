#!/usr/bin/env python
"""doc-12 layer-grid analysis (rerun, post-4a12ce4 fixed-cell code).

Reads results/phase1/subexp2/<level>/ (12 cells x 100 files = 2000 traj each;
fleet ratchet-era data archived in subexp2_ratchetera/) and emits
results/phase1/analysis/doc12_table.json: the 12-row paper table —
  induced OOD (clean-start, steps>=1, mask 0b010011; denominator = all traj),
  terminal validity, dwell/deep step shares (d_min_hist steps >= 1,
  <1.3 / <0.5 A), E_hull median of valid candidates, rej_mean, nfe_med,
  plus a wall-expansion sanity check: the protocol is fixed-cell, so the only
  cell variation is the per-candidate INITIAL draw (log-Gram prior, sigma_latt
  = 0.1 * sigma_max), giving ~+-20..31% within a composition, but post-4a12ce4 NO cell
  may exceed 1.3x the per-composition median (fleet ratchet-era l1l2l3/l1l4
  had such outliers, up to ~5x; archived in subexp2_ratchetera/).
  When given --fleet-dir, the same check runs on the archived fleet cells as
  a positive/negative control (bare/l1/l1l2: clean; l1l2l3/l1l4: expanded).
stdout: single-layer marginal contrasts available in the doc-12 grid
(nested pairs) with Wilson 95% CI on the induced-rate deltas.
Metric conventions mirrored from analyze_phase1.cell_metrics and
verify_l1prime.cell_metrics (paper Table 8 / H1a' fleet numbers).
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("ap1", REPO / "scripts/analyze_phase1.py")
ap1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap1)

OUT = ap1.OUT
ANALYSIS = ap1.ANALYSIS

# doc-12 levels, table order; legacy name "l1l4" = the FULL stack {1,2,3,4}.
LEVELS = ["bare", "l1", "l2", "l3", "l4", "l1l2", "l1l3",
          "l1l2l3", "l1l2l4", "l1l3l4", "l2l3l4", "l1l4"]
SUBSET = {"bare": "", "l1": "1", "l2": "2", "l3": "3", "l4": "4",
          "l1l2": "12", "l1l3": "13", "l1l2l3": "123", "l1l2l4": "124",
          "l1l3l4": "134", "l2l3l4": "234", "l1l4": "1234"}
DWELL_DMIN, DEEP_DMIN = 1.3, 0.5
OOD_MASK = 0b010011


def wilson(k, n, z=1.96):
    """Wilson score interval half-width-free pair: returns (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    d = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - d) / den), min(1.0, (c + d) / den)


def dwell_deep(cands):
    hists = [c["d_min_hist"][1:] for c in cands if len(c.get("d_min_hist", [])) > 1]
    steps = np.concatenate(hists) if hists else np.array([])
    if not len(steps):
        return None, None
    return float((steps < DWELL_DMIN).mean()), float((steps < DEEP_DMIN).mean())


EXPAND_FACTOR = 1.3  # |det L| > 1.3x per-composition median = wall-expansion signature


def cell_volumes(cands):
    """Per-composition final-cell |det L| arrays (wall-expansion check).

    NOTE: fixed-cell protocol (update_lattice=False): each candidate's cell is
    drawn ONCE from the log-Gram prior at initialization (sigma_latt =
    0.1 * sigma_max in make_initial) and held constant for every ALD step, so
    final |det L| varies across candidates within a composition (~+-20..31%)
    but never during a trajectory -- and the same (composition, seed, cand)
    carries the identical cell in all 12 levels.  The pre-fix ratchet
    signature was REFLECTION cells pushed
    far beyond that band (fleet l1l2l3/l1l4: up to ~5x the composition median;
    see subexp2_ratchetera/).  We therefore report, per composition,
    the natural max rel. deviation and the count of candidates above
    EXPAND_FACTOR x the composition median (must be 0 for all doc-12 cells).
    """
    by_comp = {}
    for c in cands:
        lat = c["final"].get("structure", {}).get("lattice")
        if lat is not None:
            by_comp.setdefault(c["composition"], []).append(
                abs(float(np.linalg.det(np.asarray(lat, float)))))
    return by_comp


def vol_summary(cands):
    """dict(vol_maxdev, vol_n_expanded) over per-composition medians."""
    vols = cell_volumes(cands)
    if not vols:
        return None
    maxdev, n_exp = 0.0, 0
    for comp, vv in vols.items():
        v = np.asarray(vv)
        d = float(np.max(np.abs(v - np.median(v))) / np.median(v))
        maxdev = max(maxdev, d)
        n_exp += int((v > EXPAND_FACTOR * np.median(v)).sum())
    return {"vol_maxdev": maxdev, "vol_n_expanded": n_exp}


def load_cell(level, root="subexp2"):
    """Load one level's candidates from OUT/<root>/<level>/ (root may name the
    archived fleet dir for cross-era checks)."""
    recs = []
    for t in ap1.load_subexp(root):
        if t.get("level") == level:
            recs.append(t)
    if not recs and root != "subexp2":  # fall back to direct file scan
        p = OUT / root / level
        recs = [json.loads(f.read_text()) for f in sorted(p.glob("*.json"))] \
            if p.is_dir() else []
    cands = [c for t in recs for c in ap1.cands_of(t)]
    m = ap1.cell_metrics(cands)
    m["dwell"], m["deep"] = dwell_deep(cands)
    vs = vol_summary(cands)
    if vs is None:
        m["vol_maxdev"], m["vol_n_expanded"] = None, None
    else:
        m.update(vs)
    return cands, m


def per_cell_md(payload):
    """One detailed statistics table per doc-12 cell (12 tables, ALD-only grid)."""
    lines = ["# doc-12 layer grid -- per-cell statistics tables (rerun)",
             "",
             f"Era: {payload['era']}",
             "All cells: ALD, sigma_max = 1.0, NFE 200, 20 compositions x 5 seeds x "
             "20 candidates = 2000 trajectories.",
             "Rates are trajectory shares; E_hull in meV/atom over valid candidates.",
             ""]
    for lv in LEVELS:
        c = payload["cells"].get(lv)
        if not c or not c.get("n_traj"):
            continue
        n = c["n_traj"]
        ind_k = round(c["induced_ood_rate"] * n)
        lines += [
            f"## {lv}  (layers = {{{SUBSET[lv] or 'empty'}}})",
            "",
            "| metric | value |",
            "|---|---|",
            f"| trajectories | {n} |",
            f"| induced OOD (steps >= 1, clean start) | {c['induced_ood_rate']*100:.2f}% "
            f"({ind_k}/{n}) |",
            f"| start-OOD floor (step 0) | {c['start_ood_rate']*100:.1f}% |",
            f"| terminal validity | {c['validity']*100:.1f}% |",
            f"| dwell share (d_min < 1.3 A) | {(c['dwell'] or 0)*100:.1f}% |",
            f"| deep share (d_min < 0.5 A) | {(c['deep'] or 0)*100:.1f}% |",
            f"| E_hull median / mean / p90 | {c['e_hull_med_meV']:.1f} / "
            f"{c['e_hull_mean_meV']:.1f} / {c['e_hull_p90_meV']:.0f} |",
            f"| d_min median (all / valid) | {c['d_min_med']:.2f} / "
            f"{c['d_min_med_valid']:.2f} A |",
            f"| SafetyMonitor rejections (mean) | {c['rej_mean']:.1f} |",
            f"| NFE (median) | {c['nfe_med']:.0f} |",
            f"| failure modes T1 / T2 / force / NaN | {c['t1_rate']*100:.1f}% / "
            f"{c['t2_rate']*100:.1f}% / {c['force_rate']*100:.1f}% / "
            f"{c['nan_rate']*100:.1f}% |",
            f"| cell-volume outliers / maxdev | {c['vol_n_expanded']} / "
            f"{c['vol_maxdev']:.3f} |",
            "",
        ]
    path = ANALYSIS / "doc12_percell_tables.md"
    path.write_text("\n".join(lines))
    return path


def main():
    fleet_dir = None
    if len(sys.argv) > 1:
        fleet_dir = Path(sys.argv[1])
    cells = {lv: load_cell(lv)[1] for lv in LEVELS}
    table = {lv: {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in m.items() if k != "buckets"}
             for lv, m in cells.items()}
    payload = {
        "era": "doc12-rerun (post-4a12ce4 fixed-cell code; supersedes the "
               "fleet subexp2 cells of summary.json archived in "
               "results/phase1/subexp2_ratchetera/)",
        "cells": table,
    }
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "doc12_table.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {per_cell_md(payload)}")

    print("=== wall-expansion sanity (post-fix: 0 expanded candidates expected) ===")
    for lv in LEVELS:
        m = cells[lv]
        flag = "" if m["vol_n_expanded"] == 0 else "  <-- EXPANSION!"
        print(f"  {lv:8s} natural maxdev {m['vol_maxdev']:.3f}  "
              f"n_expanded(>1.3x comp-med) {m['vol_n_expanded']}{flag}")
    if fleet_dir is not None:
        print(f"  --- fleet-era control (archived {fleet_dir.name}/) ---")
        for lv in ["bare", "l1", "l1l2", "l1l2l3", "l1l4"]:
            m = load_cell(lv, root=fleet_dir.name)[1]
            print(f"  {lv:8s} natural maxdev {m['vol_maxdev']:.3f}  "
                  f"n_expanded {m['vol_n_expanded']}"
                  f"{'   <-- expected: polluted era' if m['vol_n_expanded'] else ''}")

    # Single-layer marginal pairs: add layer x to a grid row lacking x and
    # compare to the row with x (both must exist in the doc-12 grid).
    pairs = {
        1: [("bare", "l1"), ("l2", "l1l2"), ("l3", "l1l3"), ("l2l3l4", "l1l4")],
        2: [("bare", "l2"), ("l1", "l1l2"), ("l1l3", "l1l2l3"), ("l1l3l4", "l1l4")],
        3: [("bare", "l3"), ("l1", "l1l3"), ("l1l2", "l1l2l3"), ("l1l2l4", "l1l4")],
        4: [("bare", "l4"), ("l1l2", "l1l2l4"), ("l1l3", "l1l3l4"), ("l1l2l3", "l1l4")],
    }
    print("\n=== single-layer marginals (delta induced-OOD, pp; 95% CI) ===")
    for layer in (1, 2, 3, 4):
        rows = []
        for a, b in pairs[layer]:
            ma, mb = cells[a], cells[b]
            if not (ma["n_traj"] and mb["n_traj"]):
                continue
            ka, kb = round(ma["induced_ood_rate"] * ma["n_traj"]), \
                round(mb["induced_ood_rate"] * mb["n_traj"])
            dp = (mb["induced_ood_rate"] - ma["induced_ood_rate"]) * 100
            lo_a, hi_a = wilson(ka, ma["n_traj"])
            lo_b, hi_b = wilson(kb, mb["n_traj"])
            ci = 100 * math.hypot(hi_a - lo_a, hi_b - lo_b) / 2  # 95% on the delta
            rows.append((a, b, dp, ci))
        for a, b, dp, ci in rows:
            print(f"  +L{layer} on {{{SUBSET[a] or 'bare'}:>4s}} -> "
                  f"{{{SUBSET[b]}:>4s}}: {dp:+6.2f} pp  (CI {ci:.1f})")
        if len(rows) >= 2:
            w = sum(1 / (r[3] ** 2) for r in rows)
            wdp = sum(r[2] / (r[3] ** 2) for r in rows) / w
            print(f"  L{layer} pooled marginal: {wdp:+.2f} pp over {len(rows)} contexts")

    print("\n=== headline table (paper Table 8 / H1a' style) ===")
    print(f"{'level':8s} {'sub':5s} {'ind%':>6s} {'valid%':>7s} {'dwell%':>7s} "
          f"{'deep%':>6s} {'E_hull':>9s} {'rej':>6s} {'nfe':>6s}")
    for lv in LEVELS:
        m = cells[lv]
        eh = "  n/a " if m["e_hull_med_meV"] is None else f"{m['e_hull_med_meV']:8.1f}"
        print(f"{lv:8s} {SUBSET[lv]:5s} {m['induced_ood_rate']*100:6.1f} "
              f"{m['validity']*100:7.1f} {(m['dwell'] or 0)*100:7.1f} "
              f"{(m['deep'] or 0)*100:6.1f} {eh} {m['rej_mean']:6.2f} "
              f"{m['nfe_med']:6.0f}")
    print(f"\nwrote {ANALYSIS / 'doc12_table.json'}")


if __name__ == "__main__":
    main()
