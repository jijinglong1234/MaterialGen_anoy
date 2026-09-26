#!/usr/bin/env python
"""subexp1 l1l4 sigma-curve analysis -- post-fix rerun.

The 20 cells s{sigma}_l1l4_{ald,pfode} of results/phase1/subexp1/ were re-run
on the post-4a12ce4 fixed-cell code (fleet ratchet-era data archived in
results/phase1/subexp1_l1l4_ratchetera/).  This script pools each
cell and reports, per sigma:

  files      non-empty json count (100 = complete; partial cells flagged)
  ind%       sampler-induced OOD rate (clean start, steps >= 1, mask 0b010011,
             denominator = all trajectories; analyze_phase1 convention)
  floor%     start-OOD rate (step 0, reported separately)
  valid%     terminal validity
  dwell/deep share of accepted steps (d_min_hist steps >= 1) with
             d_min < 1.3 / < 0.5 A
  E_hull     median over valid final structures (meV)
  rej/nfe    mean SafetyMonitor rejections / median NFE
  n_exp      candidates with final |det L| > 1.3x per-composition median
             (wall-expansion sanity; post-fix must match the bare baseline
             of <=2/2000 per cell -- fleet l1l4 cells ran 37-753/2000)

Controls printed alongside when present:
  - bare / l1 cells of the same sigma+sampler (clean in both eras), and
  - the archived fleet l1l4 cell (subexp1_l1l4_ratchetera/).

NOTE on PF-ODE naming: the ODE-arm "l1l4" condition carries L1 + L3 only
(no L2 noise injection, no L4 monitor on the ODE path); ALD-arm "l1l4" is the
full stack {1,2,3,4}.  rej/nfe patterns differ accordingly.

Output: results/phase1/analysis/subexp1_l1l4_postfix.json
Usage:  python scripts/analyze_subexp1_l1l4.py [--sigma 1.0] [--complete-only]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("ap1", REPO / "scripts/analyze_phase1.py")
ap1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap1)

spec2 = importlib.util.spec_from_file_location("a12", REPO / "scripts/analyze_doc12.py")
a12 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(a12)

OUT = ap1.OUT
ANALYSIS = ap1.ANALYSIS
FLEET = "subexp1_l1l4_ratchetera"
SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
SAMPLERS = ["ald", "pfode"]


def load_cell(root: str, tag: str):
    """Load one tag dir directly (root = 'subexp1' or the fleet archive name)."""
    p = OUT / root / tag
    if not p.is_dir():
        return None, 0
    files = [f for f in sorted(p.glob("*.json")) if f.stat().st_size > 0]
    cands = []
    for f in files:
        try:
            cands.extend(ap1.cands_of(json.loads(f.read_text())))
        except Exception:
            pass
    return cands, len(files)


def metrics(cands):
    if not cands:
        return None
    m = ap1.cell_metrics(cands)
    m["dwell"], m["deep"] = a12.dwell_deep(cands)
    vs = a12.vol_summary(cands)
    m.update(vs or {})
    return m


def row(name, cands, nfiles):
    m = metrics(cands)
    if m is None:
        return {"cell": name, "files": nfiles, "n_traj": 0}
    return {
        "cell": name, "files": nfiles, "n_traj": m["n_traj"],
        "induced": m["induced_ood_rate"] * 100,
        "floor": m["start_ood_rate"] * 100,
        "validity": m["validity"] * 100,
        "dwell": (m["dwell"] or 0) * 100,
        "deep": (m["deep"] or 0) * 100,
        "e_hull_med_meV": m["e_hull_med_meV"],
        "e_hull_p90_meV": m["e_hull_p90_meV"],
        "rej_mean": m["rej_mean"], "nfe_med": m["nfe_med"],
        "n_exp": m.get("vol_n_expanded"), "vol_maxdev": m.get("vol_maxdev"),
    }


def fmt(r):
    if r.get("n_traj", 0) == 0:
        return f"{r['cell']:16s} {'--':>5s}"
    eh = "  n/a" if r["e_hull_med_meV"] is None else f"{r['e_hull_med_meV']:7.1f}"
    return (f"{r['cell']:16s} {r['files']:4d}f {r['n_traj']:5d} {r['induced']:6.2f} "
            f"{r['floor']:6.1f} {r['validity']:6.1f} {r['dwell']:6.1f} "
            f"{r['deep']:5.1f} {eh} {r['rej_mean']:6.1f} {r['nfe_med']:5.0f} "
            f"{str(r['n_exp']):>4s}")


HEADER = (f"{'cell':16s} {'files':>5s} {'ntraj':>5s} {'ind%':>6s} {'floor%':>6s} "
          f"{'valid%':>6s} {'dwell%':>6s} {'deep%':>5s} {'E_hull':>7s} "
          f"{'rej':>6s} {'nfe':>5s} {'nexp':>4s}")

MD_COLS = ["files", "n_traj", "induced", "floor", "validity", "dwell", "deep",
           "e_hull_med_meV", "e_hull_p90_meV", "rej_mean", "nfe_med",
           "n_exp", "vol_maxdev"]
MD_HEAD = ("| condition | files | ntraj | ind% | floor% | valid% | dwell% | deep% | "
           "E_hull med | E_hull p90 | rej | nfe | nexp | vol maxdev |")
MD_SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def md_row(label, r):
    if not r or not r.get("n_traj"):
        return f"| {label} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |"
    eh = "n/a" if r["e_hull_med_meV"] is None else f"{r['e_hull_med_meV']:.1f}"
    p90 = "n/a" if r.get("e_hull_p90_meV") is None else f"{r['e_hull_p90_meV']:.0f}"
    md = "n/a" if r.get("vol_maxdev") is None else f"{r['vol_maxdev']:.3f}"
    flag = "" if r.get("complete", True) else " (PARTIAL)"
    return (f"| {label}{flag} | {r['files']} | {r['n_traj']} | {r['induced']:.2f} | "
            f"{r['floor']:.1f} | {r['validity']:.1f} | {r['dwell']:.1f} | "
            f"{r['deep']:.1f} | {eh} | {p90} | {r['rej_mean']:.1f} | "
            f"{r['nfe_med']:.0f} | {r['n_exp']} | {md} |")


def md_report():
    """Per-cell tables split by arm (one table per completed sigma cell)."""
    lines = ["# subexp1 l1l4 post-fix rerun -- per-cell tables",
             "",
             "Era: post-4a12ce4 fixed-cell code. Controls bare/l1 are clean in both",
             "eras; `l1l4 fleet` rows come from the ratchet-era archive",
             "`results/phase1/subexp1_l1l4_ratchetera/`.",
             "PF-ODE `l1l4` = L1 + L3 only (no L2/L4 on the ODE path).",
             "E_hull in meV; nexp = final |det L| > 1.3x per-composition median;",
             "maxdev = max rel. deviation of |det L| within a composition.",
             ""]
    todo = []
    for samp, arm in (("ald", "ALD"), ("pfode", "PF-ODE")):
        lines.append(f"## Arm: {arm}")
        lines.append("")
        for sigma in SIGMAS:
            tag = f"s{sigma:g}_l1l4_{samp}"
            r = load_cell("subexp1", tag)
            cands, nfiles = r
            if nfiles == 0:
                todo.append(tag)
                continue
            rr = row(tag, cands, nfiles)
            rr["complete"] = nfiles == 100
            lines.append(f"### sigma = {sigma:g} -- {arm}")
            lines.append("")
            lines.append(MD_HEAD)
            lines.append(MD_SEP)
            for ctrl in ("bare", "l1"):
                c, nf = load_cell("subexp1", f"s{sigma:g}_{ctrl}_{samp}")
                if c:
                    cr = row(f"s{sigma:g}_{ctrl}_{samp}", c, nf)
                    cr["complete"] = nf == 100
                    lines.append(md_row(f"{ctrl} (control)", cr))
            fc, fnf = load_cell(FLEET, tag)
            if fc:
                fr = row(f"fleet {tag}", fc, fnf)
                fr["complete"] = fnf == 100
                lines.append(md_row("l1l4 fleet (archive)", fr))
            lines.append(md_row("**l1l4 post-fix**", rr))
            lines.append("")
    if todo:
        lines.append("## Pending cells (no files yet at report time)")
        lines.append("")
        lines += [f"- {t}" for t in todo]
        lines.append("")
    (ANALYSIS / "subexp1_l1l4_tables.md").write_text("\n".join(lines))
    return ANALYSIS / "subexp1_l1l4_tables.md"


SIGMA_LAYER = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]   # sigma range of the user-facing split
# Ladder names that exist at every sigma in subexp1.  l1l2l3 was added
# for the Phase-2 C8/headline set and backfilled over the whole
# ladder, so it is no longer a sigma=1-only condition.
PER_SIGMA_CONDS = ["bare", "l1", "l1l2l3", "l1l4"]


def _subexp1_ald_row(cond, sigma):
    tag = f"s{sigma:g}_{cond}_ald"
    cands, nf = load_cell("subexp1", tag)
    if not cands:
        return None
    r = row(tag, cands, nf)
    r["complete"] = nf == 100
    return r


def sigma_layer_md():
    """Per-sigma ALD layer tables.

    The 12-condition layer grid (doc-12) was executed at sigma = 1.0 only; the
    planned sigma sub-scan of Sub-exp 2 was never run.  Off sigma = 1 the ALD
    layer evidence is the subexp1 ladder {bare, l1, l1l2l3, full-stack l1l4},
    which now covers every sigma.  This file makes that coverage explicit
    instead of implying a grid that does not exist.

    All rows are post-fix (the Cholesky cell-initialization fix); the
    earlier revision of this file was computed on pre-fix trajectories and its
    numbers are not comparable -- in particular the pre-fix l1l4 columns and
    the "l1l2l3 missing below sigma = 1" coverage note.
    """
    doc12 = json.loads((ANALYSIS / "doc12_table.json").read_text())["cells"]
    lines = ["# ALD layer ablation by sigma -- coverage + per-sigma tables",
             "",
             "Sources, all post-Cholesky-fix and all 10 candidates/seed:",
             "  - sigma = 1.0: 12-condition layer grid = doc-12 (1000 traj/cell, ALD),",
             "  - sigma < 1 and the ladder rows: subexp1 {bare, l1, l1l2l3, l1l4 = full",
             "    stack {1,2,3,4}} (1000 traj/cell, ALD). The planned Sub-exp 2 sigma",
             "    sub-scan was never executed, so the intermediate subsets are still",
             "    sigma = 1.0 only. PF-ODE is excluded by request.",
             "",
             "Reading the ladder: l1l2l3 is the Phase-2 C8 set and dominates the full",
             "stack l1l4 on every column at every sigma (L4 is net-harmful: its",
             "rejection gate tests the pre-move state, so a rejected step cannot move",
             "and cannot induce OOD -- the zero induced rate it appears to buy is an",
             "absorbing state, paid for in validity, dwell and E_hull).",
             "",
             "## Coverage matrix (12 layer conditions x 6 sigma)", "",
             "| layers | " + " | ".join(f"sigma={s:g}" for s in SIGMA_LAYER) + " |",
             "|---" * (len(SIGMA_LAYER) + 1) + "|"]
    for lv in a12.LEVELS:
        cells_mark = []
        for s in SIGMA_LAYER:
            if s == 1.0:
                ok = bool(doc12.get(lv, {}).get("n_traj"))
            else:
                ok = lv in PER_SIGMA_CONDS and _subexp1_ald_row(lv, s) is not None
            cells_mark.append("X" if ok else "-")
        lines.append(f"| {lv} | " + " | ".join(cells_mark) + " |")
    lines.append("")

    for s in SIGMA_LAYER:
        lines += [f"## sigma = {s:g} -- ALD", ""]
        lines += [MD_HEAD, MD_SEP]
        if s == 1.0:
            for lv in a12.LEVELS:
                c = doc12.get(lv)
                if not c or not c.get("n_traj"):
                    continue
                rr = {"files": 100, "n_traj": c["n_traj"],
                      "induced": c["induced_ood_rate"] * 100,
                      "floor": c["start_ood_rate"] * 100,
                      "validity": c["validity"] * 100,
                      "dwell": (c["dwell"] or 0) * 100,
                      "deep": (c["deep"] or 0) * 100,
                      "e_hull_med_meV": c["e_hull_med_meV"],
                      "e_hull_p90_meV": c["e_hull_p90_meV"],
                      "rej_mean": c["rej_mean"], "nfe_med": c["nfe_med"],
                      "n_exp": c["vol_n_expanded"], "vol_maxdev": c["vol_maxdev"]}
                lines.append(md_row(f"{lv} (doc-12)", rr))
            # Cross-checks: the same condition exists in two independently run
            # grids (doc-12 and the subexp1 ladder), so agreement is evidence
            # that neither run is anomalous.
            for xc in ("l1l2l3", "l1l4"):
                r = _subexp1_ald_row(xc, 1.0)
                if r:
                    lines.append(md_row(f"{xc} (subexp1 cross-check)", r))
        else:
            for cond in PER_SIGMA_CONDS:
                rr = _subexp1_ald_row(cond, s)
                lines.append(md_row(cond, rr) if rr else f"| {cond} | missing |")
            missing = [lv for lv in a12.LEVELS if lv not in PER_SIGMA_CONDS]
            lines.append("")
            lines.append("Missing at this sigma (never run): " + ", ".join(missing))
        lines.append("")
    path = ANALYSIS / "ald_layer_by_sigma.md"
    path.write_text("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--complete-only", action="store_true")
    args = ap.parse_args()

    sigmas = [args.sigma] if args.sigma is not None else SIGMAS
    payload = {"era": "subexp1-l1l4-postfix (fixed-cell code, "
                      "allow_lattice_scale=False); fleet archive: "
                      "results/phase1/subexp1_l1l4_ratchetera/",
               "cells": {}}
    for sigma in sigmas:
        for samp in SAMPLERS:
            tag = f"s{sigma:g}_l1l4_{samp}"
            cands, nfiles = load_cell("subexp1", tag)
            if nfiles == 0:
                continue
            payload["cells"][tag] = row(tag, cands, nfiles)
            payload["cells"][tag]["complete"] = nfiles == 100
    if args.sigma is None:
        # controls + fleet comparison, printed only
        for sigma in sigmas:
            for samp in SAMPLERS:
                tag = f"s{sigma:g}_l1l4_{samp}"
                if tag not in payload["cells"]:
                    continue
                print(f"\n=== sigma={sigma:g}  {samp.upper()} ===")
                print(HEADER)
                for ctrl in ("bare", "l1"):
                    c, nf = load_cell("subexp1", f"s{sigma:g}_{ctrl}_{samp}")
                    if c:
                        print(fmt(row(f"s{sigma:g}_{ctrl}_{samp}", c, nf)) +
                              ("   (control)" if True else ""))
                c, nf = load_cell(FLEET, tag)
                if c:
                    print(fmt(row(f"fleet {tag}", c, nf)) + "   (fleet, polluted)")
                r = payload["cells"][tag]
                if r.get("n_traj"):
                    print(fmt({**r, "cell": tag}) +
                          ("" if r["complete"] else "   (PARTIAL)") + "   (post-fix)")

    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "subexp1_l1l4_postfix.json").write_text(json.dumps(payload, indent=1))
    md_path = md_report()
    print(f"\nwrote {md_path}")
    print(f"wrote {sigma_layer_md()}")

    n_complete = sum(1 for v in payload["cells"].values()
                     if v.get("complete"))
    n_partial = sum(1 for v in payload["cells"].values()
                    if v.get("n_traj") and not v.get("complete"))
    print(f"\nwrote {ANALYSIS / 'subexp1_l1l4_postfix.json'}")
    print(f"cells: {n_complete} complete, {n_partial} partial")


if __name__ == "__main__":
    main()
