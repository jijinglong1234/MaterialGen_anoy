#!/usr/bin/env python
"""Diagnostic analysis of the completed subexp2 (doc-12 layer grid) of the
NC=10 cholesky re-run.

Scope: subexp2 only (12 levels x 20 compositions x 5 seeds x 10 candidates =
12000 trajectories, 1200 files), all carrying init_cell_gen=cholesky-fix.
subexp2 is complete -- this is a per-subexperiment read-out, NOT
the official `analyze_phase1.py` run (that one refuses to write while the other
four subexperiments are still in flight, and stays authoritative).

Writes to results/phase1/analysis/diag_subexp2_nc10/:
  table_levels.md / .csv       12-level headline table (paper Table 8 columns)
  table_layer_marginals.md     single-layer contrasts over the doc-12 contexts
  table_vs_prefix.md           pre-fix (LEGACY doc12_table.json) vs this run
  fig_levels_headline.{pdf,png}
  fig_layer_marginals.{pdf,png}
  fig_heatmap.{pdf,png}        level x composition, induced OOD and E_hull
  cells.json                   full per-level + per-(level, comp) metrics

Metric conventions are imported, not re-implemented: analyze_phase1.cell_metrics
(OOD mask 0b010011, steps >= 1, valid-only E_hull) and analyze_doc12
(dwell/deep d_min shares, wall-expansion sanity, Wilson intervals).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ap1 = _load("ap1", REPO / "scripts/analyze_phase1.py")
d12 = _load("d12", REPO / "scripts/analyze_doc12.py")

LEVELS = d12.LEVELS                     # 12 doc-12 subsets, table order
SUBSET = d12.SUBSET
PAIRS = {                               # single-layer contexts (from analyze_doc12)
    1: [("bare", "l1"), ("l2", "l1l2"), ("l3", "l1l3"), ("l2l3l4", "l1l4")],
    2: [("bare", "l2"), ("l1", "l1l2"), ("l1l3", "l1l2l3"), ("l1l3l4", "l1l4")],
    3: [("bare", "l3"), ("l1", "l1l3"), ("l1l2", "l1l2l3"), ("l1l2l4", "l1l4")],
    4: [("bare", "l4"), ("l1l2", "l1l2l4"), ("l1l3", "l1l3l4"), ("l1l2l3", "l1l4")],
}
COMP_ORDER = None                       # filled from the data, sorted by element
PREFIX_TABLE = ap1.ANALYSIS / "doc12_table.json"   # LEGACY (pre-fix) copy
OUTDIR = ap1.ANALYSIS / "diag_subexp2_nc10"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def gather():
    """{level: [cands]}, {level: {comp: [cands]}}, n_files."""
    recs = ap1.load_subexp("subexp2")
    by_lv, by_lvc = {}, {}
    for t in recs:
        lv = t.get("level")
        if lv not in LEVELS:
            continue
        cs = ap1.cands_of(t)
        by_lv.setdefault(lv, []).extend(cs)
        by_lvc.setdefault(lv, {}).setdefault(t.get("composition"), []).extend(cs)
    return by_lv, by_lvc, len(recs)


def metrics(cands):
    m = ap1.cell_metrics(cands)
    m["dwell"], m["deep"] = d12.dwell_deep(cands)
    vs = d12.vol_summary(cands)
    if vs:
        m.update(vs)
    return m


def pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
HEAD = ("| level | layers | traj | induced OOD | start floor | validity | "
        "dwell<1.3 | deep<0.5 | E_hull med | E_hull p90 | rej | NFE med |")


def table_levels(cells: dict, n_files: int) -> str:
    lines = [
        "# sub-exp 2: doc-12 layer grid -- headline table",
        "",
        "Era: **NC=10 cholesky re-run** (post-fix initial cells; "
        f"`init_cell_gen=cholesky-fix`), {n_files}/1200 files.",
        "",
        "Protocol: ALD, sigma_max = 1.0, NFE 200, fixed cell, 20 compositions x "
        "5 seeds x **10 candidates** = 1000 trajectories per level (the pre-fix "
        "era had 20 candidates / 2000).",
        "",
        "Definitions mirror `analyze_phase1.cell_metrics`: induced OOD counts "
        "trajectories whose first OOD step (mask 0b010011) is at step >= 1 "
        "(step 0 = initial-structure floor, reported separately); validity / "
        "dwell / deep are trajectory shares; E_hull is the median over valid "
        "candidates; rej = mean SafetyMonitor rejections per trajectory.",
        "",
        HEAD,
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lv in LEVELS:
        c = cells[lv]
        lines.append(
            f"| {lv} | {{{SUBSET[lv] or '—'}}} | {c['n_traj']} | "
            f"{pct(c['induced_ood_rate'])} | {pct(c['start_ood_rate'])} | "
            f"{pct(c['validity'])} | {pct(c['dwell'])} | {pct(c['deep'])} | "
            f"{num(c['e_hull_med_meV'])} | {num(c['e_hull_p90_meV'], 0)} | "
            f"{num(c['rej_mean'], 2)} | {num(c['nfe_med'], 0)} |")
    lines += ["", "Cell-volume sanity (`analyze_doc12.vol_summary`, fixed-cell "
                  "protocol: only the initial draw may vary):", "",
              "| level | max rel. deviation from comp. median | candidates >1.3x |",
              "|---|---|---|"]
    for lv in LEVELS:
        c = cells[lv]
        lines.append(f"| {lv} | {num(c.get('vol_maxdev'), 3)} | "
                     f"{c.get('vol_n_expanded', '—')} |")
    return "\n".join(lines) + "\n"


def table_csv(cells: dict, path: Path) -> None:
    keys = ["n_traj", "induced_ood_rate", "start_ood_rate", "ood_rate",
            "validity", "dwell", "deep", "e_hull_med_meV", "e_hull_mean_meV",
            "e_hull_p90_meV", "d_min_med", "d_min_med_valid", "rej_mean",
            "nfe_med", "t1_rate", "t2_rate", "force_rate", "nan_rate",
            "collapse_rate", "vol_maxdev", "vol_n_expanded"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "layers"] + keys)
        for lv in LEVELS:
            c = cells[lv]
            w.writerow([lv, SUBSET[lv] or ""] +
                       [c.get(k) if not isinstance(c.get(k), float)
                        else round(c[k], 6) for k in keys])


def table_marginals(cells: dict) -> str:
    lines = [
        "# sub-exp 2: single-layer marginal contrasts (doc-12 contexts)",
        "",
        "Each row adds one layer L to a context that lacks it and compares the "
        "same grid row with L present. Delta induced-OOD in percentage points "
        "(negative = the layer removes OOD), 95% CI on the delta from Wilson "
        "intervals of the two rates. E_hull delta = median(valid) difference in "
        "meV/atom (negative = the layer improves the final energy).",
        "",
        "| layer | context | -> context+L | d induced OOD (pp) | CI | d E_hull (meV) |",
        "|---|---|---|---|---|---|",
    ]
    pooled = {}
    for layer in (1, 2, 3, 4):
        rows, wsum, wxs, dehs = [], 0.0, 0.0, []
        for a, b in PAIRS[layer]:
            ma, mb = cells[a], cells[b]
            if not (ma["n_traj"] and mb["n_traj"]):
                continue
            ka = round(ma["induced_ood_rate"] * ma["n_traj"])
            kb = round(mb["induced_ood_rate"] * mb["n_traj"])
            dp = (mb["induced_ood_rate"] - ma["induced_ood_rate"]) * 100
            lo_a, hi_a = d12.wilson(ka, ma["n_traj"])
            lo_b, hi_b = d12.wilson(kb, mb["n_traj"])
            ci = 100 * math.hypot(hi_a - lo_a, hi_b - lo_b) / 2
            eha, ehb = ma["e_hull_med_meV"], mb["e_hull_med_meV"]
            deh = None if (eha is None or ehb is None) else ehb - eha
            if deh is not None:
                dehs.append(deh)
            lines.append(f"| L{layer} | {{{SUBSET[a] or '—'}}} | "
                         f"{{{SUBSET[b]}}} | {dp:+.2f} | ±{ci:.1f} | "
                         f"{'—' if deh is None else f'{deh:+.1f}'} |")
            rows.append((a, b, dp, ci))
            w = 1 / (ci ** 2) if ci > 0 else 0
            wsum += w
            wxs += w * dp
        if rows:
            pooled[layer] = (wxs / wsum if wsum else float("nan"),
                             float(np.mean(dehs)) if dehs else None,
                             len(dehs))
            p, m, k = pooled[layer]
            lines.append(f"| **L{layer} pooled** | | | **{p:+.2f}** | "
                         f"(inverse-variance over {len(rows)} contexts) | "
                         f"{'—' if m is None else f'{m:+.1f}'} "
                         f"({k} contexts) |")
    return "\n".join(lines) + "\n"


def table_vs_prefix(cells: dict) -> str:
    lines = [
        "# sub-exp 2: pre-fix (LEGACY) vs this run",
        "",
        "Left = `analysis/doc12_table.json` (pre-fix initial cells, 20 candidates, "
        "2000 traj/level; **do not cite**, see "
        "`analysis/LEGACY_pre_cholesky.md`). Right = this re-run "
        "(post-fix cells, 10 candidates, 1000 traj/level). Directional reading "
        "only -- the candidate counts differ.",
        "",
        "| level | induced OOD pre | induced OOD now | validity pre | validity now "
        "| E_hull med pre | E_hull med now |",
        "|---|---|---|---|---|---|---|",
    ]
    pre = json.loads(PREFIX_TABLE.read_text())["cells"] if PREFIX_TABLE.exists() else {}
    for lv in LEVELS:
        c = cells[lv]
        p = pre.get(lv, {})
        lines.append(
            f"| {lv} | {pct(p.get('induced_ood_rate'))} | {pct(c['induced_ood_rate'])} "
            f"| {pct(p.get('validity'))} | {pct(c['validity'])} "
            f"| {num(p.get('e_hull_med_meV'))} | {num(c['e_hull_med_meV'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_levels(plt, cells: dict) -> None:
    x = np.arange(len(LEVELS))
    ind = np.array([cells[lv]["induced_ood_rate"] for lv in LEVELS])
    ci = np.array([(d12.wilson(round(cells[lv]["induced_ood_rate"] * cells[lv]["n_traj"]),
                               cells[lv]["n_traj"])[1] -
                    d12.wilson(round(cells[lv]["induced_ood_rate"] * cells[lv]["n_traj"]),
                               cells[lv]["n_traj"])[0]) / 2 * 100
                   for lv in LEVELS])
    val = np.array([cells[lv]["validity"] for lv in LEVELS]) * 100
    ehm = np.array([cells[lv]["e_hull_med_meV"] or np.nan for lv in LEVELS])
    ehp = np.array([cells[lv]["e_hull_p90_meV"] or np.nan for lv in LEVELS])
    start = np.array([cells[lv]["start_ood_rate"] for lv in LEVELS]) * 100

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.2, 0.8, 1.1], "hspace": 0.18})
    ax = axs[0]
    ax.bar(x, start, color=ap1.GRID, label="start floor (step 0)")
    ax.bar(x, ind * 100, bottom=start, color=ap1.C[0], label="induced (steps >= 1)")
    ax.errorbar(x, start + ind * 100, yerr=ci, fmt="none", ecolor=ap1.INK_S, elinewidth=0.8, capsize=2)
    ax.set_ylabel("trajectory OOD %")
    ax.legend(fontsize=7, ncols=2, loc="upper right")
    ax.set_ylim(0, 105)

    ax = axs[1]
    ax.bar(x, val, color=ap1.C[2])
    ax.set_ylabel("validity %")
    ax.set_ylim(0, 105)

    ax = axs[2]
    ax.plot(x, ehm, "o-", color=ap1.C[1], ms=4, lw=1.3, label="E_hull median")
    ax.plot(x, ehp, "s--", color=ap1.C[3], ms=3, lw=1.0, alpha=0.8, label="E_hull p90")
    ax.set_yscale("log")
    ax.set_ylim(30, 1e6)
    ax.set_ylabel("E_hull (meV/atom, log)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(0.985, 0.06, "L4 cells: median/p90 dominated by\nnear-overlap structures (see README)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=ap1.INK_MUTED)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([f"{lv}\n{{{SUBSET[lv] or '—'}}}" for lv in LEVELS],
                           fontsize=7.5)
    for a in axs:
        a.set_xlim(-0.6, len(LEVELS) - 0.4)
        a.grid(axis="x", visible=False)
    axs[0].set_title("sub-exp 2 layer grid, NC=10 cholesky re-run "
                     "(ALD, sigma_max = 1.0, 1000 traj/level)", loc="left", fontsize=10)
    ap1.save(plt, str(OUTDIR / "fig_levels_headline"))


def fig_marginals(plt, cells: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ys = np.arange(4)[::-1]
    pooled_o, pooled_e, eh_pts = [], [], []
    for layer in (1, 2, 3, 4):
        dps, cis, dehs = [], [], []
        for a, b in PAIRS[layer]:
            ma, mb = cells[a], cells[b]
            ka = round(ma["induced_ood_rate"] * ma["n_traj"])
            kb = round(mb["induced_ood_rate"] * mb["n_traj"])
            dp = (mb["induced_ood_rate"] - ma["induced_ood_rate"]) * 100
            la, ha = d12.wilson(ka, ma["n_traj"])
            lb, hb = d12.wilson(kb, mb["n_traj"])
            dps.append(dp)
            cis.append(100 * math.hypot(ha - la, hb - lb) / 2)
            if ma["e_hull_med_meV"] is not None and mb["e_hull_med_meV"] is not None:
                dehs.append(mb["e_hull_med_meV"] - ma["e_hull_med_meV"])
        jitter = np.linspace(-0.16, 0.16, len(dps))
        axs[0].errorbar(dps, np.full(len(dps), ys[layer - 1]) + jitter, xerr=cis,
                        fmt="o", ms=3.5, lw=0.9, color=ap1.C[layer - 1],
                        alpha=0.75, capsize=2)
        w = [1 / c ** 2 if c > 0 else 0 for c in cis]
        po = float(np.sum(np.array(dps) * w) / np.sum(w)) if np.sum(w) else np.nan
        axs[0].plot([po], [ys[layer - 1]], "D", ms=7, color=ap1.C[layer - 1],
                    markeredgecolor="white", markeredgewidth=0.6)
        pooled_o.append(po)
        if dehs:
            axs[1].plot(dehs, np.full(len(dehs), ys[layer - 1]) + jitter[:len(dehs)],
                        "o", ms=3.5, color=ap1.C[layer - 1], alpha=0.75)
            eh_pts += [(v, ys[layer - 1]) for v in dehs]
            pe = float(np.mean(dehs))
            axs[1].plot([pe], [ys[layer - 1]], "D", ms=7, color=ap1.C[layer - 1],
                        markeredgecolor="white", markeredgewidth=0.6)
            pooled_e.append(pe)
    axs[0].axvline(0, color=ap1.BASELINE, lw=1)
    axs[0].set_xlabel("d induced OOD (pp), adding the layer")
    axs[1].axvline(0, color=ap1.BASELINE, lw=1)
    axs[1].set_xscale("symlog", linthresh=100)
    axs[1].set_xlabel("d E_hull median (meV/atom, symlog)")
    if eh_pts:
        v, y = max(eh_pts) if abs(max(eh_pts)[0]) >= abs(min(eh_pts)[0]) else min(eh_pts)
        axs[1].annotate(f"{v:+.0f}", (v, y), textcoords="offset points",
                        xytext=(-6, 5), ha="right", fontsize=6.5,
                        color=ap1.INK_MUTED)
    for a in axs:
        a.set_yticks(ys)
        a.set_yticklabels([f"L{i}" for i in (1, 2, 3, 4)])
        a.set_ylim(-0.6, 3.6)
    fig.suptitle("single-layer marginals over the doc-12 contexts "
                 "(dots = contexts, diamonds = pooled)", fontsize=9, x=0.01, ha="left")
    ap1.save(plt, str(OUTDIR / "fig_layer_marginals"))


def fig_heatmap(plt, by_lvc: dict) -> None:
    import matplotlib.colors as mcolors

    comps = sorted({c for lv in LEVELS for c in by_lvc.get(lv, {})})
    ood = np.full((len(LEVELS), len(comps)), np.nan)
    eh = np.full((len(LEVELS), len(comps)), np.nan)
    for i, lv in enumerate(LEVELS):
        for j, cp in enumerate(comps):
            cs = by_lvc.get(lv, {}).get(cp)
            if not cs:
                continue
            m = ap1.cell_metrics(cs)
            ood[i, j] = m["induced_ood_rate"]
            if m["e_hull_med_meV"] is not None:
                eh[i, j] = m["e_hull_med_meV"]
    fig, axs = plt.subplots(2, 1, figsize=(9.6, 5.4), sharex=True)
    im0 = axs[0].imshow(ood, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    axs[0].set_title("induced OOD rate (50 traj/cell; white = exactly 0, "
                     "LiF has none at any level)", loc="left", fontsize=9)
    fig.colorbar(im0, ax=axs[0], pad=0.01, fraction=0.02)
    # log colour scale: easy cells (~40 meV) and the L4 near-overlap blow-ups
    # (up to ~5e4 meV) must both be readable in one panel
    lo = max(10.0, float(np.nanmin(eh)))
    im1 = axs[1].imshow(eh, cmap="Blues", aspect="auto",
                        norm=mcolors.LogNorm(vmin=lo, vmax=float(np.nanmax(eh))))
    axs[1].set_title("E_hull median (meV/atom, valid candidates, log scale)",
                     loc="left", fontsize=9)
    fig.colorbar(im1, ax=axs[1], pad=0.01, fraction=0.02)
    for a in axs:
        a.set_yticks(range(len(LEVELS)))
        a.set_yticklabels([f"{lv} {{{SUBSET[lv] or '—'}}}" for lv in LEVELS], fontsize=7)
        a.grid(visible=False)
    axs[1].set_xticks(range(len(comps)))
    axs[1].set_xticklabels(comps, rotation=90, fontsize=7)
    ap1.save(plt, str(OUTDIR / "fig_heatmap"))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    by_lv, by_lvc, n_files = gather()
    cells = {lv: metrics(by_lv.get(lv, [])) for lv in LEVELS}
    comp_cells = {lv: {cp: metrics(cs) for cp, cs in by_lvc.get(lv, {}).items()}
                  for lv in LEVELS}

    (OUTDIR / "table_levels.md").write_text(table_levels(cells, n_files))
    table_csv(cells, OUTDIR / "table_levels.csv")
    (OUTDIR / "table_layer_marginals.md").write_text(table_marginals(cells))
    (OUTDIR / "table_vs_prefix.md").write_text(table_vs_prefix(cells))
    (OUTDIR / "cells.json").write_text(json.dumps(
        {"era": "NC=10 cholesky re-run", "n_files": n_files,
         "n_cand": 10, "levels": cells, "by_composition": comp_cells}, indent=1))

    plt = ap1.setup_mpl()
    fig_levels(plt, cells)
    fig_marginals(plt, cells)
    fig_heatmap(plt, by_lvc)

    print(table_levels(cells, n_files))
    print(table_marginals(cells))
    print(f"\nwrote {OUTDIR}/ (tables md+csv, cells.json, 3 figures)")


if __name__ == "__main__":
    main()
