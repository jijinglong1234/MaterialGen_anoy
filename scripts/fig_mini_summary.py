#!/usr/bin/env python3
"""Mini-experiment summary figures — fresh subexp1 fleet data.

Figure 1: sigma-failure curves (induced + total OOD) from complete fleet cells
          (sigma<=0.5, n~1800-2000 per cell; partial sigma=0.7 hollow markers).
Figure 2: protection cost-benefit breakdown at sigma=0.5 (induced OOD,
          validity, E_hull median).

Palette: validated categorical instance (same tokens as scripts/analyze_phase1.py).
Outputs: results/phase1/analysis/mini_summary/fig_mini_{ood,breakdown}.{png,pdf}
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "mini_summary")
os.makedirs(OUT, exist_ok=True)

# Validated categorical palette + ink tokens (dataviz reference instance)
C = {"bare": "#e34948", "l1": "#2a78d6", "l1l4": "#1baf7a"}   # bare=status-warning red, L1=blue, L1L4=aqua
INK_P, INK_S, INK_MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LS = {"ald": "-", "pfode": "--"}
LAB = {"bare": "bare", "l1": "L1", "l1l4": "L1–L4"}

# ---- load aggregated fleet data ----
rows = json.load(open("/tmp/subexp1_agg_clean.json"))
# rows: [sigma, protection, sampler, {n, ood, start, induced, t1, t2, force, nan, valid, eh, wt}]
cells = {(r[0], r[1], r[2]): r[3] for r in rows}
SIGMAS = sorted({r[0] for r in rows})
MIN_N = 500  # hollow markers below this (partial cells)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.color": INK_S, "ytick.color": INK_S,
    "legend.frameon": False, "figure.dpi": 150,
})

# ============ Figure 1: sigma-failure curves ============
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True)
for ax, key, title in zip(axes, ["induced", "ood"],
                          ["sampler-induced OOD", "total trajectory OOD"]):
    for prot in ["bare", "l1", "l1l4"]:
        for smp in ["ald", "pfode"]:
            xs, ys, partial = [], [], []
            for s in SIGMAS:
                if (s, prot, smp) not in cells:
                    continue
                a = cells[(s, prot, smp)]
                xs.append(s)
                ys.append(a[key])
                partial.append(a["n"] < MIN_N)
            full = [p for x, y, p in zip(xs, ys, partial) if not p]
            if not full:
                continue
            # complete cells: solid joined line
            fxs = [x for x, p in zip(xs, partial) if not p]
            fys = [y for x, y, p in zip(xs, ys, partial) if not p]
            ax.plot(fxs, fys, LS[smp], color=C[prot], lw=2.0, marker="o",
                    ms=5.5, mfc="white", mew=1.6, zorder=3)
            # partial cells: hollow markers only
            for x, y, p in zip(xs, ys, partial):
                if p:
                    ax.plot([x], [y], ls="none", color=C[prot], marker="o",
                            ms=7, mfc="white", mew=1.8, alpha=0.85, zorder=4)
    ax.set_xlabel("σ (noise scale)")
    ax.set_ylabel("fraction of trajectories")
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    if key == "ood":
        # fitted sigma_c on fresh bare-ALD data (sigmoid collapse model, §6.1)
        ax.axvline(0.417, color=INK_MUTED, lw=1.2, ls=":", zorder=1)
        ax.annotate("σc ≈ 0.42 (bare ALD fit)", xy=(0.417, 0.92),
                    xytext=(0.46, 0.83), fontsize=8.5, color=INK_MUTED,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
axes[0].set_xticks(SIGMAS); axes[0].set_xticklabels([str(s) for s in SIGMAS])
legend = [
    Line2D([0], [0], color=C["bare"], lw=2, marker="o", ms=5.5, mfc="white", label="bare"),
    Line2D([0], [0], color=C["l1"], lw=2, marker="o", ms=5.5, mfc="white", label="L1"),
    Line2D([0], [0], color=C["l1l4"], lw=2, marker="o", ms=5.5, mfc="white", label="L1–L4"),
    Line2D([0], [0], color=INK_S, lw=2, ls="-", label="ALD"),
    Line2D([0], [0], color=INK_S, lw=2, ls="--", label="PF-ODE"),
    Line2D([0], [0], color=INK_S, lw=0, marker="o", ms=7, mfc="white",
           mew=1.8, label="n < 500 (in flight)"),
]
fig.legend(handles=legend, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.16),
           fontsize=8.5, handlelength=2.2, columnspacing=1.4)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_mini_ood.{ext}", bbox_inches="tight")
plt.close(fig)

# ============ Figure 2: sigma=0.5 protection cost-benefit ============
s = 0.5
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
metrics = [("induced", "sampler-induced OOD", "fraction", "%"),
           ("valid", "validity", "fraction", "%"),
           ("eh", "E_hull median", "eV/atom", "")]

for ax, (key, title, ylab, fmt) in zip(axes, metrics):
    xpos, labels, vals, cols, hatches = [], [], [], [], []
    for i, prot in enumerate(["bare", "l1", "l1l4"]):
        for smp in ["ald", "pfode"]:
            a = cells[(s, prot, smp)]
            xpos.append(len(xpos)); labels.append(f"{LAB[prot]}\n{smp}")
            v = a[key] * 100 if key in ("induced", "valid") else a[key] * 1000
            vals.append(v)
            cols.append(C[prot]); hatches.append("" if smp == "ald" else "//")
    bars = ax.bar(xpos, vals, width=0.72, color=cols, edgecolor=INK_P, lw=0.6,
                  hatch=hatches)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.06 * max(vals)),
                f"{v:.0f}{fmt}", ha="center", va="bottom", fontsize=7.8, color=INK_P)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylab)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7); ax.set_axisbelow(True)
    if key == "eh":
        ax.set_ylabel("E_hull (meV/atom)")
axes[0].set_title("sampler-induced OOD @ σ=0.5", fontsize=10)
fig.suptitle("σ = 0.5 · 20 compositions · 5 seeds · 20 candidates (n = 1800/arm)",
             fontsize=9, color=INK_S, y=1.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_mini_s0.5.{ext}", bbox_inches="tight")
plt.close(fig)

print("figures written to", OUT)
