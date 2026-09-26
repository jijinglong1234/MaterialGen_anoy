"""Sub-exp 1 figures (arms corrected, ALD/PF-ODE split): sigma-failure curves
from the live post-Cholesky tree.

Two outputs, both PDF+PNG, light surface #fcfcfb, ink #0b0b0b:

  results/phase1/analysis/subexp1_sigma_failure.{pdf,png}   2 x 3 panels.
      Top row ALD, bottom row PF-ODE --- the two samplers are never drawn on
      shared axes.  ALD: OOD failure, structure quality, validity.  PF-ODE:
      OOD failure, NFE-cap burn, validity.  Panel letters (a)-(f) feed the
      caption of Fig. fig:sigma-failure.

  results/phase1/analysis/subexp1_layer_grid.{pdf,png}      1 x 3 panels.
      The sigma = 1.0 layer grid (results/phase1/subexp2, ALD, 12 subsets),
      i.e. the figure form of Table tab:layer-dissection.  This is the only
      place the off-ladder subsets {2}, {3}, {1,2}, {1,3} exist: the sigma
      ladder backfilled l1l2l3 alone, and {2,3} was never run at all.  Kept as
      a separate file because it duplicates the table --- include it only if
      the layout wants the visual.

Both are drawn at their printed size (\textwidth = 6.5 in) so the point sizes
below are the point sizes on the page; the earlier 10.5 in canvas rendered at
width=0.8\textwidth put 9 pt labels on the page at ~4.5 pt.

Arm mapping (see the note below): the protected arm is
PROT_SRC[sampler], because the runner token "l1l4" means different layer sets
on the two paths.

Arm correction.  The protected arm is the adopted three-layer set
L1--L3 (docs/phase2_frozen_config.md, F2-a: wall off, alpha=1e-3, l1l2l3), and
on ALD that is the runner's l1l2l3 cell.  The script used to plot the l1l4
cell on BOTH paths --- on ALD that token is the full {1,2,3,4} stack, i.e. it
carries the L4 monitor the paper reports as net-harmful --- which is why
panels (c)/(d) disagreed with Table tab:sigma-failure and with the text of
Sec. 5.2 (validity 90.9% vs the guard-free 100%, median E_hull 968 vs 753
meV/atom at sigma = 5).  There is no l1l2l3 PF-ODE cell (L2 density-adaptive
noise and L4 have no ODE counterpart), so the ODE arm stays l1l4, which
carries L1+L3 on that path and reproduces the table's PF-ODE row exactly.
The ALD l1l4 cell (the full four-layer stack) is not drawn at all: it is the
arm the paper rejects, and on the OOD axis it is indistinguishable from
L1--L3, so it would only hide the adopted curve.  Its E_hull/validity penalty
stays a table/text claim in Sec. 5.2.

Palette: dataviz reference slots 1-3 from analyze_phase1.C (blue/orange/aqua).
Identity is hue = arm, so no series relies on hue alone to be told apart.
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from analyze_subexp1 import load_all, load_doc12, SIGMAS, DOC12, DOC12_SUBSET  # noqa: E402
from analyze_phase1 import fit_sigmoid  # noqa: E402

OUT = os.path.join(_ROOT, "results", "phase1", "analysis")
os.makedirs(OUT, exist_ok=True)
# The paper keeps its own copy of each figure beside its .tex (self-contained
# submission bundle, \graphicspath{{./}}), so mirror the PDFs there on every
# write -- otherwise a re-plot silently leaves the paper rendering the old one.
PAPER_DIR = os.path.join(_ROOT, "docs", "paper")
TEXTWIDTH = 6.5  # inches; the paper's \textwidth, so 1 pt here is 1 pt printed


def save_fig(fig, name):
    """Write to results/ and mirror the PDF into the paper directory."""
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, facecolor="#fcfcfb")
    if name.endswith(".pdf"):
        os.makedirs(PAPER_DIR, exist_ok=True)
        shutil.copy2(path, os.path.join(PAPER_DIR, name))

# --- style (light surface, paper-friendly) ---
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
# dataviz reference slots (analyze_phase1.C): 1 blue, 2 orange, 3 aqua
C_BARE, C_L1, C_PROT = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.labelsize": 7,
    "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.pad": 2,
    "ytick.major.pad": 2, "legend.fontsize": 6.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})
LW, MS = 1.5, 3.6

T = load_all()
SIGMA_FMT = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.5", "2", "3", "5"]
xs = np.array(SIGMAS)

# Paper-facing series, per sampler.  "prot" = the L1--L3 stack; on ALD that is
# the l1l2l3 cell, on PF-ODE the l1l4 cell (L1+L3 by construction there).
# The same three arms on both rows, so one legend covers the whole figure.
# The ALD full-stack cell (l1l4) is deliberately NOT drawn: it is the arm the
# paper rejects, and its OOD curve is indistinguishable from L1--L3's (both
# sit at ~0), so it would only add a line that hides the adopted one.
ALD_ARMS = [("bare", C_BARE, "Bare", "-"),
            ("l1", C_L1, "L1", "-"),
            ("l1l2l3", C_PROT, "L1–L3", "-")]
ODE_ARMS = [("bare", C_BARE, "Bare", "-"),
            ("l1", C_L1, "L1", "-"),
            ("l1l4", C_PROT, "L1–L3", "-")]


def yvals(token, smp, key):
    """Per-sigma series for one cell token, under one sampler.

    key="induced_all" applies the pooled-table normalization (clean-start
    numerator, all-trajectory denominator; the analyze_phase1 convention) ---
    the figure must match Table tab:sigma-failure, whose caption states it.
    """
    m = [T[(s, token, smp)] for s in SIGMAS]
    if key == "induced_all":
        return np.array([x["induced_k"] / max(x["n"], 1) for x in m])
    return np.array([x[key] for x in m])


def plot_row(ax, arms, smp, key, ylabel, title, ymax=None, pct=True):
    """One panel of one row.  No in-axes legend: at print size a 4-entry box
    covers the curves it is explaining, so the row legend is drawn above the
    row by draw_row_legend() instead."""
    for token, col, lab, ls in arms:
        y = yvals(token, smp, key)
        ax.plot(xs, y * 100 if pct else y, lw=LW, color=col, label=lab, ls=ls,
                ms=MS, marker="o", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(SIGMAS)
    # Rotated: ten sigma labels on a 2 in-wide log axis collide when laid flat.
    ax.set_xticklabels(SIGMA_FMT, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel(r"$\sigma$ (initial noise)", labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_title(title, color=INK, pad=3)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.tick_params(axis="both", which="both", length=2)


def draw_legend(fig, axs, arms):
    """One figure-level legend above the top row.

    Both rows carry the same three arms, so a single key covers them; it
    spans the grid rather than sitting centred over the middle panel, and
    clears the panel titles (~0.035 of the figure height at this font size).
    """
    handles = [plt.Line2D([0], [0], color=c, lw=LW, ls=ls, marker="o", ms=MS,
                          label=lab) for _t, c, lab, ls in arms]
    b = axs[0].get_position()
    right = axs[-1].get_position().x1
    fig.legend(handles=handles, loc="lower center", ncol=len(arms),
               frameon=False, handlelength=1.7, columnspacing=1.4,
               bbox_to_anchor=(b.x0, b.y1 + 0.048, right - b.x0, 0.02),
               mode="expand")


fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.0))
fig.patch.set_facecolor("#fcfcfb")
# Fixed margins rather than tight_layout: the legend lives in figure
# coordinates above the top row, so that gap has to be reserved up front.
fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.155,
                    wspace=0.34, hspace=0.62)

# ---- row 1: ALD ----
plot_row(axes[0, 0], ALD_ARMS, "ald", "induced_all",
         "sampler-induced OOD [%]", "(a) ALD: OOD failure curves")
# Fitted collapse scale for bare ALD, computed from the same series panel (a)
# draws rather than read from summary.json.
#
# History of this number, because it has moved twice.  The literal used to be
# hardcoded (0.4489, from the pre-Cholesky summary); it was then read from
# summary.json subexp1.sigmoid["bare|ald|mace"].ood, which is the fit of the
# *total* rate -- a curve that includes the start-OOD floor this panel does not
# draw, and which is fitted by the fixed-amplitude form, so it was never the
# same quantity (0.462 A).  Now fit_sigmoid() fits the induced
# curve too (its `np.all(ys < 0.5)` guard used to reject every curve that
# saturates below half, i.e. every induced curve).  But summary.json is written
# by a full analyze_phase1 pass, and the artifact in the tree predates that
# fix, so its `induced` entry does not exist and this script raised SystemExit
# on every run.
#
# The dependency is dropped rather than refreshed: the fit is taken over
# exactly the array plot_row() draws, so the rule on the panel cannot drift
# from the curve under it even if the artifact is stale again.  Fitting the
# induced curve gives sigma_c = 0.334 A, amplitude 32.1% -- the numbers the
# paper quotes in Sec. 5.2 / Fig. 2(a) / Appendix D.
_SIG_FIT = fit_sigmoid(SIGMAS, yvals("bare", "ald", "induced_all"))
if not _SIG_FIT["fit_ok"] or _SIG_FIT["sigma_c"] is None:
    raise SystemExit(
        "no sigmoid fit for the bare-ALD induced curve -- panel (a) would have "
        f"no rule to draw (got {_SIG_FIT!r}); the induced series is "
        f"{yvals('bare', 'ald', 'induced_all')!r}, which should saturate "
        "near 0.32 with a midpoint near sigma = 0.33 A.")
_SIGMA_C_BARE = _SIG_FIT["sigma_c"]
ax_a = axes[0, 0]
ax_a.axvline(_SIGMA_C_BARE, ls=":", color=INK, lw=1.1, zorder=2)
# Offset to the right of the rule: centred on it, the dotted line runs through
# the label's own text.
ax_a.annotate(rf"$\sigma_c$ = {_SIGMA_C_BARE:.3f} $\AA$",
              xy=(_SIGMA_C_BARE * 1.06, 0.72), xycoords=("data", "axes fraction"),
              ha="left", va="center", fontsize=6.5, color=INK)
# pct=False: plot_row otherwise scales by 100, which mislabels an eV/atom
# quantity as if it were a percentage
plot_row(axes[0, 1], ALD_ARMS, "ald", "e_hull",
         "median $E_{hull}$ [eV/atom]", "(b) ALD: structure quality", pct=False)
plot_row(axes[0, 2], ALD_ARMS, "ald", "validity",
         "valid final structures [%]", "(c) ALD: validity")

# ---- row 2: PF-ODE ----
plot_row(axes[1, 0], ODE_ARMS, "pfode", "induced_all",
         "sampler-induced OOD [%]", "(d) PF-ODE: OOD failure curves")
plot_row(axes[1, 1], ODE_ARMS, "pfode", "cap",
         "at NFE cap [%]", "(e) PF-ODE: NFE-cap burn", ymax=105)
plot_row(axes[1, 2], ODE_ARMS, "pfode", "validity",
         "valid final structures [%]", "(f) PF-ODE: validity")

# Row labels: the sampler is the row's identity, so it is stated once per row
# rather than repeated in every legend and title.
for r, txt in enumerate(("ALD", "PF-ODE")):
    b = axes[r, 0].get_position()
    fig.text(0.018, (b.y0 + b.y1) / 2, txt, rotation=90, va="center",
             ha="center", fontsize=8.5, color=INK, weight="bold")
draw_legend(fig, axes[0], ALD_ARMS)
for p in ("subexp1_sigma_failure.pdf", "subexp1_sigma_failure.png"):
    save_fig(fig, p)
plt.close(fig)
print("saved:", os.path.join(OUT, "subexp1_sigma_failure.{pdf,png}"))

# ---------------------------------------------------------------------------
# Layer grid at sigma = 1.0 (12 subsets, ALD) --- the off-ladder subsets
# {2}, {3}, {1,2}, {1,3} have no noise scan; this is where they live.
# ---------------------------------------------------------------------------
G = load_doc12()
# Rank by induced rate: the ordering itself carries the result, since every
# low-OOD subset is one that contains L3 and every high-OOD one does not.
order = sorted(DOC12, key=lambda c: (G[c]["induced"], c))
LAB = {c: ("$\\emptyset$" if not DOC12_SUBSET[c] else
           "$\\{" + ",".join(str(i) for i in sorted(DOC12_SUBSET[c])) + "\\}$")
       + (" †" if 4 in DOC12_SUBSET[c] else "") for c in DOC12}
ys = np.arange(len(order))[::-1]

fg, gax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.7), sharey=True)
fg.patch.set_facecolor("#fcfcfb")
IND_COL = {"L3": C_PROT, "no L3": C_BARE}
# Scales and limits first: the value labels below place themselves relative to
# the bar end, which is a data-coordinate offset on a linear axis and a
# multiplicative one on the log axis of panel (c).
for ax, xlabel, title, scale in (
        (gax[0], "sampler-induced OOD [%]", "(a) OOD failure", "linear"),
        (gax[1], "valid final structures [%]", "(b) validity", "linear"),
        (gax[2], "median $E_{hull}$ [meV/atom]", "(c) structure quality", "log")):
    ax.set_xscale(scale)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_title(title, color=INK, pad=3)
    ax.tick_params(axis="both", which="both", length=2)
for c in order:
    col = IND_COL["L3" if 3 in DOC12_SUBSET[c] else "no L3"]
    y = ys[order.index(c)]
    for ax, v, fmt, pad in ((gax[0], G[c]["induced"] * 100, "{:.2f}", 0.35),
                            (gax[1], G[c]["validity"] * 100, "{:.1f}", 1.0),
                            (gax[2], (G[c]["e_hull"] or 1e-3) * 1000, "{:.0f}", 1.12)):
        ax.barh(y, v, color=col, height=0.66)
        # Value at the bar end.  In the OOD panel the {1,2,3} and {1,2,3,4}
        # bars are exactly zero-length and would otherwise read as missing
        # data, which is the whole point of the panel.
        tx = v * pad if ax.get_xscale() == "log" else v + pad
        ax.text(tx, y, fmt.format(v), va="center", ha="left", fontsize=5.5,
                color=MUTED)
gax[1].set_xlim(0, 118)
gax[2].set_xlim(1, 2e5)
gax[0].set_yticks(ys)
gax[0].set_yticklabels([LAB[c] for c in order], fontsize=6.5)
gax[0].set_ylim(-0.7, len(order) - 0.3)
for ax in gax:
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.5, zorder=0)
fg.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=IND_COL["L3"],
                                 label="contains L3 (reflecting constraint)"),
                   plt.Rectangle((0, 0), 1, 1, color=IND_COL["no L3"],
                                 label="no L3")],
          loc="lower center", ncol=2, frameon=False, fontsize=6.5,
          bbox_to_anchor=(0.5, -0.02),
          title="† contains L4 (divergence guard); subsets are the runner's layer sets",
          title_fontsize=6.5)
fg.tight_layout(rect=[0, 0.10, 1, 1])
for p in ("subexp1_layer_grid.pdf", "subexp1_layer_grid.png"):
    save_fig(fg, p)
plt.close(fg)
print("saved:", os.path.join(OUT, "subexp1_layer_grid.{pdf,png}"))
