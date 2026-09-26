"""Phase 2 figures: the frozen generation-quality benchmark.

The Phase-2 grid (2700 tasks, 27,000 trajectories, eSEN) is the sampler half of
the generation-quality protocol of Appendix G: six configurations on three
datasets, ten candidates per test composition per seed, pooled over five seeds
to 50 per cell, nine configuration--sampler cells per dataset.  This script
draws the executed half.  The baseline half was never run and is not drawn.

Three outputs, all PDF+PNG, light surface #fcfcfb, ink #0b0b0b:

  results/phase2/analysis/phase2_mp20_quality.{pdf,png}      2 x 3 panels.
      MP-20 only (the dataset the main text reports; Perov-5 and Carbon-24
      live in the appendix figure below).  Panels (a)-(f) feed the caption of
      Fig. fig:phase2-mp20: validity, stability, S.U.N., E_hull, match rate and
      coverage, AMSD.  Covers the protocol's own metric list.

  results/phase2/analysis/phase2_appendix_datasets.{pdf,png}  2 x 3 panels.
      Row 1 Perov-5, row 2 Carbon-24, on the three quality axes.  Read with
      the two qualifications carried in the appendix text: Carbon-24's S.U.N.
      is 0 by criterion degeneracy (not sampler failure), and its three
      PF-ODE arms are budget-truncated.

  results/phase2/analysis/phase2_carbon_diagnostics.{pdf,png}  1 x 3 panels.
      The qualifications themselves: (a) the S.U.N. funnel with P(Novel|Stable)
      over each dataset, (b) the PF-ODE evaluation cap, (c) the unphysical
      E_hull share, which is Perov-5-only and drawn for Perov-5 alone.

Both are drawn at their printed size (\textwidth = 6.5 in) so the point sizes
below are the point sizes on the page.

Colour encodes the defense state, reusing the paper's existing arm semantics
from plot_subexp1.py -- bare / L1 / L1-L3 -- so the same three hues mean the
same three arms across both figures.  Where one panel carries two metrics they
are separated by hatching, not by a second hue, so no series relies on hue
alone.  Error bars are the spread over the 20 cells (not over trajectories).

Data.  Aggregates and per-cell values come from
results/phase2/summary/phase2_<dataset>_esen.json.  The S.U.N. funnel in panel
(a) of the diagnostics figure is NOT in that file: the stored counts mix
denominators (n_sun additionally requires uniqueness, so n_stable and n_novel
cannot reconstruct the 2x2).  It is recomputed here from the per-task payloads
through the same run_phase2 pooling path the canonical analysis uses, so the
two agree by construction rather than by transcription.
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
import run_phase2 as rp  # noqa: E402

OUT = os.path.join(_ROOT, "results", "phase2", "analysis")
os.makedirs(OUT, exist_ok=True)
# The paper keeps its own copy of each figure beside its .tex (self-contained
# submission bundle, \graphicspath{{./}}), so mirror the PDFs there on every
# write -- otherwise a re-plot silently leaves the paper rendering the old one.
PAPER_DIR = os.path.join(_ROOT, "docs", "paper")
SUMMARY = os.path.join(_ROOT, "results", "phase2", "summary")


def save_fig(fig, name):
    """Write to results/ and mirror the PDF into the paper directory."""
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    if name.endswith(".pdf"):
        os.makedirs(PAPER_DIR, exist_ok=True)
        shutil.copy2(path, os.path.join(PAPER_DIR, name))
TEXTWIDTH = 6.5  # inches; the paper's \textwidth, so 1 pt here is 1 pt printed

# --- style (light surface, paper-friendly; same block as plot_subexp1.py) ---
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
SURFACE = "#fcfcfb"

DATASETS = ("mp_20", "perov_5", "carbon_24")
DS_LABEL = {"mp_20": "MP-20", "perov_5": "Perov-5", "carbon_24": "Carbon-24"}

#: One line per dataset for the appendix stability panels, where the axis is
#: scaled to the data and the numbers are therefore small.  Carbon-24's is
#: load-bearing: its S.U.N. is exactly zero in all nine cells and must not be
#: read as a sampler failure (see the funnel panel and Appendix G).
#: Both are short on purpose: the panels are 1.9 in wide and these sit inside
#: them, so a full sentence runs under the neighbouring panel's y-axis label.
STAB_NOTE = {
    "perov_5": "axis scaled to data (max 31%)",
    "carbon_24": "S.U.N. = 0 in all nine arms\n(criterion degeneracy; App. G)",
}

#: The nine cells, in draw order.  ALD runs all six configurations; PF-ODE runs
#: only the three that carry the exploration--safety axis (Appendix G).
ALD_CFGS = ("C1", "C3", "C5", "C8", "C9", "C11")
PFODE_CFGS = ("C5", "C8", "C11")
ARMS = [(c, "ald") for c in ALD_CFGS] + [(c, "pfode") for c in PFODE_CFGS]
#: Slot index where the PF-ODE block starts -- the separator is drawn here.
SPLIT = len(ALD_CFGS)

#: Defense state of each cell, which is what the hue encodes.  Keyed off the
#: layer set the run itself recorded, so a re-frozen grid cannot silently
#: recolour the bars -- an unknown layer set fails loudly in state_of().
STATE_BY_LAYERS = {frozenset(): "bare", frozenset({1}): "L1",
                   frozenset({1, 2, 3}): "L1-L3"}
STATE_COLOR = {"bare": C_BARE, "L1": C_L1, "L1-L3": C_PROT}


def load(dataset):
    """One dataset's canonical summary, with a loud failure if it is absent."""
    path = os.path.join(SUMMARY, f"phase2_{dataset}_esen.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing -- run `python scripts/run_phase2.py --analyze "
            f"--nnp esen` first; the figures read only its output.")
    return json.load(open(path))


def agg(doc, cfg, sampler):
    """The aggregate block of one cell, keyed as run_phase2.analyze writes it."""
    key = f"{cfg}_{sampler}"
    if key not in doc["configs"]:
        raise SystemExit(
            f"cell {key} absent from {doc['dataset']}/esen -- the frozen grid is "
            f"expected to carry all nine; re-run `--analyze`.")
    return doc["configs"][key]


def state_of(block):
    """Defense state of a cell, checked against the layers the run recorded."""
    layers = frozenset(block["config_spec"]["layers_subset"])
    if layers not in STATE_BY_LAYERS:
        raise SystemExit(
            f"unexpected layer set {sorted(layers)} in {block['config_spec']['name']}; "
            f"the figures key the hues on bare / L1 / L1-L3 and cannot colour it.")
    return STATE_BY_LAYERS[layers]


def mstat(block, metric):
    """(mean, std-over-cells) of one aggregate metric, as fractions."""
    a = block["aggregate"].get(metric)
    if not a or a.get("mean") is None:
        raise SystemExit(f"aggregate metric {metric!r} missing/empty in "
                         f"{block['config_spec']['name']}")
    return a["mean"], (a["std"] or 0.0)


def dyn(block, metric):
    """One dynamics metric of a cell, averaged over its cells.

    ``capped_rate`` and ``nfe_mean`` are NOT top-level aggregate metrics: the
    per-cell values live under ``per_cell[cell]["dynamics"]`` and the summary
    collapses them to a plain float under ``aggregate["dynamics"]``.  Reading
    them off ``aggregate`` directly raises KeyError; reading them with ``.get``
    would silently draw zeros, which is the failure mode this helper exists to
    prevent.
    """
    d = block["aggregate"].get("dynamics") or {}
    if metric not in d:
        raise SystemExit(f"aggregate.dynamics[{metric!r}] missing in "
                         f"{block['config_spec']['name']}")
    return float(d[metric])


def frac_unphysical(block):
    """Share of the cell's valid candidates whose E_hull is below the floor.

    Also a nested value (``aggregate["e_hull_valid"]["frac_unphysical"]``), not
    a top-level aggregate metric --- an earlier `.get(...) or 0.0` here drew the
    Perov-5 panel as a flat zero.
    """
    v = block["aggregate"].get("e_hull_valid") or {}
    if v.get("frac_unphysical") is None:
        raise SystemExit(f"e_hull_valid.frac_unphysical missing in "
                         f"{block['config_spec']['name']}")
    return float(v["frac_unphysical"])


def e_hull_med(block):
    """(mean, std, n_cells) of the cell's median PHYSICAL E_hull, meV/atom.

    Cells whose E_hull population is partly unphysical report median_physical;
    averaging the raw median instead would mix NNP extrapolations into the
    structure-quality axis (see E_HULL_FLOOR).

    ``n_cells`` is returned and reported on the panel because it is NOT always
    20.  A cell that is unphysical throughout has no physical median at all,
    and on Perov-5 several cells are in exactly that state under every arm --
    so the panel's mean is over the surviving cells and has to say so rather
    than look like a 20-cell average.
    """
    # median_physical, matching the CSV's e_hull_med_phys_meV column.  Inside
    # this block the two keys are equal by construction (the block is already
    # the physical subsample), but reading the named one keeps the figure and
    # the table from drifting apart if that ever stops being true.
    vals = [pc["e_hull_valid_physical"].get("median_physical")
            for pc in block["per_cell"].values()]
    vals = [v for v in vals if v is not None]
    if not vals:
        raise SystemExit(f"no physical E_hull medians in "
                         f"{block['config_spec']['name']}")
    return (float(np.mean(vals)) * 1000.0, float(np.std(vals)) * 1000.0,
            len(vals), len(block["per_cell"]))


def funnel(dataset):
    """The stable/novel 2x2 over all nine cells, pooled over cells.

    Recomputed from the per-task payloads because the stored counts cannot
    reconstruct it (see module docstring).  Cells are pooled rather than
    averaged so the contingency is a count of candidates, not of rates.
    """
    counts = {"S&N": 0, "S&~N": 0, "~S&N": 0, "~S&~N": 0}
    for cfg, sampler in ARMS:
        tasks = rp._load_tasks(dataset, cfg, sampler)
        if not tasks:
            raise SystemExit(f"no task files for {dataset}/{cfg}/{sampler}")
        by_cell = {}
        for t in tasks:
            by_cell.setdefault(t["cell"]["id"], []).append(t)
        for ts in by_cell.values():
            pool = rp._pooled_records(
                [c for t in ts for c in rp.ap.cands_of(t)],
                [r for t in ts for r in t["records"]],
                ts[0]["cell"]["formula"])
            for rec in pool:
                s, n = bool(rec.stable), bool(rec.novel)
                counts["S&N" if s and n else "S&~N" if s else
                       "~S&N" if n else "~S&~N"] += 1
    return counts


def arm_axis(ax, ymax=None):
    """Shared x-axis: nine cells, ALD block then PF-ODE block, with a divider."""
    xs = np.arange(len(ARMS))
    ax.set_xticks(xs)
    ax.set_xticklabels([c for c, _ in ARMS], fontsize=6.5)
    ax.axvline(SPLIT - 0.5, color="#c3c2b7", lw=0.9, ls="-", zorder=1)
    # Block labels ride under the tick labels, in axes coordinates so they do
    # not collide with the tick text at this panel height.
    ax.annotate("ALD", xy=((SPLIT - 1) / 2 / (len(ARMS) - 1), -0.155),
                xycoords="axes fraction", ha="center", va="top", fontsize=6.5,
                color=INK, annotation_clip=False)
    ax.annotate("PF-ODE", xy=((len(ARMS) - 1 + SPLIT) / 2 / (len(ARMS) - 1), -0.155),
                xycoords="axes fraction", ha="center", va="top", fontsize=6.5,
                color=INK, annotation_clip=False)
    ax.set_xlim(-0.65, len(ARMS) - 0.35)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.tick_params(axis="both", which="both", length=2)


def draw_metric(ax, doc, metric, title, ylabel, ymax=None, scale=100.0):
    """One bar panel: nine cells, hue = defense state, error bar = cell spread.

    ``scale`` converts the stored metric to the axis unit: 100 for the rate
    metrics, which are stored as fractions, and 1 for AMSD, which is stored in
    Angstrom.  Applying the rate conversion to AMSD drew panel (f) 100x too
    tall -- 0.6-1.8 A rendered as 60-180 A against a ylabel reading ``AMSD
    [\\AA]`` (a reviewer read the inflated axis as a real
    diversity finding).
    """
    m, s = zip(*[mstat(agg(doc, c, smp), metric) for c, smp in ARMS])
    for i, (c, smp) in enumerate(ARMS):
        ax.bar(i, m[i] * scale, color=STATE_COLOR[state_of(agg(doc, c, smp))],
               width=0.72, zorder=3)
        ax.errorbar(i, m[i] * scale, yerr=s[i] * scale, fmt="none", ecolor=MUTED,
                    elinewidth=0.7, capsize=1.4, capthick=0.7, zorder=4)
    arm_axis(ax, ymax)
    ax.set_title(title, color=INK, pad=3)
    ax.set_ylabel(ylabel, labelpad=1)


def draw_e_hull(ax, doc, title):
    """Median physical E_hull, log axis: the axis spans 29--613 meV/atom.

    A cell with no physical median drops out of the mean, so when any arm
    loses cells the panel prints the retained range -- a 12-of-20 mean must
    not be read as a 20-of-20 one.
    """
    ms = [e_hull_med(agg(doc, c, smp)) for c, smp in ARMS]
    m, s = [v[0] for v in ms], [v[1] for v in ms]
    n_min, n_tot = min(v[2] for v in ms), ms[0][3]
    ax.set_yscale("log")
    for i, (c, smp) in enumerate(ARMS):
        ax.bar(i, max(m[i], 1.0), color=STATE_COLOR[state_of(agg(doc, c, smp))],
               width=0.72, zorder=3)
        ax.errorbar(i, m[i], yerr=[[min(s[i], m[i] - 1.0)], [s[i]]], fmt="none",
                    ecolor=MUTED, elinewidth=0.7, capsize=1.4, capthick=0.7,
                    zorder=4)
    arm_axis(ax)
    ax.set_ylim(1, 3000)
    ax.set_title(title, color=INK, pad=3)
    ax.set_ylabel(r"median $E_{\mathrm{hull}}$ [meV/atom]", labelpad=1)
    if n_min < n_tot:
        ax.annotate(f"{n_min}-{n_tot} cells", xy=(0.97, 0.94),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=5.8, color=INK)


def draw_pair(ax, doc, m1, m2, title, ylabel, ymax=None, note=None):
    """Two metrics in one panel, separated by hatch (never by a second hue).

    ``ymax=None`` scales the axis to the data.  On the appendix datasets the
    fixed 0--100 axis is wrong: Perov-5's stable rate peaks at 31% and
    Carbon-24's at 2%, so a 100-wide axis renders both as a flat line at zero
    and hides the difference the panel exists to show.  The main-text MP-20
    panel keeps the shared 0--100 so its bars stay comparable with figure 1.
    """
    # tuple(), not zip(): the scaling pass below consumes each series, so a
    # lazy zip would leave the draw loop with nothing left to iterate.
    series = [tuple(zip(*[mstat(agg(doc, c, smp), metric) for c, smp in ARMS]))
              for metric in (m1, m2)]
    if ymax is None:
        top = max(max(m) * 100 + max(s) * 100 for m, s in series)
        ymax = float(np.ceil(top * 1.15 / 5.0) * 5.0)
    for (m, s), hatched in zip(series, (False, True)):
        for i, (c, smp) in enumerate(ARMS):
            ax.bar(i + (0.19 if hatched else -0.19), m[i] * 100,
                   color=STATE_COLOR[state_of(agg(doc, c, smp))],
                   width=0.36, zorder=3,
                   hatch="////" if hatched else None,
                   edgecolor=SURFACE if hatched else "none", linewidth=0.0)
            ax.errorbar(i + (0.19 if hatched else -0.19), m[i] * 100,
                        yerr=s[i] * 100, fmt="none", ecolor=MUTED,
                        elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=4)
    arm_axis(ax, ymax)
    ax.set_title(title, color=INK, pad=3)
    ax.set_ylabel(ylabel, labelpad=1)
    if note is not None:
        # Opaque background: on these panels the error bars of the tallest arm
        # reach the top of the axes, and bare text over them is unreadable.
        ax.annotate(note, xy=(0.5, 0.965), xycoords="axes fraction",
                    ha="center", va="top", fontsize=5.4, color=INK,
                    linespacing=1.3,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))


def state_legend(fig, axs, extra=None):
    """Figure-level key: the three defense states, plus the hatch if used."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLOR[k],
                             label={"bare": "bare (no protection)",
                                    "L1": "L1 (analytic wall)",
                                    "L1-L3": "L1-L3 (full stack)"}[k])
               for k in ("bare", "L1", "L1-L3")]
    if extra:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=MUTED,
                                     hatch="////", edgecolor=SURFACE,
                                     label=extra))
    b = axs.flat[0].get_position()
    right = axs.flat[len(axs.flat) - 1].get_position().x1
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, handlelength=1.7, columnspacing=1.4,
               bbox_to_anchor=(b.x0, b.y1 + 0.045, right - b.x0, 0.02),
               mode="expand")


# ---------------------------------------------------------------------------
# Figure 1 -- MP-20 quality (main text)
# ---------------------------------------------------------------------------
mp = load("mp_20")

fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.0))
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.175,
                    wspace=0.34, hspace=0.62)

draw_metric(axes[0, 0], mp, "validity_rate",
            "(a) terminal validity", "valid [%]", ymax=112)
draw_metric(axes[0, 1], mp, "stable_rate",
            "(b) thermodynamic stability", r"$E_{\mathrm{hull}} < 100$ meV/atom [%]",
            ymax=100)
draw_metric(axes[0, 2], mp, "sun_rate",
            "(c) S.U.N. rate", "stable, unique, novel [%]")
draw_e_hull(axes[1, 0], mp, "(d) structure quality")
draw_pair(axes[1, 1], mp, "match_rate", "coverage",
          "(e) reference agreement", "[%]", ymax=100)
draw_metric(axes[1, 2], mp, "amsd",
            "(f) diversity", r"AMSD [\AA]", scale=1.0)

state_legend(fig, axes, extra="hatched: coverage in (e)")
for p in ("phase2_mp20_quality.pdf", "phase2_mp20_quality.png"):
    save_fig(fig, p)
plt.close(fig)
print("saved:", os.path.join(OUT, "phase2_mp20_quality.{pdf,png}"))

# ---------------------------------------------------------------------------
# Figure 2 -- Perov-5 and Carbon-24 quality (appendix)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.2))
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.175,
                    wspace=0.34, hspace=0.70)

for row, dataset in enumerate(("perov_5", "carbon_24")):
    doc = load(dataset)
    ttl = DS_LABEL[dataset]
    draw_metric(axes[row, 0], doc, "validity_rate",
                f"({chr(97 + row * 3)}) {ttl}: terminal validity", "valid [%]",
                ymax=112)
    draw_pair(axes[row, 1], doc, "stable_rate", "sun_rate",
              f"({chr(98 + row * 3)}) {ttl}: stability", "[%]",
              note=STAB_NOTE[dataset])
    draw_e_hull(axes[row, 2], doc, f"({chr(99 + row * 3)}) {ttl}: quality")

state_legend(fig, axes, extra="hatched: S.U.N. in the middle column")
for p in ("phase2_appendix_datasets.pdf", "phase2_appendix_datasets.png"):
    save_fig(fig, p)
plt.close(fig)
print("saved:", os.path.join(OUT, "phase2_appendix_datasets.{pdf,png}"))

# ---------------------------------------------------------------------------
# Figure 3 -- the two qualifications (appendix)
# ---------------------------------------------------------------------------
fig, gax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.1))
fig.patch.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30, wspace=0.34)

# (a) S.U.N. funnel: the 2x2 of stable against novel, per dataset.
FUN = {ds: funnel(ds) for ds in DATASETS}
SEG = [("S&N", "S and N", C_PROT),
       ("S&~N", "S, not N", C_L1),
       ("~S&N", "N, not S", C_BARE),
       ("~S&~N", "neither", MUTED)]
xs = np.arange(len(DATASETS))
bottom = np.zeros(len(DATASETS))
for key, lab, col in SEG:
    vals = np.array([FUN[ds][key] for ds in DATASETS], dtype=float)
    tot = np.array([sum(FUN[ds].values()) for ds in DATASETS], dtype=float)
    gax[0].bar(xs, vals / tot * 100, bottom=bottom / tot * 100, color=col,
               width=0.62, label=lab, zorder=3)
    bottom = bottom + vals
gax[0].set_xticks(xs)
gax[0].set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=6.5)
gax[0].set_xlim(-0.65, len(DATASETS) - 0.35)
# Headroom for the P(N|S) row: at ylim 100 the labels sat outside the axes and
# were clipped away, taking the panel's whole diagnostic with them.
gax[0].set_ylim(0, 118)
gax[0].set_yticks([0, 20, 40, 60, 80, 100])
gax[0].set_ylabel("share of all candidates [%]", labelpad=1)
gax[0].set_title("(a) S.U.N. funnel", color=INK, pad=3)
gax[0].tick_params(axis="both", which="both", length=2)
gax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              frameon=False, fontsize=6, handlelength=1.2, columnspacing=1.0)
# The whole point of the panel: on Carbon-24 the upper segment (stable AND
# novel) is empty in this run because stable and novel came out disjoint here.
# That disjointness is an outcome, not a degeneracy of the metrics: a new
# allotrope inside the hull window would fill both.  The drawn P(Novel|Stable)
# is 0.00 there against 0.88 and 1.00 elsewhere.
# The metric name sits once, centred, and each bar carries only its value --
# three "P(N|S)=x.xx" strings on a 1.9 in axis overran each other.
for i, ds in enumerate(DATASETS):
    p = FUN[ds]["S&N"] / max(FUN[ds]["S&N"] + FUN[ds]["S&~N"], 1)
    gax[0].annotate(f"{p:.2f}", xy=(i, 102), ha="center", va="bottom",
                    fontsize=5.8, color=INK)
gax[0].annotate("P(Novel | Stable)", xy=(0.5, 112), xycoords=("axes fraction",
                                                              "data"),
                ha="center", va="bottom", fontsize=5.8, color=INK)

# (b) the PF-ODE evaluation cap: what fraction never converged.
for i, ds in enumerate(DATASETS):
    doc = load(ds)
    caps = [dyn(agg(doc, c, "pfode"), "capped_rate") * 100
            for c in PFODE_CFGS]
    nfes = [dyn(agg(doc, c, "pfode"), "nfe_mean") for c in PFODE_CFGS]
    off = (np.arange(len(PFODE_CFGS)) - 1) * 0.26
    gax[1].bar(i + off, caps, width=0.24,
               color=[STATE_COLOR[state_of(agg(doc, c, "pfode"))]
                      for c in PFODE_CFGS], zorder=3)
    # One NFE label per dataset, not one per bar: the three arms sit within a
    # few percent of each other on every dataset, so three separate labels
    # overlapped into an unreadable smear.  The per-arm values are in the
    # appendix table; the panel only needs to say whether the arm converged.
    gax[1].annotate(f"NFE {min(nfes):.0f}-{max(nfes):.0f}",
                    xy=(i, max(caps) + 4.0), ha="center", va="bottom",
                    fontsize=5.8, color=MUTED)
gax[1].set_xticks(xs)
gax[1].set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=6.5)
gax[1].set_xlim(-0.65, len(DATASETS) - 0.35)
gax[1].set_ylim(0, 108)
gax[1].set_ylabel("trajectories hitting the cap [%]", labelpad=1)
gax[1].set_title("(b) PF-ODE budget truncation", color=INK, pad=3)
gax[1].tick_params(axis="both", which="both", length=2)

# (c) the unphysical-E_hull share.  This is a Perov-5 problem and not a sampler
#     problem: it is flat across bare and protected arms alike, and it is
#     exactly zero on the other two datasets under all nine arms -- so plotting
#     those two would add eighteen invisible bars and a second, competing hue
#     legend.  They are stated in the annotation instead, and the hue stays the
#     defense state, the same three hues as figures 1 and 2.
pv = load("perov_5")
vals = [frac_unphysical(agg(pv, c, smp)) * 100 for c, smp in ARMS]
gax[2].bar(np.arange(len(ARMS)), vals, width=0.72, zorder=3,
           color=[STATE_COLOR[state_of(agg(pv, c, smp))] for c, smp in ARMS])
arm_axis(gax[2])
gax[2].set_ylim(0, 84)
gax[2].set_ylabel(r"$E_{\mathrm{hull}}$ below the $-0.05$ floor [%]", labelpad=1)
gax[2].set_title("(c) unphysical $E_{\\mathrm{hull}}$ (Perov-5)", color=INK,
                 pad=3)
gax[2].annotate("MP-20 and Carbon-24: 0.0% in all\n"
                "nine arms -- a Perov-5 chemistry\n"
                "effect, not a sampler effect",
                xy=(0.5, 0.96), xycoords="axes fraction", ha="center",
                va="top", fontsize=5.3, color=INK)
# (b) and (c) share the defense-state hue, so they share one key, spanning both
# panels; (a) is a composition and keeps its own legend under itself.
handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLOR[k],
                         label={"bare": "bare", "L1": "L1",
                                "L1-L3": "L1-L3"}[k])
           for k in ("bare", "L1", "L1-L3")]
b1, b2 = gax[1].get_position(), gax[2].get_position()
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           handlelength=1.2, columnspacing=1.4,
           bbox_to_anchor=(b1.x0, 0.055, b2.x1 - b1.x0, 0.02),
           mode="expand")

for p in ("phase2_carbon_diagnostics.pdf", "phase2_carbon_diagnostics.png"):
    save_fig(fig, p)
plt.close(fig)
print("saved:", os.path.join(OUT, "phase2_carbon_diagnostics.{pdf,png}"))

print("funnel:", {ds: FUN[ds] for ds in DATASETS})
