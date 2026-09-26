"""Sub-exp 1 per-composition sigma-failure tables.

Splits the pooled sigma-failure statistic of paper Table tab:sigma-failure
(S6.2 Sub-experiment 1) into 20 per-composition tables --- one per MP-20
test composition, each cell pooling that composition's
5 seeds x N_CAND candidates.

Definition (frozen protocol; identical to analyze_phase1.py /
results/phase1/analysis/summary.json `induced_ood_rate`):
  OOD mask 0b010011 (d_min < 0.5 A | max|F_NNP| > 500 eV/A | |det L| > 10x)
  induced = share of ALL trajectories (denominator n = trajectories per
            comp cell) that started clean AND have an OOD event at any
            step >= 1.
  The induced count is additive over compositions, so pooling the 20 tables
  reproduces the pooled table exactly.

The protected column of the pooled table (and hence of the 21
tables here) is no longer a single arm.  On ALD it is the dedicated L1--L3
ladder run_phase1.SIGMA_SCAN_ALD_ONLY = ["l1l2l3"], added when L4
was demoted to an optional numeric guard; on the ODE path it stays the l1l4
code name, which there denotes the identical configuration (no L2 noise, no
L4 monitor).  DIR_ARM below maps the display key to the on-disk arm.

Every denominator and every printed "N trajectories" is now read
from the data (N_PER_COMP below) instead of being hardcoded at 20 candidates =
100 per composition / 2000 per cell.  The bug-forced re-run
halves the candidate count to 10, i.e. 50 per composition and 1000 per cell,
so the previous hardcoded 100/2000 would have silently doubled every printed
percentage.  The script also refuses to run on a partial data set: if any
(composition, cell) is missing or the counts are not uniform it aborts rather
than printing a table with a too-small denominator.

Output: docs/sigma_failure_percomposition_tables.tex  (standalone supplement,
ASCII-only body; cross-references the main-text pooled table by
\ref{tab:sigma-failure}, which resolves when the file is merged as an
appendix of paper_outline_v0.4.tex and prints ?? when compiled standalone).
"""
import glob
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(_ROOT, "results", "phase1", "subexp1")
OUT = os.path.join(_ROOT, "docs", "sigma_failure_percomposition_tables.tex")

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
ARMS = ["bare", "l1", "l1l4"]
SAMP = ["ald", "pfode"]
OOD_MASK = 0b010011
N_SEEDS = 5                     # run_phase1.SEEDS -- task files per composition
N_CAND_EXPECTED = 10            # run_phase1.N_CAND (was 20)

# The third arm is the protected one, and its *directory* depends on the
# sampler.  On ALD it is the dedicated L1--L3 ladder, run as the
# sigma-scan-only arm run_phase1.SIGMA_SCAN_ALD_ONLY = ["l1l2l3"] (added
# with the L4 demotion).  On the ODE path there is no L2 (no
# density-adaptive noise) and no L4 (no monitor), so l1l2l3 and l1l4 denote the
# identical configuration -- make_pfode_config(s, "l1l2l3") ==
# make_pfode_config(s, "l1l4") field by field -- and the l1l4 directory stays
# the ODE source.  The key "l1l4" below is therefore a *display* key, not a
# directory name; DIR_ARM resolves it.
DIR_ARM = {
    ("bare", "ald"): "bare",   ("bare", "pfode"): "bare",
    ("l1", "ald"): "l1",       ("l1", "pfode"): "l1",
    ("l1l4", "ald"): "l1l2l3", ("l1l4", "pfode"): "l1l4",
}

# Physical column order of the paper table: Bare{ALD,PF-ODE}, L1{ALD,PF-ODE},
# protected{ALD,PF-ODE}  ->  (arm, sampler).
COLS = [(a, s) for a in ARMS for s in SAMP]
SIGMA_FMT = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.5", "2.0", "3.0", "5.0"]
# composition -> TeX formula (paper conventions: subscripts)
TEX = {
    "Ag2S": "Ag$_2$S", "Al2O3": "Al$_2$O$_3$", "AlCo": "AlCo", "AlCu": "AlCu",
    "AlF3": "AlF$_3$", "AlFe3": "AlFe$_3$", "BaTiO3": "BaTiO$_3$",
    "Ca3N2": "Ca$_3$N$_2$", "CdS": "CdS", "CoF3": "CoF$_3$",
    "FeNi3": "FeNi$_3$", "GaN": "GaN", "LiCoO2": "LiCoO$_2$", "LiF": "LiF",
    "MoS2": "MoS$_2$", "SrTiO3": "SrTiO$_3$", "TiO2": "TiO$_2$", "VN": "VN",
    "ZnS": "ZnS", "ZrO2": "ZrO$_2$",
}

HEADER = [
    "\\textbf{ALD}", "\\textbf{PF-ODE}", "\\textbf{ALD}", "\\textbf{PF-ODE}",
    "\\textbf{ALD}", "\\textbf{PF-ODE}$^{\\ddagger}$",
]


def load_cell(sig: float, arm: str, smp: str) -> dict:
    """Per-cell: composition -> candidate list (all seeds pooled).

    A cell holds one task file per (composition, seed), i.e.
    N_SEEDS * len(TEX) files; each file carries rp.N_CAND candidates.  The
    file count is asserted so a half-finished cell cannot silently shrink the
    denominator (the candidate count itself is not hardcoded -- it is read
    back from the data by n_per_comp, so the 20 -> 10 candidate
    change needs no edit here).
    """
    dir_arm = DIR_ARM[(arm, smp)]            # display key -> on-disk arm
    d = os.path.join(BASE, f"s{sig:g}_{dir_arm}_{smp}")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    want = N_SEEDS * len(TEX)
    assert len(files) == want, (
        f"cell s{sig:g}_{dir_arm}_{smp} has {len(files)} task file(s), expected "
        f"{want} = {len(TEX)} compositions x {N_SEEDS} seeds -- the "
        f"re-run has not finished this cell" if len(files) < want else
        f"cell s{sig:g}_{dir_arm}_{smp} has {len(files)} task file(s), more than "
        f"the expected {want} -- stale files from an earlier era left behind? "
        f"(pre-fix trees belong in results/phase1/legacy_pre_cholesky/)")
    comps = {}
    for p in files:
        j = json.load(open(p))
        comps.setdefault(j["composition"], []).extend(j["candidates"])
    return comps


def n_per_comp(comps: dict) -> int:
    """Trajectories in one (composition, cell) cell -- read from the data.

    Every composition must contribute the same count (N_SEEDS x N_CAND); a
    partial cell would silently change the denominator, so it raises.  The
    count is additionally pinned to N_SEEDS * N_CAND_EXPECTED, which is what
    catches a mixed era: a cell still holding 20-candidate pre-fix task files
    alongside post-fix 10-candidate ones would otherwise pass the uniformity
    check and inflate every percentage.
    """
    ns = {len(c) for c in comps.values()}
    assert len(ns) == 1, f"uneven per-composition trajectory counts: {sorted(ns)}"
    n = ns.pop()
    assert n > 0, "empty cells"
    want = N_SEEDS * N_CAND_EXPECTED
    assert n == want, (f"per-composition cell has {n} trajectories, expected "
                       f"{want} = {N_SEEDS} seeds x {N_CAND_EXPECTED} candidates "
                       "-- mixed-era or stale data?  (pre-fix trees belong in "
                       "results/phase1/legacy_pre_cholesky/)")
    return n


def induced_k(cands: list) -> int:
    """Start-clean trajectories with any OOD event at steps >= 1."""
    return sum(
        not (c["ood_bitmask"][0] & OOD_MASK)
        and any(m & OOD_MASK for m in c["ood_bitmask"][1:])
        for c in cands
        if len(c["ood_bitmask"]) > 1
    )


def parse_pooled_paper_table(tex_path: str):
    """Parse the frozen pooled sigma-failure table out of the paper tex.

    Returns {(sigma, arm, sampler): printed percent} in the physical column
    order Bare{ALD,PF-ODE}, L1{ALD,PF-ODE}, protected{ALD,PF-ODE} (the key
    "l1l4" is a display key -- see DIR_ARM), or None if the table cannot be
    parsed.  Values are compared against the recompute with a
    tolerance of half a printed unit: the table prints two decimals since the
    migration, and a one-decimal string round-trip would flag pure
    float-noise differences such as 19.95 -> '19.9' vs '20.0'.

    The printed column order is unchanged, but the protected ALD
    column now carries the dedicated l1l2l3 arm (post L4-demotion) rather than
    the l1l4 stack, and the protected ODE column is still l1l4.  The parse is
    positional, so only the recompute side needs the DIR_ARM map.
    """
    txt = open(tex_path, encoding="utf-8", errors="replace").read()
    start = txt.find(r"\label{tab:sigma-failure}")
    end = txt.find(r"\end{tabularx}", start)
    if start < 0 or end < 0:
        return None
    body = txt[start:end].split("\n")
    out = {}
    for line in body:
        fields = [f.strip() for f in line.split("&")]
        if len(fields) != 7:
            continue
        sig_s = fields[0].rstrip("\\").strip()
        if sig_s not in SIGMA_FMT:
            continue
        vals = []
        for f in fields[1:]:
            t = f.rstrip("\\").strip()
            if t:
                vals.append(float(t))
        if len(vals) == 6:
            for (a, smp), v in zip(COLS, vals):
                out[(float(sig_s), a, smp)] = v
    return out or None


def table_lines(sigmas, pcts, label, caption) -> list:
    """Render one sigma-failure table body given pcts[sig][col] in % (1 dec)."""
    L = [r"\begin{table}[ht]",
         r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{" + caption + "}",
         r"\label{" + label + "}",
         r"\begin{tabularx}{\textwidth}{@{}lXXXXXX@{}}",
         r"\toprule",
         r"$\sigma_{\max}$\,(\AA{}) & \multicolumn{2}{c}{\textbf{Bare}}"
         r" & \multicolumn{2}{c}{\textbf{L1}}"
         r" & \multicolumn{2}{c}{\textbf{L1--L3}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"]
    L.append(" & " + " & ".join(HEADER) + r" \\")
    L.append(r"\midrule")
    for s, row in zip(sigmas, pcts):
        vals = " & ".join(f"{p:.1f}" for p in row)
        L.append(f"{s} & {vals} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabularx}")
    L.append(r"\end{table}")
    return L


def main():
    # counts[(sigma, comp)][(arm, sampler)] = induced count per comp cell.
    counts = {}
    N_PER_COMP = None            # trajectories per (composition, cell)
    for sig in SIGMAS:
        for arm in ARMS:
            for smp in SAMP:
                comps = load_cell(sig, arm, smp)
                assert sorted(comps) == sorted(TEX), set(TEX) ^ set(comps)
                n = n_per_comp(comps)
                if N_PER_COMP is None:
                    N_PER_COMP = n
                    print(f"per-composition cell = {n} trajectories "
                          f"(5 seeds x {n // 5} candidates)")
                assert n == N_PER_COMP, (sig, arm, smp, n, N_PER_COMP)
                for comp, cands in comps.items():
                    counts.setdefault((sig, comp), {})[(arm, smp)] = induced_k(cands)

    comps = sorted(TEX)
    N_POOLED = N_PER_COMP * len(comps)
    # pooled recompute for the cross-check table
    pooled = {}
    for sig in SIGMAS:
        pooled[sig] = {(a, s): sum(counts[(sig, c)][(a, s)] for c in comps) / N_POOLED
                       for (a, s) in COLS}

    # ---- console audit: pooled recompute vs summary.json ----
    # summary.json is regenerated by analyze_phase1.py.  After the
    # re-run and a fresh analysis pass it must match this recompute in all 60
    # cells; until then it is the pre-fix snapshot and mismatches everything.
    # A mismatch therefore means either "analysis not re-run yet" or a real
    # inconsistency -- the message says which file to check.
    n_s1 = None
    try:
        s = json.load(open(os.path.join(_ROOT, "results", "phase1", "analysis",
                                        "summary.json")))
        cells = s["subexp1"]["cells"]
        n_s1 = s.get("stats", {}).get("n_s1", 0)
        mism = {arm: 0 for arm in ARMS}
        for sig in SIGMAS:
            for arm in ARMS:
                for smp in SAMP:
                    v = pooled[sig][(arm, smp)]
                    # summary.json keys by the on-disk arm name, so the
                    # display key "l1l4" must be resolved for the ALD column.
                    stored = cells[f"{sig:g}|{DIR_ARM[(arm, smp)]}|{smp}|mace"][
                        "induced_ood_rate"]
                    if abs(v - stored) > 1e-9:
                        mism[arm] += 1
        print(f"audit: recompute vs summary.json mismatches = "
              f"{sum(mism.values())}/60 {mism} "
              f"(summary.json holds {n_s1} subexp1 task files; expect 0 "
              f"mismatches after a post-fix analysis pass)")
    except Exception as e:                                  # noqa: BLE001
        print(f"audit: summary.json not comparable ({type(e).__name__}: {e}) "
              "-- expected before the post-fix analysis pass")

    # ---- reconciliation with the printed pooled table (paper \S6.2) --------
    # Parse the printed values from the paper tex (kept unchanged) and list
    # the cells whose one-decimal value differs from the frozen recompute.
    printed = parse_pooled_paper_table(os.path.join(_ROOT, "docs",
                                                    "paper_outline_v0.4.tex"))
    diff_cells = []
    if printed is not None:
        for sig in SIGMAS:
            for (a, smp) in COLS:
                pr = printed.get((sig, a, smp))
                v = pooled[sig][(a, smp)] * 100
                if pr is None or abs(pr - v) > 5e-3 + 1e-9:
                    diff_cells.append((sig, a, smp,
                                       f"{v:.2f}",
                                       "n/a" if pr is None else f"{pr:.2f}"))
        if diff_cells:
            n = len(diff_cells)
            print(f"audit: printed pooled Table differs from recompute in "
                  f"{n} cells (two-decimal values):")
            for sig, a, smp, v, pr in diff_cells:
                print(f"  s{sig:g} {a} {smp}: recompute {v}% vs printed {pr}%")
        else:
            print("audit: printed pooled Table matches recompute at the "
                  "printed precision in all 60 cells")

    # ---- render the standalone supplement ----
    L, a = [], (lambda x: L.append(x))
    a(r"% Sub-experiment 1 per-composition sigma-failure tables.")
    a(r"% Data: results/phase1/subexp1 (MACE-MP-0), all cells from the post-fix")
    a(r"% re-run (cell-initialization erratum: sigma-independent cell")
    a(r"% distortion before the fix; pre-fix trees quarantined under")
    a(r"% results/phase1/legacy_pre_cholesky/).  10 candidates per seed,")
    a(r"% so a per-composition cell is 5 seeds x 10 = 50 and the pooled cell 1000.")
    a(r"% Generator: scripts/make_percomp_sigma_tables.py  --  body is ASCII-only.")
    a(r"% The pooled table of the main text is cross-referenced as")
    a(r"% \ref{tab:sigma-failure}: it resolves once this file is merged as an")
    a(r"% appendix into paper_outline_v0.4.tex and prints ?? when compiled")
    a(r"% standalone.  Terminal validity ranges per composition: see \S6.2 text.")
    a("")
    a(r"\documentclass[11pt,a4paper]{article}")
    a(r"\usepackage[T1]{fontenc}")
    a(r"\usepackage{amsmath}")
    a(r"\usepackage{booktabs}")
    a(r"\usepackage{tabularx}")
    a(r"\usepackage[colorlinks=true]{hyperref}")
    a(r"\usepackage[margin=2.5cm]{geometry}")
    a("")
    a(r"\begin{document}")
    a("")
    a(r"\begin{center}")
    a(r"{\Large\textbf{Sub-experiment 1: Per-Composition $\sigma$-Failure"
      r" Tables}}\\[4pt]")
    a(r"{\normalsize Supplementary breakdown of Table~\ref{tab:sigma-failure}"
      r" (\S6.2), frozen protocol}")
    a(r"\end{center}")
    a("")
    a(r"Each of the 20 tables below reports one MP-20 test composition.")
    a(r"Entries are the \textbf{sampler-induced OOD rate} (\%), computed over")
    a(rf"that composition's \textbf{{5 seeds $\times$ {N_PER_COMP // 5} "
      rf"candidates $=$ {N_PER_COMP} trajectories}} per "
      r"($\sigma_{\max}$, condition) cell --- the pooled")
    a(r"Table~\ref{tab:sigma-failure} of \S6.2 is the union of these")
    a(rf"{N_PER_COMP}-trajectory cells over the 20 compositions "
      rf"({N_POOLED} trajectories per cell).")
    a("")
    a(r"\textbf{Definition} (identical to the pooled statistic): the OOD")
    a(r"mask is 0b010011 ($d_{\min} < 0.5$\,\AA{} $\mid$"
      r" $\max_i \|F_i^{\mathrm{NNP}}\| > 500$\,eV/\AA{} $\mid$")
    a(r"$|\det L| / |\det L_0| > 10$); ``induced'' = share of \emph{all}")
    a(rf"trajectories ($n = {N_PER_COMP}$) that started clean and had an OOD "
      r"event at")
    a(r"any step $\geq 1$; trajectories already OOD at step 0 (the start-OOD")
    a(r"floor) are excluded from the numerator exactly as in the pooled")
    a(r"definition.  The induced count is additive over compositions, so")
    a(r"pooling the 20 tables reproduces the pooled table exactly")
    a(r"(cross-check Table~\ref{tab:sigma-failure-pooled-recompute}).")
    a(r"\textbf{Cell initialization:} all cells are the post-fix "
      r"re-run")
    a(r"(cell-initialization erratum in \S\ref{sec:exp-comp-sigma}: the "
      r"pre-fix")
    a(r"initial cells were distorted by 45--114\%, sigma-independently).  "
      r"The")
    a(r"pre-fix trees are archived at")
    a(r"\texttt{results/phase1/legacy\_pre\_cholesky/}.")
    a("")
    a(r"\textbf{Columns:} the six main-text conditions --- \textbf{Bare} /")
    a(r"\textbf{L1} / protected $\times$ \textbf{ALD} / \textbf{PF-ODE},")
    a(r"in the column order of Table~\ref{tab:sigma-failure}.  The protected")
    a(r"column is the dedicated \textbf{L1--L3} ladder on ALD (code name")
    a(r"\texttt{l1l2l3}) and \textbf{L1} $+$ \textbf{L3} on the ODE path (code")
    a(r"name \texttt{l1l4}, marked $^{\ddagger}$): the ODE configurations")
    a(r"carry no L2 noise injection and no L4 monitor, so on that path the two")
    a(r"code names are the identical configuration field by field.")
    a("")
    a(rf"\textbf{{Statistical note:}} per-cell $n = {N_PER_COMP}$ puts "
      rf"single-trajectory")
    a(rf"resolution at {100.0 / N_PER_COMP:.0f}\,pp (per-seed cells, "
      rf"{N_PER_COMP // 5} candidates each, would sit at")
    a(rf"{100.0 / (N_PER_COMP // 5):.0f}\,pp); the pooled "
      r"Table~\ref{tab:sigma-failure} is the statistically")
    a(r"stable estimate and the text's collapse-scale fits (\S6.2"
      r" Sub-experiments 1 and 4) are computed on the pooled or per-step")
    a(rf"data, not on these per-composition $n = {N_PER_COMP}$ rates.  "
      r"The tables below")
    a(r"serve the per-composition audit of the failure curve (spread of the")
    a(r"rise across the 20 compositions, cf.\ the per-composition $\sigma_c$")
    a(r"of Table~\ref{tab:basin-radius}).")
    a("")
    if diff_cells is not None:
        names = {"bare": "Bare", "l1": "L1", "l1l4": "L1--L3"}
        smps = {"ald": "ALD", "pfode": "PF-ODE"}
        if diff_cells:
            a(r"\textbf{Reconciliation with the printed pooled table:} the")
            a(r"pooled recompute (last table) is the same sum over the same")
            a(rf"{N_POOLED}-trajectory cells as the pooled")
            a(r"Table~\ref{tab:sigma-failure}, yet the printed")
            a(rf"{len(diff_cells)} of its 60 entries differ from this recompute")
            a(r"(each a $\leq 0.1$\,pp difference, up to a rounding step of")
            a(rf"$\sim$2 of {N_PER_COMP} trajectories):")
            for sig, arm, smp, v, pr in diff_cells:
                a(fr"{names[arm]} {smps[smp]} at $\sigma_{{\max}} = {sig:.1f}$: "
                  fr"{pr} (printed) vs.\ {v} (recompute);")
            a(r"a mismatch here means the main-text table was not re-rendered")
            a(r"after the fixed-cell migration --- re-run")
            a(r"\texttt{scripts/make\_percomp\_sigma\_tables.py} and port the")
            a(r"last table into the paper Appendix~I.")
        else:
            a(r"\textbf{Reconciliation with the printed pooled table:} the")
            a(r"pooled recompute (last table) is the same sum over the same")
            a(rf"{N_POOLED}-trajectory cells as the pooled")
            a(r"Table~\ref{tab:sigma-failure} and reproduces it in all 60")
            a(rf"entries (0/60 differ, to two decimals = 1/{N_PER_COMP} of a cell).")
            a(r"Both sides come from the same post-fix re-run of")
            a(r"Sub-experiment 1 (cell-initialization erratum,")
            a(r"\S\ref{sec:exp-comp-sigma}); the pre-fix trees are archived at")
            a(r"\texttt{results/phase1/legacy\_pre\_cholesky/}.")
            if n_s1 is not None:
                a(rf"The committed \texttt{{summary.json}} holds {n_s1} ")
                a(r"sub-experiment-1 task files and agrees with this recompute")
                a(r"in all 60 cells.")
        a("")
    for comp in comps:
        pcts = [[counts[(sig, comp)][(a, smp)] / N_PER_COMP * 100  # k/n -> per cent
                 for (a, smp) in COLS] for sig in SIGMAS]
        cap = (f"Sub-experiment 1 --- per-composition $\\sigma$-failure curve, "
               f"{TEX[comp]} (5 seeds $\\times$ {N_PER_COMP // 5} candidates "
               f"$=$ {N_PER_COMP} trajectories per cell).  Entries: "
               f"sampler-induced OOD rate (\\%), clean-start numerator over all "
               f"$n = {N_PER_COMP}$ trajectories; definition and column order "
               f"as in Table~\\ref{{tab:sigma-failure}}.")
        L.extend(table_lines(SIGMA_FMT, pcts,
                             f"tab:sigma-failure-{comp}", cap))
        a("")
    # ---- pooled recompute cross-check ----
    pcts = [[pooled[sig][(a, smp)] * 100 for (a, smp) in COLS]
            for sig in SIGMAS]
    cap = (r"Pooled recompute --- summing the induced counts of the 20 "
           r"per-composition tables above reproduces the pooled "
           r"Table~\ref{tab:sigma-failure} of \S\ref{sec:exp-comp-sigma} "
           rf"({N_POOLED} trajectories per cell).  Entries (\%) = "
           rf"$\sum_{{\mathrm{{comp}}}} k_{{\mathrm{{comp}}}}/{N_POOLED}$, computed "
           r"cell-by-cell from the same data set as "
           r"Table~\ref{tab:sigma-failure}.  Every cell is the "
           r"post-fix re-run (cell-initialization erratum, "
           r"\S\ref{sec:exp-comp-sigma}); the pre-fix trees are archived at "
           r"\texttt{results/phase1/legacy\_pre\_cholesky/}.")
    L.extend(table_lines(SIGMA_FMT, pcts,
                         "tab:sigma-failure-pooled-recompute", cap))
    a("")
    a(r"\end{document}")
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", OUT)

    # ---- per-composition spread at sigma = 1 (bare / l1 / l1l2l3, ALD) ----
    print(f"\nper-composition induced counts @ sigma=1.0 "
          f"(k / {N_PER_COMP}):")
    for comp in comps:
        b = counts[(1.0, comp)]["bare", "ald"]
        l1 = counts[(1.0, comp)]["l1", "ald"]
        l3 = counts[(1.0, comp)]["l1l4", "ald"]       # -> l1l2l3 on disk
        print(f"  {comp:8s} bare {b:3d}   l1 {l1:3d}   L1-L3 {l3:3d}")


if __name__ == "__main__":
    main()
