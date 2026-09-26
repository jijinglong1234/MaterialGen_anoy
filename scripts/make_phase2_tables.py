"""Phase 2 result tables: the frozen generation-quality grid.

Writes docs/phase2_tables.tex as BODY-ONLY LaTeX -- complete tabular
environments with no table float, no caption, no label -- following
docs/comp_sigma_c_table.tex.  The paper is self-contained (it \\inputs only
math_commands.tex), so the bodies are pasted into their floats in
docs/paper/conference.tex rather than \\input, and this file stays
the source of truth for the numbers.

Four bodies:

  tab:phase2-mp20      MP-20, the nine frozen cells.  Main text (Sec. 6.3).
  tab:phase2-baseline  our MP-20 reference against the third-party row, with
                       the columns that make the two non-comparable visible.
  tab:phase2-appendix  Perov-5 and Carbon-24, same columns, 18 rows.
  tab:phase2-reserved  the protocol slots Phase 2 did NOT fill.

Every number is read from results/phase2/summary/phase2_esen.csv (the canonical
27-row summary) or phase2_third_party.csv.  Nothing is transcribed by hand and
nothing is recomputed: if a cell is missing the script aborts rather than
printing a shorter table.

Two printing conventions, both deliberate:

  * Cells carry the point estimate only.  The spread is over the 20 test
    compositions, not over trajectories, and it is wide enough that a
    mean +- sd column pair would triple the table width to say what the Notes
    row says once.  Same convention as tab:sigma-failure.
  * Percentages are printed to one decimal and E_hull to whole meV/atom.  The
    E_hull values are medians over a per-cell subsample that is not always all
    20 cells (Perov-5), so more digits would be false precision.
"""
import csv
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(_ROOT, "results", "phase2", "summary")
OUT = os.path.join(_ROOT, "docs", "phase2_tables.tex")

#: The nine frozen cells, in draw order, with the spec each one froze
#: (docs/phase2_frozen_config.md Sec. 5).  The layer set and the lattice flag
#: are properties of the cell, not of the sampler, so they are stated once here
#: and the table does not need two more columns to repeat them.
CELLS = [
    ("C1",  "ald",   "0.5", "---",     "on"),
    ("C3",  "ald",   "1.5", "---",     "on"),
    ("C5",  "ald",   "0.5", "---",     "fixed"),
    ("C8",  "ald",   "0.5", "L1--L3",  "on"),
    ("C9",  "ald",   "1.5", "L1--L3",  "on"),
    ("C11", "ald",   "0.5", "L1",      "on"),
    ("C5",  "pfode", "0.5", "---",     "on"),
    ("C8",  "pfode", "0.5", "L1--L3",  "on"),
    ("C11", "pfode", "0.5", "L1",      "on"),
]
N_ALD = 6                       # rows before the ALD | PF-ODE rule


def load():
    """The canonical summary as {(dataset, config, sampler): row}."""
    path = os.path.join(SUMMARY, "phase2_esen.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 27:
        raise SystemExit(
            f"{path} has {len(rows)} rows, expected 27 (3 datasets x 9 cells) "
            f"-- re-run `python scripts/run_phase2.py --analyze --nnp esen`.")
    out = {}
    for r in rows:
        key = (r["dataset"], r["config"], r["sampler"])
        if key in out:
            raise SystemExit(f"duplicate row {key} in {path}")
        out[key] = r
    for ds in ("mp_20", "perov_5", "carbon_24"):
        for cfg, smp, _s, _l, _u in CELLS:
            if (ds, cfg, smp) not in out:
                raise SystemExit(f"cell {ds}/{cfg}/{smp} missing from {path}")
    return out


def f(row, key, scale=1.0, nd=1):
    """One cell, or a loud failure -- never a blank that reads as zero."""
    v = row.get(key)
    if v in (None, ""):
        raise SystemExit(f"column {key!r} empty for "
                         f"{row['dataset']}/{row['config']}/{row['sampler']}")
    return f"{float(v) * scale:.{nd}f}"


def cap(row):
    """The NFE-cap column.  ALD has no evaluation cap -- it is a stochastic
    sampler with a fixed step budget, not a solver with a convergence
    criterion -- so its column is '---', not a literal 0.0 that would read as
    'converged for free'."""
    return "---" if row["sampler"] == "ald" else f(row, "capped_rate", 100)


#: Column specs, one per body.  The number of columns is what the notes rows'
#: \multicolumn spans have to match, so it is stated once here and not
#: re-written inside the template.
SPEC_12 = r"@{}llc*{9}{r}@{}"       # 3 labels + 9 numeric: the quality tables
SPEC_9 = r"@{}lc*{7}{r}@{}"         # 2 labels + 7 numeric: the comparison.
# 9 columns, not 10: the Dataset column was dropped because both rows are
# MP-20 (the caption says so).  At 10 columns the float overran \textwidth by
# 98pt, and the surviving overflow is closed in the paper by \footnotesize and
# a 3pt \tabcolsep on that one float -- the body below is width-neutral.


def cells(hdr, body_rows, label=None, spec=SPEC_12):
    """A body-only tabular.  ``body_rows`` are pre-joined strings."""
    L = []
    if label:
        L.append(f"% ---- {label} ----")
    L.append(r"\begin{tabular}{" + spec + "}")
    L.append(r"\toprule")
    L.append(hdr + r" \\")
    L.append(r"\midrule")
    L.extend(body_rows)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return L


def main():
    R = load()
    L = ["% Generated by scripts/make_phase2_tables.py; do not edit.",
         "% Bodies only: paste each into its own table float.", ""]

    # ------------------------------------------------------------------
    # tab:phase2-mp20
    # ------------------------------------------------------------------
    hdr = (r"\textbf{Cell} & \textbf{$\sigma_{\max}$} & \textbf{Layers} & "
           r"\textbf{Valid} & \textbf{Stable} & \textbf{S.U.N.} & "
           r"\textbf{Match} & \textbf{Cover.} & \textbf{AMSD} & "
           r"\textbf{R-KL} & \textbf{$E_{\text{hull}}$} & \textbf{Cap}")
    body = []
    for i, (cfg, smp, sig, lay, upd) in enumerate(CELLS):
        if i == N_ALD:
            body.append(r"\midrule")
        r = R[("mp_20", cfg, smp)]
        name = f"{cfg}" + (r"$^{\dag}$" if upd == "fixed" else "")
        body.append(
            f"{name} & {sig} & {lay} & "
            f"{f(r, 'validity_rate', 100)} & {f(r, 'stable_rate', 100)} & "
            f"{f(r, 'sun_rate', 100)} & {f(r, 'match_rate', 100)} & "
            f"{f(r, 'coverage', 100)} & {f(r, 'amsd')} & "
            f"{f(r, 'r_angle_kl', 1, 2)} & "
            f"{f(r, 'e_hull_med_phys_meV', 1, 0)} & "
            f"{cap(r)} \\\\")
    body += [
        r"\midrule",
        r"\multicolumn{12}{@{}p{0.96\textwidth}}{\footnotesize All rates in "
        r"\%; AMSD in \AA; R-KL is a KL divergence in nats against the "
        r"composition's reference angular distribution, so lower is closer to "
        r"the reference; $E_{\text{hull}}$ in meV/atom. Rows 1--6 are ALD, "
        r"rows 7--9 PF-ODE. Each cell pools 20 test compositions $\times$ 5 "
        r"seeds $\times$ 10 candidates $= 1{,}000$ trajectories; the spread "
        r"over the 20 cells is drawn in Fig.~\ref{fig:phase2-mp20} and is not "
        r"repeated here. Stable is $E_{\text{hull}} < 100$\,meV/atom. "
        r"$E_{\text{hull}}$ is a median over cells, so it is a median of "
        r"medians rather than a pooled median. Cap is the share of PF-ODE "
        r"trajectories exhausting the 1{,}000-evaluation budget. "
        r"$^{\dag}$C5 holds the cell fixed; every other row updates it.} \\",
    ]
    L += cells(hdr, body, "tab:phase2-mp20")
    L.append("")

    # ------------------------------------------------------------------
    # tab:phase2-baseline
    # ------------------------------------------------------------------
    # The two stability columns are the whole point of this table and the
    # reason it is not a like-for-like comparison.  Our arms resolve a hull
    # entry for every candidate on MP-20, so "all candidates" and "candidates
    # with a known hull" are the same denominator and the two columns agree.
    # The third-party row's hull is known for 7.9% of its samples, so its
    # all-candidates rate is 0.063 where its known-hull rate is 0.797 -- an
    # order of magnitude that is a property of the hull lookup, not of the
    # generator.  Printing them side by side is what stops the 0.063 from
    # being read as a bad generator.
    tp_path = os.path.join(SUMMARY, "phase2_third_party.csv")
    with open(tp_path) as fh:
        tp = list(csv.DictReader(fh))
    if len(tp) != 1:
        raise SystemExit(f"{tp_path} has {len(tp)} rows; the comparison table "
                         f"is written for exactly one third-party row.")
    tp = tp[0]
    ours = R[("mp_20", "C11", "pfode")]
    # AMSD and R-angle KL are deliberately absent.  Both are pooled over all
    # samples for the third-party row and averaged over composition-matched
    # cells for ours, so the two entries are not the same statistic; a column
    # that looks comparable but is not is worse than no column.  Their
    # exclusion is stated in the note.
    hdr = (r"\textbf{Arm} & \textbf{Potential} & "
           r"\textbf{Valid} & \textbf{S.U.N.} & \textbf{Match} & "
           r"\textbf{Cover.} & \textbf{$E_{\text{hull}}$} & "
           r"\textbf{Stable (all)} & \textbf{Stable (known hull)}")
    body = [
        f"{r'Ours (C11, PF-ODE)'} & eSEN & "
        f"{f(ours, 'validity_rate', 100)} & {f(ours, 'sun_rate', 100)} & "
        f"{f(ours, 'match_rate', 100)} & {f(ours, 'coverage', 100)} & "
        f"{f(ours, 'e_hull_med_phys_meV', 1, 0)} & "
        f"{f(ours, 'stable_rate', 100)} & {f(ours, 'stable_rate', 100)} \\\\",
        f"{r'DiffCSP (uncond.)'} & MACE & "
        f"{f(tp, 'validity_rate', 100)} & {f(tp, 'sun_rate', 100)} & "
        f"{f(tp, 'match_rate', 100)} & {f(tp, 'coverage', 100)} & "
        f"{f(tp, 'e_hull_med_phys_meV', 1, 0)} & "
        f"{f(tp, 'stable_rate_all', 100)} & "
        f"{f(tp, 'stable_rate', 100)} \\\\",
        r"\midrule",
        r"\multicolumn{9}{@{}p{0.96\textwidth}}{\footnotesize The third-party "
        r"row's two stability columns deliberately differ. \emph{All} counts "
        r"every sample, \emph{known hull} counts only those whose hull entry "
        r"the lookup resolved: for the third-party row that is "
        f"{float(tp['frac_hull_known']) * 100:.1f}\\% of "
        f"$n = {tp['n']}$ samples, so its all-candidates rate is set by the "
        r"lookup rather than by the generator, while ours has a resolved hull "
        r"for every MP-20 candidate and the two columns coincide. Its "
        r"$E_{\text{hull}}$ is likewise a pooled median over those "
        r"hull-known samples, ours a median of per-composition medians. Three "
        r"further differences are not correctable in the table: the scored "
        r"third-party arm is \emph{unconditional}, so it does not satisfy the "
        r"protocol's identical-test-composition condition; it is evaluated "
        r"with MACE where our row uses eSEN; and its diversity metrics (AMSD, "
        r"R-angle KL) are pooled while ours are composition-matched, so both "
        r"are omitted rather than shown as if comparable. Direction is "
        r"informative; magnitudes are not comparable across the two rows.} \\",
    ]
    L += cells(hdr, body, "tab:phase2-baseline", spec=SPEC_9)
    L.append("")

    # ------------------------------------------------------------------
    # tab:phase2-appendix
    # ------------------------------------------------------------------
    hdr = (r"\textbf{Cell} & \textbf{$\sigma_{\max}$} & \textbf{Layers} & "
           r"\textbf{Valid} & \textbf{Stable} & \textbf{S.U.N.} & "
           r"\textbf{Match} & \textbf{Cover.} & \textbf{AMSD} & "
           r"\textbf{R-KL} & \textbf{$E_{\text{hull}}$} & \textbf{Cap}")
    body = []
    for ds, lab in (("perov_5", "Perov-5"), ("carbon_24", "Carbon-24")):
        if ds != "perov_5":
            body.append(r"\midrule")
        body.append(r"\multicolumn{12}{@{}l}{\emph{" + lab + r"}} \\")
        for i, (cfg, smp, sig, lay, upd) in enumerate(CELLS):
            if i == N_ALD:
                body.append(r"\midrule")
            r = R[(ds, cfg, smp)]
            name = f"{cfg}" + (r"$^{\dag}$" if upd == "fixed" else "")
            body.append(
                f"{name} & {sig} & {lay} & "
                f"{f(r, 'validity_rate', 100)} & {f(r, 'stable_rate', 100)} & "
                f"{f(r, 'sun_rate', 100)} & {f(r, 'match_rate', 100)} & "
                f"{f(r, 'coverage', 100)} & {f(r, 'amsd')} & "
                f"{f(r, 'r_angle_kl', 1, 2)} & "
                f"{f(r, 'e_hull_med_phys_meV', 1, 0)} & "
                f"{cap(r)} \\\\")
    # Perov-5's two qualifications, read rather than written down: the E_hull
    # column's cell count and the unphysical share are per-arm quantities and
    # both vary across the nine rows.
    pv = json.load(open(os.path.join(SUMMARY, "phase2_perov_5_esen.json")))
    pv_blocks = [pv["configs"][f"{c}_{s}"] for c, s, _x, _y, _z in CELLS]
    pv_n = [b["aggregate"]["e_hull_median_over_cells"]["n"] for b in pv_blocks]
    pv_bad = [b["aggregate"]["e_hull_valid"]["frac_unphysical"] * 100
              for b in pv_blocks]
    body += [
        r"\midrule",
        r"\multicolumn{12}{@{}p{0.96\textwidth}}{\footnotesize Columns as in "
        r"Table~\ref{tab:phase2-mp20}. \emph{Perov-5}: $E_{\text{hull}}$ is a "
        f"median over {min(pv_n)}--{max(pv_n)} of the 20 cells, because "
        f"{min(pv_bad):.0f}--{max(pv_bad):.0f}\\% of its candidates have no "
        r"physical hull value at all (all below the $-0.05$\,eV/atom floor), "
        r"so that column is a median over a selected subsample as well as a "
        r"median of per-cell medians. The dropout is not arm-specific: it "
        r"appears under all nine. \emph{Carbon-24}: \textbf{S.U.N. is 0 in "
        r"every row by criterion degeneracy, not by sampler failure} --- "
        r"Stable and Novel are exactly disjoint on a single-composition "
        r"dataset, so the collapsed rate cannot be read as quality here --- "
        r"and its three PF-ODE rows are budget-truncated rather than "
        r"converged (Cap 88.6--89.3\%, every cell's median at the "
        r"1{,}000-step ceiling), so those three rows measure a partial "
        r"trajectory. Both qualifications are stated in full in the "
        r"surrounding text.} \\",
    ]
    L += cells(hdr, body, "tab:phase2-appendix")
    L.append("")

    # ------------------------------------------------------------------
    # tab:phase2-reserved
    # ------------------------------------------------------------------
    L += [
        "% ---- tab:phase2-reserved ----",
        r"\begin{tabularx}{\textwidth}{@{}lX@{}}",
        r"\toprule",
        r"\textbf{Measurement} & \textbf{Quantity to be reported} \\",
        r"\midrule",
        r"Score mixing & quality versus $\lambda$; bare and augmented "
        r"$\lambda = 0$ overlay \\",
        r"Curl violation & $\mathbb{E}\|\nabla\times s\|_F$ for "
        r"potential-based versus learned scores \\",
        r"Efficiency & evaluations to $E_{\text{hull}} < 100$\,meV/atom versus "
        r"MD and basin hopping \\",
        r"Cross-potential & per-potential validity, S.U.N., $E_{\text{hull}}$; "
        r"rank correlation \\",
        r"DFT transfer & rank correlation, relaxation displacement, stability "
        r"retention \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", OUT, f"({len(L)} lines)")

    # ---- console cross-check, so the numbers that go into the prose can be
    #      read off the same run that wrote the tables ----
    print("\nMP-20 reference points for the Sec. 6.3 prose:")
    for key, lab in ((("mp_20", "C1", "ald"), "bare ALD"),
                     (("mp_20", "C8", "ald"), "L1-L3 ALD"),
                     (("mp_20", "C5", "pfode"), "bare PF-ODE"),
                     (("mp_20", "C8", "pfode"), "L1-L3 PF-ODE"),
                     (("mp_20", "C11", "pfode"), "L1 PF-ODE"),
                     (("mp_20", "C3", "ald"), "bare ALD, sigma 1.5"),
                     (("mp_20", "C9", "ald"), "L1-L3 ALD, sigma 1.5")):
        r = R[key]
        print(f"  {lab:22s} valid {float(r['validity_rate'])*100:5.1f}  "
              f"stable {float(r['stable_rate'])*100:5.1f}  "
              f"sun {float(r['sun_rate'])*100:5.1f}  "
              f"e_hull {float(r['e_hull_med_phys_meV']):7.1f}  "
              f"cap {float(r['capped_rate'])*100:5.1f}")


if __name__ == "__main__":
    main()
