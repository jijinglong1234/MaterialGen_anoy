"""
Phase 1 analysis — aggregate results/phase1/* into summary.json, figures, report.

Re-runnable at any fleet progress: pools whatever JSONs exist per condition cell,
fits the §6.1 sigmoid collapse model, computes the M1 gate metrics, and renders
Figure 1 (4 panels) + supplementary figures + docs/phase1_report.md.

Outputs:
  results/phase1/analysis/summary.json         pooled metrics + fits
  results/phase1/analysis/figures/fig1_{a,b,c,d}.pdf/.png   Figure 1 panels
  results/phase1/analysis/figures/fig_sm_heatmap.png        SafetyMonitor scan
  results/phase1/analysis/figures/fig_s5_pauli.png          Pauli sensitivity
  results/phase1/analysis/figures/fig_rc_esen.png           eSEN recheck vs MACE
  docs/phase1_report.md                                     report + M1 gates

Usage:
    python scripts/analyze_phase1.py            # full analysis (whatever data exists)
    python scripts/analyze_phase1.py --no-figs  # metrics only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase1"
ANALYSIS = OUT / "analysis"
FIGS = ANALYSIS / "figures"

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
OOD_DMIN, OOD_FMAX = 0.5, 500.0
#: run_phase1.PF_MAX_NFE.  PF-ODE integrates with an adaptive DOPRI5(4) step
#: and stops at this many RHS evaluations; the ALD arm is fixed at K*M = 200
#: and can never reach it.  Share of candidates at the cap is reported as
#: ``capped_rate`` because a capped trajectory is a *truncated* integration,
#: not a converged PF-ODE sample: at sigma = 1.5 the measured share is 93%, so
#: without this column a summary row silently reads as ODE-converged quality.
NFE_CAP = 1000
# ALD protection arms of the sigma scan.  The protected ALD arm is
# the dedicated l1l2l3 ladder (run_phase1.SIGMA_SCAN_ALD_ONLY), so the sigma
# plots and the report read L1--L3, not the l1l4 stack.  The PF-ODE side is
# unaffected: on the ODE path l1l2l3 and l1l4 are the same configuration field
# by field (no L2 density noise, no L4 monitor), so the hardcoded
# ["bare", "l1l4"] lists below stay correct and the l1l4 directory remains the
# ODE source.  The eSEN recheck arm has its own literal list (it ran l1l4).
PROTECTIONS = ["bare", "l1", "l1l2l3"]


def ode_prot(prot):
    """PF-ODE counterpart of an ALD protection arm.

    The protected ALD arm is the sigma-scan-only ``l1l2l3`` ladder, but no
    PF-ODE run ever carried that name -- on the ODE path the protected arm is
    ``l1l4``, the *identical* configuration field by field
    (make_pfode_config(s, "l1l2l3") == make_pfode_config(s, "l1l4"): no L2
    density noise, no L4 monitor on the ODE path).  Looking up subexp1 cells
    under the ALD name would silently return None and blank the PF-ODE curve.
    """
    return "l1l4" if prot == "l1l2l3" else prot
SAMPLERS = ["ald", "pfode"]
RECHECK_COMPS = ["SrTiO3", "ZnS", "LiF", "GaN", "FeNi3"]

# Validated categorical palette (dataviz reference; adjacent-pair CVD ΔE>=8,
# normal-vision >=15 in both modes) + ink tokens for the light surface.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_P, INK_S, INK_MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#1c5cab"]  # ordinal ramp 250..600


def prot_label(p: str, sampler: str | None = None) -> str:
    if sampler == "pfode" and p == "l1l4":
        return "L1+L3 (refl.)"
    return {"bare": "bare", "l1": "L1", "l1l2": "L1+L2", "l1l2l3": "L1+L2+L3",
            "l1l4": "L1-L4"}[p]


# ---------------------------------------------------------------------------
# Loading / pooling
# ---------------------------------------------------------------------------

# cell-init guard.  Every payload produced after the noise_lattice
# cholesky fix carries this marker; files without it ran in a distorted
# initial cell (sigma-independent 45-114% stretch, see
# results/phase1/legacy_pre_cholesky/README.md) and must never be
# pooled with post-fix data.  The legacy trees were moved out of
# results/phase1/ entirely, so this guard is the second line of defence: it
# catches a stray copy, a resumed run writing into a stale tree, or a manual
# file drop.  The value is imported from run_phase1 (the writer) rather than
# repeated here: a mismatch between writer and reader would silently reject
# every new file.
import run_phase1 as _rp                                      # noqa: E402

INIT_CELL_GEN = _rp.INIT_CELL_GEN
if _rp.N_CAND != 10:                # re-run protocol
    print(f"WARNING: run_phase1.N_CAND = {_rp.N_CAND}, not 10 -- per-cell "
          "trajectory counts will not match the paper's 1000-trajectory cells")
LEGACY_DIR = "results/phase1/legacy_pre_cholesky"
ALLOW_LEGACY = False           # set by --allow-legacy (archaeology only)
_warned: set = set()


def load_subexp(subexp: str, allow_legacy: bool | None = None) -> list[dict]:
    allow = ALLOW_LEGACY if allow_legacy is None else allow_legacy
    records, legacy = [], 0
    base = OUT / subexp
    if not base.exists():
        return records
    for path in sorted(base.rglob("*.json")):
        try:
            task = json.loads(path.read_text())
        except Exception:
            continue
        if not allow and task.get("init_cell_gen") != INIT_CELL_GEN:
            legacy += 1
            continue
        records.append(task)
    if legacy and subexp not in _warned:
        _warned.add(subexp)
        print(f"  [cell-init guard] {subexp}: skipped {legacy} file(s) lacking "
              f"init_cell_gen={INIT_CELL_GEN!r} -- pre-fix initial cells, "
              f"archive: {LEGACY_DIR}/ (use --allow-legacy to include them)")
    return records


_PLAIN_FORMULA = re.compile(r"(?:[A-Z][a-z]?\d*)+")


def formula_atom_count(formula: str) -> int:
    """Atoms in one formula unit of a plain reduced formula (Al2O3 -> 5).

    Equivalent to pymatgen Composition(formula).num_atoms for the Phase 1
    composition set; anything with brackets, decimals or a non-plain form
    falls back to pymatgen.
    """
    if _PLAIN_FORMULA.fullmatch(formula or ""):
        return sum(int(cnt) if cnt else 1 for cnt in
                   (m.group(1) for m in re.finditer(r"[A-Z][a-z]?(\d*)", formula)))
    from pymatgen.core import Composition
    return int(Composition(formula).num_atoms)


def cands_of(task: dict):
    """Candidate records of one task payload, with E_hull in true eV/atom.

    Units history: the generator divided pymatgen's already
    per-atom get_e_above_hull() by the composition's formula-unit atom count
    again (run_phase1.HullEvaluator.e_hull), so those files store
    E_hull / n_fu with n_fu = 2-5 depending on the composition.  Such legacy
    files carry no "e_hull_units" marker and are rescaled here, in place, by
    that composition-constant factor; the task dict is stamped afterwards so
    repeated calls cannot rescale twice.
    """
    cands = task.get("candidates", [])
    comp = task.get("composition")
    if cands and comp and task.get("e_hull_units") != "eV/atom":
        factor = formula_atom_count(comp)
        for c in cands:
            fin = c.get("final") or {}
            eh = fin.get("e_hull")
            if eh is not None:
                fin["e_hull"] = eh * factor
        task["e_hull_units"] = "eV/atom"
    return cands


def cell_metrics(cands: list[dict]) -> dict:
    """Pooled per-cell metrics over all candidates of a condition cell.

    Empty input returns the full schema (rates 0, medians None) so consumers
    can rely on key presence and guard on n_traj.

    OOD accounting: step 0 (the initial structure) is excluded
    from OOD spike counting.  The σ_max cartesian-noise protocol produces an
    unavoidable initial-condition floor (d_min<0.5 or |F|>500 already at
    step 0 once σ_max approaches the basin radius); the defense layers act on
    the sampling dynamics, so the gates measure *sampler-induced* OOD
    (steps ≥ 1).  The floor is reported separately as `start_ood_rate`."""
    n = len(cands)
    base = {"n_traj": n, "ood_rate": 0.0, "validity": 0.0,
            "buckets": dict.fromkeys(("clean", "t1", "force", "t2", "nan", "mixed"), 0.0),
            "d_min_med": None, "d_min_med_valid": None,
            "e_hull_mean_meV": None, "e_hull_med_meV": None, "e_hull_p90_meV": None,
            "nfe_mean": None, "nfe_med": None, "rej_mean": None,
            "capped_rate": 0.0,
            "t1_rate": 0.0, "t2_rate": 0.0, "force_rate": 0.0, "nan_rate": 0.0,
            "collapse_rate": 0.0,
            "t3_rate": 0.0,  # structurally absent under fixed cell
            "single_mode_rate": 0.0, "mixed_mode_rate": 0.0,
            "start_ood_rate": 0.0, "induced_ood_rate": 0.0}
    if n == 0:
        return base
    ood = [c for c in cands if c["ood_steps"] > 0]
    valid = [c for c in cands if c["valid"]]
    dmin_all = [c["final"]["d_min"] for c in cands
                if c["final"]["d_min"] is not None and np.isfinite(c["final"]["d_min"])]
    dmin_valid = [c["final"]["d_min"] for c in valid
                  if c["final"]["d_min"] is not None and np.isfinite(c["final"]["d_min"])]
    eh = [c["final"]["e_hull"] for c in valid
          if c["final"]["e_hull"] is not None and np.isfinite(c["final"]["e_hull"])]

    # OOD events on steps >= 1 only (see docstring).  Stored bitmasks are
    # per recorded step; step 0 is the initial structure.
    def ood_after_start(c: dict) -> bool:
        bm = c.get("ood_bitmask") or []
        return any(m & 0b010011 for m in bm[1:]) if len(bm) > 1 else False

    start_ood = [c for c in cands
                 if (c.get("ood_bitmask") or [0])[0] & 0b010011]
    induced = [c for c in cands if ood_after_start(c)
               and not ((c.get("ood_bitmask") or [0])[0] & 0b010011)]

    # exclusive trajectory buckets: clean / single-mode / mixed
    n_t1 = sum(1 for c in cands if c["type1_steps"] > 0)
    n_t2 = sum(1 for c in cands if c["type2_steps"] > 0)
    n_force = sum(1 for c in cands if any(m & 0b000010 for m in (c["ood_bitmask"] or [])[1:]))
    n_nan = sum(1 for c in cands if c["nan_steps"] > 0)
    # lattice collapse (paper section 4.2): Type I under a shrunken cell;
    # structurally zero in fixed-cell Phase 1, expected only in upd_lat runs
    n_collapse = sum(1 for c in cands if any(m & 0b1000000 for m in (c["ood_bitmask"] or [])))
    modes = [(c["type1_steps"] > 0, c["type2_steps"] > 0,
              any(m & 0b000010 for m in (c["ood_bitmask"] or [])[1:]), c["nan_steps"] > 0) for c in cands]
    n_single = sum(1 for m in modes if sum(m) == 1)
    n_mixed = sum(1 for m in modes if sum(m) >= 2)

    m = {
        "n_traj": n,
        "ood_rate": len(ood) / n,
        "start_ood_rate": len(start_ood) / n,
        "induced_ood_rate": len(induced) / n,
        "validity": len(valid) / n,
        "buckets": bucket_fractions(cands),
        "d_min_med": float(np.median(dmin_all)) if dmin_all else None,
        "d_min_med_valid": float(np.median(dmin_valid)) if dmin_valid else None,
        "e_hull_mean_meV": float(np.mean(eh) * 1000) if eh else None,
        "e_hull_med_meV": float(np.median(eh) * 1000) if eh else None,
        "e_hull_p90_meV": float(np.percentile(eh, 90) * 1000) if eh else None,
        "nfe_mean": float(np.mean([c["nfe"] for c in cands])),
        "nfe_med": float(np.median([c["nfe"] for c in cands])),
        "capped_rate": sum(1 for c in cands if c["nfe"] >= NFE_CAP) / n,
        "rej_mean": float(np.mean([c["n_rejections"] for c in cands])),
        "t1_rate": n_t1 / n, "t2_rate": n_t2 / n, "force_rate": n_force / n,
        "nan_rate": n_nan / n, "collapse_rate": n_collapse / n,
        "single_mode_rate": n_single / n, "mixed_mode_rate": n_mixed / n,
    }
    return m


def bucket_fractions(cands: list[dict]) -> dict[str, float]:
    """Exclusive trajectory buckets summing to 1: clean, t1, t2, force, nan, mixed."""
    n = max(len(cands), 1)
    frac = {"clean": 0.0, "t1": 0.0, "force": 0.0, "t2": 0.0, "nan": 0.0, "mixed": 0.0}
    for c in cands:
        ms = (c["type1_steps"] > 0, c["type2_steps"] > 0,
              any(m & 0b000010 for m in c["ood_bitmask"]), c["nan_steps"] > 0)
        k = sum(ms)
        if k == 0:
            frac["clean"] += 1
        elif k == 1:
            frac[("t1", "t2", "force", "nan")[ms.index(True)]] += 1
        else:
            frac["mixed"] += 1
    return {k: v / n for k, v in frac.items()}


def fit_sigmoid(sigmas: list[float], y: list[float]):
    """Sigmoid fit in σ; returns dict {sigma_c, tau, amplitude, fit_ok}.

    Tries both orientations — validity(σ) = 1/(1+exp((σ−σc)/τ)) (decreasing)
    and the OOD-spike curve 1/(1+exp((σc−σ)/τ)) (increasing) — with and
    without a saturation amplitude A (the measured OOD curve saturates below
    1 because most trajectories recover; a 2-param sigmoid then pins σc to
    the bound).  Returns the best fit.  (Fixed: the previous code
    fit the decreasing form to the rising OOD curve, which pinned σc to the
    bounds.)

    The `np.all(ys < 0.5)` early return was removed.  It rejected
    every curve that saturates *below* half --- which is every *induced*-OOD
    curve, since those saturate at 0.31 (bare) down to 0.01 (the stack).  The
    rejection is what forced the paper's bare-ALD σc to be taken from the
    *total* rate (which does exceed 0.5, thanks to the start-OOD floor) while
    the figure drew it on the induced curve: a scale from one curve annotated
    onto another.  The amplitude form handles these curves; the amplitude
    bound was widened to 1e-4 to reach the stack's ~1%.
    """
    from scipy.optimize import curve_fit

    xs = np.asarray(sigmas, float)
    ys = np.clip(np.asarray(y, float), 1e-4, 1 - 1e-4)
    if len(xs) < 4 or np.all(ys > 0.5):
        return {"fit_ok": False, "sigma_c": None, "tau": None, "amplitude": None}
    best, best_err = None, np.inf
    for rising in (False, True):
        for with_amp in (False, True):
            def f(x, sc, tau, amp=None):
                s = 1.0 / (1.0 + np.exp((x - sc) / tau))
                s = s if not rising else 1.0 - s
                return amp * s if with_amp else s
            try:
                p0 = (float(xs[0] + 0.5 * (xs[-1] - xs[0])), 0.3)
                bounds = ([0.01, 0.01], [10.0, 10.0])
                if with_amp:
                    p0 = (*p0, float(min(max(ys.max(), 0.05), 0.95)))
                    bounds = ([0.01, 0.01, 1e-4], [10.0, 10.0, 1.0])
                (sc, tau, amp) = (None, None, None)
                popt, _ = curve_fit(f, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
                sc, tau = popt[0], popt[1]
                amp = popt[2] if with_amp else 1.0
                err = float(np.mean((f(xs, *popt) - ys) ** 2))
                if err < best_err:
                    best, best_err = (sc, tau, amp), err
            except Exception:
                pass
    if best is None:
        return {"fit_ok": False, "sigma_c": None, "tau": None, "amplitude": None}
    sc, tau, amp = best
    return {"fit_ok": True, "sigma_c": float(sc), "tau": float(tau),
            "amplitude": float(amp)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def analyze_subexp1(records: list[dict]) -> dict:
    """Per (σ, protection, sampler, nnp) cells + per-composition sigmoid fits."""
    cells: dict = {}
    for t in records:
        key = (float(t["sigma"]), t["protection"], t["sampler"], t["nnp"])
        cells.setdefault(key, []).extend(cands_of(t))
    out = {"cells": {}, "sigmoid": {}, "comp_sigma_c": {}}
    for (sigma, prot, sampler, nnp), cands in sorted(cells.items()):
        out["cells"][f"{sigma:g}|{prot}|{sampler}|{nnp}"] = cell_metrics(cands)
    # per (prot, sampler, nnp) sigmoid over the σ grid
    for (prot, sampler, nnp) in sorted({(p, s, n) for (_, p, s, n) in cells}):
        sig, ys = [], []
        for sigma in SIGMAS:
            m = out["cells"].get(f"{sigma:g}|{prot}|{sampler}|{nnp}")
            if m and m["n_traj"]:
                sig.append(sigma)
                ys.append(m["validity"])
        fit = fit_sigmoid(sig, ys)
        out["sigmoid"][f"{prot}|{sampler}|{nnp}"] = {
            "validity": {**fit, "n_sigma": len(sig)}}
        # OOD-spike sigmoid of the *total* rate, i.e. including the start-OOD
        # floor.  Kept because the M1 gate and the phase-1 summaries quote it,
        # but it is NOT the curve panel (a) of the sigma figure draws.
        sig, ys = [], []
        for sigma in SIGMAS:
            m = out["cells"].get(f"{sigma:g}|{prot}|{sampler}|{nnp}")
            if m and m["n_traj"]:
                sig.append(sigma)
                ys.append(m["ood_rate"])
        fit = fit_sigmoid(sig, ys)
        out["sigmoid"][f"{prot}|{sampler}|{nnp}"]["ood"] = {
            **fit, "n_sigma": len(sig)}
        # Sigmoid of the sampler-*induced* rate: the failure curve of Fig 1a.
        # This is the one the paper should quote as sigma_c, since the induced
        # rate is what the clean-start argument of Sec. 4.1 defines.  Its
        # amplitude is the quantity that says the failure curve is gone.
        sig, ys = [], []
        for sigma in SIGMAS:
            m = out["cells"].get(f"{sigma:g}|{prot}|{sampler}|{nnp}")
            if m and m["n_traj"]:
                sig.append(sigma)
                ys.append(m["induced_ood_rate"])
        fit = fit_sigmoid(sig, ys)
        out["sigmoid"][f"{prot}|{sampler}|{nnp}"]["induced"] = {
            **fit, "n_sigma": len(sig)}
    # per-composition σc (validity collapse), pooled over seeds
    comps = sorted({t["composition"] for t in records})
    n_atoms = {t["composition"]: t["n_atoms"] for t in records if "n_atoms" in t}
    for prot, sampler, nnp in sorted({(p, s, n) for (_, p, s, n) in cells}):
        for comp in comps:
            sig, ys = [], []
            for sigma in SIGMAS:
                key = f"{sigma:g}|{prot}|{sampler}|{nnp}"
                cc = [c for c in cells.get((sigma, prot, sampler, nnp), [])
                      if c.get("composition") == comp or c.get("formula") == comp]
                if cc:
                    m = cell_metrics(cc)
                    if m["n_traj"]:
                        sig.append(sigma)
                        ys.append(m["validity"])
            fit = fit_sigmoid(sig, ys)
            if fit["fit_ok"]:
                n = n_atoms.get(comp, np.nan)
                out["comp_sigma_c"][f"{comp}|{prot}|{sampler}|{nnp}"] = {
                    "sigma_c": fit["sigma_c"], "tau": fit["tau"],
                    "amplitude": fit["amplitude"], "n_atoms": n,
                    # d = 3N + 9: joint config-space dimensionality including
                    # the lattice DOF (paper Sub-exp 4, Corollary reliability)
                    "R_basin_A": float(fit["sigma_c"] * np.sqrt(3 * n + 9)) if np.isfinite(n) else None}
    return out


def analyze_subexp2(records: list[dict]) -> dict:
    levels = ["bare", "l1", "l1l2", "l1l2l3", "l1l4"]
    out = {"cells": {}}
    cands_by = {lv: [] for lv in levels}
    for t in records:
        lv = t.get("level")
        if lv in cands_by:
            cands_by[lv].extend(cands_of(t))
    for lv in levels:
        out["cells"][lv] = cell_metrics(cands_by[lv])
    return out


def analyze_smscan(records: list[dict]) -> dict:
    out = {"cells": {}}
    by = {}
    for t in records:
        by.setdefault((t["p"], t["d_safe"]), []).extend(cands_of(t))
    for (p, d), cands in sorted(by.items()):
        out["cells"][f"p{p:g}_d{d:g}"] = cell_metrics(cands)
    return out


def analyze_subexp5(records: list[dict]) -> dict:
    out = {"cells": {}}
    pairs = sorted({t["pair"] for t in records})
    for pair in pairs:
        by = {}
        for t in records:
            if t["pair"] != pair:
                continue
            tag = f"zbl" if t.get("A") is None else f"a{t['A']:g}_b{t['B']:g}"
            by.setdefault(tag, []).extend(cands_of(t))
        out["cells"][pair] = {tag: cell_metrics(c) for tag, c in sorted(by.items())}
    return out


def analyze_recheck(records: list[dict]) -> dict:
    out = {"cells": {}}
    by = {}
    for t in records:
        by.setdefault((t["sigma"], t["protection"], t["sampler"], t["nnp"]),
                      []).extend(cands_of(t))
    for (sigma, prot, sampler, nnp), cands in sorted(by.items()):
        out["cells"][f"{sigma:g}|{prot}|{sampler}|{nnp}"] = cell_metrics(cands)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "font.family": "DejaVu Sans",
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK_S, "ytick.color": INK_S,
        "axes.labelcolor": INK_P, "text.color": INK_P,
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def save(plt, name):
    for ext in ("pdf", "png"):
        plt.savefig(FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def fig1(plt, s1: dict, s2: dict, s1_records: list | None = None) -> None:
    """Figure 1, 4 panels (paper §6.1 headline).

    Panel (a) per the paper's Figure 1a spec: dual axis — left: validity
    rate for the three ALD conditions; right: OOD spike rate (ALD dashed,
    PF-ODE dotted in a lighter shade); dashed vertical line at the fitted
    σc for bare NNP; inset: d_min distribution bare vs. augmented (L1) at
    σ_max=1.0 (ALD), showing the augmented score's preventive effect on
    atom overlap (the validity threshold 0.5 Å is marked).
    """
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    # ---- (a) σ-failure curves: dual-axis validity / OOD + d_min inset ------
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax2 = ax.twinx()
    nnp = "mace"
    for pi, prot in enumerate(PROTECTIONS):
        # left axis: ALD validity_rate(σ)
        vy = []
        for sigma in SIGMAS:
            m = s1["cells"].get(f"{sigma:g}|{prot}|ald|{nnp}")
            vy.append(m["validity"] if m and m["n_traj"] else np.nan)
        ax.plot(SIGMAS, vy, "-", color=C[pi], lw=1.7, marker="o", ms=3.5,
                label=f"ALD {prot_label(prot)} validity", zorder=4)
        # right axis: OOD spike rate — ALD dashed, PF-ODE dotted (lighter)
        oa, op = [], []
        for sigma in SIGMAS:
            ma = s1["cells"].get(f"{sigma:g}|{prot}|ald|{nnp}")
            mp = s1["cells"].get(f"{sigma:g}|{ode_prot(prot)}|pfode|{nnp}")
            oa.append(ma["ood_rate"] if ma and ma["n_traj"] else np.nan)
            op.append(mp["ood_rate"] if mp and mp["n_traj"] else np.nan)
        ax2.plot(SIGMAS, oa, "--", color=C[pi], lw=1.2, zorder=3)
        ax2.plot(SIGMAS, op, ":", color=C[pi], lw=1.2, alpha=0.45, zorder=2)
    # dashed vertical line at the fitted σc (bare NNP validity collapse)
    fit = s1["sigmoid"].get("bare|ald|mace", {})
    if fit and fit["validity"]["fit_ok"]:
        sc = fit["validity"]["sigma_c"]
        ax.axvline(sc, ls=(0, (4, 3)), lw=1.0, color=INK_MUTED, zorder=1)
        ax.text(sc, 0.97, rf"$\sigma_c$={sc:.2f}", color=INK_S, fontsize=7,
                ha="center", va="top")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"$\sigma_{\max}$")
    ax.set_ylabel("validity rate (ALD)", color=INK_P)
    ax2.set_ylabel("OOD spike rate", color=INK_S)
    ax.set_ylim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="y", colors=INK_P)
    ax2.tick_params(axis="y", colors=INK_S)
    handles = [Line2D([], [], color=C[pi], lw=1.7, ls="-", marker="o", ms=3.5,
                      label=f"ALD {prot_label(prot)} validity")
               for pi, prot in enumerate(PROTECTIONS)]
    handles += [
        Line2D([], [], color=INK_S, lw=1.2, ls="--", label="OOD — ALD"),
        Line2D([], [], color=INK_S, lw=1.2, ls=":", alpha=0.45, label="OOD — PF-ODE"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", ncol=1)
    ax.set_title("(a) σ-failure curves", loc="left", fontsize=10, pad=6)

    # Inset: d_min distribution, ALD σ_max=1.0, bare vs. augmented (L1)
    if s1_records:
        db, dl = [], []
        for t in s1_records:
            if t.get("sampler") != "ald" or t.get("nnp") != "mace" \
                    or t.get("sigma") != 1.0:
                continue
            prot = t.get("protection")
            if prot not in ("bare", "l1"):
                continue
            for c in t.get("candidates", []):
                d = c.get("final", {}).get("d_min")
                if d is not None and np.isfinite(d):
                    (db if prot == "bare" else dl).append(float(d))
        if db and dl:
            inset = ax.inset_axes([0.56, 0.12, 0.40, 0.36])
            hi = max(max(db), max(dl), 1.0)
            bins = np.linspace(0.0, hi, 40)
            inset.hist(db, bins=bins, histtype="step", lw=1.2, color=C[0],
                       label="bare")
            inset.hist(dl, bins=bins, histtype="step", lw=1.2, color=C[1],
                       label="L1")
            inset.axvline(0.5, ls=":", lw=0.8, color=INK_MUTED)
            inset.set_xlim(0.0, None)
            inset.tick_params(labelsize=6)
            inset.legend(fontsize=6, frameon=False)
            inset.set_title(r"$d_{\min}$ @ $\sigma_{\max}$=1.0 (ALD)",
                            fontsize=6.5, loc="left")
    save(plt, "fig1a")

    # ---- (b) layer dissection bars at σ=1.0 (subexp 2) --------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    levels = ["bare", "l1", "l1l2", "l1l2l3", "l1l4"]
    surv = [1 - s2["cells"][lv]["ood_rate"] for lv in levels]
    n_prev, prev_surv = 0, 1.0
    for i, (lv, s) in enumerate(zip(levels, surv)):
        ax.bar(i, s, width=0.6, color=SEQ_BLUE[i], edgecolor="white",
               linewidth=0.5, zorder=3)
        ax.text(i, s + 0.02, f"{s*100:.0f}%", ha="center", va="bottom",
                fontsize=8, color=INK_P)
        gain = s - prev_surv
        if i > 0 and gain > 0.005:
            ax.annotate(f"+{gain*100:.0f} pt", (i - 0.5, (prev_surv + s) / 2),
                        ha="center", va="center", fontsize=7, color=INK_S,
                        xytext=(0, -9), textcoords="offset points")
        prev_surv = s
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([prot_label(lv) for lv in levels], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("trajectory survival")
    ax.set_title("(b) protection layers, ALD " + r"$\sigma_{\max}=1.0$",
                 loc="left", fontsize=10, pad=6)
    save(plt, "fig1b")

    # ---- (c) failure-mode composition at σ=1.0 (ALD) ----------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    from matplotlib.patches import Rectangle

    keys = ["clean", "t1", "force", "t2", "nan", "mixed"]
    names = {"clean": "clean", "t1": "Type I (overlap)", "force": "force >500",
             "t2": "Type II (E<−50)", "nan": "non-finite", "mixed": "mixed"}
    colors = dict(zip(keys, C[:len(keys)]))
    for pi, prot in enumerate(PROTECTIONS):
        m = s1["cells"].get(f"1|{prot}|ald|mace")
        if not m or not m["n_traj"]:
            continue
        cell_frac = m["buckets"]
        bottom = 0.0
        for k in keys:
            v = cell_frac[k]
            if v <= 0:
                continue
            ax.bar(pi, v, bottom=bottom, width=0.62, color=colors[k],
                   edgecolor="white", linewidth=0.5, zorder=3)
            bottom += v
        ax.text(pi, bottom + 0.01, f"{bottom*100:.0f}%", ha="center",
                fontsize=8, color=INK_P)
    ax.set_xticks(range(len(PROTECTIONS)))
    ax.set_xticklabels([prot_label(p) for p in PROTECTIONS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("trajectory fraction, ALD " + r"$\sigma_{\max}=1.0$")
    ax.legend([Rectangle((0, 0), 1, 1, color=colors[k]) for k in keys],
              [names[k] for k in keys], fontsize=7, loc="upper left",
              frameon=False)
    ax.set_title("(c) failure-mode composition", loc="left", fontsize=10, pad=6)
    save(plt, "fig1c")

    # ---- (d) PF-ODE robustness: OOD rate vs ALD, with <1% threshold -------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for label, (sampler, prot) in {
        "ALD bare": ("ald", "bare"), "PF-ODE bare": ("pfode", "bare"),
        "PF-ODE L1+L3": ("pfode", "l1l4"),
    }.items():
        ys = []
        for sigma in SIGMAS:
            m = s1["cells"].get(f"{sigma:g}|{prot}|{sampler}|mace")
            ys.append(m["ood_rate"] if m and m["n_traj"] else np.nan)
        idx = 0 if "PF" not in label else (1 if label == "PF-ODE bare" else 2)
        ax.plot(SIGMAS, ys, "-" if "PF" not in label else "--",
                color=C[idx], lw=1.6, marker="o", ms=3.5, label=label, zorder=3)
    ax.axhline(0.01, ls=(0, (3, 3)), lw=0.9, color=INK_MUTED)
    ax.text(5.15, 0.014, "1% target", color=INK_MUTED, fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"$\sigma_{\max}$")
    ax.set_ylabel("trajectory OOD rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(d) PF-ODE robustness", loc="left", fontsize=10, pad=6)
    save(plt, "fig1d")


def fig_sm(plt, sm: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ps = sorted({c.split("_")[0] for c in sm["cells"]})
    ds = sorted({c.split("_")[1] for c in sm["cells"]})
    surv = np.full((len(ps), len(ds)), np.nan)
    for i, p in enumerate(ps):
        for j, d in enumerate(ds):
            m = sm["cells"].get(f"{p}_{d}")
            if m and m["n_traj"]:
                surv[i, j] = 1 - m["ood_rate"]
    im = ax.imshow(surv, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    for i in range(len(ps)):
        for j in range(len(ds)):
            if np.isfinite(surv[i, j]):
                ax.text(j, i, f"{surv[i,j]*100:.0f}%", ha="center", va="center",
                        fontsize=8, color=INK_P)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([d.split("d")[1] for d in ds])
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels([p.split("p")[1] for p in ps])
    ax.set_xlabel(r"$d_{\mathrm{safe}}$ (Å)")
    ax.set_ylabel(r"SafetyMonitor $p$")
    ax.set_title("SafetyMonitor scan: survival, ALD L1-L4 " + r"$\sigma_{\max}=1.0$",
                 loc="left", fontsize=10, pad=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="survival")
    save(plt, "fig_sm_heatmap")


def fig_s5(plt, s5: dict) -> None:
    pairs = sorted(s5["cells"])
    nrow, ncol = len(pairs), 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.2, 2.1 * nrow),
                             constrained_layout=True)
    as_ = [1e2, 1e3, 1e4, 1e5, 1e6]
    bs = [1.0, 3.0, 10.0]
    for r, pair in enumerate(pairs):
        for col, (metric, cmap, vmin, vmax) in enumerate(
                [("ood_rate", "Reds", 0.0, 1.0), ("e_hull_mean_meV", "Blues", None, None)]):
            ax = axes[r, col]
            m = np.full((len(as_), len(bs)), np.nan)
            for i, a in enumerate(as_):
                for j, b in enumerate(bs):
                    cell = s5["cells"][pair].get(f"a{a:g}_b{b:g}")
                    if cell and cell["n_traj"]:
                        v = cell[metric]
                        m[i, j] = np.nan if v is None else v
            im = ax.imshow(m, cmap=cmap, aspect="auto", interpolation="nearest")
            zbl = s5["cells"][pair].get("zbl")
            zbl_v = zbl[metric] if zbl and zbl["n_traj"] else None
            for i in range(len(as_)):
                for j in range(len(bs)):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i,j]*100:.0f}" if metric == "ood_rate"
                                else f"{m[i,j]:.0f}", ha="center", va="center",
                                fontsize=7, color=INK_P)
            ax.set_xticks(range(len(bs)))
            ax.set_xticklabels([f"{b:g}" for b in bs])
            ax.set_yticks(range(len(as_)))
            ax.set_yticklabels([f"{a:.0e}" for a in as_])
            ax.set_title(f"{pair}: {metric}" + (f"  (ZBL ref {zbl_v})" if zbl_v is not None else ""),
                         fontsize=8)
            ax.grid(False)
            fig.colorbar(im, ax=ax)
    fig.suptitle("Pauli sensitivity A8: OOD rate and E_hull vs (A, B)", fontsize=11)
    save(plt, "fig_s5_pauli")


def fig_rc(plt, rc: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    sigmas = [0.5, 1.0, 2.0, 5.0]
    styles = {"bare": 0, "l1": 1, "l1l4": 2}
    for nnp_i, nnp in enumerate(["mace", "esen"]):
        for prot, ci in styles.items():
            ys = []
            for sigma in sigmas:
                m = rc["cells"].get(f"{sigma:g}|{prot}|ald|{nnp}")
                ys.append(m["ood_rate"] if m and m["n_traj"] else np.nan)
            ax.plot(sigmas, ys, "-" if nnp == "mace" else "--", color=C[ci],
                    lw=1.6, marker="o", ms=3.5,
                    label=f"{nnp.upper()} {prot_label(prot)}" if nnp == "mace"
                    else f"eSEN {prot_label(prot)}", zorder=3)
    pfys = []
    for sigma in sigmas:
        m = rc["cells"].get(f"{sigma:g}|bare|pfode|esen")
        pfys.append(m["ood_rate"] if m and m["n_traj"] else np.nan)
    ax.plot(sigmas, pfys, ":", color=C[3], lw=1.6, marker="s", ms=3.5,
            label="eSEN PF-ODE bare", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{\max}$")
    ax.set_ylabel("trajectory OOD rate (5-comp subset)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("eSEN recheck vs MACE", loc="left", fontsize=10, pad=6)
    save(plt, "fig_rc_esen")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def gate_table(s1: dict, s2: dict) -> list[str]:
    """M1 gate checks; returns markdown rows.

    Gates measure *sampler-induced* OOD (steps ≥ 1; the initial-condition
    floor is reported separately as start_ood_rate) — see
    cell_metrics docstring.  M1-2 tests the full four-layer stack (paper H1b:
    "the full four-layer architecture … reduces OOD spikes to near zero");
    the L1-only rate is reported for honesty (the plan's own n=5 validation,
    docs/experiment_plan_v1.md line 131, expected L1-only to be *worse* than
    bare at σ=1.0).
    """
    rows = []
    # M1-1: rising OOD-spike sigmoid of the bare ALD condition (the failure
    # curve of Fig 1a).  σc = the collapse scale where OOD spikes begin.
    fit = s1["sigmoid"].get("bare|ald|mace", {}).get("ood") or {}
    # M1-2: full-stack induced-OOD reduction vs bare at σ=1.0
    mb, ml = None, None
    if s1["cells"].get("1|bare|ald|mace", {}).get("n_traj"):
        mb = s1["cells"]["1|bare|ald|mace"]
    if s1["cells"].get("1|l1l4|ald|mace", {}).get("n_traj"):
        ml = s1["cells"]["1|l1l4|ald|mace"]
    if mb and ml and mb["induced_ood_rate"] is not None:
        red = 1 - ml["induced_ood_rate"] / mb["induced_ood_rate"] \
            if mb["induced_ood_rate"] > 0 else None
    else:
        red = None
    # M1-3: bare PF-ODE induced-OOD, max over the full σ scan (paper H1c)
    pf_oods = []
    for sigma in SIGMAS:
        m = s1["cells"].get(f"{sigma:g}|bare|pfode|mace")
        if m and m["n_traj"]:
            pf_oods.append(m["induced_ood_rate"])
    pf_ood_max = max(pf_oods) if pf_oods else None
    checks = [
        ("M1-1 sigmoid collapse", "σc∈[0.3,2.0], τ<1.5 (bare ALD, OOD curve)",
         None, fit),
        ("M1-2 L1–L4 OOD reduction", ">70% sampler-induced reduction at σ=1.0",
         red, None),
        ("M1-3 PF-ODE robustness", "<1% sampler-induced OOD at ALL σ (max over scan)",
         pf_ood_max, None),
    ]
    for name, crit, val, fit in checks:
        if val is not None:
            status = ("PASS" if (name.startswith("M1-2") and val > 0.7) or
                      (name.startswith("M1-3") and val < 0.01) else "FAIL")
            disp = f"{val*100:.1f}% (max over σ)" if name.startswith("M1-3") \
                else f"{val*100:.1f}%"
        elif fit and fit.get("fit_ok"):
            in_range = 0.3 <= fit["sigma_c"] <= 2.0 and fit["tau"] < 1.5
            status = "PASS" if in_range else "FAIL"
            disp = (f"σc={fit['sigma_c']:.2f}, τ={fit['tau']:.2f}"
                    + (f", A={fit.get('amplitude', 1):.2f}"
                       if "amplitude" in fit else ""))
        else:
            status, disp = "TBD", "data incomplete"
        rows.append(f"| {name} | {crit} | {disp} | {status} |")
    return rows


def write_report(s1, s2, sm, s5, rc, stats: dict) -> None:
    lines = []
    a = lines.append
    a("# Phase 1 report — completeness boundary verification (§6.1)")
    a("")
    a(f"_Generated {stats['time']} · subexp1 {stats['n_s1']}/6000 tasks "
      f"({stats['n_s1_pct']:.0f}%), subexp2 {stats['n_s2']}/1200, "
      f"sm {stats['n_sm']}/225, subexp5 {stats['n_s5']}/400, "
      f"recheck {stats['n_rc']}/400 tasks_")
    a("")
    a("## M1 gate evaluation")
    a("")
    a("| Gate | Criterion | Measured | Status |")
    a("|---|---|---|---|")
    a("\n".join(gate_table(s1, s2)))
    a("")
    a("## 1. σ-failure curves (sub-exp 1)")
    a("")
    a("Total trajectory OOD rate (any OOD step; includes the initial-condition"
      " floor).  Sampler-induced OOD (steps ≥ 1, clean start) drives the M1"
      " gates and is tabulated after this table.")
    a("")
    a("| σ | ALD bare | ALD L1 | ALD L1-L4 | PF-ODE bare | PF-ODE L1+L3 |")
    a("|---|---|---|---|---|---|")
    for sigma in SIGMAS:
        row = [f"{sigma:g}"]
        for sampler in SAMPLERS:
            for prot in (PROTECTIONS if sampler == "ald" else ["bare", "l1l4"]):
                m = s1["cells"].get(f"{sigma:g}|{prot}|{sampler}|mace")
                row.append(f"{m['ood_rate']*100:.1f}%" if m and m["n_traj"] else "—")
        a("| " + " | ".join(row) + " |")
    a("")
    a("Sampler-induced OOD rate (OOD events on steps ≥ 1, trajectory started"
      " clean):")
    a("")
    a("| σ | ALD bare | ALD L1 | ALD L1-L3 | PF-ODE bare | PF-ODE L1+L3 | start-OOD floor |")
    a("|---|---|---|---|---|---|---|")
    for sigma in SIGMAS:
        row = [f"{sigma:g}"]
        for sampler in SAMPLERS:
            for prot in (PROTECTIONS if sampler == "ald" else ["bare", "l1l4"]):
                m = s1["cells"].get(f"{sigma:g}|{prot}|{sampler}|mace")
                if m and m["n_traj"]:
                    row.append(f"{m['induced_ood_rate']*100:.1f}%")
                else:
                    row.append("—")
        m = s1["cells"].get(f"{sigma:g}|bare|ald|mace")
        row.append(f"{m['start_ood_rate']*100:.1f}%" if m and m["n_traj"] else "—")
        a("| " + " | ".join(row) + " |")
    a("")
    a("## 2. Layer dissection at σ=1.0 (sub-exp 2)")
    a("")
    a("| Level | OOD rate | induced OOD | validity | E_hull med (meV) | NFE med |")
    a("|---|---|---|---|---|---|")
    for lv in ["bare", "l1", "l1l2", "l1l2l3", "l1l4"]:
        m = s2["cells"][lv]
        eh = m["e_hull_med_meV"]
        nfe = m["nfe_med"]
        a(f"| {prot_label(lv)} | {m['ood_rate']*100:.1f}% | "
          f"{m['induced_ood_rate']*100:.1f}% | {m['validity']*100:.1f}% "
          f"| {eh if eh is None else round(eh,1)} | {nfe if nfe is None else round(nfe,0)} |")
    a("")
    a("## 2b. Lattice collapse (paper section 4.2)")
    a("")
    a("Collapse = Type I under a cell shrunk below half the reference volume —"
      " the collective failure mode of stress-driven lattice updates. Under the"
      " fixed-cell Phase 1 protocol it is structurally absent; the counter "
      "fires only in upd_lat runs (Phase 2 C1/C3/C8/C9/C11).")
    a("")
    a("| Level | collapse rate |")
    a("|---|---|")
    for lv in ["bare", "l1", "l1l2", "l1l2l3", "l1l4"]:
        m = s2["cells"][lv]
        cr = m.get("collapse_rate")
        a(f"| {prot_label(lv)} | {cr*100:.1f}%" if cr is not None else f"| {prot_label(lv)} | — |")
    a("")
    a("## 3. SafetyMonitor scan (sub-exp sm)")
    a("")
    a("Survival (1 − OOD rate), ALD L1-L4 σ=1.0 on 5 compositions:")
    a("")
    a("| p \\ d_safe | " + " | ".join(["0.7", "1.0", "1.2"]) + " |")
    a("|---|---|---|---|")
    for p in ["p1", "p2", "p3"]:
        # producer keys are f"p{p:g}_d{d:g}" with float d (no "d" prefix);
        # the old lookup added a literal "d", rendering "p1_dd0.7" and blanking the table
        cells = [sm["cells"].get(f"{p}_d{d:g}") for d in [0.7, 1.0, 1.2]]
        vals = []
        for c in cells:
            vals.append(f"{c['ood_rate']*100:.1f}%" if c and c["n_traj"] else "—")
        a(f"| {p[1]} | " + " | ".join(vals) + " |")
    a("")
    a("## 4. Pauli sensitivity (sub-exp 5)")
    a("")
    for pair in sorted(s5["cells"]):
        cells = s5["cells"][pair]
        a(f"**{pair}**: OOD rate — ZBL ref "
          f"{cells.get('zbl', {}).get('ood_rate', 0)}; "
          f"active A range (B=10): "
          f"{min(cells.get(f'a{a:g}_b10', {'ood_rate': 1})['ood_rate'] for a in [1e2,1e3,1e4,1e5,1e6]):.0%}–"
          f"{max(cells.get(f'a{a:g}_b10', {'ood_rate': 1})['ood_rate'] for a in [1e2,1e3,1e4,1e5,1e6]):.0%}")
    a("")
    a("## 5. eSEN recheck (sub-exp rc)")
    a("")
    a("| σ | eSEN bare ALD | eSEN L1 | eSEN L1-L4 | eSEN PF-ODE bare |")
    a("|---|---|---|---|---|")
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        row = [f"{sigma:g}"]
        for prot in ["bare", "l1", "l1l4"]:
            m = rc["cells"].get(f"{sigma:g}|{prot}|ald|esen")
            row.append(f"{m['ood_rate']*100:.1f}%" if m and m["n_traj"] else "—")
        m = rc["cells"].get(f"{sigma:g}|bare|pfode|esen")
        row.append(f"{m['ood_rate']*100:.1f}%" if m and m["n_traj"] else "—")
        a("| " + " | ".join(row) + " |")
    a("")
    a("## 6. Recommended production configuration")
    a("")
    a("_Filled once sub-exp 1 completes; see summary.json `sigmoid` + `cells`._")
    a("")
    a("## 7. Figures")
    a("")
    a("`results/phase1/analysis/figures/fig1a.pdf` … `fig1d.pdf`, "
      "`fig_sm_heatmap.png`, `fig_s5_pauli.png`, `fig_rc_esen.png`")
    Path(REPO / "docs" / "phase1_report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--subexp", default=None, help="only analyze this subexp (1/2/sm/5/rc)")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="include pre-fix files (distorted initial cells, "
                         "see results/phase1/legacy_pre_cholesky/); only "
                         "for archaeology, never for paper numbers")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the analysis artefacts even if the post-fix "
                         "re-run is incomplete (diagnostics only; the numbers "
                         "must not be cited)")
    args = ap.parse_args()
    global ALLOW_LEGACY
    ALLOW_LEGACY = args.allow_legacy

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    import datetime

    s1r = load_subexp("subexp1") if args.subexp in (None, "1") else []
    s2r = load_subexp("subexp2") if args.subexp in (None, "2") else []
    smr = load_subexp("smscan") if args.subexp in (None, "sm") else []
    s5r = load_subexp("subexp5") if args.subexp in (None, "5") else []
    rcr = load_subexp("recheck") if args.subexp in (None, "rc") else []
    s1 = analyze_subexp1(s1r) if args.subexp in (None, "1") else {}
    s2 = analyze_subexp2(s2r) if args.subexp in (None, "2") else {}
    sm = analyze_smscan(smr) if args.subexp in (None, "sm") else {}
    s5 = analyze_subexp5(s5r) if args.subexp in (None, "5") else {}
    rc = analyze_recheck(rcr) if args.subexp in (None, "rc") else {}

    stats = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_s1": len(s1r), "n_s2": len(s2r), "n_sm": len(smr),
        "n_s5": len(s5r), "n_rc": len(rcr),
    }
    # Progress against the task matrix, not a hardcoded cell count: this read
    # `/ 60.0` (the 6-arm ladder) and printed 116.7% once the l1l2l3 arm was
    # backfilled and the l3 arm added (80 cells).
    stats["n_s1_pct"] = 100.0 * stats["n_s1"] / max(
        len(_rp.build_tasks("1", next((r.get("nnp", "mace") for r in s1r),
                                      "mace"))), 1)

    # never overwrite the analysis artefacts from a partial run.
    # The legacy trees were moved into legacy_pre_cholesky/ while the
    # bug-forced re-run runs for days, so a premature analysis pass -- on an
    # empty tree, or mid-flight with a third of the tasks done -- would
    # otherwise overwrite summary.json and the figures with a truncated data
    # set that still *looks* like a finished analysis.
    # The expected file counts come from the task matrix itself (run_phase1),
    # so a protocol edit cannot leave a stale constant behind here.
    se_of = {"n_s1": "1", "n_s2": "2", "n_sm": "sm", "n_s5": "5", "n_rc": "rc"}
    nnp = next((r.get("nnp", "mace") for r in s1r), "mace")
    missing = {}
    for k, se in se_of.items():
        if args.subexp not in (None, se):
            continue
        exp = len(_rp.build_tasks(se, nnp))
        if stats[k] < exp:
            missing[k] = (stats[k], exp)
    if missing and not args.allow_partial:
        print("REFUSING to write analysis artefacts: the post-fix re-run is "
              f"incomplete under {OUT}/")
        for k, (got, exp) in missing.items():
            print(f"  {k[2:]:3} {got:5d} / {exp} tasks"
                  f"{'  (not started)' if got == 0 else ''}")
        print(f"  (legacy pre-fix cells are quarantined in {LEGACY_DIR}/; "
              "the re-run has not fully landed yet)")
        print("  re-run with --allow-partial to analyse a partial data set "
              "(diagnostics only -- do NOT cite the result)")
        raise SystemExit(2)

    # WARNING -- `--subexp N` is a CLOBBER, not a merge.  Every section other
    # than N is `{}` in this dict (see the s2 = ... if args.subexp in (None,
    # "2") else {} block above), so `--subexp 1` writes 7000 live sub-exp-1
    # records alongside `"subexp2": {}` and silently destroys what the last
    # full pass put there.  To regenerate one section, run the FULL pass (no
    # --subexp): all five trees are complete, so the gate below passes, and
    # that is the only way to come out with a whole artifact.
    summary = {"stats": stats, "subexp1": s1, "subexp2": s2, "smscan": sm,
               "subexp5": s5, "recheck": rc}
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=1))

    if not args.no_figs:
        plt = setup_mpl()
        if s1 and s2:
            fig1(plt, s1, s2, s1_records=s1r)
        if sm and sm["cells"]:
            fig_sm(plt, sm)
        if s5 and s5["cells"]:
            fig_s5(plt, s5)
        if rc and rc["cells"]:
            fig_rc(plt, rc)

    write_report(s1, s2, sm, s5, rc, stats)
    n_traj = sum(v["n_traj"] for v in s1["cells"].values())
    print(f"analyzed: s1 {stats['n_s1']} tasks/{n_traj} traj, s2 {stats['n_s2']}, "
          f"sm {stats['n_sm']}, s5 {stats['n_s5']}, rc {stats['n_rc']}")
    print(f"summary -> results/phase1/analysis/summary.json, report -> docs/phase1_report.md")


if __name__ == "__main__":
    main()
