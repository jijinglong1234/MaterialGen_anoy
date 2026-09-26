"""Sub-exp 1 full-fleet analysis (post mini-legacy re-run).

Data: results/phase1/subexp1/ — 6000 files x 20 cands = 120000 trajectories,
100% under the frozen protocol.  The 300 mini-era files
(5 probe sigmas x RECHECK x {42,123}, 10 cands) were re-run
(see results/phase1/subexp1_mini_legacy/).

Column order is STANDARD: [ALD bare, ALD l1, ALD l1l4 | PF bare, PF l1, PF l1l4]
(lesson: the first draft interleaved [ALD x, PF x] pairs and
mislabeled the header).

Definitions (frozen, same as verify_v3_cap.py run_cell):
  OOD mask 0b010011 (d_min<0.5 | max|F_NNP|>500 | |det|>10x)
  induced = share of ALL trajectories that started clean and saw OOD at any
            step>=1 (denominator = all traj, NOT the clean-start subset; see
            agg() -- the clean-start-conditional share is ~1.5x higher)
  total   = share of ALL trajectories with OOD at any step (incl. step 0)
  dwell   = share of accepted steps with d_min < 1.3 A
  deep    = share of accepted steps with d_min < 0.5 A
  e_hull  = median final formation-energy-above-hull (eV/atom)
  validity= share of candidates with a valid final structure
  cap     = share of trajectories hitting the NFE cap (nfe >= 1000)

Gates:
  M1-1 sigma_c sigmoid (bare ALD total): sigma_c in [0.3, 2.0], tau < 1.5
  M1-2 L1-L4 OOD reduction > 70% at sigma=1 (l1l4 vs bare)
  M1-3 PF-ODE absolute < 1% (dual-report with the relative claim)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(_ROOT, "results", "phase1", "subexp1")
SUBEXP2 = os.path.join(_ROOT, "results", "phase1", "subexp2")

# shared loader (metric conventions + the pre-fix E_hull units fix)
import importlib.util as _ilu

# analyze_phase1 imports run_phase1, which imports materialgen -- and the
# importlib load below runs that whole chain.  Without the repo root on the
# path the first `python scripts/plot_subexp1.py` of a session dies with
# "No module named 'materialgen'" before it draws anything.
sys.path.insert(0, _ROOT)

_spec = _ilu.spec_from_file_location("ap1", os.path.join(_ROOT, "scripts",
                                                         "analyze_phase1.py"))
ap1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(ap1)

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
ARMS = ["bare", "l1", "l1l4"]
SAMP = ["ald", "pfode"]
# the C8 layer set {l1,l2,l3} was backfilled over the whole sigma
# ladder on ALD only (results/phase1/subexp1/s<sigma>_l1l2l3_ald/).  It is the
# arm the Phase-2 grid carries, so it belongs in the ladder table next to the
# full stack.  There is no PF-ODE l1l2l3 dir: L2 (density-adaptive noise) and
# L4 have no PF-ODE counterpart, so on that arm the C8 set is L1+L3 and the
# nearest existing columns are bare / l1.
# L3 alone (the reflecting constraint, no L1 caps and no L2 noise)
# was added over the same ladder.  Until it existed every protected
# ladder arm bundled L1+L3, so nothing separated the analytic wall from the
# constraint it complements.  ALD only, same reason as l1l2l3.
ALD_ONLY = ["l1l2l3", "l3"]
# The 12-subset layer grid at sigma = 1.0 (results/phase1/subexp2, ALD only, the
# "doc-12" grid behind Table tab:layer-dissection).  It is the ONLY place the
# off-ladder subsets {2}, {3}, {1,2}, {1,3} exist: the sigma ladder backfilled
# l1l2l3 alone, so none of them have a noise scan.  {2,3} was never run at all
# (the grid covers 12 of the 16 subsets).
DOC12 = ["bare", "l1", "l2", "l3", "l4", "l1l2", "l1l3", "l1l2l3",
         "l1l2l4", "l1l3l4", "l2l3l4", "l1l4"]
DOC12_SUBSET = {
    "bare": frozenset(),           "l1": frozenset({1}),
    "l2": frozenset({2}),          "l3": frozenset({3}),
    "l4": frozenset({4}),          "l1l2": frozenset({1, 2}),
    "l1l3": frozenset({1, 3}),     "l1l2l3": frozenset({1, 2, 3}),
    "l1l2l4": frozenset({1, 2, 4}), "l1l3l4": frozenset({1, 3, 4}),
    "l2l3l4": frozenset({2, 3, 4}), "l1l4": frozenset({1, 2, 3, 4}),
}
OOD_MASK = 0b010011
DWELL = 1.3
DEEP = 0.5
NFE_CAP = 1000
# Standard column order: ALD bare, l1, l1l2l3, l1l4 | PF bare, l1, l1l4
COLS = ([(a, "ald") for a in ARMS + ALD_ONLY] + [(a, "pfode") for a in ARMS])
HDRS = ([f"ALD {a}" for a in ARMS + ALD_ONLY]
        + [f"PF {a}" for a in ARMS])


def agg(cands: list) -> dict:
    clean = [c for c in cands if not (c["ood_bitmask"][0] & OOD_MASK)]
    induced_k = sum(any(m & OOD_MASK for m in c["ood_bitmask"][1:]) for c in clean)
    total_k = sum(any(m & OOD_MASK for m in c["ood_bitmask"]) for c in cands)
    hists = [h[1:] for h in (c["d_min_hist"] for c in cands) if len(h) > 1]
    steps = np.concatenate(hists) if hists else np.array([])
    # valid-only, to match the cell_metrics convention of analyze_phase1 /
    # analyze_subexp1_l1l4 (the numbers quoted in the paper: "median of valid
    # structures").  Panel (c) of the sigma figure had been
    # pooling invalid candidates as well, which disagreed with the table.
    eh = [c["final"]["e_hull"] for c in cands
          if c.get("valid", True) and c["final"]["e_hull"] is not None]
    nfes = np.array([c["nfe"] for c in cands]) if cands else np.array([])
    valid = [c for c in cands if c.get("valid", True)]
    return {
        # Denominator = ALL trajectories.  This is the analyze_phase1.cell_metrics
        # and analyze_doc12 convention, and it is what the paper's sub-experiment-1
        # caption states verbatim ("clean-start trajectories with an OOD event at
        # steps >= 1, normalized by all trajectories").  Dividing by len(clean)
        # instead gives the conditional P(OOD | clean start), which reads ~1.5x
        # higher (50.9% vs 33.0% for bare ALD at sigma = 1) and is not comparable
        # with the start-OOD floor, which IS normalized by all trajectories.
        # this function used the clean-start denominator until now.
        "induced": induced_k / max(len(cands), 1),
        "induced_clean_cond": induced_k / max(len(clean), 1),
        "induced_k": induced_k,
        "n_clean": len(clean),
        "total": total_k / max(len(cands), 1),
        "total_k": total_k,
        "n": len(cands),
        "dwell": float((steps < DWELL).mean()) if steps.size else 0.0,
        "deep": float((steps < DEEP).mean()) if steps.size else 0.0,
        "e_hull": float(np.median(eh)) if eh else None,
        "validity": len(valid) / max(len(cands), 1),
        "cap": float((nfes >= NFE_CAP).mean()) if nfes.size else 0.0,
    }


LIVE_GEN = "cholesky-fix"


def _load_cell(tag: str, base: str = BASE) -> list:
    """Candidates of one cell directory, refusing anything off the live tree.

    results/phase1/README.md splits the tree into two eras: the post-fix
    Cholesky re-run (live, referenceable) and the pre-fix trajectories
    (legacy, run on sigma-independently distorted initial cells, which no
    analysis-layer correction can undo).  Every live payload stamps
    `init_cell_gen`; a payload without it is legacy by definition.  Failing
    loudly here keeps the two eras from being pooled into one figure.
    """
    cands = []
    for p in sorted(glob.glob(os.path.join(base, tag, "*.json"))):
        task = json.load(open(p))
        gen = task.get("init_cell_gen")
        if gen != LIVE_GEN:
            raise RuntimeError(
                f"{p}: init_cell_gen={gen!r}, expected {LIVE_GEN!r} -- legacy "
                "pre-Cholesky payload (see results/phase1/README.md)")
        # ap1.cands_of also applies the pre-fix E_hull units correction
        # (per-composition /n_fu) so that the e_hull column below is in eV/atom.
        cands.extend(ap1.cands_of(task))
    return cands


def load_all() -> dict:
    """T[(sigma, arm, sampler)] -> agg dict."""
    T = {}
    for s in SIGMAS:
        for a in ARMS:
            for smp in SAMP:
                T[(s, a, smp)] = agg(_load_cell(f"s{s:g}_{a}_{smp}"))
        for a in ALD_ONLY:
            T[(s, a, "ald")] = agg(_load_cell(f"s{s:g}_{a}_ald"))
    return T


def load_doc12() -> dict:
    """tag -> agg for the 12-subset sigma=1.0 layer grid (ALD, subexp2).

    Read from the live tree rather than analysis/doc12_table.json so a stale
    artifact cannot leak into a figure; the two agree cell-for-cell.
    """
    return {c: agg(_load_cell(c, SUBEXP2)) for c in DOC12}


def show(T: dict, key: str, fmt: str = "{:.1%}") -> None:
    """fmt is a proportion format ({:.1%} = 63.0%); pass "{:.3f}" for units."""
    print(f"--- {key} ---")
    print("sigma | " + "   ".join(f"{h:>8}" for h in HDRS))
    for s in SIGMAS:
        row = [T[(s, a, smp)][key] for (a, smp) in COLS]
        print(f"{s:>5} | " + "   ".join(fmt.format(v) for v in row))
    print()


def main():
    T = load_all()
    show(T, "induced")
    show(T, "total")
    show(T, "dwell")
    show(T, "deep")
    show(T, "e_hull", "{:.3f}")
    show(T, "validity")
    show(T, "cap")

    # ---- M1-1: sigma_c sigmoid on bare-ALD total OOD ----
    from scipy.optimize import curve_fit
    xs = np.array(SIGMAS)
    ys = np.array([T[(s, "bare", "ald")]["total"] for s in SIGMAS])
    f = lambda x, A, sc, tau: A / (1 + np.exp(-(x - sc) / tau))
    try:
        (A, sc, tau), _ = curve_fit(f, xs, ys, p0=[0.7, 0.5, 0.1], maxfev=20000)
        print(f"M1-1 sigmoid (bare ALD total): A={A:.3f}, sigma_c={sc:.3f}, tau={tau:.3f} "
              f"-> {'PASS' if 0.3 <= sc <= 2.0 and tau < 1.5 else 'FAIL'}")
    except Exception as e:
        print("M1-1 fit failed:", e)

    # ---- M1-2: reduction > 70% at sigma=1 ----
    for smp in SAMP:
        b = T[(1.0, "bare", smp)]["induced"]
        for a in ("l1l4",) + (("l1l2l3",) if smp == "ald" else ()):
            l = T[(1.0, a, smp)]["induced"]
            print(f"M1-2 {smp} {a} @sigma=1: {b:.1%} -> {l:.1%} (cut {(1 - l/b):.1%}) "
                  f"-> {'PASS' if (1 - l/b) > 0.70 else 'FAIL'}")
    # PF with L1 only (paper's PF-ODE claim uses L1')
    b, l = T[(1.0, "bare", "pfode")]["induced"], T[(1.0, "l1", "pfode")]["induced"]
    print(f"M1-2 PF l1-only @sigma=1: {b:.1%} -> {l:.1%} (cut {(1 - l/b):.1%})")
    # Diagnostic, NOT a gate: L3 alone against the bundled L1+L3 arm, to show
    # how much of the ladder's reduction the constraint carries by itself.
    b = T[(1.0, "bare", "ald")]["induced"]
    for a in ("l3", "l1l2l3"):
        l = T[(1.0, a, "ald")]["induced"]
        print(f"M1-2 ALD {a} @sigma=1: {b:.1%} -> {l:.1%} (cut {(1 - l/b):.1%})")

    # ---- M1-3: PF absolute < 1% ----
    for a in ARMS:
        v = max(T[(s, a, "pfode")]["induced"] for s in SIGMAS)
        print(f"M1-3 PF {a} max induced over sigmas: {v:.1%} -> {'PASS' if v < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()
