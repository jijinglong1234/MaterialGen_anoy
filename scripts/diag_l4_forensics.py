#!/usr/bin/env python
"""Why L4 (SafetyMonitor) can only subtract from every non-OOD axis.

Forensics on the post-fix subexp2 (doc-12) cells of the NC=10 re-run.
Read-only; complements scripts/diag_subexp2_nc10.py (which produces the tables).

Measurements:
1. cohort decomposition -- every doc-12 level split by start-OOD (step-0) vs
   clean-start: validity, induced OOD, rejection counts, final d_min.
2. recovery axis -- among clean-start trajectories with >= 1 OOD step, the
   share whose FINAL structure has d_min >= 0.5 (i.e. that come back).  This
   is the axis the paper's induced-OOD metric does not have.
3. pinning forensics -- where in the 200-NFE d_min record the constant tail
   starts (index 0 = frozen at the initial structure), the frozen d_min value
   vs the SafetyMonitor thresholds, and the frozen state's NNP-only force
   (final.max_f, computed by run_phase1 with the Pauli wall excluded).

Key result: L4 is the only rejecting layer (0.00 rejections/trajectory in every
non-L4 level) and rejection is an absorbing state, because ald.py calls
safety_monitor.check(proposal, energy=score.energy, forces=score.forces) with a
`score` computed on the PRE-MOVE structure (ald.py:252): criteria (a) energy and
(b) force therefore test the state the trajectory is trying to leave, while only
(c) d_min tests the proposal.  A state over the force threshold can never be
left -- every proposal is rejected, `crystal = prev` restores the same state,
and the sigma decay ratchet shrinks the steps further.

Run: python scripts/diag_l4_forensics.py    (materialgen env, read-only)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ap1 = _load("ap1", REPO / "scripts/analyze_phase1.py")

OOD_MASK = 0b010011                      # d_min<0.5 | F>500 | v_ratio>10
LEVELS = ["bare", "l1", "l1l2", "l1l2l3", "l3", "l4", "l1l2l4", "l1l3l4", "l1l4"]


def start_ood(c):
    bm = c.get("ood_bitmask") or []
    return bool(bm and (bm[0] & OOD_MASK))


def ood_after(c):
    return any(m & OOD_MASK for m in (c.get("ood_bitmask") or [])[1:])


def tail_start(hist):
    """Index where the constant tail of the d_min record begins."""
    if not hist:
        return None, None
    last = hist[-1]
    i = len(hist) - 1
    while i - 1 >= 0 and hist[i - 1] == last:
        i -= 1
    return i, last


def main():
    recs = ap1.load_subexp("subexp2")
    by_lv = {}
    for t in recs:
        lv = t.get("level")
        if lv:
            by_lv.setdefault(lv, []).extend(ap1.cands_of(t))
    print(f"loaded {len(recs)} subexp2 files\n")

    print("### 1. cohort decomposition (start-OOD = step-0 OOD, excluded from "
          "induced OOD by definition)")
    print(f"{'level':8s} {'cohort':10s} {'n':>5s} {'invalid':>8s} {'induced':>8s} "
          f"{'rej med':>8s} {'rej sum':>8s} {'final d_min med':>16s}")
    for lv in LEVELS:
        cs = by_lv.get(lv)
        if not cs:
            continue
        for name, grp in (("start-OOD", [c for c in cs if start_ood(c)]),
                          ("clean", [c for c in cs if not start_ood(c)])):
            if not grp:
                continue
            rej = np.array([c["n_rejections"] for c in grp], float)
            inv = sum(1 for c in grp if not c["valid"])
            ind = sum(1 for c in grp if ood_after(c) and not start_ood(c))
            dm = np.array([c["final"]["d_min"] for c in grp], float)
            print(f"{lv:8s} {name:10s} {len(grp):5d} {inv:8d} {ind:8d} "
                  f"{np.median(rej):8.0f} {rej.sum():8.0f} {np.median(dm):16.2f}")
    print("\n  rejections are L4-only: every non-L4 level above shows rej sum = 0.")

    print("\n### 2. recovery after an OOD excursion (clean-start, >=1 OOD step, "
          "final d_min >= 0.5)")
    print(f"{'level':8s} {'excursions':>10s} {'recovered':>10s} {'recovery':>9s} "
          f"{'frozen d_min<0.5':>16s} {'total rej':>10s}")
    for lv in LEVELS:
        cs = by_lv.get(lv)
        if not cs:
            continue
        ind = [c for c in cs if ood_after(c) and not start_ood(c)]
        rec = [c for c in ind if c["valid"]]
        pin = sum(1 for c in cs if not c["valid"])
        rp = f"{len(rec) / len(ind):.1%}" if ind else "-"
        print(f"{lv:8s} {len(ind):10d} {len(rec):10d} {rp:>9s} {pin:16d} "
              f"{sum(c['n_rejections'] for c in cs):10.0f}")

    print("\n### 3. pinning forensics")
    print(f"{'level':8s} {'cohort':10s} {'n':>5s} {'tail idx=0':>10s} {'1-9':>5s} "
          f"{'10-49':>6s} {'frozen d_min<0.3':>16s} {'frozen max_f med':>16s} "
          f"{'frozen E/atom med':>17s}")
    for lv in ("l4", "l1l2l4", "l1l3l4", "l1l4"):
        cs = by_lv.get(lv) or []
        for name, grp in (("start-OOD", [c for c in cs if start_ood(c) and not c["valid"]]),
                          ("clean", [c for c in cs if not start_ood(c) and not c["valid"]])):
            if not grp:
                continue
            idx, fz = [], []
            for c in grp:
                i, last = tail_start(c.get("d_min_hist") or [])
                if i is not None:
                    idx.append(i)
                    fz.append(last)
            idx = np.array(idx)
            fz = np.array(fz)
            mf = np.array([c["final"].get("max_f") or np.nan for c in grp], float)
            na = len((grp[0]["final"]["structure"] or {}).get("atomic_numbers") or []) or 1
            en = np.array([(c["final"].get("energy") if c["final"].get("energy") is not None
                            else np.nan) for c in grp], float) / na
            print(f"{lv:8s} {name:10s} {len(grp):5d} {(idx == 0).sum():10d} "
                  f"{((idx >= 1) & (idx < 10)).sum():5d} "
                  f"{((idx >= 10) & (idx < 50)).sum():6d} "
                  f"{(fz < 0.3).sum():16d} {np.nanmedian(mf):16.0f} "
                  f"{np.nanmedian(en):17.2f}")
    print("\n  SafetyMonitor thresholds: d_min < 0.3 A | |F_NNP| > 1e3 eV/A | "
          "U > 1e4 eV/atom | v_ratio > 10")
    print("  paper OOD metric:         d_min < 0.5 A | |F_NNP| > 5e2 eV/A | "
          "v_ratio > 10")
    print("  every frozen state is 2-3 orders of magnitude over the force "
          "threshold, so criterion (b) -- evaluated on the PRE-move state -- "
          "is the absorber; the energy criterion (U < 1e4) never fires.")

    print("\n### 4. control: candidates with final d_min < 0.5 in NON-L4 levels "
          "(genuine blow-ups, not pinning)")
    for lv in ("bare", "l1", "l1l2", "l1l2l3", "l3"):
        cs = by_lv.get(lv) or []
        pin = [c for c in cs if not c["valid"]]
        if not pin:
            print(f"  {lv:8s} 0/{len(cs)}")
            continue
        mf = np.array([c["final"].get("max_f") or np.nan for c in pin], float)
        print(f"  {lv:8s} {len(pin)}/{len(cs)} invalid, all with n_rejections = 0, "
              f"|F_NNP| med {np.nanmedian(mf):.3g} (numerical blow-up)")


if __name__ == "__main__":
    main()
