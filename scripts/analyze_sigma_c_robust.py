"""Robust per-composition collapse scale sigma_c, bare score under ALD.

Written to answer two reviewer objections to the published Table 9
(`tab:basin-radius`) and the ">17x" spread claim in the paper's Sec. 4.2.

  (1) SIZE CONFOUND.  `comp_sigma_c.py` defines sigma_c by the per-step
      collision rate crossing an *absolute* threshold (0.005).  A larger cell
      has more pairs, hence more chances to be within 0.5 A per step, hence
      reaches any fixed rate at a smaller sigma.  Across the 20 MP-20
      compositions the pair count spans 1-120 (2-16 atoms), wider than the
      spread the crossing is used to claim, and the rank correlation between
      the published sigma_c and cell size is rho ~ -0.8 (p < 1e-4).

  (2) ARBITRARY CONSTANTS.  The crossing needs a rule for cells whose rate is
      exactly 0; the published script floors the rate at 1e-12, an
      undocumented choice that moves the smallest sigma_c by ~25%.

Two definitions are reported side by side:

  (a) CROSS -- the published definition (log-log interpolation of the running
      maximum of the rate to the fixed threshold), kept so the old numbers
      stay reproducible, now at three zero-floors and under the paper's own
      clean-start, steps >= 1 accounting.
  (b) MIDPOINT -- the scale of a three-parameter saturating sigmoid fitted in
      log sigma.  A per-composition multiplicative chance factor cancels
      exactly here (rescaling the rate rescales the amplitude A and leaves the
      midpoint), so this estimator is size-invariant by construction: a
      residual size correlation under (b) means size changes the shape of the
      curve, not merely the number of chances.  Reported only when the
      midpoint falls inside the scanned range; otherwise the composition is
      censored rather than extrapolated.

Both carry cluster bootstrap CIs resampling *trajectories* (the 50 per cell are
the independent units; the ~10,000 steps inside them are not), plus a
threshold-free statistic: the collision rate at the single fixed condition
sigma = 5.0, raw and per-pair.

Two departures from `comp_sigma_c.py`'s accounting, both required by the
paper's own OOD convention (Sec. 4.1): the rate is taken over trajectories that
started clean (step 0 not OOD) and over steps >= 1 only.  The published script
pools every trajectory and includes step 0, whose distances are properties of
the initial cell rather than of the sampler.

Usage:
  python scripts/analyze_sigma_c_robust.py [--root results/phase1/subexp1]
      [--arm bare_ald] [--boot 400] [--out results/phase1/analysis/sigma_c_robust.json]
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
D_OOD = 0.5          # A, the OOD distance threshold (analyze_phase1)
THRESHOLD = 0.005    # published CROSS threshold: one collision per 200 steps
FLOORS = [1e-12, 1e-4, 1e-3]   # zero-cell floors scanned for the CROSS estimator
REF_SIGMA = 5.0      # fixed condition for the threshold-free statistic


# ---------------------------------------------------------------------------
# Loading: one Cell = one (composition, sigma) = 50 trajectories
# ---------------------------------------------------------------------------

class Cell:
    """Pre-digested payload of one (composition, sigma) cell.

    Per-trajectory OOD indicator sequences are kept because the published
    "collision rate" counts *every* step spent below 0.5 A, so it mixes two
    different things: how often a trajectory enters the OOD state, and how
    long it then stays there.  Both are separated here (`entry`, `residence`).
    """

    __slots__ = ("sigma", "n_atoms", "clean", "seqs")

    def __init__(self, sigma, n_atoms, clean, seqs):
        self.sigma = sigma
        self.n_atoms = n_atoms              # (m,) int
        self.clean = clean                  # (m,) bool, step 0 not OOD
        self.seqs = seqs                    # list of (T,) bool, steps >= 1

    def __len__(self):
        return len(self.n_atoms)

    def stats(self, idx=None):
        """(raw rate, per-pair rate, entry hazard, residence, entries, steps,
        mean n_pairs) over clean-start trajectories."""
        n, clean, seqs = self.n_atoms, self.clean, self.seqs
        if idx is not None:
            n, clean = n[idx], clean[idx]
            seqs = [self.seqs[i] for i in idx]
        n_atoms, steps, events, entries, dwell = [], 0, 0, 0, 0
        for na, cl, s in zip(n, clean, seqs):
            if not cl or not na or na <= 1 or s.size == 0:
                continue
            e = int(np.count_nonzero(s))
            # an entry is a step at which the trajectory is OOD and the
            # previous step was not (the state at step 1 is an entry if it is
            # OOD, because step 0 is required clean by construction)
            ent = int(np.count_nonzero(s & ~np.concatenate(([False], s[:-1]))))
            steps += s.size
            events += e
            entries += ent
            dwell += e
            n_atoms.append(int(na))
        if steps == 0 or not n_atoms:
            return (np.nan,) * 7
        n_pairs = float(np.mean([a * (a - 1) / 2.0 for a in n_atoms]))
        raw = events / steps
        haz = entries / steps
        # residence: mean OOD steps per entry (0 when nothing ever entered)
        res = (dwell / entries) if entries else 0.0
        return raw, raw / n_pairs, haz, res, entries, steps, n_pairs


def load_cell(comp, sigma, root, arm, prefix=""):
    # `prefix` reads the step-size control tree, whose cells are tagged
    # "m4_s<sigma>_<arm>" so that no existing glob can mistake them for scan
    # output (run_phase1.build_tasks, subexp "fine").
    d = os.path.join(root, f"{prefix}s{sigma:g}_{arm}")
    n_atoms, clean, seqs = [], [], []
    for f in sorted(glob.glob(os.path.join(d, f"{comp}_seed*.json"))):
        with open(f) as fh:
            task = json.load(fh)
        n = task.get("n_atoms")
        for c in task.get("candidates", []):
            hist = c.get("d_min_hist") or []
            if not hist or n is None:
                continue
            # step 0 is the initial state: this is the clean-start test of
            # Sec. 4.1, evaluated in the same distance channel the collision
            # rate is defined on.
            d0 = hist[0]
            arr = np.asarray([x for x in hist[1:]
                              if x is not None and np.isfinite(x)], float)
            n_atoms.append(int(n))
            clean.append(d0 is not None and np.isfinite(d0) and d0 >= D_OOD)
            seqs.append(arr < D_OOD)
    return Cell(sigma, np.array(n_atoms, int), np.array(clean, bool), seqs)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _cross(sig, rate, threshold=THRESHOLD, floor=FLOORS[0]):
    """First crossing of `threshold` by the running max of `rate`, with `floor`
    standing in for cells whose rate is exactly 0 (published definition)."""
    sig, rate = np.asarray(sig, float), np.maximum.accumulate(np.asarray(rate, float))
    ok = np.isfinite(rate)
    if not ok.any():
        return np.nan
    sig, rate = sig[ok], rate[ok]
    if rate[0] > threshold:
        return float(sig[0])
    if rate[-1] < threshold:
        return np.nan                       # censored above the scan edge
    return float(np.exp(np.interp(np.log(threshold),
                                  np.log(np.maximum(rate, floor)), np.log(sig))))


def _sigmoid(x, A, xc, tau):
    return A / (1.0 + np.exp(-(x - xc) / tau))


def _midpoint(sig, rate):
    """Saturating-sigmoid scale in log sigma; (NaN, A) when not identifiable.

    A curve still climbing at the last grid point has no midpoint and a flat
    curve has no amplitude; both are censored rather than extrapolated.
    """
    from scipy.optimize import curve_fit
    x = np.log(np.asarray(sig, float))
    y = np.asarray(rate, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if y.size < 4 or y.max() <= 0:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (A, xc, tau), _ = curve_fit(
                _sigmoid, x, y, p0=[min(0.9, 2 * y.max()), x.mean(), 0.5],
                bounds=([y.max(), x.min() - 2.0, 1e-3], [1.0, x.max() + 4.0, 5.0]),
                maxfev=40000)
    except Exception:
        return np.nan, np.nan
    if not (x.min() <= xc <= x.max()):
        return np.nan, float(A)             # midpoint outside the scan: censored
    return float(np.exp(xc)), float(A)


RATE, PPRATE, HAZ, RES, ENTRIES, STEPS, NPAIRS = range(7)


def curve_of(cells, idxs=None, key=RATE):
    """(sigmas, values) over the cells that yielded a finite value."""
    sig, val = [], []
    for j, c in enumerate(cells):
        v = c.stats(idxs[j] if idxs else None)[key]
        if np.isfinite(v):
            sig.append(c.sigma)
            val.append(v)
    return np.asarray(sig), np.asarray(val)


def ci(vals, lo=2.5, hi=97.5, min_frac=0.5):
    """Percentile CI; NaN unless enough replicates actually resolved."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < max(10, min_frac * len(vals)):
        return (np.nan, np.nan, int(v.size))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)), int(v.size))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1/subexp1")
    ap.add_argument("--arm", default="bare_ald")
    ap.add_argument("--prefix", default="",
                    help="cell-tag prefix.  'm4_' reads the step-size control "
                         "tree results/phase1/subexp1_fine, whose cells are "
                         "tagged m4_s<sigma>_<arm> by run_phase1.")
    ap.add_argument("--sigmas", default=None,
                    help="comma-separated sigma grid; default the 10-point "
                         "scan.  The fine control tree carries fewer points, "
                         "so it must be passed explicitly.")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default="results/phase1/analysis/sigma_c_robust.json")
    args = ap.parse_args()

    sigmas = ([float(x) for x in args.sigmas.split(",")] if args.sigmas
              else list(SIGMAS))
    if REF_SIGMA not in sigmas:
        sys.exit(f"the sigma grid must contain the reference sigma "
                 f"{REF_SIGMA:g} (got {sigmas})")

    probe_dir = os.path.join(args.root, f"{args.prefix}s{sigmas[0]:g}_{args.arm}")
    comps = sorted({os.path.basename(f).rsplit("_seed", 1)[0]
                    for f in glob.glob(os.path.join(probe_dir, "*.json"))})
    if not comps:
        sys.exit(f"no cells under {probe_dir}")

    cells = {c: [load_cell(c, s, args.root, args.arm, args.prefix)
                 for s in sigmas] for c in comps}
    n_tr = int(np.median([len(c) for c in cells[comps[0]]]))
    print(f"arm={args.arm}  prefix='{args.prefix}'  compositions={len(comps)}  "
          f"trajectories/cell~{n_tr}  grid={len(sigmas)} pts  "
          f"clean-start, steps>=1, d_min<{D_OOD} A")

    rng = np.random.default_rng(0)
    rows, per_comp = [], {}
    iref = sigmas.index(REF_SIGMA)

    for c in comps:
        cl = cells[c]
        sig, rate = curve_of(cl)
        cross = {f"{f:g}": _cross(sig, rate, THRESHOLD, f) for f in FLOORS}
        mid, A = _midpoint(sig, rate)
        # non-parametric saturation check: how far is the published rate still
        # climbing between the last two grid points (3 -> 5 on the standard
        # 10-point scan)?  ~1 means the curve has plateaued and a midpoint
        # exists; >>1 means it has not, so the sigmoid midpoint is not
        # identifiable.  Taken from the grid rather than hardcoded to 3/5 so
        # the same code reads the shorter step-size-control grid.
        v3, v5 = cl[-2].stats()[RATE], cl[iref].stats()[RATE]
        sat = float(v5 / v3) if (np.isfinite(v3) and v3 > 0) else np.nan

        # --- bootstrap CIs (trajectory-level) -----------------------------
        ref = cl[iref]

        def resample(key, ref=ref):
            idx = rng.integers(0, len(ref), len(ref))
            return ref.stats(idx)[key]

        def resample_curve(stat, cl=cl):
            idxs = [rng.integers(0, len(x), len(x)) for x in cl]
            return stat(*curve_of(cl, idxs))

        cb = np.array([resample_curve(lambda sg, rt: _cross(sg, rt))
                       for _ in range(args.boot)])
        mb = np.array([resample_curve(lambda sg, rt: _midpoint(sg, rt)[0])
                       for _ in range(args.boot)])
        hb = np.array([resample(HAZ) for _ in range(args.boot)])
        rb = np.array([resample(RES) for _ in range(args.boot)])
        qb = np.array([resample(PPRATE) for _ in range(args.boot)])

        raw, pp, haz, res, entries, steps, n_pairs = ref.stats()
        n_atoms = int(round((1.0 + np.sqrt(1.0 + 8.0 * n_pairs)) / 2.0))

        rows.append(dict(comp=c, n_atoms=n_atoms, n_pairs=n_pairs,
                         cross=cross["1e-12"], cross_ci=ci(cb), cross_floors=cross,
                         mid=mid, mid_ci=ci(mb), A=A, sat=sat,
                         r5=raw, q5=pp, q5_ci=ci(qb),
                         haz=haz, haz_ci=ci(hb), res=res, res_ci=ci(rb),
                         entries=entries, steps=steps))
        per_comp[c] = dict(n_atoms=n_atoms, n_pairs=n_pairs, sigma=sig.tolist(),
                           rate=rate.tolist(), cross=cross, midpoint=mid,
                           amplitude=A, saturation_last_two=sat,
                           midpoint_ci=ci(mb), cross_ci=ci(cb),
                           rate_at_ref=raw, per_pair_at_ref=pp,
                           per_pair_at_ref_ci=ci(qb),
                           entry_hazard_at_ref=haz, entry_hazard_ci=ci(hb),
                           residence_at_ref=res, residence_ci=ci(rb),
                           entries=entries, steps=steps)

    # ------------------------------------------------------------------
    # Size association: the point of the exercise
    # ------------------------------------------------------------------
    from scipy import stats

    def corr(key):
        xs = [np.log(r["n_pairs"]) for r in rows
              if r[key] is not None and np.isfinite(r[key])]
        ys = [np.log(r[key]) for r in rows
              if r[key] is not None and np.isfinite(r[key])]
        if len(xs) < 5:
            return (float("nan"), float("nan"), len(xs))
        rho, p = stats.spearmanr(xs, ys)
        return (float(rho), float(p), len(xs))

    rho = {k: corr(k) for k in ("cross", "mid", "r5", "q5", "haz", "res")}

    def spread(key):
        v = np.array([r[key] for r in rows
                      if r[key] is not None and np.isfinite(r[key]) and r[key] > 0])
        return (float(v.max() / v.min()) if v.size else float("nan"), int(v.size))

    sp = {k: spread(k) for k in ("r5", "q5", "haz", "res")}

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    def val(v, fmt="{:.3f}", censor="  >5 "):
        return fmt.format(v) if v is not None and np.isfinite(v) else censor

    def brack(cc):
        return f"[{cc[0]:.2f},{cc[1]:.2f}]" if np.isfinite(cc[0]) else ""

    print(f"\n{'comp':>8} {'atoms':>5} {'pairs':>6} {'CROSS':>7} {'[95% CI]':>15} "
          f"{'MIDPOINT':>9} {'[95% CI]':>15} {'A':>5} "
          f"{'r%g/r%g' % (sigmas[-1], sigmas[-2]):>6} "
          f"{'rate@5':>8} {'entry@5':>8} {'resid':>6} {'per-pair@5':>11}")
    for r in sorted(rows, key=lambda r: (not np.isfinite(r["mid"]),
                                         r["mid"] if np.isfinite(r["mid"]) else 0.0)):
        print(f"{r['comp']:>8} {r['n_atoms']:>5} {r['n_pairs']:>6.0f} "
              f"{val(r['cross']):>7} {brack(r['cross_ci']):>15} "
              f"{val(r['mid'], '{:.3f}', '  cens'):>9} {brack(r['mid_ci']):>15} "
              f"{val(r['A'], '{:.2f}', '  -  '):>5} {val(r['sat'], '{:.1f}', '  - '):>6} "
              f"{r['r5']:>8.4f} {r['haz']:>8.4f} {r['res']:>6.1f} {r['q5']:>11.3e}")

    print("\nrank association with cell size (Spearman on logs):")
    for k, label in (("cross", "CROSS     "), ("mid", "MIDPOINT  "),
                     ("r5", "rate@5    "), ("q5", "per-pair@5"),
                     ("haz", "entry@5   "), ("res", "residence ")):
        rr, pp, nn = rho[k]
        print(f"  {label} vs n_pairs: rho={rr:+.3f} p={pp:.2g} (n={nn})")

    obs = [r["cross"] for r in rows if np.isfinite(r["cross"])]
    md = [r["mid"] for r in rows if np.isfinite(r["mid"])]
    print(f"\nCROSS:    {len(obs)}/{len(rows)} resolved; "
          f"spread {max(obs)/min(obs):.1f}x over {min(obs):.2f}-{max(obs):.2f} A")
    if md:
        print(f"MIDPOINT: {len(md)}/{len(rows)} resolved; "
              f"spread {max(md)/min(md):.1f}x over {min(md):.2f}-{max(md):.2f} A")

    for f in FLOORS:
        v = [x for x in (r["cross_floors"][f"{f:g}"] for r in rows) if np.isfinite(x)]
        if v:
            print(f"  CROSS floor={f:g}: {len(v)}/{len(rows)} resolved, "
                  f"{min(v):.2f}-{max(v):.2f} A, spread {max(v)/min(v):.1f}x")

    print(f"\nspread at sigma={REF_SIGMA:g} over {len(rows)} compositions:")
    for k, label in (("r5", "rate (published statistic)"), ("q5", "per-pair rate"),
                     ("haz", "entry hazard"), ("res", "residence (steps/entry)")):
        print(f"  {label:<28} {sp[k][0]:8.1f}x   (n={sp[k][1]})")

    # Sanity: the published statistic should factor into hazard x residence.
    print("\ndecomposition check, mean over compositions: "
          f"rate@5 / (entry@5 x residence) = "
          f"{np.nanmean([r['r5'] / (r['haz'] * max(r['res'], 1e-12)) for r in rows]):.3f}")

    with open(args.out, "w") as fh:
        json.dump(dict(arm=args.arm, prefix=args.prefix, sigmas=sigmas,
                       comps=comps, n_traj=n_tr,
                       per_comp=per_comp, rho=rho,
                       spread={k: dict(ratio=sp[k][0], n=sp[k][1]) for k in sp},
                       thresholds=dict(d_ood=D_OOD, cross=THRESHOLD,
                                       floors=FLOORS, ref_sigma=REF_SIGMA)),
                  fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
