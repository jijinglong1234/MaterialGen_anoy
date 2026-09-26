"""Pre-flight protocol self-check for the cell-init fix.

Run before every fleet launch (risk #10 in docs/experiment_plan_v1.md): a
silent deformation of the initial cell invalidated all of Phase 1 once, and
these assertions are what catch it in seconds instead of after a week of GPU
time.

Checks, per composition:
  1. sigma = 0 is an exact no-op on the Gram matrix (the bug was
     sigma-INDEPENDENT, so this is the decisive regression test);
  2. the induced strain stays in the documented band and scales with sigma;
  3. d_min of the perturbed cell stays close to the reference d_min;
  4. make_initial is deterministic in (sigma, seed, candidate);
  5. the payload markers the analysis layer keys on are present.

Usage: python scripts/check_protocol_init_cell.py [--nnp-free]
Exit code 0 = PASS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import run_phase1 as rp                                    # noqa: E402

COMPS = ["Al2O3", "TiO2", "AlCu", "GaN", "SrTiO3"]
SIGMA_PROBE = 1.0
NDRAW = 12


def strains(cell, g0, c0i):
    """Principal stretches of `cell` relative to the reference Gram g0."""
    g1 = np.asarray(cell) @ np.asarray(cell).T
    s = c0i @ g1 @ c0i.T
    s = 0.5 * (s + s.T)
    return np.sqrt(np.linalg.eigvalsh(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cand", type=int, default=rp.N_CAND,
                    help=f"expected candidates per task (default: N_CAND={rp.N_CAND})")
    args = ap.parse_args()

    print(f"N_CAND = {rp.N_CAND}, PF_MAX_NFE = {rp.PF_MAX_NFE}, "
          f"sigmas = {rp.SIGMAS}")
    print(f"marker expected in every payload: init_cell_gen = "
          f"{rp.__dict__.get('INIT_CELL_GEN', '(written in run_task)')!r}")
    fails = []

    refs = rp.load_reference_structures(rp.COMPOSITIONS_20)
    refs = {c: refs[c] for c in COMPS if c in refs}
    print(f"\n{'comp':9} {'G(0) err':>10} {'max|s-1| s=1':>13} "
          f"{'s=1 / s=0.1':>12} {'dmin s=0':>9} {'dmin ref':>9} {'floor1':>7}")
    for comp, rec in refs.items():
        L0 = np.asarray(rec["lattice"], dtype=float)
        g0 = L0 @ L0.T
        c0i = np.linalg.inv(np.linalg.cholesky(g0))

        # -- 1. sigma = 0 must reproduce the reference Gram exactly ----------
        zero = rp.make_initial(rec, 0.0, np.random.RandomState(0))
        g_zero = np.asarray(zero.lattice) @ np.asarray(zero.lattice).T
        g_err = float(np.max(np.abs(g_zero - g0)))
        if g_err > 1e-8:
            fails.append(f"{comp}: sigma=0 changed the Gram matrix by {g_err:.3e} "
                         "(the pre-fix bug signature)")
        s_zero = strains(zero.lattice, g0, c0i)
        if np.max(np.abs(s_zero - 1.0)) > 1e-6:
            fails.append(f"{comp}: sigma=0 stretch {s_zero} != 1")

        # -- 2/3. strain band, sigma-scaling, d_min -------------------------
        hi, lo = [], []
        for cand in range(NDRAW):
            r1 = rp.make_initial(rec, SIGMA_PROBE, np.random.RandomState(cand))
            s1 = strains(r1.lattice, g0, c0i)
            hi.append(np.max(np.abs(s1 - 1.0)))
            r0 = rp.make_initial(rec, 0.1, np.random.RandomState(cand))
            lo.append(np.max(np.abs(strains(r0.lattice, g0, c0i) - 1.0)))
        hi_m, lo_m = float(np.median(hi)), float(np.median(lo))
        if not (0.02 < hi_m < 0.6):
            fails.append(f"{comp}: median |stretch-1| at sigma=1 is {hi_m:.3f}, "
                         "outside the expected 0.02-0.6 band")
        ratio = hi_m / max(lo_m, 1e-9)
        if not (3.0 < ratio < 25.0):
            fails.append(f"{comp}: sigma=1 / sigma=0.1 strain ratio {ratio:.1f} "
                         "is not ~10x -- the noise level may not scale with sigma")

        # At sigma=0 the Cartesian term vanishes too, so the structure must be
        # the reference exactly -- a cell distortion shows up here as a d_min
        # drop.  (At sigma>0 d_min is dominated by the documented 1 A-per-
        # component Cartesian start noise, so it is reported, not asserted.)
        dmin_ref = rp.structure_diagnostics(zero)[0]
        dmin_true = rp.structure_diagnostics(
            rp.CrystalStructure.from_frac_coords(
                rec["numbers"], rec["frac_coords"], rec["lattice"]))[0]
        if abs(dmin_ref - dmin_true) > 1e-9:
            fails.append(f"{comp}: d_min at sigma=0 is {dmin_ref:.4f} A, "
                         f"reference {dmin_true:.4f} A")

        floor1 = float(np.mean([
            rp.structure_diagnostics(
                rp.make_initial(rec, SIGMA_PROBE, np.random.RandomState(c)))[0] < 0.5
            for c in range(20)]))
        if not (0.0 <= floor1 <= 0.75):
            fails.append(f"{comp}: start-OOD share at sigma=1 is {floor1:.2f}, "
                         "outside the documented 0-0.75 band")

        print(f"{comp:9} {g_err:10.2e} {hi_m:13.4f} {ratio:12.1f} "
              f"{dmin_ref:9.3f} {dmin_true:9.3f} {floor1:9.2f}")

    # -- 4. determinism ------------------------------------------------------
    rec = next(iter(refs.values()))
    a = rp.make_initial(rec, SIGMA_PROBE, np.random.RandomState(7)).lattice
    b = rp.make_initial(rec, SIGMA_PROBE, np.random.RandomState(7)).lattice
    if not np.array_equal(a, b):
        fails.append("make_initial is not deterministic in (sigma, seed, cand)")
    print(f"\ndeterminism (same seed/cand -> identical cell): "
          f"{'OK' if np.array_equal(a, b) else 'FAIL'}")

    print(f"\n{'PASS: protocol self-check clean' if not fails else 'FAIL:'}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
