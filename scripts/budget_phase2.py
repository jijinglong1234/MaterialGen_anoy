#!/usr/bin/env python
"""Phase 2 — GPU-budget re-computation and scale trade-off (plan step 4).

Phase 2's size is set by four knobs (datasets x configs x samplers x cells x
seeds x candidates); the frozen grid is 3 x 6 x 2 x 20 x 5 x 10 = 36,000
trajectories.  The plan's ~220 GPU-h figure predates the Cholesky
re-run and the protocol edits, so this script *measures* the
per-trajectory cost instead of restating it:

  * cost model    per-(sampler, protection) seconds-per-trajectory-per-atom
                  fitted from the Phase-1 fleet JSONs (candidates[].wall_time_s
                  / candidates[].nfe / payload n_atoms), which ran the same
                  frozen protocol code Phase 2 imports;
  * scale factors Phase 2 differs from Phase 1 in three ways, each priced
                  separately (see FACTORS below);
  * grid prices   the frozen grid plus the alternatives on the step-4 table
                  (n_cand 10 vs 20, cells, seeds, config subsets), so the
                  trade-off is a choice between priced options rather than a
                  single number.

Phase-1 inputs: MP-20 only (20 cells, 2-16 atoms, fixed cell, no stress).
Phase-2 extras priced from measurements where they exist:

  stress / upd_lat   the ALD step gains a stress evaluation.  Measured in the
                     preflight (results/phase2/preflight/preflight_summary*.json
                     wall_time_s vs the fixed-cell fleet).  read from
                     --preflight if present, else a conservative factor.
  dataset cells      cost is ~linear in n_atoms for these cell sizes; the
                     per-atom rate is what Phase 1 measures, and each dataset's
                     cell size distribution is read from the runner itself
                     (run_phase2.cells_for) so the two stay in sync.

Output: results/phase2/summary/phase2_budget.json (+ a printed table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_phase1 as rp                                            # noqa: E402
import run_phase2 as r2                                            # noqa: E402

OUT = REPO / "results" / "phase2" / "summary"

#: Phase-2 sampling parameters per config (run_phase2.CONFIGS), used to pick
#: the matching Phase-1 sigma arm for the cost model.
CONFIG_SIGMA = {"C1": 0.5, "C3": 1.5, "C5": 0.5, "C8": 0.5, "C9": 1.5,
                "C11": 0.5}
#: Phase-1 protection arm that carries the same layer set as each config.
#: C8/C9 are l1l2l3 in the fleet (SIGMA_SCAN_ALD_ONLY); C10 is not in the
#: streamlined subset.  The ALD l1l2l3 arm exists at every sigma.
CONFIG_PROT = {"C1": "bare", "C3": "bare", "C5": "bare", "C8": "l1l2l3",
               "C9": "l1l2l3", "C11": "l1"}
#: PF-ODE has no L2/L4, so its protected arm is the fleet's l1l4 directory
#: (== L1+L3 field by field, see analyze_phase1.ode_prot).
CONFIG_PROT_ODE = {"C1": "bare", "C3": "bare", "C5": "bare", "C8": "l1l4",
                   "C9": "l1l4", "C11": "l1"}


def phase1_rates() -> dict:
    """Per-(sampler, prot, sigma) seconds per trajectory and per NFE from the fleet."""
    base = REPO / "results" / "phase1" / "subexp1"
    acc: dict = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        # directory name: s<sigma>_<prot>_<sampler>
        stem = d.name
        try:
            sig_s, prot, sampler = stem[1:].split("_")[0], None, None
        except Exception:
            continue
        parts = stem[1:].split("_")
        if len(parts) != 3:
            continue
        sig_s, prot, sampler = parts
        sigma = float(sig_s)
        key = (sampler, prot, sigma)
        for f in d.glob("*.json"):
            try:
                payload = json.loads(f.read_text())
            except Exception:
                continue
            if payload.get("init_cell_gen") != rp.INIT_CELL_GEN:
                continue
            n_at = payload.get("n_atoms") or 0
            for c in payload.get("candidates", []):
                w = c.get("wall_time_s")
                nfe = c.get("nfe")
                if not w or not n_at:
                    continue
                acc.setdefault(key, []).append((w, nfe, n_at))
    out = {}
    for key, rows in acc.items():
        w = np.array([r[0] for r in rows], dtype=float)
        nfe = np.array([r[1] for r in rows], dtype=float)
        nat = np.array([r[2] for r in rows], dtype=float)
        out[key] = {
            "n": len(rows),
            "s_per_traj": float(np.median(w)),
            "s_per_traj_mean": float(w.mean()),
            "s_per_nfe": float(np.median(w / np.maximum(nfe, 1))),
            "s_per_atom": float(np.median(w / np.maximum(nat, 1))),
            "nfe_med": float(np.median(nfe)),
            "n_atoms_med": float(np.median(nat)),
            "capped_rate": float((nfe >= rp.PF_MAX_NFE).mean()),
        }
    return out


def dataset_cells(dataset: str, nnp: str = "mace") -> tuple[list, np.ndarray]:
    """The dataset's 20 Phase-2 cells plus each cell's atom count.

    ``Cell.meta`` carries no atom count on MP-20/Perov-5 (only Carbon-24
    records one), so the count is read from the reference record the cell
    indexes -- the same record the task will load.
    """
    from materialgen.data.registry import get_dataset

    spec = get_dataset(dataset)
    refs = spec.load_reference(nnp, "test")
    cells = r2.cells_for(spec, refs)
    n_atoms = np.array([len(refs[c.ref_index]["numbers"]) for c in cells],
                       dtype=float)
    return cells, n_atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnp", default="mace", choices=["mace", "esen"])
    ap.add_argument("--n-cand", type=int, default=r2.N_CAND)
    ap.add_argument("--preflight", default=str(REPO / "results" / "phase2" /
                                               "preflight" / "preflight_summary.json"))
    ap.add_argument("--updlat-probe", default=str(REPO / "results" / "phase2" /
                                                  "summary" / "phase2_updlat_cost.json"))
    ap.add_argument("--json", default=str(OUT / "phase2_budget.json"))
    args = ap.parse_args()

    rates = phase1_rates()
    print(f"Phase-1 cell costs measured for {len(rates)} (sampler, prot, sigma) arms\n")

    # ---- upd_lat price ----------------------------------------------------
    # The Phase-1 fleet ran fixed-cell and the preflight JSONs persist no wall
    # time, so this factor comes from probe_updlat_cost.py: C1 vs C5 on the same
    # cell in the same process, i.e. the same task with and without the
    # stress-driven lattice channel (run_phase2.CONFIGS differ in nothing else).
    # Measured at n_cand=3 over AlCu/FeNi3/MoS2/SrTiO3/ZnS:
    # median ratio 0.97 -- the stress evaluation is lost in the step's noise.
    # The fallback is the physical worst case if the probe file is absent.
    probe = Path(args.updlat_probe)
    if probe.exists():
        stress_factor = float(json.loads(probe.read_text())["factor_median"])
        src = f"measured ({probe.name})"
    else:
        stress_factor = 1.5
        src = "ESTIMATE (probe missing)"
    print(f"upd_lat cost factor: {stress_factor:.3f} [{src}]\n")

    # ---- Phase-2 grid price ------------------------------------------------
    cells = {ds: dataset_cells(ds, args.nnp) for ds in r2.DATASETS}
    print("cells per dataset (n_atoms):")
    atoms = {}
    for ds, (cs, na) in cells.items():
        atoms[ds] = na
        print(f"  {ds:10s} n={len(cs)} atoms min/med/max = "
              f"{na.min():.0f}/{np.median(na):.0f}/{na.max():.0f}")
    print()

    rows = []
    total = 0.0
    for ds in r2.DATASETS:
        for cfg in r2.PHASE2_SUBSET:
            sigma = CONFIG_SIGMA[cfg]
            upd = r2.CONFIGS[cfg].update_lattice
            for sampler in r2.SAMPLERS:
                prot = (CONFIG_PROT if sampler == "ald" else CONFIG_PROT_ODE)[cfg]
                key = (sampler, prot, sigma)
                r = rates.get(key)
                if r is None:
                    rows.append({"dataset": ds, "config": cfg, "sampler": sampler,
                                 "status": "no phase-1 arm", "key": list(key)})
                    continue
                n_traj = len(cells[ds][0]) * len(rp.SEEDS) * args.n_cand
                # Cost is ~linear in n_atoms at these cell sizes, so carry the
                # Phase-1 rate to another dataset by the ratio of median cell
                # sizes (MP-20's arm median vs this dataset's cell median).
                size = float(np.median(atoms[ds])) / r["n_atoms_med"]
                s = r["s_per_traj"] * size
                if upd:
                    s *= stress_factor
                gpu_h = n_traj * s / 3600.0
                total += gpu_h
                rows.append({"dataset": ds, "config": cfg, "sampler": sampler,
                             "sigma": sigma, "prot": prot, "n_traj": n_traj,
                             "size_factor": round(size, 3),
                             "s_per_traj": round(s, 2),
                             "s_per_traj_measured": round(r["s_per_traj"], 2),
                             "nfe_med": r["nfe_med"], "capped_rate": r["capped_rate"],
                             "upd_lat": upd, "gpu_h": round(gpu_h, 1)})

    print(f"{'dataset':10s} {'cfg':4s} {'smp':6s} {'traj':>6s} {'s/traj':>8s} "
          f"{'cap%':>6s} {'GPU-h':>8s}")
    for r in rows:
        if r.get("status"):
            print(f"{r['dataset']:10s} {r['config']:4s} {r['sampler']:6s}  -- {r['status']} {r['key']}")
            continue
        print(f"{r['dataset']:10s} {r['config']:4s} {r['sampler']:6s} "
              f"{r['n_traj']:6d} {r['s_per_traj']:8.1f} {r['capped_rate']*100:5.1f}% "
              f"{r['gpu_h']:8.1f}")
    print(f"\nTOTAL (n_cand={args.n_cand}): {total:.0f} sequential GPU-hours "
          f"= {total/4:.1f} h on 4 GPUs, {total/(4*3):.1f} h at 3 workers/GPU")

    # ---- scale alternatives -------------------------------------------------
    alts = {}
    for nc in (5, 10, 20):
        alts[f"n_cand={nc}"] = round(total * nc / args.n_cand, 0)
    for ncell in (10, 20):
        alts[f"n_cells={ncell}"] = round(total * ncell / 20, 0)
    for nseed in (3, 5):
        alts[f"n_seeds={nseed}"] = round(total * nseed / 5, 0)
    subsets = {
        "frozen 6 configs": r2.PHASE2_SUBSET,
        "core 4 (C1,C3,C8,C9)": ("C1", "C3", "C8", "C9"),
        "protected 3 (C8,C9,C11)": ("C8", "C9", "C11"),
    }
    for name, cfgs in subsets.items():
        alt = sum(r.get("gpu_h", 0.0) for r in rows if r["config"] in cfgs)
        alts[name] = round(alt, 0)
    alts["mp_20 only"] = round(sum(r.get("gpu_h", 0.0) for r in rows
                                   if r["dataset"] == "mp_20"), 0)
    alts["ald only"] = round(sum(r.get("gpu_h", 0.0) for r in rows
                                 if r["sampler"] == "ald"), 0)
    alts["pfode only"] = round(sum(r.get("gpu_h", 0.0) for r in rows
                                   if r["sampler"] == "pfode"), 0)
    print("\nscale alternatives (sequential GPU-h):")
    for k, v in alts.items():
        print(f"  {k:28s} {v:8.0f}")

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"nnp": args.nnp, "n_cand": args.n_cand, "stress_factor": stress_factor,
         "phase1_rates": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rates.items()},
         "rows": rows, "total_gpu_h": round(total, 1), "alternatives": alts},
        indent=1))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
