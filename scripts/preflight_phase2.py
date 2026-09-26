"""Phase 2 preflight: upd_lat smoke + beta/NFE/sigma_min checks + travel metrics.

Phase 2 (plan \S6.2, configs C1/C3/C5/C8/C9/C11) enables stress-driven lattice
updates for the first time in the fleet pipeline.  This preflight answers the
four open questions from the design discussions before committing
~250 GPU-h to the full Phase 2:

  A  upd_lat smoke:  sampler {ALD, PF-ODE} x prot {bare, l1/l1l4} x
                     sigma {0.5, 1.5} x comp {SrTiO3, ZnS, FeNi3} x
                     seed {42, 123} x 10 candidates, update_lattice=True.
                     Gates: lattice collapse rate (vol < 0.5 V0), Type III
                     (vol > 10 V0), induced OOD vs the fixed-cell fleet
                     baselines, E_hull, validity, and lattice drift.
  B  beta check:     ALD L1L4 sigma=0.5, beta in {11.6 (1000K), 38.68 (300K)}
                     --- alpha=1e-3 was calibrated at 300K; 1000K weakens the
                     drift 3.3x relative to noise.
  C  NFE budget:     ALD L1L4 sigma=0.5, K in {100 (NFE 200), 200 (NFE 400)}
                     --- the ald_budget discussion deferred 200->400 to
                     the protected configs; here it is measured under L1L4.
  D  stretched min:  ALD L1L4 sigma=0.5, sigma_min in {0.01, 0.05}
                     --- end-state anneal floor of the design discussion.
  E  lattice fix round (post-cell-A finding): the stress-driven
     update under L1L4 deterministically INFLATES Pauli-strong cells (SrTiO3
     both seeds both sigma: |dL|/|L0| ~ 0.9, vol x7.5, E_hull 0.47-10.4 eV;
     ZnS immune).  Mitigations on the worst cells (SrTiO3 s0.5/s1.5 x both
     seeds, FeNi3 s0.5, ZnS s0.5 control, seed42):
       E-w   volume wall: reject steps inflating past 2.0 x V0 (mirror of
             the collapse wall at 0.5) — ALDConfig.lattice_vol_max_ratio
       E-a4  alpha_lattice 1e-4 (weaker ratchet)
       E-a5  alpha_lattice 1e-5
       E-g   reopen gate: lattice_reopen_frac 0.02 AND d_min >= 0.7 A —
             the cell adapts only at true equilibrium
       E-wa  volume wall 2.0 + alpha_lattice 1e-4 (combined)

Every trajectory additionally records min-image travel metrics (path length,
end-to-end displacement) from its accepted-step positions --- the fleet JSONs
do not store positions, so protected-config travel metrics are re-measured
here (unresolved in the fleet).

Output: results/phase2/preflight/preflight_summary_<nnp>.json
Usage:   python scripts/preflight_phase2.py [--shard k/N] [--nnp esen] [--device cuda:0]

NNP default: eSEN (Phase-2 benchmark runs eSEN as the primary NNP per the
paper's protocol; the Phase-1 M0 degradation rule exempts only the subexp1/3
large sigma scans).  MACE remains selectable for reruns of the
preflight, whose cells were measured on MACE-MP-0.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import run_phase1 as rp  # frozen fleet protocol helpers (import-only reuse)
from run_phase2 import _is_sticky_cuda_error as rp_is_sticky_cuda

OUT = rp.REPO / "results" / "phase2" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

COMPS_A = ["SrTiO3", "ZnS", "FeNi3"]   # oxide / sulfide / intermetallic
COMPS_B = ["SrTiO3", "FeNi3"]
COMPS_C = ["SrTiO3", "ZnS"]
COMPS_D = ["SrTiO3"]
SEEDS = [42, 123]
N_CAND = 10
SIGMA_A = [0.5, 1.5]
BETAS = [11.6, 38.68]                  # 1000 K / 300 K
KS = [100, 200]                        # NFE 200 / 400 (M=2)
SIGMA_MINS = [0.01, 0.05]
ALPHA_LATTICE = 0.001                  # ALDConfig default (paper C1 practice)

# sampler arms: PF-ODE runs L1 without reflection (H1c fleet finding: L3 is
# net-harmful on the ODE arm); ALD runs {bare, l1l4} per the Phase-2 plan.
ALD_PROTS = ["bare", "l1l4"]
PF_PROTS = ["bare", "l1"]


def build_tasks():
    tasks = []
    # A: upd_lat smoke
    for sigma in SIGMA_A:
        for comp in COMPS_A:
            for seed in SEEDS:
                for prot in ALD_PROTS:
                    tasks.append({"cell": "A", "sampler": "ald", "prot": prot,
                                  "sigma": sigma, "comp": comp, "seed": seed,
                                  "beta": 38.68, "K": 100, "sigma_min": 0.01,
                                  "update_lattice": True})
                for prot in PF_PROTS:
                    tasks.append({"cell": "A", "sampler": "pfode", "prot": prot,
                                  "sigma": sigma, "comp": comp, "seed": seed,
                                  "beta": 38.68, "K": 100, "sigma_min": 0.01,
                                  "update_lattice": True})
    # B: beta check
    for beta in BETAS:
        for comp in COMPS_B:
            for seed in SEEDS:
                tasks.append({"cell": "B", "sampler": "ald", "prot": "l1l4",
                              "sigma": 0.5, "comp": comp, "seed": seed,
                              "beta": beta, "K": 100, "sigma_min": 0.01,
                              "update_lattice": True})
    # C: NFE budget
    for K in KS:
        for comp in COMPS_C:
            for seed in SEEDS:
                tasks.append({"cell": "C", "sampler": "ald", "prot": "l1l4",
                              "sigma": 0.5, "comp": comp, "seed": seed,
                              "beta": 38.68, "K": K, "sigma_min": 0.01,
                              "update_lattice": True})
    # D: stretched sigma_min
    for smin in SIGMA_MINS:
        for seed in SEEDS:
            tasks.append({"cell": "D", "sampler": "ald", "prot": "l1l4",
                          "sigma": 0.5, "comp": "SrTiO3", "seed": seed,
                          "beta": 38.68, "K": 100, "sigma_min": smin,
                          "update_lattice": True})
    # E: lattice fix round.  Mitigation overrides on the
    # distortion-prone cells; 5 candidates (distortion is a deterministic
    # attractor, so 5 suffices for screening).
    E_CELLS = [("SrTiO3", 0.5, 42), ("SrTiO3", 0.5, 123),
               ("SrTiO3", 1.5, 42), ("SrTiO3", 1.5, 123),
               ("FeNi3", 0.5, 42), ("ZnS", 0.5, 42)]
    E_MITS = {
        "E-w":  {"lat_vol_max": 2.0, "alpha_lat": 1e-3, "frac": 0.1, "dmin": 0.0},
        "E-a4": {"lat_vol_max": float("inf"), "alpha_lat": 1e-4, "frac": 0.1, "dmin": 0.0},
        "E-a5": {"lat_vol_max": float("inf"), "alpha_lat": 1e-5, "frac": 0.1, "dmin": 0.0},
        "E-g":  {"lat_vol_max": float("inf"), "alpha_lat": 1e-3, "frac": 0.02, "dmin": 0.7},
        "E-wa": {"lat_vol_max": 2.0, "alpha_lat": 1e-4, "frac": 0.1, "dmin": 0.0},
    }
    for mit, ov in E_MITS.items():
        for comp, sigma, seed in E_CELLS:
            tasks.append({"cell": mit, "sampler": "ald", "prot": "l1l4",
                          "sigma": sigma, "comp": comp, "seed": seed,
                          "beta": 38.68, "K": 100, "sigma_min": 0.01,
                          "update_lattice": True, "n_cand": 5, **ov})
    # F: patched-L3 validation.  ReflectionOperator expansion is
    # now clamped to the reference cell (never inflates past V0) and gated on
    # update_lattice; the lattice-update channel runs alpha=1e-4 + vol wall
    # 2.0 (the E-a4/E-w fixes for the FeNi3 channel).  Re-runs the worst
    # preflight cells expecting drift ~ 0 and E_hull ~ fixed-cell baseline.
    F_CELLS = [("SrTiO3", 0.5, 42), ("SrTiO3", 0.5, 123),
               ("SrTiO3", 1.5, 42), ("SrTiO3", 1.5, 123),
               ("FeNi3", 0.5, 42), ("ZnS", 0.5, 42)]
    for comp, sigma, seed in F_CELLS:
        tasks.append({"cell": "F", "sampler": "ald", "prot": "l1l4",
                      "sigma": sigma, "comp": comp, "seed": seed,
                      "beta": 38.68, "K": 100, "sigma_min": 0.01,
                      "update_lattice": True, "n_cand": 5,
                      "lat_vol_max": 2.0, "alpha_lat": 1e-4,
                      "frac": 0.1, "dmin": 0.0})
    # F2: post-freeze rerun.  The F round is void twice over:
    # its cells went through the pre-Cholesky `noise_lattice` (see
    # LEGACY_pre_cholesky.md), and it carried the legacy full-stack
    # name "l1l4".  The frozen Phase-2 layer set is C8 = l1l2l3 (L4 is the
    # net-harmful ablation, not a defense), so the L3-ratchet validation has to
    # be re-measured on the configuration that will actually launch.
    #
    # The load-bearing question is the lattice channel.  The F round concluded
    # "lat_vol_max=2.0 + alpha_lat=1e-4 + L3 clamp", but only the third part is
    # in the code: run_phase2.CONFIGS carries alpha_lat = 0.001 (the ALDConfig
    # default) and never sets lattice_vol_max_ratio, so it runs with no
    # inflation wall at all.  Either the 4a12ce4 L3 clamp alone holds the cell
    # and the CONFIGS are fine, or the config mitigation is still load-bearing
    # and CONFIGS must be amended before the full run.  F2-a / F2-b separate
    # those two hypotheses; F2-c restores the old F arm for the L4 contrast on
    # the upd_lat channel (doc-12 is fixed-cell, so no existing cell covers it).
    F2_OV = {
        "F2-a": {"lat_vol_max": float("inf"), "alpha_lat": 1e-3},
        "F2-b": {"lat_vol_max": 2.0, "alpha_lat": 1e-4},
        "F2-c": {"lat_vol_max": 2.0, "alpha_lat": 1e-4},
    }
    F2_PROT = {"F2-a": "l1l2l3", "F2-b": "l1l2l3", "F2-c": "l1l4"}
    for arm, ov in F2_OV.items():
        for comp, sigma, seed in F_CELLS:
            tasks.append({"cell": arm, "sampler": "ald", "prot": F2_PROT[arm],
                          "sigma": sigma, "comp": comp, "seed": seed,
                          "beta": 38.68, "K": 100, "sigma_min": 0.01,
                          "update_lattice": True, "n_cand": 10,
                          "frac": 0.1, "dmin": 0.0, **ov})
    return tasks


# ---------------------------------------------------------------------------
# Scores / samplers (fleet convention + preflight overrides)
# ---------------------------------------------------------------------------

def make_score2(calc, prot: str, beta: float, nnp: str = "esen"):
    """Like rp.make_score but with beta override and stress enabled (upd_lat)."""
    from materialgen.core.score_function import AugmentedScore, NNPScore, PauliScore

    nnp_ = NNPScore(calc, beta=beta, compute_stress=True, label=rp.nnp_label(nnp))
    if prot == "bare":
        return nnp_
    pauli = PauliScore(pauli=rp.PauliRepulsion(params=None, deep_wall=False),
                       beta=beta)
    return AugmentedScore(nnp_, pauli)


def make_ald_config2(sigma_max, prot, K, sigma_min, update_lattice, t=None):
    cfg = rp.make_ald_config(sigma_max, prot)
    cfg.K = K
    cfg.sigma_min = sigma_min
    cfg.update_lattice = update_lattice
    cfg.lattice_reopen_frac = t.get("frac", 0.1) if t else 0.1
    cfg.alpha_lattice = t.get("alpha_lat", ALPHA_LATTICE) if t else ALPHA_LATTICE
    cfg.lattice_vol_max_ratio = t.get("lat_vol_max", float("inf")) if t else float("inf")
    cfg.lattice_min_dmin = t.get("dmin", 0.0) if t else 0.0
    cfg.save_trajectory = True
    return cfg


def make_pfode_config2(sigma_max, prot, update_lattice):
    cfg = rp.make_pfode_config(sigma_max, prot)
    cfg.update_lattice = update_lattice
    cfg.lattice_reopen_frac = 0.1
    return cfg


def make_sampler(cfg, prot, sampler_name):
    if sampler_name == "ald":
        full = prot == "l1l4"
        monitor = None
        if full:
            from materialgen.core.safety_monitor import MonitorThresholds, SafetyMonitor
            monitor = SafetyMonitor(thresholds=MonitorThresholds(), p=2.0,
                                    gamma_base=0.5)
        from materialgen.samplers.ald import ALDSampler
        return ALDSampler(cfg, safety_monitor=monitor)
    from materialgen.samplers.pf_ode import PFODESampler
    return PFODESampler(cfg, reflection=None)  # no L3 on the ODE arm (H1c)


# ---------------------------------------------------------------------------
# Trajectory metrics (lattice-aware min-image)
# ---------------------------------------------------------------------------

def travel_metrics(positions: list, lattices: list) -> dict:
    """positions: list of (N,3) frac arrays; lattices: list of (3,3).

    min-image per step with the step's own lattice (upd_lat varies the cell).
    """
    d_cart = []
    for i in range(len(positions) - 1):
        d_frac = positions[i + 1] - positions[i]
        d_frac -= np.round(d_frac)          # min-image (wrap correction)
        d_cart.append(d_frac @ lattices[i])
    d_cart = np.array(d_cart)               # (S, N, 3)
    if len(d_cart) == 0:
        return dict(path_len=0.0, disp_sum=0.0, step_max=0.0,
                    e2e_med=0.0, e2e_max=0.0, n_steps=0)
    path_len = float(np.sum(np.linalg.norm(d_cart, axis=2)))
    disp_sum = float(np.sum(np.linalg.norm(d_cart, axis=2)))
    step_max = float(np.max(np.linalg.norm(d_cart, axis=2)))
    delta = positions[-1] - positions[0]
    delta -= np.round(delta)
    e2e = np.linalg.norm(delta @ lattices[-1], axis=1)
    return dict(path_len=path_len, disp_sum=disp_sum, step_max=step_max,
                e2e_med=float(np.median(e2e)), e2e_max=float(np.max(e2e)),
                n_steps=len(d_cart))


# ---------------------------------------------------------------------------
# Candidate runner: rp.run_candidate + positions/travel/volume capture
# ---------------------------------------------------------------------------

def run_candidate_traj(score, sampler, ref_record, sigma_max, seed, cand, hull):
    rng = np.random.RandomState(seed * 1000 + cand)
    initial = rp.make_initial(ref_record, sigma_max, rng)

    sampler_cfg = getattr(sampler, "config", None)
    if sampler_cfg is not None:
        sampler_cfg.seed = seed * 1000 + cand + 500000

    diag = rp.DiagnosticScore(score)
    t0 = time.time()
    res = sampler.sample(diag, initial)
    nfe, n_rej = res.nfe, getattr(res, "n_rejections", 0)

    # Positions: PF-ODE -> accepted-step path; ALD -> save_trajectory states.
    positions, lattices = None, None
    if getattr(res, "path", None):
        diag.records = []
        pos, lat = [], []
        for st in res.path:
            diag.compute(st)
            pos.append(st.frac_coords)
            lat.append(st.lattice)
        positions, lattices = np.array(pos), np.array(lat)
    elif getattr(res, "trajectory", None) and len(res.trajectory):
        pos, lat = [], []
        for st in res.trajectory:
            pos.append(st.frac_coords)
            lat.append(st.lattice)
        positions = np.concatenate([[initial.frac_coords], pos])
        lattices = np.concatenate([[initial.lattice], lat])

    final = res.structure
    r_final = score.compute(final)
    d_min_f, cov_def_f = rp.structure_diagnostics(final)
    f_nnp_f = (r_final.metadata or {}).get("nnp_forces", r_final.forces)
    max_f_f = float(np.linalg.norm(f_nnp_f, axis=1).max()) if f_nnp_f is not None and len(f_nnp_f) else None
    e_final = r_final.energy
    finite_ok = np.isfinite(d_min_f) and (max_f_f is None or np.isfinite(max_f_f)) \
        and (e_final is None or np.isfinite(e_final))
    valid = bool(finite_ok and d_min_f >= rp.OOD_DMIN)

    masks, agg = rp.classify_steps(diag.records, final.num_atoms, initial.volume)
    e_hull = None
    if e_final is not None and np.isfinite(e_final):
        e_hull = hull.e_hull(final.properties.get("formula")
                             or ref_record["metadata"]["pretty_formula"],
                             e_final, final.num_atoms)

    out = {
        "seed": seed, "cand": cand, "nfe": nfe, "n_rejections": n_rej,
        "valid": valid, "ood_steps": agg["n_ood_steps"],
        "type1_steps": agg["type1_steps"], "type2_steps": agg["type2_steps"],
        "type3_steps": agg["type3_steps"], "nan_steps": agg["n_nan_steps"],
        "collapse_steps": agg["collapse_steps"],
        "ood_bitmask": masks,
        "e_hull": e_hull,
        "wall_time_s": time.time() - t0,
    }
    # lattice drift: |L_final - L_initial| / |L_initial| (Frobenius)
    if positions is not None and len(lattices):
        out["travel"] = travel_metrics(positions, lattices)
        L0 = lattices[0]
        out["lat_drift"] = float(np.linalg.norm(lattices[-1] - L0)
                                 / max(np.linalg.norm(L0), 1e-12))
        vol_ratio = [float(np.linalg.det(L) / np.linalg.det(L0))
                     for L in lattices]
        out["vol_max_ratio"] = float(max(vol_ratio))
        out["vol_final_ratio"] = float(vol_ratio[-1])
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def cell_label(t):
    return (f"{t['cell']}|{t['sampler']}|{t['prot']}|s{t['sigma']:g}|"
            f"{t['comp']}|seed{t['seed']}|b{t['beta']:g}|K{t['K']}|"
            f"min{t['sigma_min']:g}")


def pool(cands):
    n = len(cands)
    if n == 0:
        return None
    OOD = 0b010011   # paper §6.1: d_min<0.5 | max|F_NNP|>500 | Type III
    start_ood = sum(1 for c in cands if c["ood_bitmask"]
                    and c["ood_bitmask"][0] & OOD)
    ind = sum(1 for c in cands if c["ood_bitmask"]
              and not (c["ood_bitmask"][0] & OOD)
              and any(m & OOD for m in c["ood_bitmask"][1:]))
    ehs = [c["e_hull"] for c in cands if c["e_hull"] is not None]
    travel = [c["travel"] for c in cands if c.get("travel")]
    out = {
        "n_traj": n,
        "ood_rate": sum(1 for c in cands if c["ood_steps"]) / n,
        "start_ood_rate": start_ood / n,
        "induced_ood_rate": ind / n,
        "validity": sum(1 for c in cands if c["valid"]) / n,
        "collapse_rate": sum(1 for c in cands if c["collapse_steps"]) / n,
        "type3_rate": sum(1 for c in cands if c["type3_steps"]) / n,
        "nfe_med": float(np.median([c["nfe"] for c in cands])),
        "e_hull_med_meV": float(np.median(ehs)) * 1000 if ehs else None,
        # units fix: HullEvaluator now returns true eV/atom.  Rows
        # in files written before the fix are in the pre-fix scale (true
        # eV/atom divided by the reduced-formula atom count, 2-5) and cannot
        # be rescaled -- these summaries persist no candidates.  Always
        # compare rows of the same era, or re-run the cell.
        "e_hull_units": "eV/atom",
        "lat_drift_med": float(np.median([c["lat_drift"] for c in cands
                                          if "lat_drift" in c])),
        "vol_max_ratio_p90": float(np.percentile(
            [c["vol_max_ratio"] for c in cands if "vol_max_ratio" in c], 90))
            if any("vol_max_ratio" in c for c in cands) else None,
    }
    if travel:
        out["path_len_med"] = float(np.median([t["path_len"] for t in travel]))
        out["e2e_med"] = float(np.median([t["e2e_med"] for t in travel]))
        out["e2e_q75"] = float(np.percentile([t["e2e_med"] for t in travel], 75))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None,
                    help="k/N: keep tasks whose index %% N == k")
    ap.add_argument("--cells", default=None,
                    help="cell prefix filter, comma-separated (e.g. 'A,F' or 'E')")
    ap.add_argument("--sampler", default=None, choices=["ald", "pfode"],
                    help="sampler filter (e.g. launch the ALD half of A)")
    ap.add_argument("--nnp", default="esen", choices=["mace", "esen"],
                    help="force field backend (default esen: Phase-2 primary NNP)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None,
                    help="suffix inserted into the output file name "
                         "(preflight_summary_<tag>_<nnp>.json).  A partial-cell "
                         "run must pass this: without it, '--cells F2' would "
                         "overwrite the legacy summary.")
    args = ap.parse_args()

    tasks = build_tasks()
    if args.cells:
        keep = [c.strip() for c in args.cells.split(",") if c.strip()]
        tasks = [t for t in tasks if any(t["cell"].startswith(c) for c in keep)]
    if args.sampler:
        tasks = [t for t in tasks if t["sampler"] == args.sampler]
    if args.shard:
        k, N = (int(x) for x in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % N == k]

    calc = rp.load_calculator(args.nnp, args.device)
    refs = rp.load_reference_structures(
        sorted({t["comp"] for t in tasks}))
    hull = rp.HullEvaluator(rp.load_hull_table())

    # E_hull calibration (fleet convention, run_task pattern): NNP energy of
    # the reference (primitive) structure, keyed internally by calc_ref_energy.
    for comp in sorted({t["comp"] for t in tasks}):
        ref = refs[comp]
        ref_crystal = rp.CrystalStructure.from_frac_coords(
            ref["numbers"], ref["frac_coords"], ref["lattice"])
        e_ref = rp.calc_ref_energy(calc, ref_crystal)
        hull.calibrate(comp, e_ref, len(ref["numbers"]),
                       float(ref["metadata"].get("formation_energy_per_atom", 0.0)))

    # NNP-suffixed outputs: the preflight (MACE-MP-0) keeps its
    # legacy file names; eSEN runs write *_esen.* so backends never clobber.
    nnp_suffix = "" if args.nnp == "mace" else f"_{args.nnp}"
    tag = f"_{args.tag}" if args.tag else ""
    summary_path = OUT / f"preflight_summary{tag}{nnp_suffix}.json"
    if args.shard:
        # shard files merge on completion; single-worker writes the summary
        summary_path = OUT / (f"preflight_shard{tag}_"
                              f"{args.shard.replace('/', '_')}{nnp_suffix}.json")

    def flush_rows() -> None:
        """Persist progress after every cell.

        A cell costs minutes on eSEN and the run is killable from outside
        (the eSEN batch lost 5 finished rows per shard to a
        SIGKILL mid-cell, because the only write was the one after the loop).
        Rewriting on each cell makes a killed run resume from what it has.
        """
        summary_path.write_text(json.dumps({"nnp": args.nnp, "rows": rows,
                                            "tasks": len(tasks)},
                                           indent=1, default=str))

    # Resume: a killed shard restarts on the cells its last flush did not cover.
    rows = {}
    if summary_path.exists():
        try:
            rows = json.loads(summary_path.read_text()).get("rows", {})
        except Exception:
            rows = {}
        if rows:
            print(f"resuming: {len(rows)} cells already on disk in "
                  f"{summary_path.name}", flush=True)

    for t in tasks:
        label = cell_label(t)
        if rows.get(label, {}).get("n_traj"):
            continue                       # already computed by an earlier pass
        score = make_score2(calc, t["prot"], t["beta"], nnp=args.nnp)
        if t["sampler"] == "ald":
            cfg = make_ald_config2(t["sigma"], t["prot"], t["K"],
                                   t["sigma_min"], t["update_lattice"], t)
        else:
            cfg = make_pfode_config2(t["sigma"], t["prot"], t["update_lattice"])
        sampler = make_sampler(cfg, t["prot"], t["sampler"])
        n_cand = t.get("n_cand", N_CAND)
        try:
            cands = [run_candidate_traj(score, sampler, refs[t["comp"]],
                                        t["sigma"], t["seed"], cand, hull)
                     for cand in range(n_cand)]
        except Exception as exc:
            # A dead CUDA context fails every later cell too, so stop cleanly
            # and let the supervisor restart this shard -- the flush above
            # means the restart resumes at this cell.  See run_phase2's
            # _is_sticky_cuda_error for the failure this was written against
            # (eSEN).
            if rp_is_sticky_cuda(exc):
                flush_rows()
                print(f"ABORT shard: sticky CUDA error in {label}: {exc}",
                      flush=True)
                return 3
            raise
        rows[label] = pool(cands)
        m = rows[label]
        print(f"{label:58s} n={m['n_traj']:3d} ood={m['ood_rate']*100:5.1f}% "
              f"ind={m['induced_ood_rate']*100:5.1f}% "
              f"valid={m['validity']*100:4.1f}% "
              f"collapse={m['collapse_rate']*100:4.2f}% "
              f"eHull={m['e_hull_med_meV'] if m['e_hull_med_meV'] else float('nan'):7.0f} "
              f"e2e={m.get('e2e_med', float('nan')):5.2f} "
              f"latDrift={m['lat_drift_med']:.3f}", flush=True)
        flush_rows()

    flush_rows()
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
