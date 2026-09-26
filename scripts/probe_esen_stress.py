"""probe_esen_stress.py — eSEN stress-channel probe (Phase-2 upd_lat gate).

Phase 2 (plan §6.2) is the first fleet stage to enable stress-driven lattice
updates, and Phase 2's default NNP is eSEN (decision).  Every Phase-1
eSEN task ran fixed-cell (compute_stress=False), so the eSEN stress channel has
never been exercised; NNPScore swallows stress failures silently (a calculator
without get_stress produces lattice_score=None and the sampler silently skips
lattice updates — ald.py gates on `score.lattice_score is not None`).  This
probe verifies the channel before the ~250 GPU-h Phase-2 fleet is committed.

Gates:
  G1  interface   calc.get_stress() succeeds on ASE atoms (finite 6-vector).
  G2  pipeline    NNPScore(compute_stress=True).compute() returns a non-None
                  stress tensor AND lattice_score (what ALD/PF-ODE consume).
  G3  strain      +2% volume strain produces a finite elastic response
                  (|dtrace| > 1e-3 eV/A^3) whose sign and magnitude agree with
                  MACE on the same cell (same sign, |ratio| within [0.1x, 10x]).
                  The sign itself is NNP-specific (it encodes where the MP
                  reference sits relative to that NNP's own equilibrium volume),
                  so no absolute direction is required — MACE is the reference.
  G4  smoke (eSEN only): full ALD L1L4 update_lattice trajectories (1 cand per
      cell) with the frozen F-fix config (vol wall 2.0 + alpha_lat 1e-4,
      from the preflight) — completes without pathology, cell stays within
      [0.5, 2.0] x V0 throughout, and each trajectory is either valid or shows
      measurable lattice drift (the stress channel responding).

Run TWICE — once per conda env (eSEN lives in `materialgen-esen`, fairchem-core
1.10; MACE in `materialgen`; they cannot share an env):
    # materialgen-esen env, GPU 0:
    python scripts/probe_esen_stress.py --nnp esen
    # materialgen env (same GPU or another):
    python scripts/probe_esen_stress.py --nnp mace
The second run auto-merges and writes probe_esen_stress_compare.json with the
final verdict; exit code 0 = all gates pass.

Output: results/phase2/preflight/probe_esen_stress_{nnp}.json (+ _compare.json)
Usage:  python scripts/probe_esen_stress.py [--nnp esen] [--comps SrTiO3,ZnS,FeNi3]
                                            [--device cuda:0] [--no-smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import run_phase1 as rp          # frozen fleet protocol helpers (import-only)
import preflight_phase2 as pf2   # Phase-2 config/sampler/candidate helpers

OUT = rp.REPO / "results" / "phase2" / "preflight"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_COMPS = ["SrTiO3", "ZnS", "FeNi3"]   # oxide / sulfide / intermetallic
VOL_STRAIN = 1.02                          # +2% volume for the strain gate
DTRACE_MIN = 1e-3                          # eV/A^3, ~160 MPa at 1 eV/A^3
BETA = 38.68                               # 300 K (Phase-2 practice, pf2 B-cells)
ALPHA_LAT = 1e-4                           # frozen F-fix config
VOL_WALL = 2.0                             # frozen F-fix config

# Smoke cells: the F-fix validated worst cases, 1 candidate each.
SMOKE_CELLS = [("SrTiO3", 0.5, 42), ("ZnS", 0.5, 42)]


def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """ASE Voigt [xx,yy,zz,yz,xz,xy] -> (3,3) symmetric tensor."""
    s = np.array(v, dtype=float)
    return np.array([[s[0], s[5], s[4]],
                     [s[5], s[1], s[3]],
                     [s[4], s[3], s[2]]])


def ase_stress_atoms(calc, crystal):
    """Energy -> forces -> stress on the raw ASE interface (G1 path)."""
    atoms = crystal.ase_atoms
    atoms.calc = calc
    try:
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        stress = atoms.get_stress()          # Voigt (6,), eV/A^3
        return energy, np.asarray(forces, dtype=float), np.asarray(stress, dtype=float)
    finally:
        atoms.calc = None


def env_info(nnp: str) -> dict:
    import importlib.metadata as md
    info = {"nnp": nnp, "torch": "n/a"}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    for pkg in (["fairchem-core", "fairchem.core", "ocpmodels"] if nnp == "esen"
                else ["mace"]):
        try:
            info[pkg] = md.version(pkg)
        except Exception:
            info[pkg] = "not-installed"
    return info


def probe_nnp(nnp: str, comps: list[str], device: str, run_smoke: bool) -> dict:
    from materialgen.core.crystal import CrystalStructure
    from materialgen.core.score_function import NNPScore

    calc = rp.load_calculator(nnp, device)
    label = rp.nnp_label(nnp)
    print(f"[{nnp}] calc loaded ({label}), fairchem/mace env below")
    score = NNPScore(calc, beta=BETA, compute_stress=True, label=label)

    refs = rp.load_reference_structures(comps)
    out: dict = {"nnp": nnp, "comps": {}, "strain": {}, "gates": {}, "smoke": {}}

    # ---- G1 / G2 / G3 per composition ----
    for comp in comps:
        ref = refs[comp]
        cr = CrystalStructure.from_frac_coords(ref["numbers"], ref["frac_coords"],
                                               ref["lattice"])
        row: dict = {}
        # G1: raw ASE interface
        try:
            e, f, sv = ase_stress_atoms(calc, cr)
            sv = np.asarray(sv).ravel()
            finite = bool(np.all(np.isfinite(sv)) and sv.size == 6)
            row["G1_ok"] = finite
            row["raw_stress_voigt"] = [float(x) for x in sv] if finite else None
            row["G1_exc"] = None
            s_ff = np.linalg.norm(voigt_to_tensor(sv)) if finite else None
            row["raw_stress_fro"] = float(s_ff) if s_ff is not None else None
            row["raw_energy_eV"] = float(e)
            row["raw_max_f"] = float(np.linalg.norm(f, axis=1).max())
        except Exception as exc:
            row["G1_ok"] = False
            row["G1_exc"] = f"{type(exc).__name__}: {exc}"
            row["raw_stress_fro"] = None

        # G2: NNPScore pipeline (what the samplers actually consume)
        try:
            res_eq = score.compute(cr)
            has_stress = res_eq.stress is not None
            row["G2_ok"] = bool(has_stress
                                and res_eq.lattice_score is not None
                                and np.all(np.isfinite(res_eq.stress)))
            row["pipeline_stress_fro"] = (
                float(np.linalg.norm(res_eq.stress)) if has_stress else None)
            row["pipeline_lattice_score_fro"] = (
                float(np.linalg.norm(res_eq.lattice_score))
                if res_eq.lattice_score is not None else None)
            row["pipeline_energy_eV"] = float(res_eq.energy)
            row["pipeline_max_f"] = (
                float(np.linalg.norm(res_eq.forces, axis=1).max())
                if res_eq.forces is not None and len(res_eq.forces) else None)
            eq_trace = float(np.trace(res_eq.stress)) if has_stress else None
            # G3: +2% volume strain
            L2 = ref["lattice"] * VOL_STRAIN ** (1 / 3)
            cr2 = CrystalStructure.from_frac_coords(ref["numbers"],
                                                    ref["frac_coords"], L2)
            res_2 = score.compute(cr2)
            t2 = float(np.trace(res_2.stress)) if res_2.stress is not None else None
            dtrace = (t2 - eq_trace) if (eq_trace is not None and t2 is not None) else None
            row["strain_dtrace_eV_A3"] = dtrace
            row["strain_gate_ok"] = bool(
                dtrace is not None and abs(dtrace) > DTRACE_MIN)
        except Exception as exc:
            row["G2_ok"] = False
            row["G2_exc"] = f"{type(exc).__name__}: {exc}"
        out["comps"][comp] = row
        print(f"[{nnp}] {comp:7s} G1={row.get('G1_ok')} "
              f"G2={row.get('G2_ok')} "
              f"dtrace={row.get('strain_dtrace_eV_A3')}")

    out["gates"]["G1_interface"] = all(c["G1_ok"] for c in out["comps"].values())
    out["gates"]["G2_pipeline"] = all(c["G2_ok"] for c in out["comps"].values())
    out["gates"]["G3_strain_local"] = all(
        c.get("strain_gate_ok") for c in out["comps"].values())

    # ---- G4: end-to-end ALD upd_lat smoke (eSEN only; F-fix config) ----
    if run_smoke and nnp == "esen":
        hull = rp.HullEvaluator(rp.load_hull_table())
        for comp in sorted({c for c, _, _ in SMOKE_CELLS}):
            ref_crystal = CrystalStructure.from_frac_coords(
                refs[comp]["numbers"], refs[comp]["frac_coords"],
                refs[comp]["lattice"])
            e_ref = rp.calc_ref_energy(calc, ref_crystal)
            hull.calibrate(comp, e_ref, len(refs[comp]["numbers"]),
                           float(refs[comp]["metadata"].get(
                               "formation_energy_per_atom", 0.0)))
        out["smoke"]["config"] = {"cell": "S", "prot": "l1l4",
                                  "alpha_lat": ALPHA_LAT,
                                  "lat_vol_max": VOL_WALL, "frac": 0.1}
        for comp, sigma, seed in SMOKE_CELLS:
            t = {"sampler": "ald", "prot": "l1l4", "sigma": sigma,
                 "comp": comp, "seed": seed, "beta": BETA, "K": 100,
                 "sigma_min": 0.01, "update_lattice": True, "n_cand": 1,
                 "lat_vol_max": VOL_WALL, "alpha_lat": ALPHA_LAT,
                 "frac": 0.1, "dmin": 0.0}
            score_c = pf2.make_score2(calc, t["prot"], t["beta"], nnp=nnp)
            cfg = pf2.make_ald_config2(t["sigma"], t["prot"], t["K"],
                                       t["sigma_min"], t["update_lattice"], t)
            sampler = pf2.make_sampler(cfg, t["prot"], t["sampler"])
            c = pf2.run_candidate_traj(score_c, sampler, refs[comp], t["sigma"],
                                       t["seed"], 0, hull)
            out["smoke"][f"{comp}|s{sigma:g}"] = {
                "valid": c["valid"], "ood_steps": c["ood_steps"],
                "lat_drift": c.get("lat_drift"),
                "vol_max_ratio": c.get("vol_max_ratio"),
                "vol_final_ratio": c.get("vol_final_ratio"),
                "e_hull_eV_atom": c["e_hull"], "wall_time_s": c["wall_time_s"]}
            ld = c.get("lat_drift")
            print(f"[{nnp}] smoke {comp} valid={c['valid']} "
                  f"lat_drift={ld if ld is None else f'{ld:.2e}'} "
                  f"vol_max={c.get('vol_max_ratio')} "
                  f"vol_final={c.get('vol_final_ratio')}", flush=True)
        good = all(
            (r.get("valid") is True or (r.get("lat_drift") or 0.0) > 1e-6)
            and r.get("vol_max_ratio") is not None and r["vol_max_ratio"] <= VOL_WALL
            and r.get("vol_final_ratio") is not None
            and 0.5 <= r["vol_final_ratio"] <= VOL_WALL
            for k, r in out["smoke"].items() if k != "config")
        out["gates"]["G4_smoke"] = bool(good)
    else:
        out["gates"]["G4_smoke"] = None   # skipped on the MACE side

    out["env_info"] = env_info(nnp)
    return out


def merge_and_verdict(mace: dict, esen: dict) -> tuple[dict, int]:
    v: dict = {"mace": mace, "esen": esen, "verdict": {}}
    print("\n===== probe verdict =====")
    # G1/G2: eSEN side decisive.
    for g in ("G1_interface", "G2_pipeline"):
        v["verdict"][g] = "PASS" if esen["gates"].get(g) else "FAIL"
        print(f"{g:16s} {v['verdict'][g]}   (eSEN: {esen['gates'].get(g)})")
    # G3: local sign/magnitude for each NNP + cross-NNP sign agreement.
    ok = True
    for comp in PROBE_COMPS:
        cm = mace.get("comps", {}).get(comp, {})
        ce = esen.get("comps", {}).get(comp, {})
        d_m = cm.get("strain_dtrace_eV_A3")
        d_e = ce.get("strain_dtrace_eV_A3")
        same_sign = (d_m is not None and d_e is not None
                     and np.sign(d_m) == np.sign(d_e))
        mag_ok = (d_m is not None and d_e is not None
                  and 0.1 <= abs(d_e) / max(abs(d_m), 1e-12) <= 10.0)
        row_ok = bool(same_sign and mag_ok and d_e is not None
                      and abs(d_e) > DTRACE_MIN)
        ok &= row_ok
        dm_s = f"{d_m:+.4f}" if d_m is not None else "None"
        de_s = f"{d_e:+.4f}" if d_e is not None else "None"
        print(f"G3_strain {comp:7s} dtrace mace={dm_s} esen={de_s} "
              f"(same_sign={same_sign}, mag={mag_ok})")
    v["verdict"]["G3_strain"] = "PASS" if ok else "FAIL"
    # G4: eSEN smoke.
    g4 = esen["gates"].get("G4_smoke")
    v["verdict"]["G4_smoke"] = {True: "PASS", False: "FAIL"}.get(g4, "SKIP")
    print(f"{'G4_smoke':16s} {v['verdict']['G4_smoke']}")
    return v, 0 if all(x == "PASS" for x in v["verdict"].values()) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnp", default="esen", choices=["mace", "esen"])
    ap.add_argument("--comps", default=",".join(PROBE_COMPS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the G4 ALD upd_lat smoke trajectory")
    args = ap.parse_args()
    comps = [c.strip() for c in args.comps.split(",") if c.strip()]
    run_smoke = (args.nnp == "esen") and not args.no_smoke

    t0 = time.time()
    result = probe_nnp(args.nnp, comps, args.device, run_smoke)
    result["wall_time_s"] = time.time() - t0

    path = OUT / f"probe_esen_stress_{args.nnp}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {path}")

    other = OUT / ("probe_esen_stress_esen.json"
                   if args.nnp == "mace" else "probe_esen_stress_mace.json")
    if other.exists():
        other_data = json.loads(other.read_text())
        verdict, code = (merge_and_verdict(other_data, result)
                         if args.nnp == "esen"
                         else merge_and_verdict(result, other_data))
        cpath = OUT / "probe_esen_stress_compare.json"
        cpath.write_text(json.dumps(verdict, indent=1, default=str))
        print(f"wrote {cpath}")
        print("exit:", code)
        sys.exit(code)
    print(f"\n{args.nnp} probe done. Run the other backend "
          f"(--nnp {'mace' if args.nnp == 'esen' else 'esen'}) to get the "
          f"cross-NNP verdict.")


if __name__ == "__main__":
    main()
