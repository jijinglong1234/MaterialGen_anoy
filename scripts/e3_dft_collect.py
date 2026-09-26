"""E-3: read the DFT jobs back and compare them with the NNP.

Tier 1 (single points, scripts/e3_dft_select.py staged the inputs) answers the
reviewer's question directly: at the geometries each protection arm produced,
where is the NNP wrong according to DFT?

Two numbers, and only the first one is the headline:

  force RMSE   ||F_DFT - F_NNP|| over all atoms, per structure.  Forces are
               comparable between the two methods with no reference-state
               correction -- this is a property of the two potentials at one
               geometry, which is exactly what a single point buys.

  energy excess  (E_DFT - E_NNP)/N_atom, then subtract that composition's own
               mean over its clean structures.  Total energies are not on a
               common zero between a PAW calculation and a learned potential,
               so the raw difference is meaningless; removing a per-composition
               offset leaves "how much worse is the NNP here than its usual
               error for this chemistry", which is the quantity the flagged-
               vs-clean contrast is about.  Reported, not headlined: with 12
               structures per stratum and a per-composition offset fitted on
               the same sample, it is the noisier of the two.

Non-convergence is a result, not a hole: a geometry on which the SCF will not
reach EDIFF is a geometry DFT itself cannot describe with these settings, and
the per-arm share of such structures is reported alongside the RMSE rather
than dropped silently.  Those structures contribute a row with no force RMSE
and a `scf_converged: false` flag; every table states both denominators.

Tier 2 (relaxations) reuses the same parser and adds the two relaxation
outcomes: how far the ions move, and how much energy the relaxation recovers.

Output: results/phase1/analysis/e3_dft_tier<N>.json (+ stdout tables).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "results", "phase1", "analysis")


# --------------------------------------------------------------------------
# OUTCAR / OSZICAR parsing
# --------------------------------------------------------------------------

def parse_outcar(path: str) -> dict:
    """Energy, forces and convergence from one OUTCAR.

    Anchors, all of them literal VASP output that has been stable across 5.x
    and 6.x (verified against the reference run in apps/test):
      "free  energy   TOTEN  ="   -> the final total energy
      "POSITION ... TOTAL-FORCE"  -> the last force block, 6 columns per row,
                                     first three cartesian in Angstrom, last
                                     three the force in eV/Angstrom
      "aborting loop because EDIFF is reached"  -> the electronic loop exited
                                     on convergence rather than on NELM
      "reached required accuracy" -> ionic loop converged (tier 2 only)
    """
    out = {"energy": None, "forces": None, "scf_converged": False,
           "ionic_converged": None, "n_ionic_steps": None,
           "final_positions": None, "final_lattice": None}
    if not os.path.exists(path):
        return out
    txt = open(path, errors="ignore").read()
    lines = txt.splitlines()

    for ln in reversed(lines):
        if "free  energy   TOTEN  =" in ln:
            out["energy"] = float(ln.split("=")[1].split()[0])
            break
    out["scf_converged"] = "aborting loop because EDIFF is reached" in txt

    idx = [i for i, ln in enumerate(lines) if "TOTAL-FORCE" in ln]
    if idx:
        i = idx[-1]
        rows = []
        for ln in lines[i + 2:]:
            if ln.strip().startswith("---") or not ln.strip():
                break
            parts = ln.split()
            if len(parts) < 6:
                break
            rows.append([float(x) for x in parts[3:6]])
        if rows:
            out["forces"] = np.asarray(rows, float)

    # Ionic step count: VASP labels each ionic block in OSZICAR with "F=".
    osz = os.path.join(os.path.dirname(path), "OSZICAR")
    if os.path.exists(osz):
        n = sum(1 for ln in open(osz, errors="ignore") if "F=" in ln)
        out["n_ionic_steps"] = n
    out["ionic_converged"] = "reached required accuracy" in txt

    cont = os.path.join(os.path.dirname(path), "CONTCAR")
    if os.path.exists(cont):
        lat, pos = read_poscar(cont)
        if lat is not None:
            out["final_lattice"] = lat
            out["final_positions"] = pos
    return out


def read_poscar(path: str):
    """Lattice (3,3) and cartesian positions (N,3) from a POSCAR/CONTCAR."""
    try:
        lines = open(path).read().splitlines()
        scale = float(lines[1].split()[0])
        lat = np.array([[float(x) for x in lines[2 + k].split()[:3]]
                        for k in range(3)]) * scale
        head = lines[5].split()
        # VASP 5/6 writes the symbols on line 6 unless it is a bare count line
        if all(h.replace(".", "").lstrip("-").isdigit() for h in head):
            symbols = None
            counts = [int(x) for x in head]
            start = 6
        else:
            symbols = head
            counts = [int(x) for x in lines[6].split()]
            start = 7
        mode = lines[start].strip().lower()
        cartesian = mode.startswith(("c", "k"))
        n = sum(counts)
        coords = np.array([[float(x) for x in lines[start + 1 + k].split()[:3]]
                           for k in range(n)])
        if not cartesian:
            coords = coords @ lat
        else:
            coords = coords * scale
        return lat, coords
    except Exception:
        return None, None


def read_poscar_symbols(path: str):
    lines = open(path).read().splitlines()
    head = lines[5].split()
    if all(h.replace(".", "").lstrip("-").isdigit() for h in head):
        return None
    out = []
    for s, c in zip(head, [int(x) for x in lines[6].split()]):
        out += [s] * c
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def structure_row(d: str, tier: int) -> dict | None:
    meta_p = os.path.join(d, "meta.json")
    if not os.path.exists(meta_p):
        return None
    meta = json.load(open(meta_p))
    res = parse_outcar(os.path.join(d, "OUTCAR"))
    row = {**{k: meta[k] for k in
              ("tag", "arm", "groups", "composition", "seed", "cand_idx",
               "d_min", "n_atoms", "sigma", "final_ood_bits", "ever_ood",
               "nnp_energy", "nnp_energy_per_atom", "elements", "kpoints")},
           "dir": os.path.relpath(d, _ROOT),
           "ran": res["energy"] is not None,
           "scf_converged": res["scf_converged"],
           "dft_energy": res["energy"]}

    nnp_f_p = os.path.join(d, "nnp_forces.dat")
    nnp_f = np.loadtxt(nnp_f_p) if os.path.exists(nnp_f_p) else None
    if res["energy"] is not None:
        row["dft_energy_per_atom"] = res["energy"] / meta["n_atoms"]
        row["energy_excess_raw"] = (res["energy"] / meta["n_atoms"]
                                    - meta["nnp_energy"] / meta["n_atoms"])
    if res["forces"] is not None and nnp_f is not None and len(res["forces"]) == len(nnp_f):
        diff = res["forces"] - nnp_f
        row["force_rmse"] = float(np.sqrt((diff ** 2).sum() / len(diff)))
        row["force_mae"] = float(np.abs(diff).mean())
        row["dft_force_max"] = float(np.linalg.norm(res["forces"], axis=1).max())
        row["nnp_force_max"] = float(np.linalg.norm(nnp_f, axis=1).max())
    if tier == 2:
        lat0, pos0 = read_poscar(os.path.join(d, "POSCAR"))
        if (res["final_positions"] is not None and lat0 is not None
                and len(pos0) == len(res["final_positions"])):
            # Displacement in fractional space, then through the *initial*
            # cell -- the relaxation holds the cell fixed (ISIF=2), so the two
            # cells agree and this is a plain cartesian RMSD.
            df = (res["final_positions"] - pos0) @ np.linalg.inv(lat0)
            df -= np.round(df)
            row["relax_rmsd"] = float(np.sqrt(
                (((df @ lat0) ** 2).sum(axis=1)).mean()))
        if row.get("dft_energy_per_atom") is not None:
            row["relax_n_ionic_steps"] = res["n_ionic_steps"]
            row["relax_ionic_converged"] = res["ionic_converged"]
    return row


def add_energy_excess(rows: list[dict]) -> None:
    """Turn the raw DFT-NNP offset into an excess over each chemistry's own.

    A PAW total energy and a learned potential's total energy do not share a
    zero, so the per-structure difference carries an arbitrary per-composition
    constant.  Estimate that constant from the composition's clean structures
    only -- fitting it on structures that include the flagged ones would let
    the thing being measured define its own baseline -- and subtract it.  What
    is left is how far this geometry's NNP error sits from the NNP's usual
    error for this chemistry.
    """
    by_comp: dict[str, list] = {}
    for r in rows:
        if r.get("energy_excess_raw") is not None:
            by_comp.setdefault(r["composition"], []).append(r)
    for comp, sel in by_comp.items():
        base_pool = [r for r in sel if "clean" in r["groups"] and r["scf_converged"]]
        if not base_pool:
            base_pool = sel
        base = float(np.mean([r["energy_excess_raw"] for r in base_pool]))
        for r in sel:
            r["energy_excess"] = r["energy_excess_raw"] - base
            r["energy_offset_used"] = base


def summarize(rows: list[dict], keys, label: str) -> list[dict]:
    """Mean/median force RMSE per stratum, with both denominators stated."""
    out = []
    for arm, group in keys:
        sel = [r for r in rows if r["arm"] == arm and group in r["groups"]]
        if not sel:
            continue
        ran = [r for r in sel if r["ran"]]
        conv = [r for r in ran if r["scf_converged"]]
        rm = [r["force_rmse"] for r in conv if r.get("force_rmse") is not None]
        ex = [r["energy_excess"] for r in conv if r.get("energy_excess") is not None]
        out.append({
            "arm": arm, "group": group, "n_staged": len(sel), "n_ran": len(ran),
            "n_scf_converged": len(conv),
            "scf_fail_share_ran": (1 - len(conv) / len(ran)) if ran else None,
            "force_rmse_mean": float(np.mean(rm)) if rm else None,
            "force_rmse_median": float(np.median(rm)) if rm else None,
            "force_rmse_q90": float(np.percentile(rm, 90)) if rm else None,
            "force_rmse_max": float(np.max(rm)) if rm else None,
            "d_min_median": float(np.median([r["d_min"] for r in sel])),
            "d_min_min": float(np.min([r["d_min"] for r in sel])),
            "energy_excess_median": float(np.median(ex)) if ex else None,
        })
        fail = (1 - len(conv) / len(ran)) * 100 if ran else float("nan")
        print(f"{label:8s} {arm:8s} {group:8s} staged={len(sel):3d} ran={len(ran):3d} "
              f"conv={len(conv):3d} SCFfail={fail:5.1f}% "
              f"RMSE mean={np.mean(rm) if rm else float('nan'):6.3f} "
              f"med={np.median(rm) if rm else float('nan'):6.3f} "
              f"max={np.max(rm) if rm else float('nan'):6.3f}  "
              f"d_min~{np.median([r['d_min'] for r in sel]):.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.path.join(_ROOT, "results", "interim", "e3_dft", f"tier{args.tier}")
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no staged structures under {root}")
    rows = [r for r in (structure_row(d, args.tier) for d in dirs) if r]
    add_energy_excess(rows)
    print(f"tier {args.tier}: {len(rows)} staged, "
          f"{sum(1 for r in rows if r['ran'])} produced an OUTCAR, "
          f"{sum(1 for r in rows if r['scf_converged'])} converged the SCF\n")

    keys = [(a, g) for a in sorted({r["arm"] for r in rows})
            for g in ("pop", "flagged", "clean")]
    print("force RMSE (eV/A) of NNP against DFT at the staged geometry:")
    table = summarize(rows, keys, f"tier{args.tier}")

    # The within-arm flagged-vs-clean contrast, which is the direct test of
    # the detector (only arms that still have flagged structures can show it).
    print("\nflagged vs clean, within arms that have both:")
    for arm in sorted({r["arm"] for r in rows}):
        a = [t for t in table if t["arm"] == arm]
        fl = [t for t in a if t["group"] == "flagged" and t["force_rmse_mean"]]
        cl = [t for t in a if t["group"] == "clean" and t["force_rmse_mean"]]
        if fl and cl:
            ratio = fl[0]["force_rmse_mean"] / cl[0]["force_rmse_mean"]
            print(f"  {arm:8s} flagged {fl[0]['force_rmse_mean']:.3f} vs "
                  f"clean {cl[0]['force_rmse_mean']:.3f} eV/A  ->  x{ratio:.2f}")

    if args.tier == 2:
        print("\nrelaxation outcome:")
        print(f"{'arm':8s} {'group':8s} {'n':>4s} {'RMSD_med':>9s} "
              f"{'RMSD_max':>9s} {'DE/atom_med':>12s}")
        for t in table:
            sel = [r for r in rows if r["arm"] == t["arm"] and t["group"] in r["groups"]
                   and r.get("relax_rmsd") is not None]
            if not sel:
                continue
            rm = [r["relax_rmsd"] for r in sel]
            print(f"{t['arm']:8s} {t['group']:8s} {len(sel):4d} "
                  f"{np.median(rm):9.3f} {np.max(rm):9.3f}")

    # Mark what finished, so a re-submit skips it.  A run counts as finished
    # when it produced an energy -- an SCF that hit NELM still wrote one, and
    # re-running it would reproduce the same non-convergence at the same cost.
    # The marker is written by this script rather than by the Slurm job, so
    # "DONE" always means "collected", which is the property the submit script
    # actually needs.
    marked = 0
    for r in rows:
        if r["ran"]:
            with open(os.path.join(_ROOT, r["dir"], "DONE"), "w") as fh:
                fh.write(f"tier={args.tier} ran={r['ran']} "
                         f"scf_converged={r['scf_converged']}\n")
            marked += 1
    if marked:
        print(f"marked {marked} finished run(s) DONE "
              "(re-submitting will skip them)")

    out = args.out or os.path.join(OUT_DIR, f"e3_dft_tier{args.tier}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"tier": args.tier, "n_staged": len(rows),
                   "n_ran": sum(1 for r in rows if r["ran"]),
                   "n_scf_converged": sum(1 for r in rows if r["scf_converged"]),
                   "summary": table, "structures": rows}, fh, indent=1,
                  default=float)
    print(f"\nwrote {os.path.relpath(out, _ROOT)}")


if __name__ == "__main__":
    main()
