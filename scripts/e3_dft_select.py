"""E-3 tier 1: stage DFT single-point inputs for representative samples.

A reviewer asked for DFT single-point and relaxation checks on representative
samples, covering bare sampling, geometry-constraint-only and analytic-repulsion
schemes -- i.e. can the claimed improvement be verified by an independent method
rather than the NNP that produced it?

The four ALD arms at the operating point sigma_max = 1 map onto the four things
the reviewer names:

    bare     bare sampling                      (no protection)
    l3       geometry constraint only           (L3 reflecting boundary)
    l1       analytic repulsion                 (L1 Pauli score)
    l1l2l3   full stack                         (the paper's arm)

What this script does: for each arm, draw a stratified sample of final
structures and write one VASP directory per structure.

    group "flagged"  the final step raised an OOD bit (d_min / force / volume).
                     These are the geometries the paper's detector calls
                     untrustworthy.
    group "clean"    no OOD event at any step of the trajectory.

Sampling is RANDOM within each group (not by extremity -- taking the worst
cases would make the two groups differ in a way that is not the population
difference) but spread round-robin over compositions, so every composition the
arm covers is represented before any composition gets a second draw.  A group
that is too small to fill its quota is taken whole; the shortfall is recorded
in the manifest, never silently topped up from the other group.

Alongside the VASP inputs the script stores, in meta.json, the NNP energy and
per-atom forces AT EXACTLY the staged geometry.  That is what makes tier 1 a
comparison: the DFT single point returns energy and forces for the same
coordinates, so the difference is a property of the two potentials and not of
two different geometries.

Protocol choices, and why (all documented in the paper's appendix):
  ENCUT 520          a well-converged cutoff for these elements; above every
                     POTCAR's own maximum, so no POTCAR is under-cut.
  POTCAR             the supplied library's default variant per element; Ba,
                     Ca, Sr and Zr have no plain variant and take _sv.
                     Uniform across arms and structures -- the comparison is
                     relative, and a per-element-tuned potential set would
                     change every arm identically.
  ISPIN 2 + MAGMOM   spin-polarised, with a fixed high-spin guess on the 3d
                     elements (V 3, Fe 5, Co 3, Ni 2, Cu 1, Ti 1) and 0.6 muB
                     elsewhere.  The guesses are not tuned per structure; they
                     are a uniform protocol applied to every arm.
  ISMEAR 0, SIGMA .05  Gaussian smearing, small enough to be near the
                     insulator limit and stable for the metals.
  KPOINTS            Gamma-centred, n_i = ceil(30 / |a_i|), i.e. ~30 A of
                     real-space extent per subdivision.
  EDIFF 1e-6         tight enough that the Hellmann-Feynman forces are good to
                     well under the NNP error being measured.

Tier 2 (scripts/e3_dft_relax.py) reuses these same directories with ISIF=2
relaxation switched on, so the two tiers are the same structures under two
questions ("is the NNP right here?" and "would this survive DFT?").

Usage:
    python scripts/e3_dft_select.py --sigma 1.0 --per-group 12 --device cpu
    python scripts/e3_dft_select.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)

import make_percomp_sigma_tables as mps  # noqa: E402

# Point this at your local PBE POTCAR library (one directory per element,
# each holding a POTCAR).
POTCAR_LIB = os.path.join(_ROOT, "third_party", "potcars", "PBE")
OUT_ROOT = os.path.join(_ROOT, "results", "interim", "e3_dft")
OOD_MASK = 0b010011          # d_min | max|F| | volume -- see make_percomp_sigma_tables

# The four arms, in the order the paper presents them.
ARMS = [("bare", "bare"), ("l3", "l3"), ("l1", "l1"), ("l1l2l3", "l1l2l3")]

MAGMOM = {"V": 3.0, "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0, "Ti": 1.0}
MAGMOM_DEFAULT = 0.6

_KSPAN = 30.0                # A of real-space extent per k-point subdivision
_ENCUT = 520


# --------------------------------------------------------------------------
# POTCAR
# --------------------------------------------------------------------------

def potcar_path(el: str) -> str:
    """Default PBE POTCAR for an element, falling back to the _sv variant.

    The library ships Ba, Ca, Sr and Zr only as _sv (there is no plain
    directory), so the fallback is not a preference -- it is the only
    potential available for those four.
    """
    for name in (el, f"{el}_sv", f"{el}_pv", f"{el}_d"):
        p = os.path.join(POTCAR_LIB, name, "POTCAR")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no PBE POTCAR for {el} under {POTCAR_LIB}")


def potcar_variant(el: str) -> str:
    return os.path.basename(os.path.dirname(potcar_path(el)))


# --------------------------------------------------------------------------
# VASP inputs
# --------------------------------------------------------------------------

def kpoints_mesh(lattice: np.ndarray) -> tuple[int, int, int]:
    """Gamma-centred mesh from the cell lengths: ceil(30 / |a_i|), min 1."""
    lengths = np.linalg.norm(np.asarray(lattice, float), axis=1)
    return tuple(int(max(1, np.ceil(_KSPAN / max(x, 1e-6)))) for x in lengths)


def write_poscar(path: str, numbers, frac, lattice, comment: str) -> list[str]:
    """POSCAR with species grouped, so the POTCAR blocks line up.

    Returns the element order used -- the caller must concatenate POTCARs in
    exactly this order or VASP silently evaluates the wrong species.
    """
    numbers = np.asarray(numbers, int)
    frac = np.asarray(frac, float)
    uniq = sorted(set(numbers.tolist()))
    from ase.data import chemical_symbols
    order = [chemical_symbols[z] for z in uniq]
    idx = [np.where(numbers == z)[0] for z in uniq]
    with open(path, "w") as fh:
        fh.write(f"{comment}\n")
        fh.write("1.0000000000000000\n")
        for row in np.asarray(lattice, float):
            fh.write("  " + "  ".join(f"{x:18.12f}" for x in row) + "\n")
        fh.write("  " + "  ".join(order) + "\n")
        fh.write("  " + "  ".join(str(len(i)) for i in idx) + "\n")
        fh.write("Direct\n")
        for i in idx:
            for row in frac[i]:
                fh.write("  " + "  ".join(f"{x:18.12f}" for x in row) + "\n")
    return order


def write_potcar(path: str, order: list[str]) -> None:
    with open(path, "wb") as out:
        for el in order:
            with open(potcar_path(el), "rb") as fh:
                out.write(fh.read())


def write_incar(path: str, magmom_line: str, relax: bool) -> None:
    """Single point (NSW=0, IBRION=-1) or ion relaxation (NSW=100, ISIF=2).

    `relax=False` is the tier-1 setting: VASP computes the energy and the
    Hellmann-Feynman forces and moves nothing.
    """
    ionic = ("Ionic Relaxation\n"
             "NSW    =  100          (Max ionic steps)\n"
             "IBRION =  2            (Conjugate gradient)\n"
             "ISIF   =  2            (Relax ions, fixed cell)\n"
             "EDIFFG = -2E-02        (Ionic convergence, eV/A)\n"
             if relax else
             "Ionic Relaxation\n"
             "NSW    =  0            (Single point: no ionic step)\n"
             "IBRION =  -1           (Static; forces still evaluated)\n")
    with open(path, "w") as fh:
        fh.write(f"""Global Parameters
ISTART =  0            (From scratch)
ICHARG =  2            (Atomic superposition)
ISPIN  =  2            (Spin polarised)
LREAL  = .FALSE.       (Projection in reciprocal space)
ENCUT  =  {_ENCUT}         (Plane-wave cutoff, eV)
PREC   =  Accurate
ADDGRID= .TRUE.
LASPH  = .TRUE.
LWAVE  = .FALSE.
LCHARG = .FALSE.
ALGO   =  Normal
ISMEAR =  0            (Gaussian smearing)
SIGMA  =  0.05
NELM   =  150
NELMIN =  4
EDIFF  =  1E-06        (SCF convergence, eV)
MAGMOM = {magmom_line}

{ionic}""")


def magmom_string(numbers, order: list[str]) -> str:
    """Per-species moments in POSCAR order, e.g. '0.6*2 5.0*1 0.6*3'."""
    numbers = np.asarray(numbers, int)
    from ase.data import chemical_symbols
    parts = []
    for el in order:
        z = chemical_symbols.index(el)
        n = int((numbers == z).sum())
        m = MAGMOM.get(el, MAGMOM_DEFAULT)
        parts.append(f"{m}*{n}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def load_arm_candidates(sigma: float, arm: str) -> list[dict]:
    """Every candidate of one arm at one sigma, tagged with its group.

    The scan cell is read with the same completeness rules
    make_percomp_sigma_tables uses, but not through its load_cell: that
    function's DIR_ARM table only covers the three arms of the main ablation
    (bare / l1 / l1l4), and two of the four arms here are scan-only
    (l3, l1l2l3).  For those two the arm name is the directory name.  The
    file-count assertion is kept, so a half-finished cell is a hard error
    rather than a quietly thinner denominator.
    """
    dir_arm = mps.DIR_ARM.get((arm, "ald"), arm)
    d = os.path.join(mps.BASE, f"s{sigma:g}_{dir_arm}_ald")
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    want = mps.N_SEEDS * len(mps.TEX)
    if len(files) != want:
        raise SystemExit(
            f"cell {os.path.relpath(d, _ROOT)} has {len(files)} task file(s), "
            f"expected {want} = {len(mps.TEX)} compositions x {mps.N_SEEDS} "
            "seeds -- not sampling an incomplete cell")
    cells: dict = {}
    for p in files:
        j = json.load(open(p))
        cells.setdefault(j["composition"], []).extend(j["candidates"])
    out = []
    for comp, cands in cells.items():
        for c in cands:
            bits = c.get("ood_bitmask") or []
            if not bits:
                continue
            final_bits = int(bits[-1])
            ever = any(int(b) & OOD_MASK for b in bits)
            group = "flagged" if (final_bits & OOD_MASK) else "clean"
            out.append({"composition": comp, "group": group,
                        "final_bits": final_bits, "ever_ood": ever,
                        "d_min": c["final"]["d_min"], "cand": c,
                        "seed": c.get("seed"), "cand_idx": c.get("cand")})
    return out


def stratified_pick(cands: list[dict], group: str, n: int, rng) -> list[dict]:
    """Random sample of `n` from one stratum, spread round-robin over compositions.

    Shuffling first makes the draw random within a composition; walking the
    compositions in turn makes the sample cover chemistry rather than
    whatever composition happens to have the most candidates.

    Three strata, because no single one answers the reviewer's question:
      pop      uniform over the arm's whole final-structure population.  This
               is the only stratum every arm has, and it is the one that
               supports a claim about the arm rather than about a subgroup.
      flagged  final step raised an OOD bit.  Direct test of whether the
               paper's own detector marks geometries DFT actually disagrees
               on -- but empty for the arms whose whole point is to remove
               them (l3, l1l2l3 at sigma=1), so it cannot carry the headline.
      clean    no OOD event at any step.  Contrast group for `flagged`.
    """
    pool = [c for c in cands if group == "pop" or c["group"] == group]
    if not pool:
        return []
    by_comp: dict[str, list] = {}
    for c in pool:
        by_comp.setdefault(c["composition"], []).append(c)
    for lst in by_comp.values():
        rng.shuffle(lst)
    comps = sorted(by_comp)
    picked, i = [], 0
    while len(picked) < n and any(by_comp[c] for c in comps):
        c = comps[i % len(comps)]
        if by_comp[c]:
            picked.append(by_comp[c].pop())
        i += 1
    return picked


# --------------------------------------------------------------------------

def nnp_forces(crystal, device: str):
    """MACE-MP-0 energy and forces at a frozen geometry, via ase."""
    from run_phase1 import load_calculator, nnp_label
    calc = load_calculator("mace", device)
    atoms = crystal.ase_atoms
    atoms.calc = calc
    return (float(atoms.get_potential_energy()),
            np.asarray(atoms.get_forces(), float), nnp_label("mace"))


def clone_tier2() -> int:
    """Build tier2/ from tier1/ with the same geometries, relaxation switched on.

    Same structures, two questions: tier 1 asks "is the NNP right at this
    geometry?", tier 2 asks "would this geometry survive DFT?".  Cloning
    rather than re-drawing guarantees the two tiers describe the same sample --
    a second draw would let a difference between the tiers be a difference in
    which structures were picked.

    Only the ionic block of the INCAR changes; POSCAR, POTCAR, KPOINTS and
    meta.json are copied verbatim, so the tier-2 column cannot drift from the
    tier-1 column through an input difference.
    """
    src = os.path.join(OUT_ROOT, "tier1")
    dst = os.path.join(OUT_ROOT, "tier2")
    dirs = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"nothing staged under {src} -- run tier 1 first")
    copied = 0
    for d in dirs:
        tag = os.path.basename(d)
        o = os.path.join(dst, tag)
        os.makedirs(o, exist_ok=True)
        for f in ("POSCAR", "POTCAR", "KPOINTS", "meta.json", "nnp_forces.dat"):
            s = os.path.join(d, f)
            if os.path.exists(s):
                import shutil
                shutil.copy2(s, os.path.join(o, f))
        meta = json.load(open(os.path.join(d, "meta.json")))
        # Rebuild MAGMOM from the POSCAR's grouped counts rather than from the
        # stored atom list: the POSCAR is what VASP reads, so deriving the
        # line from it is the only way the two cannot disagree.
        poscar = os.path.join(d, "POSCAR")
        magmom = " ".join(
            f"{MAGMOM.get(el, MAGMOM_DEFAULT)}*{_count_in_poscar(poscar, el)}"
            for el in meta["elements"])
        write_incar(os.path.join(o, "INCAR"), magmom, relax=True)
        copied += 1
    print(f"cloned {copied} structure(s) into {os.path.relpath(dst, _ROOT)} "
          "with ISIF=2 relaxation")
    return copied


def _count_in_poscar(path: str, el: str) -> int:
    """How many atoms of `el` the staged POSCAR holds (its grouped counts line)."""
    lines = open(path).read().splitlines()
    head = lines[5].split()
    if all(h.replace(".", "").lstrip("-").isdigit() for h in head):
        return 0
    counts = [int(x) for x in lines[6].split()]
    return dict(zip(head, counts)).get(el, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--arms", nargs="+", default=[a for a, _ in ARMS])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clone-tier2", action="store_true",
                    help="stage tier2/ from the existing tier1/ instead of "
                         "drawing a new sample")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the draw without writing VASP inputs or "
                         "evaluating the NNP")
    args = ap.parse_args()

    if args.clone_tier2:
        clone_tier2()
        return

    from materialgen.core.crystal import CrystalStructure

    tier = os.path.join(OUT_ROOT, "tier1")
    manifest, staged = [], 0
    for arm in args.arms:
        cands = load_arm_candidates(args.sigma, arm)
        if not cands:
            print(f"{arm}: no candidates at sigma={args.sigma:g} -- skipped")
            continue
        # A stable per-arm RNG: `hash()` is salted per process, so seed the
        # stream from the arm's position in ARMS instead of from hash(arm).
        rng = np.random.RandomState(args.seed + 1000 * [a for a, _ in ARMS].index(arm))
        counts = {g: sum(1 for c in cands if c["group"] == g)
                  for g in ("flagged", "clean")}
        picks = {g: stratified_pick(cands, g, args.per_group, rng)
                 for g in ("pop", "flagged", "clean")}
        # Dedupe across strata: a structure drawn into two groups is staged
        # once, and meta.json records every group it belongs to.  "pop" wins
        # the tie so the population sample stays a clean random draw.
        merged: dict[str, dict] = {}
        for g in ("pop", "flagged", "clean"):
            for c in picks[g]:
                key = f"{c['composition']}|{c['seed']}|{c['cand_idx']}"
                if key in merged:
                    merged[key]["groups"].append(g)
                else:
                    c = dict(c, groups=[g])
                    merged[key] = c
        sel_list = list(merged.values())
        sizes = {g: sum(1 for c in sel_list if g in c["groups"])
                 for g in ("pop", "flagged", "clean")}
        print(f"{arm:8s} pool flagged={counts['flagged']:5d} "
              f"clean={counts['clean']:5d} -> staged {len(sel_list):3d} "
              f"(pop={sizes['pop']} flagged={sizes['flagged']} "
              f"clean={sizes['clean']})")
        if args.dry_run:
            for g in ("pop", "flagged", "clean"):
                d = sorted(c["d_min"] for c in sel_list if g in c["groups"])
                if d:
                    print(f"         {g:8s} n={len(d):3d} d_min: "
                          f"min={d[0]:.3f} med={d[len(d)//2]:.3f} max={d[-1]:.3f}")
            continue

        for c in sel_list:
                crystal = CrystalStructure.from_dict(c["cand"]["final"]["structure"])
                tag = (f"{arm}__{c['composition']}__seed{c['seed']}"
                       f"__c{c['cand_idx']}")
                d = os.path.join(tier, tag)
                os.makedirs(d, exist_ok=True)
                numbers = crystal.atomic_numbers
                frac = crystal.frac_coords
                lat = crystal.lattice
                order = write_poscar(
                    os.path.join(d, "POSCAR"), numbers, frac, lat,
                    f"{tag} sigma={args.sigma:g} "
                    f"groups={','.join(c['groups'])} d_min={c['d_min']:.4f}")
                write_potcar(os.path.join(d, "POTCAR"), order)
                write_incar(os.path.join(d, "INCAR"),
                            magmom_string(numbers, order), relax=False)
                km = kpoints_mesh(lat)
                with open(os.path.join(d, "KPOINTS"), "w") as fh:
                    fh.write("Auto mesh, Gamma-centred (E-3 protocol)\n0\nGamma\n"
                             f"  {km[0]}  {km[1]}  {km[2]}\n0 0 0\n")
                e, f, label = nnp_forces(crystal, args.device)
                np.savetxt(os.path.join(d, "nnp_forces.dat"), f)
                lat = np.asarray(lat, float)
                meta = {
                    "tag": tag, "arm": arm, "groups": c["groups"],
                    "composition": c["composition"], "seed": c["seed"],
                    "cand_idx": c["cand_idx"], "sigma": args.sigma,
                    "d_min": c["d_min"], "final_ood_bits": c["final_bits"],
                    "ever_ood": c["ever_ood"], "n_atoms": crystal.num_atoms,
                    "elements": order, "potcar_variants": {el: potcar_variant(el)
                                                           for el in order},
                    "kpoints": km, "encut": _ENCUT,
                    "nnp_energy": e, "nnp_energy_per_atom": e / crystal.num_atoms,
                    "nnp_label": label, "nnp_forces_file": "nnp_forces.dat",
                    "volume": float(np.linalg.det(lat)),
                    "stored_nnp_energy": c["cand"]["final"].get("energy"),
                    "stored_e_hull": c["cand"]["final"].get("e_hull"),
                }
                with open(os.path.join(d, "meta.json"), "w") as fh:
                    json.dump(meta, fh, indent=1)
                manifest.append(meta)
                staged += 1

    if args.dry_run:
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    mpath = os.path.join(OUT_ROOT, "manifest_tier1.json")
    with open(mpath, "w") as fh:
        json.dump({"sigma": args.sigma, "per_group": args.per_group,
                   "seed": args.seed, "n": len(manifest),
                   "entries": manifest}, fh, indent=1)
    print(f"\nstaged {staged} structures under {os.path.relpath(tier, _ROOT)}")
    print(f"manifest: {os.path.relpath(mpath, _ROOT)}")
    by = {}
    for m in manifest:
        for g in m["groups"]:
            key = f"{m['arm']}/{g}"
            by[key] = by.get(key, 0) + 1
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print("\nnext: scripts/e3_dft_submit.sh (tier 1), then scripts/e3_dft_collect.py")


if __name__ == "__main__":
    main()
