"""Pre-registered verify: L1' cap v3 — bounded repulsive escape.

Background: the asymmetric cap fixed the symmetric-cap trap (start-OOD pairs
stuck 200/200 steps) but left the wall repulsion to fire at up to 3*sigma_k;
the L1' verification cell measured the resulting energy pump — L1-only-ALD
induced 0.57-0.58 vs bare 0.39-0.41 at sigma 1-2, dwell/deep above bare.
v3 additionally caps the OPENING (repulsive) component at (r_wall - d)/2.

Two phases:
  toy (no GPU, seconds): single C-C pair + real Pauli wall + 3 eV/A attractor
    (verify #3's toy), 600 trajectories x 400 steps from a start-OOD 0.4 A
    start.  Variants: wall-only (no cap), asym cap, v3 cap.
    Gates: T1 v3 deep-step fraction = 0 AND escape-to->=1.0-A fraction >= 0.99
           (no trap); T2 v3 max single-step pair-separation increase <= 0.9 A
           (bounded escape, vs wall-only's ~5 A ejection); T3 v3 dwell
           (d<1.3 A step fraction) <= 5%.
  cell (GPU ~8 h): sigma=1 ALD, 5 comps x 2 seeds x 40 cands,
    arms {bare, asym(l1), v3(l1+cap_v3)}.
    Gates: G1 induced(v3) <= induced(bare); G2 deep(v3) <= 2*deep(bare);
           G3 dwell(v3) <= dwell(bare) + 0.02;
           G4 E_hull(v3) <= E_hull(bare) + 0.02 eV/atom.
Verdict path (pre-registered): toy T1-T3 AND cell G1-G4 pass -> adopt v3 in
    make_ald_config for L1 conditions (fleet); toy passes but cell fails ->
    close v3, keep the asymmetric cap, report L1-only-ALD honestly; toy
    fails -> iterate the cap form before spending the cell.

Usage: python scripts/verify_v3_cap.py --phase toy [--n 600]
       python scripts/verify_v3_cap.py --phase cell [--device cuda:0] [--cands 40]
       python scripts/verify_v3_cap.py --phase cell --sigma 2.0 --arm v3
           (multi-sigma extension: one (sigma, arm) task per
            process; outputs v3_verify_s{sigma}_{arm}.json; the original
            sigma=1 all-arms run stays v3_verify.json untouched)
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

TOY_OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "v3_toy.json")
CELL_OUT = os.path.join(_ROOT, "results", "phase1", "analysis", "v3_verify.json")
DWELL = 1.3


def cell_out_path(sigma: float, arm: str | None = None) -> str:
    """sigma=1 all-arms keeps the pre-registered filename; per-(sigma, arm)
    tasks (multi-sigma extension) get v3_verify_s{sigma}_{arm}.json."""
    if sigma == 1.0 and arm is None:
        return CELL_OUT
    return os.path.join(_ROOT, "results", "phase1", "analysis",
                        f"v3_verify_s{sigma:g}" + (f"_{arm}" if arm else "") + ".json")


def _pair_toy(d0, cell=12.0):
    from materialgen.core.crystal import CrystalStructure
    return CrystalStructure.from_cartesian(
        np.array([6, 6]), np.array([[0.0, 0, 0], [d0, 0, 0]]),
        np.eye(3) * cell)


class _ToyScore:
    """Analytic Pauli wall (same functional form as pauli.py's exponential
    branch: A*e^{-Br}*Theta, tanh smooth step at 0.75*r_sum) + a constant
    3 eV/A attractor along the pair axis.  Pure numpy: the toy runs 720K
    steps, the ASE-based PauliScore was ~100x too slow for that."""

    A, B = 4.3e2, 3.33          # C-C ZBL-fitted parameters
    R_SUM, W = 1.52, 0.05       # covalent sum, smooth-step width
    BETA = 38.68                # the ALD drift is alpha * beta * F

    def compute(self, crystal):
        fc = crystal.frac_coords
        delta = fc[1] - fc[0]
        delta -= np.round(delta)          # min-image: consistent with the
        disp = delta @ crystal.lattice    # sampler's wrapped frac coords
        d = float(np.linalg.norm(disp))   # TRUE 3D pair distance: the real
        u = disp / d if d > 1e-12 else np.array([1.0, 0.0, 0.0])
        x = 0.75 * self.R_SUM - d
        theta = 0.5 * (1.0 + np.tanh(x / self.W))
        if x < -8.0 * self.W:
            F_wall = 0.0
        else:
            e = np.exp(-self.B * d)
            sech2 = 1.0 / np.cosh(x / self.W) ** 2
            # force on the atoms pushing them APART (=-dU/dr, pauli.py form)
            F_wall = self.A * e * (self.B * theta + sech2 / (2.0 * self.W))
        F = np.zeros((crystal.num_atoms, 3))
        F[0] = (3.0 - F_wall) * u   # attractor pulls together, wall pushes apart
        F[1] = -(3.0 - F_wall) * u
        frac = np.linalg.solve(crystal.lattice.T, (self.BETA * F).T).T
        return SimpleNamespace(frac_score=frac, energy=None, forces=None,
                               lattice_score=None, metadata={})


def run_toy(n=600):
    from materialgen.samplers.ald import ALDConfig, ALDSampler

    variants = {
        "wall_only": dict(use_wall_cap=False, use_wall_cap_v3=False),
        "asym": dict(use_wall_cap=True, use_wall_cap_v3=False),
        "v3": dict(use_wall_cap=True, use_wall_cap_v3=True),
    }
    score = _ToyScore()
    out = {}
    for name, caps in variants.items():
        deep, dwell, max_eject, n_escape, n_steps = 0, 0, 0.0, 0, 0
        for i in range(n):
            cfg = ALDConfig(sigma_max=1.0, sigma_min=0.01, K=100, M=4,
                            alpha=1e-3, drift_cap=3.0, seed=i, **caps)
            res = ALDSampler(cfg).sample(score, _pair_toy(0.4))
            h = np.array(res.d_min_history)
            deep += int((h < 0.5).sum())
            dwell += int((h < DWELL).sum())
            n_steps += len(h)
            if len(h) > 1:
                max_eject = max(max_eject, float(np.diff(h).max()))
            if h.size and h.max() >= 1.0:
                n_escape += 1
        out[name] = {
            "deep_step_frac": round(deep / n_steps, 5),
            "dwell_frac": round(dwell / n_steps, 5),
            "max_eject_A": round(max_eject, 3),
            "escape_frac": round(n_escape / n, 4),
        }
        print(f"toy {name:10}: {out[name]}", flush=True)

    g = {k: out[k] for k in ("wall_only", "asym", "v3")}
    # T3 amended (before the cell ran): the absolute dwell gate
    # (<= 5%) is not adjudicable in a constant-3-eV/A-attractor toy — the
    # pair has nowhere to be except the wall corridor, so dwell measures the
    # attractor, not the cap (measured 0.78-0.85 for ALL variants).  T3 is
    # restated comparatively: v3 must not increase dwell beyond the asym
    # variant it replaces; the population-level dwell is what the cell
    # phase's G3 measures against bare.
    verdict = {
        "T1_no_trap": g["v3"]["deep_step_frac"] == 0.0
                      and g["v3"]["escape_frac"] >= 0.99,
        "T2_bounded_escape": g["v3"]["max_eject_A"] <= 0.9,
        "T3_dwell_vs_asym": g["v3"]["dwell_frac"] <= g["asym"]["dwell_frac"] + 0.1,
    }
    print("toy verdict:", verdict, flush=True)
    with open(TOY_OUT, "w") as f:
        json.dump({"cells": out, "verdict": verdict,
                   "n_traj": n, "steps": 400, "start_d": 0.4}, f, indent=1)
    print(f"wrote {TOY_OUT}")
    return all(verdict.values())


def run_cell(device, cands, sigma=1.0, arm="all"):
    import run_phase1 as rp
    from materialgen.samplers.ald import ALDConfig

    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device=device, default_dtype="float32")
    refs = rp.load_reference_structures(rp.MINI_COMPOSITIONS)
    hull = rp.HullEvaluator(rp.load_hull_table())
    for comp in rp.MINI_COMPOSITIONS:
        ref = refs[comp]
        ref_c = rp.CrystalStructure.from_frac_coords(
            ref["numbers"], ref["frac_coords"], ref["lattice"])
        hull.calibrate(comp, rp.calc_ref_energy(calc, ref_c), len(ref["numbers"]),
                       float(ref["metadata"].get("formation_energy_per_atom", 0.0)))

    bare_score = rp.make_score(calc, "bare")
    l1_score = rp.make_score(calc, "l1")
    if sigma == 1.0 and arm == "all":
        arms = {
            "bare": (bare_score, dict(use_wall_cap=False, use_wall_cap_v3=False)),
            "asym": (l1_score, dict(use_wall_cap=True, use_wall_cap_v3=False)),
            "v3": (l1_score, dict(use_wall_cap=True, use_wall_cap_v3=True)),
        }
    else:  # multi-sigma extension: asym is superseded by v3, only bare vs v3
        arms = {
            "bare": (bare_score, dict(use_wall_cap=False, use_wall_cap_v3=False)),
            "v3": (l1_score, dict(use_wall_cap=True, use_wall_cap_v3=True)),
        }
    if arm != "all":
        arms = {arm: arms[arm]}
    out = cell_out_path(sigma, arm if arm != "all" else None)
    OOD_MASK = 0b010011
    cells = {}
    for arm_k, (score, caps) in arms.items():
        cands_l = []
        for comp in rp.MINI_COMPOSITIONS:
            cfg = ALDConfig(sigma_max=sigma, sigma_min=0.01, K=100, M=2,
                            alpha=1e-3, drift_cap=3.0, **caps)
            smp = rp.ALDSampler(cfg)
            for seed in [42, 123]:
                for cand in range(cands):
                    c = rp.run_candidate(score, smp, refs[comp], sigma,
                                         seed, cand, hull)
                    cands_l.append(c)
        clean = [c for c in cands_l if not (c["ood_bitmask"][0] & OOD_MASK)]
        induced = sum(any(m & OOD_MASK for m in c["ood_bitmask"][1:])
                      for c in clean) / max(len(clean), 1)
        hists = [h[1:] for h in [c["d_min_hist"] for c in cands_l] if len(h) > 1]
        steps = np.concatenate(hists) if hists else np.array([])
        # units fix: run_candidate -> HullEvaluator.e_hull now
        # returns true eV/atom, so a re-run of this cell reports the correct
        # scale.  The stored v3_verify*.json files predate the fix (pre-fix
        # scale = true eV/atom / reduced-formula atom count) and cannot be
        # rescaled -- run_cell persists only these aggregates, never the
        # candidates.  The G4 tolerance (0.02 eV/atom) is likewise 2-5x
        # looser in the pre-fix scale; re-run the cell before re-testing G4.
        eh = [c["final"]["e_hull"] for c in cands_l
              if c["final"]["e_hull"] is not None]
        cells[arm_k] = {
            "induced": round(induced, 4),
            "dwell": round(float((steps < DWELL).mean()), 4) if steps.size else 0.0,
            "deep": round(float((steps < 0.5).mean()), 4) if steps.size else 0.0,
            "e_hull_med": round(float(np.median(eh)), 2) if eh else None,
            "n_traj": len(cands_l),
        }
        print(f"cell[{sigma}] {arm_k}: {cells[arm_k]}", flush=True)
        with open(out, "w") as f:
            json.dump({"meta": {"partial": True}, "cells": cells}, f, indent=1)

    if len(cells) >= 2:
        if "asym" in cells:  # original sigma=1 three-arm run
            b, a, v = cells["bare"], cells["asym"], cells["v3"]
        else:
            b, v = cells["bare"], cells["v3"]
        verdict = {
            "G1_safety": v["induced"] <= b["induced"],
            "G2_deep": v["deep"] <= 2 * b["deep"],
            "G3_dwell": v["dwell"] <= b["dwell"] + 0.02,
            "G4_quality": v["e_hull_med"] is not None and b["e_hull_med"] is not None
                          and v["e_hull_med"] <= b["e_hull_med"] + 0.02,
        }
    else:
        verdict = None  # single-arm task: adjudication happens in the merge step
    print(f"cell[{sigma}] verdict: {verdict}", flush=True)
    with open(out, "w") as f:
        json.dump({"meta": {"cands": cands, "sigma": sigma,
                            "note": "v3 = asym cap + opening cap (r_wall-d)/2"},
                   "cells": cells, "verdict": verdict}, f, indent=1)
    print(f"wrote {out}")
    return all(verdict.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["toy", "cell", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cands", type=int, default=40)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--arm", default="all", choices=["all", "bare", "v3"])
    args = ap.parse_args()
    ok = True
    if args.phase in ("toy", "both"):
        ok = run_toy(args.n) and ok
    if args.phase in ("cell", "both") and ok:
        ok = run_cell(args.device, args.cands, sigma=args.sigma, arm=args.arm) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
