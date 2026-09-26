"""
Phase 0.5 — MD / Basin-Hopping baseline runner (CPU, eSEN).

Protocol (experiment_plan_v1.md 0.5, paper 6.0.6):
    - 10 MP-20 compositions, stratified by chemistry:
      oxides SrTiO3/TiO2/Al2O3/BaTiO3, sulfides ZnS/MoS2/CdS,
      fluoride LiF, nitride GaN, intermetallic FeNi3
    - MD: Langevin 2000 -> 100 K linear cooling, 5 rates (n_steps), 10 seeds
    - BH: T=1000 K, n_hops in {50, 100, 200, 500}, 10 seeds
    - Calculator: eSEN (OMAT24/esen_30m_mptrj.pt) on CPU

Each (composition, method, param, seed) is an independent task; results are
written as one JSON per task under results/baselines/{md,bh}/ so a crashed
worker never invalidates finished tasks. Run many workers in parallel with
OMP_NUM_THREADS pinned (see logs/run_baselines.sh).

Usage:
    python scripts/run_baselines.py --pilot          # 1 MD + 1 BH task, prints timing
    python scripts/run_baselines.py --method md --shard 0/128   # worker shard
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from materialgen.baselines.basin_hopping import run_basin_hopping
from materialgen.baselines.md_annealing import make_random_initial, run_simulated_annealing
from materialgen.core.crystal import CrystalStructure

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "baselines"
PROCESSED = REPO / "data" / "processed" / "mp_20" / "test.pkl"

COMPOSITIONS = [
    "SrTiO3", "TiO2", "Al2O3", "BaTiO3",   # oxides
    "ZnS", "MoS2", "CdS",                  # sulfides
    "LiF",                                 # fluoride
    "GaN",                                 # nitride
    "FeNi3",                               # intermetallic
]

MD_STEP_BUDGETS = [2000, 5000, 10000, 20000, 40000]   # 5 cooling rates
BH_HOPS = [50, 100, 200, 500]
SEEDS = list(range(10))


def load_reference_structures() -> dict:
    """Lowest-e_above_hull structure per target composition from mp_20 test."""
    import pickle

    with open(PROCESSED, "rb") as f:
        records = pickle.load(f)
    best: dict = {}
    for r in records:
        meta = r["metadata"]
        formula = meta.get("pretty_formula")
        if formula not in COMPOSITIONS:
            continue
        ehull = float(meta.get("e_above_hull", np.inf))
        if formula not in best or ehull < best[formula][0]:
            best[formula] = (ehull, r)
    refs = {f: CrystalStructure.from_frac_coords(r["numbers"], r["frac_coords"],
                                                 r["lattice"])
            for f, (_, r) in best.items()}
    missing = set(COMPOSITIONS) - set(refs)
    if missing:
        raise RuntimeError(f"No reference structure for {missing}")
    return refs


def build_tasks():
    tasks = []
    for comp in COMPOSITIONS:
        for n_steps in MD_STEP_BUDGETS:
            for seed in SEEDS:
                tasks.append({"method": "md", "composition": comp,
                              "n_steps": n_steps, "seed": seed})
        for n_hops in BH_HOPS:
            for seed in SEEDS:
                tasks.append({"method": "bh", "composition": comp,
                              "n_hops": n_hops, "seed": seed})
    return tasks


def task_path(task) -> Path:
    tag = (f"{task['composition']}_"
           + (f"steps{task['n_steps']}" if task["method"] == "md"
              else f"hops{task['n_hops']}")
           + f"_seed{task['seed']}.json")
    return OUT / task["method"] / tag


def run_task(task, calc, refs):
    comp = task["composition"]
    ref = refs[comp]
    initial = make_random_initial(ref.atomic_numbers, ref.lattice,
                                  dilation=1.5, seed=task["seed"])
    t0 = time.time()
    if task["method"] == "md":
        res = run_simulated_annealing(initial, calc, n_steps=task["n_steps"],
                                      seed=task["seed"])
        payload = {
            "best_energy": res.best_energy, "final_energy": res.final_energy,
            "nfe_md": res.nfe_md, "nfe_relax": res.nfe_relax,
            "energy_trace": res.energy_trace,
        }
    else:
        res = run_basin_hopping(initial, calc, n_hops=task["n_hops"],
                                seed=task["seed"])
        payload = {
            "best_energy": res.best_energy, "nfe": res.nfe,
            "n_hops": res.n_hops, "n_accepted": res.n_accepted,
            "energies": res.energies,
        }
    payload.update({
        **task,
        "wall_time_s": time.time() - t0,
        "n_atoms": initial.num_atoms,
        "best_structure": res.best_structure.to_dict(),
    })
    path = task_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="run one small MD and one small BH task, print timing")
    parser.add_argument("--method", choices=["md", "bh"], default=None)
    parser.add_argument("--shard", default=None,
                        help="k/N: run tasks where index %% N == k")
    parser.add_argument("--device", default="cuda",
                        help="cpu or cuda (eSEN on GPU is ~25x faster; "
                             "CUDA_VISIBLE_DEVICES selects the card)")
    args = parser.parse_args()

    from materialgen.nnp.esen import load_esen_calculator

    calc = load_esen_calculator(device=args.device)
    refs = load_reference_structures()
    tasks = build_tasks()

    if args.pilot:
        for task in [{"method": "md", "composition": "LiF", "n_steps": 200, "seed": 0},
                     {"method": "bh", "composition": "LiF", "n_hops": 3, "seed": 0}]:
            p = run_task(task, calc, refs)
            nfe = p.get("nfe_md", 0) + p.get("nfe_relax", 0) + p.get("nfe", 0)
            print(f"PILOT {task['method']}: {p['wall_time_s']:.1f}s wall, "
                  f"~{nfe} NFE, {p['wall_time_s']/max(nfe,1)*1000:.0f} ms/NFE, "
                  f"N={p['n_atoms']}")
        return

    if args.method:
        tasks = [t for t in tasks if t["method"] == args.method]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % n == k]

    for task in tasks:
        if task_path(task).exists():
            continue
        try:
            run_task(task, calc, refs)
        except Exception as exc:  # keep worker alive; log and skip
            print(f"FAILED {task}: {exc}", flush=True)


if __name__ == "__main__":
    main()
