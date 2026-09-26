"""Multi-sigma v3-cap verification driver.

Pre-registered extension of verify_v3_cap.py's cell phase to sigma in
{2, 3, 5} x {bare, v3} — the gate before the full Phase-1 fleet: v3 was
validated only at sigma=1 (G1-G4 all pass), and the plan doc defers its
sigma>=2 behaviour to the fleet; this targeted cell closes that gap at
~15 GPU-h instead of risking 6000 fleet tasks.

Each task is one (sigma, arm) pair = 5 comps x 2 seeds x 20 cands = 200
trajectories, launched as an independent process pinned to one GPU.

GPU layout: cuda:0 is idle; cuda:1-4 are shared with a concurrent job that
uses ~67 GB each, so MACE-medium inference (~2-4 GB) fits in the remainder.

Usage:
    python scripts/run_v3_multisigma.py                 # launch all 6 tasks
    python scripts/run_v3_multisigma.py --watch         # poll until done
    python scripts/run_v3_multisigma.py --merge         # adjudicate G1-G4
"""
import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(_ROOT, "results", "phase1", "analysis")
LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

SIGMAS = [2.0, 3.0, 5.0]
ARMS = ["bare", "v3"]
CANDS = 20
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"]
# Interpreter used for the subprocesses: the one running this script.
PYTHON = sys.executable
MERGE_OUT = os.path.join(ANALYSIS, "v3_verify_multisigma.json")


def tasks():
    return [(s, a) for s in SIGMAS for a in ARMS]


def task_log(sigma, arm):
    return os.path.join(LOG_DIR, f"v3verify_s{sigma:g}_{arm}.log")


def merge():
    """Adjudicate G1-G4 per sigma from the six per-task JSONs."""
    out = {"meta": {"sigma": SIGMAS, "arms": ARMS, "cands": CANDS,
                    "n_traj_per_task": 5 * 2 * CANDS,
                    "note": "gate before full Phase-1 fleet: v3 must pass "
                            "G1-G4 at every sigma in {2,3,5}"},
           "cells": {}, "verdict": {}}
    all_pass = True
    for s in SIGMAS:
        cells = {}
        for a in ARMS:
            p = os.path.join(ANALYSIS, f"v3_verify_s{s:g}_{a}.json")
            if not os.path.exists(p):
                print(f"MISSING {p}")
                all_pass = False
                continue
            d = json.load(open(p))
            cells[a] = d["cells"][a]
        out["cells"][f"s{s:g}"] = cells
        if set(cells) == {"bare", "v3"}:
            b, v = cells["bare"], cells["v3"]
            g = {
                "G1_safety": v["induced"] <= b["induced"],
                "G2_deep": v["deep"] <= 2 * b["deep"],
                "G3_dwell": v["dwell"] <= b["dwell"] + 0.02,
                "G4_quality": v["e_hull_med"] is not None
                              and b["e_hull_med"] is not None
                              and v["e_hull_med"] <= b["e_hull_med"] + 0.02,
            }
            out["verdict"][f"s{s:g}"] = g
            all_pass = all_pass and all(g.values())
            print(f"sigma={s}: {g}")
        else:
            out["verdict"][f"s{s:g}"] = None
            print(f"sigma={s}: INCOMPLETE ({sorted(cells)})")
    out["all_pass"] = all_pass
    with open(MERGE_OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'ALL GATES PASS' if all_pass else 'GATES FAIL'} -> {MERGE_OUT}")
    return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="poll running tasks until all finish")
    ap.add_argument("--merge", action="store_true",
                    help="adjudicate from existing per-task JSONs")
    args = ap.parse_args()

    if args.merge:
        sys.exit(0 if merge() else 1)

    # dispatch: gpu0 takes 2 tasks (sequential), gpu1-4 one each
    plan = {}
    for i, t in enumerate(tasks()):
        gpu = GPUS[i % 5]
        plan.setdefault(gpu, []).append(t)
    print("dispatch plan:", {g: [f"s{s:g}-{a}" for s, a in ts] for g, ts in plan.items()})

    procs = []
    for gpu, ts in plan.items():
        for s, a in ts:
            log = open(task_log(s, a), "w")
            cmd = [PYTHON, os.path.join(_ROOT, "scripts", "verify_v3_cap.py"),
                   "--phase", "cell", "--device", gpu, "--cands", str(CANDS),
                   "--sigma", str(s), "--arm", a]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=_ROOT)
            procs.append((s, a, p))
            print(f"launched s{s:g}-{a} on {gpu} pid={p.pid} -> {task_log(s, a)}",
                  flush=True)
            time.sleep(5)  # stagger model loads on the shared GPUs

    if args.watch:
        try:
            while any(p.poll() is None for _, _, p in procs):
                time.sleep(120)
                done = sum(1 for _, _, p in procs if p.poll() is not None)
                print(f"[{time.strftime('%H:%M')}] {done}/6 tasks finished",
                      flush=True)
            rc = [p.returncode for _, _, p in procs]
            print("return codes:", rc)
            sys.exit(0 if all(r == 0 for r in rc) else 1)
        except KeyboardInterrupt:
            print("interrupted; tasks keep running in background")
            sys.exit(130)


if __name__ == "__main__":
    main()
