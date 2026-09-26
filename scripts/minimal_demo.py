#!/usr/bin/env python3
"""
Minimal Experiment: Score-Based Sampling with NNP Forces
=========================================================

This script demonstrates the core idea of MaterialGen:
    s(x) = ∇_x log p(x) = -β ∇_x U(x)

using a Lennard-Jones potential for Argon crystal (no NNP installation required).

What this experiment shows:
    1. An energy function U(x) implies a Boltzmann distribution p(x) ∝ e^{-βU(x)}
    2. The score of this distribution is s(x) = β · F(x), where F = -∇U
    3. We can sample from p(x) using Langevin dynamics with this score
    4. The Langevin dynamics naturally relaxes perturbed structures to low energy
    5. This is the SAME mathematics as diffusion-based generation,
       but with a physics-based score (no training needed)

Physical setup:
    - Argon FCC crystal (4 atoms in conventional cell)
    - Lennard-Jones potential: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    - ε = 0.0104 eV, σ = 3.40 Å (standard Ar parameters)
    - β = 38.68 eV⁻¹ (T = 300 K)

Algorithm:
    Annealed Langevin Dynamics (ALD):
        x_{k+1} = x_k + α · σ²_k · β · F(x_k) + √(2α) · σ_k · z_k

    where:
        σ_k = σ_max · (σ_min/σ_max)^(k/K)   (geometric schedule)
        z_k ~ N(0, I)                        (Gaussian noise)

Expected results:
    - Energy decreases from ~1-10 eV to near the LJ minimum (~-0.04 eV/atom)
    - Structure relaxes from random perturbation back to FCC
    - The Langevin path is a physically plausible trajectory (unlike pure data-driven diffusion)

References:
    - This implements Type A (ALD): annealed Langevin dynamics
    - MACE score pattern: https://github.com/ACEsuit/mace
    - Score-based models: Song et al., ICLR 2021
    - Score↔Force: Elijošius et al., Nature Comms 2025

Usage:
    python scripts/minimal_demo.py                    # LJ potential (no deps)
    python scripts/minimal_demo.py --backend lj       # Same as above, explicit
    python scripts/minimal_demo.py --noise 0.3        # Larger noise perturbation
    python scripts/minimal_demo.py --steps 500        # More integration steps
    python scripts/minimal_demo.py --plot             # Save energy trajectory plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.io import write as ase_write

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from materialgen.core.crystal import CrystalStructure
from materialgen.core.score_function import NNPScore
from materialgen.utils.constants import BETA_300K, BETA_500K


# ==============================================================================
# Experiment Configuration
# ==============================================================================

def create_argon_fcc() -> CrystalStructure:
    """Create a 2×2×2 Ar FCC supercell (32 atoms)."""
    # ase.build.bulk creates conventional cell (4 atoms for FCC)
    atoms = bulk("Ar", crystalstructure="fcc", a=5.26)  # Ar lattice constant
    atoms = atoms.repeat((2, 2, 2))  # 32 atoms
    return CrystalStructure(atoms)


def create_lj_score(beta: float = BETA_300K) -> NNPScore:
    """
    Create NNPScore with Lennard-Jones calculator.

    Standard Ar LJ parameters:
        σ = 3.40 Å, ε = 0.0104 eV (≈ 120 K · kB)
    """
    calc = LennardJones(sigma=3.40, epsilon=0.0104, rc=10.0)
    return NNPScore(calculator=calc, beta=beta, label="LennardJones(Ar)")


# ==============================================================================
# Annealed Langevin Dynamics (Type A)
# ==============================================================================

def annealed_langevin_dynamics(
    crystal: CrystalStructure,
    score_fn: NNPScore,
    sigma_max: float = 0.5,
    sigma_min: float = 0.01,
    num_levels: int = 100,
    steps_per_level: int = 3,
    step_size: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run annealed Langevin dynamics with NNP-defined score.

    Algorithm (Type A):
        For k = 0, 1, ..., K-1:                     # K noise levels
            σ_k = σ_max · (σ_min/σ_max)^(k/K)       # geometric schedule
            For i = 1, 2, ..., M:                    # M steps per level
                s = score_fn(crystal)                # NNP score: s = β · F
                z ~ N(0, I)                          # Gaussian noise
                x_{t+1} = x_t + α · σ² · s + √(2α) · σ · z
                Enforce PBC (wrap to [0,1))

    Args:
        crystal: Initial CrystalStructure (typically with added noise).
        score_fn: NNPScore that provides s(x) = β · F(x).
        sigma_max: Maximum noise level (controls exploration).
        sigma_min: Minimum noise level (controls final precision).
        num_levels: K, number of noise levels.
        steps_per_level: M, Langevin steps per noise level.
        step_size: α, step size for Langevin dynamics.
        seed: Random seed.
        verbose: Whether to print progress.

    Returns:
        dict with:
            - 'final_structure': CrystalStructure at the end
            - 'energy_trace': list of energies at each step
            - 'force_trace': list of max force magnitudes
            - 'structures_trace': list of intermediate CrystalStructures
            - 'nfe': number of score function evaluations
            - 'wall_time': computation time in seconds
    """
    rng = np.random.RandomState(seed)
    current = crystal.copy()
    n_atoms = current.num_atoms

    # Geometric noise schedule
    sigmas = sigma_max * (sigma_min / sigma_max) ** (np.arange(num_levels) / num_levels)

    # Trace collection
    energy_trace = []
    force_trace = []
    structures_trace = []

    total_steps = num_levels * steps_per_level
    t_start = time.perf_counter()

    step_counter = 0
    for k, sigma in enumerate(sigmas):
        sigma_sq = sigma * sigma

        for m in range(steps_per_level):
            # --- Compute score: s(x) = β · F(x) ---
            result = score_fn(current)
            cart_score = result.cart_score  # (N, 3), = β·forces

            # --- Langevin step (on fractional coordinates) ---
            # We use fractional space for PBC convenience:
            #   frac_{t+1} = frac_t + α · σ² · s_frac + √(2α) · σ · z
            noise = rng.normal(0, 1, (n_atoms, 3))
            delta_frac = (
                step_size * sigma_sq * result.frac_score
                + np.sqrt(2.0 * step_size) * sigma * noise
            )

            new_frac = current.frac_coords + delta_frac
            # Enforce PBC: wrap to [0, 1)
            new_frac = new_frac % 1.0

            current.frac_coords = new_frac

            # --- Tracing ---
            energy_trace.append(result.energy)
            force_trace.append(np.max(np.abs(result.forces)))
            if step_counter % max(1, total_steps // 20) == 0:
                structures_trace.append(current.copy())

            step_counter += 1

        if verbose and (k % max(1, num_levels // 10) == 0):
            print(
                f"  σ={sigma:.4f} | step={step_counter:4d}/{total_steps} | "
                f"E={result.energy:.4f} eV | F_max={np.max(np.abs(result.forces)):.4f} eV/Å"
            )

    wall_time = time.perf_counter() - t_start

    return {
        "final_structure": current,
        "energy_trace": energy_trace,
        "force_trace": force_trace,
        "structures_trace": structures_trace,
        "nfe": score_fn.n_calls,
        "wall_time": wall_time,
        "total_steps": total_steps,
    }


# ==============================================================================
# Analysis
# ==============================================================================

def analyze_results(results: dict, original: CrystalStructure, score_fn: NNPScore):
    """Print analysis of the sampling results."""
    final = results["final_structure"]
    energy_trace = np.array(results["energy_trace"])
    force_trace = np.array(results["force_trace"])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Energy statistics
    print(f"\nEnergy (per atom):")
    print(f"  Initial:    {energy_trace[0] / final.num_atoms:.6f} eV/atom")
    print(f"  Final:      {energy_trace[-1] / final.num_atoms:.6f} eV/atom")
    print(f"  Minimum:    {energy_trace.min() / final.num_atoms:.6f} eV/atom")
    print(f"  ΔE:         {(energy_trace[0] - energy_trace[-1]):.4f} eV total")

    # LJ reference: FCC Ar minimum ≈ -0.087 eV/atom at σ=3.40Å, ε=0.0104eV
    # Actually -8.6 * ε ≈ -0.089 eV/atom for FCC
    print(f"  LJ FCC ref: ≈ -0.089 eV/atom")

    # Force statistics
    print(f"\nForces (max per atom):")
    print(f"  Initial:    {force_trace[0]:.4f} eV/Å")
    print(f"  Final:      {force_trace[-1]:.4f} eV/Å")
    print(f"  Minimum:    {force_trace.min():.4f} eV/Å")

    # Structure statistics
    print(f"\nStructure:")
    print(f"  Min distance:   {final.get_minimum_distance():.4f} Å")
    print(f"  Volume:         {final.volume:.2f} Å³")
    print(f"  Volume/atom:    {final.volume / final.num_atoms:.2f} Å³/atom")

    # Validity check
    min_dist = final.get_minimum_distance()
    print(f"\nValidity:")
    print(f"  Atom overlap:   {'FAIL' if min_dist < 2.0 else 'PASS'} (min dist={min_dist:.2f} Å)")
    print(f"  Forces relaxed: {'YES' if force_trace[-1] < 0.1 else 'NOT FULLY'} (F_max={force_trace[-1]:.4f} eV/Å)")

    # Performance
    print(f"\nPerformance:")
    print(f"  NFE (score evals): {results['nfe']}")
    print(f"  Wall time:         {results['wall_time']:.2f} s")
    print(f"  ms/NFE:            {1000 * results['wall_time'] / results['nfe']:.2f} ms")

    # Temperature analysis
    print(f"\nScore function:")
    print(f"  {score_fn}")

    # Score-force relationship verification
    # For the initial structure, verify |s| / |F| = β
    result = score_fn(original)
    ratio = np.linalg.norm(result.cart_score) / max(np.linalg.norm(result.forces), 1e-10)
    print(f"\nScore-Force Relationship (verification):")
    print(f"  s(x) = β · F(x)")
    print(f"  |s| / |F| = {ratio:.4f}  (should equal β = {score_fn.beta:.4f})")
    print(f"  Error: {abs(ratio - score_fn.beta):.2e}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MaterialGen Minimal Experiment: Score-Based Sampling with NNP Forces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/minimal_demo.py                    # Basic run (LJ Ar, default params)
  python scripts/minimal_demo.py --noise 0.3        # Larger initial perturbation
  python scripts/minimal_demo.py --steps 500        # More integration steps
  python scripts/minimal_demo.py --plot             # Save energy trajectory plot
  python scripts/minimal_demo.py --temp 500         # Run at T=500K
  python scripts/minimal_demo.py --repeat 5         # Run 5 times, report statistics
        """,
    )
    parser.add_argument("--backend", default="lj", choices=["lj"],
                        help="Calculator backend (default: lj = Lennard-Jones Ar)")
    parser.add_argument("--noise", type=float, default=0.15,
                        help="Std of Gaussian noise added to fractional coords (default: 0.15)")
    parser.add_argument("--sigma-max", type=float, default=0.5,
                        help="Max noise level for ALD schedule (default: 0.5)")
    parser.add_argument("--sigma-min", type=float, default=0.01,
                        help="Min noise level for ALD schedule (default: 0.01)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of noise levels K (default: 100)")
    parser.add_argument("--steps-per-level", type=int, default=3,
                        help="Langevin steps per noise level M (default: 3)")
    parser.add_argument("--step-size", type=float, default=0.1,
                        help="Langevin step size α (default: 0.1)")
    parser.add_argument("--temp", type=float, default=300,
                        help="Temperature in Kelvin (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--plot", action="store_true",
                        help="Save energy trajectory plot to results/")
    parser.add_argument("--save-structures", action="store_true",
                        help="Save generated structures to results/")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # ---- Setup ----
    beta = 1.0 / (8.617333262145e-5 * args.temp)
    score_fn = create_lj_score(beta=beta)
    original = create_argon_fcc()
    original_energy = score_fn(original).energy

    if not args.quiet:
        print("=" * 60)
        print("MaterialGen Minimal Experiment")
        print("=" * 60)
        print(f"\nSystem:     Ar FCC, {original.num_atoms} atoms")
        print(f"Backend:    Lennard-Jones (σ=3.40 Å, ε=0.0104 eV)")
        print(f"Temperature: {args.temp} K (β = {beta:.2f} eV⁻¹)")
        print(f"Noise std:  {args.noise} (fractional coordinates)")
        print(f"Reference energy: {original_energy:.4f} eV ({original_energy/original.num_atoms:.6f} eV/atom)")
        print(f"\nALD schedule:")
        print(f"  σ_max = {args.sigma_max}, σ_min = {args.sigma_min}")
        print(f"  K = {args.steps}, M = {args.steps_per_level}")
        print(f"  α = {args.step_size}")
        print(f"  Total Langevin steps: {args.steps * args.steps_per_level}")

    # ---- Run ----
    all_energy_traces = []
    all_final_energies = []

    for run in range(args.repeat):
        run_seed = args.seed + run
        if not args.quiet and args.repeat > 1:
            print(f"\n--- Run {run+1}/{args.repeat} (seed={run_seed}) ---")

        # Perturb the structure
        perturbed = original.add_noise_to_frac(args.noise, seed=run_seed * 100)
        score_fn.reset_stats()

        # Run ALD
        results = annealed_langevin_dynamics(
            crystal=perturbed,
            score_fn=score_fn,
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            num_levels=args.steps,
            steps_per_level=args.steps_per_level,
            step_size=args.step_size,
            seed=run_seed,
            verbose=not args.quiet,
        )

        all_energy_traces.append(np.array(results["energy_trace"]))
        all_final_energies.append(results["energy_trace"][-1])

        if args.repeat == 1 or run == 0:
            analyze_results(results, original, score_fn)

    # ---- Multi-run statistics ----
    if args.repeat > 1:
        final_energies = np.array(all_final_energies)
        print(f"\n--- Multi-run Statistics (N={args.repeat}) ---")
        print(f"Final energy: {final_energies.mean():.4f} ± {final_energies.std():.4f} eV")
        print(f"            : {final_energies.mean()/original.num_atoms:.6f} ± {final_energies.std()/original.num_atoms:.6f} eV/atom")
        print(f"Min: {final_energies.min():.4f}, Max: {final_energies.max():.4f}")

    # ---- Plot ----
    if args.plot:
        _save_plot(all_energy_traces, original, args)

    # ---- Save structures ----
    if args.save_structures:
        _save_structures(results, args)


def _save_plot(energy_traces: list, original: CrystalStructure, args):
    """Save energy trajectory plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARNING] matplotlib not installed, skipping plot. Install with: pip install matplotlib")
        return

    output_dir = Path("results/minimal_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_atoms = original.num_atoms

    for i, trace in enumerate(energy_traces):
        ax1.plot(trace / n_atoms, alpha=0.7, label=f"Run {i+1}" if len(energy_traces) > 1 else None)
    ax1.axhline(y=-0.089, color="r", linestyle="--", label="LJ FCC minimum")
    ax1.set_xlabel("Langevin Step")
    ax1.set_ylabel("Energy (eV/atom)")
    ax1.set_title("Energy During Annealed Langevin Dynamics")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Second subplot: zoom on final steps
    for i, trace in enumerate(energy_traces):
        ax2.plot(trace[-50:] / n_atoms, alpha=0.7)
    ax2.axhline(y=-0.089, color="r", linestyle="--", label="LJ FCC minimum")
    ax2.set_xlabel("Langevin Step (last 50)")
    ax2.set_ylabel("Energy (eV/atom)")
    ax2.set_title("Zoom: Final Convergence")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "energy_trajectory.png", dpi=150)
    plt.savefig(output_dir / "energy_trajectory.pdf")
    print(f"\nPlot saved to: {output_dir}/energy_trajectory.png")


def _save_structures(results: dict, args):
    """Save initial and final structures."""
    output_dir = Path("results/minimal_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    final = results["final_structure"]
    ase_write(str(output_dir / "final_structure.cif"), final.ase_atoms, format="cif")
    ase_write(str(output_dir / "final_structure.extxyz"), final.ase_atoms, format="extxyz")
    print(f"\nStructures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
