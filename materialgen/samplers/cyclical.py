"""
CyclicalSampler — Type D: Cyclical / Reheating Schedule

Functionality:
    Implements cyclical noise scheduling where σ(t) oscillates between σ_min
    and σ_max over multiple cycles, analogous to simulated tempering but in
    the diffusion/score framework.

    Each cycle consists of:
    1. Heating phase: σ increases (more noise → escape current basin)
    2. Cooling phase: σ decreases (less noise → converge to new basin)

    This is designed to overcome the fundamental limitation of monotonic
    annealing: once you descend into a local minimum, you cannot escape.

    Cycle shape options:

    1. Sinusoidal: σ(t) = σ_min + (σ_max-σ_min) · |sin(πt/T_cycle)|
       - Smooth, continuous transition
       - Most natural choice, inspired by physical tempering

    2. Sawtooth (fast-heat, slow-cool):
       - Heating: σ jumps quickly to σ_max (encourage escape)
       - Cooling: σ decreases gradually (thorough exploration of new basin)
       - More aggressive exploration than sinusoidal

    3. Step (discrete reheating):
       - Run at σ_min for N_converge steps
       - Jump to σ_max for N_heat steps (randomize)
       - Jump back to σ_min for N_converge steps
       - Simplest to implement and analyze

    Algorithm (sinusoidal):
        For cycle = 1 to N_cycles:
            For t = 0 to T_cycle:
                σ = σ_min + (σ_max-σ_min) · |sin(πt/T_cycle)|
                x = langevin_step(x, score_fn, σ)
                # Optionally: record energy, if energy plateaus → trigger early reheat

    Key hyperparameters:
    - σ_max: Maximum reheating noise level
      Critical: must be within NNP's reliability range (strongly recommend ≤ 1.0-1.5)
    - σ_min: Minimum noise level (convergence target)
    - T_cycle: Steps per cycle (50-200 recommended)
    - N_cycles: Number of cycles (3-10 recommended)
    - cycle_shape: "sinusoidal" | "sawtooth" | "step"
    - Early reheating: if energy hasn't decreased in N_stall steps, reheat early

    NNP suitability: ⭐⭐ (worst among 6 schedulers)
    - Repeatedly enters high-noise regions where NNP score is unreliable
    - At σ_max, NNP forces may point toward unphysical configurations
    - Recommend conservative σ_max (≤ 1.0) when using this scheduler
    - The cyclical nature means errors at high noise accumulate across cycles

    Expected behavior:
    - Best exploration IF NNP is reliable at σ_max
    - Risk: may generate unphysical structures due to accumulated NNP errors
    - Potentially discovers more diverse polymorphs than monotonic schedulers
    - Should be compared with ALD at same total NFE to assess benefit of reheating

    Analysis output:
    - Per-cycle energy traces
    - Number of distinct basins visited per cycle
    - Correlation between σ_max and OOD rate

Dependencies:
    numpy
    materialgen.core.sampler.Sampler
    materialgen.core.score_function.ScoreFunction
    materialgen.core.crystal.CrystalStructure
"""
