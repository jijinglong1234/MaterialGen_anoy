"""
Tests for sampler implementations.

Test cases:
    1. test_ald_reaches_equilibrium:
       For a simple 1D potential U(x) = x²/2 (harmonic oscillator),
       verify that ALD sampling recovers the Boltzmann distribution
       p(x) ∝ exp(-βx²/2) with correct variance 1/β.

    2. test_pf_ode_consistency:
       For the same harmonic potential, verify that PF-ODE produces
       samples from the same distribution as ALD (same marginals).

    3. test_underdamped_reaches_equilibrium:
       For harmonic potential, verify Underdamped Langevin recovers
       the correct configurational distribution (marginal over x).

    4. test_cyclical_exploration:
       For a double-well potential U(x) = (x²-1)², verify that
       Cyclical sampling visits both wells while monotonic ALD
       may get stuck in one well.

    5. test_fp_optimal_convergence:
       For harmonic potential, verify that FP-Optimal schedule
       achieves lower KL divergence than geometric ALD at the
       same NFE.

    6. test_u_adaptive_avoids_ood:
       Using a synthetic NNP with artificially inflated forces
       in certain regions, verify that U-Adaptive reduces the
       fraction of steps in those regions compared to ALD.

    7. test_sampler_determinism:
       Verify that samplers with fixed seed produce identical
       results across runs (reproducibility).

    8. test_pbc_enforcement:
       Verify that all samplers correctly enforce periodic
       boundary conditions (fractional coordinates ∈ [0, 1)).

    9. test_lattice_evolution:
       Verify that lattice parameters evolve during sampling
       and remain physically valid (positive volume, reasonable angles).

Dependencies:
    pytest, numpy, torch
    materialgen.samplers.*
    materialgen.core.sampler.Sampler
    materialgen.core.score_function.ScoreFunction
"""
