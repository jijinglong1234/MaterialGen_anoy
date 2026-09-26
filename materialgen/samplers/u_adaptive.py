"""
UAdaptiveSampler — Type F: Uncertainty-Adaptive Schedule

Functionality:
    Implements an adaptive noise schedule where σ(t) is controlled in real-time
    by the NNP's uncertainty estimate u(x). This is the most NNP-aware scheduler
    — it slows down when the NNP is uncertain and accelerates when the NNP is
    confident.

    The dynamics are ALD-based but with adaptive σ:
    dx = α · σ²(x) · s(x) dt + √(2α) · σ(x) · dW_t

    where σ(x) = g(u(x)) — noise scale as a function of uncertainty.

    The key insight: standard schedulers apply the same σ(t) regardless of the
    NNP's confidence. When NNP uncertainty is high (OOD), standard schedulers
    blindly use inaccurate scores, leading to unphysical samples. U-Adaptive
    detects high uncertainty and either maintains or increases σ (staying in
    the diffusive, exploration regime) rather than converging with bad forces.

    Adaptive rules:

    1. Threshold-based:
       if u(x) > u_high:
           σ = min(σ + Δσ_up, σ_max)    # increase noise, escape OOD region
       elif u(x) < u_low:
           σ = max(σ - Δσ_down, σ_min)  # decrease noise, converge
       else:
           σ unchanged                   # maintain

    2. Continuous mapping:
       σ(x) = σ_min + (σ_max-σ_min) · sigmoid((u(x)-u_0)/τ)
       - Smooth, differentiable (useful for analysis)
       - τ controls the sharpness of the transition
       - u_0 is the "acceptable uncertainty" threshold

    3. PI controller:
       σ(t+1) = σ(t) + Kp·(u_target - u(x)) + Ki·Σ(u_target - u(x))
       - Proportional-integral control from control theory
       - Maintains uncertainty near a target level
       - More responsive than threshold-based

    4. Rejection-based:
       Propose: x' = x + α·σ²·s(x) + √(2α)·σ·z
       if u(x') < u_reject:
           accept: x = x'
       else:
           reject: x unchanged, potentially increase σ

    Uncertainty sources (see materialgen/nnp/uncertainty.py):
    - Ensemble NNP standard deviation
    - Descriptor-based OOD distance
    - Evidential uncertainty (eIP)
    - MC-Dropout variance

    Safety features:
    - σ_max hard cap: never exceeds the NNP's reliability limit
    - Panic mode: if u(x) exceeds u_critical, backtrack to last safe state
    - Uncertainty logging: full u(x) trajectory saved for analysis

    Key hyperparameters:
    - u_low, u_high: Uncertainty thresholds
    - σ_max, σ_min: Noise bounds (σ_max should be conservative, ≤ 1.5)
    - Δσ_up, Δσ_down: Rate of noise change
    - u_critical: Panic threshold
    - uncertainty_source: "ensemble" | "descriptor" | "evidential" | "mc_dropout"
    - adaptive_rule: "threshold" | "continuous" | "pi_controller" | "rejection"

    Expected behavior:
    - Most robust to NNP OOD failures
    - May be slower (more NFE) due to cautious exploration
    - Generates the most physically reliable structures
    - Ideal for systems where physical correctness is critical
    - The natural pairing with NNP-based score — directly addresses the core OOD bottleneck

Dependencies:
    numpy
    materialgen.core.sampler.Sampler
    materialgen.core.score_function.ScoreFunction
    materialgen.core.crystal.CrystalStructure
    materialgen.nnp.uncertainty.UncertaintyEstimator
"""
