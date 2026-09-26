"""
UnderdampedSampler — Type C: Underdamped Langevin Dynamics

Functionality:
    Implements second-order (underdamped) Langevin dynamics with momentum:
    dx = v dt
    dv = -γ(t) v dt - β(t) ∇U(x) dt + √(2γ(t)/m) dW_t

    where v is the auxiliary momentum variable, γ(t) is the friction coefficient,
    and β(t) is the time-dependent inverse temperature.

    This is the closest to physical MD, but with tunable friction and temperature
    schedules. The momentum allows the system to "coast" through low-energy
    barriers that would trap overdamped dynamics.

    Splitting scheme (BAOAB — recommended for configurational sampling):
        The BAOAB splitting (Leimkuhler & Matthews, 2013) preserves the
        Boltzmann distribution better than Velocity Verlet:

        B (dt/2): v ← v + (dt/2) · (-β∇U(x))
        A (dt/2): x ← x + (dt/2) · v
        O (dt):   v ← exp(-γdt) · v + √(1-exp(-2γdt)) · N(0, 1/β)
        A (dt/2): x ← x + (dt/2) · v
        B (dt/2): v ← v + (dt/2) · (-β∇U(x))

    Alternative schemes:
    - ABOBA: Similar to BAOAB, different operator ordering
    - VV (Velocity Verlet): Standard MD integrator (symplectic)
    - GJF (Grønbech-Jensen-Farago): Better for large time steps
    - Euler-Maruyama: Simplest, lowest accuracy

    Friction scheduling options (γ(t)):

    1. Constant: γ(t) = γ₀
       - Simplest, no tuning of friction
       - Choose γ₀ ∈ {0.1, 0.5, 1.0, 5.0, 10.0}
       - γ₀ < 1 → underdamped (inertia dominates)
       - γ₀ > 5 → near-overdamped (recovers ALD behavior)

    2. Decaying: γ(t) = γ_max · (γ_min/γ_max)^(t/T)
       - High friction early (stable, overdamped-like)
       - Low friction late (momentum for barrier crossing)

    3. Two-phase: γ = γ_low for t < T/2, γ = γ_high for t ≥ T/2
       - Exploration with inertia → convergence with damping

    Temperature scheduling options (β(t)):
    - Constant: β(t) = β_target
    - Annealed: β(t) = β_min → β_max (cooling schedule)
    - Coupled to friction: β(t) = β_target · min(1, γ(t)/γ_ref)

    Key hyperparameters:
    - γ_min, γ_max: Friction range
    - β_min, β_max: Inverse temperature range
    - dt: Time step (typically 0.5-2.0 fs equivalent)
    - splitting: "BAOAB" | "ABOBA" | "VV" | "GJF"
    - mass: Effective mass for all atoms (or per-species masses)

    Expected behavior:
    - Better barrier crossing than ALD (due to inertia)
    - More physical trajectories (resembles MD)
    - Higher risk of NNP OOD: momentum pushes into unreliable regions
    - Carbon-24: should discover more polymorphs than ALD

Dependencies:
    numpy, torch
    materialgen.core.sampler.Sampler
    materialgen.core.score_function.ScoreFunction
    materialgen.core.crystal.CrystalStructure
"""
