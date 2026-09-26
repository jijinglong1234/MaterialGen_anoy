"""
FPOptimalSampler — Type E: Fokker-Planck Optimal Control

Functionality:
    Formulates the sampling problem as an optimal control problem:
    find a(t), b(t) that minimize the terminal KL divergence between the
    sampled distribution p_T and the target Boltzmann distribution p_B ∝ e^{-βU}.

    The general dynamics are:
    dx = a(t) · s(x) dt + b(t) · dW_t

    where s(x) = -β ∇U_NNP(x). The optimal a(t), b(t) minimize:
    J = D_KL(p_T || p_B) + λ · E[∫₀ᵀ L(a(t), b(t)) dt]

    This scheduler is the most theoretically grounded but also the most
    complex. It provides the optimal "temperature schedule" for the NNP score.

    Theoretical background:
    By the Fokker-Planck equation, the distribution p_t evolves as:
    ∂p_t/∂t = -∇ · (a(t) s(x) p_t) + (b²(t)/2) ∇² p_t

    The effective temperature is: e(t) = b²(t) / (2a(t))

    For exponential convergence of D_KL(p_t || p_B), the optimal effective
    temperature schedule often takes the form: e(t) ∝ e^{-ct} (Tzen & Raginsky, 2019).

    Implementation options:

    1. Exponential schedule (analytical approximation):
       e(t) = e_0 · (e_T/e_0)^(t/T), where e_0 = σ²_max, e_T = 1/β
       a(t) = ε · e(t)
       b(t) = √(2ε · e(t))
       - Simple, based on theoretical results for log-Sobolev distributions
       - Works well when the energy landscape is approximately convex

    2. Power-law schedule:
       e(t) = e_0 · (1 + c·t)^(-p)
       - Slower decay than exponential, better for rough landscapes
       - p > 1 for tempered convergence

    3. Numerically optimized (EM-based):
       - Run initial sampling with exponential schedule
       - Estimate p_t along the path using density estimation
       - Optimize e(t) to minimize estimated D_KL at each t
       - Re-run with optimized schedule
       - More expensive but adapts to the specific energy landscape

    4. Score-informed schedule:
       - Use the magnitude of NNP forces ||∇U|| to adjust e(t)
       - Higher ||∇U|| → decrease e(t) faster (landscape is steep)
       - Lower ||∇U|| → decrease e(t) slower (landscape is flat, need more time)
       - Balances exploration across different landscape regions

    Key hyperparameters:
    - schedule_type: "exponential" | "power_law" | "numerical" | "score_informed"
    - e_0, e_T: Effective temperature range
    - c, p: Parameters for power-law schedule
    - λ: Control cost weight (higher λ → slower schedule)

    Expected behavior:
    - Theoretically optimal convergence to p_B for convex-like landscapes
    - May underperform on highly non-convex landscapes (theoretical optimality
      assumes certain regularity conditions that crystal energy landscapes violate)
    - Good for benchmarking: provides the "best possible" monotonic schedule
    - Useful for understanding the fundamental limits of NNP score sampling

Dependencies:
    numpy, torch
    materialgen.core.sampler.Sampler
    materialgen.core.score_function.ScoreFunction
    materialgen.core.crystal.CrystalStructure
"""
