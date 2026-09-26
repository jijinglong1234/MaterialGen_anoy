"""
Curl violation analysis for learned score functions.

Functionality:
    A key theoretical advantage of the NNP score is that it is guaranteed to be
    curl-free (conservative force field): ∇ × s_NNP = ∇ × (-β∇U) = 0.

    Learned scores from diffusion models do NOT have this guarantee. This module
    quantifies the curl violation and correlates it with generation quality.

    compute_curl_violation(score_fn, crystals, t) -> dict:
        Estimates the curl of the learned score field numerically:
        curl(s)_ij = ∂s_j/∂x_i - ∂s_i/∂x_j

        Since the score field is high-dimensional (3N for N atoms), we compute:
        1. Pairwise curl: For each pair of atoms (a,b), compute the 3×3 curl tensor
           by finite differences along 6 perturbation directions.
        2. Frobenius norm: |curl(s)|_F at each atomic position.
        3. Global curl metric: E_x[|curl(s)(x)|_F].

        For NNP score: this should be exactly 0 (up to numerical precision).
        For learned score: expect non-zero values, especially in OOD regions.

    curl_field_analysis(results: dict) -> dict:
        Analyzes the spatial distribution of curl violation:
        - Curl vs energy: does curl increase in high-energy regions?
        - Curl vs t (diffusion time): does curl change along the diffusion path?
        - Curl vs atom type: which elements have larger curl violations?
        - Curl vs coordination: does curl concentrate near defects/surfaces?
        - Correlation with generation failure: structures with high curl →
          more likely to be unphysical after relaxation?

    Practical implementation:
        Uses finite differences with adaptive step size.
        For atom i, coordinate α:
        ∂s_j/∂x_{i,α} ≈ (s_j(x + h e_{i,α}) - s_j(x - h e_{i,α})) / (2h)
        where h ≈ 1e-4 Å (tuned for NNP precision).

        Computational cost: 2 × 3N forward passes per structure.
        For large N, uses randomized subspace approximation (Hutchinson trace).

    Expected findings (hypothesis-driven):
        - Learned scores have non-zero curl, NNP scores have zero curl
        - Curl magnitude correlates with distance from training data
        - Curl is largest near saddle points (transition states)
        - High-curl structures are more likely to be unphysical
        - PF-ODE is less affected by curl than stochastic methods

Dependencies:
    numpy, scipy, torch
    materialgen.core.score_function.ScoreFunction, ScoreResult
    materialgen.core.crystal.CrystalStructure
"""
