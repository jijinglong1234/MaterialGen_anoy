"""
Score field comparison: NNP score vs Learned score.

Functionality:
    This is the core analysis module for the paper's central comparison.
    It computes quantitative metrics comparing s_NNP(x) = -β∇U(x) and
    s_learned(x, t) from a trained diffusion model.

    ScoreComparator class:
        Orchestrates the full score comparison pipeline.

    Sampling strategies for comparison points:
    1. Natural diffusion path: x_t = α_t x_0 + σ_t ε, t ∈ [0, T]
       - Samples along the actual diffusion trajectory
       - Most relevant for understanding diffusion behavior
    2. Energy-stratified: sample from different energy ranges
       - Low energy: < 0.1 eV/atom above hull
       - Medium energy: 0.1-0.5 eV/atom
       - High energy: > 0.5 eV/atom
    3. Saddle/transition state: near phase transition pathways
       - Most physically relevant for NNP reliability analysis

    Point-wise metrics:
    - angular_deviation(x, t): arccos(s_NNP·s_learned / (|s_NNP||s_learned|))
      in degrees [0, 180]. 0° = perfect agreement.
    - relative_magnitude(x, t): |s_learned| / |s_NNP|
      >1: learned score is stronger; <1: learned score is weaker.
    - per_atom_coherence(x, t): 1/N Σ_i |s_NNP_i·s_learned_i| / (|s_NNP_i||s_learned_i|)
      Per-atom direction agreement, robust to global scaling differences.
    - force_residual(x): |F_NNP - F_learned| / |F_NNP|
      Raw force difference normalized by NNP force magnitude.

    Distribution-level metrics:
    - score_wasserstein: W₂ distance between normalized score magnitude distributions
    - field_divergence: E_x[|s_NNP(x) - s_learned(x,t)|²]
    - spearman_rank: rank correlation of score magnitudes across atoms

    Analysis outputs:
    - d_angle vs t plot (key figure: shows when learning deviates from physics)
    - d_angle vs energy plot (shows where learning deviates from physics)
    - Score magnitude distribution comparison
    - Atom-type-resolved score comparison (which elements are harder to learn?)
    - Per-coordination-environment analysis

    Usage:
        comparator = ScoreComparator(nnp_score, learned_score)
        results = comparator.compare(
            dataset=mp20_dataset,
            t_values=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            num_samples_per_t=100,
        )
        comparator.plot_all(results, save_dir="figures/score_compare/")

Dependencies:
    numpy, scipy, torch
    materialgen.core.score_function.ScoreFunction, ScoreResult
    materialgen.core.crystal.CrystalStructure
    materialgen.data.datasets.CrystalDataset
"""
