"""
Lambda mixing experiment: quantifying the marginal value of learned scores.

Functionality:
    This is a key experiment for the paper that precisely measures how much
    the learned score adds beyond the NNP score.

    LambdaMixingExperiment:
        Samples structures using a hybrid score:
        s_hybrid(x, t, λ) = (1-λ) · s_NNP(x) + λ · s_learned(x, t)
        for λ ∈ {0.0, 0.1, 0.2, ..., 0.9, 1.0}.

        This creates a continuous interpolation between:
        - λ = 0.0: pure NNP score (zero-training baseline)
        - λ = 1.0: pure learned score (full data-driven model)
        - 0 < λ < 1: hybrid (NNP with some learned correction)

    Experimental design:
        For each λ:
        - Generate N structures (e.g., N=500) using all 6 schedulers
        - Evaluate: validity, match rate, coverage, E_hull, diversity
        - Compute curl violation at each λ
        - Compare with pure NNP (λ=0) and pure learned (λ=1)

    Expected curve shapes (hypothesis-driven):
        Hypothesis A: "Diminishing returns"
            Quality ∝ 1 - exp(-c·λ)  → most value from small λ
            Implication: NNP score carries most information

        Hypothesis B: "Threshold effect"
            Quality ≈ constant for λ < λ*, then jumps at λ*
            Implication: learned score provides a specific correction
            that kicks in at a critical λ

        Hypothesis C: "Linear improvement"
            Quality ∝ λ
            Implication: NNP score and learned score are complementary

        Hypothesis D: "Optimal intermediate"
            Quality peaks at 0 < λ* < 1
            Implication: optimal combination of physics + data

    Analysis outputs:
    - λ-quality curves for each metric (line plot, x=λ)
    - Per-scheduler λ analysis (some schedulers benefit more from learned score)
    - Per-t (diffusion time) λ analysis: is learned score more valuable early or late?
    - Statistical significance test: does λ=0 significantly differ from λ=1?
    - Cost-benefit analysis: NFE vs quality for different λ

    This experiment directly answers: "Do we really need to train a generative
    model, or can we just use an NNP?"

Dependencies:
    numpy, scipy, torch
    materialgen.core.score_function.ScoreFunction, HybridScore
    materialgen.samplers.* (all 6 samplers)
    materialgen.eval.metrics.StructureEvaluator
"""
