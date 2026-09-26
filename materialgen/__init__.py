"""
MaterialGen: Zero-Training Score-Based Generative Modeling for Crystal Structures

This package implements the core idea: using neural network interatomic potentials (NNPs)
as score functions for generative sampling of crystal structures, without training any
generative model.

The fundamental relationship is:
    s(x) = ∇_x log p(x) = -β ∇_x U_NNP(x)

where U_NNP is a pre-trained neural network potential (MACE, CHGNet, M3GNet),
β = 1/(kT) is the inverse temperature, and s(x) is the score function used in
score-based diffusion / Langevin sampling.

Key features:
- Zero training: No generative model training required
- Physics-guaranteed: Score is a conservative force field (curl-free)
- Multi-scheduler: 6 sampling paradigms (ALD, PF-ODE, Underdamped, Cyclical, FP-Optimal, U-Adaptive)
- Multi-NNP: Supports MACE, CHGNet, M3GNet backends
- Comprehensive evaluation: Validity, stability, match rate, coverage, curl violation analysis

Subpackages:
- core: Base abstractions (ScoreFunction, Scheduler, Sampler, CrystalStructure)
- nnp: Neural network potential wrappers
- baselines: Interface to learned score models (DiffCSP, MatterGen, CDVAE)
- samplers: Sampling algorithm implementations (6 scheduling strategies)
- data: Dataset loading and crystal structure I/O
- eval: Evaluation metrics and DFT relaxation pipeline
- analysis: Score field comparison, curl analysis, visualization
- experiments: Experiment runner, configuration, logging
- utils: Physical constants, coordinate transforms, periodic BCs
"""

__version__ = "0.1.0"
