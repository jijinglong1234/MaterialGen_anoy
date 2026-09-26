"""
MatterGenScore — interface to MatterGen learned score model.

Functionality:
    Wraps the MatterGen model (Microsoft Research, 2024) for use as a
    learned score baseline. MatterGen uses a joint diffusion over atomic
    types, coordinates, and lattice parameters.

    This wrapper:
    - Loads pre-trained MatterGen checkpoint
    - Computes learned score s_learned(x, t) for comparison with NNP score
    - Generates structures via MatterGen's native sampling pipeline
    - Provides the same ScoreFunction interface as NNPScore

    MatterGen's score space:
    - Score for atomic coordinates (fractional or Cartesian, configurable)
    - Score for lattice parameters
    - Score for atomic types (discrete diffusion component)
    - Note: The atomic type score has no NNP equivalent (NNP doesn't handle
      alchemical transformations), so comparison focuses on coordinate and
      lattice scores.

    Key comparison capability:
        s_NNP(x) vs s_MatterGen(x, t) — comparison restricted to coordinate
        and lattice components. Atomic type component is excluded from
        comparison metrics.

    Usage:
        mattergen = MatterGenScore(checkpoint_path="mattergen.pt", device="cuda")
        result = mattergen.compute(crystal, t=0.3)
        # result.frac_score.shape = (N, 3)
        # result.lattice_score.shape = (3, 3)

Dependencies:
    torch, numpy
    MatterGen checkpoint and model definition
    materialgen.core.score_function.ScoreFunction, ScoreResult
"""
