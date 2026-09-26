"""
DiffCSPScore — interface to DiffCSP / DiffCSP++ learned score model.

Functionality:
    Wraps the DiffCSP (Diffusion for Crystal Structure Prediction) model
    for use as a learned score baseline. DiffCSP operates on fractional
    coordinates + lattice parameters and uses an E(3)-equivariant architecture.

    This wrapper:
    - Loads pre-trained DiffCSP / DiffCSP++ checkpoint
    - Computes learned score s_learned(x, t) at any diffusion time t
    - Generates structures via DiffCSP's native sampling (for baseline comparison)
    - Provides the same ScoreFunction interface as NNPScore for direct comparison

    DiffCSP's score space:
    - Score for fractional coordinates: s_frac(x, t) ∈ R^{N×3}
    - Score for lattice parameters: s_lattice(L, t) ∈ R^{3×3}
    - Both are time-dependent (conditioned on noise level t)

    Key comparison capability:
        For the same crystal structure x at noise level t, we can compute:
        - s_NNP(x) = -β ∇U(x) (time-independent, from NNP)
        - s_learned(x, t) (time-dependent, from DiffCSP)
        And compare them using score field analysis tools.

    Usage:
        diffcsp = DiffCSPScore(checkpoint_path="diffcsp_mp20.pt", device="cuda")
        result = diffcsp.compute(crystal, t=0.5)  # ScoreResult at t=0.5
        structures = diffcsp.generate(num_samples=100)  # native generation

    Note on coordinate system:
        DiffCSP uses fractional coordinates. The NNP score in Cartesian space
        must be transformed for comparison: s_frac = A^T · s_cart (where A = lattice matrix).
        This module handles the transformation automatically.

Dependencies:
    torch, numpy
    DiffCSP checkpoint and model definition (from official repository)
    materialgen.core.score_function.ScoreFunction, ScoreResult
    materialgen.core.crystal.CrystalStructure
"""
