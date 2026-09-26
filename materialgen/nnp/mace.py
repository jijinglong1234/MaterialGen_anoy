"""
MACENNP — wrapper for MACE-MP-0 / MACE-MP-1 universal potential.

Functionality:
    Wraps the MACE (Multipole Atomic Cluster Expansion) model from
    https://github.com/ACEsuit/mace for use as an NNP backend.

    This wrapper:
    - Loads the pre-trained MACE-MP-0 (or MACE-MP-1) checkpoint
    - Converts CrystalStructure → ase.Atoms → MACE input
    - Computes energy, forces, and optionally stress
    - Converts MACE output → NNPResult in standard units
    - Handles batch inference via torch DataLoader

    Supported MACE versions:
    - MACE-MP-0 (trained on Materials Project, 89 elements)
    - MACE-MP-1 (if available, broader coverage)
    - Custom fine-tuned MACE models (via checkpoint path)

    MACE-specific features:
    - Full E(3) equivariance guarantees rotation-invariant predictions
    - Compute scaling: O(N) with atomic cluster expansion radius
    - Default dtype: float64 (required for stable force computation)
    - Supports CUDA acceleration

    Usage:
        mace = MACENNP(model_path="MACE-MP-0.model", device="cuda", dtype="float64")
        result = mace.predict(crystal)  # -> NNPResult with energy, forces, stress

    Stress computation:
        MACE supports virial stress: σ_ij = (1/V) Σ_k m_k v_k,i v_k,j + (1/V) Σ_k r_k,i · f_k,j
        This is used to compute lattice score: s_L = -β · ∂U/∂L

Dependencies:
    mace (pip install mace-torch)
    torch, numpy, ase
    materialgen.nnp.base.NNPBackend, NNPResult
"""
