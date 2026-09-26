"""
CHGNetNNP — wrapper for CHGNet universal potential with charge information.

Functionality:
    Wraps the CHGNet (Crystal Hamiltonian Graph Neural Network) model from
    https://github.com/CederGroupHub/chgnet for use as an NNP backend.

    This wrapper:
    - Loads the pre-trained CHGNet checkpoint (trained on MPtrj dataset, 1.5M structures)
    - Converts CrystalStructure → pymatgen.Structure → CHGNet input
    - Computes energy, forces, stress, and magnetic moments
    - Converts CHGNet output → NNPResult in standard units
    - Handles charge-informed predictions

    CHGNet-specific features:
    - Charge-aware: predicts atomic oxidation states alongside energies
    - Magnetic: predicts magnetic moments (useful for magnetic materials)
    - Trained on non-equilibrium MD trajectories (MPtrj), broader than MP relaxations
    - Supports both CPU and CUDA inference

    Usage:
        chgnet = CHGNetNNP(device="cuda")
        result = chgnet.predict(crystal)  # -> NNPResult with energy, forces, stress
        magmoms = result.metadata.get("magmoms")  # magnetic moments if relevant

    Note on OOD behavior:
        CHGNet was trained on MPtrj (MD trajectories at various temperatures),
        so it has seen some non-equilibrium configurations. However, the
        systematic softening problem (Deng et al., 2025) still applies to
        highly perturbed structures encountered in early diffusion steps.

Dependencies:
    chgnet (pip install chgnet)
    torch, numpy, pymatgen
    materialgen.nnp.base.NNPBackend, NNPResult
"""
