"""
M3GNetNNP — wrapper for M3GNet universal potential with 3-body interactions.

Functionality:
    Wraps the M3GNet (Materials 3-body Graph Network) model from
    https://github.com/materialsvirtuallab/m3gnet for use as an NNP backend.

    This wrapper:
    - Loads the pre-trained M3GNet checkpoint (trained on MPF.2021.2.8)
    - Converts CrystalStructure → pymatgen.Structure → M3GNet input
    - Computes energy, forces, and stress
    - Converts M3GNet output → NNPResult in standard units

    M3GNet-specific features:
    - 3-body interactions explicitly modeled (beyond pairwise)
    - Universal coverage: 89 elements from the periodic table
    - Trained on ~190k structures from Materials Project
    - Lightweight compared to MACE (fewer parameters, faster inference)

    Usage:
        m3gnet = M3GNetNNP(device="cuda")
        result = m3gnet.predict(crystal)  # -> NNPResult

    Note:
        M3GNet has been shown to have more severe OOD softening than MACE
        and CHGNet (Deng et al., 2025). It is primarily used as a secondary
        NNP for ensemble consensus validation, not as the primary score provider.

Dependencies:
    m3gnet (pip install m3gnet)
    torch, numpy, pymatgen
    materialgen.nnp.base.NNPBackend, NNPResult
"""
