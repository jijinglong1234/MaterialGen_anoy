"""
EnsembleNNP — multi-NNP consensus wrapper for robust force and uncertainty estimation.

Functionality:
    Wraps multiple NNP backends (e.g., MACE + CHGNet + M3GNet) as an ensemble.
    Provides:
    1. Consensus energy/forces via weighted averaging
    2. Force uncertainty via ensemble standard deviation (per-atom)
    3. OOD detection via ensemble disagreement
    4. Configurable aggregation: mean, median, weighted mean, trimmed mean

    This is critical for:
    - UncertaintyAdaptiveScheduler: uses ensemble_std as u(x) to control σ(t)
    - Reliability checking: flags structures where NNPs disagree strongly
    - Evaluating NNP-dependence of results

    Ensembling strategies:
    - "mean": equally weighted average of all NNP outputs
    - "trimmed_mean": discard highest and lowest, average the rest
    - "median": per-atom median force direction (robust to one NNP failure)
    - "mace_only": use MACE primarily, others for uncertainty only

    Usage:
        ensemble = EnsembleNNP(
            backends=[MACENNP(...), CHGNetNNP(...), M3GNetNNP(...)],
            strategy="trimmed_mean",
        )
        result = ensemble.predict(crystal)
        # result.forces = consensus forces
        # result.uncertainty = per-atom std dev across ensemble

Dependencies:
    numpy, torch
    materialgen.nnp.base.NNPBackend, NNPResult
    materialgen.nnp.mace.MACENNP
    materialgen.nnp.chgnet.CHGNetNNP
    materialgen.nnp.m3gnet.M3GNetNNP
"""
