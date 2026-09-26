"""
UncertaintyEstimator — uncertainty quantification for NNP predictions.

Functionality:
    Provides multiple UQ methods to estimate when the NNP is operating
    out-of-distribution (OOD). This is essential for:
    - U-Adaptive scheduler (controls noise level based on uncertainty)
    - Safe exploration (rejects steps into high-uncertainty regions)
    - Failure analysis (correlates uncertainty with generation quality)

    UQ Methods:

    1. EnsembleDisagreement:
       - Uses EnsembleNNP to compute per-atom force standard deviation
       - u(x) = max_i std_{m ∈ ensemble}(F_m^{(i)}) or mean std
       - Simple, effective, requires multiple NNP evaluations

    2. DescriptorDistance:
       - Computes SOAP/ACSF descriptor of local atomic environments
       - Compares against training set descriptor distribution
       - u(x) = Mahalanobis distance to training set centroid
       - Single NN pass, no ensemble needed

    3. EvidentialNNP (if available — eIP, Nature Comms 2026):
       - Single forward pass yields both prediction and uncertainty
       - Distinguishes aleatoric (data noise) vs epistemic (model uncertainty)
       - Most efficient option when supported

    4. MC-Dropout (if NNP supports dropout layers):
       - Multiple forward passes with dropout enabled
       - Variance across passes = epistemic uncertainty
       - Less reliable than ensemble (Kahle & Zipoli, 2021)

    5. LatentSpaceDensity:
       - Estimate p(x) in the NNP's learned latent space
       - Low density → OOD → high uncertainty
       - Requires access to NNP's internal representations

    Usage:
        estimator = UncertaintyEstimator(method="ensemble", ensemble=ensemble_nnp)
        u = estimator.estimate(crystal)  # -> float (scalar uncertainty)
        u_per_atom = estimator.estimate_per_atom(crystal)  # -> (N,) array

Dependencies:
    numpy, torch, scipy (for Mahalanobis), dscribe (for SOAP descriptors)
    materialgen.nnp.ensemble.EnsembleNNP
    materialgen.nnp.base.NNPResult
"""
