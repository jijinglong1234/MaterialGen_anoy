"""
CDVAEWrapper — interface to CDVAE (Crystal Diffusion Variational Autoencoder).

Functionality:
    Wraps the CDVAE model for use as a data-driven generative baseline.
    Note: CDVAE's score function is defined in a learned latent space,
    NOT in coordinate space. Therefore, direct score field comparison
    with NNP score is NOT possible.

    However, CDVAE is included because:
    1. It is a key baseline for generation quality comparison
    2. The latent score can be analyzed: what properties of the learned
       latent space correlate with physical energy?
    3. It provides a contrast case: methods with inaccessible score fields

    This wrapper:
    - Loads pre-trained CDVAE (encoder + diffusion prior + decoder)
    - Generates structures via latent diffusion + decoding
    - Provides the decoder Jacobian for approximate score analysis:
      s_coord(x) ≈ (∂dec/∂z)^(-1) · s_latent(z)

    Note on score comparison limitation:
        CDVAE's score s_latent(z) operates on latent variables z, not atomic
        coordinates. The NNP score operates on coordinates. Direct comparison
        requires the decoder Jacobian, which is expensive (O(N²)) and potentially
        ill-conditioned. For this reason, CDVAE is used only for generation
        quality comparison, not for score field analysis.

    Usage:
        cdvae = CDVAEWrapper(checkpoint_path="cdvae_mp20.pt", device="cuda")
        structures = cdvae.generate(num_samples=100)  # latent sampling + decoding

Dependencies:
    torch, numpy
    CDVAE checkpoint and model definition (from official repository)
    materialgen.core.crystal.CrystalStructure
"""
