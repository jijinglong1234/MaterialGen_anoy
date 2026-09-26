"""
Tests for score_function module.

Test cases:
    1. test_nnp_score_curl_free:
       Verify that ∇ × s_NNP = 0 (within numerical precision).
       For a set of random crystals, compute the curl violation and assert it is < 1e-8.

    2. test_nnp_score_magnitude:
       Verify that the NNP score magnitude is reasonable:
       - Not NaN or inf
       - Force magnitudes are in physical range (< 100 eV/Å)
       - Score vector is not zero for non-equilibrium structures

    3. test_score_coordinate_transform:
       Verify that s_frac = L^T · s_cart correctly transforms the score:
       - Compute Cartesian score from NNP
       - Transform to fractional coordinates
       - Verify that the dot product s_frac · Δfrac gives the same energy
         change as s_cart · Δcart for small displacements

    4. test_hybrid_score_consistency:
       Verify that HybridScore(λ=0) == NNPScore and HybridScore(λ=1) == LearnedScore
       within numerical tolerance.

    5. test_score_batch_consistency:
       Verify that batch computation gives the same results as individual
       computation for a list of structures.

    6. test_lattice_score_from_stress:
       Verify that lattice score s_L = -β · V · σ · L^{-T} is consistent:
       - For a small lattice perturbation δL, the energy change from NNP
         should match s_L : δL (Frobenius inner product).

    7. test_nnp_score_temperature_scaling:
       Verify that s(x; β=100) = (100/38.68) · s(x; β=38.68) at 300K.
       The score should scale linearly with β.

Dependencies:
    pytest, numpy, torch
    materialgen.core.score_function.*
    materialgen.core.crystal.CrystalStructure
    materialgen.nnp.*
"""
