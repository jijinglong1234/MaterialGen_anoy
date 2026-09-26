"""
Periodic boundary condition utilities.

Functionality:
    minimum_image(dr: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        Applies the minimum image convention to displacement vectors.
        For each vector dr_i, finds the periodic image closest to 0.
        Uses the algorithm: dr - L · round(L^{-1} · dr)

    wrap_coordinates(frac_coords: np.ndarray) -> np.ndarray:
        Wraps fractional coordinates to [0, 1) range.
        Handles both single atom and batch inputs.

    get_neighbor_list(frac_coords, lattice, cutoff) -> tuple:
        Computes neighbor list: (src_idx, dst_idx, offsets) within cutoff.
        Uses a cell-list algorithm for O(N) scaling.
        Returns neighbor pairs with periodic image offsets.

    compute_displacement_matrix(frac1, frac2, lattice) -> np.ndarray:
        Computes all-pairs displacement vectors with minimum image convention.
        Returns: (N1, N2, 3) array of displacement vectors.

    compute_distance_matrix(frac_coords, lattice) -> np.ndarray:
        Computes all-pairs distance matrix with minimum image convention.
        Returns: (N, N) symmetric matrix.

    apply_pbc_displacement(frac_coords, displacement, lattice) -> np.ndarray:
        Applies a displacement to fractional coordinates and wraps to [0,1).

    make_supercell_frac(frac_coords, scaling) -> np.ndarray:
        Generates fractional coordinates for a supercell.
        scaling = (na, nb, nc) — number of replicas in each direction.

Dependencies:
    numpy
    materialgen.core.crystal.CrystalStructure
"""
