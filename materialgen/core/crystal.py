"""
CrystalStructure — primary data class for representing periodic crystal structures.

Design:
    Internally wraps an ase.Atoms object for compatibility with NNP backends
    (MACE, CHGNet, M3GNet all accept ase.Atoms or can convert from it).
    Provides convenience accessors for fractional coordinates, lattice, etc.

Reference implementations:
    - MACE: https://github.com/ACEsuit/mace (uses ase.Atoms)
    - CHGNet: https://github.com/CederGroupHub/chgnet (uses pymatgen.Structure)
    - DiffCSP: https://github.com/jiaor17/DiffCSP (custom representation)
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import get_distances as ase_get_distances


def symmetrize_lattice_update(L: np.ndarray, dL: np.ndarray) -> np.ndarray:
    """
    Project a raw lattice-matrix displacement onto the strain subspace.

    The stress tensor is symmetric, so the lattice score s_L = -beta*V*sigma*L^{-T}
    carries only 6 physical (strain) degrees of freedom (paper section 6.0.5);
    a raw 9-component update additionally drives a rotation gauge, which has
    no physical content under the fractional-coordinate dynamics, and can
    drift the cell out of the positive-definite domain.  Projection via the
    velocity-gradient symmetrization removes the rotation component exactly:

        A = dL · L^{-1},   dL_sym = (A + A^T)/2 · L

    Args:
        L: current lattice matrix (rows are lattice vectors).
        dL: raw displacement of L (e.g. coeff * s_L for ALD, dL/dsigma for PF-ODE).
    """
    A = dL @ np.linalg.inv(L)
    return 0.5 * (A + A.T) @ L


class CrystalStructure:
    """
    Represents a periodic crystal structure.

    Wraps ase.Atoms internally for seamless NNP backend compatibility.
    Provides fast access to Cartesian/fractional coordinates and lattice.
    """

    def __init__(
        self,
        atoms: Atoms,
        properties: Optional[dict] = None,
    ):
        """
        Initialize from an ase.Atoms object.

        Args:
            atoms: ASE Atoms with periodic boundary conditions and cell.
            properties: Optional dict of extra properties (energy, forces, stress, etc.).
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms, got {type(atoms)}")
        if not np.all(atoms.pbc):
            raise ValueError("CrystalStructure requires periodic boundary conditions (pbc=True)")

        self._atoms = atoms.copy()
        self.properties = properties or {}

    # ---- Factory constructors ----

    @classmethod
    def from_cartesian(
        cls,
        atomic_numbers: np.ndarray,
        cart_coords: np.ndarray,
        lattice: np.ndarray,
        properties: Optional[dict] = None,
    ) -> "CrystalStructure":
        """
        Create from Cartesian coordinates and lattice.

        Args:
            atomic_numbers: (N,) int array
            cart_coords: (N, 3) float array in Å
            lattice: (3, 3) float array — rows are lattice vectors a1, a2, a3
            properties: Optional dict of extra properties.
        """
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=cart_coords,
            cell=lattice,
            pbc=True,
        )
        return cls(atoms, properties)

    @classmethod
    def from_frac_coords(
        cls,
        atomic_numbers: np.ndarray,
        frac_coords: np.ndarray,
        lattice: np.ndarray,
        properties: Optional[dict] = None,
    ) -> "CrystalStructure":
        """
        Create from fractional coordinates and lattice.

        Args:
            atomic_numbers: (N,) int array
            frac_coords: (N, 3) float array ∈ [0, 1)
            lattice: (3, 3) float array — rows are lattice vectors
            properties: Optional dict of extra properties.
        """
        scaled = np.clip(frac_coords, 0.0, 1.0 - 1e-12)
        cart = scaled @ lattice
        return cls.from_cartesian(atomic_numbers, cart, lattice, properties)

    # ---- Properties ----

    @property
    def num_atoms(self) -> int:
        return len(self._atoms)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atoms.get_atomic_numbers().copy()

    @property
    def atom_types(self) -> list:
        return list(self._atoms.get_chemical_symbols())

    @property
    def cart_coords(self) -> np.ndarray:
        """Cartesian coordinates (N, 3) in Å."""
        return self._atoms.get_positions().copy()

    @cart_coords.setter
    def cart_coords(self, positions: np.ndarray):
        self._atoms.set_positions(positions)

    @property
    def frac_coords(self) -> np.ndarray:
        """Fractional coordinates (N, 3) ∈ [0, 1)."""
        return self._atoms.get_scaled_positions(wrap=True)

    @frac_coords.setter
    def frac_coords(self, scaled: np.ndarray):
        scaled = np.clip(scaled, 0.0, 1.0 - 1e-12)
        self._atoms.set_scaled_positions(scaled)

    @property
    def lattice(self) -> np.ndarray:
        """Lattice matrix (3, 3) — rows are lattice vectors a1, a2, a3."""
        return self._atoms.get_cell().array.copy()

    @lattice.setter
    def lattice(self, cell: np.ndarray):
        self._atoms.set_cell(cell)

    @property
    def volume(self) -> float:
        return self._atoms.get_volume()

    @property
    def chemical_formula(self) -> str:
        return self._atoms.get_chemical_formula()

    @property
    def ase_atoms(self) -> Atoms:
        """Return the underlying ase.Atoms object (non-destructive access)."""
        return self._atoms.copy()

    # ---- Convenience methods ----

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON persistence."""
        return {
            "atomic_numbers": self.atomic_numbers.tolist(),
            "frac_coords": self.frac_coords.tolist(),
            "lattice": self.lattice.tolist(),
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CrystalStructure":
        """Deserialize from dictionary."""
        return cls.from_frac_coords(
            atomic_numbers=np.array(d["atomic_numbers"]),
            frac_coords=np.array(d["frac_coords"]),
            lattice=np.array(d["lattice"]),
            properties=d.get("properties", {}),
        )

    def copy(self) -> "CrystalStructure":
        """Return a deep copy."""
        return CrystalStructure(self._atoms.copy(), copy.deepcopy(self.properties))

    def add_noise_to_frac(self, std: float, seed: Optional[int] = None) -> "CrystalStructure":
        """
        Add Gaussian noise to fractional coordinates and return a NEW structure.
        The original is not modified.

        Args:
            std: Standard deviation of Gaussian noise in fractional space.
            seed: Optional random seed for reproducibility.

        Returns:
            New CrystalStructure with perturbed fractional coordinates.
        """
        rng = np.random.RandomState(seed)
        noisy_frac = self.frac_coords + rng.normal(0, std, (self.num_atoms, 3))
        noisy_frac = np.clip(noisy_frac, 0.0, 1.0 - 1e-12)
        return CrystalStructure.from_frac_coords(
            self.atomic_numbers, noisy_frac, self.lattice.copy(), copy.deepcopy(self.properties)
        )

    def noise_lattice(self, sigma: float, rng: Optional[np.random.RandomState] = None) -> "CrystalStructure":
        """
        Return a NEW structure with MatterGen-style multiplicative lattice
        noise (log-Gram space, Jain et al. 2024, arXiv:2312.03687 §3.2):
        additive isotropic Gaussian noise in the log-domain of the Gram
        matrix G = L·L^T, symmetrized so the perturbation is a pure strain.

            logG' = logG + sigma * (eps + eps^T)/2,  eps ~ N(0, 1)
            L'    = cholesky(exp(logG'))          (LOWER factor, numpy convention)

        Convention: ``L`` stores lattice vectors as ROWS with
        G = L @ L.T, and ``np.linalg.cholesky(G)`` returns the LOWER factor C
        with C @ C.T = G -- so C is the correct L' directly.  Adding ``.T``
        here (as the original code did, copying scipy's upper-factor
        convention) returns a matrix whose Gram matrix is C.T @ C != G, i.e.
        a sigma-INDEPENDENT gross distortion of the cell; see the bugfix note
        below.  Do not reintroduce a transpose.

        Properties: the perturbation is multiplicative in lattice space
        (hence always positive-definite and volume-preserving up to the
        strain), isotropic over strains, and respects lattice symmetry —
        unlike independent per-vector Gaussian noise, which can flip a
        vector's direction and degrade the cell.  The rotation gauge is
        fixed by the Cholesky factor (rotation is redundant for the
        fractional-coordinate dynamics).  The original is not modified.

        Args:
            sigma: Noise level in log-Gram space.  Protocol value
                sigma_latt = 0.1 * sigma_max (paper section 6.0.5).
            rng: RandomState; a fresh one is created if not given.
        """
        rng = rng or np.random.RandomState()
        L = self.lattice
        G = L @ L.T
        lam, U = np.linalg.eigh(G)
        logG = (U * np.log(np.maximum(lam, 1e-12))) @ U.T
        eps = rng.normal(0.0, 1.0, (3, 3))
        eps = 0.5 * (eps + eps.T)          # symmetric: pure strain, no rotation
        lam2, U2 = np.linalg.eigh(logG + sigma * eps)
        G_new = (U2 * np.exp(lam2)) @ U2.T
        # BUGFIX: np.linalg.cholesky(G) returns the LOWER factor C
        # with C @ C.T = G (scipy.linalg.cholesky with lower=False returns the
        # upper factor U with U.T @ U = G -- the `.T` below is right only under
        # that convention).  With numpy, L_new = C.T gives L_new @ L_new.T =
        # C.T @ C != G_new, so the returned cell did NOT have the intended Gram
        # matrix: the error is sigma-INDEPENDENT and large (principal stretches
        # 0.45-2.14 for real MP cells), i.e. every make_initial() call produced
        # a grossly distorted cell regardless of sigma.  C (no transpose) does
        # satisfy C @ C.T = G_new, so the cell is the input cell strained by
        # exp(sigma*eps) up to a rigid rotation -- the documented intent.  The
        # rotation gauge is still fixed by the Cholesky factor.
        L_new = np.linalg.cholesky(G_new)    # rows are lattice vectors
        return CrystalStructure.from_frac_coords(
            self.atomic_numbers, self.frac_coords, L_new,
            copy.deepcopy(self.properties),
        )

    def add_noise_to_cart(self, std: float, seed: Optional[int] = None) -> "CrystalStructure":
        """
        Add Gaussian noise to Cartesian coordinates and return a NEW structure.

        Args:
            std: Standard deviation in Å.
            seed: Optional random seed.
        """
        rng = np.random.RandomState(seed)
        noisy_cart = self.cart_coords + rng.normal(0, std, (self.num_atoms, 3))
        return CrystalStructure.from_cartesian(
            self.atomic_numbers, noisy_cart, self.lattice.copy(), copy.deepcopy(self.properties)
        )

    def get_distance_matrix(self) -> np.ndarray:
        """All-pairs distance matrix with minimum image convention (N, N)."""
        return self._atoms.get_all_distances(mic=True)

    def get_minimum_distance(self) -> float:
        """Minimum pairwise distance (useful for overlap detection)."""
        dm = self.get_distance_matrix()
        np.fill_diagonal(dm, np.inf)
        return dm.min()

    def __repr__(self) -> str:
        return (
            f"CrystalStructure({self.chemical_formula}, "
            f"n_atoms={self.num_atoms}, "
            f"volume={self.volume:.2f} Å³)"
        )

    def __len__(self) -> int:
        return self.num_atoms
