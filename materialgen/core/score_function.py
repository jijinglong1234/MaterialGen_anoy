"""
ScoreFunction — computes the score s(x) = ∇_x log p(x) for crystal structures.

Core insight (verified by Elijošius et al., Nature Comms, 2025):
    For a system at thermal equilibrium with Boltzmann distribution
        p(x) ∝ exp(-β U(x))
    the score function IS the negative force:
        s(x) = ∇_x log p(x) = -β ∇_x U(x) = β · F(x)

    where F(x) = -∇U(x) are the forces (energy gradients) from any calculator.

Design:
    The NNP (or any ASE Calculator) provides U(x) and F(x) = -∇U(x).
    NNPScore transforms: s(x) = β * F(x), i.e., s(x) = -β * ∇U(x).

    This module follows the patterns established by:
    - MACE: https://github.com/ACEsuit/mace (ASE Calculator interface)
    - CHGNet: https://github.com/CederGroupHub/chgnet (predict_structure)
    - DiffCSP: https://github.com/jiaor17/DiffCSP (score from denoising network)

Architecture:
    ScoreFunction (ABC)
    ├── NNPScore       — wraps any ASE Calculator (LJ, MACE, CHGNet, ...)
    │                     computes s(x) = β * forces
    └── HybridScore    — s_λ = (1-λ)·s_NNP + λ·s_learned
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..utils.constants import BETA_300K

if TYPE_CHECKING:
    from .crystal import CrystalStructure


# ==============================================================================
# ScoreResult — container for the computed score
# ==============================================================================

@dataclass
class ScoreResult:
    """
    Result of a score computation on a crystal structure.

    Attributes:
        cart_score: (N, 3) array — score for Cartesian coordinates (eV/Å units).
            This is the primary score used for Langevin dynamics in Cartesian space.
        frac_score: (N, 3) array — score for fractional coordinates.
            Related to cart_score by: frac_score = L^{-T} · cart_score
            (frac displacement d maps to cartesian d @ L, hence the inverse
            transform; the naive L^T form amplifies the step by |L|^2).
        lattice_score: (3, 3) array or None — score for lattice matrix L.
            s_L = -β · ∂U/∂L = -β · V · σ · L^{-T}
            where σ is the stress tensor and V = det(L).
        energy: float or None — the potential energy U(x) in eV.
        forces: (N, 3) array or None — the raw forces F = -∇U (eV/Å).
        stress: (3, 3) array or None — the stress tensor σ (eV/Å³).
        uncertainty: (N,) array or None — per-atom force uncertainty (if available).
        is_conservative: bool — True if ∇×s = 0. Always True for NNP-derived scores.
        metadata: dict — extra information (timing, NNP backend info, etc.).
    """
    cart_score: np.ndarray                # (N, 3)
    frac_score: np.ndarray                # (N, 3)
    lattice_score: Optional[np.ndarray]   # (3, 3) or None
    energy: Optional[float] = None
    forces: Optional[np.ndarray] = None   # (N, 3)
    stress: Optional[np.ndarray] = None   # (3, 3)
    uncertainty: Optional[np.ndarray] = None
    is_conservative: bool = True          # NNP scores are always curl-free
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        fmean = np.linalg.norm(self.cart_score) / max(len(self.cart_score), 1)
        return (
            f"ScoreResult(energy={self.energy:.4f} eV, "
            f"|s|_mean={fmean:.4f}, "
            f"conservative={self.is_conservative})"
        )

    @property
    def score_magnitude(self) -> float:
        """Frobenius norm of the Cartesian score."""
        return float(np.linalg.norm(self.cart_score))

    @property
    def force_magnitude(self) -> float:
        """Frobenius norm of forces (if available)."""
        if self.forces is None:
            return 0.0
        return float(np.linalg.norm(self.forces))


# ==============================================================================
# ScoreFunction — abstract base
# ==============================================================================

class ScoreFunction(ABC):
    """
    Abstract base class for score functions.

    A ScoreFunction computes s(x) = ∇_x log p(x), the gradient of the
    log-probability density with respect to the structure coordinates.
    """

    @abstractmethod
    def compute(self, crystal: "CrystalStructure", t: Optional[float] = None) -> ScoreResult:
        """
        Compute the score for a crystal structure.

        Args:
            crystal: CrystalStructure to evaluate.
            t: Diffusion time (ignored for time-independent scores like NNP).

        Returns:
            ScoreResult with score vectors, energy, and metadata.
        """
        ...

    def __call__(
        self,
        crystal: "CrystalStructure",
        t: Optional[float] = None,
    ) -> ScoreResult:
        """Convenience: call compute() directly."""
        return self.compute(crystal, t)

    def batch_compute(
        self,
        crystals: list["CrystalStructure"],
        t: Optional[float] = None,
    ) -> list[ScoreResult]:
        """Default sequential batch computation. Override for parallel."""
        return [self.compute(c, t) for c in crystals]


# ==============================================================================
# NNPScore — wraps an ASE Calculator to produce the physical score
# ==============================================================================

class NNPScore(ScoreFunction):
    """
    Score function derived from a neural network potential (or any ASE Calculator).

    The fundamental relationship:
        s(x) = -β ∇_x U(x) = β · F(x)

    where F(x) = -∇U(x) are the forces from the calculator, and β = 1/(k_B T).

    This implementation follows the ASE Calculator pattern, unifying LJ, MACE,
    CHGNet, and any other backend through a single interface.

    Reference:
        MACE Calculator: https://github.com/ACEsuit/mace/blob/main/mace/calculators/mace.py
            - Uses ase.Atoms.get_forces() → returns -∇U
            - Uses ase.Atoms.get_stress() → returns σ (Voigt 6-vector)
        CHGNet: https://github.com/CederGroupHub/chgnet
            - model.predict_structure() → {'f': forces, 's': stress}

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Any ASE Calculator attached to an Atoms object. Examples:
        - LennardJones() for Ar — no NNP required, good for testing
        - MACECalculator(model_path="MACE-MP-0.model") for universal NNP
        - CHGNetCalculator() for CHGNet
    beta : float
        Inverse temperature 1/(k_B T) in eV⁻¹. Default: β(300K) ≈ 38.68 eV⁻¹.
    compute_stress : bool
        Whether to compute lattice score from stress tensor.
    label : str
        Human-readable label for logging.
    """

    def __init__(
        self,
        calculator,
        beta: float = BETA_300K,
        compute_stress: bool = True,
        label: str = "NNPScore",
    ):
        self._calc = calculator
        self.beta = beta
        self.compute_stress_flag = compute_stress
        self.label = label

        # Internal tracking
        self._n_calls: int = 0
        self._total_time: float = 0.0

    # ---- Properties ----

    @property
    def calculator(self):
        """The underlying ASE Calculator."""
        return self._calc

    @property
    def temperature(self) -> float:
        """Effective temperature in Kelvin."""
        return 1.0 / (8.617333262145e-5 * self.beta) if self.beta > 0 else float("inf")

    @property
    def n_calls(self) -> int:
        """Number of times compute() has been called (NFE counter)."""
        return self._n_calls

    # ---- Core computation ----

    def compute(
        self,
        crystal: "CrystalStructure",
        t: Optional[float] = None,
    ) -> ScoreResult:
        """
        Compute NNP score: s(x) = -β ∇U(x) = β · F(x).

        Steps:
        1. Attach the calculator to the structure's ASE Atoms
        2. Compute energy U, forces F = -∇U (via get_forces), stress σ
        3. Transform: s_cart = β * F  (score = negative force scaled by β)
        4. Transform to fractional coordinates: s_frac = L^T · s_cart
        5. Compute lattice score from stress: s_L = -β · V · σ · L^{-T}
        6. Detach calculator (cleanup)
        """
        import time
        t0 = time.perf_counter()

        atoms = crystal.ase_atoms
        atoms.calc = self._calc

        try:
            # --- Energy ---
            energy = atoms.get_potential_energy()  # eV

            # --- Forces: F = -∇U (eV/Å) ---
            forces = atoms.get_forces()  # (N, 3), negative gradient of energy

            # --- Score: s_cart = -β ∇U = β · F ---
            # (forces ARE -∇U, so score = β * forces)
            cart_score = self.beta * forces  # (N, 3)

            # --- Stress & lattice score ---
            stress = None
            lattice_score = None
            if self.compute_stress_flag:
                try:
                    # get_stress returns Voigt notation (6,) → convert to (3,3)
                    stress_voigt = atoms.get_stress()  # eV/Å³, Voigt order [xx,yy,zz,yz,xz,xy]
                    stress = self._voigt_to_tensor(stress_voigt)
                    lattice_score = self._compute_lattice_score(crystal.lattice, stress)
                except Exception:
                    # Some calculators don't support stress
                    pass

            # --- Fractional score ---
            # s_frac = L^{-T} · s_cart.  A step d_frac moves the cartesian
            # position by d_frac @ L, so the score must be divided by L, not
            # multiplied: the L^T form amplified the ALD/PF-ODE drift by
            # |L|^2 ~ 15 (cubic 3.9 Å) and pushed first steps ~3 Å through
            # the Pauli wall (fix).
            lattice = crystal.lattice
            frac_score = np.linalg.solve(lattice.T, cart_score.T)  # (3, N)
            frac_score = frac_score.T               # (N, 3)

        finally:
            atoms.calc = None  # Always detach

        # --- Bookkeeping ---
        dt = time.perf_counter() - t0
        self._n_calls += 1
        self._total_time += dt

        return ScoreResult(
            cart_score=cart_score,
            frac_score=frac_score,
            lattice_score=lattice_score,
            energy=energy,
            forces=forces,
            stress=stress,
            uncertainty=None,  # NNPScore alone doesn't provide uncertainty
            is_conservative=True,
            metadata={
                "backend": self.label,
                "beta": self.beta,
                "temperature_K": self.temperature,
                "compute_time_s": dt,
                "n_calls": self._n_calls,
                "t_diffusion": t,
            },
        )

    # ---- Lattice score from stress ----

    def _compute_lattice_score(self, lattice: np.ndarray, stress: np.ndarray) -> np.ndarray:
        """
        Compute lattice score from stress tensor.

        For the lattice matrix L (3×3, rows are vectors), the derivative of
        energy with respect to lattice is:
            ∂U/∂L = V · σ · L^{-T}

        where V = det(L) is the cell volume and σ is the stress tensor.

        The score for the lattice is then:
            s_L = -β · ∂U/∂L = -β · V · σ · L^{-T}
        """
        return lattice_score_from_stress(lattice, stress, self.beta)

    @staticmethod
    def _voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
        """
        Convert Voigt notation [xx, yy, zz, yz, xz, xy] to 3×3 stress tensor.
        ASE uses this convention for get_stress().
        """
        xx, yy, zz, yz, xz, xy = voigt[0], voigt[1], voigt[2], voigt[3], voigt[4], voigt[5]
        return np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ])

    # ---- Statistics ----

    def get_stats(self) -> dict:
        """Get cumulative computation statistics."""
        return {
            "n_calls": self._n_calls,
            "total_time_s": self._total_time,
            "avg_time_s": self._total_time / max(self._n_calls, 1),
            "beta": self.beta,
            "temperature_K": self.temperature,
        }

    def reset_stats(self):
        """Reset call counters."""
        self._n_calls = 0
        self._total_time = 0.0

    def __repr__(self) -> str:
        return (
            f"NNPScore(backend={self.label}, β={self.beta:.2f} eV⁻¹, "
            f"T={self.temperature:.0f} K, calls={self._n_calls})"
        )


# ==============================================================================
# HybridScore — weighted combination of two scores
# ==============================================================================

class HybridScore(ScoreFunction):
    """
    Weighted combination of two score functions.

    s_λ(x, t) = (1 - λ) · s_A(x) + λ · s_B(x, t)

    Used for the λ-mixing experiment:
    - λ = 0: pure NNP score
    - λ = 1: pure learned score
    - 0 < λ < 1: hybrid

    This experiment directly quantifies the marginal value of learning
    over physics.
    """

    def __init__(
        self,
        score_a: ScoreFunction,   # typically NNPScore
        score_b: ScoreFunction,   # typically LearnedScore
        lam: float = 0.5,
        label: str = "HybridScore",
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"λ must be in [0, 1], got {lam}")
        self.score_a = score_a
        self.score_b = score_b
        self.lam = lam
        self.label = label

    @property
    def lam(self) -> float:
        return self._lam

    @lam.setter
    def lam(self, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"λ must be in [0, 1], got {value}")
        self._lam = value

    def compute(
        self,
        crystal: "CrystalStructure",
        t: Optional[float] = None,
    ) -> ScoreResult:
        """Compute s_λ = (1-λ)·s_A + λ·s_B."""
        result_a = self.score_a.compute(crystal, t)
        result_b = self.score_b.compute(crystal, t)

        # Weighted combination
        cart_score = (1 - self.lam) * result_a.cart_score + self.lam * result_b.cart_score
        frac_score = (1 - self.lam) * result_a.frac_score + self.lam * result_b.frac_score

        # Lattice score
        lattice_score = None
        if result_a.lattice_score is not None and result_b.lattice_score is not None:
            lattice_score = (1 - self.lam) * result_a.lattice_score + self.lam * result_b.lattice_score
        elif result_a.lattice_score is not None:
            lattice_score = (1 - self.lam) * result_a.lattice_score
        elif result_b.lattice_score is not None:
            lattice_score = self.lam * result_b.lattice_score

        # Hybrid is NOT guaranteed conservative unless both scores are
        is_conservative = result_a.is_conservative and result_b.is_conservative

        return ScoreResult(
            cart_score=cart_score,
            frac_score=frac_score,
            lattice_score=lattice_score,
            energy=result_a.energy,  # Use NNP energy if available
            forces=result_a.forces,
            stress=result_a.stress,
            uncertainty=result_a.uncertainty,
            is_conservative=is_conservative,
            metadata={
                "lambda": self.lam,
                "score_a": result_a.metadata,
                "score_b": result_b.metadata,
                "backend": self.label,
            },
        )

    def __repr__(self) -> str:
        return f"HybridScore(λ={self.lam:.2f}, a={self.score_a}, b={self.score_b})"


# ==============================================================================
# Factory function
# ==============================================================================

def build_score_function(
    backend: str,
    calculator=None,
    beta: float = BETA_300K,
    **kwargs,
) -> ScoreFunction:
    """
    Build a ScoreFunction from a backend name.

    Args:
        backend: "lj" (Lennard-Jones), "mace" (MACE), "chgnet", "m3gnet",
                 or path to a calculator.
        calculator: Pre-built ASE Calculator (optional, takes precedence).
        beta: Inverse temperature.

    Returns:
        A ScoreFunction instance.

    Usage:
        # Lennard-Jones (no NNP needed)
        from ase.calculators.lj import LennardJones
        calc = LennardJones(sigma=3.4, epsilon=0.0104)
        score_fn = build_score_function("lj", calculator=calc)

        # MACE (if installed)
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_path="MACE-MP-0.model", device="cpu")
        score_fn = build_score_function("mace", calculator=calc)
    """
    if calculator is not None:
        return NNPScore(calculator, beta=beta, label=backend, **kwargs)
    else:
        # Auto-create based on backend name
        if backend == "lj":
            from ase.calculators.lj import LennardJones
            calc = LennardJones()
            return NNPScore(calc, beta=beta, label="LennardJones", **kwargs)
        elif backend == "mace":
            try:
                from mace.calculators import MACECalculator
                calc = MACECalculator(model_path="MACE-MP-0.model", device="cpu")
                return NNPScore(calc, beta=beta, label="MACE-MP-0", **kwargs)
            except ImportError:
                raise ImportError("MACE not installed. Install with: pip install mace-torch")
        elif backend == "chgnet":
            try:
                # CHGNet uses pymatgen.Structure, not ASE Calculator directly.
                # Use the CHGNetCalculator wrapper in nnp/chgnet.py instead.
                raise NotImplementedError("Use CHGNetCalculator wrapper (see nnp/chgnet.py)")
            except ImportError:
                raise ImportError("CHGNet not installed. Install with: pip install chgnet")
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'lj', 'mace', or provide a calculator.")


# ==============================================================================
# Lattice score helper — shared by NNPScore and PauliScore
# ==============================================================================

def lattice_score_from_stress(lattice: np.ndarray, stress: np.ndarray,
                              beta: float) -> Optional[np.ndarray]:
    """
    s_L = -beta * V * sigma * L^{-T}  (V = |det L|).

    The lattice-matrix derivative of the energy is dU/dL = V·σ·L^{-T};
    with s_L = -β·dU/dL this is the score of the lattice degrees of freedom
    (paper section 4.2).  Summing the NNP and Pauli stress tensors before
    applying this map yields the lattice score of the augmented potential
    U_NNP + U_Pauli.
    """
    V = np.abs(np.linalg.det(lattice))
    if V <= 0 or stress is None:
        return None
    L_inv_T = np.linalg.inv(lattice).T
    return -beta * V * stress @ L_inv_T


# ==============================================================================
# PauliScore — score of the analytic Pauli repulsion potential (Layer 1)
# ==============================================================================

class PauliScore(ScoreFunction):
    """
    Score function of the analytic Pauli repulsion potential.

        s_Pauli(x) = -beta * grad U_Pauli(x)

    Zero-training, exactly conservative, active only below the covalent
    radius sum.  Also provides the lattice score from the pair-potential
    virial (paper section 4.2): the wall's compressive stress resists
    lattice collapse, so the augmentation acts on the lattice degrees of
    freedom as well as on individual pairs.
    """

    def __init__(self, pauli=None, beta: float = BETA_300K, label: str = "PauliScore"):
        from .pauli import PauliRepulsion
        self.pauli = pauli or PauliRepulsion()
        self.beta = beta
        self.label = label
        self._n_calls: int = 0

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def compute(self, crystal: "CrystalStructure", t: Optional[float] = None) -> ScoreResult:
        atoms = crystal.ase_atoms
        energy, forces, stress = self.pauli.energy_forces_stress(atoms)
        cart_score = self.beta * forces
        # see NNPScore.compute: frac = cart @ inv(L), not L^T @ cart
        frac_score = np.linalg.solve(crystal.lattice.T, cart_score.T).T
        lattice_score = lattice_score_from_stress(crystal.lattice, stress, self.beta)
        self._n_calls += 1
        return ScoreResult(
            cart_score=cart_score,
            frac_score=frac_score,
            lattice_score=lattice_score,
            energy=energy,
            forces=forces,
            stress=stress,
            uncertainty=None,
            is_conservative=True,
            metadata={
                "backend": self.label,
                "n_active_pairs": self.pauli.active_pair_count(atoms),
                "t_diffusion": t,
            },
        )

    def __repr__(self) -> str:
        return f"PauliScore(beta={self.beta:.2f}, {self.pauli!r})"


# ==============================================================================
# AugmentedScore — NNP + Pauli (physics decomposition, Layer 1 assembly)
# ==============================================================================

class AugmentedScore(ScoreFunction):
    """
    Physics-augmented score (paper Eq. augmented-score):

        s_aug(x) = -beta * grad [ U_NNP(x) + U_Pauli(x) ]

    Long-range chemistry from the NNP, short-range universality from the
    analytic Pauli potential. Exactly conservative by construction (sum of
    two gradients of scalar potentials). The Pauli term vanishes at
    equilibrium, so the NNP score is preserved verbatim in the reliable
    region.
    """

    def __init__(
        self,
        nnp_score: ScoreFunction,
        pauli_score: Optional[PauliScore] = None,
        label: str = "AugmentedScore",
    ):
        self.nnp_score = nnp_score
        self.pauli_score = pauli_score or PauliScore(beta=getattr(nnp_score, "beta", BETA_300K))
        self.label = label

    @property
    def n_calls(self) -> int:
        return getattr(self.nnp_score, "n_calls", 0)

    def compute(self, crystal: "CrystalStructure", t: Optional[float] = None) -> ScoreResult:
        r_nnp = self.nnp_score.compute(crystal, t)
        r_pauli = self.pauli_score.compute(crystal, t)

        energy = None
        if r_nnp.energy is not None and r_pauli.energy is not None:
            energy = r_nnp.energy + r_pauli.energy
        forces = None
        if r_nnp.forces is not None and r_pauli.forces is not None:
            forces = r_nnp.forces + r_pauli.forces

        # The augmented lattice score and stress are the sums of the NNP and
        # Pauli contributions: the Pauli wall's compressive virial resists
        # lattice collapse (paper section 4.2).
        stress = None
        if r_nnp.stress is not None and r_pauli.stress is not None:
            stress = r_nnp.stress + r_pauli.stress
        elif r_nnp.stress is not None:
            stress = r_nnp.stress
        lattice_score = None
        if r_nnp.lattice_score is not None and r_pauli.lattice_score is not None:
            lattice_score = r_nnp.lattice_score + r_pauli.lattice_score
        elif r_nnp.lattice_score is not None:
            lattice_score = r_nnp.lattice_score
        elif r_pauli.lattice_score is not None:
            lattice_score = r_pauli.lattice_score

        return ScoreResult(
            cart_score=r_nnp.cart_score + r_pauli.cart_score,
            frac_score=r_nnp.frac_score + r_pauli.frac_score,
            lattice_score=lattice_score,
            energy=energy,
            forces=forces,
            stress=stress,
            uncertainty=r_nnp.uncertainty,
            is_conservative=True,
            metadata={
                "backend": self.label,
                "nnp": r_nnp.metadata,
                "pauli": r_pauli.metadata,
                "pauli_active_pairs": r_pauli.metadata.get("n_active_pairs", 0),
                # force decomposition: downstream OOD/validity checks must judge
                # the NNP-only force (the Pauli wall legitimately exceeds any
                # finite |F| threshold at short range; it is not extrapolation)
                "nnp_forces": r_nnp.forces,
                "pauli_forces": r_pauli.forces,
                "t_diffusion": t,
            },
        )

    def __repr__(self) -> str:
        return f"AugmentedScore(nnp={self.nnp_score!r}, pauli={self.pauli_score!r})"
