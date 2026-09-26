"""
NNPBackend — abstract base class for all neural network potential backends.

Functionality:
    Defines the minimal interface that every NNP implementation must satisfy.
    All energy/force outputs are converted to consistent units (eV, eV/Å, eV/Å³).

    Abstract Base Class: NNPBackend
        - Must implement: predict(crystal) -> NNPResult
        - Can optionally implement: predict_batch(crystals) -> list[NNPResult]

    NNPResult (dataclass):
        energy: float  # total potential energy U(x) in eV
        forces: np.ndarray (N, 3)  # -∇U(x) in eV/Å (negative gradient of energy)
        stress: np.ndarray (3, 3) | None  # stress tensor in eV/Å³ (for lattice score)
        uncertainty: np.ndarray (N,) | None  # per-atom force uncertainty (if supported)
        site_energies: np.ndarray (N,) | None  # per-atom energy decomposition

    Key responsibilities:
    1. Unit conversion: Each backend may use different units internally;
       NNPBackend normalizes to eV / Å / eV/Å³.
    2. Periodic boundary handling: The backend must correctly handle PBC in
       energy and force computation.
    3. Cell relaxation: Optional interface for variable-cell relaxation.
    4. Stress computation: Required for lattice score s_L = -β · ∂U/∂L.
       ∂U/∂L_ij = V · Σ_k σ_ik · (L⁻¹)_kj (where σ is the stress tensor).

    Additional methods (optional):
        get_cutoff() -> float  # interaction cutoff radius in Å
        get_supported_elements() -> set[int]  # atomic numbers supported
        supports_stress() -> bool  # whether stress tensor is computed
        supports_uncertainty() -> bool  # whether uncertainty is provided

Dependencies:
    numpy, torch, abc
    materialgen.core.crystal.CrystalStructure
"""
