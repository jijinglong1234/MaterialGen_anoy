"""
PauliRepulsion — analytic short-range repulsive potential (Layer 1, physics decomposition).

Functionality:
    Implements the universal Pauli repulsion augmentation from the paper
    (paper_outline_v0.4, Eq. pauli-potential):

        U_Pauli(x) = sum_{i<j} A_{ZiZj} * exp(-B_{ZiZj} * r_ij)
                     * Theta(0.75*(R_i + R_j) - r_ij)

    where Theta(d) = 0.5*(1 + tanh(d/w)) is a smooth step (w = 0.05 Angstrom)
    that activates only below 0.75 x the covalent-radius sum (the protocol's
    Type I completeness boundary — calibration; the wall must be
    inert at equilibrium bond distances, which sit AT or BELOW the covalent
    sum), and A, B are element-pair parameters fitted to the ZBL universal
    screening potential.

    Key properties (by construction):
    - Zero training: parameters are analytic, tabulated per element pair.
    - Exactly conservative: gradient of a scalar pair potential.
    - Universal: depends only on atomic numbers, not chemical environment.
    - Vanishing at equilibrium: U == 0 when all d_ij > 0.75*(R_i + R_j).

    Parameter source:
    - ``fit_exp_params()`` derives (A, B) for any element pair from the ZBL
      potential in closed form (force and energy continuity at
      r* = 0.75*(R_i + R_j) - 0.1 Angstrom, just inside the wall's
      activation threshold). This is the single source of truth;
      ``scripts/fit_pauli_params.py`` regenerates data/pauli/pauli_params.json
      from it.
    - ``PauliRepulsion`` loads the JSON table when available and auto-fits
      on the fly for any missing pair, so the class works standalone.

Dependencies:
    numpy, ase.neighborlist
    materialgen.utils.constants.COVALENT_RADII
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.constants import COVALENT_RADII

# e^2/(4*pi*eps0) in eV*Angstrom
_E2_OVER_4PI_EPS0 = 14.3996454784255
_BOHR_RADIUS = 0.529177210903  # Angstrom

_DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "pauli" / "pauli_params.json"
)
_DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "pauli" / "probe_calibration.json"
)


# ==============================================================================
# ZBL reference potential and closed-form exponential fit
# ==============================================================================

def _zbl_screening_length(Zi: int, Zj: int) -> float:
    """ZBL screening length a = 0.8854*a0 / (Zi^0.23 + Zj^0.23) in Angstrom."""
    return 0.8854 * _BOHR_RADIUS / (Zi ** 0.23 + Zj ** 0.23)


def _zbl_phi(x: float) -> Tuple[float, float]:
    """ZBL screening function Phi(x) and its derivative dPhi/dx."""
    terms = [
        (0.1818, -3.2),
        (0.5099, -0.9423),
        (0.2802, -0.4029),
        (0.02817, -0.2016),
    ]
    phi, dphi = 0.0, 0.0
    for c, k in terms:
        e = c * np.exp(k * x)
        phi += e
        dphi += k * e
    return phi, dphi


def zbl_energy(Zi: int, Zj: int, r: float) -> float:
    """ZBL universal screening potential energy in eV at distance r (Angstrom)."""
    a = _zbl_screening_length(Zi, Zj)
    phi, _ = _zbl_phi(r / a)
    return _E2_OVER_4PI_EPS0 * Zi * Zj * phi / r


def zbl_force(Zi: int, Zj: int, r: float) -> float:
    """ZBL repulsive force -dU/dr in eV/Angstrom at distance r. Analytic."""
    a = _zbl_screening_length(Zi, Zj)
    x = r / a
    phi, dphi = _zbl_phi(x)
    dUdr = _E2_OVER_4PI_EPS0 * Zi * Zj * (dphi / (a * r) - phi / r**2)
    return -dUdr


def fit_exp_params(
    Zi: int,
    Zj: int,
    radii: Optional[Dict[int, float]] = None,
    match_offset: float = 0.1,
    threshold_frac: float = 0.75,
) -> Tuple[float, float]:
    """
    Closed-form exponential fit A*exp(-B*r) to the ZBL potential.

    Matches energy and force at r* = threshold_frac*(R_i + R_j) - match_offset
    (paper section 6.0.4, with the threshold calibration: the fit
    follows the wall activation at 0.75 x the covalent sum, i.e. just inside
    the Type I boundary):
        B = F_ZBL(r*) / U_ZBL(r*),  A = U_ZBL(r*) * exp(B*r*)

    Returns:
        (A in eV, B in Angstrom^-1)
    """
    radii = radii or COVALENT_RADII
    r_star = threshold_frac * (radii[int(Zi)] + radii[int(Zj)]) - match_offset
    if r_star <= 0.05:
        r_star = 0.05  # guard for very small radius sums
    U = zbl_energy(Zi, Zj, r_star)
    F = zbl_force(Zi, Zj, r_star)
    B = F / U
    A = U * np.exp(B * r_star)
    return float(A), float(B)


def pair_key(Zi: int, Zj: int) -> str:
    """Canonical element-pair key, e.g. '6-8' for C-O."""
    a, b = int(Zi), int(Zj)
    return f"{min(a, b)}-{max(a, b)}"


def core_energy_deriv(
    r: float, A: float, B: float, r_wall: float,
    deep_wall: bool, deep_wall_frac: float = 0.6,
) -> Tuple[float, float]:
    """
    (u_core, du_core/dr) of the pair repulsion core — L1' item 3.

    For r >= r_match = deep_wall_frac * r_wall the core is the ZBL-fitted
    exponential A*exp(-B*r).  Below r_match the exponential tail is replaced
    by a power-law floor U = C*(r_match/r)^p with C^1 matching at r_match:
    C = A*exp(-B*r_match) and, necessarily for a one-term power law,
    p = B*r_match (a free p in the design note's 4-6 range is incompatible
    with C^1 matching; tabulated pairs give p ~ 2.2-3.4).  The floor keeps
    the repulsive physics (U diverges as r^-p instead of saturating at A)
    while replacing the exponential tail's curvature with a polynomial one
    (the plan's L1' design item 3).
    """
    if deep_wall:
        r_match = deep_wall_frac * r_wall
        if r < r_match:
            C = A * np.exp(-B * r_match)
            p = B * r_match
            u = C * (r_match / r) ** p
            dudr = -p * C * r_match ** p / r ** (p + 1)
            return u, dudr
    e = np.exp(-B * r)
    return A * e, -A * B * e


# ==============================================================================
# PauliRepulsion — energy/force evaluation with PBC
# ==============================================================================

class PauliRepulsion:
    """
    Analytic Pauli repulsion pair potential with periodic boundary conditions.

    Args:
        params: Optional dict {"Zi-Zj": {"A": eV, "B": Angstrom^-1}}. Missing
            pairs are auto-fitted from ZBL on first use.
        params_path: Optional JSON table to load (defaults to
            data/pauli/pauli_params.json when it exists).
        radii: Covalent radii dict {Z: Angstrom}; defaults to Cordero et al.
            (2008) from utils.constants.
        w: Smooth-step width in Angstrom (paper: 0.05).
        cutoff_buffer: Neighbor-list buffer beyond the smooth-step tail.
        deep_wall (L1' item 3): replace the exponential tail below
            deep_wall_frac*r_wall with a C^1-matched power-law floor.
        deep_wall_frac: floor switch point as a fraction of r_wall (0.6).
        calibration (L1' item 2): per-pair A scale factors; when None, the
            probe-calibration table data/pauli/probe_calibration.json is
            auto-loaded if present.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Dict[str, float]]] = None,
        params_path: Optional[str] = None,
        radii: Optional[Dict[int, float]] = None,
        w: float = 0.05,
        cutoff_buffer: float = 0.3,
        threshold_frac: float = 0.75,
        deep_wall: bool = False,
        deep_wall_frac: float = 0.6,
        calibration: Optional[Dict[str, float]] = None,
        calibration_path: Optional[str] = None,
    ):
        self.radii = {int(z): float(r) for z, r in (radii or COVALENT_RADII).items()}
        self.w = float(w)
        self.cutoff_buffer = float(cutoff_buffer)
        # L1' items 2-3: deep-wall power-law floor (item 3) and
        # probe-driven entry-force calibration (item 2).  Both are off by
        # default so the class stays a faithful Eq. pauli-potential evaluator;
        # run_phase1.make_score enables deep_wall for every L1-containing
        # condition, and the probe-calibration table (data/pauli/
        # probe_calibration.json) is auto-loaded when present.
        self.deep_wall = bool(deep_wall)
        self.deep_wall_frac = float(deep_wall_frac)
        self._calibration: Dict[str, float] = {}
        if calibration:
            self._calibration = {str(k): float(v) for k, v in calibration.items()}
        else:
            cpath = Path(calibration_path) if calibration_path else _DEFAULT_CALIBRATION_PATH
            if cpath.exists():
                with open(cpath) as f:
                    table = json.load(f).get("pairs", {})
                self._calibration = {
                    str(k): float(v["scale"]) for k, v in table.items()
                }
        # Activation threshold as a fraction of the covalent-radius sum
        # (calibration): real equilibrium bonds sit AT or BELOW
        # the covalent sum (Ti-O 1.97 A vs 2.13 A sum), so a wall centered at
        # the sum is fully ON at equilibrium and pushes bonds apart.  The
        # wall defends the protocol's Type I completeness boundary instead
        # (0.75 x sum) and is inert at equilibrium distances.
        self.threshold_frac = float(threshold_frac)

        self._params: Dict[str, Tuple[float, float]] = {}
        # Explicit params take precedence; then JSON table; auto-fit fills gaps.
        if params:
            for k, v in params.items():
                self._params[str(k)] = (float(v["A"]), float(v["B"]))
        path = Path(params_path) if params_path else _DEFAULT_PARAMS_PATH
        if path.exists():
            with open(path) as f:
                table = json.load(f)["pairs"]
            for k, v in table.items():
                self._params.setdefault(k, (float(v["A"]), float(v["B"])))

    # ---- Parameter access ----

    def get_params(self, Zi: int, Zj: int) -> Tuple[float, float]:
        """(A, B) for an element pair; auto-fits from ZBL if not tabulated."""
        key = pair_key(Zi, Zj)
        if key not in self._params:
            self._params[key] = fit_exp_params(
                Zi, Zj, self.radii, threshold_frac=self.threshold_frac,
            )
        return self._params[key]

    @property
    def cutoff(self) -> float:
        """Neighbor-list cutoff: max covalent sum + smooth-step tail."""
        max_rsum = max(self.radii.values()) * 2.0
        return max_rsum + 5.0 * self.w + self.cutoff_buffer

    # ---- Core evaluation ----

    def energy_forces(self, atoms) -> Tuple[float, np.ndarray]:
        """
        Compute U_Pauli and its forces for an ase.Atoms object (PBC-aware).

        Returns:
            (energy in eV, forces (N, 3) in eV/Angstrom) where forces = -grad U.
        """
        energy, forces, _ = self.energy_forces_stress(atoms)
        return energy, forces

    def energy_forces_stress(self, atoms) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute U_Pauli, its forces, and the pair-potential virial stress.

        Returns:
            (energy in eV, forces (N, 3) in eV/Angstrom, stress (3, 3) in
            eV/Angstrom^3).  stress follows the ASE convention (positive =
            tensile); the Pauli wall gives compressive stress along
            compressed bonds, so the lattice score derived from it resists
            lattice collapse (paper section 4.2: the Pauli augmentation acts
            on the lattice degrees of freedom as well as on individual pairs).
        """
        from ase.neighborlist import neighbor_list

        numbers = atoms.get_atomic_numbers()
        n = len(atoms)
        forces = np.zeros((n, 3))
        energy = 0.0
        stress = np.zeros((3, 3))
        if n < 2:
            return energy, forces, stress

        i_idx, j_idx, d, D = neighbor_list("ijdD", atoms, self.cutoff)
        if len(i_idx) == 0:
            return energy, forces, stress

        # Deduplicate unordered pairs, keeping the MINIMUM-IMAGE entry per
        # pair.  With cutoff comparable to the cell size, ASE returns many
        # periodic images of the same pair; first-occurrence selection
        # evaluated phantom long-range images and missed real overlaps
        # (E_pauli 0.7 eV vs 28.8 eV on the SrTiO3 reference —
        # fix).
        keys = np.maximum(i_idx, j_idx) * n + np.minimum(i_idx, j_idx)
        order = np.lexsort((d, keys))               # per key, ascending d
        first = order[np.unique(keys[order], return_index=True)[1]]

        w = self.w
        for k in first:
            i, j = int(i_idx[k]), int(j_idx[k])
            r = float(d[k])
            vec = D[k]
            Zi, Zj = int(numbers[i]), int(numbers[j])
            if Zi not in self.radii or Zj not in self.radii:
                continue
            r_sum = self.radii[Zi] + self.radii[Zj]
            r_wall = self.threshold_frac * r_sum
            x = r_wall - r
            # Smooth step Theta(x) = 0.5(1+tanh(x/w)); skip deep-tail pairs.
            if x < -8.0 * w:
                continue
            A, B = self.get_params(Zi, Zj)
            A = A * self._calibration.get(pair_key(Zi, Zj), 1.0)
            theta = 0.5 * (1.0 + np.tanh(x / w))
            u_core, du_core_dr = core_energy_deriv(
                r, A, B, r_wall, self.deep_wall, self.deep_wall_frac,
            )
            energy += u_core * theta

            # dU/dr = du_core/dr * Theta + u_core * dTheta/dr
            # (Theta' = -(1/2w)*sech^2(x/w); for the exponential branch this
            # reduces to the pre-L1' form -A*B*e^{-Br}*Theta + A*e^{-Br}*Theta')
            cosh_term = np.cosh(x / w)
            dtheta_dr = -(1.0 / (2.0 * w)) / (cosh_term * cosh_term)
            dUdr = du_core_dr * theta + u_core * dtheta_dr

            if r < 1e-8:
                # Exact overlap: pick an arbitrary separation direction.
                unit = np.array([1.0, 0.0, 0.0])
            else:
                unit = vec / r  # points i -> j
            f_pair = dUdr * unit  # force on i; points away from j when dUdr < 0
            forces[i] += f_pair
            forces[j] -= f_pair

            # Pair virial (ASE convention, positive = tensile):
            #     sigma = (1/V) * sum_pairs (dU/dr)/r * outer(D, D)
            # with D = r_j - r_i the minimum-image vector (D points i -> j,
            # so r_i - r_j = -D and the standard -1/V * sum f_a (x) r_a form
            # reduces to the expression above).  dUdr < 0 for the repulsive
            # wall, hence compressive (negative) stress along compressed
            # bonds.
            if r < 1e-8:
                dd = np.outer(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
            else:
                dd = np.outer(vec, vec) / r
            stress += dUdr * dd

        V = float(atoms.get_volume())
        if V > 0:
            stress /= V
        else:
            stress = np.zeros((3, 3))

        return energy, forces, stress

    def active_pair_count(self, atoms, theta_threshold: float = 1e-3) -> int:
        """Number of pairs with Theta above threshold (Pauli term 'active')."""
        from ase.neighborlist import neighbor_list

        numbers = atoms.get_atomic_numbers()
        n = len(numbers)
        if n < 2:
            return 0
        i_idx, j_idx, d, _ = neighbor_list("ijdD", atoms, self.cutoff)
        if len(i_idx) == 0:
            return 0
        keys = np.maximum(i_idx, j_idx) * n + np.minimum(i_idx, j_idx)
        order = np.lexsort((d, keys))               # min-image per pair
        first = order[np.unique(keys[order], return_index=True)[1]]
        count = 0
        for k in first:
            Zi = int(numbers[i_idx[k]])
            Zj = int(numbers[j_idx[k]])
            r_sum = self.radii.get(Zi, 0.0) + self.radii.get(Zj, 0.0)
            theta = 0.5 * (1.0 + np.tanh((self.threshold_frac * r_sum - d[k]) / self.w))
            if theta > theta_threshold:
                count += 1
        return count

    def __repr__(self) -> str:
        return (
            f"PauliRepulsion(w={self.w}, n_pairs_tabulated={len(self._params)}, "
            f"deep_wall={self.deep_wall}, n_pairs_calibrated={len(self._calibration)})"
        )
