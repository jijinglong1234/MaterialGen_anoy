"""
SafetyMonitor — step-level rejection and phase-aware sigma decay (Layer 4).

Functionality:
    Implements the post-step safety check of paper_outline_v0.4 (section 4.3,
    Layer 4; thresholds in section 6.0.5). After a Langevin step is finalized,
    the step is REJECTED if any of:

        (a) U(x) > energy_per_atom_max (default 1e4 eV/atom) — energy explosion
        (b) max_i |F_i| > force_max (default 1e3 eV/Angstrom) — force explosion
        (c) min_ij d_ij < d_min_abs (default 0.3 Angstrom) — hard overlap
        (d) |det L_new| / |det L_ref| > volume_ratio_max (default 10) — cell explosion

    On rejection the sampler backtracks and reduces sigma with the
    phase-aware decay factor:

        gamma_decay(sigma) = 1 - (1 - gamma_base) * (1 - sigma/sigma_max)^p

Dependencies:
    numpy
    materialgen.core.crystal.CrystalStructure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .crystal import CrystalStructure


@dataclass
class MonitorThresholds:
    """Rejection thresholds (paper section 6.0.5 defaults)."""
    energy_per_atom_max: float = 1e4   # eV/atom
    force_max: float = 1e3             # eV/Angstrom
    d_min_abs: float = 0.3             # Angstrom
    volume_ratio_max: float = 10.0     # |det L_new| / |det L_ref|


@dataclass
class MonitorDecision:
    """Outcome of a single safety check."""
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    energy_per_atom: Optional[float] = None
    max_force: Optional[float] = None
    d_min: Optional[float] = None
    volume_ratio: Optional[float] = None


class SafetyMonitor:
    """
    Layer-4 safety monitor.

    Args:
        thresholds: MonitorThresholds instance (paper defaults if omitted).
        reference_volume: Reference cell volume for the volume-ratio check
            (typically the initial structure's volume). If None, the first
            checked structure's volume becomes the reference.
        p: Phase-aware decay exponent (paper: 2.0).
        gamma_base: Base decay factor (paper: 0.5).
    """

    def __init__(
        self,
        thresholds: Optional[MonitorThresholds] = None,
        reference_volume: Optional[float] = None,
        p: float = 2.0,
        gamma_base: float = 0.5,
    ):
        self.thresholds = thresholds or MonitorThresholds()
        self.reference_volume = reference_volume
        self.p = float(p)
        self.gamma_base = float(gamma_base)
        self.n_checks: int = 0
        self.n_rejections: int = 0

    # ---- Core check ----

    def check(
        self,
        crystal: CrystalStructure,
        energy: Optional[float] = None,
        forces: Optional[np.ndarray] = None,
    ) -> MonitorDecision:
        """
        Evaluate all four rejection criteria for a proposed structure.

        Args:
            crystal: Proposed structure.
            energy: Total potential energy in eV (optional).
            forces: (N, 3) forces in eV/Angstrom (optional).

        Returns:
            MonitorDecision with accepted flag and triggered reasons.
        """
        th = self.thresholds
        reasons: List[str] = []

        e_per_atom = None
        if energy is not None and crystal.num_atoms > 0:
            e_per_atom = energy / crystal.num_atoms
            if e_per_atom > th.energy_per_atom_max:
                reasons.append(f"energy_explosion({e_per_atom:.3g} eV/atom > {th.energy_per_atom_max:.3g})")

        max_f = None
        if forces is not None and len(forces) > 0:
            max_f = float(np.linalg.norm(forces, axis=1).max())
            if max_f > th.force_max:
                reasons.append(f"force_explosion({max_f:.3g} eV/A > {th.force_max:.3g})")

        d_min = None
        if crystal.num_atoms >= 2:
            d_min = crystal.get_minimum_distance()
            if d_min < th.d_min_abs:
                reasons.append(f"hard_overlap(d_min={d_min:.3f} A < {th.d_min_abs:.3f})")

        if self.reference_volume is None:
            self.reference_volume = crystal.volume
        v_ratio = None
        if self.reference_volume and self.reference_volume > 0:
            v_ratio = crystal.volume / self.reference_volume
            if v_ratio > th.volume_ratio_max:
                reasons.append(f"volume_explosion(ratio={v_ratio:.3g} > {th.volume_ratio_max:.3g})")

        self.n_checks += 1
        accepted = len(reasons) == 0
        if not accepted:
            self.n_rejections += 1

        return MonitorDecision(
            accepted=accepted,
            reasons=reasons,
            energy_per_atom=e_per_atom,
            max_force=max_f,
            d_min=d_min,
            volume_ratio=v_ratio,
        )

    # ---- Phase-aware sigma decay ----

    def sigma_decay_factor(self, sigma: float, sigma_max: float) -> float:
        """
        Phase-aware decay factor (paper section 4.3, Layer 4):

            gamma(sigma) = 1 - (1 - gamma_base) * (1 - sigma/sigma_max)^p

        Equals 1 at sigma = sigma_max (gentle early: the annealing schedule
        itself still descends, so little intervention is needed) and
        gamma_base at low sigma (strong late-phase decay: rejections near
        convergence are the costly ones).
        """
        if sigma_max <= 0:
            return self.gamma_base
        frac = min(max(1.0 - sigma / sigma_max, 0.0), 1.0)
        return 1.0 - (1.0 - self.gamma_base) * frac ** self.p

    @property
    def rejection_rate(self) -> float:
        return self.n_rejections / max(self.n_checks, 1)

    def __repr__(self) -> str:
        return (
            f"SafetyMonitor(rejections={self.n_rejections}/{self.n_checks}, "
            f"p={self.p}, gamma_base={self.gamma_base})"
        )
