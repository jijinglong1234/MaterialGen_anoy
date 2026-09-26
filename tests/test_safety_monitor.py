"""
Phase 0.2 unit tests: SafetyMonitor (Layer 4).
"""

import numpy as np
import pytest

from materialgen.core.crystal import CrystalStructure
from materialgen.core.safety_monitor import SafetyMonitor, MonitorThresholds


def _crystal(d: float = 3.0, cell: float = 12.0) -> CrystalStructure:
    return CrystalStructure.from_cartesian(
        np.array([6, 6]), np.array([[0.0, 0, 0], [d, 0, 0]]), np.eye(3) * cell,
    )


class TestThresholds:
    def test_clean_structure_accepted(self):
        mon = SafetyMonitor(reference_volume=12.0**3)
        decision = mon.check(_crystal(), energy=-10.0, forces=np.zeros((2, 3)))
        assert decision.accepted

    def test_energy_explosion_rejected(self):
        mon = SafetyMonitor(reference_volume=12.0**3)
        decision = mon.check(_crystal(), energy=3e4, forces=np.zeros((2, 3)))
        assert not decision.accepted
        assert any("energy_explosion" in r for r in decision.reasons)

    def test_force_explosion_rejected(self):
        mon = SafetyMonitor(reference_volume=12.0**3)
        forces = np.array([[2e3, 0, 0], [0, 0, 0]])
        decision = mon.check(_crystal(), energy=-10.0, forces=forces)
        assert not decision.accepted
        assert any("force_explosion" in r for r in decision.reasons)

    def test_hard_overlap_rejected(self):
        mon = SafetyMonitor(reference_volume=12.0**3)
        decision = mon.check(_crystal(d=0.2), energy=-10.0, forces=np.zeros((2, 3)))
        assert not decision.accepted
        assert any("hard_overlap" in r for r in decision.reasons)

    def test_volume_explosion_rejected(self):
        mon = SafetyMonitor(reference_volume=12.0**3)
        big = _crystal(cell=30.0)
        decision = mon.check(big, energy=-10.0, forces=np.zeros((2, 3)))
        assert not decision.accepted
        assert any("volume_explosion" in r for r in decision.reasons)


class TestSigmaDecay:
    def test_decay_factor_bounds(self):
        """Paper formula verbatim: gamma = 1 - (1-gamma_base)(1-sigma/sigma_max)^p.

        gamma(sigma_max) = 1 (gentle early: the annealing schedule itself
        still descends); gamma(0) = gamma_base (strong late-phase decay:
        rejections near convergence are the costly ones).
        """
        mon = SafetyMonitor(p=2.0, gamma_base=0.5)
        assert mon.sigma_decay_factor(1.0, 1.0) == pytest.approx(1.0)
        assert mon.sigma_decay_factor(0.0, 1.0) == pytest.approx(0.5)
        # Monotonic increase with sigma.
        vals = [mon.sigma_decay_factor(s, 1.0) for s in [0.0, 0.25, 0.5, 0.75, 1.0]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))
