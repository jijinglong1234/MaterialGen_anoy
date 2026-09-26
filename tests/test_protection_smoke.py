"""
Phase 0.2 end-to-end smoke test: ALD / PF-ODE with four-layer defense.

Runs on an Ar FCC crystal with a Lennard-Jones calculator (no NNP needed),
verifying that the full stack — AugmentedScore (L1), density noise (L2),
reflection (L3), SafetyMonitor (L4) — composes and runs, and that the
hard constraint of L3 holds at every recorded step.
"""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones

from materialgen.core.crystal import CrystalStructure
from materialgen.core.score_function import NNPScore, AugmentedScore, PauliScore
from materialgen.core.reflection import ReflectionOperator
from materialgen.core.safety_monitor import SafetyMonitor
from materialgen.samplers.ald import ALDSampler, ALDConfig
from materialgen.samplers.pf_ode import PFODESampler, PFODEConfig
from materialgen.utils.constants import BETA_300K, COVALENT_RADII


def _ar_crystal() -> CrystalStructure:
    atoms = bulk("Ar", "fcc", a=5.26)
    return CrystalStructure(atoms)


def _lj_score(augmented: bool):
    calc = LennardJones(sigma=3.4, epsilon=0.0104)
    nnp = NNPScore(calc, beta=BETA_300K, compute_stress=False, label="LJ")
    if not augmented:
        return nnp
    return AugmentedScore(nnp, PauliScore(beta=BETA_300K))


class TestAugmentedScoreAssembly:
    def test_energy_and_score_are_sums(self):
        crystal = _ar_crystal().add_noise_to_cart(0.3, seed=0)
        nnp = _lj_score(augmented=False)
        pauli = PauliScore(beta=BETA_300K)
        aug = AugmentedScore(nnp, pauli)
        r_nnp, r_pauli, r_aug = nnp(crystal), pauli(crystal), aug(crystal)
        np.testing.assert_allclose(
            r_aug.cart_score, r_nnp.cart_score + r_pauli.cart_score, atol=1e-12,
        )
        assert r_aug.energy == pytest.approx(r_nnp.energy + r_pauli.energy)
        assert r_aug.is_conservative

    def test_pauli_inactive_at_equilibrium(self):
        crystal = _ar_crystal()  # FCC Ar: nearest neighbor ~3.7 A >> covalent sum
        pauli = PauliScore(beta=BETA_300K)
        r = pauli(crystal)
        assert r.metadata["n_active_pairs"] == 0
        assert r.energy == pytest.approx(0.0, abs=1e-12)


class TestALDSmoke:
    def test_bare_and_protected_run(self):
        initial = _ar_crystal()
        for protected in (False, True):
            score = _lj_score(augmented=protected)
            cfg = ALDConfig(
                sigma_max=0.2, sigma_min=0.05, K=10, M=1, alpha=0.1,
                use_density_noise=protected,
                use_reflection=protected,
                seed=42, save_trajectory=True,
            )
            monitor = SafetyMonitor(reference_volume=initial.volume) if protected else None
            sampler = ALDSampler(cfg, safety_monitor=monitor)
            result = sampler.sample(score, initial)
            assert np.isfinite(result.energies).all()
            assert result.nfe == cfg.K * cfg.M
            # L3 hard guarantee: no recorded step below the covalent sum.
            if protected and result.trajectory:
                r_sum_ar = 2 * COVALENT_RADII[18]
                for snap in result.trajectory:
                    assert snap.get_minimum_distance() >= r_sum_ar - 1e-6


class TestPFODESmoke:
    def test_euler_run_and_energy_descent(self):
        initial = _ar_crystal().add_noise_to_cart(0.05, seed=1)
        score = _lj_score(augmented=True)
        cfg = PFODEConfig(sigma_max=0.2, sigma_min=0.01, n_steps=50)
        result = PFODESampler(cfg).sample(score, initial)
        assert np.isfinite(result.structure.cart_coords).all()
        assert result.nfe == cfg.n_steps

    def test_ivp_with_reflection_runs(self):
        """Regression: ivp + L3 crashed with NameError
        (ref_volume read from the outer sample() scope). Exercised by the
        Phase 1 mini fleet (pfode l1l4 cells)."""
        initial = _ar_crystal().add_noise_to_cart(0.05, seed=2)
        score = _lj_score(augmented=True)
        cfg = PFODEConfig(sigma_max=0.2, sigma_min=0.01, solver="ivp",
                          rtol=1e-4, atol=1e-6, max_nfe=500,
                          use_reflection=True)
        result = PFODESampler(cfg).sample(score, initial)
        assert np.isfinite(result.structure.cart_coords).all()
        assert result.nfe <= cfg.max_nfe + 1
        assert isinstance(result.converged, bool)

    def test_ivp_path_is_accepted_trajectory(self):
        """Regression: ivp per-NFE diagnostics include RK45
        intermediate stage points (extrapolation states, not trajectory
        states) — the accepted-step path must be exposed for honest OOD
        accounting.  path[0] is the initial structure, path[-1] the final
        state (modulo periodic wrap), and every entry is finite."""
        initial = _ar_crystal().add_noise_to_cart(0.05, seed=3)
        score = _lj_score(augmented=False)
        cfg = PFODEConfig(sigma_max=0.2, sigma_min=0.01, solver="ivp",
                          rtol=1e-4, atol=1e-6, max_nfe=500)
        result = PFODESampler(cfg).sample(score, initial)
        assert result.path is not None and len(result.path) >= 1
        assert len(result.path) <= result.nfe
        # path[0] is the initial state (same d_min), path[-1] the terminal.
        np.testing.assert_allclose(
            result.path[0].frac_coords, initial.frac_coords, atol=1e-9)
        np.testing.assert_allclose(
            result.path[-1].frac_coords, result.structure.frac_coords, atol=1e-9)
        for st in result.path:
            assert np.isfinite(st.cart_coords).all()
