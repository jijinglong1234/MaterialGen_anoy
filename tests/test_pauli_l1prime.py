"""L1' items 2-3 unit tests: deep-wall power-law floor and
probe-driven entry-force calibration of the Pauli potential.

Item 3 (deep wall): below r_match = 0.6 * r_wall the exponential tail
A*exp(-B*r) is replaced by U = C*(r_match/r)^p with C^1 matching at r_match
(C = A*exp(-B*r_match), p = B*r_match).  Item 2 (calibration): per-pair
A scale factors loaded from data/pauli/probe_calibration.json when present.
"""
import numpy as np
import pytest
from ase import Atoms

from materialgen.core.pauli import (
    PauliRepulsion,
    core_energy_deriv,
    pair_key,
)

# Ti-O: r_sum = 2.26 A (Cordero), r_wall = 0.75*2.26 = 1.695 A
_TI, _O = 22, 8
BOX = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]


def _tio_atoms(d):
    """Two-atom Ti-O cell in a large box, pair distance d along x."""
    return Atoms(numbers=[_TI, _O], positions=[[0, 0, 0], [d, 0, 0]],
                 cell=BOX, pbc=True)


def _r_wall(pauli, Zi=_TI, Zj=_O):
    return pauli.threshold_frac * (pauli.radii[Zi] + pauli.radii[Zj])


class TestCoreEnergyDeriv:
    """core_energy_deriv: C^1 matching at r_match (item 3)."""

    def test_c1_continuity_at_rmatch(self):
        A, B, r_wall = 690.0, 2.91, 1.695
        r_m = 0.6 * r_wall
        u_lo, du_lo = core_energy_deriv(r_m - 1e-6, A, B, r_wall, True)
        u_hi, du_hi = core_energy_deriv(r_m + 1e-6, A, B, r_wall, True)
        np.testing.assert_allclose(u_lo, u_hi, rtol=1e-5)
        np.testing.assert_allclose(du_lo, du_hi, rtol=1e-5)
        # exact values pinned by the C^1 constraint
        C = A * np.exp(-B * r_m)
        p = B * r_m
        np.testing.assert_allclose(u_lo, C, rtol=1e-5)
        np.testing.assert_allclose(du_lo, -p * C / r_m, rtol=1e-5)
        assert 2.0 < p < 4.0  # tabulated pairs give p = B*r_m ~ 2.2-3.4

    def test_exp_branch_unchanged(self):
        A, B, r_wall = 690.0, 2.91, 1.695
        r = 1.5  # > r_match
        u, du = core_energy_deriv(r, A, B, r_wall, True)
        np.testing.assert_allclose(u, A * np.exp(-B * r), rtol=1e-12)
        np.testing.assert_allclose(du, -A * B * np.exp(-B * r), rtol=1e-12)

    def test_off_switch_reproduces_exponential(self):
        A, B, r_wall = 690.0, 2.91, 1.695
        r = 0.3  # deep region, but deep_wall=False keeps the exponential
        u, du = core_energy_deriv(r, A, B, r_wall, False)
        np.testing.assert_allclose(u, A * np.exp(-B * r), rtol=1e-12)
        np.testing.assert_allclose(du, -A * B * np.exp(-B * r), rtol=1e-12)


class TestDeepWall:
    """Full-potential behavior with deep_wall=True."""

    def test_deep_wall_diverges_beyond_exp_plateau(self):
        old = PauliRepulsion(deep_wall=False, calibration={})
        new = PauliRepulsion(deep_wall=True, calibration={})
        r_m = 0.6 * _r_wall(new)
        d = 0.3 * r_m  # deep inside the floor region
        e_old, f_old = old.energy_forces(_tio_atoms(d))
        e_new, f_new = new.energy_forces(_tio_atoms(d))
        # the power-law floor keeps growing where the exponential plateaus
        # at A ~ 690 eV: U_pow/U_exp = (r_m/d)^p * e^{-B r_m/2} ~ 4.6 here,
        # and the force ratio is ~15 (p/r vs B)
        assert e_new > 3 * e_old
        assert abs(f_new[0, 0]) > 5 * abs(f_old[0, 0])
        # still repulsive and Newton-paired
        assert f_new[0, 0] < 0 and f_new[1, 0] > 0
        np.testing.assert_allclose(f_new[0], -f_new[1], atol=1e-12)

    def test_identical_at_and_above_rmatch(self):
        old = PauliRepulsion(deep_wall=False, calibration={})
        new = PauliRepulsion(deep_wall=True, calibration={})
        r_wall = _r_wall(new)
        for d in [r_wall, 0.8 * r_wall]:
            e_old, f_old = old.energy_forces(_tio_atoms(d))
            e_new, f_new = new.energy_forces(_tio_atoms(d))
            np.testing.assert_allclose(e_new, e_old, rtol=1e-12)
            np.testing.assert_allclose(f_new, f_old, rtol=1e-12)

    def test_forces_match_finite_difference_in_floor(self):
        pauli = PauliRepulsion(deep_wall=True, calibration={})
        d = 0.5  # deep floor region
        atoms = _tio_atoms(d)
        _, forces = pauli.energy_forces(atoms)
        pos = atoms.positions.copy()
        for a in range(2):
            for c in range(3):
                h = 1e-4
                pos_p, pos_m = pos.copy(), pos.copy()
                pos_p[a, c] += h
                pos_m[a, c] -= h
                atoms.set_positions(pos_p)
                ep, _ = pauli.energy_forces(atoms)
                atoms.set_positions(pos_m)
                em, _ = pauli.energy_forces(atoms)
                fd = -(ep - em) / (2 * h)  # force = -dU/dx
                np.testing.assert_allclose(
                    forces[a, c], fd, rtol=1e-2, atol=0.5,
                    err_msg=f"atom {a} coord {c}: analytic vs finite diff",
                )
                atoms.set_positions(pos)


class TestCalibration:
    """Item 2: per-pair A scaling (probe-driven entry-force calibration)."""

    def test_scale_is_linear_in_A(self):
        base = PauliRepulsion(deep_wall=False, calibration={})
        scaled = PauliRepulsion(deep_wall=False, calibration={pair_key(_TI, _O): 3.0})
        d = _r_wall(base)  # theta = 0.5 at the activation center
        e_b, f_b = base.energy_forces(_tio_atoms(d))
        e_s, f_s = scaled.energy_forces(_tio_atoms(d))
        np.testing.assert_allclose(e_s, 3.0 * e_b, rtol=1e-12)
        np.testing.assert_allclose(f_s, 3.0 * f_b, rtol=1e-12)

    def test_unlisted_pair_unchanged(self):
        pauli = PauliRepulsion(calibration={"13-16": 5.0})  # Al-S, absent in cell
        e, f = pauli.energy_forces(_tio_atoms(_r_wall(pauli)))
        ref = PauliRepulsion(calibration={})
        e_r, f_r = ref.energy_forces(_tio_atoms(_r_wall(ref)))
        np.testing.assert_allclose(e, e_r, rtol=1e-12)
        np.testing.assert_allclose(f, f_r, rtol=1e-12)

    def test_scale_applies_on_top_of_deep_wall(self):
        a = PauliRepulsion(deep_wall=True, calibration={pair_key(_TI, _O): 2.0})
        b = PauliRepulsion(deep_wall=True, calibration={})
        d = 0.3  # floor region: energy linear in A there too
        e_a, f_a = a.energy_forces(_tio_atoms(d))
        e_b, f_b = b.energy_forces(_tio_atoms(d))
        np.testing.assert_allclose(e_a, 2.0 * e_b, rtol=1e-12)
        np.testing.assert_allclose(f_a, 2.0 * f_b, rtol=1e-12)
