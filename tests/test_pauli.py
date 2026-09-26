"""
Phase 0.2 unit tests: Pauli repulsion potential.

Covers:
    1. Analytic forces vs central finite differences (tol 1e-5).
    2. Vanishing at equilibrium (no pairs below covalent sum -> U = 0, F = 0).
    3. Two-atom synthetic test: C-C at d = 0.8/1.0/1.2 Angstrom is repulsive,
       with energy/force increasing as d decreases; consistency with ZBL at r*.
"""

import numpy as np
import pytest
from ase import Atoms

from materialgen.core.pauli import (
    PauliRepulsion, fit_exp_params, zbl_energy, zbl_force,
)
from materialgen.utils.constants import COVALENT_RADII


def _two_carbon(d: float, cell: float = 15.0) -> Atoms:
    return Atoms(
        numbers=[6, 6],
        positions=[[0, 0, 0], [d, 0, 0]],
        cell=[cell, cell, cell],
        pbc=True,
    )


class TestTwoAtomSynthetic:
    """Ground-truth check of the Pauli wall on C-C pairs."""

    @pytest.mark.parametrize("d", [0.8, 1.0, 1.2])
    def test_force_is_repulsive_and_points_apart(self, d):
        pauli = PauliRepulsion()
        atoms = _two_carbon(d)
        energy, forces = pauli.energy_forces(atoms)
        # C-C covalent sum is 1.52 A: all three distances are inside the wall.
        assert energy > 0, f"U_Pauli should be positive at d={d}"
        # Force on atom 0 must point in -x (away from atom 1), atom 1 in +x.
        assert forces[0, 0] < 0, f"force on atom 0 should push -x at d={d}"
        assert forces[1, 0] > 0, f"force on atom 1 should push +x at d={d}"
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)

    def test_wall_steepens_as_d_shrinks(self):
        pauli = PauliRepulsion()
        energies, forces = [], []
        for d in [1.2, 1.0, 0.8]:
            e, f = pauli.energy_forces(_two_carbon(d))
            energies.append(e)
            forces.append(abs(f[0, 0]))
        assert energies[0] < energies[1] < energies[2]
        assert forces[0] < forces[1] < forces[2]

    def test_matches_zbl_at_fit_point(self):
        """At r* = 0.75*R_sum - 0.1 the exponential equals ZBL energy/force."""
        pauli = PauliRepulsion()
        r_star = 0.75 * 2 * COVALENT_RADII[6] - 0.1  # 1.04 A for C-C
        energy, forces = pauli.energy_forces(_two_carbon(r_star))
        A, B = fit_exp_params(6, 6)
        theta = 0.5 * (1.0 + np.tanh(0.1 / pauli.w))
        np.testing.assert_allclose(energy / theta, zbl_energy(6, 6, r_star), rtol=1e-10)
        # Force includes the Theta' term; check magnitude is close to ZBL force.
        assert abs(forces[0, 0]) > 0.5 * zbl_force(6, 6, r_star)
        assert abs(forces[0, 0]) < 2.0 * zbl_force(6, 6, r_star)


class TestFiniteDifference:
    """Analytic forces vs numerical gradient of U."""

    def test_forces_match_finite_difference(self):
        pauli = PauliRepulsion()
        atoms = Atoms(
            numbers=[6, 6, 8, 14],
            positions=[
                [0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.3, 1.2, 0.2], [1.0, 1.0, 1.1],
            ],
            cell=[15.0, 15.0, 15.0],
            pbc=True,
        )
        _, forces = pauli.energy_forces(atoms)
        delta = 1e-4
        pos0 = atoms.get_positions()
        num_forces = np.zeros_like(pos0)
        for i in range(len(atoms)):
            for a in range(3):
                pos = pos0.copy(); pos[i, a] += delta
                atoms.set_positions(pos); e_plus, _ = pauli.energy_forces(atoms)
                pos = pos0.copy(); pos[i, a] -= delta
                atoms.set_positions(pos); e_minus, _ = pauli.energy_forces(atoms)
                num_forces[i, a] = -(e_plus - e_minus) / (2 * delta)
        atoms.set_positions(pos0)
        np.testing.assert_allclose(forces, num_forces, atol=1e-5, rtol=1e-4)


class TestVanishing:
    def test_no_pairs_below_covalent_sum_gives_zero(self):
        pauli = PauliRepulsion()
        atoms = Atoms(
            numbers=[6, 6],
            positions=[[0, 0, 0], [3.0, 0, 0]],  # 3.0 A >> 1.52 A covalent sum
            cell=[15.0, 15.0, 15.0],
            pbc=True,
        )
        energy, forces = pauli.energy_forces(atoms)
        assert energy == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_active_pair_count(self):
        pauli = PauliRepulsion()
        assert pauli.active_pair_count(_two_carbon(1.0)) == 1
        assert pauli.active_pair_count(_two_carbon(3.0)) == 0
