"""
Phase 0.2 unit tests: reflecting boundary operator (Layer 3).

Covers:
    1. 100 random overlapping configurations all converge to the constraint
       manifold (d_ij >= 0.75*(R_i + R_j), the protocol Type I boundary).
    2. Minimal intervention: non-violating structures are left unchanged.
    3. Determinism: same input -> same output.
"""

import numpy as np
import pytest

from materialgen.core.crystal import CrystalStructure
from materialgen.core.reflection import ReflectionOperator
from materialgen.utils.constants import COVALENT_RADII


def _random_overlapping(n: int, seed: int) -> CrystalStructure:
    """Random structure with guaranteed short distances but a FEASIBLE
    constraint manifold (region large enough to separate n atoms)."""
    rng = np.random.RandomState(seed)
    numbers = rng.choice([6, 8, 14], size=n)
    cart = rng.uniform(0, 7.0, (n, 3))
    lattice = np.eye(3) * 12.0
    return CrystalStructure.from_cartesian(numbers, cart, lattice)


class TestConvergence:
    @pytest.mark.parametrize("seed", range(100))
    def test_random_overlap_converges(self, seed):
        op = ReflectionOperator()
        crystal = _random_overlapping(8, seed)
        result = op.reflect(crystal)
        assert result.converged, f"seed {seed}: not converged after {result.passes} passes"
        assert result.n_violations_after == 0
        # Hard guarantee: every pair satisfies the Type I boundary
        # (threshold_frac x covalent sum — calibration).
        dm = result.crystal.get_distance_matrix()
        np.fill_diagonal(dm, np.inf)
        numbers = result.crystal.atomic_numbers
        i, j = np.triu_indices(len(numbers), k=1)
        r_sums = np.array([COVALENT_RADII[numbers[a]] + COVALENT_RADII[numbers[b]]
                           for a, b in zip(i, j)])
        assert np.all(dm[i, j] >= op.threshold_frac * r_sums - 1e-6)


class TestInfeasible:
    def test_dense_infeasible_expands_lattice(self):
        """Pathologically dense config (empty constraint manifold): the
        operator must not hang; the reflecting boundary
        acts on the lattice degrees of freedom (paper section 4.2) — the
        cell is expanded until the min-image constraint manifold is
        nonempty, then the atomic sweep converges."""
        op = ReflectionOperator()
        rng = np.random.RandomState(0)
        # 20 Si in a 4 A cell: packing density proves the manifold is empty.
        crystal = CrystalStructure.from_cartesian(
            np.full(20, 14),
            rng.uniform(0, 4.0, (20, 3)),
            np.eye(3) * 4.0,
        )
        result = op.reflect(crystal)
        assert result.converged
        assert result.lattice_scale > 1.0
        assert result.n_violations_after == 0
        assert result.passes <= op.max_passes

    def test_max_scale_caps_expansion(self):
        """A low max_scale cap: the required expansion exceeds it, and the
        operator exits gracefully with converged=False (Layer 4 rejects)."""
        op = ReflectionOperator(max_scale=2.0)
        # collapsed cell (a=1.0 cubic) whose constraint needs scale ~2.28
        crystal = CrystalStructure.from_cartesian(
            np.array([14, 14]),
            np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]),
            np.eye(3) * 1.0,
        )
        result = op.reflect(crystal)
        assert result.converged is False
        assert result.lattice_scale == 1.0  # cap blocked the expansion

    def test_degenerate_cell_does_not_hang(self):
        """Near-zero-volume cell: reflection bails out without entering the
        neighbor list (which ASE cannot handle on degenerate cells)."""
        op = ReflectionOperator()
        crystal = CrystalStructure.from_cartesian(
            np.array([14, 14]),
            np.array([[0.0, 0, 0], [0.0, 0, 0]]),
            np.diag([4.0, 4.0, 1e-9]),
        )
        result = op.reflect(crystal)
        assert result.converged is False
        assert result.passes == 0


class TestOuterWall:
    """Outer (Type III) wall: |det L| <= f_out*reference_volume (paper 4.2)."""

    def test_inflated_cell_capped_to_volume_cap(self):
        """A cell inflated past f_out*V0 is isotropically rescaled to the cap."""
        op = ReflectionOperator()  # f_out = 10
        # single atom: no pairs, the atomic sweep is trivially converged, so
        # only the outer wall acts.  100 A cubic cell, V0 = 1 A^3 -> cap 10.
        crystal = CrystalStructure.from_cartesian(
            np.array([14]), np.array([[0.5, 0.5, 0.5]]), np.eye(3) * 100.0,
        )
        result = op.reflect(crystal, reference_volume=1.0)
        assert result.volume_capped
        assert result.converged
        assert abs(result.crystal.volume - 10.0) < 1e-6

    def test_outer_wall_inert_below_cap(self):
        """Volumes at or below f_out*V0 are untouched."""
        op = ReflectionOperator()
        crystal = CrystalStructure.from_frac_coords(
            np.array([6, 6]),
            np.array([[0.0, 0, 0], [0.5, 0, 0]]),
            np.eye(3) * 4.0,  # vol 64, V0 = 64 -> cap 640
        )
        result = op.reflect(crystal, reference_volume=64.0)
        assert result.volume_capped is False
        np.testing.assert_allclose(
            result.crystal.lattice, crystal.lattice, atol=1e-12,
        )

    def test_outer_wall_inert_without_reference(self):
        """No reference volume -> the wall is inert (backward compatible)."""
        op = ReflectionOperator()
        crystal = CrystalStructure.from_cartesian(
            np.array([14]), np.array([[0.5, 0.5, 0.5]]), np.eye(3) * 100.0,
        )
        result = op.reflect(crystal)
        assert result.volume_capped is False
        np.testing.assert_allclose(
            result.crystal.lattice, crystal.lattice, atol=1e-12,
        )

    def test_cap_creating_infeasible_inner_reports_failure(self):
        """Downscaling to the cap can make the inner constraint infeasible;
        the operator reports converged=False (Layer 4 backstops)."""
        op = ReflectionOperator()
        # 10 A cubic cell (vol 1000), V0 = 1 -> cap 10 -> cell 2.154 A.
        # Two C at 0.1 frac apart: 0.215 A min-image < 1.275 A threshold, and
        # the half-cell separation (1.077 A) cannot reach it -> infeasible.
        crystal = CrystalStructure.from_frac_coords(
            np.array([6, 6]),
            np.array([[0.0, 0, 0], [0.1, 0, 0]]),
            np.eye(3) * 10.0,
        )
        result = op.reflect(crystal, reference_volume=1.0)
        assert result.volume_capped
        assert result.converged is False
        assert abs(result.crystal.volume - 10.0) < 1e-6


class TestMinimalIntervention:
    def test_non_violating_structure_unchanged(self):
        op = ReflectionOperator()
        crystal = CrystalStructure.from_cartesian(
            np.array([6, 6]),
            np.array([[0.0, 0, 0], [3.0, 0, 0]]),
            np.eye(3) * 12.0,
        )
        result = op.reflect(crystal)
        assert result.n_violations_before == 0
        assert result.passes == 0
        np.testing.assert_allclose(
            result.crystal.cart_coords, crystal.cart_coords, atol=1e-12,
        )

    def test_deterministic(self):
        op = ReflectionOperator()
        crystal = _random_overlapping(8, 7)
        r1 = op.reflect(crystal)
        r2 = op.reflect(crystal)
        np.testing.assert_allclose(r1.crystal.cart_coords, r2.crystal.cart_coords, atol=1e-12)
