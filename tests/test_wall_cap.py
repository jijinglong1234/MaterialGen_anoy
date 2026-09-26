"""
Phase 0.2 unit tests: L1' wall-relative drift cap (verify #3).

The Pauli wall (L1) mirrors compressed pairs back; without a cap, a strong
attractive drift can dive deep under the wall in one step, and the mirrored
displacement overshoots to ~5.5 A (toy measurement; real cells are ~5 A,
i.e. the atom lands inside a neighbor).  L1' tightens the per-atom drift cap
near a neighbor to (d_min - 0.5)/4 (-> 0 at the 0.5 A OOD threshold), limiting
the penetration so the mirror step cannot overshoot.

Covers:
    1. per_atom_dmin: per-atom minimum-image nearest-neighbor distance.
    2. Without the cap, a strong attractive drift closes a 2.0 A pair to
       below the 0.5 A OOD threshold in one step (deep penetration).
    3. With the cap, the same drift is throttled to (d-0.5)/4 and the pair
       stays well above the threshold (no deep penetration).
    4. Wall cap is inert when the pair is far (no behaviour change on
       normal steps).
    5. Asymmetric cap: a repulsive drift on a pair already
       below the threshold is NOT zeroed -- the pair escapes the wall
       (the symmetric cap trapped start-OOD pairs for whole trajectories).
"""

import numpy as np
import pytest
from types import SimpleNamespace

from materialgen.core.crystal import CrystalStructure
from materialgen.samplers.ald import ALDConfig, ALDSampler, per_atom_dmin


class _AttractScore:
    """Constant attractive drift: atom 0 pulled +x, atom 1 pulled -x.

    With ``repel=True`` the directions flip (atom 0 -x, atom 1 +x): a
    repulsive drift pushing the pair apart, as the Pauli wall does for an
    overlapping pair. The ALD step is drift = alpha_k * frac_score, so
    frac_score must encode ``disp / alpha_k`` Angstrom of cartesian
    displacement to give an uncapped drift of exactly ``disp`` Angstrom
    per atom.
    """

    def __init__(self, disp: float, alpha: float, repel: bool = False):
        self.disp = float(disp)
        self.alpha = float(alpha)
        self.repel = repel

    def compute(self, crystal):
        n = crystal.num_atoms
        s = -1.0 if self.repel else 1.0
        F_cart = np.zeros((n, 3))
        F_cart[0, 0] = s * self.disp / self.alpha
        F_cart[1, 0] = -s * self.disp / self.alpha
        frac = np.linalg.solve(crystal.lattice.T, F_cart.T).T
        return SimpleNamespace(frac_score=frac, energy=None, forces=None,
                               lattice_score=None)


def _pair(pair_dist: float, cell: float = 12.0) -> CrystalStructure:
    lattice = np.eye(3) * cell  # cell must be > pair_dist: min-image = pair
    return CrystalStructure.from_cartesian(
        np.array([6, 6]), np.array([[0.0, 0, 0], [pair_dist, 0, 0]]), lattice)


def _one_step(pair_dist: float, use_wall_cap: bool, disp: float = 1.0,
              cell: float = 12.0, repel: bool = False):
    cfg = ALDConfig(sigma_max=1.0, sigma_min=1.0, K=1, M=1, alpha=1e-3,
                    drift_cap=3.0, use_wall_cap=use_wall_cap, seed=0)
    sampler = ALDSampler(cfg)
    res = sampler.sample(_AttractScore(disp, cfg.alpha, repel),
                         _pair(pair_dist, cell))
    return res.structure.get_minimum_distance()


def test_per_atom_dmin():
    crys = CrystalStructure.from_frac_coords(
        np.array([6, 6]), np.array([[0.0, 0, 0], [0.5, 0, 0]]),
        np.eye(3) * 4.0,
    )
    np.testing.assert_allclose(per_atom_dmin(crys), [2.0, 2.0], atol=1e-9)


def test_uncapped_drift_penetrates_below_ood_threshold():
    """A 1 A attractive drift on a 2.0 A pair: without L1' the pair closes to
    d < 0.5 A in one step (the deep-penetration failure L1' fixes)."""
    d = _one_step(2.0, use_wall_cap=False)
    assert d < 0.5, f"expected deep penetration, got d_min={d:.3f}"


def test_wall_cap_throttles_drift_and_blocks_penetration():
    """Same drift with L1': the cap (2.0-0.5)/4 = 0.375 A throttles the step,
    so the pair stays well above the OOD threshold."""
    d = _one_step(2.0, use_wall_cap=True)
    assert d >= 0.5, f"L1' failed to block penetration, d_min={d:.3f}"
    assert d > 1.0, f"expected strong throttling, d_min={d:.3f}"


def test_wall_cap_inert_for_distant_pairs():
    """Pair at 14 A in a 40 A cell: (14-0.5)/4 = 3.375 A >= the 3*sigma_k cap
    (3.0 A), so the wall cap never binds and on/off trajectories coincide."""
    d_on = _one_step(14.0, use_wall_cap=True, cell=40.0)
    d_off = _one_step(14.0, use_wall_cap=False, cell=40.0)
    assert abs(d_on - d_off) < 0.2, f"cap should be inert far from wall: {d_on} vs {d_off}"


def test_wall_cap_does_not_trap_pair_below_threshold():
    """mini-run finding: the symmetric cap zeroed ALL drift for
    pairs already below the 0.5 A OOD threshold (start-OOD), trapping them in
    the collision state for 200/200 OOD steps (Pauli repulsion disabled too).
    With the asymmetric cap the repulsive drift is untouched: a 0.4 A pair
    pushed apart by a 1 A repulsive drift must leave the wall region."""
    d = _one_step(0.4, use_wall_cap=True, disp=1.0, repel=True)
    assert d > 1.0, f"pair trapped below the wall: d_min={d:.3f}"
