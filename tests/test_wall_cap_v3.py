"""L1' cap v3 unit tests: bounded repulsive escape.

The asymmetric cap fixed the symmetric-cap trap but left the
wall repulsion to fire at up to 3*sigma_k — the L1' verification cell
measured the resulting energy pump: L1-only-ALD induced
0.57-0.58 vs bare 0.39-0.41 at sigma 1-2.  v3 additionally caps the
OPENING (repulsive) component at (r_wall - d)/2: deep pairs still climb
out of the wall (no trap) but land near the wall radius instead of being
ejected multi-Angstrom (no pump).

Covers:
    1. A 3 A repulsive drift on a 0.4 A pair is bounded to the wall radius.
    2. v3 still escapes (no trap) where the symmetric cap froze the pair.
    3. The closing component is unchanged vs. the asymmetric cap.
    4. v3 is inert outside the wall.
    5. v3 without use_wall_cap is a no-op.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from materialgen.core.crystal import CrystalStructure
from materialgen.samplers.ald import ALDConfig, ALDSampler, WALL_THRESHOLD_FRAC


class _AttractScore:
    """Constant cartesian drift: atom 0 pulled +x, atom 1 pulled -x
    (``repel=True`` flips the directions — the Pauli wall on an overlap).
    frac_score encodes ``disp / alpha`` Angstrom of cartesian displacement."""

    def __init__(self, disp: float, alpha: float, repel: bool = False):
        self.disp, self.alpha, self.repel = float(disp), float(alpha), repel

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
    lattice = np.eye(3) * cell
    return CrystalStructure.from_cartesian(
        np.array([6, 6]), np.array([[0.0, 0, 0], [pair_dist, 0, 0]]), lattice)


def _one_step(pair_dist: float, use_wall_cap: bool, disp: float = 1.0,
              cell: float = 12.0, repel: bool = False, v3: bool = False):
    cfg = ALDConfig(sigma_max=1.0, sigma_min=1.0, K=1, M=1, alpha=1e-3,
                    drift_cap=3.0, use_wall_cap=use_wall_cap,
                    use_wall_cap_v3=v3, seed=0)
    sampler = ALDSampler(cfg)
    res = sampler.sample(_AttractScore(disp, cfg.alpha, repel),
                         _pair(pair_dist, cell))
    return res.structure.get_minimum_distance()


def test_v3_bounds_escape_to_wall_radius():
    """3 A repulsion on a 0.4 A pair: the opening cap (r_wall_tail-0.4)/2
    lands the pair at the wall-force tail (r_wall + 8w = 1.54 A for C-C),
    not at 0.4 + 2*3 = 6.4 A.  The cap domain includes the smooth-step tail
    (8*w = 0.4 A) where the wall force still fires tens of eV/A."""
    r_tail = WALL_THRESHOLD_FRAC * 2 * 0.76 + 8 * 0.05  # 1.14 + 0.4 = 1.54
    d = _one_step(0.4, use_wall_cap=True, disp=3.0, repel=True, v3=True)
    assert abs(d - r_tail) < 0.25, f"escape should land near r_wall+8w, got {d:.3f}"
    assert d < 0.4 + 2.0, f"escape not bounded (energy pump): d_min={d:.3f}"


def test_v3_still_escapes_no_trap():
    """The symmetric cap froze pairs below the threshold; v3 must still let a
    0.4 A pair climb out under a 1 A repulsion."""
    d = _one_step(0.4, use_wall_cap=True, disp=1.0, repel=True, v3=True)
    assert d > 0.8, f"pair trapped below the wall: d_min={d:.3f}"


def test_v3_closing_unchanged_vs_asym():
    """v3 only touches the opening component: a purely closing drift must be
    throttled identically to the asymmetric cap."""
    d_asym = _one_step(2.0, use_wall_cap=True, disp=1.0)
    d_v3 = _one_step(2.0, use_wall_cap=True, disp=1.0, v3=True)
    assert abs(d_v3 - d_asym) < 0.1, f"{d_v3:.3f} vs {d_asym:.3f}"


def test_v3_inert_outside_wall():
    """Pair at 3.0 A (beyond r_wall 1.14): cap_open = inf, so v3 and the
    asymmetric cap give the same step."""
    d_asym = _one_step(3.0, use_wall_cap=True, disp=0.5)
    d_v3 = _one_step(3.0, use_wall_cap=True, disp=0.5, v3=True)
    assert abs(d_v3 - d_asym) < 0.1, f"{d_v3:.3f} vs {d_asym:.3f}"


def test_v3_without_wall_cap_is_noop():
    """use_wall_cap_v3 without use_wall_cap: the v3 branch lives inside the
    use_wall_cap gate, so the step is the bare (uncapped) one."""
    d_bare = _one_step(0.4, use_wall_cap=False, disp=3.0, repel=True)
    d_v3only = _one_step(0.4, use_wall_cap=False, disp=3.0, repel=True, v3=True)
    assert abs(d_v3only - d_bare) < 0.1
