"""
ReflectionOperator — hard geometric constraint on the covalent boundary (Layer 3).

Functionality:
    Implements the reflecting boundary condition of paper_outline_v0.4
    (Algorithm 1, Reflected Langevin Step). After a proposed Langevin step,
    any atom pair violating

        d_ij >= f_typ * (R_i^cov + R_j^cov),  f_typ = 0.75

    is mirrored back across the constraint boundary: each atom is displaced
    by half the penetration depth along the separation direction. Because
    fixing one pair may create new violations, the operator iterates until
    convergence (typically 2-5 passes) or max_passes.

    The 0.75 x covalent-sum threshold (calibration) matches the
    protocol's Type I completeness boundary: real equilibrium bonds sit at
    or below the covalent sum (Ti-O 1.97 A vs 2.13 A sum), so reflecting at
    the full sum would fire on every step of an equilibrium structure.

    The operator also enforces the OUTER (Type III) wall of the constraint
    manifold M = {x : |det L| <= f_out*|det L0|} (paper section 4.2): when a
    reference volume is supplied, a cell inflated past f_out*|det L0| is
    isotropically rescaled to the cap (f_out = 10, matching the SafetyMonitor
    Type III threshold).  The boundary therefore acts on the lattice degrees
    of freedom in both directions -- expansion against lattice collapse,
    capping against lattice explosion.

    Properties (by construction):
    - Hard guarantee: after convergence, all d_ij >= 0.75*(R_i + R_j) (up to tol).
    - Minimal intervention: only violating pairs are moved, by the minimal
      displacement restoring the constraint.
    - Zero NFE cost: purely geometric (positions + tabulated radii).

Dependencies:
    numpy, ase.neighborlist
    materialgen.core.crystal.CrystalStructure
    materialgen.utils.constants.COVALENT_RADII
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .crystal import CrystalStructure
from ..utils.constants import COVALENT_RADII


@dataclass
class ReflectionResult:
    """Outcome of a reflection pass.

    Attributes:
        crystal: Reflected structure (new CrystalStructure).
        n_violations_before: Pairs below the covalent sum before reflection.
        n_violations_after: Pairs still below after the last pass.
        passes: Number of iteration passes executed.
        converged: True if no violations remain (within tol).
        total_displacement: Sum of per-atom displacement magnitudes (Angstrom).
        lattice_scale: Cumulative isotropic cell-expansion factor applied to
            make the min-image constraint satisfiable (1.0 = no expansion;
            > 1.0 means the boundary acted on the lattice degrees of freedom,
            paper section 4.2).
        volume_capped: True if the outer (Type III) wall fired: the cell was
            inflated past f_out*reference_volume and was isotropically
            rescaled to the cap (paper section 4.2).
    """
    crystal: CrystalStructure
    n_violations_before: int
    n_violations_after: int
    passes: int
    converged: bool
    total_displacement: float
    lattice_scale: float = 1.0
    volume_capped: bool = False


class ReflectionOperator:
    """
    Mirror-reflection operator on the constraint manifold
    M = {x : d_ij(x) >= 0.75*(R_i + R_j) for all i < j} (the protocol's
    Type I completeness boundary — calibration).

    The boundary acts on the lattice degrees of freedom as well as on
    individual pairs (paper section 4.2): if the atomic pass cannot converge
    because the cell is so small that no placement of the atoms satisfies
    the min-image constraint (a lattice collapse, where d_min drops below
    the covalent sum for all pairs at once), the cell is expanded
    isotropically until the constraint manifold becomes nonempty, and the
    atomic reflection is retried.

    Args:
        radii: Covalent radii dict {Z: Angstrom}; defaults to Cordero 2008.
        max_passes: Maximum iteration passes per atomic pass (paper: 20).
        tol: Distance tolerance in Angstrom for constraint satisfaction.
        threshold_frac: Constraint threshold as a fraction of the covalent
            sum (0.75 = protocol Type I; the reflection is inert at
            equilibrium bond distances).
        max_expansions: Maximum cell expansions before giving up (each
            expansion re-runs the atomic pass).
        max_scale: Maximum allowed isotropic cell expansion factor; beyond
            this the structure is left unreflected (Layer 4 rejects it).
        outer_volume_frac: The outer (Type III) wall cap f_out: a cell with
            volume > f_out*reference_volume is isotropically rescaled to the
            cap when a reference volume is supplied to reflect() (paper
            section 4.2; 10 = SafetyMonitor Type III threshold).  The wall is
            inert without a reference volume.
    """

    def __init__(
        self,
        radii: Optional[Dict[int, float]] = None,
        max_passes: int = 20,
        tol: float = 1e-6,
        threshold_frac: float = 0.75,
        max_expansions: int = 3,
        max_scale: float = 100.0,
        outer_volume_frac: float = 10.0,
    ):
        self.radii = {int(z): float(r) for z, r in (radii or COVALENT_RADII).items()}
        self.max_passes = int(max_passes)
        self.tol = float(tol)
        self.threshold_frac = float(threshold_frac)
        self.max_expansions = int(max_expansions)
        self.max_scale = float(max_scale)
        self.outer_volume_frac = float(outer_volume_frac)
        self._max_rsum = max(self.radii.values()) * 2.0
        # ASE's neighbor list hangs on degenerate (near-zero-volume) cells;
        # a collapse trajectory could in principle approach such a cell, so
        # reflection bails out gracefully instead of spinning.
        self._vol_floor = 1e-6  # Angstrom^3

    # ---- Violation detection ----

    def count_violations(self, crystal: CrystalStructure) -> int:
        """Number of atom pairs below threshold_frac*(R_i + R_j) (min image)."""
        pairs = self._violating_pairs(crystal.ase_atoms, tol=self.tol)
        return len(pairs)

    def _violating_pairs(self, atoms, tol: float = 0.0):
        """List of (i, j, d, r_sum, unit) for pairs with d < f*(R_i+R_j) - tol,
        f = threshold_frac.  unit points from j to i (minimum image).
        """
        from ase.neighborlist import neighbor_list

        numbers = atoms.get_atomic_numbers()
        n = len(numbers)
        if n < 2:
            return []
        i_idx, j_idx, d, D = neighbor_list("ijdD", atoms, self._max_rsum)
        if len(i_idx) == 0:
            return []
        # Min-image dedupe per unordered pair: first-occurrence selection
        # with cutoff comparable to the cell size evaluated phantom long-range
        # images and missed real overlaps (same defect as PauliScore, fixed).
        keys = np.maximum(i_idx, j_idx) * n + np.minimum(i_idx, j_idx)
        order = np.lexsort((d, keys))               # per key, ascending d
        first = order[np.unique(keys[order], return_index=True)[1]]

        out = []
        for k in first:
            i, j = int(i_idx[k]), int(j_idx[k])
            r_sum = self.radii[int(numbers[i])] + self.radii[int(numbers[j])]
            dist = float(d[k])
            if dist < self.threshold_frac * r_sum - tol:
                if dist < 1e-8:
                    unit = np.array([1.0, 0.0, 0.0])  # degenerate: arbitrary axis
                else:
                    unit = -D[k] / dist  # D points i->j, so -D/d points j->i
                out.append((i, j, dist, r_sum, unit))
        return out

    # ---- Reflection ----

    def _atomic_pass(self, atoms, positions) -> tuple:
        """
        One atomic mirror-reflection sweep on ``positions`` (in place).

        Returns (converged, n_before, passes, total_disp, violating):
        ``violating`` is the last pass's violation list (needed by the
        lattice-expansion step when the sweep does not converge).
        """
        total_disp = 0.0
        passes = 0
        n_before = 0
        violating = []
        for p in range(self.max_passes):
            atoms.set_positions(positions)
            violating = self._violating_pairs(atoms, tol=self.tol)
            if p == 0:
                n_before = len(violating)
            if not violating:
                return True, n_before, passes, total_disp, violating
            passes = p + 1
            for i, j, dist, r_sum, unit in violating:
                delta = self.threshold_frac * r_sum - dist  # penetration depth
                positions[i] += 0.5 * delta * unit
                positions[j] -= 0.5 * delta * unit
                total_disp += delta
        return False, n_before, passes, total_disp, violating

    def _lattice_expansion_factor(self, atoms, violating) -> float:
        """
        Isotropic cell scale that makes the min-image constraint satisfiable.

        A pair is fixable by atomic reflection iff its threshold is below the
        covering radius of the lattice; the covering radius is at least
        l_min/2 (l_min = shortest lattice vector), so l_min >= 2*f*max_rsum
        guarantees convergence.  Per pair, the minimum-image distance scales
        linearly with the cell, giving the smaller minimal factor
        (f*r_sum)/d_current; the guaranteed backstop is taken as the maximum.
        """
        from itertools import product

        c = 1.0
        for i, j, dist, r_sum, unit in violating:
            target = self.threshold_frac * r_sum + 2.0 * self.tol
            c = max(c, target / max(dist, 1e-4))
        # Guaranteed backstop: estimate l_min over the 26 +/-1 coefficient
        # combinations of the cell rows (a subset of all integer combinations,
        # so an overestimate of the true l_min — safe, it only expands more
        # than necessary).
        L = atoms.get_cell()
        lmin = min(
            np.linalg.norm(cx[0] * L[0] + cx[1] * L[1] + cx[2] * L[2])
            for cx in product((-1, 0, 1), repeat=3)
            if cx != (0, 0, 0)
        )
        if lmin > 1e-12:
            c = max(c, 2.0 * self.threshold_frac * self._max_rsum / lmin)
        return float(c)

    def reflect(
        self,
        crystal: CrystalStructure,
        reference_volume: Optional[float] = None,
        allow_lattice_scale: bool = True,
    ) -> ReflectionResult:
        """
        Project a structure onto the constraint manifold via mirror reflection.

        Acts on the lattice degrees of freedom as well as on individual pairs
        (paper section 4.2), in both directions:
        - INNER (collapse): if the atomic sweep cannot converge because the
          cell is too small for the min-image constraint (a lattice collapse),
          the cell is expanded isotropically and the sweep is retried.
          With ``allow_lattice_scale=False`` (fixed-cell protocol) the
          expansion is skipped and non-convergence is reported for Layer 4
          to backstop; with a ``reference_volume`` the expansion is clamped
          to the reference cell — the cell recovers from collapse but never
          inflates past |det L0| (preflight hardening).
        - OUTER (Type III explosion): if ``reference_volume`` is given (the
          trajectory's initial cell volume, |det L0|) and the cell is inflated
          past outer_volume_frac*|det L0|, the cell is isotropically rescaled
          to the cap and the sweep is retried (the downscale shrinks min-image
          distances, so fresh inner violations are possible; if they cannot be
          resolved the operator reports converged=False and Layer 4 backstops).

        Returns a new CrystalStructure; the input is not modified.
        """
        atoms = crystal.ase_atoms
        positions = atoms.get_positions()

        # Degenerate-cell guard: the neighbor list hangs on near-zero-volume
        # cells, and no expansion can restore a genuinely flat lattice.  Bail
        # out; Layer 4 will reject the structure.
        if float(atoms.get_volume()) < self._vol_floor:
            return ReflectionResult(
                crystal=crystal,
                n_violations_before=len(crystal.atomic_numbers),
                n_violations_after=len(crystal.atomic_numbers),
                passes=0,
                converged=False,
                total_displacement=0.0,
                lattice_scale=1.0,
            )

        converged, n_before, passes, total_disp, violating = self._atomic_pass(atoms, positions)
        lattice_scale = 1.0
        volume_capped = False

        if not converged and allow_lattice_scale:
            # Atomic reflection saturated: the cell cannot accommodate the
            # constraint under the minimum-image convention.  Expand the
            # lattice (the collective mode of paper section 4.2) and retry.
            # Preflight hardening: the unbounded expansion was a
            # deterministic inflation ratchet — a clash event expands the cell,
            # atoms relax, the next clash expands again, up to just under the
            # 10x Type III cap, destroying E_hull while evading the OOD counter
            # (SrTiO3 l1l4: |dL|/|L0| ~ 0.9, vol x7.5, E_hull 0.47-10.4 eV;
            # alpha_lattice 1e-3 -> 1e-5 changes nothing).  Two guards:
            #   (1) allow_lattice_scale=False (fixed-cell protocol): the cell
            #       is frozen by contract, so expansion is skipped and the
            #       non-convergence is reported for Layer 4 to backstop.
            #   (2) with a reference volume, expansion is clamped to the
            #       reference cell — the cell may recover from collapse
            #       (expanding back toward |det L0|) but never inflate past it.
            for _ in range(self.max_expansions):
                if (reference_volume is not None and reference_volume > 0.0
                        and allow_lattice_scale):
                    vol_now = float(atoms.get_volume())
                    if vol_now >= reference_volume * (1.0 + 1e-9):
                        break                      # already at/past the ref
                    c = min(self._lattice_expansion_factor(atoms, violating),
                            (reference_volume / vol_now) ** (1.0 / 3.0))
                else:
                    c = self._lattice_expansion_factor(atoms, violating)
                if c <= 1.0 + 1e-9 or c > self.max_scale:
                    break
                atoms.set_cell(c * atoms.get_cell(), scale_atoms=True)
                atoms.wrap()
                positions = atoms.get_positions()  # resync: scaled + wrapped
                lattice_scale *= c
                converged, _, passes2, disp2, violating = self._atomic_pass(atoms, positions)
                total_disp += disp2
                passes += passes2
                if converged:
                    break

        # ---- Outer (Type III) wall: lattice explosion cap ----
        # The constraint manifold includes |det L| <= f_out*|det L0| (paper
        # section 4.2; f_out = 10 matches the SafetyMonitor Type III
        # threshold).  Project an inflated cell back to the cap isotropically;
        # the downscale shrinks min-image distances, so the atomic sweep is
        # retried once (it may become infeasible, in which case converged=False
        # and Layer 4 backstops).
        if (
            reference_volume is not None
            and reference_volume > 0.0
            and self.outer_volume_frac > 0.0
        ):
            vol = float(atoms.get_volume())
            cap = self.outer_volume_frac * reference_volume
            if vol > cap:
                scale = (cap / vol) ** (1.0 / 3.0)
                atoms.set_cell(scale * atoms.get_cell(), scale_atoms=True)
                atoms.wrap()
                positions = atoms.get_positions()  # resync: scaled + wrapped
                volume_capped = True
                conv2, _, passes2, disp2, _ = self._atomic_pass(atoms, positions)
                converged = conv2
                total_disp += disp2
                passes += passes2

        atoms.set_positions(positions)
        atoms.wrap()
        result_crystal = CrystalStructure(atoms, dict(crystal.properties))
        n_after = 0 if converged else len(self._violating_pairs(atoms, tol=self.tol))

        return ReflectionResult(
            crystal=result_crystal,
            n_violations_before=n_before,
            n_violations_after=n_after,
            passes=passes,
            converged=converged,
            total_displacement=total_disp,
            lattice_scale=lattice_scale,
            volume_capped=volume_capped,
        )

    def __repr__(self) -> str:
        return f"ReflectionOperator(max_passes={self.max_passes}, tol={self.tol})"
