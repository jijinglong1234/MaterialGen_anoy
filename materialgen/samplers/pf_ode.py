"""
PFODESampler — Type B: Probability Flow ODE (deterministic sampling).

Functionality:
    Implements the PF-ODE of paper_outline_v0.4 (section 4.4.B) in the
    sigma parametrization of the VE process:

        dx/dsigma = -sigma * s(x)          (sigma decreasing sigma_max -> sigma_min)

    (from dx = sigma*dsigma/dt * beta * grad U dt and s = -beta * grad U).

    When lattice updates are enabled, the lattice degrees of freedom follow
    the same ODE with the stress-derived lattice score s_L (paper section
    4.4.B):  dL/dsigma = -sigma * s_L.  As in ALD, the update is gated to
    sigma < lattice_reopen_frac * sigma_max (plan risk #7 mitigation: the
    NNP stress is only trusted near equilibrium).

    Deterministic: no re-noising, hence the inherent robustness to NNP score
    errors analyzed in Theorem (pfode-robust). Typically run with Layer 1
    (Pauli-augmented score) only.

    Solvers:
    - "euler": fixed-step Euler in sigma (baseline, NFE ~ n_steps).
    - "ivp": scipy.solve_ivp adaptive RK45 with per-step reflection (L3)
      applied after each accepted internal step via dense output chunks.

Dependencies:
    numpy, scipy (optional, for "ivp")
    materialgen.core.crystal.CrystalStructure
    materialgen.core.score_function.ScoreFunction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.crystal import CrystalStructure, symmetrize_lattice_update
from ..core.reflection import ReflectionOperator


@dataclass
class PFODEConfig:
    """Configuration for PFODESampler (paper section 6.0.5 defaults)."""
    sigma_max: float = 0.5
    sigma_min: float = 0.01
    solver: str = "euler"           # "euler" | "ivp"
    n_steps: int = 2000             # Euler steps (NFE ~ n_steps)
    rtol: float = 1e-4              # ivp relative tolerance
    atol: float = 1e-6              # ivp absolute tolerance
    max_nfe: int = 5000             # hard cap on score evaluations
    use_reflection: bool = False    # L3 (usually unnecessary for PF-ODE)
    reflection_max_passes: int = 20
    # Lattice DOF (paper section 4.4.B): dL/dsigma = -sigma * s_L.  Fixed
    # cell by default, matching ALD.  When enabled, the update is gated to
    # sigma < lattice_reopen_frac * sigma_max (plan risk #7 mitigation: the
    # NNP stress tensor is only trusted near equilibrium, and an ungated
    # stress-driven update is precisely the mechanism that triggers lattice
    # collapse, paper section 4.2).
    update_lattice: bool = False
    lattice_reopen_frac: float = 0.1
    save_trajectory: bool = False


@dataclass
class PFODEResult:
    """Result of one PF-ODE trajectory."""
    structure: CrystalStructure
    energies: List[float] = field(default_factory=list)
    trajectory: List[CrystalStructure] = field(default_factory=list)
    nfe: int = 0
    converged: bool = True
    # Accepted-step path of the "ivp" solver (sol.t/sol.y endpoints).  The
    # per-NFE score evaluations inside RK45 include intermediate stage points,
    # which are extrapolation states of the step, NOT trajectory states —
    # recording diagnostics there (as run_phase1's per-NFE wrapper does)
    # produces spurious OOD spikes (fixed).  `path` is the honest
    # trajectory; populated only by the ivp branch.
    path: Optional[List[CrystalStructure]] = None
    # Noise level the integration actually reached.  The ivp solver
    # integrates in sigma from sigma_max to sigma_min, so a trajectory that
    # exhausts max_nfe stops at an *intermediate* sigma and its terminal
    # structure is a partially annealed one.  Without this the truncation is
    # invisible in the output: the cell reports "1000 NFE, not converged" and
    # nothing about how far down the schedule the trajectory got, which is
    # what a capped OOD rate has to be read against.  A value
    # equal to cfg.sigma_min is a completed schedule; anything larger stopped
    # early.  None means the run predates the field.
    sigma_final: Optional[float] = None


class PFODESampler:
    """
    Deterministic probability-flow sampler.

    Args:
        config: PFODEConfig.
        reflection: ReflectionOperator (created if L3 enabled and not given).
    """

    def __init__(
        self,
        config: Optional[PFODEConfig] = None,
        reflection: Optional[ReflectionOperator] = None,
    ):
        self.config = config or PFODEConfig()
        self.reflection = reflection or (
            ReflectionOperator(max_passes=self.config.reflection_max_passes)
            if self.config.use_reflection else None
        )

    # ---- Vector field ----

    def _score(self, score_fn, numbers, lattice, frac_flat: np.ndarray):
        """Full score result for flattened fractional coordinates.

        The euler branch needs the lattice score s_L as well as the
        fractional-coordinate score, so the raw result object is returned
        (one NNP evaluation serves both).
        """
        crystal = CrystalStructure.from_frac_coords(
            numbers, frac_flat.reshape(-1, 3) % 1.0, lattice,
        )
        return score_fn.compute(crystal)

    # ---- Single trajectory ----

    def sample(self, score_fn, initial: CrystalStructure) -> PFODEResult:
        """Integrate the PF-ODE from sigma_max down to sigma_min."""
        cfg = self.config
        numbers = initial.atomic_numbers
        lattice = initial.lattice
        x = initial.frac_coords.reshape(-1) % 1.0

        result = PFODEResult(structure=initial.copy())
        ref_volume = float(initial.volume)
        sigmas = np.linspace(cfg.sigma_max, cfg.sigma_min, cfg.n_steps + 1)

        if cfg.solver == "euler":
            result.sigma_final = cfg.sigma_min
            for k in range(cfg.n_steps):
                if result.nfe >= cfg.max_nfe:
                    result.converged = False
                    result.sigma_final = float(sigmas[k])
                    break
                sigma_k = sigmas[k]
                dsigma = sigmas[k + 1] - sigmas[k]  # negative
                score = self._score(score_fn, numbers, lattice, x)
                result.nfe += 1
                x = (x - sigma_k * score.frac_score.reshape(-1) * dsigma) % 1.0  # dx/dsigma = -sigma * s
                # Lattice DOF: dL/dsigma = -sigma * s_L (paper section 4.4.B),
                # gated to the final decade of the schedule (plan risk #7) —
                # dsigma < 0, so the update grows the lattice in the direction
                # of the stress-driven score.  The raw displacement is
                # projected onto the strain (6-component) subspace
                # (symmetrize_lattice_update): the stress tensor is symmetric,
                # and the rotation gauge has no physical content under the
                # fractional-coordinate dynamics (paper section 6.0.5).
                if (
                    cfg.update_lattice
                    and score.lattice_score is not None
                    and sigma_k < cfg.lattice_reopen_frac * cfg.sigma_max
                ):
                    lat_new = self._lattice_step(lattice, score.lattice_score,
                                                 -sigma_k * dsigma)
                    if lat_new is not None:
                        lattice = lat_new
                if self.reflection is not None:
                    x, lattice = self._reflect(numbers, lattice, x, ref_volume)
                if cfg.save_trajectory and (k % max(cfg.n_steps // 20, 1) == 0):
                    result.trajectory.append(
                        CrystalStructure.from_frac_coords(numbers, x.reshape(-1, 3), lattice)
                    )
            result.structure = CrystalStructure.from_frac_coords(
                numbers, x.reshape(-1, 3) % 1.0, lattice, dict(initial.properties),
            )
        elif cfg.solver == "ivp":
            # _sample_ivp sets result.structure itself; do NOT rebuild it
            # here (a fall-through rebuild used the never-updated outer x and
            # silently returned the initial structure — fixed).
            result = self._sample_ivp(score_fn, numbers, lattice, x, result)
        else:
            raise ValueError(f"Unknown solver: {cfg.solver}")

        return result

    def _reflect(
        self, numbers, lattice, x: np.ndarray, reference_volume: Optional[float] = None
    ) -> tuple:
        """L3 reflection; returns (frac_flat, lattice).

        The lattice may come back expanded: when the min-image constraint
        manifold is empty under the current cell (a lattice collapse), the
        reflecting boundary acts on the lattice degrees of freedom as well
        (paper section 4.2).  The ODE state must then follow the new cell.
        With reference_volume (the trajectory's initial cell |det L0|) the
        outer (Type III) wall also caps |det L| at f_out*|det L0|.
        """
        crystal = CrystalStructure.from_frac_coords(numbers, x.reshape(-1, 3), lattice)
        ref = self.reflection.reflect(
            crystal, reference_volume=reference_volume,
            allow_lattice_scale=self.config.update_lattice)
        return ref.crystal.frac_coords.reshape(-1), ref.crystal.lattice

    def _sample_ivp(self, score_fn, numbers, lattice, x0, result: PFODEResult) -> PFODEResult:
        """Adaptive integration via scipy RK45 with a hard NFE cap.

        With lattice updates enabled, the ODE state is the flattened
        3N+9-tuple (frac, L) (paper section 4.4.B: d = 3N+9); the lattice is
        read from the state inside the rhs closure (never the outer
        variable), and the final structure is built from the terminal state.
        """
        from scipy.integrate import solve_ivp

        cfg = self.config
        n = len(numbers)
        # |det L0| of the trajectory's initial cell (paper §6.1 OOD metric,
        # reflection outer wall): captured before integration — in the
        # lattice-update branch `lattice` evolves away from it (fixed:
        # previously read the outer sample()'s `ref_volume`,
        # which is out of scope here — every ivp+reflection run crashed
        # with NameError; now exercised by the Phase 1 mini fleet, which
        # found it, and covered by tests/test_samplers.py).
        ref_volume = float(abs(np.linalg.det(lattice)))

        if cfg.update_lattice:
            lattice0 = np.asarray(lattice)  # initial cell (degenerate fallback)
            state0 = np.concatenate([x0, lattice0.reshape(-1)])
            # sigma of the last evaluation that actually moved the state.  Once
            # the cap is hit the RHS returns zeros, so the state freezes while
            # sigma keeps decreasing to sigma_min -- sol.t[-1] would report a
            # completed schedule for a stalled run.  This is the honest
            # "noise level reached" (see PFODEResult.sigma_final).
            reached = [float(cfg.sigma_max)]

            def rhs(sigma, state):
                if result.nfe >= cfg.max_nfe:
                    return np.zeros_like(state)
                result.nfe += 1
                reached[0] = float(sigma)
                x_part = state[: 3 * n]
                lat = state[3 * n :].reshape(3, 3)
                score = score_fn.compute(
                    CrystalStructure.from_frac_coords(
                        numbers, x_part.reshape(-1, 3) % 1.0, lat,
                    )
                )
                d = -sigma * score.frac_score.reshape(-1)
                if (
                    score.lattice_score is not None
                    and sigma < cfg.lattice_reopen_frac * cfg.sigma_max
                ):
                    # symmetric-strain projection (6-component parameterization,
                    # paper section 6.0.5): removes the rotation gauge from the
                    # lattice ODE, which otherwise accumulates spurious rotation
                    d = np.concatenate([
                        d,
                        symmetrize_lattice_update(lat, -sigma * score.lattice_score).reshape(-1),
                    ])
                else:
                    d = np.concatenate([d, np.zeros(9)])
                return d

            sol = solve_ivp(
                rhs, (cfg.sigma_max, cfg.sigma_min), state0,
                method="RK45", rtol=cfg.rtol, atol=cfg.atol,
            )
            x_end = sol.y[: 3 * n, -1] % 1.0
            lattice = sol.y[3 * n :, -1].reshape(3, 3)
            if np.linalg.det(lattice) <= 0.0 or not np.all(np.isfinite(lattice)):
                # the adaptive solver stepped the 9-component state outside the
                # positive-definite domain; the strain projection makes this
                # rare, but if it happens fall back to the initial cell and
                # flag the trajectory as not converged.
                lattice = lattice0
                result.converged = False
        else:
            # Coordinate-only state (behavior identical to before the lattice
            # DOF was added).
            reached = [float(cfg.sigma_max)]   # see the lattice branch above

            def rhs(sigma, x):
                if result.nfe >= cfg.max_nfe:
                    return np.zeros_like(x)
                result.nfe += 1
                reached[0] = float(sigma)
                return -sigma * self._score(score_fn, numbers, lattice, x).frac_score.reshape(-1)

            sol = solve_ivp(
                rhs, (cfg.sigma_max, cfg.sigma_min), x0,
                method="RK45", rtol=cfg.rtol, atol=cfg.atol,
            )
            x_end = sol.y[:, -1] % 1.0

        if self.reflection is not None:
            x_end, lattice = self._reflect(numbers, lattice, x_end, ref_volume)
        # and-ed with the current flag, not assigned over it: the lattice
        # branch above clears it when the adaptive solver stepped the cell out
        # of the positive-definite domain, and a plain assignment here
        # silently re-flagged those trajectories as converged again whenever
        # solve_ivp itself reported success.
        result.converged = (result.converged and bool(sol.success)
                            and result.nfe < cfg.max_nfe)
        # The noise level the terminal structure actually belongs to: the last
        # sigma at which the state moved.  For a converged run
        # this sits within one RK45 step of sigma_min.
        result.sigma_final = reached[0]
        # Accepted-step path: sol.t/sol.y endpoints (the honest trajectory —
        # see PFODEResult.path; excludes RK45 intermediate stage points).
        n_steps = sol.y.shape[1]
        result.path = []
        for j in range(n_steps):
            xj = sol.y[: 3 * n, j] % 1.0
            latj = (sol.y[3 * n :, j].reshape(3, 3)
                    if cfg.update_lattice else np.asarray(lattice))
            result.path.append(
                CrystalStructure.from_frac_coords(numbers, xj.reshape(-1, 3), latj)
            )
        # Stash final state back through the result structure path.
        result.structure = CrystalStructure.from_frac_coords(
            numbers, x_end.reshape(-1, 3), lattice,
        )
        return result

    def __repr__(self) -> str:
        cfg = self.config
        return f"PFODESampler(solver={cfg.solver}, sigma_max={cfg.sigma_max}, L3={self.reflection is not None})"
