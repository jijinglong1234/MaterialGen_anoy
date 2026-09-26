"""
ALDSampler — Type A: Annealed Langevin Dynamics with four-layer OOD defense.

Functionality:
    Implements the reflected annealed Langevin dynamics of paper_outline_v0.4
    (section 4.4.A):

        x <- x + alpha_k * s_aug(x) + sqrt(2*alpha_k) * sigma_k * eps

    The update takes the score *undivided*.  An earlier version of this
    docstring described a VE-consistent normalization s_used = s_aug/sigma_k^2
    that "cancels identically in the drift"; it does not cancel, and the code
    never applied it — only the comment did.  With drift b = -alpha_k*beta*F
    and diffusion D = alpha_k*sigma_k^2 the Einstein relation gives

        p_inf(x) ~ exp( integral b/D ) = exp(-beta*U(x) / sigma_k^2),

    i.e. the VE process at level k, which is the intended target: the paper's
    stationary law is the sigma_k-level noised Boltzmann distribution, not
    e^{-beta U}.  Dividing the score by sigma_k^2 would instead give
    e^{-beta*U/sigma_k^4}, four powers steeper than intended.  The paper's
    step formula was corrected to match this implementation;
    this docstring is the implementation-side record of the same fix.

    The update is in fractional coordinates, with geometric noise spacing
    sigma_k = sigma_max * (sigma_min/sigma_max)^(k/K) and per-level step size
    alpha_k = alpha_0 * (sigma_k/sigma_max) (paper section 6.0.5).

    Defense layers (all optional, default off = bare NNP):
    - L1: handled upstream by passing an AugmentedScore (NNP + Pauli).
    - L2: density-adaptive noise, eps_i scaled by 1/(1 + gamma*rho_i),
      rho_i = sum_j exp(-(r_ij/r_cut)^2) (paper section 4.3, Layer 2).
    - L3: reflecting boundary via ReflectionOperator after each step.
    - L4: SafetyMonitor rejection with phase-aware sigma decay.

    NFE accounting: one score evaluation per Langevin step; rejected steps
    still cost the evaluation that produced them.

Dependencies:
    numpy
    materialgen.core.crystal.CrystalStructure
    materialgen.core.score_function.ScoreFunction
    materialgen.core.reflection.ReflectionOperator
    materialgen.core.safety_monitor.SafetyMonitor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.crystal import CrystalStructure
from ..core.reflection import ReflectionOperator
from ..core.safety_monitor import SafetyMonitor

# Type-I completeness threshold (protocol): the Pauli wall
# activates at WALL_THRESHOLD_FRAC times the covalent-radius sum.  Kept in
# sync with PauliRepulsion.threshold_frac and the reflection operator.
WALL_THRESHOLD_FRAC = 0.75
# Smooth-step width of the Pauli activation (PauliRepulsion.w): the wall
# force tail reaches ~8*w beyond the activation radius, which is where the
# v3 opening cap must also apply (see the cap block below).
WALL_SMOOTH_W = 0.05


@dataclass
class ALDConfig:
    """Configuration for ALDSampler (paper section 6.0.5 defaults)."""
    sigma_max: float = 0.5
    sigma_min: float = 0.01
    K: int = 100                    # noise levels
    M: int = 2                      # Langevin steps per level (NFE = K*M)
    alpha: float = 1e-3             # base step size alpha_0 (re-calibrated
                                    # for the sigma-normalized
                                    # score; 0.1 was ~100x too large)
    drift_cap: float = 3.0          # per-atom drift cap in units of sigma_k
                                    # (protocol fix: raw Boltzmann
                                    # scores are unbounded — compressed bonds
                                    # give |F| ~ tens of eV/A, so alpha*beta*F
                                    # produced multi-Angstrom steps that wrap
                                    # the periodic cell and smashed atoms
                                    # together)
    # Layer-1 wall-relative drift caps (verify #3). Near a neighbor
    # the per-atom cap tightens to (d_min - 0.5)/4 so a compressed pair
    # cannot dive deep under the Pauli wall and be mirrored back with
    # multi-Angstrom overshoot (toy measured 5.5 A -> 2.1 A; no lateral
    # bypass, dwell < 0.4%).  Adopted for every L1-containing condition
    # (paper §6.2.1: enabled on every L1-containing condition of the
    # full-scale run).
    # asymmetric fix: the cap applies only to the drift component
    # closing toward the nearest neighbor. Capping the total vector trapped
    # start-OOD pairs below the wall (repulsion zeroed, 200/200-OOD stuck
    # trajectories in the mini run); the closing-only cap keeps the verify #3
    # approach-side guarantee and lets the repulsion eject pairs from below.
    use_wall_cap: bool = False
    # Layer-1 cap v3: additionally bound the OPENING (repulsive)
    # escape component at (r_wall_tail - d)/2, where r_wall_tail = r_wall +
    # 8*w is the far edge of the wall-force tail: the cap vanishes exactly
    # where the wall force does (a cap vanishing at r_wall itself would leave
    # the tail's entry-region ejection uncapped).  The asymmetric cap alone
    # lets the wall repulsion fire at up to 3*sigma_k, flinging atoms into
    # neighbors (energy pump: L1-only-ALD induced 0.57-0.58 vs bare
    # 0.39-0.41 at sigma 1-2, Layer-1 verification cell).  v3
    # keeps the escape (no trap) but limits its speed — a deep pair climbs
    # out at <= (r_wall_tail - d)/2 per atom per step and comes to rest just
    # outside the wall, where the wall force vanishes, rather than being
    # ejected multi-Angstrom.  Requires use_wall_cap.
    use_wall_cap_v3: bool = False
    alpha_lattice: float = 0.001
    update_lattice: bool = False    # fixed-cell default (paper C5/C10 practice)
    # plan risk #7 mitigation: the stress tensor of the NNP is only trusted
    # near equilibrium, so lattice updates engage exclusively at
    # sigma_k < lattice_reopen_frac * sigma_max (the lattice "reopens" in
    # the final decade of the schedule; without the gate, stress-driven
    # updates during the high-noise phase can trigger lattice collapse).
    lattice_reopen_frac: float = 0.1
    # Phase-2 lattice hardening (preflight): the stress-driven
    # update opened a second failure mode — deterministic cell INFLATION
    # (the mirror of collapse: the Pauli virial ratchets the cell up to
    # several x V0, stopping just below the Type III counter).  Two guards:
    # lattice_vol_max_ratio rejects a step that inflates the cell past
    # ratio*|det L0| (inf = off; mirror of the collapse wall at 0.5), and
    # lattice_min_dmin gates reopening on minimum pair distance so the cell
    # only adapts at true equilibrium (0.0 = off).
    lattice_vol_max_ratio: float = float("inf")
    lattice_min_dmin: float = 0.0
    # L2: density-adaptive noise
    use_density_noise: bool = False
    density_gamma: float = 0.5
    density_rcut: float = 4.0
    # L3: reflecting boundary
    use_reflection: bool = False
    reflection_max_passes: int = 20
    # bookkeeping
    seed: Optional[int] = None
    save_trajectory: bool = False


@dataclass
class ALDResult:
    """Result of one ALD trajectory."""
    structure: CrystalStructure
    energies: List[float] = field(default_factory=list)
    d_min_history: List[float] = field(default_factory=list)
    trajectory: List[CrystalStructure] = field(default_factory=list)
    nfe: int = 0
    n_rejections: int = 0
    ood_events: int = 0            # steps failing the paper OOD metric (§6.1):
                                   # d_min < 0.5 A, max|F_NNP| > 500 eV/A, or
                                   # |det L|/|det L0| > 10 (Type III)
    converged: bool = True
    # Drift-cap diagnostics (step-size control).  The per-atom
    # drift cap is a numerical stabiliser, not part of the SDE, and a finer
    # integrator needs it less: cap_steps counts steps on which at least one
    # atom was clipped, cap_atoms the total clipped atom-steps.  They are the
    # evidence that the "fine" control arm removes step-size error and cap
    # clipping together, which is what its interpretation requires.  Purely
    # additive: no arithmetic below depends on them.
    cap_steps: int = 0
    cap_atoms: int = 0


def geometric_schedule(sigma_max: float, sigma_min: float, K: int) -> np.ndarray:
    """sigma_k = sigma_max * (sigma_min/sigma_max)^(k/K), k = 0..K (K+1 values)."""
    return sigma_max * (sigma_min / sigma_max) ** (np.arange(K + 1) / K)


def local_density(crystal: CrystalStructure, r_cut: float = 4.0) -> np.ndarray:
    """
    Local atomic density rho_i = sum_{j != i} exp(-(r_ij/r_cut)^2).

    Neighbor sum truncated at 1.5*r_cut (Gaussian tail exp(-2.25) ~ 0.1).
    """
    from ase.neighborlist import neighbor_list

    n = crystal.num_atoms
    rho = np.zeros(n)
    if n < 2:
        return rho
    atoms = crystal.ase_atoms
    i_idx, j_idx, d = neighbor_list("ijd", atoms, 1.5 * r_cut)
    contrib = np.exp(-((d / r_cut) ** 2))
    np.add.at(rho, i_idx, contrib)
    np.add.at(rho, j_idx, contrib)
    return rho


def per_atom_dmin(crystal: "CrystalStructure", return_dirs: bool = False,
                  return_rwall: bool = False):
    """Per-atom minimum-image distance to the nearest neighbor (Angstrom).

    With return_dirs, also returns the unit vector (cartesian) from each
    atom to its nearest neighbor.  With return_rwall (requires return_dirs),
    additionally returns the Type-I wall radius of each atom's nearest pair,
    WALL_THRESHOLD_FRAC*(R_i + R_nn) — the Pauli activation threshold; cap
    v3 uses it to bound the repulsive escape speed.
    O(n^2) over the primitive cell (n <= ~100); used only when Layer 1's
    wall-relative caps (use_wall_cap) are enabled, so the cost is not on
    the hot path.
    """
    fc = crystal.frac_coords
    d = fc[:, None, :] - fc[None, :, :]
    d -= np.round(d)
    disp = d @ crystal.lattice
    dist = np.linalg.norm(disp, axis=2)
    np.fill_diagonal(dist, np.inf)
    if return_dirs:
        nn = dist.argmin(axis=1)
        dmin = dist[np.arange(len(fc)), nn]
        # disp[i, j] = r_i - r_j points from the neighbor to atom i; the cap
        # needs the closing direction (atom i -> neighbor), so negate.
        dirs = -disp[np.arange(len(fc)), nn] / dmin[:, None]
        if return_rwall:
            from ..utils.constants import COVALENT_RADII

            numbers = crystal.atomic_numbers
            radii = np.array([COVALENT_RADII.get(int(z), 0.0) for z in numbers])
            rwall = WALL_THRESHOLD_FRAC * (radii + radii[nn])
            return dmin, dirs, rwall
        return dmin, dirs
    return dist.min(axis=1)


class ALDSampler:
    """
    Annealed Langevin dynamics sampler with optional four-layer defense.

    Args:
        config: ALDConfig.
        reflection: ReflectionOperator (created if L3 enabled and not given).
        safety_monitor: SafetyMonitor (created if L4 used; pass explicitly
            to enable Layer 4). None disables Layer 4.
    """

    def __init__(
        self,
        config: Optional[ALDConfig] = None,
        reflection: Optional[ReflectionOperator] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
    ):
        self.config = config or ALDConfig()
        self.reflection = reflection or (
            ReflectionOperator(max_passes=self.config.reflection_max_passes)
            if self.config.use_reflection else None
        )
        self.safety_monitor = safety_monitor

    # ---- Single trajectory ----

    def sample(self, score_fn, initial: CrystalStructure) -> ALDResult:
        """Run one annealed Langevin trajectory from ``initial``."""
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)
        crystal = initial.copy()
        sigmas = geometric_schedule(cfg.sigma_max, cfg.sigma_min, cfg.K)
        # Reference volume = the trajectory's initial cell |det L0|: used by
        # the Layer-4 volume-ratio check and by the reflection operator's
        # outer (Type III) wall (|det L| <= f_out*|det L0|, paper section 4.2).
        ref_volume = float(crystal.volume)
        if self.safety_monitor is not None and self.safety_monitor.reference_volume is None:
            self.safety_monitor.reference_volume = ref_volume

        result = ALDResult(structure=crystal)
        sigma_current = cfg.sigma_max

        for k in range(cfg.K):
            sigma_k = min(sigmas[k], sigma_current)
            alpha_k = cfg.alpha * (sigma_k / cfg.sigma_max)

            for _ in range(cfg.M):
                score = score_fn.compute(crystal)
                result.nfe += 1

                # --- Drift in fractional coordinates ---
                # The score enters undivided (see the module docstring, and the
                # correction): drift alpha_k*s_aug against diffusion
                # alpha_k*sigma_k^2 gives the level-k VE stationary law
                # e^{-beta U/sigma_k^2}, which is the intended target.  The
                # "sigma_k^2 prefactor" that an earlier comment described
                # cancelling was never applied by this code, and applying it
                # would give e^{-beta U/sigma_k^4}.  The step size is held to
                # the noise scale instead by the calibrated alpha_0 and by the
                # drift cap below.
                drift = alpha_k * score.frac_score

                # --- Per-atom drift cap (protocol fix) ---
                # alpha_k*beta*F is unbounded: compressed bonds give |F| of
                # tens-hundreds of eV/A, hence multi-Angstrom steps that wrap
                # the periodic cell and collide atoms (measured: l1 sigma=0.1,
                # 49 eV/A -> 1.9 A step -> d_min 1.47 -> 0.665 in one step).
                # Cap each atom's cartesian drift displacement at
                # drift_cap*sigma_k so the drift stays noise-relative.
                if cfg.drift_cap > 0:
                    cap = cfg.drift_cap * sigma_k
                    drift_cart = drift @ crystal.lattice
                    norms = np.linalg.norm(drift_cart, axis=1)
                    scale = np.minimum(1.0, cap / np.maximum(norms, 1e-12))
                    n_capped = int(np.count_nonzero(scale < 1.0))
                    if n_capped:
                        result.cap_steps += 1
                        result.cap_atoms += n_capped
                    drift_cart = drift_cart * scale[:, None]
                    if cfg.use_wall_cap and crystal.num_atoms > 1:
                        # Layer-1 caps: only the component CLOSING toward
                        # the nearest neighbor is throttled (asymmetric cap
                        # fix).
                        # The symmetric version capped the total vector at
                        # (d_min-0.5)/4, which for pairs already below the
                        # 0.5 A OOD threshold (start-OOD, 8-30% of sigma=1
                        # starts) zeroed the repulsive drift too, trapping
                        # the pair in the collision state for the whole
                        # trajectory (mini-run: 200/200-OOD stuck candidates
                        # in l1 conditions). Capping only the closing
                        # component preserves the verify #3 approach-side
                        # guarantee (a purely closing drift is unchanged)
                        # while the Pauli repulsion keeps firing at the 3*sigma
                        # cap below the wall, ejecting start-OOD pairs.
                        rw = per_atom_dmin(crystal, True, cfg.use_wall_cap_v3)
                        dmin, dirs = rw[0], rw[1]
                        wall_cap = np.maximum(0.0, (dmin - 0.5) / 4.0)
                        closing = np.einsum("ij,ij->i", drift_cart, dirs)
                        excess = closing - wall_cap
                        drift_cart -= np.where(excess > 0, excess, 0.0)[:, None] * dirs
                        if cfg.use_wall_cap_v3:
                            # v3: bound the OPENING (repulsive
                            # escape) component at (r_wall + 8*w - d)/2, the
                            # cap vanishing at the tail's far edge where the
                            # wall force itself does.  Applied to the drift
                            # left after the closing cap; a deep pair still
                            # climbs out (no trap) but lands just outside the
                            # wall instead of being ejected multi-Angstrom
                            # (no energy pump into neighbors).  The cap
                            # domain extends to
                            # r_wall + 8*w: the smooth-step gradient term
                            # (sech^2/(2w), w=0.05) keeps the wall force at
                            # tens of eV/A up to ~0.4 A OUTSIDE the nominal
                            # activation radius — the toy showed
                            # that capping only inside r_wall leaves the
                            # entry-region ejection untouched (3.7-4.7 A
                            # single-step jumps from d ~ 1.15-1.22 A).
                            rwall = rw[2] + 8.0 * WALL_SMOOTH_W
                            away = -np.einsum("ij,ij->i", drift_cart, dirs)
                            cap_open = np.where(
                                dmin < rwall,
                                np.maximum((rwall - dmin) / 2.0, 0.0), np.inf)
                            excess_open = away - cap_open
                            drift_cart += np.where(excess_open > 0, excess_open,
                                                   0.0)[:, None] * dirs
                    drift = drift_cart @ np.linalg.inv(crystal.lattice)

                # --- Noise (L2: density-adaptive) ---
                # eps is a cartesian displacement; convert to fractional
                # (eps_frac = eps_cart @ inv(L)) so a frac step maps back to
                # the intended cartesian move. The raw (unit-covariance)
                # eps made the noise |L| ~ 4x too large on the 3.9 A cell.
                eps = rng.normal(0.0, 1.0, (crystal.num_atoms, 3))
                if cfg.use_density_noise:
                    rho = local_density(crystal, cfg.density_rcut)
                    eps = eps / (1.0 + cfg.density_gamma * rho)[:, None]
                eps = np.linalg.solve(crystal.lattice.T, eps.T).T
                noise = np.sqrt(2.0 * alpha_k) * sigma_k * eps

                prev = crystal
                new_frac = crystal.frac_coords + drift + noise
                crystal = CrystalStructure.from_frac_coords(
                    crystal.atomic_numbers, new_frac % 1.0, crystal.lattice,
                    dict(crystal.properties),
                )

                # --- Optional lattice update ---
                # Stress-driven, gated to sigma_k < lattice_reopen_frac*sigma_max
                # (plan risk #7): the NNP stress is unreliable on the highly
                # perturbed structures of the high-noise phase, and an
                # ungated update is precisely the mechanism that triggers
                # lattice collapse (paper section 4.2).
                if (
                    cfg.update_lattice
                    and score.lattice_score is not None
                    and sigma_k < cfg.lattice_reopen_frac * cfg.sigma_max
                ):
                    # Phase-2 hardening (preflight): equilibrium
                    # gate + inflation wall, see ALDConfig.lattice_min_dmin /
                    # lattice_vol_max_ratio.
                    if cfg.lattice_min_dmin > 0.0 and \
                            float(per_atom_dmin(crystal).min()) < cfg.lattice_min_dmin:
                        new_lat = None
                    else:
                        new_lat = self._lattice_step(
                            crystal.lattice, score.lattice_score,
                            cfg.alpha_lattice * sigma_k**2,
                        )
                        if (new_lat is not None
                                and cfg.lattice_vol_max_ratio < float("inf")
                                and np.linalg.det(new_lat) / ref_volume
                                > cfg.lattice_vol_max_ratio):
                            new_lat = None
                    if new_lat is not None:
                        crystal.lattice = new_lat

                # --- L3: reflecting boundary ---
                # reference_volume enables the outer (Type III) wall: a cell
                # inflated past f_out*|det L0| is rescaled back to the cap
                # (paper section 4.2).
                if self.reflection is not None:
                    # allow_lattice_scale=False under the fixed-cell protocol
                    # (update_lattice=False): the frozen cell must not be
                    # expanded by the inner collapse-recovery loop (preflight
                    # found it a deterministic inflation ratchet).
                    ref = self.reflection.reflect(
                        crystal, reference_volume=ref_volume,
                        allow_lattice_scale=cfg.update_lattice)
                    crystal = ref.crystal

                # --- L4: safety monitor ---
                if self.safety_monitor is not None:
                    # judge the NNP-only force: the Pauli wall legitimately
                    # exceeds any finite |F| threshold at short range and must
                    # not trigger rejection (fix)
                    md = getattr(score, "metadata", None) or {}
                    f_monitor = md.get("nnp_forces", score.forces)
                    decision = self.safety_monitor.check(
                        crystal, energy=score.energy, forces=f_monitor,
                    )
                    if not decision.accepted:
                        crystal = prev  # backtrack
                        result.n_rejections += 1
                        decay = self.safety_monitor.sigma_decay_factor(sigma_k, cfg.sigma_max)
                        # floor at sigma_min: a rejection cascade must not
                        # freeze the trajectory at zero effective noise
                        sigma_current = max(sigma_k * decay, cfg.sigma_min)
                        continue
                    sigma_current = sigma_k

                # --- Diagnostics ---
                d_min = crystal.get_minimum_distance() if crystal.num_atoms >= 2 else np.inf
                result.d_min_history.append(d_min)
                # OOD event per the paper metric (§6.1): d_min < 0.5 Å,
                # max NNP-only force > 500 eV/Å, or |det L|/|det L0| > 10 (Type III).
                # The force criterion judges the NNP-only force: the Pauli wall
                # legitimately exceeds 500 eV/Å at short range (same convention
                # as the L4 check above).
                md = getattr(score, "metadata", None) or {}
                f_ood = md.get("nnp_forces", getattr(score, "forces", None))
                max_f_ood = float(np.linalg.norm(f_ood, axis=1).max()) \
                    if f_ood is not None and len(f_ood) else None
                v_ratio = crystal.volume / ref_volume if ref_volume > 0 else None
                if (d_min < 0.5
                        or (max_f_ood is not None and max_f_ood > 500.0)
                        or (v_ratio is not None and v_ratio > 10.0)):
                    result.ood_events += 1
                if score.energy is not None:
                    result.energies.append(score.energy)
                if cfg.save_trajectory:
                    result.trajectory.append(crystal.copy())

        result.structure = crystal
        return result

    # ---- Lattice strain parameterization ----

    @staticmethod
    def _lattice_step(L: np.ndarray, s_L: np.ndarray, coeff: float) -> Optional[np.ndarray]:
        """
        Symmetric-strain lattice update (paper section 6.0.5).

        The raw displacement coeff*s_L is projected onto the strain
        (6-component) subspace via the shared velocity-gradient
        symmetrization (crystal.symmetrize_lattice_update), removing the
        spurious rotation gauge; see that function for the rationale.  If
        the projected cell is degenerate or non-finite the update is skipped
        and the previous cell retained (the gate only opens near
        equilibrium, where such failure indicates NNP stress breakdown
        rather than physics).

        Args:
            L: current lattice (rows are lattice vectors).
            s_L: lattice score (3, 3).
            coeff: scalar step coefficient (alpha_lattice*sigma^2 for ALD).
        """
        from ..core.crystal import symmetrize_lattice_update

        L_new = L + symmetrize_lattice_update(L, coeff * s_L)
        if np.linalg.det(L_new) <= 0.0 or not np.all(np.isfinite(L_new)):
            return None
        return L_new

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"ALDSampler(sigma_max={cfg.sigma_max}, K={cfg.K}, M={cfg.M}, "
            f"L2={cfg.use_density_noise}, L3={self.reflection is not None}, "
            f"L4={self.safety_monitor is not None})"
        )
