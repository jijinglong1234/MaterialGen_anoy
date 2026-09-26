"""
Sampler — abstract interface for running the sampling process.

NOTE: this module is currently a docstring-only placeholder —
the abstract `Sampler` class, factory, and `SampleResult` described below
are NOT implemented, and the concrete samplers (ALDSampler, PFODESampler,
UnderdampedSampler, CyclicalSampler, UAdaptiveSampler, FPOptimalSampler)
live in `materialgen/samplers/` as self-contained implementations with
their own config/result dataclasses.  The placeholder is kept deliberately
(do not delete) so the unified interface can be recovered from this
specification if the samplers are ever refactored onto a common ABC;
the docstring below is then the source of truth for that interface.

The Sampler combines a ScoreFunction and a Scheduler to generate crystal structures
from the Boltzmann distribution defined by the score. This is the main entry point
for structure generation.

Functionality:
    Abstract Base Class: Sampler
        - run(score_fn, scheduler, initial_state) -> list[CrystalStructure]
        - Generates structures by iterating the scheduler's dynamics

    Each concrete sampler implements a specific dynamical system:

    LangevinSampler:
        - Implements dx = a(t) · s(x) dt + b(t) · dW_t (Euler-Maruyama)
        - Used by: ALD, Cyclical, U-Adaptive schedulers
        - Supports Metropolis-Hastings accept/reject correction
        - Options: step_size, num_steps_per_level, save_trajectory

    ODESampler:
        - Implements dx = v(x, t) dt using adaptive ODE solvers
        - Used by: PF-ODE scheduler
        - Supports: Euler, RK4, DOPRI5(4), adaptive Heun
        - Options: rtol, atol, max_step, min_step

    UnderdampedSampler:
        - Implements the coupled (x, v) dynamics with BAOAB splitting
        - Used by: Underdamped scheduler
        - Splitting scheme: B (momentum update, dt/2) → A (position update, dt/2)
                           → O (Ornstein-Uhlenbeck, dt) → A (dt/2) → B (dt/2)
        - Options: splitting_scheme ("BAOAB" | "ABOBA" | "VV")

    FPOptimalSampler:
        - Implements dx = a(t) · s(x) dt + b(t) · dW_t with FP-optimal a(t), b(t)
        - Used by: FP-Optimal scheduler
        - May use adaptive step sizing based on local energy curvature

Public API:
    class Sampler(ABC):
        @abstractmethod
        def sample(
            self,
            score_fn: ScoreFunction,
            scheduler: Scheduler,
            num_samples: int,
            initial_prior: str | np.ndarray = "gaussian",
            seed: int | None = None,
            callback: Callable | None = None,
        ) -> list[CrystalStructure]: ...

        def sample_single(self, ...) -> CrystalStructure  # convenience for one sample

    class SampleResult:
        structures: list[CrystalStructure]  # final generated structures
        trajectory: list[list[CrystalStructure]] | None  # optional full trajectory
        stats: dict  # step count, acceptance rate, energy history, timing

    Factory function:
        build_sampler(sampler_type: str, config: dict) -> Sampler
        Supported types: "langevin", "ode", "underdamped", "fp_optimal"

Dependencies:
    numpy, torch
    materialgen.core.score_function.ScoreFunction, ScoreResult
    materialgen.core.scheduler.Scheduler, ScheduleParams
    materialgen.core.crystal.CrystalStructure
"""
