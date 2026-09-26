"""
Sampler implementations.

Main line (paper v0.4):
    Type A: ALDSampler — Annealed Langevin Dynamics (four-layer defense ready)
    Type B: PFODESampler — Probability Flow ODE (deterministic, recommended)

Secondary paradigms (Underdamped, Cyclical, U-Adaptive) are appendix
material; FP-Optimal was dropped. Their module docstrings retain the
design specs.
"""

from .ald import ALDSampler, ALDConfig, ALDResult, geometric_schedule, local_density
from .pf_ode import PFODESampler, PFODEConfig, PFODEResult

__all__ = [
    "ALDSampler", "ALDConfig", "ALDResult", "geometric_schedule", "local_density",
    "PFODESampler", "PFODEConfig", "PFODEResult",
]
