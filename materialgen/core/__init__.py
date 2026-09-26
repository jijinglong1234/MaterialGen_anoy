"""
Core abstractions for score-based crystal generation.

This module defines the fundamental interfaces that all other modules implement:
- ScoreFunction: Computes the score (gradient of log-probability) from a crystal structure
- CrystalStructure: Primary data class representing a periodic crystal
- PauliRepulsion: Analytic short-range repulsive potential (Layer 1)
- ReflectionOperator: Hard geometric constraint on the covalent boundary (Layer 3)
- SafetyMonitor: Step-level rejection and sigma decay (Layer 4)

Scheduler / Sampler ABCs are design-stage (see scheduler.py / sampler.py docs);
concrete samplers live in materialgen.samplers.
"""

from .crystal import CrystalStructure
from .score_function import (
    ScoreFunction,
    ScoreResult,
    NNPScore,
    HybridScore,
    PauliScore,
    AugmentedScore,
    build_score_function,
)
from .pauli import PauliRepulsion, fit_exp_params, zbl_energy, zbl_force
from .reflection import ReflectionOperator, ReflectionResult
from .safety_monitor import SafetyMonitor, MonitorDecision, MonitorThresholds

__all__ = [
    "CrystalStructure",
    "ScoreFunction", "ScoreResult", "NNPScore", "HybridScore",
    "PauliScore", "AugmentedScore", "build_score_function",
    "PauliRepulsion", "fit_exp_params", "zbl_energy", "zbl_force",
    "ReflectionOperator", "ReflectionResult",
    "SafetyMonitor", "MonitorDecision", "MonitorThresholds",
]
