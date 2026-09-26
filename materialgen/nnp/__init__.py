"""
NNP (Neural Network Potential) wrappers for score computation.

Currently implemented:
    ESENNNP — eSEN via fairchem-core (local checkpoint in OMAT24/)

MACE / CHGNet / M3GNet / ensemble wrappers are design-stage (see the
respective module docstrings). Any ASE Calculator can be used directly
with NNPScore in the meantime.
"""

try:
    from .esen import ESENNNP, load_esen_calculator
    __all__ = ["ESENNNP", "load_esen_calculator"]
except ImportError:  # fairchem-core not installed
    ESENNNP = None
    load_esen_calculator = None
    __all__ = []
