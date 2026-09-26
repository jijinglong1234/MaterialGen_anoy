"""
Experiment management: configuration, execution, and logging.

Provides a clean interface for defining and running experiments defined in
the configs/ directory, with structured logging and result persistence.
"""

from .config import ExperimentConfig, load_config, save_config
from .runner import ExperimentRunner
from .logging import ExperimentLogger

__all__ = [
    "ExperimentConfig", "load_config", "save_config",
    "ExperimentRunner",
    "ExperimentLogger",
]
