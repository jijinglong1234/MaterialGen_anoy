"""
Score field comparison and analysis tools.

This module provides the analytical backbone for comparing NNP-derived scores
with learned scores from diffusion models (DiffCSP, MatterGen).

NOTE: the score-comparison modules (score_compare, curl,
visualization, lambda_mixing) are docstring-only placeholders — the classes
and functions they describe are NOT implemented yet.  They are kept
deliberately (do not delete) as the interface specification for the future
§6.0.7 score-field analysis (Curve violation, d_angle, λ-mixing), the same
policy as core/sampler.py and core/scheduler.py.  The package must remain
importable meanwhile, so those imports are guarded below; only the real
modules (metrics) are imported unconditionally.
"""

from .metrics import D_MIN_VALID, is_valid, validity_reasons

try:  # placeholder: ScoreComparator/compare_scores not implemented yet
    from .score_compare import ScoreComparator, compare_scores
except ImportError:
    ScoreComparator = None  # type: ignore[assignment]
    compare_scores = None   # type: ignore[assignment]

try:  # placeholder: curl analysis not implemented yet
    from .curl import compute_curl_violation, curl_field_analysis
except ImportError:
    compute_curl_violation = None  # type: ignore[assignment]
    curl_field_analysis = None     # type: ignore[assignment]

try:  # placeholder: plotting not implemented yet
    from .visualization import (
        plot_score_field_2d,
        plot_angle_heatmap,
        plot_energy_angle_correlation,
        plot_t_difference,
        plot_curl_distribution,
        plot_streamline_comparison,
    )
except ImportError:
    plot_score_field_2d = None
    plot_angle_heatmap = None
    plot_energy_angle_correlation = None
    plot_t_difference = None
    plot_curl_distribution = None
    plot_streamline_comparison = None

try:  # placeholder: λ-mixing experiment not implemented yet
    from .lambda_mixing import LambdaMixingExperiment
except ImportError:
    LambdaMixingExperiment = None  # type: ignore[assignment]

__all__ = [
    "D_MIN_VALID", "is_valid", "validity_reasons",
    "ScoreComparator", "compare_scores",
    "compute_curl_violation", "curl_field_analysis",
    "plot_score_field_2d", "plot_angle_heatmap",
    "plot_energy_angle_correlation", "plot_t_difference",
    "plot_curl_distribution", "plot_streamline_comparison",
    "LambdaMixingExperiment",
]
