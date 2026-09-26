"""
Evaluation metrics for generated crystal structures (paper section 6.0.7).

The chain, in the order a benchmark applies it:

    validity    compute_validity                 overlaps, coordination, charge
    stability   HullEvaluator / compute_sun      E_hull^NNP < 100 meV/atom
    novelty     TrainingIndex / compute_novelty  not in the training split
    diversity   compute_diversity                AMSD, space groups, R-Angle
    match       compute_match_rate/coverage      against the reference set

S.U.N. is assembled by ``compute_sun`` over ``StructureRecord``s (stable AND
unique AND novel), or end-to-end by ``StructureEvaluator``, which owns the
dataset's hull, training index and matcher.

Hull tables are frozen per dataset (scripts/build_hulls.py) and
loaded through ``materialgen.data.registry.DATASETS``; ``E_HULL_UNITS`` is
eV/atom throughout.
"""

from .metrics import (
    MATCH_PRESETS,
    EvaluationReport,
    StructureEvaluator,
    StructureRecord,
    angle_histogram,
    bond_angles,
    compute_amsd,
    compute_amsd_matrix,
    compute_coverage,
    compute_match_and_coverage,
    compute_match_rate,
    compute_r_angle,
    compute_sun,
    compute_symmetry_score,
    compute_validity,
    detect_spacegroup,
    make_matcher,
    mark_unique,
)
from .stability import (
    E_HULL_STABLE,
    E_HULL_UNITS,
    HullEvaluator,
    HullTable,
    build_phase_diagram,
    canonical_formula,
    compute_energy_above_hull,
    compute_formation_energy,
    hull_summary,
    load_hull,
    system_key,
)
from .diversity import (
    TrainingIndex,
    compute_diversity,
    compute_mode_collapse_metric,
    compute_novelty,
)
from .relaxation import (
    RelaxationConfig,
    batch_relax,
    compute_relaxation_trajectory,
    relax_structure,
)

__all__ = [
    # metrics
    "MATCH_PRESETS", "EvaluationReport", "StructureEvaluator", "StructureRecord",
    "angle_histogram", "bond_angles", "compute_amsd", "compute_amsd_matrix",
    "compute_coverage", "compute_match_and_coverage", "compute_match_rate",
    "compute_r_angle", "compute_sun", "compute_symmetry_score",
    "compute_validity", "detect_spacegroup", "make_matcher", "mark_unique",
    # stability
    "E_HULL_STABLE", "E_HULL_UNITS", "HullEvaluator", "HullTable",
    "build_phase_diagram", "canonical_formula", "compute_energy_above_hull",
    "compute_formation_energy", "hull_summary", "load_hull", "system_key",
    # diversity
    "TrainingIndex", "compute_diversity", "compute_mode_collapse_metric",
    "compute_novelty",
    # relaxation
    "RelaxationConfig", "batch_relax", "compute_relaxation_trajectory",
    "relax_structure",
]
