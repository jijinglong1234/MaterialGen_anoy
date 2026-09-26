"""
run_experiment.py — CLI entry point for running MaterialGen experiments.

Functionality:
    Command-line interface to run experiments defined by configuration files.
    Supports single runs and grid searches.

Usage:
    # Run a single experiment
    python scripts/run_experiment.py --config configs/ald.yaml

    # Run all 6 schedulers on MP-20
    python scripts/run_experiment.py --config configs/benchmark_all.yaml

    # Run with overrides
    python scripts/run_experiment.py --config configs/ald.yaml \\
        --override sampler.sigma_max=0.5 sampler.num_samples=100

    # Run score field comparison only (no generation)
    python scripts/run_experiment.py --config configs/default.yaml \\
        --analysis-only score_comparison

    # Dry run (validate config without executing)
    python scripts/run_experiment.py --config configs/ald.yaml --dry-run

Arguments:
    --config PATH          Path to YAML configuration file (required)
    --override KEY=VALUE   Override specific config parameters (repeatable)
    --analysis-only NAME   Run only a specific analysis (repeatable)
    --dry-run              Validate configuration without running
    --device DEVICE        Override compute device (cuda/cpu)
    --output-dir PATH      Override output directory

Dependencies:
    argparse, yaml
    materialgen.experiments.config.load_config
    materialgen.experiments.runner.ExperimentRunner
"""
