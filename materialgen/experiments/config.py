"""
Experiment configuration via dataclasses and YAML.

Functionality:
    ExperimentConfig dataclass:
        Central configuration object that specifies all parameters for an
        experiment run. Supports YAML serialization for reproducibility.

    Config structure:
        experiment:
            name: str                # e.g., "score_comparison_mp20"
            description: str         # human-readable description
            seed: int                # random seed for reproducibility
            output_dir: str          # where to save results

        data:
            dataset: str             # "mp20" | "carbon24" | "perov5" | "custom"
            data_path: str | None    # path to custom dataset
            num_test_structures: int # how many structures to evaluate

        nnp:
            backend: str             # "mace" | "chgnet" | "m3gnet" | "ensemble"
            model_path: str          # path to checkpoint
            device: str              # "cuda" | "cpu"
            dtype: str               # "float32" | "float64"
            ensemble_backends: list[str] | None  # for EnsembleNNP

        sampler:
            type: str                # "ald" | "pf_ode" | "underdamped" |
                                     # "cyclical" | "fp_optimal" | "u_adaptive"
            sigma_max: float         # maximum noise level
            sigma_min: float         # minimum noise level
            num_steps: int           # total integration steps
            num_samples: int         # number of structures to generate
            beta: float              # inverse temperature (1/kT)
            prior: str               # "gaussian" | "physically_informed"
            # Scheduler-specific params
            solver: str | None       # for PF-ODE: "dopri5" | "rk4" | "euler"
            friction: dict | None    # for Underdamped: {min, max, schedule}
            num_cycles: int | None   # for Cyclical: number of cycles

        baseline:
            enabled: bool            # whether to run baseline comparison
            models: list[str]        # ["diffcsp", "mattergen", "cdvae"]
            checkpoint_paths: dict[str, str]  # model → checkpoint path

        analysis:
            score_comparison: bool   # whether to run score field analysis
            curl_analysis: bool      # whether to compute curl violation
            lambda_experiment: bool  # whether to run λ-mixing experiment
            lambda_values: list[float]  # e.g., [0.0, 0.2, 0.5, 0.8, 1.0]

    Usage:
        config = load_config("configs/ald_mp20.yaml")
        config.validate()  # checks all required fields
        runner = ExperimentRunner(config)
        results = runner.run()

Dependencies:
    dataclasses, yaml (pyyaml), pathlib
"""
