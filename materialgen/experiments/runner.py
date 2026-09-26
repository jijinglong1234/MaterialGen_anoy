"""
ExperimentRunner — executes configured experiments.

Functionality:
    ExperimentRunner class:
        Takes an ExperimentConfig, sets up all components, runs the experiment,
        and saves results in a structured output directory.

    Setup phase:
        1. Load dataset (from config.data)
        2. Initialize NNP backend(s) (from config.nnp)
        3. Create NNP score function: s(x) = -β∇U(x)
        4. [Optional] Load baseline learned score models
        5. Build sampler from config: scheduler + sampling algorithm
        6. Initialize evaluator with reference dataset

    Execution phase:
        1. Generate structures via sampler.sample()
        2. [Optional] Relax generated structures (NNP or DFT)
        3. Evaluate: compute all metrics via StructureEvaluator
        4. [Optional] Run score field comparison
        5. [Optional] Run curl violation analysis
        6. [Optional] Run λ-mixing experiment

    Output phase:
        output_dir/
        ├── config.yaml                  # saved config for reproducibility
        ├── generated_structures/        # CIF files of generated structures
        ├── metrics.json                 # summary metrics
        ├── metrics_detailed.json        # per-structure metrics
        ├── trajectories/                # [optional] sampling trajectories
        ├── score_comparison/            # [optional] score field analysis
        │   ├── d_angle_vs_t.pdf
        │   ├── d_angle_vs_energy.pdf
        │   ├── curl_distribution.pdf
        │   └── comparison_data.npz
        ├── lambda_experiment/           # [optional] λ-mixing results
        └── log.txt                      # detailed run log

    Parallel execution:
        Supports running multiple scheduler configs in parallel.
        Uses multiprocessing for independent runs (one per GPU if available).

Dependencies:
    numpy, torch, yaml, pathlib, multiprocessing, datetime
    materialgen.core.*
    materialgen.nnp.*
    materialgen.samplers.*
    materialgen.data.*
    materialgen.eval.*
    materialgen.analysis.*
"""
