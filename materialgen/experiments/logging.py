"""
ExperimentLogger — structured logging and metric tracking.

Functionality:
    ExperimentLogger class:
        Provides structured logging for experiment runs with:
        - Console output (rich-formatted progress bars, tables)
        - File logging (detailed step-by-step log)
        - JSON metric logging (for downstream analysis)
        - WandB / TensorBoard integration (optional)

    Logged information:
    - Experiment metadata (config hash, timestamp, git commit)
    - Progress (steps completed, structures generated, timing)
    - Metrics (validity, match rate, coverage, etc.)
    - Intermediate energies and forces during sampling
    - NNP uncertainty traces (for U-Adaptive)
    - Warnings (OOD detections, rejected steps, convergence issues)

    Output formats:
    - Console: Rich progress bars with live metrics
    - File: timestamped log with level filtering
    - JSON: machine-readable metrics at each evaluation point
    - WandB: real-time cloud logging for experiment tracking
    - CSV: tabular export for Excel/R/Python analysis

    Usage:
        logger = ExperimentLogger(output_dir, use_wandb=False)
        logger.log_config(config)
        logger.start_run()
        logger.log_step(step, {"energy": energy, "sigma": sigma})
        logger.log_evaluation(metrics_dict)
        logger.finish_run()

Dependencies:
    logging, json, datetime, pathlib, rich (optional), wandb (optional)
"""
