"""
Visualization tools for score field analysis and scheduler comparison.

Functionality:
    Each function generates publication-quality figures for the paper.

    plot_score_field_2d(crystals, scores_nnp, scores_learned, t, save_path):
        Projects crystal structures to 2D via UMAP on structure descriptors,
        then overlays NNP (red) and learned (blue) score vectors as arrow fields.
        Shows global patterns: where do the two score fields agree/diverge?

    plot_angle_heatmap(crystals, d_angle, embedding, save_path):
        Colors UMAP embedding by angular deviation d_angle between NNP and
        learned scores. Red = high disagreement, blue = high agreement.
        Overlays: energy contours, known polymorph locations, OOD regions.

    plot_energy_angle_correlation(energies, d_angles, t_values, save_path):
        Scatter plot: x=NNP energy U(x), y=d_angle, colored by t.
        Tests hypothesis: learned score deviates most at high energies.

    plot_t_difference(metrics_by_t, save_path):
        Line plot: x=t (diffusion time), y=metrics (d_angle, relative_magnitude,
        field_divergence). Multiple lines for different chemical systems.
        Key figure: "When does the learned score deviate from physics?"

    plot_curl_distribution(curl_nnp, curl_learned, save_path):
        Histogram: x=curl Frobenius norm, comparing NNP (should be ~0) vs
        learned (non-zero). Inset: curl vs energy scatter.

    plot_streamline_comparison(x0, trajectory_nnp, trajectory_learned, save_path):
        From the same initial structure, plot the path taken by NNP score
        vs learned score. Overlay on energy landscape (UMAP). Shows whether
        both paths converge to the same minimum.

    plot_scheduler_comparison(results_by_scheduler, metric, save_path):
        Bar/box plot comparing 6 schedulers on a given metric.
        x=scheduler type, y=metric value. Color by scheduler.

    plot_lambda_mixing(lambda_values, metrics, save_path):
        x=λ (mixing ratio), y=generation quality metrics.
        Shows the "value curve" of learned score over NNP score.

    plot_trajectory_energy(trajectories, save_path):
        Energy vs step for each scheduler. Compares convergence speed.

    plot_uncertainty_trace(u_values, sigma_values, save_path):
        For U-Adaptive: dual y-axis plot showing uncertainty u(x) and
        noise scale σ(x) vs step. Shows adaptive behavior.

    All plots use consistent styling with matplotlib rcParams for the paper.
    Output formats: PDF (vector) for paper, PNG for quick preview.

Dependencies:
    numpy, matplotlib, seaborn, umap-learn, scipy
    materialgen.core.crystal.CrystalStructure
"""
