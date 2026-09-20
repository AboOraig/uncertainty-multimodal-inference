"""
evaluation/plots.py
--------------------
The 3 required figures, the bonus Experiment B diagnostic, and the new
sensor-identification (failure-analysis) figure. Each function takes
already-computed data and an output path -- no computation happens here,
only plotting -- so experiments/run_experiments.py stays readable.

Every metric that comes from a LEARNED model is passed as (mean, std)
across TRAIN_SEEDS and drawn as a line + shaded band via _band(). Metrics
from the closed-form, non-learned baselines (fixed/quality/oracle WLS) have
zero seed variance by construction (they don't depend on training) and are
passed as plain arrays.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.metrics import binned_trend


def _band(ax, x, mean, std, **kwargs):
    """Draw a mean line with a ±1 std shaded band (std may be all-zero for
    deterministic quantities, in which case the band is invisible)."""
    line, = ax.plot(x, mean, **kwargs)
    ax.fill_between(x, np.asarray(mean) - np.asarray(std), np.asarray(mean) + np.asarray(std),
                     color=line.get_color(), alpha=0.15)
    return line


def plot_graph1_uncertainty_vs_error(pred_std_flat, actual_err_flat, corr_mean, corr_std,
                                      seed_shown, out_path):
    bin_x, bin_y, bin_y_std = binned_trend(pred_std_flat, actual_err_flat)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.scatter(pred_std_flat, actual_err_flat, s=3, alpha=0.05, color="steelblue",
               label=f"samples (seed {seed_shown})")
    ax.plot(bin_x, bin_y, "o-", color="darkorange", label="binned mean ± std")
    ax.fill_between(bin_x, bin_y - bin_y_std, bin_y + bin_y_std, color="darkorange", alpha=0.2)
    lims = [0, max(pred_std_flat.max(), actual_err_flat.max()) * 0.6]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x (ideal)")
    ax.set_xlabel("Predicted per-sensor uncertainty (std)")
    ax.set_ylabel("Actual per-sensor observation error")
    ax.set_title(
        f"Graph 1 — Exp. A: predicted uncertainty vs actual error\n"
        f"Pearson r = {corr_mean['pearson_r']:.3f} ± {corr_std['pearson_r']:.3f}, "
        f"Spearman ρ = {corr_mean['spearman_rho']:.3f} ± {corr_std['spearman_rho']:.3f}\n"
        f"(mean ± std across seeds)", fontsize=11
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bonus_expB(d_grid, mean_pred_std_mean, mean_pred_std_std, train_d_max, out_path):
    """mean_pred_std_mean/std: (n_d, n_sensors), aggregated across seeds."""
    n_sensors = mean_pred_std_mean.shape[1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in range(n_sensors):
        _band(ax, d_grid, mean_pred_std_mean[:, s], mean_pred_std_std[:, s],
              marker="o", label=f"sensor {s + 1}")
    ax.axvline(train_d_max, color="gray", linestyle=":", label="edge of training range")
    ax.set_xlabel("Degradation level d (controls injected noise scale)")
    ax.set_ylabel("Mean predicted uncertainty (std)")
    ax.set_title("Bonus — Exp. B: predicted uncertainty tracks injected noise\n(mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_graph2_rmse_vs_degradation(
        d_sweep,
        rmse_fixed_wls, rmse_quality_wls, rmse_oracle_wls,               # deterministic baselines
        rmse_learned_fusion_mean, rmse_learned_fusion_std,               # learned, seed-aggregated
        rmse_unc_shuffled_mean, rmse_unc_shuffled_std,
        rmse_unc_aware_mean, rmse_unc_aware_std,
        train_d_max, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(d_sweep, rmse_oracle_wls, ":", color="black", linewidth=1.5,
            label="Oracle WLS (ceiling, true noise)")
    ax.plot(d_sweep, rmse_fixed_wls, "s-", color="tab:blue", label="Fixed WLS (baseline)")
    ax.plot(d_sweep, rmse_quality_wls, "D-", color="tab:green",
            label="Quality-weighted WLS (adaptive, not learned)")
    _band(ax, d_sweep, rmse_learned_fusion_mean, rmse_learned_fusion_std,
          marker="^", color="tab:orange", label="Learned fusion (no uncertainty)")
    _band(ax, d_sweep, rmse_unc_shuffled_mean, rmse_unc_shuffled_std,
          marker="x", linestyle="--", color="gray", label="Uncertainty-aware, SHUFFLED (Exp. C ablation)")
    _band(ax, d_sweep, rmse_unc_aware_mean, rmse_unc_aware_std,
          marker="o", color="crimson", linewidth=2, label="Uncertainty-aware (proposed)")

    ax.axvline(train_d_max, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Estimation RMSE")
    ax.set_title(f"Graph 2 — Estimation error vs sensor degradation\n"
                 f"(learned methods: mean ± std across {len(rmse_unc_aware_std)}-point sweep, 5 seeds)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_graph3_calibration_shift(nom_p, nom_emp_mean, nom_emp_std, ece_id_mean, ece_id_std,
                                   shift_p, shift_emp_mean, shift_emp_std, ece_shift_mean, ece_shift_std,
                                   d_fixed, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    _band(ax, nom_p, nom_emp_mean, nom_emp_std, marker="o", color="seagreen",
          label=f"in-distribution (ECE={ece_id_mean:.3f}±{ece_id_std:.3f})")
    _band(ax, shift_p, shift_emp_mean, shift_emp_std, marker="o", color="crimson",
          label=f"shifted regime (ECE={ece_shift_mean:.3f}±{ece_shift_std:.3f})")
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage (NEES ≤ χ²(df=2) quantile)")
    ax.set_title(f"Graph 3 — Exp. D: calibration under distribution shift\n"
                 f"(fixed degradation d={d_fixed}, mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sensor_identification_accuracy(d_sweep, acc_learned_mean, acc_learned_std,
                                         acc_quality_heuristic, n_sensors, train_d_max, out_path):
    """
    Failure analysis: can the uncertainty model actually point at the
    sensor that is currently degraded? acc_quality_heuristic is the naive
    "just argmax the raw quality signal" comparator (deterministic, no
    seed variance); chance level is a flat 1/n_sensors reference line.
    """
    fig, ax = plt.subplots(figsize=(7, 5.6))
    _band(ax, d_sweep, acc_learned_mean, acc_learned_std, marker="o", color="crimson",
          linewidth=2, label="Learned uncertainty argmax (proposed)")
    ax.plot(d_sweep, acc_quality_heuristic, "D-", color="tab:green",
            label="Raw quality argmax (heuristic)")
    ax.axhline(1.0 / n_sensors, color="gray", linestyle="--", label="chance level (1/3)")
    ax.axvline(train_d_max, color="gray", linestyle=":", alpha=0.5, label="edge of training range")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Sensor-identification accuracy")
    ax.set_title("Failure analysis — does the model know WHICH sensor is degraded?\n"
                 "(undefined at d=0, no real asymmetric degradation there)\n"
                 "(mean ± std across seeds)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
