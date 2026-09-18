"""
evaluation/plots.py
--------------------
The 3 required figures plus the bonus Experiment B diagnostic. Each
function takes already-computed data and an output path -- no computation
happens here, only plotting -- so experiments/run_experiments.py stays
readable and these can be re-used/re-styled independently.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.metrics import binned_trend


def plot_graph1_uncertainty_vs_error(pred_std_flat, actual_err_flat, corr, out_path):
    bin_x, bin_y, bin_y_std = binned_trend(pred_std_flat, actual_err_flat)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(pred_std_flat, actual_err_flat, s=3, alpha=0.05, color="steelblue", label="samples")
    ax.plot(bin_x, bin_y, "o-", color="darkorange", label="binned mean ± std")
    ax.fill_between(bin_x, bin_y - bin_y_std, bin_y + bin_y_std, color="darkorange", alpha=0.2)
    lims = [0, max(pred_std_flat.max(), actual_err_flat.max()) * 0.6]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x (ideal)")
    ax.set_xlabel("Predicted per-sensor uncertainty (std)")
    ax.set_ylabel("Actual per-sensor observation error")
    ax.set_title(f"Graph 1 — Exp. A: predicted uncertainty vs actual error\n"
                 f"Pearson r = {corr['pearson_r']:.3f}, Spearman ρ = {corr['spearman_rho']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bonus_expB(d_grid, mean_pred_std_vs_d, train_d_max, out_path):
    n_sensors = mean_pred_std_vs_d.shape[1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in range(n_sensors):
        ax.plot(d_grid, mean_pred_std_vs_d[:, s], "o-", label=f"sensor {s + 1}")
    ax.axvline(train_d_max, color="gray", linestyle=":", label="edge of training range")
    ax.set_xlabel("Degradation level d (controls injected noise scale)")
    ax.set_ylabel("Mean predicted uncertainty (std)")
    ax.set_title("Graph 1b — Exp. B: predicted uncertainty tracks injected noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_graph2_rmse_vs_degradation(d_sweep, rmse_fixed_wls, rmse_learned_fusion,
                                     rmse_unc_aware, rmse_unc_shuffled, train_d_max, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d_sweep, rmse_fixed_wls, "s-", label="Fixed WLS (baseline)")
    ax.plot(d_sweep, rmse_learned_fusion, "^-", label="Learned fusion (no uncertainty)")
    ax.plot(d_sweep, rmse_unc_shuffled, "x--", color="gray",
            label="Uncertainty-aware, SHUFFLED (Exp. C ablation)")
    ax.plot(d_sweep, rmse_unc_aware, "o-", color="crimson", linewidth=2,
            label="Uncertainty-aware (proposed)")
    ax.axvline(train_d_max, color="gray", linestyle=":", alpha=0.7, label="edge of training range")
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Estimation RMSE")
    ax.set_title("Graph 2 — Estimation error vs sensor degradation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_graph3_calibration_shift(nom_p, nom_emp, ece_id, shift_p, shift_emp, ece_shift,
                                   d_fixed, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.plot(nom_p, nom_emp, "o-", color="seagreen", label=f"in-distribution (ECE={ece_id:.3f})")
    ax.plot(shift_p, shift_emp, "o-", color="crimson", label=f"shifted regime (ECE={ece_shift:.3f})")
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage (NEES ≤ χ²(df=2) quantile)")
    ax.set_title(f"Graph 3 — Exp. D: calibration under distribution shift\n(fixed degradation d={d_fixed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
