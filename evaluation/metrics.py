"""
evaluation/metrics.py
----------------------
Reusable evaluation utilities shared by all four experiments.
"""

import numpy as np
from scipy import stats


def rmse(pred, true):
    """pred, true: (n, 2) -> scalar RMSE of the 2D estimation error."""
    return np.sqrt(((pred - true) ** 2).sum(axis=1).mean())


def uncertainty_error_correlation(pred_std_flat, actual_err_flat):
    """
    Experiment A: correlation between predicted per-sensor uncertainty and
    actual per-sensor observation error. Returns both Pearson and Spearman,
    since bias faults create heavy-tailed error distributions where rank
    correlation (Spearman) is the more robust summary; Pearson is reported
    too for completeness.
    """
    r_pearson, p_pearson = stats.pearsonr(pred_std_flat, actual_err_flat)
    r_spearman, p_spearman = stats.spearmanr(pred_std_flat, actual_err_flat)
    return {
        "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
        "spearman_rho": float(r_spearman), "spearman_p": float(p_spearman),
    }


def binned_trend(x, y, nbins=25):
    """Bin x into quantile bins and return the mean/std of y in each bin --
    used to draw a clean trend line through a noisy scatter (Graph 1)."""
    bin_edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, nbins - 1)
    bin_x = np.array([x[bin_idx == b].mean() for b in range(nbins)])
    bin_y = np.array([y[bin_idx == b].mean() for b in range(nbins)])
    bin_y_std = np.array([y[bin_idx == b].std() for b in range(nbins)])
    return bin_x, bin_y, bin_y_std


def nees_calibration(mu, V, true_pos, confidence_levels=None):
    """
    Experiment D: NEES (Normalized Estimation Error Squared) consistency
    check, the standard Kalman-filter calibration diagnostic. Under a
    correctly calibrated isotropic 2D Gaussian, NEES = ||mu - true||^2 / V
    should follow a chi-squared distribution with 2 degrees of freedom.

    Returns (nominal_p, empirical_coverage, nees_values, ece) where ece is
    the mean absolute gap between nominal and empirical coverage.
    """
    if confidence_levels is None:
        confidence_levels = np.linspace(0.05, 0.95, 19)
    e = mu - true_pos
    d2 = (e ** 2).sum(axis=1) / V
    thresh = stats.chi2.ppf(confidence_levels, df=2)
    empirical = np.array([(d2 <= t).mean() for t in thresh])
    ece = float(np.mean(np.abs(confidence_levels - empirical)))
    return confidence_levels, empirical, d2, ece
