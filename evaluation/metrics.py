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


def sensor_identification_accuracy(scores, degraded_sensor, mask, d_arr, d_threshold=1e-9):
    """
    Failure-analysis metric: does argmax(scores) actually point at the
    truly degraded sensor? This directly operationalizes the "identify
    degraded observation sources" claim -- previously implied but never
    measured.

    scores: (n,3) something used to rank sensors by suspected
        unreliability (predicted variance, or raw quality as a cheap
        heuristic comparator) -- higher = more suspected degraded.
    degraded_sensor: (n,) ground-truth index of the sample's stressed sensor.
    mask: (n,3) -- samples where the true degraded sensor was dropped out
        are excluded (you can't identify a sensor that isn't there).
    d_arr: (n,) actual degradation level for that sample. At d=0 there is
        no real asymmetric degradation (the label is structurally present
        but noise is equal across sensors), so those samples are excluded
        by default.

    Returns (accuracy, n_valid). accuracy is np.nan if n_valid == 0.
    """
    predicted = np.argmax(scores, axis=1)
    target_observed = mask[np.arange(len(mask)), degraded_sensor] > 0.5
    valid = (d_arr > d_threshold) & target_observed
    if valid.sum() == 0:
        return np.nan, 0
    acc = (predicted[valid] == degraded_sensor[valid]).mean()
    return float(acc), int(valid.sum())


def aggregate_seeds(list_of_arrays):
    """
    list_of_arrays: list of length n_seeds, each an array/list of the same
    shape (e.g. an RMSE-vs-degradation curve from one training seed).
    Returns (mean, std), both with that shape, using nan-safe reduction
    (some per-seed entries, e.g. sensor-ID accuracy at d=0, are legitimately
    undefined/nan across every seed -- that column's mean/std is then nan
    too, which is correct, not a bug; the warning is suppressed since it's
    expected).
    """
    arr = np.asarray(list_of_arrays, dtype=float)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


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
