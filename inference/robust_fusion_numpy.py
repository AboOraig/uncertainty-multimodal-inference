"""
inference/robust_fusion_numpy.py
----------------------------------
Robust, ITERATIVE fusion rules -- Huber and Tukey-biweight IRLS (Iteratively
Reweighted Least Squares), motivated by the Tier-1 follow-up finding (see
README, "Follow-up: does a cross-sensor feature fix it?"): every
inverse-variance (linear, one-shot) fusion method tested -- quality
heuristic, learned v1, learned v2 -- sat at an almost identical, persistent
RMSE gap above the oracle ceiling. That's the signature of a fusion-RULE
limitation, not an uncertainty-ESTIMATE limitation: precision weighting
downweights a suspected-bad sensor smoothly and proportionally, but a bias
fault behaves close to bimodally (a sensor is either fine, or wildly off)
-- exactly the setting classical M-estimators (Huber, Tukey) were designed
for.

Given a prior per-sensor variance (from ANY source -- the quality heuristic,
or a trained uncertainty model's exp(log_var)), IRLS refines the fused
estimate by iterating:
  1. compute each sensor's residual from the current fused estimate,
     standardized by its prior std;
  2. apply a robust psi-function to that residual -- Huber (soft downweight
     past a threshold, weight ~ 1/r) or Tukey biweight (SMOOTHLY falls to
     EXACTLY zero at the threshold and stays there -- genuine hard gating,
     not just more aggressive downweighting);
  3. re-fuse with the adjusted weights.

This needs no additional training -- it's a refinement step that can be
layered on top of any prior variance source at inference time.
"""

import numpy as np
import warnings

EPS = 1e-6


def huber_psi(standardized_residual, c):
    """Soft downweight beyond threshold c: full weight inside, ~1/r falloff
    outside (never reaches exactly zero)."""
    r = np.maximum(standardized_residual, EPS)
    return np.minimum(1.0, c / r)


def tukey_psi(standardized_residual, c):
    """Tukey biweight: full weight near 0, smoothly falls to EXACTLY zero at
    r=c and stays zero beyond -- genuine hard gating with a smooth (not
    discontinuous) falloff, which is gentler on samples near the threshold
    than a hard step function while still fully excluding clear outliers."""
    r = standardized_residual
    inside = r <= c
    factor = np.where(inside, 1.0 - (r / np.maximum(c, EPS)) ** 2, 0.0)
    return np.where(inside, factor ** 2, 0.0)


PSI_FUNCTIONS = {"huber": huber_psi, "tukey": tukey_psi}


def _robust_median_init(obs, mask):
    """Coordinate-wise median across observed sensors -- a classical robust
    starting point (breakdown point 50%) for IRLS, used instead of the
    naive precision-weighted mean. Matters specifically for Huber: its psi
    is EXACTLY 1.0 (no downweighting at all) below the threshold c, so if
    the starting estimate is already pulled toward an outlier, standardized
    residuals for every sensor -- including the outlier itself -- can land
    below c on iteration 1 and the fusion never moves (verified empirically
    during development: Huber got permanently stuck at a bad initial
    estimate when started from the naive weighted mean). A median start
    sidesteps this."""
    obs_nan = np.where(mask[:, :, None] > 0.5, obs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # expected: all-dropped rows, handled by fallback below
        med = np.nanmedian(obs_nan, axis=1)
    # fallback for the (rare) all-dropped-out row: naive mean of raw obs
    all_dropped = np.all(mask < 0.5, axis=1)
    if all_dropped.any():
        med[all_dropped] = obs[all_dropped].mean(axis=1)
    return med


def irls_fusion(obs, mask, prior_var, psi_name="tukey", c=3.0, n_iters=25, damping=0.5, mad_floor=1e-3):
    """
    obs: (n,3,2), mask: (n,3), prior_var: (n,3) or (3,) prior per-sensor
    variance (from ANY source: quality^2, or exp(learned log_var)) -- used
    for the BASE inverse-variance weighting.
    psi_name: 'huber' or 'tukey'. c: robustness threshold. n_iters: number
    of reweight/refuse passes. damping: step size in (0,1] controlling how
    much of each iteration's newly-computed estimate to accept (see below).

    Residuals are standardized by a ROBUST PER-SAMPLE GROUP SCALE (a
    MAD-like estimate: 1.4826 * median residual across sensors for that
    sample), NOT by each sensor's own claimed prior std -- standardizing by
    a sensor's own claim is circular (see _robust_median_init's docstring
    for the failure mode this caused during development). The group scale
    instead asks "how far is sensor i from what its PEERS currently agree
    on", the actual cross-sensor-consistency signal this method exploits.

    UNDAMPED fixed-point IRLS is unstable in this small-n (3 sensors)
    regime: once a severe outlier gets ANY nonzero weight, mu drifts toward
    it, which SHRINKS its own residual, which INCREASES its weight next
    iteration -- a positive-feedback runaway toward trusting the outlier
    MORE. Verified empirically during development: an undamped version
    reliably converged to the bad, fully-trust-the-outlier fixed point
    instead of excluding it, even starting from the robust median. Damped
    updates (mu moves only `damping` of the way to each iteration's new
    estimate) is the standard fix and is what makes this stable.

    Returns (mu, final_w): fused (n,2) estimate and the (n,3) weights from
    the final iteration (useful as a "how suspected-bad is this sensor"
    score for failure analysis -- lower final weight = more suspected).
    """
    psi_fn = PSI_FUNCTIONS[psi_name]
    prior_var_b = np.broadcast_to(prior_var, mask.shape).astype(float)

    mu = _robust_median_init(obs, mask)
    w = mask / prior_var_b + EPS  # only used if n_iters == 0

    for _ in range(n_iters):
        resid = np.linalg.norm(obs - mu[:, None, :], axis=2)             # (n,3) raw residual magnitude
        resid_for_scale = np.where(mask > 0.5, resid, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scale = np.nanmedian(resid_for_scale, axis=1, keepdims=True) * 1.4826
        scale = np.maximum(np.nan_to_num(scale, nan=mad_floor), mad_floor)
        standardized = resid / scale
        psi = psi_fn(standardized, c)
        w = mask * psi / prior_var_b + EPS
        S = w.sum(axis=1, keepdims=True)
        mu_new = (w[:, :, None] * obs).sum(axis=1) / S
        mu = damping * mu_new + (1 - damping) * mu

    return mu, w


def tune_c(obs, mask, true_pos, prior_var, psi_name, c_grid, n_iters=25, damping=0.5):
    """Grid-search c on a VALIDATION batch (never the actual eval protocol)
    by minimizing RMSE. Returns (best_c, best_rmse, all_results)."""
    from evaluation.metrics import rmse
    results = []
    for c in c_grid:
        mu, _ = irls_fusion(obs, mask, prior_var, psi_name=psi_name, c=c, n_iters=n_iters, damping=damping)
        results.append((c, rmse(mu, true_pos)))
    best_c, best_rmse = min(results, key=lambda x: x[1])
    return best_c, best_rmse, results
