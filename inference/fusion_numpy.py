"""
inference/fusion_numpy.py
---------
Fusion rules -- a baseline ladder plus the proposed structured fusion.

Non-learned WLS baselines (all closed-form, no training, so they are
deterministic given a fixed evaluation set -- zero seed variance):

  - fixed_wls:            "conventional" baseline. Inverse-variance weights
                           from nominal (spec-sheet, d=0) per-sensor noise,
                           never adapted to instance- or field-level
                           conditions.
  - quality_weighted_wls: naive ADAPTIVE baseline. Trusts each sensor's own
                           self-reported quality signal directly as its
                           noise std, with no learning involved at all.
                           This is the baseline that answers "do you even
                           need a network, or does the raw quality signal
                           already get you most of the way there?"
  - oracle_wls:            upper-bound baseline. Uses the TRUE per-sample
                           per-sensor noise std, which no real model has
                           access to (only the simulator/evaluation code
                           does). Shows the ceiling any adaptive method is
                           chasing.

2. structured_fusion: given a PER-SENSOR predicted log-variance (from the
   uncertainty MLP), combine observations via precision weighting:

        w_i  = mask_i * exp(-log_var_i)
        S    = sum_i w_i               (total precision, floored)
        mu   = sum_i w_i * o_i / S      (fused estimate)
        V    = 1 / S                    (fused variance, isotropic per-dim)

   This is exactly a one-shot Kalman/WLS update with LEARNED per-modality
   covariances. Trained with the Gaussian negative log-likelihood of the
   fused estimate:

        NLL = log(2*pi) + log(V) + 0.5 * ||mu - x_true||^2 / V   (per sample, 2D isotropic)

   Gradient derivation (see README for the full derivation):
   writing e = mu - x_true,

        dNLL/dw_j = -1/S + 0.5*||e||^2 + sum_d e_d * (o_{j,d} - mu_d)
        dNLL/dl_j = -w_j * dNLL/dw_j          (l_j = log_var_j, w_j = mask_j * exp(-l_j))

   This is implemented directly (no autodiff needed) in nll_grad_wrt_logvar().
"""

import numpy as np

EPS = 1e-6


def _wls(obs, mask, var):
    """
    Shared WLS core. obs: (n,3,2), mask: (n,3), var: broadcastable to (n,3)
    -- either a fixed (3,) per-sensor variance, or a full (n,3) per-instance
    variance array. All three WLS baselines below are thin wrappers around
    this, differing only in what `var` they supply.
    """
    w = mask / var + EPS                      # floor so it's never exactly zero
    S = w.sum(axis=1, keepdims=True)
    mu = (w[:, :, None] * obs).sum(axis=1) / S
    return mu


def fixed_wls(obs, mask, nominal_var):
    """nominal_var: (3,) fixed per-sensor variance. Weights never adapt to
    instance conditions -- the "conventional" baseline."""
    return _wls(obs, mask, nominal_var[None, :])


def quality_weighted_wls(obs, mask, quality):
    """quality: (n,3) self-reported per-instance quality proxy, used
    directly as the assumed std (var = quality^2). Adaptive, but not
    learned -- tests whether a neural network is even necessary."""
    return _wls(obs, mask, quality ** 2)


def oracle_wls(obs, mask, oracle_var):
    """oracle_var: (n,3) GROUND-TRUTH total per-instance variance, including
    the realized bias-fault contribution (evaluation-only information --
    see data/simulate.py's oracle_var for why raw noise std alone is NOT
    enough to make this a genuine ceiling). Upper-bound baseline."""
    return _wls(obs, mask, oracle_var)


def structured_fusion_forward(obs, mask, log_var):
    """
    obs: (n, 3, 2), mask: (n, 3), log_var: (n, 3) predicted per-sensor log-variance.
    Returns mu (n, 2), V (n,), w (n, 3), S (n,)
    """
    w = mask * np.exp(-log_var) + EPS  # small floor avoids S=0 if everything is dropped
    S = w.sum(axis=1)                   # (n,)
    mu = (w[:, :, None] * obs).sum(axis=1) / S[:, None]   # (n, 2)
    V = 1.0 / S
    return mu, V, w, S


def nll_loss_and_grad(obs, mask, log_var, true_pos):
    """
    Full forward + analytic backward of the structured-fusion Gaussian NLL,
    w.r.t. the network's log_var output (n, 3). Returns (loss_scalar, dLogVar, mu, V).
    """
    n = obs.shape[0]
    mu, V, w, S = structured_fusion_forward(obs, mask, log_var)
    e = mu - true_pos                                     # (n, 2)
    sq_err = (e ** 2).sum(axis=1)                          # (n,)
    loss_per_sample = np.log(2 * np.pi) + np.log(V) + 0.5 * sq_err / V
    loss = loss_per_sample.mean()

    # dNLL/dw_j  = -1/S + 0.5*||e||^2 + sum_d e_d*(o_{j,d}-mu_d)
    term1 = (-1.0 / S)[:, None]                            # (n,1) broadcast over sensors
    term2 = (0.5 * sq_err)[:, None]                         # (n,1)
    term3 = np.einsum('nd,nsd->ns', e, obs - mu[:, None, :])  # (n,3)
    dL_dw = term1 + term2 + term3                           # (n,3)

    dL_dlogvar = -w * dL_dw / n   # divide by n for the mean-loss convention (matches MLP.backward)
    return loss, dL_dlogvar, mu, V
