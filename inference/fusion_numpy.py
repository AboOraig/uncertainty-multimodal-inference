"""
inference/fusion_numpy.py
---------
Fusion rules.

1. fixed_wls: the "conventional" baseline. Combines observations with
   inverse-variance weights computed ONCE from nominal (training-average)
   per-sensor noise, never adapted to instance- or degradation-level
   conditions.

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


def fixed_wls(obs, mask, nominal_var):
    """
    obs: (n, 3, 2), mask: (n, 3), nominal_var: (3,) fixed per-sensor variance.
    Returns fused (n, 2) estimate. Weights never adapt to instance conditions.
    """
    w = mask / nominal_var[None, :]          # (n, 3)
    w = w + EPS                               # floor so it's never exactly zero
    S = w.sum(axis=1, keepdims=True)          # (n, 1)
    mu = (w[:, :, None] * obs).sum(axis=1) / S
    return mu


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
