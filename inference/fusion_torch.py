"""
fusion_torch.py
----------------
PyTorch version of the structured (precision-weighted) fusion rule and its
Gaussian NLL loss. This replaces the *training-time* pieces of inference/fusion_numpy.py
(nll_loss_and_grad, which hand-derived dNLL/dlogvar) and bounds.py (which
hand-derived d(bounded_log_var)/d(raw)).

With autograd, we only need to write the FORWARD computation -- the exact
same math as fusion_numpy.py's structured_fusion_forward() and nll_loss_and_grad()
-- and call loss.backward() to get every gradient automatically. Compare
this file's line count to fusion_numpy.py's derivation-heavy docstring to see why
that's worth the PyTorch dependency.

Note: fusion_numpy.py's NumPy structured_fusion_forward() and fixed_wls() are
still used as-is for *evaluation* (Experiments A-D operate on plain NumPy
arrays produced by simulate.py and by predict_log_var() below) -- only the
*training* path needs autograd.
"""

import math
import torch
from configs.default import LOG_VAR_MIN, LOG_VAR_MAX

LOG_2PI = math.log(2 * math.pi)
EPS = 1e-6


def bound_log_var(raw):
    """raw (unbounded) -> log_var in [LOG_VAR_MIN, LOG_VAR_MAX] via a scaled
    sigmoid, so predictions on out-of-distribution inputs saturate instead
    of diverging. (This is bounds.py's bound_log_var(), but autograd handles
    the derivative through sigmoid for us -- no local_grad bookkeeping.)"""
    sig = torch.sigmoid(raw)
    return LOG_VAR_MIN + (LOG_VAR_MAX - LOG_VAR_MIN) * sig


def structured_fusion_forward(obs, mask, log_var):
    """
    obs: (n,3,2) tensor, mask: (n,3) tensor, log_var: (n,3) tensor.
    Returns mu (n,2), V (n,) -- identical math to fusion_numpy.py's NumPy version.
    """
    w = mask * torch.exp(-log_var) + EPS
    S = w.sum(dim=1)                                       # (n,)
    mu = (w.unsqueeze(-1) * obs).sum(dim=1) / S.unsqueeze(-1)  # (n,2)
    V = 1.0 / S
    return mu, V


def nll_loss(obs, mask, log_var, true_pos):
    """Gaussian NLL of the fused estimate (2D isotropic). Returns a scalar
    loss tensor -- call .backward() on it, no manual gradient required."""
    mu, V = structured_fusion_forward(obs, mask, log_var)
    e = mu - true_pos
    sq_err = (e ** 2).sum(dim=1)
    loss_per_sample = LOG_2PI + torch.log(V) + 0.5 * sq_err / V
    return loss_per_sample.mean(), mu, V
