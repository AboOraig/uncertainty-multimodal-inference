"""
Bounds the uncertainty network's raw output into a sane log-variance range
via a scaled sigmoid, so that predictions on out-of-distribution inputs
(e.g. degradation levels beyond the training range) saturate instead of
diverging. This is standard practice for heteroscedastic-regression heads.

LOG_VAR_MIN/MAX correspond to predicted std roughly in [0.14, 7.4], which
comfortably covers this project's noise range (best sensor std=0.3,
worst-case degraded std up to ~5.5).
"""
import numpy as np
from configs.default import LOG_VAR_MIN, LOG_VAR_MAX


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bound_log_var(raw):
    """raw -> (log_var, local_grad) where local_grad = d(log_var)/d(raw)."""
    sig = _sigmoid(raw)
    log_var = LOG_VAR_MIN + (LOG_VAR_MAX - LOG_VAR_MIN) * sig
    local_grad = (LOG_VAR_MAX - LOG_VAR_MIN) * sig * (1 - sig)
    return log_var, local_grad
