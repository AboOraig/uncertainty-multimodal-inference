"""
simulate.py
-----------
Synthetic multimodal state-estimation environment.

We simulate a 2D position that must be estimated from 3 heterogeneous
"sensors" (modalities). Each sensor has:
  - its own baseline noise level (heteroscedastic, scales with a
    controllable "degradation level" d)
  - a probability of missing data (dropout)
  - a probability of a bias fault (systematic offset)
  - a noisy self-reported "quality" signal, i.e. a proxy for its own
    current noise level, similar to an SNR / confidence channel a real
    sensor might expose. This is the observable feature the uncertainty
    model is allowed to use.

Two "noise families" are supported: Gaussian (the training regime) and
Student-t with 3 degrees of freedom (heavy-tailed, used only to build an
unseen "shifted" test regime for the distribution-shift experiment).
"""

import numpy as np
from configs.default import (
    N_SENSORS, BASE_STD, DROPOUT_P, BIAS_P, BIAS_MAG_RANGE,
    NOISE_GAIN, QUALITY_NOISE_COEF,
)


def _sample_degradation(n, d, rng):
    """d may be a scalar (fixed level) or a (lo, hi) tuple (sampled per-sample)."""
    if np.isscalar(d):
        return np.full(n, float(d))
    lo, hi = d
    return rng.uniform(lo, hi, size=n)


def generate_dataset(n, d, rng, noise_family="gaussian", quality_noise_coef=QUALITY_NOISE_COEF,
                      bias_gain=1.0):
    """
    Generate n i.i.d. multimodal observation samples.

    Parameters
    ----------
    n : int
    d : float or (lo, hi) -- degradation level (0 = nominal conditions)
    rng : np.random.Generator
    noise_family : 'gaussian' or 'student_t'  (heavy-tailed, used for shift experiments)
    quality_noise_coef : how unreliable the self-reported quality proxy is
        (higher = quality signal is a worse proxy for the true noise -> harder / shifted regime)
    bias_gain : multiplier on bias-fault probability (>1 = more faults, shifted regime)

    Returns dict with:
        true_pos   (n, 2)
        obs        (n, 3, 2)
        quality    (n, 3)      self-reported noise proxy (observable)
        mask       (n, 3)      1 = observed, 0 = dropped out
        actual_std (n, 3)      ground-truth per-sample per-sensor std (for evaluation only)
        d          (n,)        degradation level used for that sample (for evaluation only)
    """
    d_arr = _sample_degradation(n, d, rng)
    true_pos = rng.uniform(-10, 10, size=(n, 2))

    obs = np.zeros((n, N_SENSORS, 2))
    quality = np.zeros((n, N_SENSORS))
    mask = np.ones((n, N_SENSORS))
    actual_std = np.zeros((n, N_SENSORS))

    # Degradation is LOCALIZED: on each sample, one randomly chosen sensor is
    # the "stressed" modality and its noise/bias/dropout worsen with d; the
    # other sensors stay at their nominal (d=0) quality. This is what makes
    # fixed, non-adaptive weighting genuinely suboptimal -- a good fusion
    # rule has to figure out, per sample, WHICH sensor is currently degraded
    # rather than assume a single fixed noise ranking (this is the
    # "identify degraded observation sources" premise from the project spec).
    degraded_sensor = rng.integers(0, N_SENSORS, size=n)
    is_degraded = np.zeros((n, N_SENSORS), dtype=bool)
    is_degraded[np.arange(n), degraded_sensor] = True

    for i in range(N_SENSORS):
        local_d = np.where(is_degraded[:, i], d_arr, 0.0)         # only degraded when targeted
        std_i = BASE_STD[i] * (1.0 + NOISE_GAIN * local_d)         # (n,)
        actual_std[:, i] = std_i

        if noise_family == "gaussian":
            raw = rng.normal(0.0, 1.0, size=(n, 2))
        elif noise_family == "student_t":
            raw = rng.standard_t(df=3, size=(n, 2)) / np.sqrt(3.0)  # rescaled to unit std
        else:
            raise ValueError(noise_family)
        noise = raw * std_i[:, None]

        sensor_obs = true_pos + noise

        # bias faults: a systematic offset in a random direction, more likely
        # when this sensor is the currently-degraded one
        bias_prob = np.clip(BIAS_P[i] * bias_gain * (1.0 + local_d), 0, 0.9)
        faulty = rng.random(n) < bias_prob
        angle = rng.uniform(0, 2 * np.pi, size=n)
        mag = rng.uniform(*BIAS_MAG_RANGE, size=n)
        bias_vec = np.stack([np.cos(angle), np.sin(angle)], axis=1) * mag[:, None]
        sensor_obs = np.where(faulty[:, None], sensor_obs + bias_vec, sensor_obs)

        obs[:, i, :] = sensor_obs

        # dropout (also somewhat more likely when degraded)
        dropped = rng.random(n) < np.clip(DROPOUT_P[i] * (1.0 + local_d), 0, 0.9)
        mask[:, i] = (~dropped).astype(float)

        # self-reported quality proxy: a noisy estimate of the sensor's own
        # std, with an extra boost when a bias fault is active (crude
        # self-diagnostics: a faulting sensor is somewhat more likely to
        # also flag low confidence, but the signal is not perfectly reliable)
        fault_boost = np.where(faulty, rng.uniform(1.5, 3.0, size=n), 1.0)
        q = std_i * fault_boost * (1.0 + rng.normal(0.0, quality_noise_coef, size=n))
        quality[:, i] = np.clip(q, 1e-3, None)

    return dict(true_pos=true_pos, obs=obs, quality=quality, mask=mask,
                actual_std=actual_std, d=d_arr)


def featurize(batch):
    """
    Build the (n, 12) input used by the learned-fusion (direct MSE) model:
    per sensor -> [obs_x, obs_y, quality (zeroed if missing), mask]
    """
    obs, quality, mask = batch["obs"], batch["quality"], batch["mask"]
    n = obs.shape[0]
    feats = np.concatenate([
        obs * mask[:, :, None],
        (quality * mask)[:, :, None],
        mask[:, :, None],
    ], axis=2)  # (n, 3, 4)
    return feats.reshape(n, N_SENSORS * 4)


def per_sensor_features(batch):
    """
    Build the (n, 3, 4) per-sensor input used by the uncertainty model:
    [obs_x, obs_y, quality, mask] -- evaluated independently per sensor
    (this is what makes the uncertainty model "structured": it scores each
    modality on its own evidence, not a black-box mix of everything).
    """
    obs, quality, mask = batch["obs"], batch["quality"], batch["mask"]
    return np.concatenate([obs, quality[:, :, None], mask[:, :, None]], axis=2)
