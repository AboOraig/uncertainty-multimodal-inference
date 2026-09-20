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
                      bias_gain=1.0, n_degraded_sensors=1, deceptive=False,
                      deceptive_confidence_range=(0.3, 0.6), t_df=3):
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
    n_degraded_sensors : how many sensors are simultaneously "stressed" this
        sample (default 1, matching the TRAINING distribution -- Tier 2's
        shift taxonomy uses 2 or 3 to test a structural generalization gap
        never seen in training)
    deceptive : if True, a faulty sensor's self-reported quality gets
        LOWERED (falsely appears MORE confident) instead of raised -- the
        Tier-2 "spoofed sensor" shift: confidently wrong, not just wrong
    deceptive_confidence_range : how strongly a deceptive sensor lies about
        its confidence (multiplier on true std; lower = more deceptive)
    t_df : degrees of freedom for the student_t noise family (lower = more
        heavy-tailed / severe; default 3 matches the original Tier-1 shift)

    Returns dict with:
        true_pos   (n, 2)
        obs        (n, 3, 2)
        quality    (n, 3)      self-reported noise proxy (observable)
        mask       (n, 3)      1 = observed, 0 = dropped out
        actual_std (n, 3)      ground-truth per-sample per-sensor Gaussian/t
                                noise std ONLY, excluding bias (for reference)
        oracle_var (n, 3)      TRUE total per-sample per-sensor variance,
                                including the realized bias-fault contribution
                                -- this is what a genuine oracle WLS baseline
                                should use (see inference/fusion_numpy.oracle_wls)
        d          (n,)        degradation level used for that sample (for evaluation only)
        degraded_sensor (n,)   ground-truth index of ONE of the sample's
                                stressed sensors (first, if n_degraded_sensors>1;
                                for evaluation only)
        is_degraded (n, 3)     boolean, True for EVERY stressed sensor this
                                sample (generalizes degraded_sensor to
                                n_degraded_sensors>1; for evaluation only)
    """
    d_arr = _sample_degradation(n, d, rng)
    true_pos = rng.uniform(-10, 10, size=(n, 2))

    obs = np.zeros((n, N_SENSORS, 2))
    quality = np.zeros((n, N_SENSORS))
    mask = np.ones((n, N_SENSORS))
    actual_std = np.zeros((n, N_SENSORS))
    oracle_var = np.zeros((n, N_SENSORS))   # true per-instance MSE-per-dim, incl. realized bias

    # Degradation is LOCALIZED: on each sample, n_degraded_sensors randomly
    # chosen sensor(s) are "stressed" and their noise/bias/dropout worsen
    # with d; the other sensors stay at their nominal (d=0) quality. With
    # the default n_degraded_sensors=1 (the TRAINING distribution), this is
    # what makes fixed, non-adaptive weighting genuinely suboptimal -- a
    # good fusion rule has to figure out, per sample, WHICH sensor is
    # currently degraded rather than assume a fixed noise ranking (the
    # "identify degraded observation sources" premise from the project
    # spec). n_degraded_sensors>1 (Tier 2 only) tests generalization to a
    # structurally different regime never seen in training.
    k = min(n_degraded_sensors, N_SENSORS)
    degraded_idx = np.array([rng.choice(N_SENSORS, size=k, replace=False) for _ in range(n)])  # (n,k)
    is_degraded = np.zeros((n, N_SENSORS), dtype=bool)
    for j in range(k):
        is_degraded[np.arange(n), degraded_idx[:, j]] = True
    degraded_sensor = degraded_idx[:, 0]   # first stressed sensor, for backward-compatible single-index use

    for i in range(N_SENSORS):
        local_d = np.where(is_degraded[:, i], d_arr, 0.0)         # only degraded when targeted
        std_i = BASE_STD[i] * (1.0 + NOISE_GAIN * local_d)         # (n,)
        actual_std[:, i] = std_i

        if noise_family == "gaussian":
            raw = rng.normal(0.0, 1.0, size=(n, 2))
        elif noise_family == "student_t":
            raw = rng.standard_t(df=t_df, size=(n, 2)) / np.sqrt(t_df / (t_df - 2))  # rescaled to unit std
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
        bias_vec_applied = np.where(faulty[:, None], bias_vec, 0.0)
        sensor_obs = sensor_obs + bias_vec_applied

        obs[:, i, :] = sensor_obs

        # The oracle's assumed variance must include the REALIZED bias
        # contribution, not just the Gaussian noise std. Inverse-variance
        # WLS weighting assumes zero-mean noise; a sensor can have a small
        # Gaussian std (looks precise) while still carrying a large
        # deterministic bias offset this particular sample (bias faults
        # occur at a baseline rate on ANY sensor, not only the one flagged
        # "degraded" -- see bias_prob above). An oracle that only knew
        # actual_std would systematically OVER-trust such a sensor and can
        # underperform even a crude heuristic. Treating the realized bias
        # magnitude as an effective per-dimension variance contribution
        # (mag^2 split evenly across the 2 dims) is what makes this a
        # genuine performance ceiling rather than a mislabeled baseline.
        oracle_var[:, i] = std_i ** 2 + 0.5 * (bias_vec_applied ** 2).sum(axis=1)

        # dropout (also somewhat more likely when degraded)
        dropped = rng.random(n) < np.clip(DROPOUT_P[i] * (1.0 + local_d), 0, 0.9)
        mask[:, i] = (~dropped).astype(float)

        # self-reported quality proxy: a noisy estimate of the sensor's own
        # std, with an extra boost when a bias fault is active (crude
        # self-diagnostics: a faulting sensor is somewhat more likely to
        # also flag low confidence, but the signal is not perfectly
        # reliable) -- UNLESS `deceptive`, in which case a faulty sensor
        # LOWERS its reported quality value instead (falsely claims to be
        # MORE precise while it's actually biased: "confidently wrong",
        # the Tier-2 spoofed-sensor shift, never seen in training).
        if deceptive:
            fault_boost = np.where(faulty, rng.uniform(*deceptive_confidence_range, size=n), 1.0)
        else:
            fault_boost = np.where(faulty, rng.uniform(1.5, 3.0, size=n), 1.0)
        q = std_i * fault_boost * (1.0 + rng.normal(0.0, quality_noise_coef, size=n))
        quality[:, i] = np.clip(q, 1e-3, None)

    return dict(true_pos=true_pos, obs=obs, quality=quality, mask=mask,
                actual_std=actual_std, oracle_var=oracle_var, d=d_arr,
                degraded_sensor=degraded_sensor, is_degraded=is_degraded)


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


def per_sensor_features_no_quality(batch):
    """
    ABLATION (Tier 2): (n, 3, 3) per-sensor input = [obs_x, obs_y, mask] --
    quality deliberately withheld. Measures how much of the v1 model's
    performance comes from that one engineered feature vs. anything else it
    could in principle learn from raw observations alone (with only
    [obs, mask] and no quality, there is very little left to learn from per
    sensor in isolation, so this should perform close to chance/uniform
    weighting -- confirming quality is doing essentially all the work).
    """
    obs, mask = batch["obs"], batch["mask"]
    return np.concatenate([obs, mask[:, :, None]], axis=2)


def per_sensor_features_v2(batch):
    """
    FOLLOW-UP (not part of the core Tier-1 pipeline): (n, 3, 5) per-sensor
    input = [obs_x, obs_y, quality, mask, residual_mag], where residual_mag
    is sensor i's disagreement with a LEAVE-ONE-OUT consensus of the OTHER
    observed sensors:

        consensus_{-i} = mean_{j != i, mask_j=1}(obs_j)
        residual_mag_i = || obs_i - consensus_{-i} ||

    (falls back to residual_mag_i = 0 if no peer sensor is observed --
    "no information available, assume no disagreement".)

    Motivation: the v1 uncertainty model only ever sees a sensor's OWN
    [obs, quality, mask] -- it has no way to notice "sensors 1 and 2 agree
    but sensor 3 doesn't", a cross-sensor consistency cue that the
    quality-weighted-WLS heuristic baseline can't use either. This feature
    gives the model access to that cue, to test whether it can then do
    something the raw quality signal structurally cannot.
    """
    obs, quality, mask = batch["obs"], batch["quality"], batch["mask"]
    n = obs.shape[0]
    residual_mag = np.zeros((n, N_SENSORS))
    for i in range(N_SENSORS):
        peer_mask = mask.copy()
        peer_mask[:, i] = 0.0                              # exclude sensor i itself
        peer_count = peer_mask.sum(axis=1)                  # (n,)
        peer_sum = (peer_mask[:, :, None] * obs).sum(axis=1)  # (n,2)
        has_peers = peer_count > 0.5
        consensus = np.where(has_peers[:, None], peer_sum / np.maximum(peer_count, 1)[:, None], obs[:, i, :])
        residual_mag[:, i] = np.linalg.norm(obs[:, i, :] - consensus, axis=1)
    return np.concatenate([obs, quality[:, :, None], mask[:, :, None], residual_mag[:, :, None]], axis=2)
