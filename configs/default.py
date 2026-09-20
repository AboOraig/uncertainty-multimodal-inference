"""
configs/default.py
-------------------
Centralized configuration. Every magic number used by data/, models/,
inference/, training/, and experiments/ lives here, so hyperparameters are
never scattered across the codebase.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Sensor simulation (data/simulate.py)
# ---------------------------------------------------------------------------
N_SENSORS = 3
BASE_STD = np.array([0.3, 0.6, 1.0])        # sensor 1 best, sensor 3 worst
DROPOUT_P = np.array([0.05, 0.10, 0.15])
BIAS_P = np.array([0.05, 0.08, 0.12])
BIAS_MAG_RANGE = (1.5, 3.5)
NOISE_GAIN = 3.0                             # std(d) = BASE_STD * (1 + NOISE_GAIN * d)
QUALITY_NOISE_COEF = 0.15                    # reliability of the self-reported quality proxy

# ---------------------------------------------------------------------------
# Reproducibility: multi-seed training protocol
# ---------------------------------------------------------------------------
# TRAIN_SEEDS: each is an independent training run (fresh data draw + fresh
# model init). EVAL_SEED is SEPARATE and FIXED: the entire evaluation
# protocol (probe set, degradation sweep, calibration sets) is generated
# ONCE with this seed and reused identically across every training seed, so
# that observed variation reflects only training stochasticity, not also a
# shifting test set.
TRAIN_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEED = 12345

# ---------------------------------------------------------------------------
# Training (training/train_numpy.py, training/train_torch.py)
# ---------------------------------------------------------------------------
TRAIN_D_RANGE = (0.0, 0.6)                   # degradation levels seen during training
N_TRAIN_BATCHES = 3
N_TRAIN_PER_BATCH = 4000                     # -> 12,000 training samples total
HIDDEN_SIZES = (32, 32)
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
SEED = 42                                    # fallback single seed (e.g. quick smoke tests)

# ---------------------------------------------------------------------------
# Experiments (experiments/run_experiments.py)
# ---------------------------------------------------------------------------
DEGRADATION_SWEEP = np.linspace(0.0, 1.5, 16)   # Graph 2 / Experiment C: extends past training range
N_EVAL_PER_SWEEP_POINT = 2000

PROBE_D_RANGE = (0.0, 1.5)                      # Experiment A: pooled range for the correlation scatter
PROBE_N = 6000

EXP_B_D_GRID = np.linspace(0.0, 1.5, 16)        # Experiment B: uncertainty vs. degradation
EXP_B_N_PER_POINT = 1500

D_FIXED_FOR_SHIFT = 0.5                         # Experiment D / Graph 3
N_CALIBRATION_SAMPLES = 8000
SHIFT_NOISE_FAMILY = "student_t"                # unseen noise family at test time
SHIFT_QUALITY_NOISE_COEF = 0.6                  # quality proxy becomes much less reliable
SHIFT_BIAS_GAIN = 2.5                           # bias faults 2.5x more frequent

# ---------------------------------------------------------------------------
# Uncertainty-model output bounds (inference/bounds.py, inference/fusion_torch.py)
# ---------------------------------------------------------------------------
LOG_VAR_MIN = -4.0                              # predicted std floor  ~0.14
LOG_VAR_MAX = 4.0                               # predicted std ceiling ~7.4
