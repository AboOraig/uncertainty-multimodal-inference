"""
experiments/tier2_shift_taxonomy.py
--------------------------------------
Tier 1's Experiment D used one combined shift (heavy-tailed noise + noisier
quality + more bias, all at once, one severity). This replaces it with a
small taxonomy of 4 DISTINCT, individually-tested shift types, each at 2
severities, so a stress-test grid instead of one point:

  S1 heavy_tailed_noise    -- Student-t noise family instead of Gaussian
                               (mild: df=10, severe: df=3)
  S2 quality_corruption    -- the self-reported quality proxy becomes a much
                               less reliable estimate of true noise
                               (mild: coef=0.4, severe: coef=1.2)
  S3 correlated_degradation -- MULTIPLE sensors degrade simultaneously, never
                               seen in training (always exactly 1 there)
                               (mild: 2 of 3 sensors, severe: all 3)
  S4 deceptive_sensor      -- a biased sensor LOWERS its reported quality
                               value instead of raising it -- "confidently
                               wrong" rather than just wrong
                               (mild: reports 0.5-0.8x true std,
                                severe: reports 0.2-0.4x true std)

All at a fixed degradation level d=0.5 (matching Tier 1's Experiment D), so
severity differences are attributable to the shift mechanism itself, not a
confound with degradation level. Evaluated: RMSE (fixed WLS, quality WLS,
oracle WLS, v1) and calibration ECE (v1 only).

Usage:
    python -m experiments.tier2_shift_taxonomy --engine numpy
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.simulate import generate_dataset, per_sensor_features, N_SENSORS
from inference.fusion_numpy import fixed_wls, quality_weighted_wls, oracle_wls, structured_fusion_forward
from evaluation.metrics import rmse, nees_calibration, aggregate_seeds
from configs.default import BASE_STD, TRAIN_SEEDS, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH, D_FIXED_FOR_SHIFT

from experiments.run_experiments import get_engine

N_SHIFT_SAMPLES = 6000

SHIFT_CONDITIONS = {
    "in_distribution":        dict(),
    "heavy_tail_mild":        dict(noise_family="student_t", t_df=10),
    "heavy_tail_severe":      dict(noise_family="student_t", t_df=3),
    "quality_corrupt_mild":   dict(quality_noise_coef=0.4),
    "quality_corrupt_severe": dict(quality_noise_coef=1.2),
    "correlated_mild":        dict(n_degraded_sensors=2),
    "correlated_severe":      dict(n_degraded_sensors=3),
    "deceptive_mild":         dict(deceptive=True, deceptive_confidence_range=(0.5, 0.8)),
    "deceptive_severe":       dict(deceptive=True, deceptive_confidence_range=(0.2, 0.4)),
}


def build_shift_batches(eval_rng):
    """Fixed shift protocol, generated once, reused across all training seeds."""
    return {name: generate_dataset(N_SHIFT_SAMPLES, D_FIXED_FOR_SHIFT, eval_rng, **kwargs)
            for name, kwargs in SHIFT_CONDITIONS.items()}


def main(engine_name):
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", engine_name, "tier2_shift_taxonomy"))
    os.makedirs(out_dir, exist_ok=True)

    print("Building fixed shift-taxonomy evaluation protocol...")
    eval_rng = np.random.default_rng(424242)   # separate from EVAL_SEED, dedicated to this experiment
    shift_batches = build_shift_batches(eval_rng)
    nominal_var = BASE_STD ** 2

    print("Evaluating non-learned baselines (deterministic)...")
    baseline_rmse = {name: {} for name in SHIFT_CONDITIONS}
    for name, b in shift_batches.items():
        baseline_rmse[name]["fixed_wls"] = rmse(fixed_wls(b["obs"], b["mask"], nominal_var), b["true_pos"])
        baseline_rmse[name]["quality_wls"] = rmse(quality_weighted_wls(b["obs"], b["mask"], b["quality"]), b["true_pos"])
        baseline_rmse[name]["oracle_wls"] = rmse(oracle_wls(b["obs"], b["mask"], b["oracle_var"]), b["true_pos"])

    print("Retraining v1 uncertainty model per seed (same protocol as Tier 1)...")
    _, train_uncertainty_model, predict_log_var, seed_kw = get_engine(engine_name)

    v1_rmse_per_seed = {name: [] for name in SHIFT_CONDITIONS}
    v1_ece_per_seed = {name: [] for name in SHIFT_CONDITIONS}
    for seed in TRAIN_SEEDS:
        print(f"  seed {seed}...")
        train_rng = np.random.default_rng(seed)
        train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, train_rng) for _ in range(N_TRAIN_BATCHES)]
        kwargs = {"rng": train_rng} if seed_kw == "rng" else {"seed": seed}
        net, _ = train_uncertainty_model(train_batches, **kwargs)

        for name, b in shift_batches.items():
            log_var = predict_log_var(net, per_sensor_features(b))
            mu, V, w, S = structured_fusion_forward(b["obs"], b["mask"], log_var)
            v1_rmse_per_seed[name].append(rmse(mu, b["true_pos"]))
            _, _, _, ece = nees_calibration(mu, V, b["true_pos"])
            v1_ece_per_seed[name].append(ece)

    v1_rmse_agg = {name: aggregate_seeds([[v] for v in vals]) for name, vals in v1_rmse_per_seed.items()}
    v1_ece_agg = {name: aggregate_seeds([[v] for v in vals]) for name, vals in v1_ece_per_seed.items()}

    # -----------------------------------------------------------------
    # Figure 1: RMSE grouped bar chart across all 9 conditions
    # -----------------------------------------------------------------
    names = list(SHIFT_CONDITIONS.keys())
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5 * width, [baseline_rmse[n]["oracle_wls"] for n in names], width, label="Oracle WLS", color="black", alpha=0.5)
    ax.bar(x - 0.5 * width, [baseline_rmse[n]["quality_wls"] for n in names], width, label="Quality-weighted WLS", color="tab:green")
    ax.bar(x + 0.5 * width, [v1_rmse_agg[n][0][0] for n in names], width,
           yerr=[v1_rmse_agg[n][1][0] for n in names], label="v1 uncertainty-aware", color="crimson", capsize=3)
    ax.bar(x + 1.5 * width, [baseline_rmse[n]["fixed_wls"] for n in names], width, label="Fixed WLS", color="tab:blue", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Estimation RMSE")
    ax.set_title(f"Tier 2 — shift taxonomy (fixed d={D_FIXED_FOR_SHIFT}; v1 error bars = std across seeds)", fontsize=11)
    ax.legend(fontsize=8)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tier2_shift_rmse.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 2: calibration ECE across all 9 conditions (v1 only)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ece_means = [v1_ece_agg[n][0][0] for n in names]
    ece_stds = [v1_ece_agg[n][1][0] for n in names]
    bar_colors = ["seagreen" if n == "in_distribution" else "crimson" for n in names]
    ax.bar(x, ece_means, yerr=ece_stds, color=bar_colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Calibration ECE (NEES-based)")
    ax.set_title("Tier 2 — calibration error across the shift taxonomy (v1 model)\n(error bars = std across seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tier2_shift_calibration.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Numeric summary
    # -----------------------------------------------------------------
    summary = {}
    for name in names:
        summary[name] = {
            "conditions": SHIFT_CONDITIONS[name],
            "rmse_fixed_wls": baseline_rmse[name]["fixed_wls"],
            "rmse_quality_wls": baseline_rmse[name]["quality_wls"],
            "rmse_oracle_wls": baseline_rmse[name]["oracle_wls"],
            "rmse_v1_mean": float(v1_rmse_agg[name][0][0]),
            "rmse_v1_std": float(v1_rmse_agg[name][1][0]),
            "ece_v1_mean": float(v1_ece_agg[name][0][0]),
            "ece_v1_std": float(v1_ece_agg[name][1][0]),
        }
    with open(os.path.join(out_dir, "tier2_shift_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. 2 figures + tier2_shift_results.json written to {out_dir}")
    for name in names:
        s = summary[name]
        print(f"{name:24s}  quality={s['rmse_quality_wls']:.3f}  v1={s['rmse_v1_mean']:.3f}±{s['rmse_v1_std']:.3f}  "
              f"oracle={s['rmse_oracle_wls']:.3f}  ECE={s['ece_v1_mean']:.3f}±{s['ece_v1_std']:.3f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
