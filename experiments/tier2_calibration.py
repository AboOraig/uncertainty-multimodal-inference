"""
experiments/tier2_calibration.py
------------------------------------
Tier 1's Experiment D reported calibration (ECE) at a single fixed
degradation level, in-distribution vs. one combined shift. This extends it
three ways:

  1. Full ECE-vs-degradation CURVES (not a single point), swept across the
     same degradation range as Graph 2, for both the Gaussian (in-family)
     and Student-t (shifted-family) noise regimes.
  2. SHARPNESS alongside calibration -- mean predicted uncertainty (std) at
     each degradation level. Calibration alone is gameable (trivially wide
     intervals are "calibrated" but useless); reporting both together is
     the honest pair.
  3. POST-HOC RECALIBRATION: fits a single scalar variance-scaling factor
     on a HELD-OUT VALIDATION set from the shifted regime (never touching
     the actual evaluation data) by matching the first moment of the NEES
     distribution (method-of-moments: E[NEES] should equal 2 under a
     correctly-calibrated isotropic 2D Gaussian), then checks whether that
     one learned scalar recovers calibration on the actual shifted eval set.

Usage:
    python -m experiments.tier2_calibration --engine numpy
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.simulate import generate_dataset, per_sensor_features
from inference.fusion_numpy import structured_fusion_forward
from evaluation.metrics import nees_calibration, aggregate_seeds
from configs.default import (
    TRAIN_SEEDS, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH,
    DEGRADATION_SWEEP, N_EVAL_PER_SWEEP_POINT,
)

from experiments.run_experiments import get_engine

CALIB_EVAL_SEED = 314159        # dedicated, separate from EVAL_SEED
RECAL_VALIDATION_SEED = 271828  # separate again -- never touches the actual eval data
SHIFT_T_DF = 3


def build_calibration_sweep_batches():
    """Fixed Gaussian and shifted (Student-t) sweeps across the SAME
    degradation range, generated once and reused across all seeds."""
    rng = np.random.default_rng(CALIB_EVAL_SEED)
    gaussian = [generate_dataset(N_EVAL_PER_SWEEP_POINT, float(d), rng, noise_family="gaussian")
                for d in DEGRADATION_SWEEP]
    shifted = [generate_dataset(N_EVAL_PER_SWEEP_POINT, float(d), rng, noise_family="student_t", t_df=SHIFT_T_DF)
               for d in DEGRADATION_SWEEP]
    return gaussian, shifted


def build_recalibration_validation_batch():
    """Held-out validation data from the SAME shifted regime, used ONLY to
    fit the recalibration scalar -- never the actual evaluation batches."""
    rng = np.random.default_rng(RECAL_VALIDATION_SEED)
    return generate_dataset(6000, (0.3, 1.0), rng, noise_family="student_t", t_df=SHIFT_T_DF)


def fit_recalibration_scale(mu, V, true_pos):
    """Method-of-moments: under correct calibration, E[NEES] = 2 (df=2
    chi-square mean). Returns tau such that V_new = tau * V makes that hold
    on this (validation) data."""
    e = mu - true_pos
    nees = (e ** 2).sum(axis=1) / V
    return float(nees.mean() / 2.0)


def main(engine_name):
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", engine_name, "tier2_calibration"))
    os.makedirs(out_dir, exist_ok=True)

    print("Building fixed calibration-sweep evaluation protocol (Gaussian + Student-t)...")
    gaussian_batches, shifted_batches = build_calibration_sweep_batches()
    val_batch = build_recalibration_validation_batch()

    print("Retraining v1 uncertainty model per seed (same protocol as Tier 1)...")
    _, train_uncertainty_model, predict_log_var, seed_kw = get_engine(engine_name)

    ece_gauss_all, ece_shift_all = [], []
    sharp_gauss_all, sharp_shift_all = [], []
    ece_shift_recal_all = []
    tau_all = []

    for seed in TRAIN_SEEDS:
        print(f"  seed {seed}...")
        train_rng = np.random.default_rng(seed)
        train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, train_rng) for _ in range(N_TRAIN_BATCHES)]
        kwargs = {"rng": train_rng} if seed_kw == "rng" else {"seed": seed}
        net, _ = train_uncertainty_model(train_batches, **kwargs)

        # fit recalibration scalar on the held-out validation batch (this seed's model)
        log_var_val = predict_log_var(net, per_sensor_features(val_batch))
        mu_val, V_val, _, _ = structured_fusion_forward(val_batch["obs"], val_batch["mask"], log_var_val)
        tau = fit_recalibration_scale(mu_val, V_val, val_batch["true_pos"])
        tau_all.append(tau)

        ece_g, ece_s, ece_s_recal = [], [], []
        sharp_g, sharp_s = [], []
        for gb, sb in zip(gaussian_batches, shifted_batches):
            lv_g = predict_log_var(net, per_sensor_features(gb))
            mu_g, V_g, _, _ = structured_fusion_forward(gb["obs"], gb["mask"], lv_g)
            _, _, _, ece = nees_calibration(mu_g, V_g, gb["true_pos"])
            ece_g.append(ece)
            sharp_g.append(np.median(np.sqrt(V_g)))

            lv_s = predict_log_var(net, per_sensor_features(sb))
            mu_s, V_s, _, _ = structured_fusion_forward(sb["obs"], sb["mask"], lv_s)
            _, _, _, ece_before = nees_calibration(mu_s, V_s, sb["true_pos"])
            ece_s.append(ece_before)
            sharp_s.append(np.median(np.sqrt(V_s)))

            _, _, _, ece_after = nees_calibration(mu_s, tau * V_s, sb["true_pos"])
            ece_s_recal.append(ece_after)

        ece_gauss_all.append(ece_g)
        ece_shift_all.append(ece_s)
        ece_shift_recal_all.append(ece_s_recal)
        sharp_gauss_all.append(sharp_g)
        sharp_shift_all.append(sharp_s)

    ece_g_mean, ece_g_std = aggregate_seeds(ece_gauss_all)
    ece_s_mean, ece_s_std = aggregate_seeds(ece_shift_all)
    ece_s_recal_mean, ece_s_recal_std = aggregate_seeds(ece_shift_recal_all)
    sharp_g_mean, sharp_g_std = aggregate_seeds(sharp_gauss_all)
    sharp_s_mean, sharp_s_std = aggregate_seeds(sharp_shift_all)
    tau_mean, tau_std = float(np.mean(tau_all)), float(np.std(tau_all))

    print(f"\nRecalibration scale tau: {tau_mean:.3f} ± {tau_std:.3f} "
          f"(fit on validation; {'>1: model underconfident' if tau_mean > 1 else '<1: model overconfident'} under shift)")

    # -----------------------------------------------------------------
    # Figure 1: ECE vs degradation, Gaussian vs shifted, before/after recalibration
    # -----------------------------------------------------------------
    from evaluation.plots import _band
    fig, ax = plt.subplots(figsize=(8, 5.8))
    _band(ax, DEGRADATION_SWEEP, ece_g_mean, ece_g_std, marker="o", color="seagreen", label="Gaussian (in-family)")
    _band(ax, DEGRADATION_SWEEP, ece_s_mean, ece_s_std, marker="o", color="crimson", label="Student-t (shifted family)")
    _band(ax, DEGRADATION_SWEEP, ece_s_recal_mean, ece_s_recal_std, marker="s", color="tab:orange", linestyle="--",
          label=f"Student-t, RECALIBRATED (τ={tau_mean:.2f}, fit on separate validation)")
    ax.axvline(TRAIN_D_RANGE[1], color="gray", linestyle=":", alpha=0.5, label="edge of training range")
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Calibration ECE (NEES-based)")
    ax.set_title("Tier 2 — calibration error across the FULL degradation sweep\n(mean ± std across seeds)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tier2_ece_vs_degradation.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 2: sharpness (mean predicted std) vs degradation
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.8))
    _band(ax, DEGRADATION_SWEEP, sharp_g_mean, sharp_g_std, marker="o", color="seagreen", label="Gaussian (in-family)")
    _band(ax, DEGRADATION_SWEEP, sharp_s_mean, sharp_s_std, marker="o", color="crimson", label="Student-t (shifted family)")
    ax.axvline(TRAIN_D_RANGE[1], color="gray", linestyle=":", alpha=0.5, label="edge of training range")
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Sharpness: median predicted std")
    ax.set_title("Tier 2 — sharpness across the full degradation sweep\n"
                 "(companion to ECE: calibration alone is gameable by trivially wide intervals)", fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tier2_sharpness_vs_degradation.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Numeric summary
    # -----------------------------------------------------------------
    summary = {
        "recalibration_tau_mean": tau_mean, "recalibration_tau_std": tau_std,
        "d_sweep": DEGRADATION_SWEEP.tolist(),
        "ece_gaussian_mean": ece_g_mean.tolist(), "ece_gaussian_std": ece_g_std.tolist(),
        "ece_shifted_mean": ece_s_mean.tolist(), "ece_shifted_std": ece_s_std.tolist(),
        "ece_shifted_recalibrated_mean": ece_s_recal_mean.tolist(), "ece_shifted_recalibrated_std": ece_s_recal_std.tolist(),
        "sharpness_gaussian_mean": sharp_g_mean.tolist(), "sharpness_shifted_mean": sharp_s_mean.tolist(),
        "mean_ece_gaussian": float(np.mean(ece_g_mean)),
        "mean_ece_shifted": float(np.mean(ece_s_mean)),
        "mean_ece_shifted_recalibrated": float(np.mean(ece_s_recal_mean)),
        "recalibration_helps": bool(np.mean(ece_s_recal_mean) < np.mean(ece_s_mean)),
    }
    with open(os.path.join(out_dir, "tier2_calibration_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. 2 figures + tier2_calibration_results.json written to {out_dir}")
    print(f"Mean ECE: gaussian={summary['mean_ece_gaussian']:.4f}  "
          f"shifted={summary['mean_ece_shifted']:.4f}  "
          f"shifted+recal={summary['mean_ece_shifted_recalibrated']:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
