"""
experiments/run_experiments.py
--------------------------------
Main entry point. Trains both learned models with the chosen engine,
then runs Experiments A-D and writes the 3 required figures
and a results_summary.json.

Usage (from the repo root):
    python -m experiments.run_experiments --engine numpy
    python -m experiments.run_experiments --engine torch

Both engines implement the exact same simulation, fusion rules, and
experiment logic -- only the two learned models' training/inference code
differs (hand-derived NumPy backprop vs. PyTorch autograd). Results should
be qualitatively identical between engines; exact numbers will differ due
to different weight initialization and optimizer internals.
"""

import argparse
import json
import os

import numpy as np

from .data.simulate import generate_dataset, featurize, per_sensor_features, N_SENSORS
from inference.fusion_numpy import fixed_wls, structured_fusion_forward
from evaluation.metrics import rmse, uncertainty_error_correlation, nees_calibration
from evaluation.plots import (
    plot_graph1_uncertainty_vs_error, plot_bonus_expB,
    plot_graph2_rmse_vs_degradation, plot_graph3_calibration_shift,
)
from configs.default import (
    BASE_STD, SEED, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH,
    DEGRADATION_SWEEP, N_EVAL_PER_SWEEP_POINT, PROBE_D_RANGE, PROBE_N,
    EXP_B_D_GRID, EXP_B_N_PER_POINT, D_FIXED_FOR_SHIFT, N_CALIBRATION_SAMPLES,
    SHIFT_NOISE_FAMILY, SHIFT_QUALITY_NOISE_COEF, SHIFT_BIAS_GAIN,
)


def get_engine(name):
    """Returns (train_learned_fusion, train_uncertainty_model, predict_log_var,
    train_seed_kwarg) for the requested engine. Both engines expose the
    identical function signatures downstream."""
    if name == "numpy":
        from training.train_numpy import train_learned_fusion, train_uncertainty_model, predict_log_var
        rng = np.random.default_rng(SEED)
        return train_learned_fusion, train_uncertainty_model, predict_log_var, {"rng": rng}, rng
    elif name == "torch":
        from training.train_torch import train_learned_fusion, train_uncertainty_model, predict_log_var
        rng = np.random.default_rng(SEED)  # still used for data generation / ablation shuffling
        return train_learned_fusion, train_uncertainty_model, predict_log_var, {"seed": SEED}, rng
    else:
        raise ValueError(f"unknown engine: {name}")


def main(engine_name):
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", engine_name)
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_learned_fusion, train_uncertainty_model, predict_log_var, train_kwargs, rng = get_engine(engine_name)

    # -----------------------------------------------------------------
    # Training data
    # -----------------------------------------------------------------
    train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, rng)
                      for _ in range(N_TRAIN_BATCHES)]

    print(f"[{engine_name}] Training learned-fusion model (direct MSE)...")
    fusion_net, fusion_hist = train_learned_fusion(train_batches, **train_kwargs)
    print(f"  final train MSE: {fusion_hist[-1]:.4f}")

    print(f"[{engine_name}] Training uncertainty-aware model (structured NLL)...")
    unc_net, unc_hist = train_uncertainty_model(train_batches, **train_kwargs)
    print(f"  final train NLL: {unc_hist[-1]:.4f}")

    # Fixed WLS baseline: nominal (spec-sheet, d=0) variances, never adapted
    nominal_var = BASE_STD ** 2

    def run_uncertainty_aware(batch, shuffle_ablation=False, rng_local=None):
        feats = per_sensor_features(batch)
        log_var = predict_log_var(unc_net, feats)
        if shuffle_ablation:
            log_var = log_var.copy()
            for s in range(N_SENSORS):
                log_var[:, s] = rng_local.permutation(log_var[:, s])
        mu, V, w, S = structured_fusion_forward(batch["obs"], batch["mask"], log_var)
        return mu, V, log_var

    # -----------------------------------------------------------------
    # Experiment A + B
    # -----------------------------------------------------------------
    print("Running Experiment A/B...")
    probe = generate_dataset(PROBE_N, PROBE_D_RANGE, rng)
    log_var_probe = predict_log_var(unc_net, per_sensor_features(probe))
    pred_std = np.sqrt(np.exp(log_var_probe))
    actual_err = np.linalg.norm(probe["obs"] - probe["true_pos"][:, None, :], axis=2)
    observed = probe["mask"] > 0.5

    pred_std_flat = pred_std[observed]
    actual_err_flat = actual_err[observed]
    corr = uncertainty_error_correlation(pred_std_flat, actual_err_flat)
    print(f"  Experiment A: Pearson r={corr['pearson_r']:.3f}, Spearman rho={corr['spearman_rho']:.3f}")

    mean_pred_std_vs_d = np.zeros((len(EXP_B_D_GRID), N_SENSORS))
    for k, dval in enumerate(EXP_B_D_GRID):
        b = generate_dataset(EXP_B_N_PER_POINT, float(dval), rng)
        lv = predict_log_var(unc_net, per_sensor_features(b))
        mean_pred_std_vs_d[k] = np.sqrt(np.exp(lv)).mean(axis=0)

    # -----------------------------------------------------------------
    # Graph 2 + Experiment C ablation
    # -----------------------------------------------------------------
    print("Running Experiment C / Graph 2 sweep...")
    rmse_fixed_wls, rmse_learned_fusion, rmse_unc_aware, rmse_unc_shuffled = [], [], [], []
    for dval in DEGRADATION_SWEEP:
        b = generate_dataset(N_EVAL_PER_SWEEP_POINT, float(dval), rng)
        rmse_fixed_wls.append(rmse(fixed_wls(b["obs"], b["mask"], nominal_var), b["true_pos"]))
        rmse_learned_fusion.append(rmse(fusion_net.predict(featurize(b)), b["true_pos"]))
        mu_ua, _, _ = run_uncertainty_aware(b)
        rmse_unc_aware.append(rmse(mu_ua, b["true_pos"]))
        mu_shuf, _, _ = run_uncertainty_aware(b, shuffle_ablation=True, rng_local=rng)
        rmse_unc_shuffled.append(rmse(mu_shuf, b["true_pos"]))

    # -----------------------------------------------------------------
    # Experiment D + Graph 3
    # -----------------------------------------------------------------
    print("Running Experiment D / Graph 3...")
    id_batch = generate_dataset(N_CALIBRATION_SAMPLES, D_FIXED_FOR_SHIFT, rng,
                                 noise_family="gaussian")
    shift_batch = generate_dataset(N_CALIBRATION_SAMPLES, D_FIXED_FOR_SHIFT, rng,
                                    noise_family=SHIFT_NOISE_FAMILY,
                                    quality_noise_coef=SHIFT_QUALITY_NOISE_COEF,
                                    bias_gain=SHIFT_BIAS_GAIN)

    mu_id, V_id, _ = run_uncertainty_aware(id_batch)
    nom_p, nom_emp, _, ece_id = nees_calibration(mu_id, V_id, id_batch["true_pos"])
    mu_shift, V_shift, _ = run_uncertainty_aware(shift_batch)
    shift_p, shift_emp, _, ece_shift = nees_calibration(mu_shift, V_shift, shift_batch["true_pos"])
    print(f"  Calibration ECE  in-distribution={ece_id:.3f}   shifted={ece_shift:.3f}")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    plot_graph1_uncertainty_vs_error(pred_std_flat, actual_err_flat, corr,
                                      os.path.join(out_dir, "graph1_uncertainty_vs_error.png"))
    plot_bonus_expB(EXP_B_D_GRID, mean_pred_std_vs_d, TRAIN_D_RANGE[1],
                     os.path.join(out_dir, "graph1b_uncertainty_vs_noise.png"))
    plot_graph2_rmse_vs_degradation(DEGRADATION_SWEEP, rmse_fixed_wls, rmse_learned_fusion,
                                     rmse_unc_aware, rmse_unc_shuffled, TRAIN_D_RANGE[1],
                                     os.path.join(out_dir, "graph2_rmse_vs_degradation.png"))
    plot_graph3_calibration_shift(nom_p, nom_emp, ece_id, shift_p, shift_emp, ece_shift,
                                   D_FIXED_FOR_SHIFT,
                                   os.path.join(out_dir, "graph3_calibration_shift.png"))

    # -----------------------------------------------------------------
    # Numeric summary
    # -----------------------------------------------------------------
    summary = {
        "engine": engine_name,
        "experiment_A": corr,
        "experiment_C_ablation": {
            "mean_rmse_uncertainty_aware": float(np.mean(rmse_unc_aware)),
            "mean_rmse_shuffled_ablation": float(np.mean(rmse_unc_shuffled)),
            "mean_rmse_fixed_wls": float(np.mean(rmse_fixed_wls)),
            "mean_rmse_learned_fusion": float(np.mean(rmse_learned_fusion)),
        },
        "experiment_D_calibration": {"ece_in_distribution": ece_id, "ece_shifted": ece_shift},
        "d_sweep": DEGRADATION_SWEEP.tolist(),
        "rmse_fixed_wls": rmse_fixed_wls,
        "rmse_learned_fusion": rmse_learned_fusion,
        "rmse_uncertainty_aware": rmse_unc_aware,
        "rmse_uncertainty_shuffled": rmse_unc_shuffled,
    }
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Figures + results_summary.json written to {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
