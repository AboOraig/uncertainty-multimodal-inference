"""
experiments/run_experiments.py
--------------------------------
Main entry point. For the chosen engine, runs TRAIN_SEEDS independent
training runs (fresh data draw + fresh model init each time), evaluates
every one of them on a SINGLE FIXED evaluation protocol (built once from
EVAL_SEED, shared across all training seeds), and aggregates mean ± std
across seeds for every headline number and curve. The non-learned WLS
baselines (fixed / quality-weighted / oracle) are closed-form and have zero
seed variance by construction, so they're computed once, outside the seed
loop.

This isolates "is this a lucky training run" from genuine effect size --
see configs/default.py for TRAIN_SEEDS / EVAL_SEED.

Usage (from the repo root):
    python -m experiments.run_experiments --engine numpy
    python -m experiments.run_experiments --engine torch
"""

import argparse
import json
import os

import numpy as np

from data.simulate import generate_dataset, featurize, per_sensor_features, N_SENSORS
from inference.fusion_numpy import fixed_wls, quality_weighted_wls, oracle_wls, structured_fusion_forward
from evaluation.metrics import (
    rmse, uncertainty_error_correlation, nees_calibration,
    sensor_identification_accuracy, aggregate_seeds,
)
from evaluation.plots import (
    plot_graph1_uncertainty_vs_error, plot_bonus_expB,
    plot_graph2_rmse_vs_degradation, plot_graph3_calibration_shift,
    plot_sensor_identification_accuracy,
)
from configs.default import (
    BASE_STD, TRAIN_SEEDS, EVAL_SEED, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH,
    DEGRADATION_SWEEP, N_EVAL_PER_SWEEP_POINT, PROBE_D_RANGE, PROBE_N,
    D_FIXED_FOR_SHIFT, N_CALIBRATION_SAMPLES,
    SHIFT_NOISE_FAMILY, SHIFT_QUALITY_NOISE_COEF, SHIFT_BIAS_GAIN,
)


def get_engine(name):
    if name == "numpy":
        from training.train_numpy import train_learned_fusion, train_uncertainty_model, predict_log_var
        return train_learned_fusion, train_uncertainty_model, predict_log_var, "rng"
    elif name == "torch":
        from training.train_torch import train_learned_fusion, train_uncertainty_model, predict_log_var
        return train_learned_fusion, train_uncertainty_model, predict_log_var, "seed"
    raise ValueError(f"unknown engine: {name}")


# ---------------------------------------------------------------------------
# Fixed evaluation protocol -- generated once, reused identically across
# every training seed. The bonus Exp. B curve and the new sensor-ID curve
# both reuse the same degradation-sweep batches as Graph 2 (they need the
# same per-d evaluation data), rather than drawing yet another dataset.
# ---------------------------------------------------------------------------
def build_eval_protocol():
    eval_rng = np.random.default_rng(EVAL_SEED)
    probe = generate_dataset(PROBE_N, PROBE_D_RANGE, eval_rng)
    sweep_batches = [generate_dataset(N_EVAL_PER_SWEEP_POINT, float(d), eval_rng)
                      for d in DEGRADATION_SWEEP]
    id_batch = generate_dataset(N_CALIBRATION_SAMPLES, D_FIXED_FOR_SHIFT, eval_rng,
                                 noise_family="gaussian")
    shift_batch = generate_dataset(N_CALIBRATION_SAMPLES, D_FIXED_FOR_SHIFT, eval_rng,
                                    noise_family=SHIFT_NOISE_FAMILY,
                                    quality_noise_coef=SHIFT_QUALITY_NOISE_COEF,
                                    bias_gain=SHIFT_BIAS_GAIN)
    return probe, sweep_batches, id_batch, shift_batch


def evaluate_nonlearned_baselines(sweep_batches):
    """Closed-form WLS baselines: no training, so no seed variance. Computed
    once against the fixed sweep batches."""
    nominal_var = BASE_STD ** 2
    rmse_fixed, rmse_quality, rmse_oracle, sensorid_quality = [], [], [], []
    for b in sweep_batches:
        rmse_fixed.append(rmse(fixed_wls(b["obs"], b["mask"], nominal_var), b["true_pos"]))
        rmse_quality.append(rmse(quality_weighted_wls(b["obs"], b["mask"], b["quality"]), b["true_pos"]))
        rmse_oracle.append(rmse(oracle_wls(b["obs"], b["mask"], b["oracle_var"]), b["true_pos"]))
        acc, _ = sensor_identification_accuracy(b["quality"], b["degraded_sensor"], b["mask"], b["d"])
        sensorid_quality.append(acc)
    return {
        "rmse_fixed_wls": rmse_fixed,
        "rmse_quality_wls": rmse_quality,
        "rmse_oracle_wls": rmse_oracle,
        "sensorid_quality_heuristic": sensorid_quality,
    }


def run_one_seed(engine_name, seed, eval_protocol):
    probe, sweep_batches, id_batch, shift_batch = eval_protocol
    train_learned_fusion, train_uncertainty_model, predict_log_var, seed_kw = get_engine(engine_name)

    train_rng = np.random.default_rng(seed)
    train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, train_rng)
                      for _ in range(N_TRAIN_BATCHES)]
    kwargs = {"rng": train_rng} if seed_kw == "rng" else {"seed": seed}

    fusion_net, fusion_hist = train_learned_fusion(train_batches, **kwargs)
    unc_net, unc_hist = train_uncertainty_model(train_batches, **kwargs)

    # separate, deterministic-per-seed stream for the shuffle ablation, kept
    # independent of the training rng so it doesn't perturb training
    shuffle_rng = np.random.default_rng(seed + 1_000_000)

    def run_uncertainty_aware(batch, shuffle_ablation=False):
        feats = per_sensor_features(batch)
        log_var = predict_log_var(unc_net, feats)
        if shuffle_ablation:
            log_var = log_var.copy()
            for s in range(N_SENSORS):
                log_var[:, s] = shuffle_rng.permutation(log_var[:, s])
        mu, V, w, S = structured_fusion_forward(batch["obs"], batch["mask"], log_var)
        return mu, V, log_var

    out = {"fusion_final_loss": fusion_hist[-1], "unc_final_loss": unc_hist[-1]}

    # ---- Experiment A ----
    log_var_probe = predict_log_var(unc_net, per_sensor_features(probe))
    pred_std = np.sqrt(np.exp(log_var_probe))
    actual_err = np.linalg.norm(probe["obs"] - probe["true_pos"][:, None, :], axis=2)
    observed = probe["mask"] > 0.5
    out["experiment_A"] = uncertainty_error_correlation(pred_std[observed], actual_err[observed])
    out["_probe_pred_std_flat"] = pred_std[observed]     # kept only for seed 0, for the Graph 1 scatter
    out["_probe_actual_err_flat"] = actual_err[observed]

    # ---- Graph 2 sweep (learned methods) + bonus Exp. B + sensor-ID (learned) ----
    rmse_lf, rmse_ua, rmse_shuf, sensorid_learned = [], [], [], []
    mean_pred_std_vs_d = []
    for b in sweep_batches:
        rmse_lf.append(rmse(fusion_net.predict(featurize(b)), b["true_pos"]))
        mu_ua, V_ua, lv_ua = run_uncertainty_aware(b)
        rmse_ua.append(rmse(mu_ua, b["true_pos"]))
        mu_shuf, _, _ = run_uncertainty_aware(b, shuffle_ablation=True)
        rmse_shuf.append(rmse(mu_shuf, b["true_pos"]))
        mean_pred_std_vs_d.append(np.sqrt(np.exp(lv_ua)).mean(axis=0))
        acc, _ = sensor_identification_accuracy(lv_ua, b["degraded_sensor"], b["mask"], b["d"])
        sensorid_learned.append(acc)

    out["rmse_learned_fusion"] = rmse_lf
    out["rmse_uncertainty_aware"] = rmse_ua
    out["rmse_uncertainty_shuffled"] = rmse_shuf
    out["mean_pred_std_vs_d"] = mean_pred_std_vs_d
    out["sensorid_learned"] = sensorid_learned

    # ---- Experiment D calibration ----
    mu_id, V_id, _ = run_uncertainty_aware(id_batch)
    nom_p, nom_emp, _, ece_id = nees_calibration(mu_id, V_id, id_batch["true_pos"])
    mu_shift, V_shift, _ = run_uncertainty_aware(shift_batch)
    shift_p, shift_emp, _, ece_shift = nees_calibration(mu_shift, V_shift, shift_batch["true_pos"])
    out["calibration"] = {"nom_p": nom_p, "nom_emp": nom_emp, "ece_id": ece_id,
                           "shift_p": shift_p, "shift_emp": shift_emp, "ece_shift": ece_shift}
    return out


def main(engine_name):
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", engine_name))
    os.makedirs(out_dir, exist_ok=True)

    print("Building fixed evaluation protocol (shared across all seeds)...")
    eval_protocol = build_eval_protocol()
    _, sweep_batches, _, _ = eval_protocol

    print("Evaluating non-learned baselines (deterministic, zero seed variance)...")
    baselines = evaluate_nonlearned_baselines(sweep_batches)

    seed_results = []
    for seed in TRAIN_SEEDS:
        print(f"[{engine_name}] === seed {seed} ({len(seed_results) + 1}/{len(TRAIN_SEEDS)}) ===")
        r = run_one_seed(engine_name, seed, eval_protocol)
        seed_results.append(r)
        print(f"  train loss: fusion={r['fusion_final_loss']:.4f} unc={r['unc_final_loss']:.4f}  |  "
              f"Exp A: r={r['experiment_A']['pearson_r']:.3f}  |  "
              f"ECE: id={r['calibration']['ece_id']:.3f} shift={r['calibration']['ece_shift']:.3f}")

    # -----------------------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------------------
    pearson_mean, pearson_std = aggregate_seeds([[r["experiment_A"]["pearson_r"]] for r in seed_results])
    spearman_mean, spearman_std = aggregate_seeds([[r["experiment_A"]["spearman_rho"]] for r in seed_results])
    corr_mean = {"pearson_r": pearson_mean[0], "spearman_rho": spearman_mean[0]}
    corr_std = {"pearson_r": pearson_std[0], "spearman_rho": spearman_std[0]}

    rmse_lf_mean, rmse_lf_std = aggregate_seeds([r["rmse_learned_fusion"] for r in seed_results])
    rmse_ua_mean, rmse_ua_std = aggregate_seeds([r["rmse_uncertainty_aware"] for r in seed_results])
    rmse_shuf_mean, rmse_shuf_std = aggregate_seeds([r["rmse_uncertainty_shuffled"] for r in seed_results])
    stdvsd_mean, stdvsd_std = aggregate_seeds([r["mean_pred_std_vs_d"] for r in seed_results])
    sensorid_mean, sensorid_std = aggregate_seeds([r["sensorid_learned"] for r in seed_results])

    nom_p = seed_results[0]["calibration"]["nom_p"]      # confidence grid is identical every seed
    shift_p = seed_results[0]["calibration"]["shift_p"]
    nom_emp_mean, nom_emp_std = aggregate_seeds([r["calibration"]["nom_emp"] for r in seed_results])
    shift_emp_mean, shift_emp_std = aggregate_seeds([r["calibration"]["shift_emp"] for r in seed_results])
    ece_id_mean, ece_id_std = aggregate_seeds([[r["calibration"]["ece_id"]] for r in seed_results])
    ece_shift_mean, ece_shift_std = aggregate_seeds([[r["calibration"]["ece_shift"]] for r in seed_results])

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    plot_graph1_uncertainty_vs_error(
        seed_results[0]["_probe_pred_std_flat"], seed_results[0]["_probe_actual_err_flat"],
        corr_mean, corr_std, TRAIN_SEEDS[0],
        os.path.join(out_dir, "graph1_uncertainty_vs_error.png"))

    plot_bonus_expB(DEGRADATION_SWEEP, stdvsd_mean, stdvsd_std, TRAIN_D_RANGE[1],
                     os.path.join(out_dir, "bonus_expB_uncertainty_vs_noise.png"))

    plot_graph2_rmse_vs_degradation(
        DEGRADATION_SWEEP,
        baselines["rmse_fixed_wls"], baselines["rmse_quality_wls"], baselines["rmse_oracle_wls"],
        rmse_lf_mean, rmse_lf_std, rmse_shuf_mean, rmse_shuf_std, rmse_ua_mean, rmse_ua_std,
        TRAIN_D_RANGE[1], os.path.join(out_dir, "graph2_rmse_vs_degradation.png"))

    plot_graph3_calibration_shift(
        nom_p, nom_emp_mean, nom_emp_std, ece_id_mean[0], ece_id_std[0],
        shift_p, shift_emp_mean, shift_emp_std, ece_shift_mean[0], ece_shift_std[0],
        D_FIXED_FOR_SHIFT, os.path.join(out_dir, "graph3_calibration_shift.png"))

    plot_sensor_identification_accuracy(
        DEGRADATION_SWEEP, sensorid_mean, sensorid_std, baselines["sensorid_quality_heuristic"],
        N_SENSORS, TRAIN_D_RANGE[1], os.path.join(out_dir, "graph4_sensor_identification.png"))

    # -----------------------------------------------------------------
    # Numeric summary (aggregate + full per-seed raw values, for transparency)
    # -----------------------------------------------------------------
    summary = {
        "engine": engine_name,
        "train_seeds": TRAIN_SEEDS,
        "eval_seed": EVAL_SEED,
        "experiment_A": {"mean": corr_mean, "std": corr_std,
                          "per_seed": [r["experiment_A"] for r in seed_results]},
        "baseline_ladder_mean_rmse": {
            "oracle_wls_ceiling": float(np.mean(baselines["rmse_oracle_wls"])),
            "uncertainty_aware_proposed": float(np.mean(rmse_ua_mean)),
            "learned_fusion_no_uncertainty": float(np.mean(rmse_lf_mean)),
            "quality_weighted_wls_heuristic": float(np.mean(baselines["rmse_quality_wls"])),
            "fixed_wls_baseline": float(np.mean(baselines["rmse_fixed_wls"])),
            "uncertainty_aware_SHUFFLED_ablation": float(np.mean(rmse_shuf_mean)),
        },
        "experiment_D_calibration": {
            "ece_in_distribution": {"mean": float(ece_id_mean[0]), "std": float(ece_id_std[0])},
            "ece_shifted": {"mean": float(ece_shift_mean[0]), "std": float(ece_shift_std[0])},
        },
        "sensor_identification": {
            "learned_mean_over_d": sensorid_mean.tolist(),
            "learned_std_over_d": sensorid_std.tolist(),
            "quality_heuristic_over_d": baselines["sensorid_quality_heuristic"],
            "chance_level": 1.0 / N_SENSORS,
        },
        "d_sweep": DEGRADATION_SWEEP.tolist(),
        "rmse_fixed_wls": baselines["rmse_fixed_wls"],
        "rmse_quality_wls": baselines["rmse_quality_wls"],
        "rmse_oracle_wls": baselines["rmse_oracle_wls"],
        "rmse_learned_fusion": {"mean": rmse_lf_mean.tolist(), "std": rmse_lf_std.tolist(),
                                 "per_seed": [r["rmse_learned_fusion"] for r in seed_results]},
        "rmse_uncertainty_aware": {"mean": rmse_ua_mean.tolist(), "std": rmse_ua_std.tolist(),
                                    "per_seed": [r["rmse_uncertainty_aware"] for r in seed_results]},
        "rmse_uncertainty_shuffled": {"mean": rmse_shuf_mean.tolist(), "std": rmse_shuf_std.tolist(),
                                       "per_seed": [r["rmse_uncertainty_shuffled"] for r in seed_results]},
    }
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. 4 figures + results_summary.json written to {out_dir}")
    print(json.dumps(summary["baseline_ladder_mean_rmse"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
