"""
experiments/tier2_ablations.py
---------------------------------
Three new ablations, each isolating one design choice in the v1
uncertainty model against the identical fixed evaluation protocol used
throughout this project:

  A. no_quality   -- drop the quality feature entirely ([obs, mask] only).
                     Tests how much of v1's performance is quality doing
                     all the work (a lot, per the Tier-1 finding already).
  B. mse_surrogate -- same features, but trained with a decoupled per-sensor
                     MSE loss (regressing log(squared error) directly)
                     instead of the proper NLL-through-fusion objective.
                     Tests whether that coupling to the fusion rule matters.
  C. separate_nets -- 3 independent (unshared) per-sensor networks instead
                     of one shared-weight network, same NLL objective.
                     Tests the weight-sharing inductive bias (note: 3x the
                     parameters of v1, an inherent property of "no sharing").

Usage:
    python -m experiments.tier2_ablations --engine numpy
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.simulate import (
    generate_dataset, per_sensor_features, per_sensor_features_no_quality, N_SENSORS,
)
from inference.fusion_numpy import structured_fusion_forward
from evaluation.metrics import rmse, uncertainty_error_correlation, sensor_identification_accuracy, aggregate_seeds
from evaluation.plots import _band
from configs.default import TRAIN_SEEDS, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH

from experiments.run_experiments import build_eval_protocol, evaluate_nonlearned_baselines


def run_ablation_A_no_quality(seed, sweep_batches, probe):
    from training.train_numpy import train_uncertainty_model, predict_log_var
    rng = np.random.default_rng(seed)
    train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, rng) for _ in range(N_TRAIN_BATCHES)]
    net, hist = train_uncertainty_model(train_batches, rng, feature_fn=per_sensor_features_no_quality, input_dim=3)
    return _evaluate(lambda b: predict_log_var(net, per_sensor_features_no_quality(b)), sweep_batches, probe, hist)


def run_ablation_B_mse_surrogate(seed, sweep_batches, probe):
    from training.train_numpy import train_uncertainty_model_mse_surrogate, predict_log_var
    rng = np.random.default_rng(seed)
    train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, rng) for _ in range(N_TRAIN_BATCHES)]
    net, hist = train_uncertainty_model_mse_surrogate(train_batches, rng)
    return _evaluate(lambda b: predict_log_var(net, per_sensor_features(b)), sweep_batches, probe, hist)


def run_ablation_C_separate_nets(seed, sweep_batches, probe):
    from training.train_numpy import train_uncertainty_model_separate_nets, predict_log_var_separate_nets
    rng = np.random.default_rng(seed)
    train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, rng) for _ in range(N_TRAIN_BATCHES)]
    nets, hist = train_uncertainty_model_separate_nets(train_batches, rng)
    return _evaluate(lambda b: predict_log_var_separate_nets(nets, per_sensor_features(b)), sweep_batches, probe, hist)


def _evaluate(logvar_fn, sweep_batches, probe, hist):
    """Shared evaluation: RMSE + sensor-ID vs d, plus Experiment-A correlation on the probe set."""
    rmse_list, sensorid_list = [], []
    for b in sweep_batches:
        log_var = logvar_fn(b)
        mu, V, w, S = structured_fusion_forward(b["obs"], b["mask"], log_var)
        rmse_list.append(rmse(mu, b["true_pos"]))
        acc, _ = sensor_identification_accuracy(log_var, b["degraded_sensor"], b["mask"], b["d"])
        sensorid_list.append(acc)

    log_var_probe = logvar_fn(probe)
    pred_std = np.sqrt(np.exp(log_var_probe))
    actual_err = np.linalg.norm(probe["obs"] - probe["true_pos"][:, None, :], axis=2)
    observed = probe["mask"] > 0.5
    corr = uncertainty_error_correlation(pred_std[observed], actual_err[observed])

    return {"rmse": rmse_list, "sensorid": sensorid_list, "corr": corr, "final_loss": hist[-1]}


ABLATIONS = {
    "A_no_quality": run_ablation_A_no_quality,
    "B_mse_surrogate": run_ablation_B_mse_surrogate,
    "C_separate_nets": run_ablation_C_separate_nets,
}


def main(engine_name):
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", engine_name, "tier2_ablations"))
    os.makedirs(out_dir, exist_ok=True)

    print("Rebuilding the SAME fixed evaluation protocol as run_experiments.py...")
    probe, sweep_batches, _, _ = build_eval_protocol()
    baselines = evaluate_nonlearned_baselines(sweep_batches)

    tier1_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", engine_name, "results_summary.json"))
    with open(tier1_path) as f:
        tier1 = json.load(f)

    all_results = {}
    for name, fn in ABLATIONS.items():
        print(f"\n=== Ablation {name} ===")
        per_seed = []
        for seed in TRAIN_SEEDS:
            r = fn(seed, sweep_batches, probe)
            per_seed.append(r)
            print(f"  seed {seed}: final_loss={r['final_loss']:.4f}  mean RMSE={np.mean(r['rmse']):.4f}  "
                  f"Pearson r={r['corr']['pearson_r']:.3f}  mean sensor-ID={np.nanmean(r['sensorid']):.3f}")
        rmse_mean, rmse_std = aggregate_seeds([r["rmse"] for r in per_seed])
        sensorid_mean, sensorid_std = aggregate_seeds([r["sensorid"] for r in per_seed])
        pearson_mean, pearson_std = aggregate_seeds([[r["corr"]["pearson_r"]] for r in per_seed])
        all_results[name] = {
            "rmse_mean": rmse_mean, "rmse_std": rmse_std,
            "sensorid_mean": sensorid_mean, "sensorid_std": sensorid_std,
            "pearson_mean": float(pearson_mean[0]), "pearson_std": float(pearson_std[0]),
        }

    # -----------------------------------------------------------------
    # Figure: RMSE vs d, v1 + baselines + all 3 ablations
    # -----------------------------------------------------------------
    d_sweep = np.array(tier1["d_sweep"])
    rmse_v1_mean = np.array(tier1["rmse_uncertainty_aware"]["mean"])
    rmse_v1_std = np.array(tier1["rmse_uncertainty_aware"]["std"])
    rmse_quality = np.array(baselines["rmse_quality_wls"])
    rmse_oracle = np.array(baselines["rmse_oracle_wls"])
    rmse_fixed = np.array(baselines["rmse_fixed_wls"])

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(d_sweep, rmse_oracle, ":", color="black", label="Oracle WLS (ceiling)")
    ax.plot(d_sweep, rmse_quality, "D-", color="tab:green", label="Quality-weighted WLS")
    ax.plot(d_sweep, rmse_fixed, "s-", color="tab:blue", alpha=0.6, label="Fixed WLS")
    _band(ax, d_sweep, rmse_v1_mean, rmse_v1_std, marker="o", color="crimson", linewidth=2, label="v1 (full model)")
    colors = {"A_no_quality": "tab:orange", "B_mse_surrogate": "tab:purple", "C_separate_nets": "tab:brown"}
    labels = {"A_no_quality": "Ablation A: no quality feature",
              "B_mse_surrogate": "Ablation B: MSE surrogate loss",
              "C_separate_nets": "Ablation C: separate (unshared) nets"}
    for name in ABLATIONS:
        r = all_results[name]
        _band(ax, d_sweep, r["rmse_mean"], r["rmse_std"], marker="^", color=colors[name], label=labels[name])
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Estimation RMSE")
    ax.set_title("Tier 2 ablations — quality feature, NLL objective, weight sharing\n(mean ± std across seeds)",
                 fontsize=11)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tier2_ablations_rmse.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Numeric summary
    # -----------------------------------------------------------------
    summary = {
        "v1_reference": {"mean_rmse": float(np.mean(rmse_v1_mean)),
                          "pearson_r": tier1["experiment_A"]["mean"]["pearson_r"]},
        "quality_heuristic_mean_rmse": float(np.mean(rmse_quality)),
        "oracle_mean_rmse": float(np.mean(rmse_oracle)),
        "fixed_wls_mean_rmse": float(np.mean(rmse_fixed)),
        "ablations": {
            name: {
                "mean_rmse": float(np.mean(all_results[name]["rmse_mean"])),
                "mean_sensorid": float(np.nanmean(all_results[name]["sensorid_mean"])),
                "pearson_r": all_results[name]["pearson_mean"],
                "pearson_r_std": all_results[name]["pearson_std"],
            } for name in ABLATIONS
        },
    }
    with open(os.path.join(out_dir, "tier2_ablations_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Figure + tier2_ablations_results.json written to {out_dir}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
