"""
experiments/followup_cross_sensor.py
--------------------------------------
FOLLOW-UP, not part of the core Tier-1 pipeline (see run_experiments.py and
the README's "Tier 1 finding" section for context).

Tier 1's baseline ladder revealed that the v1 uncertainty-aware model
(features = [obs, quality, mask], scored independently per sensor) roughly
TIES the naive quality-weighted-WLS heuristic on both RMSE and
sensor-identification accuracy, rather than clearly beating it. The
diagnosed reason: the quality proxy is already a strong noise estimator by
construction, and the v1 model is architecturally per-sensor-independent,
so it has no way to exploit cross-sensor disagreement -- a cue the raw
quality heuristic can't use either.

This script tests the fix directly: train an otherwise-identical
uncertainty model, but with one extra input feature per sensor -- its
residual magnitude against a leave-one-out consensus of the OTHER observed
sensors (data/simulate.py's per_sensor_features_v2). If that closes (or
reverses) the gap to the quality heuristic, the diagnosis was right and the
fix works. If not, that's an equally honest result worth reporting.

Uses the exact same fixed evaluation protocol and training seeds as
run_experiments.py, so results are directly comparable to the Tier-1 numbers.

Usage:
    python -m experiments.followup_cross_sensor --engine numpy
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.simulate import per_sensor_features, per_sensor_features_v2, N_SENSORS
from inference.fusion_numpy import structured_fusion_forward
from evaluation.metrics import rmse, sensor_identification_accuracy, aggregate_seeds
from evaluation.plots import _band
from configs.default import TRAIN_SEEDS, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH

from experiments.run_experiments import build_eval_protocol, evaluate_nonlearned_baselines, get_engine
from data.simulate import generate_dataset


def run_one_seed_v2(engine_name, seed, sweep_batches):
    _, train_uncertainty_model, predict_log_var, seed_kw = get_engine(engine_name)

    train_rng = np.random.default_rng(seed)
    train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, train_rng)
                      for _ in range(N_TRAIN_BATCHES)]
    kwargs = {"rng": train_rng} if seed_kw == "rng" else {"seed": seed}

    unc_net_v2, hist_v2 = train_uncertainty_model(
        train_batches, feature_fn=per_sensor_features_v2, input_dim=5, **kwargs)

    rmse_v2, sensorid_v2 = [], []
    for b in sweep_batches:
        log_var = predict_log_var(unc_net_v2, per_sensor_features_v2(b))
        mu, V, w, S = structured_fusion_forward(b["obs"], b["mask"], log_var)
        rmse_v2.append(rmse(mu, b["true_pos"]))
        acc, _ = sensor_identification_accuracy(log_var, b["degraded_sensor"], b["mask"], b["d"])
        sensorid_v2.append(acc)
    return rmse_v2, sensorid_v2, hist_v2[-1]


def main(engine_name):
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                              "results", engine_name, "followup_cross_sensor"))
    os.makedirs(out_dir, exist_ok=True)

    print("Rebuilding the SAME fixed evaluation protocol as run_experiments.py...")
    _, sweep_batches, _, _ = build_eval_protocol()
    baselines = evaluate_nonlearned_baselines(sweep_batches)

    # Load Tier-1 v1 results for direct comparison (must have been run already)
    tier1_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                                 "results", engine_name, "results_summary.json"))
    with open(tier1_path) as f:
        tier1 = json.load(f)

    rmse_v2_all, sensorid_v2_all, final_losses = [], [], []
    for seed in TRAIN_SEEDS:
        print(f"[{engine_name}] cross-sensor follow-up, seed {seed}...")
        r_v2, s_v2, loss = run_one_seed_v2(engine_name, seed, sweep_batches)
        rmse_v2_all.append(r_v2)
        sensorid_v2_all.append(s_v2)
        final_losses.append(loss)
        print(f"  final NLL={loss:.4f}  mean RMSE={np.mean(r_v2):.4f}  mean sensor-ID={np.nanmean(s_v2):.3f}")

    rmse_v2_mean, rmse_v2_std = aggregate_seeds(rmse_v2_all)
    sensorid_v2_mean, sensorid_v2_std = aggregate_seeds(sensorid_v2_all)

    d_sweep = np.array(tier1["d_sweep"])
    rmse_v1_mean = np.array(tier1["rmse_uncertainty_aware"]["mean"])
    rmse_v1_std = np.array(tier1["rmse_uncertainty_aware"]["std"])
    sensorid_v1_mean = np.array(tier1["sensor_identification"]["learned_mean_over_d"])
    sensorid_v1_std = np.array(tier1["sensor_identification"]["learned_std_over_d"])
    rmse_quality = np.array(baselines["rmse_quality_wls"])
    rmse_oracle = np.array(baselines["rmse_oracle_wls"])
    sensorid_quality = np.array(baselines["sensorid_quality_heuristic"])

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(d_sweep, rmse_oracle, ":", color="black", label="Oracle WLS (ceiling)")
    ax.plot(d_sweep, rmse_quality, "D-", color="tab:green", label="Quality-weighted WLS (heuristic)")
    _band(ax, d_sweep, rmse_v1_mean, rmse_v1_std, marker="o", color="crimson",
          label="v1: uncertainty-aware (obs+quality+mask only)")
    _band(ax, d_sweep, rmse_v2_mean, rmse_v2_std, marker="s", color="tab:purple",
          label="v2: + cross-sensor residual feature (follow-up)")
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Estimation RMSE")
    ax.set_title("Follow-up — does a cross-sensor residual feature beat the quality heuristic?\n"
                 "(mean ± std across seeds)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "followup_rmse_comparison.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(d_sweep, sensorid_quality, "D-", color="tab:green", label="Quality argmax (heuristic)")
    _band(ax, d_sweep, sensorid_v1_mean, sensorid_v1_std, marker="o", color="crimson",
          label="v1 argmax (obs+quality+mask only)")
    _band(ax, d_sweep, sensorid_v2_mean, sensorid_v2_std, marker="s", color="tab:purple",
          label="v2 argmax (+ cross-sensor residual)")
    ax.axhline(1.0 / N_SENSORS, color="gray", linestyle="--", label="chance level")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Sensor-identification accuracy")
    ax.set_title("Follow-up — sensor identification: v1 vs v2 vs heuristic\n"
                 "(mean ± std across seeds)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "followup_sensorid_comparison.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------
    mean_rmse_v1 = float(np.mean(rmse_v1_mean))
    mean_rmse_v2 = float(np.mean(rmse_v2_mean))
    mean_rmse_quality = float(np.mean(rmse_quality))
    mean_sid_v1 = float(np.nanmean(sensorid_v1_mean))
    mean_sid_v2 = float(np.nanmean(sensorid_v2_mean))
    mean_sid_quality = float(np.nanmean(sensorid_quality))

    verdict = {
        "mean_rmse": {"v1": mean_rmse_v1, "v2": mean_rmse_v2, "quality_heuristic": mean_rmse_quality,
                       "oracle_ceiling": float(np.mean(rmse_oracle))},
        "mean_sensorid_accuracy": {"v1": mean_sid_v1, "v2": mean_sid_v2, "quality_heuristic": mean_sid_quality},
        "v2_beats_quality_heuristic_on_rmse": bool(mean_rmse_v2 < mean_rmse_quality),
        "v2_beats_v1_on_rmse": bool(mean_rmse_v2 < mean_rmse_v1),
        "v2_beats_quality_heuristic_on_sensorid": bool(mean_sid_v2 > mean_sid_quality),
    }
    with open(os.path.join(out_dir, "followup_results.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\nDone. Figures + followup_results.json written to {out_dir}")
    print(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
