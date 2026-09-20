"""
experiments/followup_robust_fusion.py
----------------------------------------
FOLLOW-UP to the follow-up. Motivated by the diagnosis in
followup_cross_sensor.py's results (README, "Follow-up: does a cross-sensor
feature fix it?"): every LINEAR (one-shot, inverse-variance) fusion method
sits at a persistent RMSE gap above the oracle, suggesting the fusion RULE
itself -- soft, proportional downweighting -- may be the bottleneck against
roughly-bimodal bias faults, not the uncertainty estimate.

This tests that directly with classical robust M-estimation (Huber and
Tukey-biweight IRLS, inference/robust_fusion_numpy.py), which can achieve
genuine hard gating (Tukey: exactly zero weight past a threshold). The
robustness threshold `c` is tuned on a SEPARATE VALIDATION set (never the
actual fixed evaluation protocol) to avoid overfitting the baseline to the
test data -- see tune_c_on_validation() below.

Tests IRLS layered on top of TWO prior variance sources:
  - the quality heuristic (var = quality^2) -- purely classical, no learning
  - the learned v1 uncertainty model's exp(log_var) -- combining learning
    with classical robust refinement

Usage:
    python -m experiments.followup_robust_fusion --engine numpy
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.simulate import generate_dataset, per_sensor_features
from inference.fusion_numpy import quality_weighted_wls
from inference.robust_fusion_numpy import irls_fusion, tune_c
from evaluation.metrics import rmse, sensor_identification_accuracy, aggregate_seeds
from evaluation.plots import _band
from configs.default import TRAIN_SEEDS, TRAIN_D_RANGE, N_TRAIN_BATCHES, N_TRAIN_PER_BATCH

from experiments.run_experiments import build_eval_protocol, evaluate_nonlearned_baselines, get_engine

VALIDATION_SEED = 999_999   # separate from EVAL_SEED and all TRAIN_SEEDS
C_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0]


def build_validation_set():
    """Held-out validation data, used ONLY for tuning c -- never touches the
    actual fixed evaluation protocol used for reported numbers."""
    val_rng = np.random.default_rng(VALIDATION_SEED)
    return [generate_dataset(3000, float(d), val_rng) for d in [0.5, 1.0, 1.5]]


def _concat(batches, key):
    return np.concatenate([b[key] for b in batches], axis=0)


def tune_c_on_validation(val_batches, prior_var, psi_name):
    obs = _concat(val_batches, "obs")
    mask = _concat(val_batches, "mask")
    true_pos = _concat(val_batches, "true_pos")
    best_c, best_rmse, _ = tune_c(obs, mask, true_pos, prior_var, psi_name, C_GRID)
    return best_c, best_rmse


def main(engine_name):
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                              "results", engine_name, "followup_robust_fusion"))
    os.makedirs(out_dir, exist_ok=True)

    print("Rebuilding the SAME fixed evaluation protocol as run_experiments.py...")
    _, sweep_batches, _, _ = build_eval_protocol()
    baselines = evaluate_nonlearned_baselines(sweep_batches)

    tier1_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                                 "results", engine_name, "results_summary.json"))
    with open(tier1_path) as f:
        tier1 = json.load(f)

    print("Building validation set and tuning c for quality-prior IRLS (Huber, Tukey)...")
    val_batches = build_validation_set()
    val_quality = _concat(val_batches, "quality")
    c_huber_quality, val_rmse_huber_q = tune_c_on_validation(val_batches, val_quality ** 2, "huber")
    c_tukey_quality, val_rmse_tukey_q = tune_c_on_validation(val_batches, val_quality ** 2, "tukey")
    val_baseline_rmse = rmse(quality_weighted_wls(_concat(val_batches, "obs"), _concat(val_batches, "mask"),
                                                    val_quality), _concat(val_batches, "true_pos"))
    print(f"  validation quality-WLS baseline RMSE: {val_baseline_rmse:.4f}")
    print(f"  tuned c: huber={c_huber_quality} (val rmse {val_rmse_huber_q:.4f}), "
          f"tukey={c_tukey_quality} (val rmse {val_rmse_tukey_q:.4f})")

    # ---- quality-prior IRLS on the FIXED eval protocol (deterministic, no seed variance) ----
    rmse_huber_q, rmse_tukey_q = [], []
    sensorid_huber_q, sensorid_tukey_q = [], []
    for b in sweep_batches:
        mu_h, w_h = irls_fusion(b["obs"], b["mask"], b["quality"] ** 2, psi_name="huber", c=c_huber_quality)
        rmse_huber_q.append(rmse(mu_h, b["true_pos"]))
        acc_h, _ = sensor_identification_accuracy(-w_h, b["degraded_sensor"], b["mask"], b["d"])
        sensorid_huber_q.append(acc_h)

        mu_t, w_t = irls_fusion(b["obs"], b["mask"], b["quality"] ** 2, psi_name="tukey", c=c_tukey_quality)
        rmse_tukey_q.append(rmse(mu_t, b["true_pos"]))
        acc_t, _ = sensor_identification_accuracy(-w_t, b["degraded_sensor"], b["mask"], b["d"])
        sensorid_tukey_q.append(acc_t)

    # ---- learned-prior (v1 model) IRLS: retrain v1 per seed, tune c once (seed 0), apply to all ----
    print("Retraining v1 uncertainty model per seed for the learned-prior IRLS variant...")
    _, train_uncertainty_model, predict_log_var, seed_kw = get_engine(engine_name)

    def train_v1(seed):
        train_rng = np.random.default_rng(seed)
        train_batches = [generate_dataset(N_TRAIN_PER_BATCH, TRAIN_D_RANGE, train_rng)
                          for _ in range(N_TRAIN_BATCHES)]
        kwargs = {"rng": train_rng} if seed_kw == "rng" else {"seed": seed}
        net, _ = train_uncertainty_model(train_batches, **kwargs)
        return net

    net0 = train_v1(TRAIN_SEEDS[0])
    val_logvar0 = predict_log_var(net0, per_sensor_features(val_batches[0]))
    # tune using seed-0 model, pooled validation batches' predicted variance
    val_prior0 = np.concatenate([np.exp(predict_log_var(net0, per_sensor_features(b))) for b in val_batches], axis=0)
    c_huber_learned, _ = tune_c_on_validation(val_batches, val_prior0, "huber")
    c_tukey_learned, _ = tune_c_on_validation(val_batches, val_prior0, "tukey")
    print(f"  tuned c (learned prior, from seed {TRAIN_SEEDS[0]}): "
          f"huber={c_huber_learned}, tukey={c_tukey_learned}")

    rmse_huber_l_all, rmse_tukey_l_all = [], []
    sensorid_huber_l_all, sensorid_tukey_l_all = [], []
    for seed in TRAIN_SEEDS:
        print(f"[{engine_name}] learned-prior IRLS, seed {seed}...")
        net = net0 if seed == TRAIN_SEEDS[0] else train_v1(seed)
        rmse_huber_l, rmse_tukey_l = [], []
        sensorid_huber_l, sensorid_tukey_l = [], []
        for b in sweep_batches:
            prior_var = np.exp(predict_log_var(net, per_sensor_features(b)))
            mu_h, w_h = irls_fusion(b["obs"], b["mask"], prior_var, psi_name="huber", c=c_huber_learned)
            rmse_huber_l.append(rmse(mu_h, b["true_pos"]))
            acc_h, _ = sensor_identification_accuracy(-w_h, b["degraded_sensor"], b["mask"], b["d"])
            sensorid_huber_l.append(acc_h)

            mu_t, w_t = irls_fusion(b["obs"], b["mask"], prior_var, psi_name="tukey", c=c_tukey_learned)
            rmse_tukey_l.append(rmse(mu_t, b["true_pos"]))
            acc_t, _ = sensor_identification_accuracy(-w_t, b["degraded_sensor"], b["mask"], b["d"])
            sensorid_tukey_l.append(acc_t)
        rmse_huber_l_all.append(rmse_huber_l)
        rmse_tukey_l_all.append(rmse_tukey_l)
        sensorid_huber_l_all.append(sensorid_huber_l)
        sensorid_tukey_l_all.append(sensorid_tukey_l)

    rmse_huber_l_mean, rmse_huber_l_std = aggregate_seeds(rmse_huber_l_all)
    rmse_tukey_l_mean, rmse_tukey_l_std = aggregate_seeds(rmse_tukey_l_all)

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    d_sweep = np.array(tier1["d_sweep"])
    rmse_v1_mean = np.array(tier1["rmse_uncertainty_aware"]["mean"])
    rmse_v1_std = np.array(tier1["rmse_uncertainty_aware"]["std"])
    rmse_quality = np.array(baselines["rmse_quality_wls"])
    rmse_oracle = np.array(baselines["rmse_oracle_wls"])

    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.plot(d_sweep, rmse_oracle, ":", color="black", label="Oracle WLS (ceiling)")
    ax.plot(d_sweep, rmse_quality, "D-", color="tab:green", label="Quality-weighted WLS (baseline)")
    ax.plot(d_sweep, rmse_huber_q, "^--", color="tab:blue",
            label=f"Quality + Huber IRLS (c={c_huber_quality}, tuned)")
    ax.plot(d_sweep, rmse_tukey_q, "v--", color="tab:cyan",
            label=f"Quality + Tukey IRLS (c={c_tukey_quality}, tuned)")
    _band(ax, d_sweep, rmse_v1_mean, rmse_v1_std, marker="o", color="crimson",
          label="v1 uncertainty-aware (no robust refinement)")
    _band(ax, d_sweep, rmse_huber_l_mean, rmse_huber_l_std, marker="^", color="tab:orange",
          label=f"v1 + Huber IRLS (c={c_huber_learned}, tuned)")
    _band(ax, d_sweep, rmse_tukey_l_mean, rmse_tukey_l_std, marker="v", color="tab:purple",
          label=f"v1 + Tukey IRLS (c={c_tukey_learned}, tuned)")
    ax.set_xlabel("Degradation level d")
    ax.set_ylabel("Estimation RMSE")
    ax.set_title("Follow-up 2 — does robust (Huber/Tukey) IRLS beat linear precision weighting?\n"
                 "(c tuned on a separate validation set; mean ± std across seeds)", fontsize=10.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "followup_robust_rmse_comparison.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------
    verdict = {
        "tuned_c": {"huber_quality": c_huber_quality, "tukey_quality": c_tukey_quality,
                    "huber_learned": c_huber_learned, "tukey_learned": c_tukey_learned},
        "validation_baseline_rmse": float(val_baseline_rmse),
        "mean_rmse": {
            "oracle_ceiling": float(np.mean(rmse_oracle)),
            "quality_heuristic": float(np.mean(rmse_quality)),
            "quality_plus_huber": float(np.mean(rmse_huber_q)),
            "quality_plus_tukey": float(np.mean(rmse_tukey_q)),
            "v1_learned": float(np.mean(rmse_v1_mean)),
            "v1_plus_huber": float(np.mean(rmse_huber_l_mean)),
            "v1_plus_tukey": float(np.mean(rmse_tukey_l_mean)),
        },
        "any_robust_variant_beats_quality_heuristic": bool(
            min(np.mean(rmse_huber_q), np.mean(rmse_tukey_q),
                np.mean(rmse_huber_l_mean), np.mean(rmse_tukey_l_mean)) < np.mean(rmse_quality)
        ),
        "mean_sensorid_accuracy": {
            "quality_heuristic": float(np.nanmean(baselines["sensorid_quality_heuristic"])),
            "quality_plus_huber": float(np.nanmean(sensorid_huber_q)),
            "quality_plus_tukey": float(np.nanmean(sensorid_tukey_q)),
            "v1_plus_huber_seed0": float(np.nanmean(sensorid_huber_l_all[0])),
            "v1_plus_tukey_seed0": float(np.nanmean(sensorid_tukey_l_all[0])),
        },
    }
    with open(os.path.join(out_dir, "followup_robust_results.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\nDone. Figure + followup_robust_results.json written to {out_dir}")
    print(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    main(args.engine)
