"""
tests/test_torch_pipeline.py
------------------------------
A fast smoke test for the PyTorch engine: tiny data, 3 epochs, just checks
that the whole pipeline runs shape-correctly and that the loss decreases.
This was NOT run by the assistant that wrote it (no PyTorch / disk space
in that sandbox) -- run this FIRST on your machine before trusting the
full experiments/run_experiments.py --engine torch run.

    python -m tests.test_torch_pipeline
"""

import numpy as np
from data.simulate import generate_dataset
from training.train_torch import train_learned_fusion, train_uncertainty_model, predict_log_var
from data.simulate import per_sensor_features
from inference.fusion_numpy import structured_fusion_forward
from evaluation.metrics import rmse

rng = np.random.default_rng(0)
batches = [generate_dataset(200, (0.0, 0.6), rng) for _ in range(2)]  # tiny, fast

print("Training learned-fusion model (3 epochs, tiny data)...")
fusion_net, fusion_hist = train_learned_fusion(batches, seed=0, epochs=3, batch_size=64)
print("  loss per epoch:", [round(x, 4) for x in fusion_hist])
assert fusion_hist[-1] < fusion_hist[0], "learned-fusion loss did not decrease -- something is wrong"

print("Training uncertainty-aware model (3 epochs, tiny data)...")
unc_net, unc_hist = train_uncertainty_model(batches, seed=0, epochs=3, batch_size=64)
print("  loss per epoch:", [round(x, 4) for x in unc_hist])
assert unc_hist[-1] < unc_hist[0], "uncertainty-model NLL did not decrease -- something is wrong"

print("Checking shapes end-to-end...")
probe = generate_dataset(50, 0.3, rng)
from data.simulate import featurize
pred = fusion_net.predict(featurize(probe))
assert pred.shape == (50, 2), f"unexpected learned-fusion output shape: {pred.shape}"

log_var = predict_log_var(unc_net, per_sensor_features(probe))
assert log_var.shape == (50, 3), f"unexpected log_var shape: {log_var.shape}"
mu, V, w, S = structured_fusion_forward(probe["obs"], probe["mask"], log_var)
assert mu.shape == (50, 2) and V.shape == (50,), "unexpected fused output shapes"
print("  fused estimate RMSE (tiny/undertrained model, expect it to be mediocre):",
      round(rmse(mu, probe["true_pos"]), 3))

print("\nAll torch pipeline smoke checks passed. Safe to run the full "
      "experiments/run_experiments.py --engine torch.")
