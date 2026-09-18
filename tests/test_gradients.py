import numpy as np
from inference.fusion_numpy import nll_loss_and_grad
from models.mlp_numpy import MLP

rng = np.random.default_rng(0)

# ---- check 1: analytic dNLL/dlogvar vs finite differences ----
n, s = 5, 3
obs = rng.normal(size=(n, s, 2))
mask = (rng.random((n, s)) > 0.2).astype(float)
mask[:, 0] = 1.0  # ensure at least one sensor always present
true_pos = rng.normal(size=(n, 2))
log_var = rng.normal(size=(n, s))

loss0, dLogVar, mu, V = nll_loss_and_grad(obs, mask, log_var, true_pos)

eps = 1e-6
num_grad = np.zeros_like(log_var)
for i in range(n):
    for j in range(s):
        lv = log_var.copy(); lv[i, j] += eps
        lp, _, _, _ = nll_loss_and_grad(obs, mask, lv, true_pos)
        lv2 = log_var.copy(); lv2[i, j] -= eps
        lm, _, _, _ = nll_loss_and_grad(obs, mask, lv2, true_pos)
        num_grad[i, j] = (lp - lm) / (2 * eps)

err = np.abs(num_grad - dLogVar)
print("NLL grad check  max abs err:", err.max(), " max rel err:",
      (err / (np.abs(num_grad) + 1e-8)).max())
assert err.max() < 1e-5, "NLL gradient mismatch!"

# ---- check 2: MLP backward vs finite differences, MSE loss ----
net = MLP([4, 8, 1], rng)
X = rng.normal(size=(6, 4))
target = rng.normal(size=(6, 1))

pred, cache = net.forward(X)
err_ = pred - target
loss = (err_ ** 2).sum(axis=1).mean()
dOut = 2 * err_
grads = net.backward(cache, dOut)

for key in ["W0", "b0", "W1", "b1"]:
    g_analytic = grads[key]
    g_num = np.zeros_like(net.params[key])
    it = np.nditer(net.params[key], flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = net.params[key][idx]
        net.params[key][idx] = orig + eps
        p1, _ = net.forward(X)
        l1 = ((p1 - target) ** 2).sum(axis=1).mean()
        net.params[key][idx] = orig - eps
        p2, _ = net.forward(X)
        l2 = ((p2 - target) ** 2).sum(axis=1).mean()
        net.params[key][idx] = orig
        g_num[idx] = (l1 - l2) / (2 * eps)
    diff = np.abs(g_num - g_analytic).max()
    print(f"MLP grad check {key}: max abs err = {diff:.3e}")
    assert diff < 1e-5, f"MLP gradient mismatch on {key}"

print("\nAll gradient checks passed.")
