"""
models/mlp_numpy.py
------
A tiny, dependency-free MLP with manual forward/backward passes, used for
both the "learned fusion" model and the "learned uncertainty" model.

No autodiff framework was available in this environment (no disk space
left for a PyTorch install), so gradients are derived and implemented by
hand. For a plain feedforward net trained with MSE this is standard
backprop. The uncertainty model's loss (Gaussian NLL evaluated on a
precision-weighted fusion of several sensors) has its gradient derived
analytically in inference/fusion_numpy.py -- see the derivation in the README.
"""

import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_activation=None):
        """sizes: e.g. [12, 32, 32, 2]. ReLU hidden layers, linear output
        (out_activation optionally applied, e.g. 'linear')."""
        self.sizes = sizes
        self.params = {}
        for l in range(len(sizes) - 1):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            # He init (ReLU-friendly)
            self.params[f"W{l}"] = rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            self.params[f"b{l}"] = np.zeros(fan_out)
        self.n_layers = len(sizes) - 1

    def forward(self, X):
        cache = {"A0": X}
        A = X
        for l in range(self.n_layers):
            Z = A @ self.params[f"W{l}"] + self.params[f"b{l}"]
            cache[f"Z{l}"] = Z
            if l < self.n_layers - 1:
                A = np.maximum(Z, 0.0)  # ReLU
            else:
                A = Z  # linear output
            cache[f"A{l+1}"] = A
        return A, cache

    def backward(self, cache, dOut):
        """dOut: gradient of loss w.r.t. network output, shape (N, out_dim)."""
        grads = {}
        dA = dOut
        N = dA.shape[0]
        for l in reversed(range(self.n_layers)):
            Z = cache[f"Z{l}"]
            if l < self.n_layers - 1:
                dZ = dA * (Z > 0)
            else:
                dZ = dA
            A_prev = cache[f"A{l}"]
            grads[f"W{l}"] = (A_prev.T @ dZ) / N
            grads[f"b{l}"] = dZ.mean(axis=0)
            dA = dZ @ self.params[f"W{l}"].T
        return grads

    def predict(self, X):
        out, _ = self.forward(X)
        return out


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
