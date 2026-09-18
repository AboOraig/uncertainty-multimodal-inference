"""
training/train_torch.py
------------------------
PyTorch training loops (autograd instead of the hand-derived backward
passes in models/mlp_numpy.py + inference/fusion_numpy.py). Same function
signatures/return types as training/train_numpy.py so
experiments/run_experiments.py can select either engine via --engine.
"""

import numpy as np
import torch
import torch.optim as optim

from models.mlp_torch import MLP
from inference.fusion_torch import bound_log_var, nll_loss
from data.simulate import featurize, per_sensor_features, N_SENSORS
from configs.default import HIDDEN_SIZES, EPOCHS, BATCH_SIZE, LEARNING_RATE


def train_learned_fusion(train_batches, seed=0, hidden=HIDDEN_SIZES, epochs=EPOCHS,
                          batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Direct MLP: concatenated multimodal features -> position estimate. MSE loss."""
    torch.manual_seed(seed)
    net = MLP([N_SENSORS * 4, *hidden, 2])
    opt = optim.Adam(net.parameters(), lr=lr)
    history = []

    X = np.concatenate([featurize(b) for b in train_batches], axis=0)
    Y = np.concatenate([b["true_pos"] for b in train_batches], axis=0)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)
    n = X_t.shape[0]

    net.train()
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            bidx = idx[start:start + batch_size]
            Xb, Yb = X_t[bidx], Y_t[bidx]

            pred = net(Xb)
            loss = ((pred - Yb) ** 2).sum(dim=1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item() * len(bidx)
        history.append(ep_loss / n)
    return net, history


def train_uncertainty_model(train_batches, seed=0, hidden=HIDDEN_SIZES, epochs=EPOCHS,
                             batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Per-sensor MLP (shared weights, applied independently to each sensor's
    own [obs, quality, mask]) -> predicted log-variance. Combined via
    structured (precision-weighted) fusion, trained with the Gaussian NLL
    of the fused estimate. See fusion_torch.py.
    """
    torch.manual_seed(seed)
    net = MLP([4, *hidden, 1])
    opt = optim.Adam(net.parameters(), lr=lr)
    history = []

    feats = np.concatenate([per_sensor_features(b) for b in train_batches], axis=0)  # (n,3,4)
    obs = np.concatenate([b["obs"] for b in train_batches], axis=0)
    mask = np.concatenate([b["mask"] for b in train_batches], axis=0)
    true_pos = np.concatenate([b["true_pos"] for b in train_batches], axis=0)

    feats_t = torch.as_tensor(feats, dtype=torch.float32)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    true_pos_t = torch.as_tensor(true_pos, dtype=torch.float32)
    n = feats_t.shape[0]

    net.train()
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            bidx = idx[start:start + batch_size]
            f_b, o_b, m_b, t_b = feats_t[bidx], obs_t[bidx], mask_t[bidx], true_pos_t[bidx]
            nb = f_b.shape[0]

            flat = f_b.reshape(nb * N_SENSORS, 4)
            raw = net(flat)
            log_var = bound_log_var(raw).reshape(nb, N_SENSORS)

            loss, mu, V = nll_loss(o_b, m_b, log_var, t_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item() * nb
        history.append(ep_loss / n)
    return net, history


def predict_log_var(net, feats):
    """feats: (n,3,4) numpy -> (n,3) numpy predicted log-variance (bounded).
    Same interface as training/train_numpy.py's predict_log_var()."""
    n = feats.shape[0]
    flat = feats.reshape(n * N_SENSORS, 4)
    net.eval()
    with torch.no_grad():
        raw = net(torch.as_tensor(flat, dtype=torch.float32))
        log_var = bound_log_var(raw)
    return log_var.numpy().reshape(n, N_SENSORS)
