import numpy as np
from models.mlp_numpy import MLP, Adam
from inference.fusion_numpy import nll_loss_and_grad
from data.simulate import featurize, per_sensor_features, N_SENSORS
from inference.bounds import bound_log_var
from configs.default import HIDDEN_SIZES, EPOCHS, BATCH_SIZE, LEARNING_RATE


def train_learned_fusion(train_batches, rng, hidden=HIDDEN_SIZES, epochs=EPOCHS,
                          batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Direct MLP: concatenated multimodal features -> position estimate. MSE loss."""
    net = MLP([N_SENSORS * 4, *hidden, 2], rng)
    opt = Adam(net.params, lr=lr)
    history = []

    X = np.concatenate([featurize(b) for b in train_batches], axis=0)
    Y = np.concatenate([b["true_pos"] for b in train_batches], axis=0)
    n = X.shape[0]

    for ep in range(epochs):
        idx = rng.permutation(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            bidx = idx[start:start + batch_size]
            Xb, Yb = X[bidx], Y[bidx]
            pred, cache = net.forward(Xb)
            err = pred - Yb
            loss = (err ** 2).sum(axis=1).mean()
            dOut = 2 * err  # backward() divides by batch size internally
            grads = net.backward(cache, dOut)
            opt.step(net.params, grads)
            ep_loss += loss * len(bidx)
        history.append(ep_loss / n)
    return net, history


def train_uncertainty_model(train_batches, rng, hidden=HIDDEN_SIZES, epochs=EPOCHS,
                             batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Per-sensor MLP (shared weights, applied independently to each sensor's
    own [obs, quality, mask]) -> predicted log-variance. The 3 predictions
    are combined via structured (precision-weighted) fusion and trained
    with the Gaussian NLL of the fused estimate (see inference/fusion_numpy.py).
    """
    net = MLP([4, *hidden, 1], rng)
    opt = Adam(net.params, lr=lr)
    history = []

    feats = np.concatenate([per_sensor_features(b) for b in train_batches], axis=0)  # (n,3,4)
    obs = np.concatenate([b["obs"] for b in train_batches], axis=0)
    mask = np.concatenate([b["mask"] for b in train_batches], axis=0)
    true_pos = np.concatenate([b["true_pos"] for b in train_batches], axis=0)
    n = feats.shape[0]

    for ep in range(epochs):
        idx = rng.permutation(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            bidx = idx[start:start + batch_size]
            f_b, o_b, m_b, t_b = feats[bidx], obs[bidx], mask[bidx], true_pos[bidx]
            nb = f_b.shape[0]

            flat = f_b.reshape(nb * N_SENSORS, 4)
            raw_flat, cache = net.forward(flat)
            log_var_flat, local_grad_flat = bound_log_var(raw_flat)
            log_var = log_var_flat.reshape(nb, N_SENSORS)

            loss, dLogVar, mu, V = nll_loss_and_grad(o_b, m_b, log_var, t_b)
            # chain through the bounded-output transform: dLoss/draw = dLoss/dlogvar * dlogvar/draw
            dLogVar = dLogVar.reshape(nb * N_SENSORS, 1) * local_grad_flat
            dLogVar = dLogVar.reshape(nb, N_SENSORS)
            # dLogVar = dMeanLoss/dlogvar, averaged over nb SAMPLES.
            # net.backward() internally divides by the number of ROWS it was
            # given, which here is nb*N_SENSORS (one row per sensor per
            # sample, since the same MLP is applied independently to each
            # sensor). Rescale so the two averaging conventions match:
            #   grads_backward = (1/(nb*S)) * sum_rows outer(A, dOut_row)
            # should equal (1/nb) * sum_rows outer(A, d(loss)/d(logvar_row))
            # => dOut_row = N_SENSORS * nb * dLogVar_row
            dOut_flat = dLogVar.reshape(nb * N_SENSORS, 1) * (nb * N_SENSORS)
            grads = net.backward(cache, dOut_flat)
            opt.step(net.params, grads)
            ep_loss += loss * nb
        history.append(ep_loss / n)
    return net, history


def predict_log_var(net, feats):
    """feats: (n, 3, 4) -> (n, 3) predicted log-variance (bounded)."""
    n = feats.shape[0]
    flat = feats.reshape(n * N_SENSORS, 4)
    raw = net.predict(flat)
    log_var, _ = bound_log_var(raw)
    return log_var.reshape(n, N_SENSORS)
