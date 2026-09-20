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
                             batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                             feature_fn=per_sensor_features, input_dim=4):
    """
    Per-sensor MLP (shared weights, applied independently to each sensor's
    own features) -> predicted log-variance. The 3 predictions are combined
    via structured (precision-weighted) fusion and trained with the
    Gaussian NLL of the fused estimate (see inference/fusion_numpy.py).

    feature_fn/input_dim let this same training loop serve a different
    per-sensor feature set without duplicating the loop -- e.g. the
    cross-sensor-residual follow-up in data/simulate.py's
    per_sensor_features_v2 (input_dim=5) vs. the core input_dim=4 default.
    """
    net = MLP([input_dim, *hidden, 1], rng)
    opt = Adam(net.params, lr=lr)
    history = []

    feats = np.concatenate([feature_fn(b) for b in train_batches], axis=0)  # (n,3,input_dim)
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

            flat = f_b.reshape(nb * N_SENSORS, input_dim)
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
    """feats: (n, 3, input_dim) -> (n, 3) predicted log-variance (bounded).
    input_dim is inferred from feats itself, so this works unchanged for
    both the core 4-feature model and the 5-feature cross-sensor follow-up."""
    n, s, input_dim = feats.shape
    flat = feats.reshape(n * s, input_dim)
    raw = net.predict(flat)
    log_var, _ = bound_log_var(raw)
    return log_var.reshape(n, s)


# ===========================================================================
# TIER 2 ABLATIONS
# ===========================================================================

def train_uncertainty_model_mse_surrogate(train_batches, rng, hidden=HIDDEN_SIZES, epochs=EPOCHS,
                                           batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                                           feature_fn=per_sensor_features, input_dim=4):
    """
    ABLATION: same architecture and features as the core v1 uncertainty
    model, but a completely different, decoupled loss -- plain MSE (in
    log-space) regressing each sensor's predicted log-variance directly
    against its OWN observed squared error log((obs_i - true_pos)^2), with
    NO coupling through the fusion rule at all (v1's proper NLL loss is
    computed on the FUSED estimate, which lets the network learn what's
    useful for good fusion specifically, not just what matches each
    sensor's marginal error in isolation).

    This tests whether that coupling matters, or whether a much simpler
    two-stage-style surrogate objective gets you the same place.
    """
    net = MLP([input_dim, *hidden, 1], rng)
    opt = Adam(net.params, lr=lr)
    history = []

    feats = np.concatenate([feature_fn(b) for b in train_batches], axis=0)   # (n,3,input_dim)
    obs = np.concatenate([b["obs"] for b in train_batches], axis=0)
    true_pos = np.concatenate([b["true_pos"] for b in train_batches], axis=0)
    sq_err = ((obs - true_pos[:, None, :]) ** 2).sum(axis=2)                  # (n,3) target
    log_target = np.log(sq_err + 1e-6)
    n = feats.shape[0]

    for ep in range(epochs):
        idx = rng.permutation(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            bidx = idx[start:start + batch_size]
            f_b, t_b = feats[bidx], log_target[bidx]
            nb = f_b.shape[0]

            flat = f_b.reshape(nb * N_SENSORS, input_dim)
            raw_flat, cache = net.forward(flat)
            log_var_flat, local_grad_flat = bound_log_var(raw_flat)

            err = log_var_flat.reshape(nb, N_SENSORS) - t_b                    # (nb,3)
            loss = (err ** 2).mean()
            # dLoss/dlogvar = 2*err/(nb*3); chain through bound_log_var same as elsewhere
            dLogVar = (2.0 * err / (nb * N_SENSORS)).reshape(nb * N_SENSORS, 1) * local_grad_flat
            dOut_flat = dLogVar * (nb * N_SENSORS)   # match MLP.backward's internal /N convention
            grads = net.backward(cache, dOut_flat)
            opt.step(net.params, grads)
            ep_loss += loss * nb
        history.append(ep_loss / n)
    return net, history


def train_uncertainty_model_separate_nets(train_batches, rng, hidden=HIDDEN_SIZES, epochs=EPOCHS,
                                           batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                                           feature_fn=per_sensor_features, input_dim=4):
    """
    ABLATION: v1 uses ONE shared-weight MLP applied independently to each
    sensor. This trains 3 SEPARATE MLPs instead (one per sensor index),
    same architecture each, same NLL-through-fusion objective as v1 -- only
    the weight-sharing inductive bias differs. Note this ablation has 3x
    the total parameters of v1 (an inherent property of "no sharing"), so
    if it still doesn't beat v1, that's a stronger endorsement of sharing,
    not a confound in sharing's favor.
    """
    nets = [MLP([input_dim, *hidden, 1], rng) for _ in range(N_SENSORS)]
    opts = [Adam(net.params, lr=lr) for net in nets]
    history = []

    feats = np.concatenate([feature_fn(b) for b in train_batches], axis=0)  # (n,3,input_dim)
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

            log_var = np.zeros((nb, N_SENSORS))
            caches, local_grads = [], []
            for s in range(N_SENSORS):
                raw_s, cache_s = nets[s].forward(f_b[:, s, :])          # (nb,1)
                lv_s, lg_s = bound_log_var(raw_s)
                log_var[:, s] = lv_s[:, 0]
                caches.append(cache_s)
                local_grads.append(lg_s)

            loss, dLogVar, mu, V = nll_loss_and_grad(o_b, m_b, log_var, t_b)  # dLogVar: (nb,3), mean-loss convention

            for s in range(N_SENSORS):
                dOut_s = (dLogVar[:, s:s + 1] * local_grads[s]) * nb   # undo nll's /nb, redo with this net's own N=nb rows
                grads_s = nets[s].backward(caches[s], dOut_s)
                opts[s].step(nets[s].params, grads_s)
            ep_loss += loss * nb
        history.append(ep_loss / n)
    return nets, history


def predict_log_var_separate_nets(nets, feats):
    """feats: (n,3,input_dim) -> (n,3) predicted log-variance (bounded),
    using the 3 separate per-sensor nets from train_uncertainty_model_separate_nets."""
    n = feats.shape[0]
    log_var = np.zeros((n, N_SENSORS))
    for s in range(N_SENSORS):
        raw_s = nets[s].predict(feats[:, s, :])
        lv_s, _ = bound_log_var(raw_s)
        log_var[:, s] = lv_s[:, 0]
    return log_var
