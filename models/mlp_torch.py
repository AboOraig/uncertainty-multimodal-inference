"""
mlp_torch.py
------------
PyTorch replacement for models/mlp_numpy.py. Same architecture convention
(sizes list -> ReLU hidden layers, linear output), but backprop is now
handled by torch.autograd instead of the hand-derived formulas in the
NumPy version.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sizes):
        """sizes: e.g. [12, 32, 32, 2]. ReLU hidden layers, linear output."""
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict(self, X):
        """X: numpy array -> numpy array. No-grad inference, matches the
        NumPy MLP's .predict() interface so downstream code (experiments/run_experiments.py) is
        unchanged regardless of which engine trained the model."""
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32)
            out = self(x)
        return out.numpy()
