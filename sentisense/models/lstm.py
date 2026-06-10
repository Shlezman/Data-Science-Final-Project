"""LSTM / GRU recurrent classifiers — same head + forward(x)->logit contract as the
transformer zoo, so they drop into train_model()/evaluate_on_test() unchanged.

Search dims exposed for Phase 6 HPO: hidden units, depth, bidirectionality, head
width (d_dense), head dropout, inter-layer (recurrent) dropout, dense activation,
temporal pooling. Lookback (window), batch size, lr, weight decay and grad-clip are
threaded from the HPO objective (see :mod:`sentisense.hpo.optuna_lstm`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh, "gelu": nn.GELU}


def _pool(seq: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce (B, T, H) over time T. mode ∈ {last, avg, max}."""
    if mode == "avg":
        return seq.mean(dim=1)
    if mode == "max":
        return seq.max(dim=1).values
    return seq[:, -1, :]  # last timestep


class _RecurrentClassifier(nn.Module):
    """Shared LSTM/GRU body → temporal pool → dense head → 1 logit."""

    def __init__(self, cell, n_features: int, *, hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.2, recurrent_dropout: float = 0.0,
                 dense_act: str = "relu", pooling: str = "last", d_dense: int = 32,
                 bidirectional: bool = False):
        super().__init__()
        self.pooling = pooling
        self.rnn = cell(
            input_size=n_features, hidden_size=hidden, num_layers=n_layers,
            batch_first=True,
            dropout=(recurrent_dropout if n_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        # Learned attention pooling: a scorer over timesteps → softmax-weighted sum.
        self.attn = nn.Linear(out_dim, 1) if pooling == "attn" else None
        act = _ACT.get(dense_act, nn.ReLU)
        self.head = nn.Sequential(
            nn.Linear(out_dim, d_dense), act(), nn.Dropout(dropout),
            nn.Linear(d_dense, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)            # (B, T, H)
        if self.attn is not None:
            weights = torch.softmax(self.attn(seq), dim=1)   # (B, T, 1)
            pooled = (seq * weights).sum(dim=1)
        else:
            pooled = _pool(seq, self.pooling)
        return self.head(pooled).squeeze(-1)


class LSTMClassifier(_RecurrentClassifier):
    def __init__(self, n_features: int, **kw):
        super().__init__(nn.LSTM, n_features, **kw)


class GRUClassifier(_RecurrentClassifier):
    def __init__(self, n_features: int, **kw):
        super().__init__(nn.GRU, n_features, **kw)


SEQUENCE_ARCHITECTURES = {"LSTM": LSTMClassifier, "GRU": GRUClassifier}
