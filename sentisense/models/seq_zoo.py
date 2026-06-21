"""Convolutional + transformer sequence classifiers for the model zoo.

Same ``forward(x: (B, T, F)) -> logit: (B,)`` contract as the recurrent zoo in
:mod:`sentisense.models.lstm`, so they drop into ``train_model`` / ``evaluate_on_test``
and the generic Optuna HPO (:mod:`sentisense.hpo.optuna_seq`) unchanged.

- :class:`TCNClassifier` — dilated causal Temporal Convolutional Network (Bai et al. 2018).
- :class:`PatchTSTClassifier` — channel-independent patched transformer (Nie et al. 2023),
  adapted to emit a single direction logit instead of a forecast head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentisense.models.lstm import _ACT, SEQUENCE_ARCHITECTURES, _pool


class _Chomp(nn.Module):
    """Trim the right padding a causal conv adds, so output length == input length."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp].contiguous() if self.chomp else x


class _TCNBlock(nn.Module):
    """Two dilated causal convs + residual (weight-normed, ReLU, dropout)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)


class TCNClassifier(nn.Module):
    """Dilated causal TCN body → temporal pool → dense head → 1 logit."""

    def __init__(self, n_features: int, *, channels: int = 64, levels: int = 4,
                 kernel_size: int = 3, dropout: float = 0.2, dense_act: str = "relu",
                 pooling: str = "last", d_dense: int = 32):
        super().__init__()
        self.pooling = pooling
        blocks, c_in = [], n_features
        for i in range(levels):
            blocks.append(_TCNBlock(c_in, channels, kernel_size, dilation=2 ** i, dropout=dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        act = _ACT.get(dense_act, nn.ReLU)
        self.head = nn.Sequential(nn.Linear(channels, d_dense), act(), nn.Dropout(dropout),
                                  nn.Linear(d_dense, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))          # (B, F, T) → (B, C, T)
        pooled = _pool(h.transpose(1, 2), self.pooling)   # (B, T, C) → (B, C)
        return self.head(pooled).squeeze(-1)


class PatchTSTClassifier(nn.Module):
    """Channel-independent patched transformer → mean-pool patches → dense head → 1 logit."""

    def __init__(self, n_features: int, *, patch_len: int = 8, stride: int = 4,
                 d_model: int = 64, n_heads: int = 4, depth: int = 2, dropout: float = 0.2,
                 dense_act: str = "relu", d_dense: int = 32, max_patches: int = 64):
        super().__init__()
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        act = _ACT.get(dense_act, nn.ReLU)
        self.head = nn.Sequential(nn.Linear(n_features * d_model, d_dense), act(),
                                  nn.Dropout(dropout), nn.Linear(d_dense, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if t < self.patch_len:                    # pad short windows on the left
            x = F.pad(x.transpose(1, 2), (self.patch_len - t, 0)).transpose(1, 2)
            t = self.patch_len
        xi = x.transpose(1, 2)                                    # (B, F, T)
        patches = xi.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B, F, P, patch_len)
        p = patches.size(2)
        tok = self.embed(patches) + self.pos[:, :p, :].unsqueeze(1)   # (B, F, P, d_model)
        tok = tok.reshape(b * f, p, -1)                          # channel-independent
        enc = self.encoder(tok).mean(dim=1)                      # (B*F, d_model) — pool patches
        return self.head(enc.reshape(b, -1)).squeeze(-1)         # (B, F*d_model) → logit


TCNClassifier.__name__ = "TCNClassifier"
PatchTSTClassifier.__name__ = "PatchTSTClassifier"

# Full sequence-classifier registry (recurrent zoo ⊕ conv/transformer zoo).
ARCHITECTURES = {**SEQUENCE_ARCHITECTURES, "TCN": TCNClassifier, "PatchTST": PatchTSTClassifier}
