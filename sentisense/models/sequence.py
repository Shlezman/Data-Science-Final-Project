"""Windowed sequence datasets + leakage-safe chronological split (port of nb cell 14).

Scaler is fit on TRAIN only; windows never straddle a split boundary; loaders keep
chronological order (shuffle=False). Adds array-returning split helpers used by the
TimeSeriesSplit HPO so the same scaling discipline applies inside CV folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sentisense.config import BATCH_SIZE, TEST_FRAC, VAL_FRAC, WINDOW_SIZE
# Torch-free split lives in features.dataset; re-exported here for callers that
# already import from models.sequence (e.g. the HPO).
from sentisense.features.dataset import chronological_split  # noqa: F401


class SequenceDataset(Dataset):
    """Sliding-window dataset; label aligns to the LAST timestep of each window."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, window: int):
        self.window = window
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.X) - self.window)

    def __getitem__(self, idx: int):
        return self.X[idx:idx + self.window], self.y[idx + self.window - 1]


def windowed_loader(X: np.ndarray, y: np.ndarray, window: int, *,
                    batch_size: int = BATCH_SIZE, shuffle: bool = False,
                    drop_last: bool = False) -> DataLoader:
    """Build a chronological windowed DataLoader from scaled arrays."""
    return DataLoader(SequenceDataset(X, y, window), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


def prepare_data(df: pd.DataFrame, *, window: int = WINDOW_SIZE,
                 val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC):
    """Chronological split → train-only scale → windowed DataLoaders (nb cell 14)."""
    X_tr, y_tr, X_va, y_va, X_te, y_te, n_features = chronological_split(
        df, val_frac=val_frac, test_frac=test_frac)
    dl_tr = windowed_loader(X_tr, y_tr, window, shuffle=False, drop_last=True)
    dl_va = windowed_loader(X_va, y_va, window, shuffle=False)
    dl_te = windowed_loader(X_te, y_te, window, shuffle=False)
    return dl_tr, dl_va, dl_te, n_features


def compute_class_weights(dl: DataLoader) -> torch.Tensor:
    """Inverse-frequency [neg_w, pos_w] from the windows a loader actually yields."""
    n_pos = n_total = 0
    for _, y in dl:
        n_pos += y.sum().item()
        n_total += y.size(0)
    n_neg = n_total - n_pos
    # Guard the degenerate single-class case to avoid div-by-zero.
    n_neg = max(n_neg, 1)
    n_pos = max(n_pos, 1)
    return torch.tensor([n_total / (2 * n_neg), n_total / (2 * n_pos)], dtype=torch.float32)
