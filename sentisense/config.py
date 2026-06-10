"""Modeling/runtime knobs (env-overridable). Cutoff + score contract live in constants.py."""

from __future__ import annotations

import os


def _int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


# Feature shaping
TOP_N_SOURCES: int = _int("SENTISENSE_TOP_N_SOURCES", 12)

# Sequence model
WINDOW_SIZE: int = _int("SENTISENSE_WINDOW_SIZE", 30)
BATCH_SIZE: int = _int("SENTISENSE_BATCH_SIZE", 64)
MAX_EPOCHS: int = _int("SENTISENSE_MAX_EPOCHS", 200)
PATIENCE: int = _int("SENTISENSE_PATIENCE", 15)
LR: float = _float("SENTISENSE_LR", 1e-4)
VAL_FRAC: float = _float("SENTISENSE_VAL_FRAC", 0.15)
TEST_FRAC: float = _float("SENTISENSE_TEST_FRAC", 0.15)

# HPO budget
OPTUNA_TRIALS: int = _int("SENTISENSE_OPTUNA_TRIALS", 100)
HPO_SEEDS: tuple[int, ...] = (42, 123, 2024)  # >=3 seeds for ablation mean±std
# Below this many trading days, cap LSTM capacity (small-data overfit guard — Gate B).
LSTM_VIABILITY_MIN_DAYS: int = _int("SENTISENSE_MIN_TRADING_DAYS", 750)

# Embeddings (Phase 4) — Hebrew-aware multilingual model.
EMBED_MODEL: str = os.environ.get("SENTISENSE_EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_BATCH: int = _int("SENTISENSE_EMBED_BATCH", 128)
# Clustering refit cadence (days) for the expanding-window MiniBatchKMeans.
CLUSTER_K: int = _int("SENTISENSE_CLUSTER_K", 8)
CLUSTER_REFIT_EVERY: int = _int("SENTISENSE_CLUSTER_REFIT_EVERY", 30)

SEED: int = _int("SENTISENSE_SEED", 42)
