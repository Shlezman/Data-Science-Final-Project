"""Phase 3 feature engineering — leakage-safe daily dataset assembly.

Ports the proven machinery from ``transformer_forecaster.ipynb`` (the locked
build-reuse decision): per-source SUM pivot + daily MEAN aggregation, the TA-125
Sun-Thu trading-calendar rollover (Fri/Sat news → next Sunday), the finance/market
merge, ``add_ta125_features`` (leak-free lagged returns / RSI / vol-z / DoW), the
close-to-close next-day target, and the hard ``<= 2023-10-07`` cutoff.
"""

from sentisense.features.dataset import (
    build_datasets,
    build_embedding_dataset,
    build_fused_dataset,
    build_sim_features,
)

__all__ = ["build_datasets", "build_embedding_dataset", "build_fused_dataset",
           "build_sim_features"]
