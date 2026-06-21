"""Phase 5/6 models — sequence datasets, training harness, LSTM/GRU, baselines.

Ported from transformer_forecaster.ipynb (chronological split, train-only scaler,
AMP training, class-weighted BCE, early stopping) so the LSTM is evaluated on
exactly the same harness/metrics as the existing transformer zoo.
"""
