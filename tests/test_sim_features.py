"""Sim feature pivot (per-mode → wide + cross-mode signals); pure, no DB."""

from __future__ import annotations

import pandas as pd

from sentisense.features.dataset import _pivot_sim_long


def _long(rows):
    cols = ["date", "mode", "dir_score", "confidence", "disagreement", "n_agents", "seeds"]
    return pd.DataFrame(rows, columns=cols)


def test_pivot_emits_per_mode_blocks_and_cross_mode_signals():
    out = _pivot_sim_long(_long([
        ("2024-03-10", "source", 0.5, 0.8, 0.2, 10, 2),
        ("2024-03-10", "flat", -0.3, 0.6, 0.3, 5, 1),
    ]))
    d = pd.Timestamp("2024-03-10")
    assert out.loc[d, "sim_source_dir_score"] == 0.5
    assert out.loc[d, "sim_flat_dir_score"] == -0.3
    assert out.loc[d, "sim_source_confidence"] == 0.8 and out.loc[d, "sim_flat_seeds"] == 1
    # cross-mode: source bull (+) vs flat bear (-) → gap 0.8, disagree (agree=0)
    assert round(out.loc[d, "sim_src_flat_gap"], 6) == 0.8
    assert out.loc[d, "sim_src_flat_agree"] == 0.0
    assert all(c.startswith("sim_") for c in out.columns)   # prefix → picked up downstream


def test_pivot_single_mode_has_no_cross_mode_cols():
    out = _pivot_sim_long(_long([("2024-03-10", "source", 0.4, 0.7, 0.1, 8, 1)]))
    assert "sim_source_dir_score" in out.columns
    assert "sim_src_flat_gap" not in out.columns   # only emitted when both modes present
