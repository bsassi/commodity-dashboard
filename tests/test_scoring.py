from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import build_summary_table, score_asset


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    close = pd.Series(values, index=idx)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1})


def test_score_asset_bounds_scores() -> None:
    frame = _frame(np.linspace(100, 160, 400))
    row = score_asset("TEST", frame, {"name": "Test", "sector": "Synthetic"}, cross_sectional_score=50)
    assert -100 <= row["Final Composite Score"] <= 100
    assert 0 <= row["Confidence Score"] <= 100
    assert row["Carry"] == "Carry indisponible avec la source de donnees actuelle."


def test_summary_handles_missing_ticker() -> None:
    assets = pd.DataFrame(
        [
            {"ticker": "GOOD", "name": "Good", "sector": "Synthetic"},
            {"ticker": "BAD", "name": "Bad", "sector": "Synthetic"},
        ]
    )
    data = {"GOOD": _frame(np.linspace(100, 120, 300)), "BAD": pd.DataFrame()}
    summary = build_summary_table(assets, data)
    assert len(summary) == 2
    assert summary["Final Composite Score"].between(-100, 100).all()
