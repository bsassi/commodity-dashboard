from __future__ import annotations

import numpy as np
import pandas as pd

from src.seasonality import calendar_month_stats, current_month_seasonality


def test_calendar_month_stats_returns_evidence_columns() -> None:
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    price = pd.Series(100.0, index=dates)
    for idx, date in enumerate(dates):
        monthly_return = 0.03 if date.month == 1 else 0.002
        price.iloc[idx] = price.iloc[idx - 1] * (1 + monthly_return) if idx else 100
    stats = calendar_month_stats(price, bootstrap_iterations=50)
    assert {"Month", "Hit Rate", "P-Value", "Evidence", "Seasonality Score"}.issubset(stats.columns)
    assert stats.loc[stats["Month"] == 1, "Observations"].iloc[0] >= 10


def test_current_month_seasonality_is_bounded() -> None:
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    price = pd.Series(1.01 ** np.arange(len(dates)), index=dates)
    result = current_month_seasonality(price)
    assert -100 <= result["Seasonality Score"] <= 100
