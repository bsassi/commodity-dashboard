from __future__ import annotations

import numpy as np
import pandas as pd

from src.drawdown import drawdown_series, drawdown_snapshot
from src.volatility import expected_shortfall, historical_var, realized_volatility


def test_realized_volatility_annualizes() -> None:
    returns = pd.Series([0.01, -0.01] * 40, index=pd.date_range("2024-01-01", periods=80))
    vol = realized_volatility(returns, window=20)
    assert vol.dropna().iloc[-1] > 0
    assert np.isclose(vol.dropna().iloc[-1], returns.tail(20).std() * np.sqrt(252))


def test_drawdown_snapshot() -> None:
    price = pd.Series([100, 120, 90, 100, 130], index=pd.date_range("2024-01-01", periods=5))
    snap = drawdown_snapshot(price)
    assert snap["Maximum Drawdown"] < 0
    assert np.isclose(drawdown_series(price.pct_change(fill_method=None)).min(), -0.25)


def test_var_and_expected_shortfall_are_tail_losses() -> None:
    returns = pd.Series([-0.10, -0.05, 0.0, 0.02, 0.04])
    assert historical_var(returns, 0.80) <= 0
    assert expected_shortfall(returns, 0.80) <= historical_var(returns, 0.80)
