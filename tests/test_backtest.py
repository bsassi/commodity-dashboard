from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import apply_signal_without_lookahead, backtest_single_asset, performance_metrics


def test_signal_is_shifted_before_return_application() -> None:
    price = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-01", periods=3))
    signal = pd.Series([0.0, 1.0, 0.0], index=price.index)
    result = apply_signal_without_lookahead(price, signal)
    assert result["position"].iloc[1] == 0.0
    assert result["strategy_return"].iloc[1] == 0.0
    assert np.isclose(result["strategy_return"].iloc[2], 0.10)


def test_backtest_single_asset_outputs_required_columns() -> None:
    idx = pd.date_range("2020-01-01", periods=320, freq="B")
    close = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    frame = pd.DataFrame({"Close": close, "High": close, "Low": close, "Open": close, "Volume": 1})
    result = backtest_single_asset(frame, strategy="time_series_momentum", params={"window": 60})
    assert {"asset_return", "raw_signal", "position", "strategy_return", "equity"}.issubset(result.columns)


def test_performance_metrics_are_defined() -> None:
    returns = pd.Series([0.01, -0.005, 0.002] * 100)
    metrics = performance_metrics(returns)
    assert "Sharpe Ratio" in metrics
    assert "Maximum Drawdown" in metrics
