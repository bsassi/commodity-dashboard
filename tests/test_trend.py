from __future__ import annotations

import pandas as pd

from src.trend import donchian_signal, price_above_sma_signal, trend_strength_snapshot


def test_trend_strength_positive_for_uptrend() -> None:
    price = pd.Series(range(1, 301), index=pd.date_range("2023-01-01", periods=300))
    frame = pd.DataFrame({"Close": price, "High": price * 1.01, "Low": price * 0.99, "Open": price, "Volume": 1})
    trend = trend_strength_snapshot(frame)
    assert trend["Trend Score"] > 20


def test_price_above_sma_signal_handles_short_history() -> None:
    price = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))
    signal = price_above_sma_signal(price, window=200)
    assert signal.fillna(0).abs().sum() == 0


def test_donchian_breakout_uses_prior_channel() -> None:
    price = pd.Series([10, 10, 10, 10, 12], index=pd.date_range("2024-01-01", periods=5))
    signal = donchian_signal(price, window=3)
    assert signal.iloc[-2] == 0
    assert signal.iloc[-1] == 1
