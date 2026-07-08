from __future__ import annotations

import numpy as np
import pandas as pd

from src.live_market import add_technical_indicators, relative_strength_index, technical_snapshot


def _ohlc(price: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": price.shift(1).fillna(price),
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Volume": 1_000,
        },
        index=price.index,
    )


def test_live_snapshot_identifies_uptrend() -> None:
    index = pd.date_range("2026-01-01 09:30", periods=160, freq="5min")
    price = pd.Series(np.linspace(100, 125, len(index)), index=index)
    indicators = add_technical_indicators(_ohlc(price))

    snapshot = technical_snapshot(indicators, interval="5m")

    assert snapshot["Technical Score"] > 25
    assert snapshot["Signal"] in {"Long Bias", "Strong Long"}
    assert snapshot["Confidence Score"] > 40


def test_live_snapshot_identifies_downtrend() -> None:
    index = pd.date_range("2026-01-01 09:30", periods=160, freq="5min")
    price = pd.Series(np.linspace(125, 100, len(index)), index=index)
    indicators = add_technical_indicators(_ohlc(price))

    snapshot = technical_snapshot(indicators, interval="5m")

    assert snapshot["Technical Score"] < -25
    assert snapshot["Signal"] in {"Short Bias", "Strong Short"}


def test_rsi_is_bounded() -> None:
    price = pd.Series([100, 101, 100.5, 102, 101, 103, 104, 103, 105, 106, 104, 107, 108, 109, 108, 110])

    rsi = relative_strength_index(price, window=5).dropna()

    assert not rsi.empty
    assert rsi.between(0, 100).all()


def test_empty_live_snapshot_is_neutral() -> None:
    snapshot = technical_snapshot(pd.DataFrame(), interval="5m")

    assert snapshot["Signal"] == "Neutral / Wait"
    assert snapshot["Technical Score"] == 0.0


def test_intraday_vwap_resets_by_session() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            "2026-01-02 09:00",
            "2026-01-02 10:00",
        ]
    )
    price = pd.Series([100.0, 110.0, 200.0, 220.0], index=index)
    frame = _ohlc(price)
    frame["Volume"] = [1_000, 1_000, 1_000, 1_000]

    indicators = add_technical_indicators(frame, interval="60m")

    assert np.isclose(indicators["VWAP"].iloc[1], 105.0)
    assert np.isclose(indicators["VWAP"].iloc[2], 200.0)
