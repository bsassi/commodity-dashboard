from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clean_price_series, clip_score


def momentum_series(price: pd.Series, window: int) -> pd.Series:
    clean = clean_price_series(price)
    return clean / clean.shift(window) - 1


def volatility_adjusted_momentum(price: pd.Series, window: int = 252, vol_window: int = 60) -> pd.Series:
    clean = clean_price_series(price)
    momentum = clean / clean.shift(window) - 1
    vol = clean.pct_change(fill_method=None).rolling(vol_window, min_periods=max(20, vol_window // 2)).std() * np.sqrt(252)
    return momentum / vol.replace(0, np.nan)


def cross_sectional_momentum(
    data: dict[str, pd.DataFrame],
    horizon: int = 63,
    acceleration_horizon: int = 21,
    vol_window: int = 60,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for ticker, frame in data.items():
        if frame.empty or "Close" not in frame.columns:
            continue
        close = clean_price_series(frame["Close"])
        if len(close) <= max(horizon, vol_window):
            rows.append(
                {
                    "Ticker": ticker,
                    "Momentum": np.nan,
                    "Risk Adjusted Momentum": np.nan,
                    "Momentum Acceleration": np.nan,
                    "Cross-Sectional Score": 0.0,
                }
            )
            continue
        mom = momentum_series(close, horizon)
        risk_adj = volatility_adjusted_momentum(close, horizon, vol_window)
        acceleration = mom.iloc[-1] - mom.shift(acceleration_horizon).iloc[-1]
        rows.append(
            {
                "Ticker": ticker,
                "Momentum": float(mom.iloc[-1]) if np.isfinite(mom.iloc[-1]) else np.nan,
                "Risk Adjusted Momentum": float(risk_adj.iloc[-1]) if np.isfinite(risk_adj.iloc[-1]) else np.nan,
                "Momentum Acceleration": float(acceleration) if np.isfinite(acceleration) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for column in ["Momentum", "Risk Adjusted Momentum", "Momentum Acceleration"]:
        ranks = result[column].rank(pct=True)
        result[f"{column} Percentile"] = ranks
    result["Cross-Sectional Score"] = (result["Risk Adjusted Momentum Percentile"] - 0.5) * 200
    result["Acceleration Score"] = (result["Momentum Acceleration Percentile"] - 0.5) * 200
    result[["Cross-Sectional Score", "Acceleration Score"]] = result[
        ["Cross-Sectional Score", "Acceleration Score"]
    ].fillna(0.0)
    result["Rotation Quadrant"] = result.apply(
        lambda row: rotation_quadrant(row["Cross-Sectional Score"], row["Acceleration Score"]), axis=1
    )
    return result


def rotation_quadrant(relative_score: float, acceleration_score: float) -> str:
    if relative_score >= 0 and acceleration_score >= 0:
        return "Leading"
    if relative_score >= 0 and acceleration_score < 0:
        return "Weakening"
    if relative_score < 0 and acceleration_score < 0:
        return "Lagging"
    return "Improving"


def momentum_snapshot(price: pd.Series) -> dict[str, float]:
    clean = clean_price_series(price)
    windows = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
    output: dict[str, float] = {}
    for label, window in windows.items():
        series = momentum_series(clean, window)
        output[f"{label} Momentum"] = float(series.iloc[-1]) if not series.dropna().empty else np.nan
    values = [v for v in output.values() if np.isfinite(v)]
    output["Average Momentum Score"] = clip_score(np.nanmean(np.tanh(np.array(values) * 4) * 100)) if values else 0.0
    return output
