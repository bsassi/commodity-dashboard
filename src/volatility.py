from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import annualization_factor, pct_rank_last


def realized_volatility(returns: pd.Series, window: int = 20, frequency: str = "daily") -> pd.Series:
    factor = annualization_factor(frequency)
    return returns.rolling(window, min_periods=max(5, window // 3)).std() * np.sqrt(factor)


def ewma_volatility(returns: pd.Series, span: int = 60, frequency: str = "daily") -> pd.Series:
    factor = annualization_factor(frequency)
    return returns.ewm(span=span, min_periods=max(10, span // 3), adjust=False).std() * np.sqrt(factor)


def downside_deviation(returns: pd.Series, window: int = 60, frequency: str = "daily") -> pd.Series:
    factor = annualization_factor(frequency)
    downside = returns.where(returns < 0, 0.0)
    return downside.rolling(window, min_periods=max(10, window // 3)).std() * np.sqrt(factor)


def semi_deviation(returns: pd.Series, frequency: str = "daily") -> float:
    downside = returns.dropna()
    downside = downside[downside < 0]
    if len(downside) < 2:
        return np.nan
    return float(downside.std() * np.sqrt(annualization_factor(frequency)))


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=max(5, window // 2)).mean()


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(1 - level))


def expected_shortfall(returns: pd.Series, level: float = 0.95) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    var = clean.quantile(1 - level)
    tail = clean[clean <= var]
    return float(tail.mean()) if not tail.empty else np.nan


def volatility_snapshot(frame: pd.DataFrame, frequency: str = "daily") -> dict[str, float | str]:
    close = frame["Close"].dropna()
    returns = close.pct_change(fill_method=None)
    vol20 = realized_volatility(returns, 20, frequency)
    vol60 = realized_volatility(returns, 60, frequency)
    vol252 = realized_volatility(returns, 252, frequency)
    vol_pct = pct_rank_last(vol60, min_periods=60)
    atr_series = atr(frame, 14) if {"High", "Low", "Close"}.issubset(frame.columns) else pd.Series(dtype=float)
    atr_norm = atr_series / close if not atr_series.empty else pd.Series(dtype=float)
    clean_returns = returns.dropna()

    if np.isnan(vol_pct):
        risk_level = "Insufficient Data"
    elif vol_pct >= 0.95:
        risk_level = "Extreme Risk"
    elif vol_pct >= 0.80:
        risk_level = "Elevated Risk"
    elif vol_pct <= 0.20:
        risk_level = "Low Risk"
    else:
        risk_level = "Normal Risk"

    return {
        "20D Volatility": float(vol20.dropna().iloc[-1]) if not vol20.dropna().empty else np.nan,
        "60D Volatility": float(vol60.dropna().iloc[-1]) if not vol60.dropna().empty else np.nan,
        "252D Volatility": float(vol252.dropna().iloc[-1]) if not vol252.dropna().empty else np.nan,
        "EWMA Volatility": float(ewma_volatility(returns, 60, frequency).dropna().iloc[-1])
        if not ewma_volatility(returns, 60, frequency).dropna().empty
        else np.nan,
        "Downside Deviation": semi_deviation(clean_returns, frequency),
        "Volatility Percentile": vol_pct,
        "Volatility ST/LT Ratio": float(vol20.dropna().iloc[-1] / vol252.dropna().iloc[-1])
        if not vol20.dropna().empty and not vol252.dropna().empty and vol252.dropna().iloc[-1] != 0
        else np.nan,
        "Volatility of Volatility": float(vol60.dropna().pct_change(fill_method=None).rolling(60).std().dropna().iloc[-1])
        if len(vol60.dropna()) > 80
        else np.nan,
        "ATR": float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else np.nan,
        "ATR / Price": float(atr_norm.dropna().iloc[-1]) if not atr_norm.dropna().empty else np.nan,
        "VaR 95": historical_var(clean_returns, 0.95),
        "Expected Shortfall 95": expected_shortfall(clean_returns, 0.95),
        "Skewness": float(stats.skew(clean_returns, nan_policy="omit")) if len(clean_returns) > 3 else np.nan,
        "Kurtosis": float(stats.kurtosis(clean_returns, nan_policy="omit")) if len(clean_returns) > 3 else np.nan,
        "Extreme Loss Frequency": float((clean_returns < clean_returns.quantile(0.01)).mean())
        if len(clean_returns) > 20
        else np.nan,
        "Worst Daily Return": float(clean_returns.min()) if not clean_returns.empty else np.nan,
        "Risk Level": risk_level,
    }
