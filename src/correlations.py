from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def returns_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for ticker, frame in data.items():
        if frame.empty or "Close" not in frame.columns:
            continue
        columns[ticker] = frame["Close"].dropna().pct_change(fill_method=None)
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).dropna(how="all")


def correlation_matrix(data: dict[str, pd.DataFrame], window: int | None = None) -> pd.DataFrame:
    returns = returns_matrix(data)
    if returns.empty:
        return pd.DataFrame()
    if window:
        returns = returns.tail(window)
    return returns.corr(min_periods=max(20, (window or 60) // 3))


def rolling_average_correlation(data: dict[str, pd.DataFrame], window: int = 60) -> pd.Series:
    returns = returns_matrix(data)
    if returns.empty or returns.shape[1] < 2:
        return pd.Series(dtype=float)

    def _avg_corr(frame: pd.DataFrame) -> float:
        corr = frame.corr().to_numpy()
        if corr.size == 0:
            return np.nan
        upper = corr[np.triu_indices_from(corr, k=1)]
        return float(np.nanmean(upper))

    return returns.rolling(window, min_periods=max(20, window // 2)).apply(lambda _: np.nan).iloc[:, 0].combine_first(
        pd.Series([_avg_corr(returns.iloc[max(0, i - window + 1) : i + 1]) for i in range(len(returns))], index=returns.index)
    )


def clustered_correlation_order(corr: pd.DataFrame) -> list[str]:
    if corr.empty or corr.shape[0] < 3:
        return list(corr.index)
    clean = corr.fillna(0).clip(-1, 1)
    distance = np.sqrt(0.5 * (1 - clean))
    np.fill_diagonal(distance.values, 0)
    condensed = squareform(distance.values, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    order = hierarchy.leaves_list(linkage)
    return list(clean.index[order])


def conditional_correlations(data: dict[str, pd.DataFrame], reference: str | None = None) -> dict[str, pd.DataFrame]:
    returns = returns_matrix(data).dropna(how="all")
    if returns.empty:
        return {}
    ref = returns[reference] if reference in returns.columns else returns.mean(axis=1)
    high_vol = ref.abs() >= ref.abs().rolling(252, min_periods=60).quantile(0.8)
    low_vol = ref.abs() <= ref.abs().rolling(252, min_periods=60).quantile(0.2)
    return {
        "Up Markets": returns[ref > 0].corr(),
        "Down Markets": returns[ref < 0].corr(),
        "High Volatility": returns[high_vol.fillna(False)].corr(),
        "Low Volatility": returns[low_vol.fillna(False)].corr(),
    }


def regime_label(trend_score: float, volatility_percentile: float) -> str:
    trend_state = "Trending" if abs(trend_score) >= 35 else "Range-Bound"
    vol_state = "High Vol" if np.isfinite(volatility_percentile) and volatility_percentile >= 0.75 else "Low Vol"
    return f"{trend_state} / {vol_state}"
