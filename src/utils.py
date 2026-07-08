from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


FREQUENCY_TO_INTERVAL = {
    "daily": "1d",
    "weekly": "1wk",
    "monthly": "1mo",
}

ANNUALIZATION_FACTORS = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def annualization_factor(frequency: str = "daily") -> int:
    return ANNUALIZATION_FACTORS.get(frequency, 252)


def yfinance_interval(frequency: str = "daily") -> str:
    return FREQUENCY_TO_INTERVAL.get(frequency, "1d")


def clip_score(value: float | int | np.floating | None, lower: float = -100.0, upper: float = 100.0) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.clip(value, lower, upper))


def safe_last(series: pd.Series, default: float = np.nan) -> float:
    clean = series.dropna()
    if clean.empty:
        return default
    value = clean.iloc[-1]
    if isinstance(value, (np.integer, np.floating, int, float)):
        return float(value)
    return value


def clean_price_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").copy()
    numeric = numeric[~numeric.index.duplicated(keep="last")]
    numeric = numeric.sort_index()
    numeric = numeric.where(numeric > 0)
    return numeric.dropna()


def ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result[~result.index.isna()]
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def pct_rank_last(series: pd.Series, min_periods: int = 20) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < min_periods:
        return np.nan
    last = clean.iloc[-1]
    return float((clean <= last).mean())


def expanding_percentile(series: pd.Series, min_periods: int = 20) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def _rank_last(window: pd.Series) -> float:
        clean = window.dropna()
        if len(clean) < min_periods:
            return np.nan
        return float((clean <= clean.iloc[-1]).mean())

    return values.expanding(min_periods=min_periods).apply(_rank_last, raw=False)


def classify_score(score: float) -> str:
    if score >= 60:
        return "Strong Long"
    if score >= 20:
        return "Moderate Long"
    if score <= -60:
        return "Strong Short"
    if score <= -20:
        return "Moderate Short"
    return "Neutral"


def classify_trend_strength(score: float) -> str:
    if score >= 60:
        return "Strong Bullish Trend"
    if score >= 20:
        return "Bullish Trend"
    if score <= -60:
        return "Strong Bearish Trend"
    if score <= -20:
        return "Bearish Trend"
    return "Neutral / Mixed"


def status_from_issues(warnings: Iterable[str], errors: Iterable[str]) -> str:
    if list(errors):
        return "Error"
    if list(warnings):
        return "Warning"
    return "OK"


def today_utc_date() -> date:
    return pd.Timestamp.utcnow().date()


def to_jsonable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
                item[key] = str(value)
            elif pd.isna(value):
                item[key] = None
            else:
                item[key] = value
        records.append(item)
    return records

