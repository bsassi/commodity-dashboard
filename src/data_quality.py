from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import status_from_issues, today_utc_date


def close_return_outliers(close: pd.Series, z_threshold: float = 8.0, abs_threshold: float = 0.25) -> int:
    returns = close.dropna().pct_change(fill_method=None).dropna()
    if returns.empty:
        return 0
    std = returns.std()
    if not np.isfinite(std) or std == 0:
        z_count = 0
    else:
        z_count = int(((returns - returns.mean()).abs() / std > z_threshold).sum())
    abs_count = int((returns.abs() > abs_threshold).sum())
    return max(z_count, abs_count)


def asset_quality_row(
    ticker: str,
    frame: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    z_threshold: float = 8.0,
    abs_threshold: float = 0.25,
) -> dict[str, Any]:
    metadata = metadata or {}
    warnings: list[str] = []
    errors: list[str] = []

    if frame is None or frame.empty or "Close" not in frame.columns:
        errors.append("No usable price history")
        return {
            "Ticker": ticker,
            "Name": metadata.get("name", ticker),
            "Sector": metadata.get("sector", "Unknown"),
            "Observations": 0,
            "Missing %": 1.0,
            "First Observation": pd.NaT,
            "Last Observation": pd.NaT,
            "Days Since Last Data": np.nan,
            "Outliers": 0,
            "Non-positive Prices": 0,
            "Status": "Error",
            "Issues": "; ".join(errors),
        }

    close = pd.to_numeric(frame["Close"], errors="coerce")
    observations = int(close.notna().sum())
    missing_pct = float(close.isna().mean()) if len(close) else 1.0
    non_positive = int((close <= 0).sum())
    if non_positive:
        errors.append(f"{non_positive} non-positive closes")
    if observations < 260:
        warnings.append("Limited history")
    if missing_pct > 0.05:
        warnings.append("More than 5% missing close values")

    first = close.dropna().index.min() if observations else pd.NaT
    last = close.dropna().index.max() if observations else pd.NaT
    if pd.isna(last):
        days_since_last = np.nan
    else:
        days_since_last = (pd.Timestamp(today_utc_date()) - pd.Timestamp(last).normalize()).days
        if days_since_last > 10:
            warnings.append("Last observation is stale")

    outliers = close_return_outliers(close, z_threshold=z_threshold, abs_threshold=abs_threshold)
    if outliers:
        warnings.append(f"{outliers} abnormal return observations")

    issues = errors + warnings
    return {
        "Ticker": ticker,
        "Name": metadata.get("name", ticker),
        "Sector": metadata.get("sector", "Unknown"),
        "Observations": observations,
        "Missing %": missing_pct,
        "First Observation": first,
        "Last Observation": last,
        "Days Since Last Data": days_since_last,
        "Outliers": outliers,
        "Non-positive Prices": non_positive,
        "Status": status_from_issues(warnings, errors),
        "Issues": "; ".join(issues),
    }


def data_quality_report(
    data: dict[str, pd.DataFrame],
    assets: pd.DataFrame,
    z_threshold: float = 8.0,
    abs_threshold: float = 0.25,
) -> pd.DataFrame:
    rows = []
    metadata = assets.set_index("ticker").to_dict(orient="index") if not assets.empty else {}
    for ticker, frame in data.items():
        rows.append(asset_quality_row(ticker, frame, metadata.get(ticker), z_threshold, abs_threshold))
    return pd.DataFrame(rows)


def enrich_assets_with_quality(assets: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if quality.empty:
        return assets
    q = quality.rename(
        columns={
            "Ticker": "ticker",
            "Status": "data_available",
            "First Observation": "first_observation",
            "Last Observation": "last_observation",
        }
    )[["ticker", "data_available", "first_observation", "last_observation"]]
    result = assets.drop(columns=[c for c in q.columns if c in assets.columns and c != "ticker"], errors="ignore")
    return result.merge(q, on="ticker", how="left")
