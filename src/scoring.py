from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .correlations import regime_label
from .drawdown import drawdown_snapshot
from .momentum import cross_sectional_momentum
from .returns import performance_snapshot, positive_month_rate
from .seasonality import current_month_seasonality
from .trend import trend_strength_snapshot
from .utils import classify_score, clean_price_series, clip_score
from .volatility import volatility_snapshot


DEFAULT_WEIGHTS = {
    "trend": 0.35,
    "time_series_momentum": 0.20,
    "cross_sectional_momentum": 0.15,
    "breakout": 0.10,
    "seasonality": 0.10,
    "risk_adjustment": 0.10,
}


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total = sum(max(v, 0) for v in weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: max(value, 0) / total for key, value in weights.items()}


def risk_penalty(volatility_percentile: float, current_drawdown: float) -> float:
    penalty = 0.0
    if np.isfinite(volatility_percentile):
        if volatility_percentile >= 0.95:
            penalty += 25
        elif volatility_percentile >= 0.80:
            penalty += 15
    if np.isfinite(current_drawdown):
        if current_drawdown <= -0.30:
            penalty += 20
        elif current_drawdown <= -0.15:
            penalty += 10
    return float(np.clip(penalty, 0, 60))


def confidence_score(
    observations: int,
    trend: dict[str, Any],
    vol: dict[str, Any],
    seasonality: dict[str, Any],
    data_status: str = "OK",
) -> float:
    score = 50.0
    score += min(observations / 1000, 1) * 20
    score += min(abs(float(trend.get("Coherence Score", 0))) / 100, 1) * 10
    r2_values = [
        value
        for key, value in dict(trend.get("Trend Components", {})).items()
        if key.startswith("r2_") and np.isfinite(value)
    ]
    if r2_values:
        score += min(float(np.nanmean(r2_values)) * 10, 10)
    if seasonality.get("Seasonality Evidence") in {"Strong Evidence", "Moderate Evidence"}:
        score += 8
    if vol.get("Risk Level") == "Extreme Risk":
        score -= 15
    elif vol.get("Risk Level") == "Elevated Risk":
        score -= 8
    if data_status == "Warning":
        score -= 10
    elif data_status == "Error":
        score -= 30
    return float(np.clip(score, 0, 100))


def directional_score(
    trend: dict[str, Any],
    cross_sectional_score: float,
    seasonality: dict[str, Any],
    weights: dict[str, float] | None = None,
    risk_quality: float = 100.0,
) -> tuple[float, dict[str, float]]:
    w = normalize_weights(weights)
    trend_score = float(trend.get("Trend Score", 0.0))
    tsmom = float(trend.get("Time-Series Momentum Score", 0.0))
    breakout = float(trend.get("Breakout Score", 0.0))
    seasonal = float(seasonality.get("Seasonality Score", 0.0))
    base_direction = np.sign(np.nanmean([trend_score, tsmom, cross_sectional_score, breakout, seasonal]))
    risk_component = base_direction * risk_quality
    components = {
        "trend": trend_score,
        "time_series_momentum": tsmom,
        "cross_sectional_momentum": cross_sectional_score,
        "breakout": breakout,
        "seasonality": seasonal,
        "risk_adjustment": risk_component,
    }
    raw = sum(w.get(key, 0.0) * value for key, value in components.items())
    return clip_score(raw), components


def data_status_for_ticker(quality: pd.DataFrame, ticker: str) -> str:
    if quality is None or quality.empty:
        return "Unknown"
    row = quality.loc[quality["Ticker"] == ticker]
    if row.empty:
        return "Unknown"
    return str(row["Status"].iloc[0])


def score_asset(
    ticker: str,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    cross_sectional_score: float = 0.0,
    weights: dict[str, float] | None = None,
    quality_status: str = "OK",
    frequency: str = "daily",
    trend_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trend_settings = trend_settings or {}
    if frame.empty or "Close" not in frame.columns or frame["Close"].dropna().empty:
        return {
            "Ticker": ticker,
            "Name": metadata.get("name", ticker),
            "Sector": metadata.get("sector", "Unknown"),
            "Last Price": np.nan,
            "Trend Signal": "Insufficient Data",
            "Momentum Signal": "Insufficient Data",
            "Trend Strength": 0.0,
            "Risk Level": "Insufficient Data",
            "Seasonality Signal": "Insufficient Data",
            "Composite Score": 0.0,
            "Confidence Score": 0.0,
            "Final Composite Score": 0.0,
            "Signal Explanation": "No usable Yahoo Finance price history was available for this ticker.",
            "Carry": "Carry indisponible avec la source de donnees actuelle.",
        }

    close = clean_price_series(frame["Close"])
    trend = trend_strength_snapshot(
        frame,
        sma_windows=trend_settings.get("sma_windows"),
        ema_windows=trend_settings.get("ema_windows"),
        donchian_windows=trend_settings.get("donchian_windows"),
        regression_windows=trend_settings.get("regression_windows"),
        momentum_windows=trend_settings.get("momentum_windows"),
        exclude_last_month=trend_settings.get("exclude_last_month_default", False),
    )
    vol = volatility_snapshot(frame, frequency=frequency)
    dd = drawdown_snapshot(close)
    seasonality = current_month_seasonality(close)
    perf = performance_snapshot(close, frequency=frequency)
    risk_pen = risk_penalty(float(vol.get("Volatility Percentile", np.nan)), float(dd.get("Current Drawdown", np.nan)))
    risk_quality = max(0.0, 100.0 - risk_pen)
    raw_score, components = directional_score(trend, cross_sectional_score, seasonality, weights, risk_quality)
    final_score = clip_score(raw_score * (1 - risk_pen / 100))
    confidence = confidence_score(len(close), trend, vol, seasonality, quality_status)
    final_score = clip_score(final_score * (0.50 + confidence / 200))
    direction = classify_score(final_score)
    one_year = close.tail(252)
    high_52 = one_year.max() if not one_year.empty else np.nan
    low_52 = one_year.min() if not one_year.empty else np.nan
    last_price = float(close.iloc[-1])
    dist_high = last_price / high_52 - 1 if np.isfinite(high_52) and high_52 else np.nan
    dist_low = last_price / low_52 - 1 if np.isfinite(low_52) and low_52 else np.nan
    momentum_signal = "Positive" if components["time_series_momentum"] > 20 else "Negative" if components["time_series_momentum"] < -20 else "Mixed"
    regime = regime_label(float(trend.get("Trend Score", 0)), float(vol.get("Volatility Percentile", np.nan)))

    row = {
        "Ticker": ticker,
        "Name": metadata.get("name", ticker),
        "Sector": metadata.get("sector", "Unknown"),
        "Sub-Sector": metadata.get("sub_sector", "Unknown"),
        "Currency": metadata.get("currency", "USD"),
        "Asset Class": metadata.get("asset_class", "Futures"),
        "Indicative Unit": metadata.get("indicative_unit", ""),
        "Last Price": last_price,
        "Distance to 52-Week High": dist_high,
        "Distance to 52-Week Low": dist_low,
        "Trend Signal": trend.get("Trend Signal"),
        "Momentum Signal": momentum_signal,
        "Trend Strength": float(trend.get("Trend Score", 0.0)),
        "Risk Level": vol.get("Risk Level"),
        "Seasonality Signal": seasonality.get("Seasonality Signal"),
        "Seasonality Evidence": seasonality.get("Seasonality Evidence"),
        "Composite Score": raw_score,
        "Direction Score": raw_score,
        "Trend Strength Score": float(trend.get("Trend Score", 0.0)),
        "Risk Score": risk_quality,
        "Signal Confidence Score": confidence,
        "Confidence Score": confidence,
        "Final Composite Score": final_score,
        "Signal Classification": direction,
        "Regime": regime,
        "Carry": "Carry indisponible avec la source de donnees actuelle.",
        **perf,
        **vol,
        **dd,
        "Positive Month Rate": positive_month_rate(close),
        "Cross-Sectional Momentum Score": cross_sectional_score,
        "Signal Explanation": build_explanation(
            metadata.get("name", ticker),
            final_score,
            trend,
            vol,
            dd,
            seasonality,
            dist_high,
            quality_status,
        ),
    }
    return row


def build_explanation(
    name: str,
    final_score: float,
    trend: dict[str, Any],
    vol: dict[str, Any],
    dd: dict[str, Any],
    seasonality: dict[str, Any],
    dist_high: float,
    quality_status: str,
) -> str:
    direction = classify_score(final_score).lower()
    clauses = [f"The signal on {name} is {direction}."]
    trend_score = float(trend.get("Trend Score", 0))
    if trend_score > 20:
        clauses.append("Trend evidence is positive across moving-average, momentum and breakout components.")
    elif trend_score < -20:
        clauses.append("Trend evidence is negative across moving-average, momentum and breakout components.")
    else:
        clauses.append("Trend evidence is mixed across horizons.")

    vol_level = vol.get("Risk Level", "Unknown")
    vol_pct = vol.get("Volatility Percentile", np.nan)
    if np.isfinite(vol_pct):
        clauses.append(f"Current volatility is in the {vol_pct:.0%} historical percentile and classified as {vol_level}.")
    else:
        clauses.append("Volatility percentile is unavailable because the history is too short.")

    current_dd = dd.get("Current Drawdown", np.nan)
    if np.isfinite(current_dd):
        clauses.append(f"The market is currently in a {current_dd:.1%} drawdown.")
    if np.isfinite(dist_high):
        clauses.append(f"Distance to the 52-week high is {dist_high:.1%}.")

    evidence = seasonality.get("Seasonality Evidence", "Insufficient Data")
    if evidence in {"Strong Evidence", "Moderate Evidence"}:
        clauses.append(f"Calendar seasonality provides {evidence.lower()} for the current month.")
    else:
        clauses.append("Current-month seasonality is weak or insufficient, so it is not treated as a robust driver.")

    if quality_status != "OK":
        clauses.append(f"Data quality status is {quality_status}, reducing confidence.")
    clauses.append("Carry is unavailable with the current Yahoo Finance data source.")
    return " ".join(clauses)


def build_summary_table(
    assets: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    quality: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    frequency: str = "daily",
    trend_settings: dict[str, Any] | None = None,
    cross_horizon: int = 63,
    acceleration_horizon: int = 21,
) -> pd.DataFrame:
    cross = cross_sectional_momentum(data, horizon=cross_horizon, acceleration_horizon=acceleration_horizon)
    cross_map = cross.set_index("Ticker")["Cross-Sectional Score"].to_dict() if not cross.empty else {}
    metadata = assets.set_index("ticker").to_dict(orient="index")
    rows = []
    for ticker, meta in metadata.items():
        row = score_asset(
            ticker=ticker,
            frame=data.get(ticker, pd.DataFrame()),
            metadata=meta,
            cross_sectional_score=float(cross_map.get(ticker, 0.0)),
            weights=weights,
            quality_status=data_status_for_ticker(quality, ticker),
            frequency=frequency,
            trend_settings=trend_settings,
        )
        if not cross.empty:
            c_row = cross.loc[cross["Ticker"] == ticker]
            if not c_row.empty:
                row["Rotation Quadrant"] = c_row["Rotation Quadrant"].iloc[0]
                row["Momentum Acceleration"] = c_row["Momentum Acceleration"].iloc[0]
                row["Risk Adjusted Momentum"] = c_row["Risk Adjusted Momentum"].iloc[0]
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("Final Composite Score", ascending=False).reset_index(drop=True)
