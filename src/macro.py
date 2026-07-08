from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import clean_price_series, clip_score


DXY_TICKER = "DX-Y.NYB"
DXY_NAME = "US Dollar Index"


def horizon_return(price: pd.Series, window: int) -> float:
    clean = clean_price_series(price)
    if len(clean) <= window:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[-window - 1] - 1)


def dxy_context(dxy_frame: pd.DataFrame) -> dict[str, Any]:
    if dxy_frame.empty or "Close" not in dxy_frame.columns or dxy_frame["Close"].dropna().empty:
        return {
            "Ticker": DXY_TICKER,
            "Name": DXY_NAME,
            "Last": np.nan,
            "1W Return": np.nan,
            "1M Return": np.nan,
            "3M Return": np.nan,
            "Trend Score": 0.0,
            "Regime": "Unavailable",
        }

    close = clean_price_series(dxy_frame["Close"])
    ret_1w = horizon_return(close, 5)
    ret_1m = horizon_return(close, 21)
    ret_3m = horizon_return(close, 63)
    sma_50 = close.rolling(50, min_periods=20).mean()
    sma_200 = close.rolling(200, min_periods=80).mean()
    ma_component = 0.0
    if not sma_50.dropna().empty and not sma_200.dropna().empty:
        ma_component = 35.0 if sma_50.iloc[-1] > sma_200.iloc[-1] else -35.0

    momentum_score = clip_score(np.nanmean([np.tanh(ret_1m * 8) * 100, np.tanh(ret_3m * 4) * 100]))
    trend_score = clip_score(0.65 * momentum_score + ma_component)
    if trend_score >= 35:
        regime = "Dollar Uptrend"
    elif trend_score <= -35:
        regime = "Dollar Downtrend"
    else:
        regime = "Dollar Neutral"

    return {
        "Ticker": DXY_TICKER,
        "Name": DXY_NAME,
        "Last": float(close.iloc[-1]),
        "1W Return": ret_1w,
        "1M Return": ret_1m,
        "3M Return": ret_3m,
        "Trend Score": trend_score,
        "Regime": regime,
        "Last Observation": close.index[-1],
    }


def rolling_dxy_beta(
    asset_price: pd.Series,
    dxy_price: pd.Series,
    beta_window: int = 126,
    corr_window: int = 63,
) -> dict[str, float]:
    asset = clean_price_series(asset_price).pct_change(fill_method=None)
    dxy = clean_price_series(dxy_price).pct_change(fill_method=None)
    joined = pd.concat({"asset": asset, "dxy": dxy}, axis=1).dropna()
    if len(joined) < max(30, min(beta_window, corr_window) // 2):
        return {"DXY Beta": np.nan, "DXY Correlation": np.nan, "DXY Observations": float(len(joined))}

    beta_sample = joined.tail(beta_window)
    variance = beta_sample["dxy"].var()
    beta = beta_sample["asset"].cov(beta_sample["dxy"]) / variance if variance and np.isfinite(variance) else np.nan
    corr_sample = joined.tail(corr_window)
    corr = corr_sample["asset"].corr(corr_sample["dxy"]) if len(corr_sample) >= 10 else np.nan
    return {
        "DXY Beta": float(beta) if np.isfinite(beta) else np.nan,
        "DXY Correlation": float(corr) if np.isfinite(corr) else np.nan,
        "DXY Observations": float(len(joined)),
    }


def dxy_pressure_score(asset_frame: pd.DataFrame, dxy_frame: pd.DataFrame, beta_window: int = 126, corr_window: int = 63) -> dict[str, Any]:
    if asset_frame.empty or dxy_frame.empty or "Close" not in asset_frame or "Close" not in dxy_frame:
        return {
            "DXY Beta": np.nan,
            "DXY Correlation": np.nan,
            "DXY Pressure Score": 0.0,
            "DXY Pressure": "Unavailable",
            "DXY Expected Impact": np.nan,
        }

    context = dxy_context(dxy_frame)
    beta_stats = rolling_dxy_beta(asset_frame["Close"], dxy_frame["Close"], beta_window, corr_window)
    beta = beta_stats["DXY Beta"]
    corr = beta_stats["DXY Correlation"]
    dxy_move = np.nanmean([context.get("1M Return", np.nan), 0.5 * context.get("3M Return", np.nan)])
    expected_impact = beta * dxy_move if np.isfinite(beta) and np.isfinite(dxy_move) else np.nan
    impact_component = np.tanh(expected_impact * 20) * 100 if np.isfinite(expected_impact) else 0.0
    relationship_strength = abs(corr) if np.isfinite(corr) else 0.35
    regime_component = float(context.get("Trend Score", 0.0)) * np.tanh(beta) * relationship_strength if np.isfinite(beta) else 0.0
    pressure_score = clip_score(0.55 * impact_component + 0.45 * regime_component)
    if pressure_score >= 15:
        pressure = "Dollar Tailwind"
    elif pressure_score <= -15:
        pressure = "Dollar Headwind"
    else:
        pressure = "Dollar Neutral"

    return {
        **beta_stats,
        "DXY 1W Return": context.get("1W Return", np.nan),
        "DXY 1M Return": context.get("1M Return", np.nan),
        "DXY 3M Return": context.get("3M Return", np.nan),
        "DXY Trend Score": context.get("Trend Score", 0.0),
        "DXY Regime": context.get("Regime", "Unavailable"),
        "DXY Expected Impact": expected_impact,
        "DXY Pressure Score": pressure_score,
        "DXY Pressure": pressure,
    }


def trade_setup(score: float) -> str:
    if score >= 55:
        return "High Conviction Long"
    if score >= 25:
        return "Long Candidate"
    if score <= -55:
        return "High Conviction Short"
    if score <= -25:
        return "Short Candidate"
    return "Watch / No Edge"


def _risk_drag(risk_level: str) -> float:
    if risk_level == "Extreme Risk":
        return 18.0
    if risk_level == "Elevated Risk":
        return 9.0
    return 0.0


def _idea_note(row: pd.Series) -> str:
    setup = str(row.get("Trade Setup", "Watch / No Edge"))
    dxy = str(row.get("DXY Pressure", "Dollar Neutral"))
    seasonality = str(row.get("Seasonality Signal", "Neutral"))
    trend = str(row.get("Trend Signal", "Neutral / Mixed"))
    risk = str(row.get("Risk Level", "Unknown"))
    if setup == "Watch / No Edge":
        return f"No clean setup: trend is {trend}, {dxy.lower()}, seasonality is {seasonality.lower()}."
    return f"{setup}: trend is {trend}, {dxy.lower()}, seasonality is {seasonality.lower()}, risk is {risk.lower()}."


def enrich_summary_with_dxy(
    summary: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    dxy_frame: pd.DataFrame,
    beta_window: int = 126,
    corr_window: int = 63,
) -> pd.DataFrame:
    if summary.empty:
        return summary

    result = summary.copy()
    feature_rows: list[dict[str, Any]] = []
    for ticker in result["Ticker"].astype(str):
        features = dxy_pressure_score(data.get(ticker, pd.DataFrame()), dxy_frame, beta_window, corr_window)
        feature_rows.append({"Ticker": ticker, **features})
    features = pd.DataFrame(feature_rows)
    result = result.merge(features, on="Ticker", how="left")

    for column in ["DXY Pressure Score", "Seasonality Score", "Cross-Sectional Momentum Score", "Final Composite Score", "Confidence Score"]:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    alignment_bonus = []
    trade_scores = []
    convictions = []
    for _, row in result.iterrows():
        components = [
            float(row["Final Composite Score"]),
            float(row["DXY Pressure Score"]),
            float(row["Seasonality Score"]),
            float(row["Cross-Sectional Momentum Score"]),
        ]
        direction = np.sign(np.nanmean(components))
        aligned = sum(1 for value in components if abs(value) >= 15 and np.sign(value) == direction)
        bonus = 6.0 * max(aligned - 1, 0) * direction if direction else 0.0
        risk_drag = _risk_drag(str(row.get("Risk Level", "")))
        raw = 0.46 * components[0] + 0.24 * components[1] + 0.16 * components[2] + 0.14 * components[3] + bonus
        if abs(raw) >= 10:
            raw -= np.sign(raw) * risk_drag
        score = clip_score(raw)
        corr = abs(float(row.get("DXY Correlation", 0.0))) if np.isfinite(row.get("DXY Correlation", np.nan)) else 0.0
        conviction = clip_score(abs(score) * 0.55 + float(row["Confidence Score"]) * 0.25 + corr * 20 - risk_drag, 0, 100)
        alignment_bonus.append(bonus)
        trade_scores.append(score)
        convictions.append(conviction)

    result["Trade Idea Score"] = trade_scores
    result["Trade Idea Conviction"] = convictions
    result["Signal Alignment Bonus"] = alignment_bonus
    result["Trade Setup"] = result["Trade Idea Score"].map(trade_setup)
    result["Trade Idea Note"] = result.apply(_idea_note, axis=1)
    return result


def top_trade_ideas(summary: pd.DataFrame, min_conviction: float = 20.0) -> pd.DataFrame:
    if summary.empty or "Trade Idea Score" not in summary:
        return pd.DataFrame()
    ideas = summary.copy()
    ideas = ideas.loc[ideas["Trade Setup"] != "Watch / No Edge"]
    ideas = ideas.loc[pd.to_numeric(ideas["Trade Idea Conviction"], errors="coerce").fillna(0.0) >= min_conviction]
    ideas["Absolute Score"] = pd.to_numeric(ideas["Trade Idea Score"], errors="coerce").abs()
    return ideas.sort_values(["Absolute Score", "Trade Idea Conviction"], ascending=False).drop(columns=["Absolute Score"])
