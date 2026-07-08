from __future__ import annotations

import numpy as np
import pandas as pd

from src.macro import dxy_context, dxy_pressure_score, enrich_summary_with_dxy


def _frame(price: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price, "Volume": 1_000}, index=price.index)


def test_dxy_pressure_detects_headwind_for_negative_beta_asset() -> None:
    index = pd.date_range("2025-01-01", periods=220, freq="B")
    dxy = pd.Series(np.linspace(100, 110, len(index)), index=index)
    asset = 100 * (dxy / dxy.iloc[0]) ** -1

    features = dxy_pressure_score(_frame(asset), _frame(dxy))

    assert features["DXY Beta"] < 0
    assert features["DXY Pressure Score"] < 0
    assert features["DXY Pressure"] == "Dollar Headwind"


def test_dxy_context_classifies_uptrend() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="B")
    dxy = pd.Series(np.linspace(95, 110, len(index)), index=index)

    context = dxy_context(_frame(dxy))

    assert context["Trend Score"] > 20
    assert context["Regime"] in {"Dollar Uptrend", "Dollar Neutral"}


def test_enrich_summary_adds_trade_setup_columns() -> None:
    index = pd.date_range("2025-01-01", periods=220, freq="B")
    dxy = pd.Series(np.linspace(100, 110, len(index)), index=index)
    asset = pd.Series(np.linspace(100, 90, len(index)), index=index)
    summary = pd.DataFrame(
        [
            {
                "Ticker": "CL=F",
                "Name": "WTI",
                "Sector": "Energy",
                "Final Composite Score": -45.0,
                "Seasonality Score": -20.0,
                "Cross-Sectional Momentum Score": -30.0,
                "Confidence Score": 70.0,
                "Risk Level": "Normal Risk",
                "Trend Signal": "Bearish Trend",
                "Seasonality Signal": "Unfavorable",
            }
        ]
    )

    enriched = enrich_summary_with_dxy(summary, {"CL=F": _frame(asset)}, _frame(dxy))

    assert "Trade Idea Score" in enriched.columns
    assert enriched["Trade Idea Score"].iloc[0] < 0
    assert enriched["Trade Setup"].iloc[0] in {"Short Candidate", "High Conviction Short"}
