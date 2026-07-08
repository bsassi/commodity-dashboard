from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import performance_bar, performance_heatmap, rotation_scatter
from src.portfolio import volatility_scaled_weights
from src.reporting import (
    configure_page,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    filter_summary,
    format_display_table,
    get_payload_from_controls,
    render_provider_warnings,
    render_sidebar,
)


configure_page("Commodity Systematic Macro Dashboard")
controls = render_sidebar()

st.title("Commodity Systematic Macro Dashboard")
st.caption("Trend-following, managed-futures style decision support using Yahoo Finance commodity futures data.")
render_provider_warnings()

with st.spinner("Loading market data and computing signals..."):
    payload = get_payload_from_controls(controls)

summary = payload["summary"]
macro = payload.get("macro", {})
if summary.empty:
    st.error("No usable market data was loaded. Check Yahoo Finance availability or reduce filters.")
    st.stop()

signals = sorted(summary["Signal Classification"].dropna().unique().tolist())
risk_levels = sorted(summary["Risk Level"].dropna().unique().tolist())
left, mid, right = st.columns([1, 1, 2])
with left:
    selected_signals = st.multiselect("Signal", signals, default=signals)
with mid:
    selected_risk = st.multiselect("Risk level", risk_levels, default=risk_levels)
with right:
    st.write("")

view = filter_summary(summary, signals=selected_signals, risk_levels=selected_risk)

bullish = int((view["Final Composite Score"] >= 20).sum())
bearish = int((view["Final Composite Score"] <= -20).sum())
neutral = int((view["Final Composite Score"].abs() < 20).sum())
top_momentum = view.sort_values("Risk Adjusted Momentum", ascending=False).head(1)
high_vol = view.sort_values("60D Volatility", ascending=False).head(1)
worst_dd = view.sort_values("Current Drawdown", ascending=True).head(1)
sector_scores = view.groupby("Sector")["Final Composite Score"].mean().sort_values()
top_trade = view.sort_values("Trade Idea Score", key=lambda s: s.abs(), ascending=False).head(1) if "Trade Idea Score" in view else view.head(0)
dxy_context = macro.get("dxy_context", {})

top_cols = st.columns(5)
top_cols[0].metric("Bullish", bullish)
top_cols[1].metric("Bearish", bearish)
top_cols[2].metric("Neutral", neutral)
top_cols[3].metric("Top Momentum", top_momentum["Ticker"].iloc[0] if not top_momentum.empty else "-")
top_cols[4].metric("Highest Vol", high_vol["Ticker"].iloc[0] if not high_vol.empty else "-")
bottom_cols = st.columns(5)
bottom_cols[0].metric("Worst Drawdown", worst_dd["Ticker"].iloc[0] if not worst_dd.empty else "-")
bottom_cols[1].metric("Strongest Sector", sector_scores.index[-1] if len(sector_scores) else "-")
bottom_cols[2].metric("Weakest Sector", sector_scores.index[0] if len(sector_scores) else "-")
bottom_cols[3].metric("Top Trade Idea", top_trade["Ticker"].iloc[0] if not top_trade.empty else "-")
bottom_cols[4].metric("DXY Regime", dxy_context.get("Regime", "-"))

st.subheader("Market Overview")
display_columns = [
    "Ticker",
    "Name",
    "Sector",
    "Last Price",
    "Daily Return",
    "Weekly Return",
    "1M Return",
    "3M Return",
    "6M Return",
    "12M Return",
    "Year-to-Date Return",
    "Distance to 52-Week High",
    "Distance to 52-Week Low",
    "20D Volatility",
    "60D Volatility",
    "252D Volatility",
    "Current Drawdown",
    "Maximum Drawdown",
    "Trend Signal",
    "Momentum Signal",
    "Trend Strength",
    "Risk Level",
    "DXY Pressure",
    "DXY Pressure Score",
    "DXY Beta",
    "Seasonality Signal",
    "Seasonality Score",
    "Composite Score",
    "Confidence Score",
    "Final Composite Score",
    "Trade Setup",
    "Trade Idea Score",
    "Trade Idea Conviction",
    "Signal Classification",
    "Signal Explanation",
]
display_columns = [column for column in display_columns if column in view.columns]
st.dataframe(format_display_table(view[display_columns]), width="stretch", hide_index=True)

download_cols = st.columns(2)
download_cols[0].download_button(
    "Export CSV",
    data=dataframe_to_csv_bytes(view[display_columns]),
    file_name="commodity_dashboard_summary.csv",
    mime="text/csv",
)
download_cols[1].download_button(
    "Export Excel",
    data=dataframe_to_excel_bytes(view[display_columns]),
    file_name="commodity_dashboard_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.subheader("Cross-Sectional Diagnostics")
chart_cols = st.columns(2)
with chart_cols[0]:
    st.plotly_chart(performance_heatmap(view), width="stretch")
with chart_cols[1]:
    st.plotly_chart(performance_bar(view, "3M Return"), width="stretch")

st.subheader("Rotation Matrix")
st.plotly_chart(rotation_scatter(view), width="stretch")

st.subheader("Theoretical Volatility-Scaled Exposure")
weights = volatility_scaled_weights(
    view,
    target_volatility=controls["target_volatility"],
    max_weight_per_asset=controls["max_weight_per_asset"],
    max_weight_per_sector=controls["max_weight_per_sector"],
    long_only=controls["long_only"],
)
st.caption(
    "Theoretical sizing only. It does not include contract multipliers, margin, liquidity, slippage, real roll costs, "
    "regulatory limits or portfolio-specific constraints."
)
st.dataframe(format_display_table(weights), width="stretch", hide_index=True)

st.caption(f"Last market-data refresh key: {controls['refresh_key']}")
