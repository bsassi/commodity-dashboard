from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.backtest import backtest_single_asset, backtest_universe, performance_metrics, portfolio_diagnostics, sensitivity_grid
from src.charts import drawdown_chart, equity_curve
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar


configure_page("Signal Backtest")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]
data = payload["data"]

st.title("Signal Backtest")
render_provider_warnings()
st.caption("Signals are computed at close t and applied from t+1. Results are research diagnostics, not recommendations.")

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

strategy = st.selectbox(
    "Strategy",
    ["trend_ensemble", "time_series_momentum", "price_above_sma", "sma_crossover", "donchian_breakout"],
)
col1, col2, col3, col4 = st.columns(4)
window = col1.number_input("Window", 20, 400, 252, 5)
short_window = col2.number_input("Short SMA", 10, 200, 50, 5)
long_window = col3.number_input("Long SMA", 50, 400, 200, 5)
deadband = col4.slider("Trend deadband", 0.00, 0.50, 0.15, 0.05)

params = {
    "window": int(window),
    "short_window": int(short_window),
    "long_window": int(long_window),
    "deadband": float(deadband),
    "volatility_window": 60,
}
portfolio_bt, asset_results = backtest_universe(
    data,
    strategy=strategy,
    params=params,
    transaction_cost_bps=controls["transaction_cost_bps"],
    rebalancing_frequency=controls["rebalancing_frequency"],
)

if portfolio_bt.empty:
    st.error("Backtest could not be computed for the selected universe.")
    st.stop()

metrics = {**performance_metrics(portfolio_bt["strategy_return"]), **portfolio_diagnostics(portfolio_bt)}
metric_cols = st.columns(8)
for column, (label, value) in zip(
    metric_cols,
    [
        ("CAGR", metrics.get("CAGR")),
        ("Vol", metrics.get("Annualized Volatility")),
        ("Sharpe", metrics.get("Sharpe Ratio")),
        ("Sortino", metrics.get("Sortino Ratio")),
        ("Calmar", metrics.get("Calmar Ratio")),
        ("Max DD", metrics.get("Maximum Drawdown")),
        ("Avg Exp", metrics.get("Average Exposure")),
        ("Avg Turnover", metrics.get("Average Turnover")),
    ],
):
    if label in {"Sharpe", "Sortino", "Calmar"}:
        column.metric(label, "-" if pd.isna(value) else f"{value:.2f}")
    else:
        column.metric(label, "-" if pd.isna(value) else f"{value:.1%}")

st.plotly_chart(equity_curve(portfolio_bt), width="stretch")
dd_frame = portfolio_bt.rename(columns={"equity": "Close"})
dd_frame["Close"] = portfolio_bt["equity"]
st.plotly_chart(drawdown_chart(dd_frame, "Strategy Drawdown"), width="stretch")

monthly = portfolio_bt["strategy_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1).to_frame("Return")
monthly["Year"] = monthly.index.year
monthly["Month"] = monthly.index.month
monthly_matrix = monthly.pivot(index="Year", columns="Month", values="Return")
fig = px.imshow(monthly_matrix, color_continuous_scale="RdBu", aspect="auto", template="plotly_white")
st.plotly_chart(fig, width="stretch")

st.subheader("Single Asset Diagnostics")
ticker = st.selectbox("Asset", sorted(asset_results.keys()))
asset_bt = asset_results[ticker]
st.dataframe(format_display_table(asset_bt.tail(20).reset_index().rename(columns={"index": "Date"})), width="stretch", hide_index=True)

if ticker in data:
    st.subheader("SMA Sensitivity")
    grid = sensitivity_grid(data[ticker], [40, 50, 60], [180, 200, 220])
    if not grid.empty:
        heat = grid.pivot(index="Short Window", columns="Long Window", values="Sharpe Ratio")
        fig = px.imshow(heat, color_continuous_scale="RdBu", aspect="auto", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
