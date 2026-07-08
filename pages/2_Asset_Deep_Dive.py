from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.charts import drawdown_chart, price_chart, seasonality_heatmap
from src.correlations import returns_matrix
from src.drawdown import top_drawdowns
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar
from src.seasonality import calendar_month_stats
from src.trend import moving_averages, trend_strength_snapshot
from src.volatility import realized_volatility


configure_page("Asset Deep Dive")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]
data = payload["data"]

st.title("Asset Deep Dive")
render_provider_warnings()

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

asset_label = st.selectbox("Commodity", summary["Ticker"] + " - " + summary["Name"])
ticker = asset_label.split(" - ")[0]
frame = data.get(ticker, pd.DataFrame()).copy()
if frame.empty:
    st.error("No usable history for this ticker.")
    st.stop()

date_min, date_max = frame.index.min().date(), frame.index.max().date()
start, end = st.date_input("Analysis period", value=(date_min, date_max), min_value=date_min, max_value=date_max)
frame = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
mode = st.radio("Price mode", ["level", "base 100", "log"], horizontal=True)

row = summary.loc[summary["Ticker"] == ticker].iloc[0]
cols = st.columns(5)
cols[0].metric("Signal", row["Signal Classification"])
cols[1].metric("Score", f"{row['Final Composite Score']:.1f}")
cols[2].metric("Confidence", f"{row['Confidence Score']:.1f}")
cols[3].metric("Risk", row["Risk Level"])
cols[4].metric("Regime", row["Regime"])
st.info(row["Signal Explanation"])

st.plotly_chart(price_chart(frame, f"{ticker} Price and Moving Averages", mode=mode), width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(drawdown_chart(frame, f"{ticker} Underwater Chart"), width="stretch")
with col2:
    returns = frame["Close"].dropna().pct_change(fill_method=None)
    vol = realized_volatility(returns, 60)
    fig = px.line(vol, title="Rolling 60D Realized Volatility", template="plotly_white")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

tabs = st.tabs(["Trend", "Seasonality", "Distribution", "Correlations", "Drawdowns"])
with tabs[0]:
    trend_settings = payload["settings"].get("trend", {})
    trend = trend_strength_snapshot(
        frame,
        sma_windows=trend_settings.get("sma_windows", [20, 50, 100, 200]),
        ema_windows=trend_settings.get("ema_windows", [20, 50, 100, 200]),
        donchian_windows=trend_settings.get("donchian_windows", [20, 55, 100, 252]),
        regression_windows=trend_settings.get("regression_windows", [20, 60, 120, 252]),
        momentum_windows=trend_settings.get("momentum_windows", [21, 63, 126, 189, 252]),
        exclude_last_month=trend_settings.get("exclude_last_month_default", False),
    )
    st.metric("Trend Strength", f"{trend['Trend Score']:.1f}", help="Bounded score combining moving averages, momentum, breakouts and regression.")
    ma = moving_averages(frame["Close"], trend_settings.get("sma_windows", [20, 50, 100, 200]), trend_settings.get("ema_windows", [20, 50, 100, 200]))
    st.dataframe(format_display_table(ma.tail(20).reset_index().rename(columns={"index": "Date"})), width="stretch", hide_index=True)
with tabs[1]:
    st.plotly_chart(seasonality_heatmap(frame["Close"]), width="stretch")
    stats = calendar_month_stats(frame["Close"], bootstrap_iterations=250)
    st.dataframe(format_display_table(stats), width="stretch", hide_index=True)
with tabs[2]:
    returns = frame["Close"].dropna().pct_change(fill_method=None).dropna()
    fig = px.histogram(returns, nbins=80, title="Daily Return Distribution", template="plotly_white")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(returns.describe().to_frame("Return").T, width="stretch")
with tabs[3]:
    matrix = returns_matrix(data)
    if ticker in matrix.columns:
        corr = matrix.corr()[ticker].sort_values(ascending=False).to_frame("Correlation")
        st.dataframe(corr, width="stretch")
with tabs[4]:
    dd_table = top_drawdowns(frame["Close"].dropna().pct_change(fill_method=None), top_n=10)
    st.dataframe(format_display_table(dd_table), width="stretch", hide_index=True)
