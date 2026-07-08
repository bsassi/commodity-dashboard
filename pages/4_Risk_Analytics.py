from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import drawdown_chart
from src.drawdown import top_drawdowns
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar


configure_page("Risk Analytics")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]
data = payload["data"]

st.title("Volatility and Risk Analytics")
render_provider_warnings()

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

risk_columns = [
    "Ticker",
    "Name",
    "Sector",
    "Risk Level",
    "20D Volatility",
    "60D Volatility",
    "252D Volatility",
    "Volatility Percentile",
    "Volatility ST/LT Ratio",
    "VaR 95",
    "Expected Shortfall 95",
    "Skewness",
    "Kurtosis",
    "Worst Daily Return",
    "Current Drawdown",
    "Maximum Drawdown",
]
risk_columns = [c for c in risk_columns if c in summary.columns]
st.dataframe(format_display_table(summary[risk_columns]), width="stretch", hide_index=True)

ticker = st.selectbox("Drawdown deep dive", summary["Ticker"])
frame = data.get(ticker, pd.DataFrame())
if not frame.empty:
    st.plotly_chart(drawdown_chart(frame, f"{ticker} Drawdown"), width="stretch")
    st.dataframe(format_display_table(top_drawdowns(frame["Close"].dropna().pct_change(fill_method=None), top_n=10)), width="stretch", hide_index=True)

st.caption("Historical VaR and Expected Shortfall are empirical estimates from available returns, not a guaranteed bound on future losses.")
