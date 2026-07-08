from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.charts import seasonality_heatmap
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar
from src.returns import monthly_returns
from src.seasonality import calendar_month_stats


configure_page("Seasonality")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]
data = payload["data"]

st.title("Seasonality")
render_provider_warnings()
st.caption("Seasonality is evidence-ranked and should be penalized when sample size, stability or significance is weak.")

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

ticker = st.selectbox("Commodity", summary["Ticker"])
frame = data.get(ticker, pd.DataFrame())
if frame.empty:
    st.error("No usable history for this ticker.")
    st.stop()

st.plotly_chart(seasonality_heatmap(frame["Close"]), width="stretch")
stats = calendar_month_stats(frame["Close"], bootstrap_iterations=500)
st.dataframe(format_display_table(stats), width="stretch", hide_index=True)

monthly = monthly_returns(frame["Close"]).to_frame("Return")
monthly["Month"] = monthly.index.month_name().str.slice(0, 3)
fig = px.box(monthly, x="Month", y="Return", title="Monthly Return Distribution", template="plotly_white")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.caption("Multiple testing risk: reviewing many assets and months increases the chance of finding patterns by luck.")
