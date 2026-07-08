from __future__ import annotations

import streamlit as st

from src.charts import performance_bar, rotation_scatter
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar


configure_page("Trend Momentum")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]

st.title("Trend and Momentum")
render_provider_warnings()

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

st.subheader("Rotation Matrix")
st.plotly_chart(rotation_scatter(summary), width="stretch")

cols = st.columns(2)
with cols[0]:
    st.plotly_chart(performance_bar(summary, "1M Return"), width="stretch")
with cols[1]:
    st.plotly_chart(performance_bar(summary, "12M Return"), width="stretch")

columns = [
    "Ticker",
    "Name",
    "Sector",
    "Trend Signal",
    "Trend Strength",
    "Momentum Signal",
    "Risk Adjusted Momentum",
    "Momentum Acceleration",
    "Cross-Sectional Momentum Score",
    "Rotation Quadrant",
    "DXY Pressure",
    "DXY Pressure Score",
    "DXY Beta",
    "DXY Correlation",
    "Trade Setup",
    "Trade Idea Score",
    "Trade Idea Conviction",
    "Final Composite Score",
    "Confidence Score",
]
columns = [c for c in columns if c in summary.columns]
st.dataframe(format_display_table(summary[columns]), width="stretch", hide_index=True)
