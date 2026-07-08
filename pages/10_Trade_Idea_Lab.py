from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.macro import top_trade_ideas
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar


configure_page("Trade Idea Lab")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]
macro = payload.get("macro", {})
dxy = macro.get("dxy_context", {})

st.title("Trade Idea Lab")
render_provider_warnings()

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

if macro.get("dxy_warnings"):
    st.warning("DXY data warning: " + "; ".join(macro["dxy_warnings"]))

macro_cols = st.columns(6)
macro_cols[0].metric("DXY", "-" if pd.isna(dxy.get("Last")) else f"{dxy.get('Last'):,.2f}")
macro_cols[1].metric("DXY 1W", "-" if pd.isna(dxy.get("1W Return")) else f"{dxy.get('1W Return'):.1%}")
macro_cols[2].metric("DXY 1M", "-" if pd.isna(dxy.get("1M Return")) else f"{dxy.get('1M Return'):.1%}")
macro_cols[3].metric("DXY 3M", "-" if pd.isna(dxy.get("3M Return")) else f"{dxy.get('3M Return'):.1%}")
macro_cols[4].metric("DXY Score", f"{float(dxy.get('Trend Score', 0.0)):.1f}")
macro_cols[5].metric("DXY Regime", str(dxy.get("Regime", "Unavailable")))

min_conviction = st.slider("Minimum conviction", 0, 80, 20, 5)
setup_options = ["All", "Long Candidates", "Short Candidates", "High Conviction Only"]
setup_filter = st.segmented_control("Setup filter", setup_options, default="All")

ideas = top_trade_ideas(summary, min_conviction=float(min_conviction))
if setup_filter == "Long Candidates":
    ideas = ideas[ideas["Trade Idea Score"] > 0]
elif setup_filter == "Short Candidates":
    ideas = ideas[ideas["Trade Idea Score"] < 0]
elif setup_filter == "High Conviction Only":
    ideas = ideas[ideas["Trade Setup"].isin(["High Conviction Long", "High Conviction Short"])]

scatter_frame = summary.copy()
scatter_frame["Trade Idea Conviction"] = pd.to_numeric(scatter_frame["Trade Idea Conviction"], errors="coerce").fillna(0.0)
scatter_frame["DXY Pressure Score"] = pd.to_numeric(scatter_frame["DXY Pressure Score"], errors="coerce").fillna(0.0)
scatter_frame["Final Composite Score"] = pd.to_numeric(scatter_frame["Final Composite Score"], errors="coerce").fillna(0.0)
fig = px.scatter(
    scatter_frame,
    x="DXY Pressure Score",
    y="Final Composite Score",
    color="Sector",
    size=np.maximum(scatter_frame["Trade Idea Conviction"], 8),
    text="Ticker",
    hover_name="Name",
    hover_data=[
        "Trade Setup",
        "Trade Idea Score",
        "DXY Pressure",
        "DXY Beta",
        "DXY Correlation",
        "Seasonality Signal",
        "Risk Level",
    ],
    template="plotly_white",
)
fig.add_hline(y=0, line_color="#a2a9b5", line_width=1)
fig.add_vline(x=0, line_color="#a2a9b5", line_width=1)
fig.update_traces(textposition="top center")
fig.update_layout(height=560, margin=dict(l=10, r=10, t=35, b=10))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Long List")
    long_cols = [
        "Ticker",
        "Name",
        "Sector",
        "Trade Setup",
        "Trade Idea Score",
        "Trade Idea Conviction",
        "DXY Pressure",
        "DXY Beta",
        "Seasonality Signal",
        "Risk Level",
    ]
    longs = ideas[ideas["Trade Idea Score"] > 0].sort_values("Trade Idea Score", ascending=False)
    st.dataframe(format_display_table(longs[[c for c in long_cols if c in longs.columns]].head(10)), width="stretch", hide_index=True)

with right:
    st.subheader("Short List")
    shorts = ideas[ideas["Trade Idea Score"] < 0].sort_values("Trade Idea Score", ascending=True)
    st.dataframe(format_display_table(shorts[[c for c in long_cols if c in shorts.columns]].head(10)), width="stretch", hide_index=True)

st.subheader("Idea Tape")
idea_cols = [
    "Ticker",
    "Name",
    "Sector",
    "Trade Setup",
    "Trade Idea Score",
    "Trade Idea Conviction",
    "Final Composite Score",
    "DXY Pressure Score",
    "DXY Pressure",
    "DXY Beta",
    "DXY Correlation",
    "Seasonality Score",
    "Seasonality Evidence",
    "Risk Level",
    "Trade Idea Note",
]
idea_cols = [column for column in idea_cols if column in ideas.columns]
st.dataframe(format_display_table(ideas[idea_cols]), width="stretch", hide_index=True)
