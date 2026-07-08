from __future__ import annotations

import streamlit as st

from src.charts import performance_heatmap, rotation_scatter
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


configure_page("Market Overview")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]

st.title("Market Overview")
render_provider_warnings()

if summary.empty:
    st.error("No usable market data was loaded.")
    st.stop()

signals = sorted(summary["Signal Classification"].dropna().unique().tolist())
risk_levels = sorted(summary["Risk Level"].dropna().unique().tolist())
sectors = sorted(summary["Sector"].dropna().unique().tolist())
col1, col2, col3 = st.columns(3)
selected_sectors = col1.multiselect("Sector filter", sectors, default=sectors)
selected_signals = col2.multiselect("Signal filter", signals, default=signals)
selected_risk = col3.multiselect("Risk filter", risk_levels, default=risk_levels)
view = filter_summary(summary, selected_sectors, selected_signals, selected_risk)

cols = st.columns(5)
cols[0].metric("Assets", len(view))
cols[1].metric("Strong Long", int((view["Signal Classification"] == "Strong Long").sum()))
cols[2].metric("Strong Short", int((view["Signal Classification"] == "Strong Short").sum()))
cols[3].metric("Warnings", int((payload["quality"]["Status"] == "Warning").sum()))
cols[4].metric("Errors", int((payload["quality"]["Status"] == "Error").sum()))

display_cols = [
    "Ticker",
    "Name",
    "Sector",
    "Last Price",
    "1M Return",
    "3M Return",
    "12M Return",
    "Year-to-Date Return",
    "Trend Signal",
    "Momentum Signal",
    "Risk Level",
    "Seasonality Signal",
    "Final Composite Score",
    "Confidence Score",
    "Signal Explanation",
]
display_cols = [c for c in display_cols if c in view.columns]
st.dataframe(format_display_table(view[display_cols]), width="stretch", hide_index=True)
col_a, col_b = st.columns(2)
col_a.download_button("Export CSV", dataframe_to_csv_bytes(view[display_cols]), "market_overview.csv", "text/csv")
col_b.download_button(
    "Export Excel",
    dataframe_to_excel_bytes(view[display_cols], "Overview"),
    "market_overview.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.plotly_chart(performance_heatmap(view), width="stretch")
st.plotly_chart(rotation_scatter(view), width="stretch")
