from __future__ import annotations

import streamlit as st

from src.charts import correlation_heatmap
from src.correlations import clustered_correlation_order, conditional_correlations, correlation_matrix
from src.reporting import configure_page, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar


configure_page("Correlation Regimes")
controls = render_sidebar()
payload = get_payload_from_controls(controls)
summary = payload["summary"]
data = payload["data"]

st.title("Correlations and Regimes")
render_provider_warnings()

window = st.selectbox("Correlation window", ["Full history", "20D", "60D", "252D"], index=2)
window_map = {"Full history": None, "20D": 20, "60D": 60, "252D": 252}
corr = correlation_matrix(data, window_map[window])
if corr.empty:
    st.error("Not enough overlapping return history for correlations.")
    st.stop()

order = clustered_correlation_order(corr)
st.plotly_chart(correlation_heatmap(corr.loc[order, order]), width="stretch")

tabs = st.tabs(["Conditional Correlations", "Regime Table"])
with tabs[0]:
    conditionals = conditional_correlations(data)
    for label, matrix in conditionals.items():
        if not matrix.empty:
            st.subheader(label)
            st.plotly_chart(correlation_heatmap(matrix.loc[matrix.index.intersection(order), matrix.columns.intersection(order)]), width="stretch")
with tabs[1]:
    st.dataframe(
        format_display_table(summary[["Ticker", "Name", "Sector", "Trend Strength", "Volatility Percentile", "Regime", "Final Composite Score"]]),
        width="stretch",
        hide_index=True,
    )
