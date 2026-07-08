from __future__ import annotations

import streamlit as st

from src.reporting import configure_page, dataframe_to_csv_bytes, format_display_table, get_payload_from_controls, render_provider_warnings, render_sidebar


configure_page("Data Quality")
controls = render_sidebar()
payload = get_payload_from_controls(controls)

st.title("Data Quality")
render_provider_warnings()

quality = payload["quality"]
download_log = payload["download_log"]
st.dataframe(format_display_table(quality), width="stretch", hide_index=True)
st.download_button("Export data-quality report", dataframe_to_csv_bytes(quality), "data_quality.csv", "text/csv")

st.subheader("Download Log")
st.dataframe(format_display_table(download_log), width="stretch", hide_index=True)

st.caption(
    "Status is based on available Yahoo Finance observations, missing close values, stale data, non-positive prices and abnormal return jumps."
)
