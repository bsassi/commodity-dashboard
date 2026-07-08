from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import live_technical_chart
from src.live_market import (
    LIVE_TIMEFRAME_PRESETS,
    YAHOO_INTERVALS,
    YAHOO_PERIODS,
    add_technical_indicators,
    download_live_asset,
    technical_component_table,
    technical_snapshot,
)
from src.reporting import configure_page, load_configuration, render_provider_warnings


configure_page("Live Commodity Stream")


@st.cache_data(show_spinner=False, ttl=20)
def cached_live_asset(
    ticker: str,
    period: str,
    interval: str,
    prepost: bool,
    refresh_key: str,
) -> tuple[pd.DataFrame, list[str]]:
    return download_live_asset(ticker=ticker, period=period, interval=interval, prepost=prepost)


def _fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "-"
    return f"{float(value):.{digits}%}"


def _fmt_minutes(value: Any) -> str:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "-"
    minutes = float(value)
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def _stale_threshold_minutes(interval: str) -> float:
    if interval.endswith("m"):
        return max(float(interval[:-1]) * 4, 20.0)
    if interval == "1d":
        return 60 * 36
    if interval == "1wk":
        return 60 * 24 * 10
    return 60 * 24 * 45


def _display_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(pd.Timestamp(value))


assets, _settings = load_configuration()
assets = assets.copy()
assets["label"] = assets["ticker"] + " - " + assets["name"]
asset_by_label = assets.set_index("label").to_dict(orient="index")

st.title("Live Commodity Stream")
st.caption("Yahoo Finance intraday futures data can be delayed or revised; treat this as decision support, not exchange-certified tick data.")
render_provider_warnings()

if "live_refresh_key" not in st.session_state:
    st.session_state["live_refresh_key"] = "initial"

sector_options = sorted(assets["sector"].dropna().unique().tolist())
control_cols = st.columns([1.3, 1.5, 1.1, 1.0])
with control_cols[0]:
    selected_sectors = st.multiselect("Sectors", sector_options, default=sector_options)

filtered_assets = assets[assets["sector"].isin(selected_sectors)] if selected_sectors else assets
if filtered_assets.empty:
    st.error("No commodity available for the selected sector filter.")
    st.stop()

label_options = filtered_assets["label"].tolist()
with control_cols[1]:
    primary_label = st.selectbox("Commodity", label_options, index=0)

with control_cols[2]:
    timeframe_label = st.selectbox("Timeframe", list(LIVE_TIMEFRAME_PRESETS.keys()), index=1)

preset = LIVE_TIMEFRAME_PRESETS[timeframe_label]
period = preset["period"]
interval = preset["interval"]
with control_cols[3]:
    chart_type = st.radio("Chart", ["Candles", "Line"], horizontal=True)

advanced = st.toggle("Advanced period / interval", value=False)
if advanced:
    advanced_cols = st.columns(4)
    with advanced_cols[0]:
        period = st.selectbox("Yahoo period", YAHOO_PERIODS, index=YAHOO_PERIODS.index(period) if period in YAHOO_PERIODS else 1)
    with advanced_cols[1]:
        interval = st.selectbox(
            "Yahoo interval",
            YAHOO_INTERVALS,
            index=YAHOO_INTERVALS.index(interval) if interval in YAHOO_INTERVALS else 2,
        )
    with advanced_cols[2]:
        prepost = st.toggle("Extended session", value=False)
    with advanced_cols[3]:
        show_volume = st.toggle("Volume pane", value=True)
else:
    prepost = False
    show_volume = True

refresh_cols = st.columns([1.2, 1.2, 1.2, 3])
with refresh_cols[0]:
    if st.button("Refresh now", type="primary"):
        st.session_state["live_refresh_key"] = pd.Timestamp.utcnow().isoformat()
with refresh_cols[1]:
    auto_refresh = st.toggle("Auto-refresh", value=False)
with refresh_cols[2]:
    refresh_seconds = st.slider("Seconds", min_value=15, max_value=300, value=60, step=15)

overlay_options = ["Moving averages", "Bollinger Bands", "Donchian Channel", "VWAP"]
overlays = st.multiselect("Overlays", overlay_options, default=["Moving averages", "Bollinger Bands", "VWAP"])

primary_asset = asset_by_label[primary_label]
ticker = str(primary_asset["ticker"])
name = str(primary_asset["name"])

with st.spinner(f"Streaming {ticker} on {period} / {interval}..."):
    frame, warnings = cached_live_asset(ticker, period, interval, prepost, st.session_state["live_refresh_key"])

if warnings:
    st.warning("\n".join(warnings))
if frame.empty:
    st.error("No live data available for the selected commodity/timeframe.")
    st.stop()

indicators = add_technical_indicators(frame)
snapshot = technical_snapshot(indicators, interval=interval)

metric_cols = st.columns(7)
metric_cols[0].metric("Last", _fmt_number(snapshot.get("Last Price")), delta=_fmt_pct(snapshot.get("Last Return")))
metric_cols[1].metric("Period", _fmt_pct(snapshot.get("Period Return")))
metric_cols[2].metric("Signal", str(snapshot.get("Signal", "-")))
metric_cols[3].metric("Score", _fmt_number(snapshot.get("Technical Score"), 1))
metric_cols[4].metric("Confidence", _fmt_number(snapshot.get("Confidence Score"), 0))
metric_cols[5].metric("Risk", str(snapshot.get("Risk State", "-")))
metric_cols[6].metric("Age", _fmt_minutes(snapshot.get("Data Age Minutes")))

age = snapshot.get("Data Age Minutes")
if pd.notna(age) and np.isfinite(age) and float(age) > _stale_threshold_minutes(interval):
    st.warning("The latest provider timestamp looks stale for this interval. Check market hours, holidays or Yahoo availability.")

st.info(str(snapshot.get("Decision Note", "")))

st.plotly_chart(
    live_technical_chart(
        indicators,
        f"{ticker} - {name} technical stream ({period} / {interval})",
        chart_type=chart_type,
        overlays=overlays,
        show_volume=show_volume,
    ),
    width="stretch",
)

tabs = st.tabs(["Technical Cockpit", "Levels", "Watchlist"])
with tabs[0]:
    component_table = technical_component_table(snapshot)
    component_table["Value"] = component_table["Value"].map(
        lambda value: _fmt_number(value, 1) if isinstance(value, (int, float, np.integer, np.floating)) else value
    )
    st.dataframe(component_table, width="stretch", hide_index=True)

with tabs[1]:
    levels = pd.DataFrame(
        [
            {"Level": "Support 20 bars", "Value": _fmt_number(snapshot.get("Support")), "Use": "Nearest observed support"},
            {"Level": "Resistance 20 bars", "Value": _fmt_number(snapshot.get("Resistance")), "Use": "Nearest observed resistance"},
            {"Level": "ATR", "Value": _fmt_number(snapshot.get("ATR")), "Use": "Current range proxy"},
            {"Level": "ATR / Price", "Value": _fmt_pct(snapshot.get("ATR Percent")), "Use": "Range intensity"},
            {"Level": "Long stop reference", "Value": _fmt_number(snapshot.get("ATR Stop Long")), "Use": "2x ATR below last"},
            {"Level": "Short stop reference", "Value": _fmt_number(snapshot.get("ATR Stop Short")), "Use": "2x ATR above last"},
            {"Level": "Last timestamp", "Value": _display_timestamp(snapshot.get("Last Timestamp")), "Use": "Provider timestamp"},
        ]
    )
    st.dataframe(levels, width="stretch", hide_index=True)

with tabs[2]:
    primary_sector = str(primary_asset.get("sector", ""))
    default_watchlist = filtered_assets.loc[filtered_assets["sector"] == primary_sector, "label"].head(5).tolist()
    if primary_label not in default_watchlist:
        default_watchlist = [primary_label] + default_watchlist[:4]
    watchlist_labels = st.multiselect("Watchlist", label_options, default=default_watchlist)
    if len(watchlist_labels) > 10:
        st.warning("Watchlist capped to the first 10 selected commodities for provider stability.")
        watchlist_labels = watchlist_labels[:10]

    rows: list[dict[str, Any]] = []
    for label in watchlist_labels:
        asset = asset_by_label[label]
        watch_ticker = str(asset["ticker"])
        watch_frame, watch_warnings = cached_live_asset(
            watch_ticker,
            period,
            interval,
            prepost,
            st.session_state["live_refresh_key"],
        )
        watch_snapshot = technical_snapshot(add_technical_indicators(watch_frame), interval=interval)
        rows.append(
            {
                "Ticker": watch_ticker,
                "Name": asset["name"],
                "Sector": asset["sector"],
                "Last": _fmt_number(watch_snapshot.get("Last Price")),
                "Period": _fmt_pct(watch_snapshot.get("Period Return")),
                "RSI": _fmt_number(watch_snapshot.get("RSI"), 1),
                "ATR / Price": _fmt_pct(watch_snapshot.get("ATR Percent")),
                "Score": _fmt_number(watch_snapshot.get("Technical Score"), 1),
                "Confidence": _fmt_number(watch_snapshot.get("Confidence Score"), 0),
                "Signal": watch_snapshot.get("Signal", "-"),
                "Risk": watch_snapshot.get("Risk State", "-"),
                "Age": _fmt_minutes(watch_snapshot.get("Data Age Minutes")),
                "Warnings": "; ".join(watch_warnings),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.caption(
    f"Refresh key: {st.session_state['live_refresh_key']} | Last app refresh: "
    f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
