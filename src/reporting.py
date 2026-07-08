from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import fetch_market_data, load_assets
from .data_quality import data_quality_report, enrich_assets_with_quality
from .scoring import build_summary_table
from .utils import load_yaml


PROVIDER_WARNING = (
    "Yahoo Finance futures tickers can represent provider-built continuous series. "
    "Returns may be affected by contract changes, adjustments and roll effects; "
    "do not interpret a contract change as certain economic performance."
)

CARRY_WARNING = "Carry indisponible avec la source de donnees actuelle."


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_configuration(root: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(root) if root else project_root()
    assets = load_assets(root_path / "config" / "assets.yaml")
    settings = load_yaml(root_path / "config" / "settings.yaml")
    return assets, settings


def build_dashboard_payload(
    root: str | Path | None = None,
    start_date: str | None = None,
    frequency: str = "daily",
    selected_sectors: list[str] | None = None,
    weights: dict[str, float] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    root_path = Path(root) if root else project_root()
    assets, settings = load_configuration(root_path)
    if selected_sectors:
        assets = assets[assets["sector"].isin(selected_sectors)].reset_index(drop=True)
    cache_dir = root_path / "data" / "cache" if settings.get("data", {}).get("local_cache", True) else None
    data, download_log = fetch_market_data(
        assets,
        start=start_date or settings.get("data", {}).get("default_start_date"),
        frequency=frequency,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    quality = data_quality_report(
        data,
        assets,
        z_threshold=settings.get("data", {}).get("outlier_zscore", 8.0),
        abs_threshold=settings.get("data", {}).get("abnormal_return_threshold", 0.25),
    )
    enriched_assets = enrich_assets_with_quality(assets, quality)
    summary = build_summary_table(
        enriched_assets,
        data,
        quality=quality,
        weights=weights or settings.get("scoring", {}).get("weights"),
        frequency=frequency,
        trend_settings=settings.get("trend", {}),
    )
    return {
        "assets": enriched_assets,
        "settings": settings,
        "data": data,
        "download_log": download_log,
        "quality": quality,
        "summary": summary,
    }


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(frame: pd.DataFrame, sheet_name: str = "Summary") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def format_display_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pct_columns = [
        column
        for column in result.columns
        if any(token in column for token in ["Return", "Volatility", "Drawdown", "Distance", "Rate", "Percentile"])
    ]
    for column in pct_columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    score_columns = [column for column in result.columns if "Score" in column or column == "Trend Strength"]
    for column in score_columns:
        if column in result and pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    price_columns = ["Last Price", "ATR"]
    for column in price_columns:
        if column in result and pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    return result


def filter_summary(
    summary: pd.DataFrame,
    sectors: list[str] | None = None,
    signals: list[str] | None = None,
    risk_levels: list[str] | None = None,
) -> pd.DataFrame:
    result = summary.copy()
    if sectors:
        result = result[result["Sector"].isin(sectors)]
    if signals:
        result = result[result["Signal Classification"].isin(signals)]
    if risk_levels:
        result = result[result["Risk Level"].isin(risk_levels)]
    return result


try:  # Streamlit is optional for unit tests and CLI imports.
    import streamlit as st
except Exception:  # pragma: no cover - depends on runtime environment
    st = None


if st is not None:  # pragma: no cover - exercised by Streamlit runtime

    @st.cache_data(show_spinner=False, ttl=3600)
    def cached_dashboard_payload(
        start_date: str,
        frequency: str,
        sectors: tuple[str, ...],
        weights: tuple[tuple[str, float], ...],
        refresh_key: str,
    ) -> dict[str, Any]:
        return build_dashboard_payload(
            start_date=start_date,
            frequency=frequency,
            selected_sectors=list(sectors),
            weights=dict(weights),
            refresh=refresh_key != "initial",
        )

else:

    def cached_dashboard_payload(
        start_date: str,
        frequency: str,
        sectors: tuple[str, ...],
        weights: tuple[tuple[str, float], ...],
        refresh_key: str,
    ) -> dict[str, Any]:
        return build_dashboard_payload(
            start_date=start_date,
            frequency=frequency,
            selected_sectors=list(sectors),
            weights=dict(weights),
            refresh=refresh_key != "initial",
        )


def configure_page(title: str = "Commodity Systematic Macro Dashboard") -> None:
    if st is None:
        return
    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            border: 1px solid #d8dde6;
            border-radius: 6px;
            padding: 12px 14px;
            background: #ffffff;
        }
        div[data-testid="stMetric"] label {
            color: #3b4658;
            font-size: 0.82rem;
        }
        .institutional-note {
            border-left: 4px solid #b11226;
            background: #f7f8fa;
            padding: 0.75rem 1rem;
            color: #1d2733;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    if st is None:
        return {}
    assets, settings = load_configuration()
    st.sidebar.title("Systematic Macro Controls")
    st.sidebar.caption("Yahoo Finance data, deterministic signals, no performance promise.")

    if st.sidebar.button("Reset to Institutional Defaults"):
        for key in list(st.session_state.keys()):
            if key.startswith("control_"):
                del st.session_state[key]
        st.rerun()

    default_start = pd.to_datetime(settings.get("data", {}).get("default_start_date", "2000-01-01")).date()
    start_date = st.sidebar.date_input("Start date", default_start, key="control_start_date")
    frequency = st.sidebar.selectbox(
        "Frequency",
        ["daily", "weekly", "monthly"],
        index=["daily", "weekly", "monthly"].index(settings.get("data", {}).get("frequency", "daily")),
        key="control_frequency",
    )
    sectors = sorted(assets["sector"].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors, key="control_sectors")

    st.sidebar.subheader("Composite weights")
    default_weights = settings.get("scoring", {}).get("weights", {})
    weight_keys = [
        ("trend", "Trend"),
        ("time_series_momentum", "TS Momentum"),
        ("cross_sectional_momentum", "XS Momentum"),
        ("breakout", "Breakout"),
        ("seasonality", "Seasonality"),
        ("risk_adjustment", "Risk Adjustment"),
    ]
    weights: dict[str, float] = {}
    for key, label in weight_keys:
        weights[key] = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value=1.0,
            value=float(default_weights.get(key, 0.0)),
            step=0.05,
            key=f"control_weight_{key}",
        )

    st.sidebar.subheader("Portfolio")
    target_vol = st.sidebar.slider("Target volatility", 0.02, 0.30, float(settings.get("portfolio", {}).get("target_volatility", 0.10)), 0.01)
    max_asset = st.sidebar.slider("Max weight per asset", 0.02, 0.50, float(settings.get("portfolio", {}).get("max_weight_per_asset", 0.15)), 0.01)
    max_sector = st.sidebar.slider("Max weight per sector", 0.05, 1.00, float(settings.get("portfolio", {}).get("max_weight_per_sector", 0.35)), 0.05)
    long_only = st.sidebar.toggle("Long-only mode", value=bool(settings.get("portfolio", {}).get("long_only", False)))
    transaction_cost_bps = st.sidebar.number_input(
        "Transaction cost bps",
        min_value=0.0,
        max_value=100.0,
        value=float(settings.get("portfolio", {}).get("transaction_cost_bps", 5)),
        step=1.0,
    )
    rebalancing = st.sidebar.selectbox(
        "Rebalancing",
        ["daily", "weekly", "monthly"],
        index=["daily", "weekly", "monthly"].index(settings.get("portfolio", {}).get("rebalancing_frequency", "monthly")),
    )

    if "refresh_key" not in st.session_state:
        st.session_state["refresh_key"] = "initial"
    if st.sidebar.button("Refresh Market Data"):
        st.session_state["refresh_key"] = pd.Timestamp.utcnow().isoformat()

    return {
        "start_date": start_date.isoformat(),
        "frequency": frequency,
        "sectors": selected_sectors,
        "weights": weights,
        "target_volatility": target_vol,
        "max_weight_per_asset": max_asset,
        "max_weight_per_sector": max_sector,
        "long_only": long_only,
        "transaction_cost_bps": transaction_cost_bps,
        "rebalancing_frequency": rebalancing,
        "refresh_key": st.session_state["refresh_key"],
    }


def get_payload_from_controls(controls: dict[str, Any]) -> dict[str, Any]:
    weights_tuple = tuple(sorted((controls.get("weights") or {}).items()))
    sectors_tuple = tuple(controls.get("sectors") or [])
    return cached_dashboard_payload(
        controls.get("start_date", "2000-01-01"),
        controls.get("frequency", "daily"),
        sectors_tuple,
        weights_tuple,
        controls.get("refresh_key", "initial"),
    )


def render_provider_warnings() -> None:
    if st is None:
        return
    st.markdown(f'<div class="institutional-note">{PROVIDER_WARNING}<br>{CARRY_WARNING}</div>', unsafe_allow_html=True)
