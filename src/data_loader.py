from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_datetime_index, load_yaml, yfinance_interval

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_assets(path: str | Path) -> pd.DataFrame:
    config = load_yaml(path)
    assets = pd.DataFrame(config.get("assets", []))
    if assets.empty:
        raise ValueError(f"No assets defined in {path}")
    assets["data_available"] = "Unknown"
    assets["first_observation"] = pd.NaT
    assets["last_observation"] = pd.NaT
    return assets


def normalize_yfinance_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        levels = frame.columns.names
        if "Ticker" in levels:
            try:
                frame = frame.xs(ticker, axis=1, level="Ticker")
            except KeyError:
                frame = frame.droplevel(-1, axis=1)
        elif ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            frame.columns = frame.columns.get_level_values(0)

    frame = ensure_datetime_index(frame)
    rename = {column: column.title().replace("Adj Close", "Adj Close") for column in frame.columns}
    frame = frame.rename(columns=rename)

    if "Close" not in frame.columns and "Adj Close" in frame.columns:
        frame["Close"] = frame["Adj Close"]
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[REQUIRED_COLUMNS + [c for c in ["Adj Close"] if c in frame.columns]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _cache_path(cache_dir: str | Path, ticker: str, start: str | None, end: str | None, interval: str) -> Path:
    safe_ticker = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
    name = f"{safe_ticker}_{start or 'max'}_{end or 'latest'}_{interval}.csv"
    return Path(cache_dir) / name


def _read_local_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return normalize_yfinance_frame(frame, ticker="")
    except Exception as exc:  # pragma: no cover - defensive cache recovery
        LOGGER.warning("Could not read local cache %s: %s", path, exc)
        return None


def _write_local_cache(path: Path, frame: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
    except Exception as exc:  # pragma: no cover - cache write should never break analysis
        LOGGER.warning("Could not write local cache %s: %s", path, exc)


def download_single_asset(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    frequency: str = "daily",
    retries: int = 3,
    pause_seconds: float = 1.0,
    local_cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    interval = yfinance_interval(frequency)
    warnings: list[str] = []

    cache_file: Path | None = None
    if local_cache_dir:
        cache_file = _cache_path(local_cache_dir, ticker, start, end, interval)
        cached = None if refresh else _read_local_cache(cache_file)
        if cached is not None and not cached.empty:
            return cached, warnings

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in runtime, not unit tests
        raise RuntimeError("yfinance is required to download market data") from exc

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            frame = normalize_yfinance_frame(raw, ticker)
            if frame.empty or frame["Close"].dropna().empty:
                warnings.append("No usable close prices returned by Yahoo Finance")
            else:
                invalid = int((frame["Close"] <= 0).sum())
                if invalid:
                    warnings.append(f"{invalid} non-positive close prices set to missing")
                    frame.loc[frame["Close"] <= 0, "Close"] = pd.NA
                if cache_file is not None:
                    _write_local_cache(cache_file, frame)
                return frame, warnings
        except Exception as exc:  # pragma: no cover - depends on network/provider
            last_error = exc
            LOGGER.warning("Download failed for %s on attempt %s: %s", ticker, attempt, exc)
            time.sleep(pause_seconds * attempt)

    message = f"Download failed after {retries} attempts"
    if last_error is not None:
        message = f"{message}: {last_error}"
    warnings.append(message)
    return pd.DataFrame(columns=REQUIRED_COLUMNS), warnings


def fetch_market_data(
    assets: pd.DataFrame | list[dict[str, Any]],
    start: str | None = None,
    end: str | None = None,
    frequency: str = "daily",
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    retries: int = 3,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    asset_frame = pd.DataFrame(assets).copy()
    data: dict[str, pd.DataFrame] = {}
    logs: list[dict[str, Any]] = []

    for _, asset in asset_frame.iterrows():
        ticker = str(asset["ticker"])
        frame, warnings = download_single_asset(
            ticker=ticker,
            start=start,
            end=end,
            frequency=frequency,
            retries=retries,
            local_cache_dir=cache_dir,
            refresh=refresh,
        )
        data[ticker] = frame
        logs.append(
            {
                "ticker": ticker,
                "name": asset.get("name", ticker),
                "rows": int(len(frame)),
                "first_observation": frame.index.min() if not frame.empty else pd.NaT,
                "last_observation": frame.index.max() if not frame.empty else pd.NaT,
                "warnings": "; ".join(warnings),
                "data_available": "Yes" if not frame.empty and frame["Close"].notna().any() else "No",
            }
        )

    return data, pd.DataFrame(logs)
