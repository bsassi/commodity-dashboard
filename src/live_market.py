from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import normalize_yfinance_frame
from .utils import clip_score


LIVE_TIMEFRAME_PRESETS: dict[str, dict[str, str]] = {
    "1D / 1m": {"period": "1d", "interval": "1m"},
    "5D / 5m": {"period": "5d", "interval": "5m"},
    "1M / 30m": {"period": "1mo", "interval": "30m"},
    "3M / 60m": {"period": "3mo", "interval": "60m"},
    "6M / 1d": {"period": "6mo", "interval": "1d"},
    "1Y / 1d": {"period": "1y", "interval": "1d"},
    "5Y / 1wk": {"period": "5y", "interval": "1wk"},
}

YAHOO_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
YAHOO_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo"]

INTERVALS_PER_YEAR = {
    "1m": 252 * 390,
    "2m": 252 * 195,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "60m": 252 * 6.5,
    "90m": 252 * 4.3,
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


def bars_per_year(interval: str) -> float:
    return float(INTERVALS_PER_YEAR.get(interval, 252))


def is_intraday_interval(interval: str) -> bool:
    return interval.endswith("m")


def download_live_asset(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    retries: int = 3,
    pause_seconds: float = 0.75,
    prepost: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("yfinance is required to stream commodity data") from exc

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                prepost=prepost,
            )
            frame = normalize_yfinance_frame(raw, ticker)
            frame = frame.dropna(subset=["Close"])
            if frame.empty:
                warnings.append("No usable live prices returned by Yahoo Finance")
            else:
                invalid = int((frame["Close"] <= 0).sum())
                if invalid:
                    warnings.append(f"{invalid} non-positive close prices removed")
                    frame = frame.loc[frame["Close"] > 0]
                if len(frame) < 30:
                    warnings.append("Short live history: some indicators may be unstable")
                return frame, warnings
        except Exception as exc:  # pragma: no cover - depends on provider/network
            last_error = exc
            time.sleep(pause_seconds * attempt)

    message = f"Live download failed after {retries} attempts"
    if last_error is not None:
        message = f"{message}: {last_error}"
    warnings.append(message)
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), warnings


def relative_strength_index(price: pd.Series, window: int = 14) -> pd.Series:
    clean = pd.to_numeric(price, errors="coerce")
    delta = clean.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi.clip(0, 100)


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    clean = pd.to_numeric(price, errors="coerce")
    fast_ema = clean.ewm(span=fast, min_periods=max(3, fast // 2), adjust=False).mean()
    slow_ema = clean.ewm(span=slow, min_periods=max(5, slow // 2), adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, min_periods=max(3, signal // 2), adjust=False).mean()
    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD Signal": signal_line,
            "MACD Histogram": macd_line - signal_line,
        }
    )


def add_technical_indicators(
    frame: pd.DataFrame,
    interval: str = "5m",
    fast_window: int = 20,
    slow_window: int = 50,
    rsi_window: int = 14,
    atr_window: int = 14,
    bollinger_window: int = 20,
    donchian_window: int = 20,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = normalize_yfinance_frame(frame, ticker="")
    close = pd.to_numeric(result["Close"], errors="coerce")
    high = pd.to_numeric(result["High"], errors="coerce").fillna(close)
    low = pd.to_numeric(result["Low"], errors="coerce").fillna(close)
    volume = pd.to_numeric(result["Volume"], errors="coerce").fillna(0)

    result["Return"] = close.pct_change(fill_method=None)
    result["SMA Fast"] = close.rolling(fast_window, min_periods=max(5, fast_window // 3)).mean()
    result["SMA Slow"] = close.rolling(slow_window, min_periods=max(10, slow_window // 3)).mean()
    result["EMA Fast"] = close.ewm(span=fast_window, min_periods=max(5, fast_window // 3), adjust=False).mean()
    result["RSI"] = relative_strength_index(close, rsi_window)
    result = result.join(macd(close))

    mid = close.rolling(bollinger_window, min_periods=max(5, bollinger_window // 2)).mean()
    sigma = close.rolling(bollinger_window, min_periods=max(5, bollinger_window // 2)).std()
    result["BB Mid"] = mid
    result["BB Upper"] = mid + 2 * sigma
    result["BB Lower"] = mid - 2 * sigma

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR"] = true_range.rolling(atr_window, min_periods=max(5, atr_window // 2)).mean()
    result["ATR Percent"] = result["ATR"] / close
    result["Realized Volatility"] = result["Return"].rolling(30, min_periods=10).std()

    min_periods = max(5, donchian_window // 2)
    result["Donchian High"] = high.shift(1).rolling(donchian_window, min_periods=min_periods).max()
    result["Donchian Low"] = low.shift(1).rolling(donchian_window, min_periods=min_periods).min()

    typical_price = (high + low + close) / 3
    valid_volume = volume.where(volume > 0)
    if is_intraday_interval(interval):
        session = pd.Series(result.index.date, index=result.index)
        session_volume = valid_volume.groupby(session).cumsum()
        session_value = (typical_price * valid_volume).groupby(session).cumsum()
        result["VWAP"] = session_value / session_volume
    else:
        rolling_volume = valid_volume.rolling(20, min_periods=5).sum()
        rolling_value = (typical_price * valid_volume).rolling(20, min_periods=5).sum()
        result["VWAP"] = rolling_value / rolling_volume
    return result


def _last_number(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) and np.isfinite(value) else np.nan


def _relation_score(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    if left > right:
        return 100.0
    if left < right:
        return -100.0
    return 0.0


def _mean_score(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    if not clean:
        return 0.0
    return clip_score(float(np.nanmean(clean)))


def _risk_state(vol_percentile: float, atr_percent: float) -> tuple[str, float]:
    if np.isfinite(vol_percentile) and vol_percentile >= 0.90:
        return "Extreme", 20.0
    if np.isfinite(atr_percent) and atr_percent >= 0.040:
        return "Extreme", 20.0
    if np.isfinite(vol_percentile) and vol_percentile >= 0.75:
        return "Elevated", 10.0
    if np.isfinite(atr_percent) and atr_percent >= 0.020:
        return "Elevated", 10.0
    if np.isfinite(vol_percentile) and vol_percentile <= 0.25:
        return "Compressed", 0.0
    return "Normal", 0.0


def classify_live_signal(score: float) -> str:
    if score >= 60:
        return "Strong Long"
    if score >= 25:
        return "Long Bias"
    if score <= -60:
        return "Strong Short"
    if score <= -25:
        return "Short Bias"
    return "Neutral / Wait"


def market_data_age_minutes(timestamp: pd.Timestamp | None) -> float:
    if timestamp is None or pd.isna(timestamp):
        return np.nan
    last = pd.Timestamp(timestamp)
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return max(float((now - last).total_seconds() / 60), 0.0)


def technical_snapshot(indicators: pd.DataFrame, interval: str = "5m") -> dict[str, Any]:
    if indicators.empty or "Close" not in indicators.columns:
        return {
            "Signal": "Neutral / Wait",
            "Technical Score": 0.0,
            "Confidence Score": 0.0,
            "Risk State": "Insufficient Data",
            "Decision Note": "No usable live history.",
        }

    clean = indicators.dropna(subset=["Close"]).copy()
    if clean.empty:
        return {
            "Signal": "Neutral / Wait",
            "Technical Score": 0.0,
            "Confidence Score": 0.0,
            "Risk State": "Insufficient Data",
            "Decision Note": "No usable live close prices.",
        }

    last = clean.iloc[-1]
    close = pd.to_numeric(clean["Close"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    last_price = float(close.iloc[-1])
    previous_price = float(close.iloc[-2]) if len(close) > 1 and pd.notna(close.iloc[-2]) else np.nan
    last_return = (last_price / previous_price - 1) if np.isfinite(previous_price) and previous_price else np.nan
    period_return = (last_price / float(close.iloc[0]) - 1) if len(close) > 1 and close.iloc[0] else np.nan

    sma_fast = _last_number(last, "SMA Fast")
    sma_slow = _last_number(last, "SMA Slow")
    ema_fast = _last_number(last, "EMA Fast")
    rsi_value = _last_number(last, "RSI")
    macd_hist = _last_number(last, "MACD Histogram")
    macd_hist_prev = _last_number(clean.iloc[-2], "MACD Histogram") if len(clean) > 1 else np.nan
    donchian_high = _last_number(last, "Donchian High")
    donchian_low = _last_number(last, "Donchian Low")
    atr_value = _last_number(last, "ATR")
    atr_percent = _last_number(last, "ATR Percent")

    slow_slope = np.nan
    if "SMA Slow" in clean and clean["SMA Slow"].dropna().shape[0] > 5:
        slow_clean = clean["SMA Slow"].dropna()
        slow_slope = _relation_score(float(slow_clean.iloc[-1]), float(slow_clean.iloc[-6]))

    trend_score = _mean_score(
        [
            _relation_score(last_price, sma_fast),
            _relation_score(last_price, sma_slow),
            _relation_score(sma_fast, sma_slow),
            _relation_score(last_price, ema_fast),
            slow_slope,
        ]
    )

    rsi_score = clip_score((rsi_value - 50) * 2) if np.isfinite(rsi_value) else np.nan
    macd_score = np.nan
    if np.isfinite(macd_hist):
        hist_direction = 75.0 if macd_hist > 0 else -75.0 if macd_hist < 0 else 0.0
        hist_acceleration = 25.0 if np.isfinite(macd_hist_prev) and macd_hist > macd_hist_prev else -25.0
        macd_score = clip_score(hist_direction + hist_acceleration)
    period_score = clip_score(np.tanh(period_return * 8) * 100) if np.isfinite(period_return) else np.nan
    momentum_score = _mean_score([rsi_score, macd_score, period_score])

    breakout_score = 0.0
    if np.isfinite(donchian_high) and np.isfinite(donchian_low) and donchian_high > donchian_low:
        if last_price > donchian_high:
            breakout_score = 100.0
        elif last_price < donchian_low:
            breakout_score = -100.0
        else:
            channel_position = ((last_price - donchian_low) / (donchian_high - donchian_low) - 0.5) * 200
            breakout_score = clip_score(channel_position)

    realized = returns.rolling(30, min_periods=10).std() * np.sqrt(bars_per_year(interval))
    realized_last = float(realized.dropna().iloc[-1]) if not realized.dropna().empty else np.nan
    vol_percentile = (
        float((realized.dropna() <= realized_last).mean())
        if np.isfinite(realized_last) and len(realized.dropna()) >= 20
        else np.nan
    )
    risk_state, risk_penalty = _risk_state(vol_percentile, atr_percent)

    directional_score = clip_score(0.45 * trend_score + 0.35 * momentum_score + 0.20 * breakout_score)
    technical_score = directional_score
    if abs(directional_score) >= 10:
        technical_score = clip_score(directional_score - np.sign(directional_score) * risk_penalty)

    signal = classify_live_signal(technical_score)
    score_direction = np.sign(technical_score)
    component_signs = [np.sign(value) for value in [trend_score, momentum_score, breakout_score] if abs(value) > 5]
    agreement = float(component_signs.count(score_direction) / len(component_signs)) if component_signs and score_direction else 0.0
    confidence = clip_score(20 + abs(technical_score) * 0.45 + agreement * 35 + min(len(clean) / 120, 1) * 20 - risk_penalty, 0, 100)

    support = float(clean["Low"].dropna().tail(20).min()) if "Low" in clean and not clean["Low"].dropna().empty else np.nan
    resistance = float(clean["High"].dropna().tail(20).max()) if "High" in clean and not clean["High"].dropna().empty else np.nan
    stop_long = last_price - 2 * atr_value if np.isfinite(atr_value) else np.nan
    stop_short = last_price + 2 * atr_value if np.isfinite(atr_value) else np.nan
    timestamp = clean.index[-1]

    if signal in {"Strong Long", "Long Bias"}:
        note = "Upside bias while price action remains above key moving averages and risk is controlled."
    elif signal in {"Strong Short", "Short Bias"}:
        note = "Downside bias while price action remains below key moving averages and momentum confirms."
    else:
        note = "Mixed technical evidence; wait for a cleaner trend, breakout or momentum confirmation."
    if risk_state in {"Elevated", "Extreme"}:
        note = f"{note} Volatility state is {risk_state.lower()}, so sizing discipline matters."

    return {
        "Last Price": last_price,
        "Previous Price": previous_price,
        "Last Return": last_return,
        "Period Return": period_return,
        "RSI": rsi_value,
        "MACD Histogram": macd_hist,
        "ATR": atr_value,
        "ATR Percent": atr_percent,
        "Realized Volatility": realized_last,
        "Volatility Percentile": vol_percentile,
        "Trend Score": trend_score,
        "Momentum Score": momentum_score,
        "Breakout Score": breakout_score,
        "Technical Score": technical_score,
        "Confidence Score": confidence,
        "Signal": signal,
        "Risk State": risk_state,
        "Support": support,
        "Resistance": resistance,
        "ATR Stop Long": stop_long,
        "ATR Stop Short": stop_short,
        "Bars": int(len(clean)),
        "Last Timestamp": timestamp,
        "Data Age Minutes": market_data_age_minutes(timestamp),
        "Decision Note": note,
    }


def technical_component_table(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Trend Score", snapshot.get("Trend Score"), "Moving-average posture and slope"),
        ("Momentum Score", snapshot.get("Momentum Score"), "RSI, MACD histogram and period return"),
        ("Breakout Score", snapshot.get("Breakout Score"), "Position versus Donchian channel"),
        ("Risk State", snapshot.get("Risk State"), "Volatility percentile and ATR/price"),
        ("Confidence Score", snapshot.get("Confidence Score"), "Component agreement, sample size and risk penalty"),
    ]
    return pd.DataFrame(rows, columns=["Component", "Value", "Interpretation"])
