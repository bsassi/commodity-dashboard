from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .returns import horizon_return
from .utils import clean_price_series, classify_trend_strength, clip_score


def moving_averages(price: pd.Series, sma_windows: list[int], ema_windows: list[int]) -> pd.DataFrame:
    clean = clean_price_series(price)
    result = pd.DataFrame({"Price": clean})
    for window in sma_windows:
        result[f"SMA {window}"] = clean.rolling(window, min_periods=max(5, window // 3)).mean()
    for window in ema_windows:
        result[f"EMA {window}"] = clean.ewm(span=window, min_periods=max(5, window // 3), adjust=False).mean()
    return result


def price_vs_ma_score(price: pd.Series, windows: list[int] | None = None) -> tuple[float, dict[str, float]]:
    windows = windows or [20, 50, 100, 200]
    clean = clean_price_series(price)
    if len(clean) < min(windows):
        return 0.0, {}
    returns = clean.pct_change(fill_method=None)
    vol = returns.rolling(60, min_periods=20).std().iloc[-1]
    components: dict[str, float] = {}
    signs = []
    for window in windows:
        ma = clean.rolling(window, min_periods=max(5, window // 3)).mean()
        if pd.isna(ma.iloc[-1]):
            continue
        distance = (clean.iloc[-1] / ma.iloc[-1] - 1) / max(vol * np.sqrt(window), 1e-9)
        sign_score = float(np.tanh(distance) * 100)
        components[f"price_vs_sma_{window}"] = sign_score
        signs.append(np.sign(sign_score))
    if not signs:
        return 0.0, components
    crossover = 0.0
    sma50 = clean.rolling(50, min_periods=20).mean()
    sma200 = clean.rolling(200, min_periods=80).mean()
    if not pd.isna(sma50.iloc[-1]) and not pd.isna(sma200.iloc[-1]):
        crossover = 100.0 if sma50.iloc[-1] > sma200.iloc[-1] else -100.0
        components["sma_50_vs_200"] = crossover
    score = np.nanmean(list(components.values())) if components else 0.0
    return clip_score(score), components


def donchian_signal(price: pd.Series, window: int = 55) -> pd.Series:
    clean = clean_price_series(price)
    min_periods = min(window, max(2, window // 3))
    upper = clean.shift(1).rolling(window, min_periods=min_periods).max()
    lower = clean.shift(1).rolling(window, min_periods=min_periods).min()
    event = pd.Series(0.0, index=clean.index)
    event = event.mask(clean > upper, 1.0)
    event = event.mask(clean < lower, -1.0)
    state = event.replace(0, np.nan).ffill().fillna(0.0)
    return state


def breakout_score(price: pd.Series, windows: list[int] | None = None) -> tuple[float, dict[str, float]]:
    windows = windows or [20, 55, 100, 252]
    components: dict[str, float] = {}
    for window in windows:
        signal = donchian_signal(price, window)
        if signal.empty:
            continue
        components[f"donchian_{window}"] = float(signal.iloc[-1] * 100)
    if not components:
        return 0.0, components
    return clip_score(np.nanmean(list(components.values()))), components


def time_series_momentum_score(
    price: pd.Series,
    windows: list[int] | None = None,
    exclude_last_month: bool = False,
) -> tuple[float, dict[str, float]]:
    windows = windows or [21, 63, 126, 189, 252]
    clean = clean_price_series(price)
    components: dict[str, float] = {}
    for window in windows:
        if exclude_last_month and window > 21:
            if len(clean) <= window:
                value = np.nan
            else:
                value = clean.shift(21).iloc[-1] / clean.shift(window).iloc[-1] - 1
        else:
            value = horizon_return(clean, window)
        if pd.isna(value):
            continue
        components[f"momentum_{window}"] = float(np.tanh(value * 4) * 100)
    if not components:
        return 0.0, components
    return clip_score(np.nanmean(list(components.values()))), components


def regression_trend(price: pd.Series, window: int = 120) -> dict[str, float]:
    clean = clean_price_series(price).tail(window)
    if len(clean) < max(20, window // 2):
        return {"slope": np.nan, "t_stat": np.nan, "r_squared": np.nan, "direction": 0.0, "score": 0.0}
    y = np.log(clean)
    x = np.arange(len(y), dtype=float)
    result = stats.linregress(x, y)
    t_stat = result.slope / result.stderr if result.stderr not in (0, None) and np.isfinite(result.stderr) else np.nan
    r_squared = result.rvalue**2
    score = np.tanh((t_stat if np.isfinite(t_stat) else 0.0) / 3.0) * r_squared * 100
    return {
        "slope": float(result.slope),
        "annualized_slope": float(np.expm1(result.slope * 252)),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "r_squared": float(r_squared),
        "direction": float(np.sign(result.slope)),
        "score": clip_score(score),
    }


def regression_score(price: pd.Series, windows: list[int] | None = None) -> tuple[float, dict[str, float]]:
    windows = windows or [20, 60, 120, 252]
    components: dict[str, float] = {}
    for window in windows:
        reg = regression_trend(price, window)
        components[f"regression_{window}"] = reg["score"]
        components[f"r2_{window}"] = reg["r_squared"]
        components[f"tstat_{window}"] = reg["t_stat"]
    score_values = [value for key, value in components.items() if key.startswith("regression_") and np.isfinite(value)]
    if not score_values:
        return 0.0, components
    return clip_score(np.nanmean(score_values)), components


def horizon_coherence(components: dict[str, float]) -> float:
    directional = [np.sign(v) for v in components.values() if np.isfinite(v) and abs(v) > 5]
    if not directional:
        return 0.0
    positive = directional.count(1.0) / len(directional)
    negative = directional.count(-1.0) / len(directional)
    dominant = max(positive, negative)
    direction = 1.0 if positive >= negative else -1.0
    return float(direction * dominant * 100)


def trend_strength_snapshot(
    frame: pd.DataFrame,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    donchian_windows: list[int] | None = None,
    regression_windows: list[int] | None = None,
    momentum_windows: list[int] | None = None,
    exclude_last_month: bool = False,
) -> dict[str, object]:
    close = frame["Close"].dropna()
    if len(close) < 30:
        return {
            "Trend Score": 0.0,
            "Trend Signal": "Insufficient Data",
            "Trend Components": {},
            "MA Score": 0.0,
            "Time-Series Momentum Score": 0.0,
            "Breakout Score": 0.0,
            "Regression Score": 0.0,
            "Coherence Score": 0.0,
        }

    ma_score, ma_components = price_vs_ma_score(close, sma_windows)
    mom_score, mom_components = time_series_momentum_score(close, momentum_windows, exclude_last_month)
    brk_score, brk_components = breakout_score(close, donchian_windows)
    reg_score, reg_components = regression_score(close, regression_windows)
    coherence = horizon_coherence({**ma_components, **mom_components, **brk_components})
    score = clip_score(0.30 * ma_score + 0.30 * mom_score + 0.20 * brk_score + 0.15 * reg_score + 0.05 * coherence)
    components = {
        **ma_components,
        **mom_components,
        **brk_components,
        **reg_components,
        "coherence": coherence,
    }
    return {
        "Trend Score": score,
        "Trend Signal": classify_trend_strength(score),
        "Trend Components": components,
        "MA Score": ma_score,
        "Time-Series Momentum Score": mom_score,
        "Breakout Score": brk_score,
        "Regression Score": reg_score,
        "Coherence Score": coherence,
    }


def sma_crossover_signal(price: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    clean = clean_price_series(price)
    short = clean.rolling(short_window, min_periods=max(10, short_window // 3)).mean()
    long = clean.rolling(long_window, min_periods=max(20, long_window // 3)).mean()
    return pd.Series(np.where(short > long, 1.0, np.where(short < long, -1.0, 0.0)), index=clean.index)


def price_above_sma_signal(price: pd.Series, window: int = 200) -> pd.Series:
    clean = clean_price_series(price)
    sma = clean.rolling(window, min_periods=max(20, window // 3)).mean()
    return pd.Series(np.where(clean > sma, 1.0, np.where(clean < sma, -1.0, 0.0)), index=clean.index)
