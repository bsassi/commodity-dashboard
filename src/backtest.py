from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .drawdown import drawdown_series
from .trend import donchian_signal, price_above_sma_signal, sma_crossover_signal
from .utils import annualization_factor, clean_price_series


def time_series_momentum_signal(price: pd.Series, window: int = 252) -> pd.Series:
    clean = clean_price_series(price)
    momentum = clean / clean.shift(window) - 1
    return pd.Series(np.where(momentum > 0, 1.0, np.where(momentum < 0, -1.0, 0.0)), index=clean.index)


def trend_ensemble_signal(
    price: pd.Series,
    windows: list[int] | None = None,
    volatility_window: int = 60,
    deadband: float = 0.15,
) -> pd.Series:
    windows = windows or [21, 63, 126, 252]
    clean = clean_price_series(price)
    if clean.empty:
        return pd.Series(dtype=float)

    returns = clean.pct_change(fill_method=None)
    vol = returns.rolling(volatility_window, min_periods=max(20, volatility_window // 2)).std()
    components = []
    for window in windows:
        raw = clean / clean.shift(window) - 1
        scaled = raw / (vol * np.sqrt(window)).replace(0, np.nan)
        components.append(np.tanh(scaled).rename(f"mom_{window}"))

    score = pd.concat(components, axis=1).mean(axis=1)
    sma_100 = clean.rolling(100, min_periods=40).mean()
    sma_200 = clean.rolling(200, min_periods=80).mean()
    ma_posture = pd.Series(0.0, index=clean.index)
    ma_posture = ma_posture.mask(clean > sma_100, 0.35).mask(clean < sma_100, -0.35)
    crossover = pd.Series(0.0, index=clean.index)
    crossover = crossover.mask(sma_100 > sma_200, 0.15).mask(sma_100 < sma_200, -0.15)
    ma_posture = ma_posture.add(crossover, fill_value=0.0)
    combined = (0.75 * score + 0.25 * ma_posture).clip(-1, 1).fillna(0.0)
    combined = combined.mask(combined.abs() < deadband, 0.0)
    return combined.clip(-1, 1)


def signal_for_strategy(price: pd.Series, strategy: str, params: dict[str, int] | None = None) -> pd.Series:
    params = params or {}
    if strategy == "price_above_sma":
        return price_above_sma_signal(price, params.get("window", 200))
    if strategy == "sma_crossover":
        return sma_crossover_signal(price, params.get("short_window", 50), params.get("long_window", 200))
    if strategy == "donchian_breakout":
        return donchian_signal(price, params.get("window", 55))
    if strategy == "time_series_momentum":
        return time_series_momentum_signal(price, params.get("window", 252))
    if strategy == "trend_ensemble":
        windows = params.get("windows")
        if windows is None:
            windows = [21, 63, 126, params.get("window", 252)]
        return trend_ensemble_signal(
            price,
            windows=windows,
            volatility_window=params.get("volatility_window", 60),
            deadband=params.get("deadband", 0.15),
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def apply_signal_without_lookahead(
    price: pd.Series,
    signal: pd.Series,
    transaction_cost_bps: float = 0.0,
    rebalancing_frequency: str = "daily",
) -> pd.DataFrame:
    clean = clean_price_series(price)
    returns = clean.pct_change(fill_method=None).fillna(0.0)
    signal = signal.reindex(clean.index).fillna(0.0).clip(-1, 1)
    if rebalancing_frequency == "weekly":
        signal = signal.resample("W-FRI").last().reindex(clean.index).ffill().fillna(0.0)
    elif rebalancing_frequency == "monthly":
        signal = signal.resample("ME").last().reindex(clean.index).ffill().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * transaction_cost_bps / 10000
    strategy_returns = position * returns - costs
    return pd.DataFrame(
        {
            "asset_return": returns,
            "raw_signal": signal,
            "position": position,
            "turnover": turnover,
            "transaction_cost": costs,
            "strategy_return": strategy_returns,
            "equity": (1 + strategy_returns).cumprod(),
        }
    )


def backtest_single_asset(
    frame: pd.DataFrame,
    strategy: str = "time_series_momentum",
    params: dict[str, int] | None = None,
    transaction_cost_bps: float = 5.0,
    rebalancing_frequency: str = "monthly",
) -> pd.DataFrame:
    price = frame["Close"].dropna()
    signal = signal_for_strategy(price, strategy, params)
    return apply_signal_without_lookahead(price, signal, transaction_cost_bps, rebalancing_frequency)


def backtest_universe(
    data: dict[str, pd.DataFrame],
    strategy: str = "time_series_momentum",
    params: dict[str, int] | None = None,
    transaction_cost_bps: float = 5.0,
    rebalancing_frequency: str = "monthly",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    asset_results: dict[str, pd.DataFrame] = {}
    returns = []
    for ticker, frame in data.items():
        if frame.empty or "Close" not in frame.columns:
            continue
        result = backtest_single_asset(frame, strategy, params, transaction_cost_bps, rebalancing_frequency)
        if result.empty:
            continue
        asset_results[ticker] = result
        returns.append(result["strategy_return"].rename(ticker))
    if not returns:
        return pd.DataFrame(), asset_results
    matrix = pd.concat(returns, axis=1).dropna(how="all").fillna(0.0)
    portfolio = matrix.mean(axis=1)
    output = pd.DataFrame({"strategy_return": portfolio})
    output["equity"] = (1 + output["strategy_return"]).cumprod()
    output["drawdown"] = drawdown_series(output["strategy_return"])
    positions = pd.concat({ticker: result["position"] for ticker, result in asset_results.items()}, axis=1).reindex(output.index)
    turnovers = pd.concat({ticker: result["turnover"] for ticker, result in asset_results.items()}, axis=1).reindex(output.index)
    output["average_abs_position"] = positions.abs().mean(axis=1)
    output["average_turnover"] = turnovers.mean(axis=1)
    output["active_assets"] = positions.abs().gt(0).sum(axis=1)
    return output, asset_results


def performance_metrics(returns: pd.Series, frequency: str = "daily") -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {}
    factor = annualization_factor(frequency)
    equity = (1 + clean).cumprod()
    years = len(clean) / factor
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else np.nan
    vol = float(clean.std() * np.sqrt(factor))
    sharpe = float((clean.mean() * factor) / vol) if vol and np.isfinite(vol) else np.nan
    downside = clean[clean < 0].std() * np.sqrt(factor)
    sortino = float((clean.mean() * factor) / downside) if downside and np.isfinite(downside) else np.nan
    dd = drawdown_series(clean)
    max_dd = float(dd.min()) if not dd.empty else np.nan
    calmar = float(cagr / abs(max_dd)) if max_dd and np.isfinite(max_dd) and max_dd < 0 else np.nan
    return {
        "CAGR": cagr,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Maximum Drawdown": max_dd,
        "Downside Deviation": float(downside) if np.isfinite(downside) else np.nan,
        "Hit Ratio": float((clean > 0).mean()),
        "Skewness": float(stats.skew(clean, nan_policy="omit")) if len(clean) > 3 else np.nan,
        "Kurtosis": float(stats.kurtosis(clean, nan_policy="omit")) if len(clean) > 3 else np.nan,
        "Average Exposure": np.nan,
        "Maximum Exposure": np.nan,
    }


def portfolio_diagnostics(backtest: pd.DataFrame) -> dict[str, float]:
    if backtest.empty:
        return {}
    return {
        "Average Exposure": float(backtest["average_abs_position"].mean()) if "average_abs_position" in backtest else np.nan,
        "Maximum Exposure": float(backtest["average_abs_position"].max()) if "average_abs_position" in backtest else np.nan,
        "Average Turnover": float(backtest["average_turnover"].mean()) if "average_turnover" in backtest else np.nan,
        "Average Active Assets": float(backtest["active_assets"].mean()) if "active_assets" in backtest else np.nan,
    }


def sensitivity_grid(frame: pd.DataFrame, short_windows: list[int], long_windows: list[int]) -> pd.DataFrame:
    rows = []
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue
            bt = backtest_single_asset(
                frame,
                strategy="sma_crossover",
                params={"short_window": short, "long_window": long},
                transaction_cost_bps=0.0,
                rebalancing_frequency="daily",
            )
            metrics = performance_metrics(bt["strategy_return"])
            rows.append({"Short Window": short, "Long Window": long, "Sharpe Ratio": metrics.get("Sharpe Ratio", np.nan)})
    return pd.DataFrame(rows)
