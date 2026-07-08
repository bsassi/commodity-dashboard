from __future__ import annotations

import numpy as np
import pandas as pd


def wealth_index(returns: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1 + returns.fillna(0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = wealth_index(returns)
    running_max = wealth.cummax()
    return wealth / running_max - 1


def drawdown_duration(drawdowns: pd.Series) -> pd.Series:
    durations: list[int] = []
    current = 0
    for value in drawdowns.fillna(0):
        if value < 0:
            current += 1
        else:
            current = 0
        durations.append(current)
    return pd.Series(durations, index=drawdowns.index)


def top_drawdowns(returns: pd.Series, top_n: int = 10) -> pd.DataFrame:
    dd = drawdown_series(returns).dropna()
    if dd.empty:
        return pd.DataFrame(columns=["start", "trough", "recovery", "depth", "days_to_trough", "total_days", "recovered"])

    episodes: list[dict[str, object]] = []
    in_drawdown = False
    start = trough = recovery = None
    trough_value = 0.0
    for dt, value in dd.items():
        if value < 0 and not in_drawdown:
            in_drawdown = True
            start = dt
            trough = dt
            trough_value = float(value)
        elif value < 0 and in_drawdown and value < trough_value:
            trough = dt
            trough_value = float(value)
        elif value >= 0 and in_drawdown:
            recovery = dt
            episodes.append(
                {
                    "start": start,
                    "trough": trough,
                    "recovery": recovery,
                    "depth": trough_value,
                    "days_to_trough": (trough - start).days if start is not None and trough is not None else np.nan,
                    "total_days": (recovery - start).days if start is not None else np.nan,
                    "recovered": True,
                }
            )
            in_drawdown = False
    if in_drawdown:
        episodes.append(
            {
                "start": start,
                "trough": trough,
                "recovery": pd.NaT,
                "depth": trough_value,
                "days_to_trough": (trough - start).days if start is not None and trough is not None else np.nan,
                "total_days": (dd.index[-1] - start).days if start is not None else np.nan,
                "recovered": False,
            }
        )

    result = pd.DataFrame(episodes)
    if result.empty:
        return result
    return result.sort_values("depth").head(top_n).reset_index(drop=True)


def drawdown_snapshot(price: pd.Series) -> dict[str, float]:
    returns = price.dropna().pct_change(fill_method=None)
    dd = drawdown_series(returns)
    durations = drawdown_duration(dd)
    current = float(dd.dropna().iloc[-1]) if not dd.dropna().empty else np.nan
    max_dd = float(dd.min()) if not dd.empty else np.nan
    current_duration = float(durations.iloc[-1]) if not durations.empty else np.nan
    max_duration = float(durations.max()) if not durations.empty else np.nan
    episodes = top_drawdowns(returns, top_n=10)
    recovered = episodes[episodes["recovered"] == True] if not episodes.empty else pd.DataFrame()  # noqa: E712
    avg_recovery = float(recovered["total_days"].mean()) if not recovered.empty else np.nan
    return {
        "Current Drawdown": current,
        "Maximum Drawdown": max_dd,
        "Current Drawdown Duration": current_duration,
        "Max Drawdown Duration": max_duration,
        "Average Recovery Days": avg_recovery,
    }
