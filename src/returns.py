from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import annualization_factor, clean_price_series


def simple_returns(price: pd.Series) -> pd.Series:
    return clean_price_series(price).pct_change(fill_method=None)


def log_returns(price: pd.Series) -> pd.Series:
    clean = clean_price_series(price)
    return np.log(clean / clean.shift(1))


def cumulative_return(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod() - 1


def horizon_return(price: pd.Series, periods: int) -> float:
    clean = clean_price_series(price)
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[-periods - 1] - 1)


def ytd_return(price: pd.Series) -> float:
    clean = clean_price_series(price)
    if clean.empty:
        return np.nan
    current_year = clean.index[-1].year
    year_prices = clean[clean.index.year == current_year]
    if len(year_prices) < 2:
        return np.nan
    return float(year_prices.iloc[-1] / year_prices.iloc[0] - 1)


def annualized_return(price: pd.Series, frequency: str = "daily", periods: int | None = None) -> float:
    clean = clean_price_series(price)
    if periods is not None:
        clean = clean.tail(periods + 1)
    if len(clean) < 2:
        return np.nan
    factor = annualization_factor(frequency)
    observations = len(clean) - 1
    total = clean.iloc[-1] / clean.iloc[0] - 1
    if observations < factor / 2:
        return np.nan
    return float((1 + total) ** (factor / observations) - 1)


def monthly_returns(price: pd.Series) -> pd.Series:
    clean = clean_price_series(price)
    monthly = clean.resample("ME").last()
    return monthly.pct_change(fill_method=None).dropna()


def performance_snapshot(price: pd.Series, frequency: str = "daily") -> dict[str, float]:
    horizons = {
        "Daily Return": 1,
        "Weekly Return": 5,
        "1M Return": 21,
        "3M Return": 63,
        "6M Return": 126,
        "12M Return": 252,
    }
    result = {label: horizon_return(price, periods) for label, periods in horizons.items()}
    result["Year-to-Date Return"] = ytd_return(price)
    result["3Y Annualized Return"] = annualized_return(price, frequency=frequency, periods=252 * 3)
    result["5Y Annualized Return"] = annualized_return(price, frequency=frequency, periods=252 * 5)
    result["Full History Annualized Return"] = annualized_return(price, frequency=frequency)
    return result


def positive_month_rate(price: pd.Series) -> float:
    returns = monthly_returns(price)
    if returns.empty:
        return np.nan
    return float((returns > 0).mean())


def best_worst_periods(price: pd.Series) -> dict[str, float]:
    mret = monthly_returns(price)
    qret = clean_price_series(price).resample("QE").last().pct_change(fill_method=None).dropna()
    return {
        "Best Month": float(mret.max()) if not mret.empty else np.nan,
        "Worst Month": float(mret.min()) if not mret.empty else np.nan,
        "Best Quarter": float(qret.max()) if not qret.empty else np.nan,
        "Worst Quarter": float(qret.min()) if not qret.empty else np.nan,
    }
