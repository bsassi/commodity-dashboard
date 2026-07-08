from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .returns import monthly_returns
from .utils import clip_score


def bootstrap_mean_ci(values: pd.Series, iterations: int = 500, confidence: float = 0.95, seed: int = 7) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(iterations, len(clean)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def month_stability_score(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 8:
        return 0.0
    first = clean.iloc[: len(clean) // 2].mean()
    second = clean.iloc[len(clean) // 2 :].mean()
    median = clean.median()
    mean = clean.mean()
    sign_consistency = int(np.sign(first) == np.sign(second) == np.sign(mean))
    mean_median = int(np.sign(mean) == np.sign(median))
    dominance_penalty = 0.0
    if clean.std() > 0:
        dominance_penalty = min(abs(clean.max() - clean.mean()), abs(clean.min() - clean.mean())) / clean.std()
    return float(np.clip((sign_consistency + mean_median) / 2 - max(0, dominance_penalty - 3) * 0.1, 0, 1))


def calendar_month_stats(price: pd.Series, min_years: int = 5, bootstrap_iterations: int = 500) -> pd.DataFrame:
    returns = monthly_returns(price)
    if returns.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"return": returns})
    frame["month"] = frame.index.month
    rows: list[dict[str, float | int | str]] = []
    for month, group in frame.groupby("month"):
        values = group["return"].dropna()
        n = len(values)
        mean = values.mean() if n else np.nan
        median = values.median() if n else np.nan
        std = values.std() if n > 1 else np.nan
        hit_rate = (values > 0).mean() if n else np.nan
        if n > 2 and std and np.isfinite(std):
            t_stat, p_value = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        else:
            t_stat, p_value = np.nan, np.nan
        ci_low, ci_high = bootstrap_mean_ci(values, iterations=bootstrap_iterations)
        stability = month_stability_score(values)
        evidence = evidence_label(n, hit_rate, p_value, mean, median, stability, min_years=min_years)
        score = seasonality_score(n, hit_rate, p_value, mean, median, stability, min_years=min_years)
        rows.append(
            {
                "Month": int(month),
                "Month Name": pd.Timestamp(year=2000, month=int(month), day=1).strftime("%b"),
                "Mean": float(mean) if np.isfinite(mean) else np.nan,
                "Median": float(median) if np.isfinite(median) else np.nan,
                "Std": float(std) if np.isfinite(std) else np.nan,
                "Observations": int(n),
                "Hit Rate": float(hit_rate) if np.isfinite(hit_rate) else np.nan,
                "T-Statistic": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "P-Value": float(p_value) if np.isfinite(p_value) else np.nan,
                "Bootstrap CI Low": ci_low,
                "Bootstrap CI High": ci_high,
                "Stability": stability,
                "Evidence": evidence,
                "Seasonality Score": score,
            }
        )
    return pd.DataFrame(rows)


def evidence_label(
    n: int,
    hit_rate: float,
    p_value: float,
    mean: float,
    median: float,
    stability: float,
    min_years: int = 5,
) -> str:
    if n < min_years or not np.isfinite(mean):
        return "Insufficient Data"
    if np.sign(mean) != np.sign(median):
        return "Weak Evidence"
    if stability < 0.5:
        return "Weak Evidence"
    if np.isfinite(p_value) and p_value < 0.05 and abs(hit_rate - 0.5) >= 0.15:
        return "Strong Evidence"
    if np.isfinite(p_value) and p_value < 0.15 and abs(hit_rate - 0.5) >= 0.10:
        return "Moderate Evidence"
    return "Weak Evidence"


def seasonality_score(
    n: int,
    hit_rate: float,
    p_value: float,
    mean: float,
    median: float,
    stability: float,
    min_years: int = 5,
) -> float:
    if n < min_years or not np.isfinite(mean) or not np.isfinite(hit_rate):
        return 0.0
    direction = np.sign(mean)
    if direction == 0 or np.sign(mean) != np.sign(median):
        return 0.0
    significance = 1.0 if np.isfinite(p_value) and p_value < 0.05 else 0.5 if np.isfinite(p_value) and p_value < 0.15 else 0.2
    hit_strength = min(abs(hit_rate - 0.5) * 2, 1.0)
    sample_strength = min(n / max(min_years * 3, 1), 1.0)
    raw = direction * 100 * (0.35 * hit_strength + 0.25 * significance + 0.25 * stability + 0.15 * sample_strength)
    return clip_score(raw)


def current_month_seasonality(price: pd.Series) -> dict[str, float | str]:
    stats_frame = calendar_month_stats(price, bootstrap_iterations=250)
    if stats_frame.empty:
        return {"Seasonality Signal": "Insufficient Data", "Seasonality Score": 0.0, "Seasonality Evidence": "Insufficient Data"}
    month = pd.Timestamp(price.dropna().index[-1]).month
    row = stats_frame.loc[stats_frame["Month"] == month]
    if row.empty:
        return {"Seasonality Signal": "Insufficient Data", "Seasonality Score": 0.0, "Seasonality Evidence": "Insufficient Data"}
    score = float(row["Seasonality Score"].iloc[0])
    evidence = str(row["Evidence"].iloc[0])
    if evidence == "Insufficient Data":
        signal = "Insufficient Data"
    elif score >= 20:
        signal = "Favorable"
    elif score <= -20:
        signal = "Unfavorable"
    else:
        signal = "Neutral"
    return {"Seasonality Signal": signal, "Seasonality Score": score, "Seasonality Evidence": evidence}


def monthly_return_matrix(price: pd.Series) -> pd.DataFrame:
    returns = monthly_returns(price)
    if returns.empty:
        return pd.DataFrame()
    matrix = returns.to_frame("Return")
    matrix["Year"] = matrix.index.year
    matrix["Month"] = matrix.index.month
    return matrix.pivot(index="Year", columns="Month", values="Return")
