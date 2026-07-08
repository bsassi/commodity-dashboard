from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_scaled_weights(
    summary: pd.DataFrame,
    target_volatility: float = 0.10,
    max_weight_per_asset: float = 0.15,
    max_weight_per_sector: float = 0.35,
    gross_exposure_cap: float = 1.0,
    long_only: bool = False,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary.iterrows():
        vol = row.get("60D Volatility", np.nan)
        signal = row.get("Final Composite Score", row.get("Composite Score", 0.0))
        if not np.isfinite(vol) or vol <= 0:
            raw = 0.0
        else:
            raw = target_volatility / vol
        normalized_signal = np.clip(signal / 100, -1, 1)
        if long_only:
            normalized_signal = max(normalized_signal, 0.0)
        weight = raw * normalized_signal
        weight = float(np.clip(weight, -max_weight_per_asset, max_weight_per_asset))
        rows.append(
            {
                "Ticker": row["Ticker"],
                "Name": row.get("Name", row["Ticker"]),
                "Sector": row.get("Sector", "Unknown"),
                "Signal": signal,
                "Estimated Volatility": vol,
                "Raw Vol Weight": raw,
                "Signal Weight": weight,
            }
        )
    weights = pd.DataFrame(rows)
    if weights.empty:
        return weights

    for sector, group in weights.groupby("Sector"):
        gross = group["Signal Weight"].abs().sum()
        if gross > max_weight_per_sector > 0:
            scale = max_weight_per_sector / gross
            weights.loc[group.index, "Signal Weight"] *= scale

    gross = weights["Signal Weight"].abs().sum()
    if gross > gross_exposure_cap > 0:
        weights["Signal Weight"] *= gross_exposure_cap / gross

    weights["Gross Contribution"] = weights["Signal Weight"].abs()
    weights["Net Contribution"] = weights["Signal Weight"]
    return weights.sort_values("Signal Weight", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def estimate_transaction_costs(weights: pd.Series, previous_weights: pd.Series | None = None, cost_bps: float = 5.0) -> float:
    aligned_prev = previous_weights.reindex(weights.index).fillna(0.0) if previous_weights is not None else 0.0
    turnover = (weights - aligned_prev).abs().sum()
    return float(turnover * cost_bps / 10000)
