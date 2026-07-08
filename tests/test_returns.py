from __future__ import annotations

import numpy as np
import pandas as pd

from src.returns import annualized_return, horizon_return, log_returns, simple_returns, ytd_return


def test_simple_and_log_returns() -> None:
    price = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-01", periods=3))
    simple = simple_returns(price)
    logs = log_returns(price)
    assert np.isclose(simple.iloc[-1], 0.10)
    assert np.isclose(logs.iloc[-1], np.log(1.10))


def test_horizon_and_ytd_returns_use_available_history() -> None:
    price = pd.Series([100.0, 105.0, 110.0], index=pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"]))
    assert np.isclose(horizon_return(price, 1), 110.0 / 105.0 - 1)
    assert np.isclose(ytd_return(price), 110.0 / 105.0 - 1)
    assert np.isnan(annualized_return(price, periods=252))
