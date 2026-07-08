from __future__ import annotations

import pandas as pd

from src.charts import live_technical_chart
from src.live_market import add_technical_indicators


def test_live_chart_uses_compressed_category_axis_without_gap_rows() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            "2026-01-05 09:00",
            "2026-01-05 10:00",
        ]
    )
    price = pd.Series([100.0, 101.0, 104.0, 105.0], index=index)
    frame = pd.DataFrame(
        {
            "Open": price,
            "High": price + 1,
            "Low": price - 1,
            "Close": price,
            "Volume": 1_000,
        },
        index=index,
    )
    indicators = add_technical_indicators(frame, interval="60m")

    fig = live_technical_chart(indicators, "Compressed axis", chart_type="Line", overlays=["Moving averages"])

    assert fig.layout.xaxis.type == "category"
    assert len(fig.data[0].x) == len(indicators)
    assert all(isinstance(value, str) for value in fig.data[0].x)
