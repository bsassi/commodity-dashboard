from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .drawdown import drawdown_series
from .seasonality import monthly_return_matrix


TEMPLATE = "plotly_white"
COLOR_SCALE = "RdBu"


def performance_bar(summary: pd.DataFrame, column: str = "3M Return") -> go.Figure:
    frame = summary.sort_values(column, ascending=True)
    fig = px.bar(frame, x=column, y="Name", color=column, color_continuous_scale=COLOR_SCALE, template=TEMPLATE)
    fig.update_layout(height=max(420, 22 * len(frame)), margin=dict(l=10, r=10, t=35, b=10), coloraxis_showscale=False)
    return fig


def performance_heatmap(summary: pd.DataFrame) -> go.Figure:
    columns = [
        "Daily Return",
        "Weekly Return",
        "1M Return",
        "3M Return",
        "6M Return",
        "12M Return",
        "Year-to-Date Return",
        "3Y Annualized Return",
        "5Y Annualized Return",
    ]
    available = [c for c in columns if c in summary.columns]
    heat = summary.set_index("Ticker")[available]
    fig = px.imshow(
        heat,
        color_continuous_scale=COLOR_SCALE,
        zmin=-max(abs(heat.min().min()), abs(heat.max().max())),
        zmax=max(abs(heat.min().min()), abs(heat.max().max())),
        aspect="auto",
        template=TEMPLATE,
    )
    fig.update_layout(height=max(420, 24 * len(heat)), margin=dict(l=10, r=10, t=35, b=10))
    return fig


def price_chart(frame: pd.DataFrame, title: str, mode: str = "level") -> go.Figure:
    close = frame["Close"].dropna()
    if mode == "base 100" and not close.empty:
        series = close / close.iloc[0] * 100
    elif mode == "log":
        series = close
    else:
        series = close
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Close", line=dict(color="#1f4e79")))
    for window, color in [(50, "#d62728"), (200, "#111111")]:
        ma = close.rolling(window, min_periods=window // 3).mean()
        fig.add_trace(go.Scatter(x=ma.index, y=ma, mode="lines", name=f"SMA {window}", line=dict(color=color, width=1)))
    fig.update_layout(template=TEMPLATE, title=title, height=480, margin=dict(l=10, r=10, t=45, b=10))
    if mode == "log":
        fig.update_yaxes(type="log")
    return fig


def drawdown_chart(frame: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    returns = frame["Close"].dropna().pct_change(fill_method=None)
    dd = drawdown_series(returns)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", mode="lines", name="Drawdown", line=dict(color="#b11226")))
    fig.update_layout(template=TEMPLATE, title=title, height=320, margin=dict(l=10, r=10, t=45, b=10))
    fig.update_yaxes(tickformat=".0%")
    return fig


def rotation_scatter(summary: pd.DataFrame) -> go.Figure:
    x = "Cross-Sectional Momentum Score"
    y = "Momentum Acceleration"
    if y not in summary.columns:
        summary = summary.copy()
        summary[y] = 0.0
    fig = px.scatter(
        summary,
        x=x,
        y=y,
        color="Sector",
        size=summary["60D Volatility"].abs().fillna(0.05),
        hover_name="Name",
        text="Ticker",
        template=TEMPLATE,
    )
    fig.add_hline(y=0, line_color="#999999", line_width=1)
    fig.add_vline(x=0, line_color="#999999", line_width=1)
    fig.update_traces(textposition="top center")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, color_continuous_scale=COLOR_SCALE, zmin=-1, zmax=1, aspect="auto", template=TEMPLATE)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def seasonality_heatmap(price: pd.Series) -> go.Figure:
    matrix = monthly_return_matrix(price)
    fig = px.imshow(matrix, color_continuous_scale=COLOR_SCALE, aspect="auto", template=TEMPLATE)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def equity_curve(backtest: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "equity" in backtest.columns:
        fig.add_trace(go.Scatter(x=backtest.index, y=backtest["equity"], mode="lines", name="Strategy"))
    fig.update_layout(template=TEMPLATE, height=420, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def live_technical_chart(
    frame: pd.DataFrame,
    title: str,
    chart_type: str = "Candles",
    overlays: list[str] | None = None,
    show_volume: bool = True,
) -> go.Figure:
    overlays = overlays or ["Moving averages", "Bollinger Bands", "VWAP"]
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, title=title, height=620, margin=dict(l=10, r=10, t=45, b=10))
        return fig

    rows = 4 if show_volume else 3
    subplot_titles = ["Price", "Volume", "RSI", "MACD"] if show_volume else ["Price", "RSI", "MACD"]
    row_heights = [0.55, 0.12, 0.16, 0.17] if show_volume else [0.62, 0.18, 0.20]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    price_row = 1
    rsi_row = 3 if show_volume else 2
    macd_row = 4 if show_volume else 3

    if chart_type == "Candles" and {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        fig.add_trace(
            go.Candlestick(
                x=frame.index,
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                name="OHLC",
                increasing_line_color="#1f7a4d",
                decreasing_line_color="#b11226",
            ),
            row=price_row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["Close"], mode="lines", name="Close", line=dict(color="#1f4e79")),
            row=price_row,
            col=1,
        )

    if "Moving averages" in overlays:
        for column, color in [("SMA Fast", "#1f4e79"), ("SMA Slow", "#111111"), ("EMA Fast", "#d17a00")]:
            if column in frame:
                fig.add_trace(
                    go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column, line=dict(color=color, width=1.4)),
                    row=price_row,
                    col=1,
                )

    if "Bollinger Bands" in overlays and {"BB Upper", "BB Lower"}.issubset(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["BB Upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="#7a8699", width=1, dash="dot"),
            ),
            row=price_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["BB Lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="#7a8699", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(122, 134, 153, 0.10)",
            ),
            row=price_row,
            col=1,
        )

    if "Donchian Channel" in overlays and {"Donchian High", "Donchian Low"}.issubset(frame.columns):
        for column, color in [("Donchian High", "#3b6f3f"), ("Donchian Low", "#8c2f2f")]:
            fig.add_trace(
                go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column, line=dict(color=color, width=1, dash="dash")),
                row=price_row,
                col=1,
            )

    if "VWAP" in overlays and "VWAP" in frame:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["VWAP"], mode="lines", name="VWAP", line=dict(color="#6b5b2a", width=1.3)),
            row=price_row,
            col=1,
        )

    if show_volume and "Volume" in frame:
        fig.add_trace(
            go.Bar(x=frame.index, y=frame["Volume"], name="Volume", marker_color="#b8c2d1", opacity=0.75),
            row=2,
            col=1,
        )

    if "RSI" in frame:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["RSI"], mode="lines", name="RSI", line=dict(color="#1f4e79", width=1.4)),
            row=rsi_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#8c2f2f", row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#3b6f3f", row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)

    if {"MACD", "MACD Signal", "MACD Histogram"}.issubset(frame.columns):
        colors = np.where(frame["MACD Histogram"] >= 0, "#1f7a4d", "#b11226")
        fig.add_trace(
            go.Bar(x=frame.index, y=frame["MACD Histogram"], name="MACD Hist", marker_color=colors, opacity=0.55),
            row=macd_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["MACD"], mode="lines", name="MACD", line=dict(color="#1f4e79", width=1.2)),
            row=macd_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["MACD Signal"], mode="lines", name="MACD Signal", line=dict(color="#d17a00", width=1.2)),
            row=macd_row,
            col=1,
        )

    fig.update_layout(
        template=TEMPLATE,
        title=title,
        height=820 if show_volume else 760,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig
