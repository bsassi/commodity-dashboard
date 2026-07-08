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


def _compressed_time_axis(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    index = pd.DatetimeIndex(frame.index)
    x_values = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in index]
    if len(index) == 0:
        return x_values, [], []

    tick_count = min(8, len(index))
    tick_positions = np.linspace(0, len(index) - 1, tick_count, dtype=int)
    tick_positions = sorted(set(int(position) for position in tick_positions))
    span_days = max((index[-1] - index[0]).total_seconds() / 86400, 0)

    tick_values = [x_values[position] for position in tick_positions]
    tick_text: list[str] = []
    for position in tick_positions:
        timestamp = index[position]
        if span_days <= 2:
            label = timestamp.strftime("%H:%M")
        elif timestamp.hour or timestamp.minute:
            label = f"{timestamp.strftime('%b')} {timestamp.day}<br>{timestamp.strftime('%H:%M')}"
        elif span_days <= 370:
            label = f"{timestamp.strftime('%b')} {timestamp.day}<br>{timestamp.year}"
        else:
            label = f"{timestamp.strftime('%b')}<br>{timestamp.year}"
        tick_text.append(label)
    return x_values, tick_values, tick_text


def live_technical_chart(
    frame: pd.DataFrame,
    title: str,
    chart_type: str = "Candles",
    overlays: list[str] | None = None,
    show_volume: bool = True,
) -> go.Figure:
    overlays = overlays or ["Moving averages"]
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, title=title, height=620, margin=dict(l=10, r=10, t=45, b=10))
        return fig

    x_values, tick_values, tick_text = _compressed_time_axis(frame)
    rows = 4 if show_volume else 3
    row_heights = [0.60, 0.10, 0.14, 0.16] if show_volume else [0.66, 0.16, 0.18]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
    )

    price_row = 1
    rsi_row = 3 if show_volume else 2
    macd_row = 4 if show_volume else 3

    if chart_type == "Candles" and {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                name="OHLC",
                showlegend=False,
                increasing_line_color="#1f7a4d",
                decreasing_line_color="#b11226",
                increasing_fillcolor="rgba(31, 122, 77, 0.42)",
                decreasing_fillcolor="rgba(177, 18, 38, 0.42)",
                hovertemplate=(
                    "%{x}<br>"
                    "Open %{open:,.2f}<br>"
                    "High %{high:,.2f}<br>"
                    "Low %{low:,.2f}<br>"
                    "Close %{close:,.2f}<extra></extra>"
                ),
            ),
            row=price_row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#1f4e79", width=1.5),
            ),
            row=price_row,
            col=1,
        )

    if "Moving averages" in overlays:
        for column, color, width in [("SMA Fast", "#235789", 1.1), ("SMA Slow", "#222222", 1.4), ("EMA Fast", "#d17a00", 1.0)]:
            if column in frame:
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=frame[column],
                        mode="lines",
                        name=column,
                        line=dict(color=color, width=width),
                    ),
                    row=price_row,
                    col=1,
                )

    if "Bollinger Bands" in overlays and {"BB Upper", "BB Lower"}.issubset(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["BB Upper"],
                mode="lines",
                name="Bollinger",
                line=dict(color="#7a8699", width=0.8, dash="dot"),
                opacity=0.75,
            ),
            row=price_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["BB Lower"],
                mode="lines",
                name="Bollinger range",
                line=dict(color="#7a8699", width=0.8, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(122, 134, 153, 0.08)",
                opacity=0.75,
                showlegend=False,
            ),
            row=price_row,
            col=1,
        )

    if "Donchian Channel" in overlays and {"Donchian High", "Donchian Low"}.issubset(frame.columns):
        for column, color, name in [("Donchian High", "#3b6f3f", "Donchian High"), ("Donchian Low", "#8c2f2f", "Donchian Low")]:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=frame[column],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=0.9, dash="dash"),
                    opacity=0.70,
                ),
                row=price_row,
                col=1,
            )

    if "VWAP" in overlays and "VWAP" in frame:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["VWAP"],
                mode="lines",
                name="VWAP",
                line=dict(color="#6b5b2a", width=1.1),
                opacity=0.82,
            ),
            row=price_row,
            col=1,
        )

    if show_volume and "Volume" in frame:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=frame["Volume"],
                name="Volume",
                marker_color="#c7d0dc",
                opacity=0.62,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if "RSI" in frame:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="#1f4e79", width=1.2),
                showlegend=False,
            ),
            row=rsi_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#8c2f2f", line_width=1, row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#3b6f3f", line_width=1, row=rsi_row, col=1)
        fig.update_yaxes(range=[10, 90], row=rsi_row, col=1)

    if {"MACD", "MACD Signal", "MACD Histogram"}.issubset(frame.columns):
        colors = np.where(frame["MACD Histogram"] >= 0, "#1f7a4d", "#b11226")
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=frame["MACD Histogram"],
                name="MACD Hist",
                marker_color=colors,
                opacity=0.42,
                showlegend=False,
            ),
            row=macd_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="#1f4e79", width=1.15),
                showlegend=False,
            ),
            row=macd_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame["MACD Signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color="#d17a00", width=1.1),
                showlegend=False,
            ),
            row=macd_row,
            col=1,
        )

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, x=0.0, xanchor="left", y=0.995, font=dict(size=15)),
        height=780 if show_volume else 720,
        margin=dict(l=8, r=8, t=54, b=76),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
            title_text="",
        ),
    )
    fig.update_yaxes(title_text="Price", row=price_row, col=1, fixedrange=False)
    if show_volume:
        fig.update_yaxes(title_text="Vol", row=2, col=1, fixedrange=True)
    fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, fixedrange=True)
    fig.update_yaxes(title_text="MACD", row=macd_row, col=1, fixedrange=False)
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x_values,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )
    fig.update_xaxes(showticklabels=False, row=price_row, col=1)
    if show_volume:
        fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=rsi_row, col=1)
    return fig
