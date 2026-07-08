# Commodity Systematic Macro Dashboard

Institutional Streamlit dashboard for commodity futures analysis using Yahoo Finance data through `yfinance`.

The application is designed as a decision-support tool for systematic macro, managed futures and trend-following research. It does not present signals as forecasts or investment recommendations. Every score is deterministic, bounded, decomposable and shown with data-quality warnings.

## Architecture

```text
commodity_dashboard/
  app.py
  requirements.txt
  README.md
  config/
    assets.yaml
    settings.yaml
  src/
    data_loader.py
    data_quality.py
    returns.py
    trend.py
    momentum.py
    volatility.py
    drawdown.py
    seasonality.py
    correlations.py
    scoring.py
    macro.py
    portfolio.py
    backtest.py
    live_market.py
    charts.py
    reporting.py
    utils.py
  pages/
    1_Market_Overview.py
    2_Asset_Deep_Dive.py
    3_Trend_Momentum.py
    4_Risk_Analytics.py
    5_Seasonality.py
    6_Correlation_Regimes.py
    7_Signal_Backtest.py
    8_Data_Quality.py
    9_Live_Commodity_Stream.py
    10_Trade_Idea_Lab.py
  tests/
    test_returns.py
    test_trend.py
    test_volatility.py
    test_seasonality.py
    test_scoring.py
    test_backtest.py
    test_live_market.py
    test_macro.py
    test_charts.py
  data/
    cache/
```

## Installation On Windows

From `c:\Users\ilyes\Documents\My Brain\Commodities\commodity_dashboard`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you use the already configured system Python, dependencies can also be installed with:

```powershell
python -m pip install -r requirements.txt
```

## Data Source

The universe is configured in [config/assets.yaml](config/assets.yaml). The default universe includes energy, precious metals, industrial metals, agriculture and livestock Yahoo Finance futures tickers.

Important limitation:

> Yahoo Finance futures tickers can represent provider-built continuous series. Returns may be affected by contract changes, adjustments and roll effects. Do not interpret a contract change as certain economic performance.

Carry and roll yield are not calculated because Yahoo Finance does not provide the futures curve required for that calculation in this setup:

```text
Carry indisponible avec la source de donnees actuelle.
```

## Live Commodity Stream

The `Live Commodity Stream` page provides an intraday technical cockpit for the configured commodity universe:

- selectable futures contract from `config/assets.yaml`;
- preset timeframes from `1D / 1m` to `5Y / 1wk`;
- advanced Yahoo period and interval controls;
- manual refresh and optional auto-refresh;
- candlestick or line chart with moving averages, Bollinger Bands, Donchian channel, VWAP, volume, RSI and MACD;
- deterministic technical score with trend, momentum, breakout, confidence, risk state, ATR levels and a compact watchlist.

The live module still uses Yahoo Finance through `yfinance`. Intraday data can be delayed, revised, unavailable outside provider windows, or constrained by Yahoo interval limits. It should be used as decision support, not as exchange-certified tick data.

## Configuration

Edit [config/settings.yaml](config/settings.yaml) to change:

- default start date and frequency;
- DXY macro ticker, beta window and correlation window;
- moving-average, Donchian, regression and momentum windows;
- composite-score weights;
- risk penalties;
- volatility-targeting and position-sizing defaults;
- backtest costs, rebalancing and warm-up assumptions.

To add an asset, add one entry to `config/assets.yaml`:

```yaml
- ticker: "NEW=F"
  name: "New Future"
  sector: "Energy"
  sub_sector: "Example"
  currency: "USD"
  asset_class: "Futures"
  indicative_unit: "contract unit"
```

The same schema can later support ETFs, indices, rates, FX, crypto or other futures if Yahoo Finance provides usable OHLC data.

## Methodology

### Data Quality

The data loader:

- retries Yahoo Finance downloads;
- handles `yfinance` MultiIndex output;
- validates OHLC columns;
- removes duplicate timestamps;
- sorts observations chronologically;
- detects missing close values, stale data, non-positive prices and abnormal return jumps;
- writes optional CSV cache files under `data/cache/`.

Each asset receives `OK`, `Warning` or `Error` status.

### Returns

The dashboard computes simple and log returns, daily, weekly, 1M, 3M, 6M, 12M, YTD, 3Y annualized, 5Y annualized and full-history annualized returns. Annualized values are suppressed when the available history is too short to make the result meaningful.

### Trend

Trend strength combines transparent components:

- price versus SMA 20, 50, 100 and 200;
- SMA 50 versus SMA 200;
- Donchian breakouts over 20, 55, 100 and 252 days;
- time-series momentum over 1M, 3M, 6M, 9M and 12M;
- log-price regression slope, t-statistic and R-squared over multiple windows;
- horizon coherence.

The signal backtest page also includes a `trend_ensemble` strategy designed for research diagnostics. It blends 1M, 3M, 6M and 12M volatility-adjusted momentum into a continuous signal, adds a moving-average posture filter and applies a configurable deadband to reduce low-conviction whipsaw trades.

The score is bounded from -100 to +100:

- +60 to +100: Strong Bullish Trend
- +20 to +60: Bullish Trend
- -20 to +20: Neutral / Mixed
- -60 to -20: Bearish Trend
- -100 to -60: Strong Bearish Trend

### Cross-Sectional Momentum

Assets are ranked by momentum, volatility-adjusted momentum and momentum acceleration. The rotation matrix classifies assets as:

- Leading
- Weakening
- Lagging
- Improving

### Volatility And Risk

The risk module computes realized volatility, EWMA volatility, downside deviation, volatility percentiles, short-term versus long-term volatility ratio, ATR, historical VaR, expected shortfall, skewness, kurtosis, extreme-loss frequency and worst daily return.

Risk labels are mostly based on each asset's own volatility percentile:

- Low Risk
- Normal Risk
- Elevated Risk
- Extreme Risk

Historical VaR and Expected Shortfall are empirical estimates from available returns. They are not guaranteed loss limits.

### Drawdown

The drawdown module computes wealth index, current drawdown, maximum drawdown, drawdown duration, average recovery time and the top historical drawdown episodes.

### Seasonality

Seasonality is calculated from monthly returns:

- mean and median return by calendar month;
- hit rate;
- volatility;
- t-statistic and p-value;
- bootstrap confidence interval;
- sub-period stability;
- mean versus median coherence.

Evidence labels are:

- Strong Evidence
- Moderate Evidence
- Weak Evidence
- Insufficient Data

The app explicitly warns about multiple-testing risk. A monthly pattern is not treated as robust if sample size, stability, significance or mean/median coherence is weak.

### Composite Score

The composite model separates:

1. Direction Score
2. Trend Strength Score
3. Risk Score
4. Signal Confidence Score
5. Final Composite Score

Default weights:

- Trend Score: 35%
- Time-Series Momentum: 20%
- Cross-Sectional Momentum: 15%
- Breakout Score: 10%
- Seasonality Score: 10%
- Risk Adjustment: 10%

Penalties reduce confidence and/or final score when volatility is extreme, drawdown is severe, data is insufficient, horizons conflict, trend fit is weak or seasonality is fragile.

The signal explanation is generated by deterministic rules. No AI API is used.

### Macro And Trade Ideas

The dashboard now downloads the US Dollar Index proxy `DX-Y.NYB` from Yahoo Finance and adds a DXY overlay to each commodity:

- rolling DXY beta;
- rolling DXY correlation;
- 1W, 1M and 3M DXY returns;
- DXY trend regime;
- DXY pressure score for each commodity;
- macro-adjusted trade idea score and conviction.

The `Trade Idea Lab` page ranks long and short candidates by combining the existing composite score, DXY pressure, current-month seasonality, cross-sectional momentum and risk state. This is a deterministic screening layer for research and discussion; it is not a trade recommendation.

The current public-data implementation does not yet include CFTC COT positioning, futures curve carry or inventory releases. Those are documented extension points because they require separate data feeds or official datasets beyond simple Yahoo Finance OHLC history.

### Volatility Scaling

The position-sizing module estimates theoretical weights:

```text
raw_weight = target_volatility / estimated_volatility
signal_weight = raw_weight * normalized_signal
```

Then it applies:

- asset caps;
- sector caps;
- gross exposure normalization;
- optional long-only mode.

The weights are theoretical. They do not automatically include contract multipliers, margin, liquidity, real slippage, roll costs, regulatory limits or portfolio-specific constraints.

## Backtest Assumptions

The backtest engine supports:

- price above or below SMA;
- SMA crossover;
- Donchian breakout;
- time-series momentum;
- equal-weight portfolio aggregation.

Anti-bias rules:

- signal at close `t` is applied from `t+1`;
- rolling windows only use data available at the calculation date;
- no future values are used for return application;
- transaction costs are applied through turnover;
- daily, weekly and monthly rebalancing are supported;
- missing histories are not backfilled before listing availability.

Implemented robustness view:

- SMA parameter sensitivity grid;
- portfolio and single-asset diagnostics;
- annual, monthly, rolling and drawdown charts.

Simplification in this first version: walk-forward optimization and macro inflation regimes are represented as documented extension points rather than full parameter-selection engines.

## Tests

Run:

```powershell
python -m pytest -q
```

Current coverage checks:

- returns and annualization behavior;
- realized volatility;
- drawdown math;
- trend score and Donchian channel logic;
- seasonality score bounds;
- composite-score bounds;
- handling of missing ticker data;
- no-lookahead position shift in backtests.

## Launch

```powershell
cd "c:\Users\ilyes\Documents\My Brain\Commodities\commodity_dashboard"
streamlit run app.py
```

The latest local verification during development started Streamlit successfully on:

```text
http://localhost:8502
```

Port 8501 was already occupied on this machine at verification time, so Streamlit selected the next free port.
