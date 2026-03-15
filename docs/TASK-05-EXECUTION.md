# TASK-05: Execution Layer

## Goal
Build the paper trading engine, pre-trade risk controls, signal-to-order translation, portfolio optimizer, and crypto data pipeline. This layer turns signals into simulated trades.

## Depends On
Task 01 (config system)
Independently testable — accepts Signal objects as input, does not require Qlib or agents.

## What to Build

### 1. Paper Trader (`src/execution/paper_trader.py`)

Simulated order execution engine.

Slippage: BUY fills at price × (1 + slippage_bps/10000), SELL at price × (1 - slippage_bps/10000).
Commission: per share (default $0.005), minimum $1.00.

Must track: positions with average cost basis (updates correctly on multiple buys at different prices), cash balance, realized P&L (on sells), unrealized P&L (mark-to-market), portfolio value history (daily snapshots), complete trade log.

Validation: reject buy if insufficient cash, reject sell if insufficient shares.

Performance metrics: total return, annualized return, Sharpe ratio (rf=4%), Sortino, max drawdown, max drawdown duration, Calmar, volatility, win rate, avg win/loss, profit factor.

Export: trade log to CSV, portfolio history to CSV.

### 2. Pre-Trade Risk Controls (`src/execution/risk_controls.py`)

Pure validation — no side effects. Takes an order + portfolio state + market data → returns pass/fail with reasons.

Checks (all of these):
- Position size: after order, position value / portfolio value ≤ max (10% equity, 5% crypto)
- Available cash: order cost ≤ cash × 0.99 (1% buffer)
- Sufficient shares: for sells
- Liquidity: ticker avg daily dollar volume ≥ $1M
- Market impact: order shares ≤ 1% of avg daily volume
- Drawdown circuit breaker: if portfolio drawdown > -15%, block ALL new buys
- Daily turnover: sum of today's order values / portfolio value ≤ 25%

### 3. Signal Bridge (`src/execution/lean_bridge.py`)

Signal model (Pydantic): ticker, direction (BUY/SELL/HOLD), strength (-1 to +1), target_weight (0 to max), confidence, source, timestamp, metadata dict.

Order model (Pydantic): ticker, side, quantity, order_type (MARKET/LIMIT/STOP), limit_price, stop_price.

Convert signals to orders: compute target shares from target_weight × portfolio_value / price, diff against current position, generate buy/sell orders. Must be idempotent.

Generate rebalance orders from target weights dict.

Write signals to JSON for LEAN consumption. Include a LEAN algorithm template string.

### 4. Portfolio Optimizer (`src/execution/portfolio_optimizer.py`)

Accept expected returns (from Qlib model) + covariance matrix (from historical data) + constraints → return target weights dict.

Methods: risk parity, mean-variance with Ledoit-Wolf shrinkage, Black-Litterman, Hierarchical Risk Parity (Lopez de Prado), maximum diversification.

Hard constraints: max 10% single equity, max 5% single crypto, total crypto 5-25%, total equity 50-90%, min 5% cash, long-only.

Output includes risk contribution breakdown by asset class.

### 5. Crypto Data Pipeline (`src/data/crypto_data.py`)

Use CCXT for Binance and Kraken access.

Fetch: OHLCV with automatic pagination (handle 1000-candle limits), funding rates (perpetual futures), open interest, orderbook snapshots.

Auto-generate investable universe: top 50 by market cap, > $10M daily volume, exclude stablecoins and wrapped tokens.

Local parquet cache with freshness checks. Rate limiting per exchange API rules.

### 6. Crypto Factors (`src/core/crypto_factors.py`)

Factors: funding rate mean-reversion (z-score of 8h rate vs 30d mean, contrarian), OI-price divergence (rolling correlation), BTC dominance signal, beta-adjusted momentum (residual vs BTC), realized volatility cone (current vol vs 1Y percentile), NVT ratio proxy (mcap / exchange volume z-scored), futures-to-spot volume ratio.

### 7. Unified Data Layer (`src/data/unified_data.py`)

Single `get_ohlcv(ticker)` that auto-detects: "/" in ticker → crypto pipeline, else → equity (Qlib/yfinance). Normalize timestamps UTC, standard OHLCV columns. Cross-asset correlation matrix. Asset metadata (asset_class, sector, market_cap, avg_volume, spread_bps).

## Inputs
- Signal objects (from agent pipeline or Qlib predictions)
- Market prices dict
- LeanConfig + RiskConfig from config system

## Outputs
- Fill objects from paper trader
- RiskCheckResult (pass/fail per check)
- Order objects from signal bridge
- Target weights dict from portfolio optimizer
- Crypto OHLCV DataFrames from crypto pipeline

## Done Criteria
- [ ] Paper trader: buy 100 shares AAPL at $200, cash decreases by ~$20K + costs
- [ ] Paper trader: buy at $100 then buy at $120 → avg cost ≈ $110
- [ ] Paper trader: sell more shares than owned → rejected (returns None)
- [ ] Paper trader: cash + positions = total_value at all times (conservation)
- [ ] Paper trader: performance metrics compute without error after 10+ trades
- [ ] Risk controls: $50K order in $100K portfolio → blocked (exceeds 10%)
- [ ] Risk controls: $3K order in $100K portfolio → passes
- [ ] Risk controls: drawdown > 15% → blocks all buys
- [ ] Signal bridge: signals convert to correct buy/sell orders
- [ ] Signal bridge: running twice with same signals produces zero new orders (idempotent)
- [ ] Portfolio optimizer: risk parity returns weights that sum to ≤ 1.0, all within constraints
- [ ] Crypto pipeline: fetches BTC/USDT daily OHLCV from Binance
- [ ] Unified data: get_ohlcv("AAPL") and get_ohlcv("BTC/USDT") both return valid DataFrames
- [ ] `pytest tests/test_execution.py` passes
