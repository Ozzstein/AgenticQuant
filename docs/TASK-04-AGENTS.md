# TASK-04: LangGraph Multi-Agent Intelligence Layer

## Goal
Build the LLM-powered analytical layer as a LangGraph state graph. Five specialist agents analyze tickers, debate, and produce structured investment decisions.

## Depends On
Task 01 (config system)
No dependency on Tasks 02-03 — this layer is independently testable with manual ticker input.

## Stack
- **Claude (Anthropic)** as primary reasoning model
- **LangGraph** for multi-agent orchestration and workflow state
- **LangChain** for external data tools
- **Pydantic** for structured outputs at every stage
- **asyncio** for parallel agent execution

## What to Build

### 1. Data Tools (`src/data/market_data.py`, `src/data/news_data.py`, `src/data/fundamental_data.py`)

LangChain @tool functions for live data access. Each tool:
- Returns concise text < 2500 chars
- Catches all exceptions, returns "TOOL_ERROR: {message}" on failure
- Has retry logic (3 retries, exponential backoff)

**Market data tools**: get_stock_price (current price + key stats + valuation ratios), get_price_history (last 30 periods OHLCV as text), get_technical_indicators (RSI, MACD, Bollinger, SMAs, ATR, stochastic, OBV, relative strength — all as interpreted text not raw numbers), get_support_resistance (swing highs/lows, volume levels, MAs as dynamic S/R), get_sector_comparison (vs sector ETF).

**News tools**: get_company_news (recent headlines from FinnHub, fallback to RSS), get_analyst_recommendations (consensus, price targets, recent changes), get_earnings_info (next date, last 4 quarters surprise, guidance).

**Fundamental tools**: get_financial_statements (income/balance/cashflow key metrics as text tables), get_financial_ratios (profitability, leverage, valuation, growth), get_insider_activity (recent transactions, net buying/selling), get_institutional_holders (top 10, ownership %).

**Macro tools**: get_macro_context (S&P level+trend, VIX, 10Y yield, DXY, Fed rate, breadth), get_fear_greed_index (crypto sentiment from alternative.me).

**Crypto tools**: get_crypto_price (via yfinance or CoinGecko), get_crypto_fundamentals (market cap, supply, TVL if available).

All tools must auto-detect crypto tickers (contains "/") vs equity tickers.

### 2. Macro Regime Detector (`src/agents/macro_regime.py`)

Classify current environment using: VIX level+trend, yield curve slope (10Y-2Y), credit spreads (if available, else skip), DXY trend, S&P 500 breadth.

Output: one of risk_on / neutral / risk_off / crisis with a confidence score.

This runs as the first node in the LangGraph pipeline. Its output is injected into every agent's context.

### 3. Five Specialist Agents (`src/agents/`)

Each agent is a function (or class) that:
- Takes the current LangGraph state (ticker, macro regime, any Qlib context)
- Has a system prompt defining its persona and analysis methodology
- Has access to its specific tools
- Returns a structured report (Pydantic model) with: recommendation, key_findings, confidence, risks_identified

Agent personas as described in CLAUDE.md:
- `fundamental_analyst.py` — CFA persona, financial statements, valuation, growth. Crypto: tokenomics, TVL.
- `sentiment_analyst.py` — Alt-data persona, news, ratings, social, whale activity, Fear & Greed.
- `technical_analyst.py` — CMT persona, indicators, patterns, exact price levels.
- `risk_manager.py` — CRO persona, **VETO POWER**, position sizing, correlation, tail risk, event risk. Crypto: smart contract + exchange risk.
- `portfolio_strategist.py` — PM persona, synthesizes all, resolves conflicts, produces final AnalysisResult.

### 4. Earnings Event Handler (`src/agents/earnings_handler.py`)

Maintain earnings calendar (yfinance or FinnHub). If earnings within 5 trading days: inject this into agent context, risk manager increases risk weighting, portfolio strategist reduces position size. Support triggering post-earnings re-analysis.

### 5. LangGraph Workflow (`src/agents/orchestrator.py`)

Build a LangGraph StateGraph with this node flow:

```
[market_context] → [macro_regime] → [parallel_analysts] → [risk_manager] → [debate_loop] → [strategist] → [output]
```

- **market_context**: gather ticker data, check earnings calendar
- **macro_regime**: classify regime, add to state
- **parallel_analysts**: run fundamental, sentiment, technical agents concurrently (use asyncio)
- **risk_manager**: review all analyst reports, flag risks, can set veto=true in state
- **debate_loop**: conditional edge — agents review each other's reports, may revise. Loops for config.max_debate_rounds times.
- **strategist**: synthesize into final AnalysisResult (Pydantic)
- **output**: return AnalysisResult

State must carry: ticker, macro_regime, qlib_context (optional), agent_reports dict, debate_round counter, risk_veto flag.

### 6. Model Configuration

Support providers: Anthropic Claude (primary), OpenAI GPT, Ollama (local).
- deep_think_model: for agent reasoning
- quick_think_model: for data extraction/summarization
Loaded from AgentConfig.

### 7. Optional: FinGPT Local Sentiment (`src/agents/fingpt_sentiment.py`)

Interface to load FinGPT from HuggingFace for local sentiment scoring. Provide a `predict_sentiment(text) → float` function. Sentiment agent can use this instead of LLM API calls when configured.

## Inputs
- Ticker string (e.g., "NVDA" or "BTC/USDT")
- Optional qlib_context dict: {qlib_alpha_score, qlib_rank, factor_signals}
- AgentConfig from config system

## Outputs
- AnalysisResult (Pydantic) as defined in CLAUDE.md
- Individual agent reports accessible via result.agent_reports

## Done Criteria
- [x] `get_stock_price("AAPL")` returns valid data as concise text
- [x] `get_company_news("NVDA")` returns recent headlines
- [x] `get_technical_indicators("MSFT")` returns interpreted indicator text
- [x] Macro regime detector returns one of risk_on/neutral/risk_off/crisis
- [x] Each agent runs independently and returns a structured report
- [x] LangGraph workflow executes the full node sequence for a single ticker
- [x] Three analyst agents run in parallel (verify with timing — should be ~1 LLM call time, not 3x)
- [x] Debate loop executes 2 rounds and agent reports show revisions
- [x] Risk manager veto flag blocks a decision when risk limits breached
- [x] Final output is a valid AnalysisResult with all fields populated
- [ ] Crypto ticker "BTC/USDT" routes to crypto-appropriate tools and analysis
- [ ] `pytest tests/test_agents.py` passes (use mocked LLM responses for unit tests)
