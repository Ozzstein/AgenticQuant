# Improvement Plan

**Last updated:** 2026-03-22

---

## Tier 1: Foundation Hardening

> Fix critical bugs — get the project running.

- [x] **1A. Real Paper Trading Persistence + Multi-Day Loop**
  - Serialize paper portfolio to JSON after every trade
  - Load on startup — enables multi-day paper trading without state loss
  - `outputs/paper_portfolio_history.jsonl` for daily NAV snapshots
  - Wire P&L attribution in `run_pipeline.py` to use actual NAV delta

- [x] **1B. CLI Integration Layer Fix**
  - Fix 4 critical CLI bugs (analyze, backtest, trade status, trade reset)
  - Add input validation and helpful error messages when API keys are missing

- [x] **1C. Docker Build Fix**
  - Fix Dockerfile multi-stage build, CMD, and env_file path
  - Add streamlit to the appropriate dependency group

---

## Tier 2: Intelligence Upgrades

> Real alpha discovery — biggest value improvement.

- [x] **2A. Real RD-Agent Integration**
  - Replace simulated `rd_agent_runner.py` with LLM-driven factor hypothesis generation
  - Claude Haiku for structured proposal generation (factors, model configs, strategies, regime changes)
  - Real two-stage backtest validation for IC/ICIR measurement
  - Claude Opus for meta-analyst research memos that feed back into proposals
  - Design spec: `docs/superpowers/specs/2026-03-22-rd-agent-llm-design.md`

- [x] **2B. Agentic Factor Backtesting**
  - Walk-forward backtest gate after IC gate in `mine_factors()`
  - `FactorBacktester` wraps `WalkForwardBacktester` + `BacktestValidator`
  - Plan: `docs/superpowers/plans/2026-03-22-agentic-factor-backtesting-plan.md`

- [x] **2C. Multi-Model Ensemble**
  - `EnsembleModel` with IC-weighted prediction blending (LightGBM + XGBoost + Linear)
  - `create_model()` factory, `model_factory` param in `WalkForwardRunner`
  - Merged: PR #9

- [x] **2D. Agent Memory / RAG**
  - Persistent memory layer for LangGraph agents across analysis sessions
  - RAG over past analyses for improved reasoning
  - Merged: PR #10

---

## Tier 3: Execution Sophistication

> Live trading readiness.

- [x] **3A. Alpaca Live Trading Bridge**
  - `AlpacaTrader` class alongside `PaperTrader` with same interface
  - Gate behind `AIQUANT_ALPACA_LIVE=true` config flag

- [x] **3B. CCXT Execution**
  - Live crypto order execution via CCXT
  - Unified interface with paper trader

- [ ] **3C. Advanced Orders**
  - Bracket orders (take-profit + stop-loss)
  - Trailing stops, time-in-force options

- [x] **3D. Intraday Scheduling**
  - Sub-daily pipeline runs for crypto and momentum strategies
  - Configurable scheduling per strategy

---

## Tier 4: Data & Analysis Expansion

> Widen the moat.

- [ ] **4A. Earnings Call Transcript NLP**
  - Transcription API for earnings call audio
  - Specialized LLM agent for transcript sentiment/insight extraction

- [ ] **4B. Alternative Data Sources**
  - Satellite imagery, social media sentiment, web traffic
  - Pluggable data source architecture

---

## Tier 5: Platform & UX

> Scale and accessibility.

- [ ] **5A. Web UI with Real-Time Updates**
  - FastAPI + React frontend replacing Streamlit
  - WebSocket-based real-time portfolio updates
  - Interactive agent debate viewer

- [ ] **5B. Multi-User Support**
  - User authentication and role-based access
  - Per-user portfolios and risk limits

- [ ] **5C. Strategy Marketplace**
  - Publish validated strategies to shared library
  - Leaderboard ranked by out-of-sample Sharpe
  - Import/export strategy definitions as YAML

- [ ] **5D. Mobile Alerts App**
  - Push notifications for trades, risk alerts, daily P&L
  - Quick approve/reject for high-conviction trades

---

## Tier 6: Cutting-Edge Research

> Long-term differentiation.

- [ ] **6A. Reinforcement Learning Portfolio Manager**
  - RL agent (PPO/SAC) replacing rule-based portfolio optimizer
  - Train on historical walk-forward data
  - Action space: position sizes for top-K assets

- [ ] **6B. Causal Factor Discovery**
  - Causal inference (DoWhy / CausalNex) replacing correlation-based IC
  - Identify factors with genuine predictive power vs. spurious correlation

- [ ] **6C. Multi-Agent Debate with Specialized LLMs**
  - Fine-tune small models (Llama-3 8B) per agent role
  - CFA agent on 10-K filings, CMT agent on chart patterns
  - Claude only for strategist synthesis node

- [ ] **6D. Synthetic Data Augmentation**
  - GANs or diffusion models for synthetic market scenarios
  - Stress-test strategies against tail-risk scenarios

- [ ] **6E. Cross-Asset Signal Propagation**
  - Lead-lag relationships across equities, crypto, commodities, FX
  - Feed cross-asset signals to macro regime detector

---

## Priority Summary

```
DONE  (Tier 1): Foundation hardening — critical bugs fixed
DONE  (Tier 2): Intelligence upgrades — all complete
NOW   (Tier 3): Execution — live trading readiness
LATER (Tier 4-6): Expansion — competitive moat
```
