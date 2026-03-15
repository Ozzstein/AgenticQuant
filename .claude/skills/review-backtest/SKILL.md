---
name: review-backtest
description: QA review of a backtest result — checks for overfitting, statistical significance, and operational feasibility
user-invocable: false
---

When asked to review a backtest result, use the `src/analysis/backtest_reviewer.py` module to perform a rigorous QA check.

Given a backtest result (either as a dict/JSON or a reference to a recent run), call `review_backtest(result)` and evaluate:

1. **Statistical Rigor**
   - Is Sharpe > 0.5? (minimum acceptable)
   - Is Sharpe > 1.0? (good)
   - Is max drawdown < 20%? (acceptable risk)
   - Is annual return > 5%? (beats inflation)

2. **Overfitting Signals**
   - Sharpe > 3.0 on historical data → likely overfit
   - Very low trade count (< 50 trades) → insufficient sample
   - Extremely high win rate (> 80%) → suspicious
   - Max drawdown < 1% → likely overfitted or data issue

3. **Operational Feasibility**
   - Annual turnover reasonable for transaction costs?
   - Position sizes within risk limits?
   - Benchmark comparison available?

Return a structured verdict: **APPROVED** / **CAUTION** / **REJECTED** with specific reasoning for each check.

Format the output as a clear report with emoji indicators:
- ✅ passing checks
- ⚠️ caution items
- ❌ failing checks

Always end with an overall recommendation and the top 2-3 things to investigate further.
