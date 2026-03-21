# TASK-09: Macro Regime Detector — HMM + RD-Agent Evolution

## Context
Tasks 01-08 are complete. The macro regime detector needs a proper implementation. We use a Hidden Markov Model that discovers regimes from historical signal data, and RD-Agent to continuously improve it.

## Depends On
All of Tasks 01-08 being functional.

## What to Build

### 1. HMM Regime Detector (`src/core/macro_regime.py`)

**New dependency:** `hmmlearn`

**Input signals (daily observations):**
- VIX level (yfinance `^VIX`)
- VIX 10-day rate of change
- Yield curve slope: 10Y minus 2Y treasury yield
- S&P 500 breadth: percentage of stocks above 200-day SMA (approximate from Qlib universe data)
- S&P 500 20-day realized volatility
- S&P 500 20-day cumulative return
- DXY 20-day rate of change
- Credit spread proxy: HYG vs TLT price-based yield difference

**Preprocessing:**
- Normalize all signals to z-scores using a rolling 252-day window. Prevents the HMM from being dominated by whichever signal has the largest absolute values.
- Forward-fill missing data up to 5 days, drop remaining NaN rows from training.

**Model:**
- Gaussian HMM with N states (default N=4, configurable)
- Covariance type: "full" (captures correlations between signals within each state)
- Fit on minimum 5 years of daily data (2018-2023 recommended — covers bull, COVID crash, recovery, 2022 bear, 2023 recovery)
- Train with multiple random initializations (n_init=10), keep highest log-likelihood to avoid local optima

**State labeling:**
After fitting, the model discovers 4 unnamed states. Auto-label them:
- Rank states by a composite risk score (weighted combination of mean VIX, mean breadth, mean momentum in each state)
- Lowest risk → "risk_on"
- Second lowest → "neutral"
- Second highest → "risk_off"
- Highest risk → "crisis"

**Daily inference:**
- Feed today's signal values into the fitted HMM
- Return: predicted regime, probability of each state, most likely state sequence for last 20 days
- Regime only changes if new state probability exceeds confidence threshold (default 0.7) — prevents flickering on noisy days

**Transition smoothing:**
- Minimum regime duration: regime must persist for at least N trading days (default 3) before the strategy selector acts on it. Prevents whipsawing.

**Retraining schedule:**
- Retrain quarterly on expanding window (always include all available history). Fits in seconds. Allows the model to incorporate new regime patterns.

### 2. Regime Evaluation Framework

The regime detector is never evaluated in isolation. It's measured by whether the strategy selector makes better decisions with it.

Build an evaluation that:
- Runs the full walk-forward pipeline with the current detector
- Measures end-to-end Sharpe, max drawdown, and regime-conditional returns
- This is what RD-Agent uses to judge its own improvements

### 3. RD-Agent Regime Evolution (extend `src/core/rd_agent_runner.py`)

New mode: `evolve-regime`. The R&D loop proposes, implements, and evaluates changes to the HMM regime detector.

**What RD-Agent can change:**

- **Input signals** — add new signals (put/call ratio, copper/gold ratio, crypto total market cap, high-yield spread) or remove signals that don't improve downstream Sharpe
- **Number of states** — try 3, 4, 5, 6 states and evaluate which count produces the best pipeline performance
- **Preprocessing** — different normalization windows (126 vs 252 days), different transformations (raw vs rate-of-change vs z-score vs percentile rank), different lookback periods for momentum signals
- **Model variant** — try replacing Gaussian HMM with other HMM emission types (GMM-HMM for multimodal distributions) or entirely different approaches (random forest classifier, LSTM) if they produce better downstream results
- **Transition logic** — confidence thresholds, minimum regime duration, hysteresis rules
- **Evaluation metric** — default is pipeline Sharpe, but could optimize for drawdown reduction or risk-adjusted return instead

**Evaluation:** every proposed change is measured by running a full walk-forward backtest with the modified detector vs the current detector. The only metric that matters is whether the change improves end-to-end trading performance. A "more accurate" regime label that doesn't improve Sharpe is worthless.

**Knowledge base:** saves what signal combinations work, optimal state counts, which model variants were tested, transition parameters. Next run builds on accumulated knowledge.

### 4. CLI Commands

Add to `scripts/run_rd_agent.py`:
- `evolve-regime --iterations 10 --budget 10.0` — run the regime R&D loop
- `copilot-regime "add put/call ratio as a signal"` — propose a specific change, RD-Agent implements and evaluates

Add to `scripts/run_pipeline.py`:
- `regime-status` — show current regime, probability per state, last 20 days history, signal values
- `regime-backtest` — evaluate current detector by running full walk-forward pipeline and reporting regime-conditional performance

### 5. Config

Add to `config/agents_config.yaml`:
```
macro_regime:
  hmm_states: 4
  confidence_threshold: 0.7
  min_regime_days: 3
  retrain_frequency: "quarterly"
  normalization_window: 252
  signals:
    - vix_level
    - vix_roc_10d
    - yield_curve_10y2y
    - sp500_breadth
    - sp500_realized_vol_20d
    - sp500_momentum_20d
    - dxy_roc_20d
    - credit_spread_proxy
```

## Inputs
- Historical signal data — at least 5 years
- Walk-forward backtester from Task 03
- Strategy selector from Task 08
- RD-Agent infrastructure from Task 06

## Outputs
- HMM regime detector classifying risk_on/neutral/risk_off/crisis daily
- Evaluation framework measuring detector quality via pipeline Sharpe
- RD-Agent evolve-regime mode for continuous improvement
- CLI commands for status, evaluation, and evolution

## Done Criteria
- [x] HMM trains on 5+ years of data and assigns 4 labeled states
- [x] State labels are auto-assigned (risk_on has lowest VIX + highest breadth)
- [x] Daily inference returns regime + probability for each state
- [x] Confidence threshold prevents flickering (regime doesn't change on low-probability days)
- [x] Minimum duration smoothing works (regime holds for at least N days)
- [x] HMM correctly identifies the March 2020 COVID crash as "crisis" and the 2021 rally as "risk_on"
- [x] Evaluation framework runs full walk-forward and reports regime-conditional Sharpe
- [x] `python scripts/run_rd_agent.py evolve-regime --iterations 3` proposes and evaluates at least one detector change
- [x] RD-Agent can add/remove a signal and measure impact on pipeline Sharpe
- [x] RD-Agent can change state count and compare results
- [x] Knowledge base saves regime evolution results across runs
- [x] `regime-status` shows current regime with probability
- [x] `regime-backtest` reports performance
- [x] Config loads all macro_regime settings correctly
- [x] `pytest tests/test_regime.py` passes
