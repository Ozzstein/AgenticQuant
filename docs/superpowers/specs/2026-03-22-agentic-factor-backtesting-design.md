# Agentic Factor Backtesting Design Spec

## Goal

Add a mandatory full walk-forward backtest gate after the IC gate in the factor discovery pipeline. Only factors that survive both the IC gate (`FactorEvaluator`) and the backtest gate (`FactorBacktester`) are admitted to the factor library.

## Architecture

```
mine_factors() loop
  │
  ├── FactorProposer.propose_factors()          → List[FactorDefinition]
  │
  ├── FactorEvaluator.evaluate_factors_batch()  → List[EvalResult]
  │     Stage 1: 90-day Spearman IC (rejects ~70%)
  │     Stage 2: 500-day walk-forward IC/ICIR
  │
  ├── [NEW] for factors where er.passed:
  │     FactorBacktester.validate_batch(factors) → List[BacktestValidationResult]
  │       ├── Fetch 500-day OHLCV via DataPipeline (once, shared across batch)
  │       ├── Build feature matrix (eval expression per ticker → wide DataFrame)
  │       ├── Build labels (next-day cross-sectional returns)
  │       ├── WalkForwardBacktester.run(features, labels, LinearModel(), topk=10)
  │       ├── BacktestValidator.validate(backtest_result) → 8 checks
  │       └── Returns: passed, sharpe, max_drawdown, checks, reason
  │
  └── Only double-gated factors saved to factor library
        FactorDefinition enriched with backtest_sharpe, backtest_max_drawdown,
        validation_checks before persistence

KB additions:
  - kb["discoveries"]        ← only fully validated (IC + backtest) factors
  - kb["backtest_failed"]    ← passed IC, failed backtest (ResearchAnalyst input)
  - kb["eval_results"]       ← IC scores for all tested factors (existing)
```

## Components

### 1. New file: `src/core/factor_backtester.py`

**Class:** `FactorBacktester`

```python
class FactorBacktester:
    def __init__(self, config: FullAppConfig | None = None) -> None: ...

    def validate(
        self,
        factor: FactorDefinition,
        tickers: list[str] | None = None,
        df: pd.DataFrame | None = None,  # pre-fetched OHLCV; avoids double-fetch
    ) -> BacktestValidationResult: ...

    def validate_batch(
        self,
        factors: list[FactorDefinition],
        tickers: list[str] | None = None,
    ) -> list[BacktestValidationResult]: ...
```

**`validate()` internals:**
1. Fetch 500-day OHLCV via `DataPipeline.yfinance_fallback()` (or reuse `df`)
2. Build feature matrix: `eval(factor.expression, {"__builtins__": {}}, ns)` per ticker → pivot to `dates × tickers` wide DataFrame
3. Build labels: next-day cross-sectional returns (same shape, aligned)
4. Stack features + labels into MultiIndex format expected by `WalkForwardBacktester`
5. Run `WalkForwardBacktester(self.config).run(features, labels, LinearModel(), topk=10)`
6. Run `BacktestValidator().validate(backtest_result)` → dict of 8 check results
7. Gate: `passed = sharpe >= config.rd_agent.min_backtest_sharpe AND checks["look_ahead_bias"] AND NOT checks["overfitting_signals"]`
8. Return `BacktestValidationResult`

**`validate_batch()` optimization:** Fetches OHLCV once, passes `df` to each `validate()` call.

**Constants (module-level):**
```python
_BT_LOOKBACK_DAYS = 500
_BT_TOPK = 10
_BT_WALK_FORWARD_MONTHS = 6
_BT_EMBARGO_DAYS = 5
```

### 2. Schema changes: `src/utils/schemas.py`

**Extend `FactorDefinition`** with optional backtest fields:
```python
backtest_sharpe: float | None = None
backtest_max_drawdown: float | None = None
validation_checks: dict[str, bool] | None = None
```

**New `BacktestValidationResult`:**
```python
class BacktestValidationResult(BaseModel):
    factor_name: str
    passed: bool
    sharpe: float | None = None
    max_drawdown: float | None = None
    checks: dict[str, bool] = {}
    reason: str = ""  # "passed" | "low_sharpe" | "eval_error" | "no_data"
```

### 3. Config change: `src/utils/config.py`

Add to `RDAgentConfig`:
```python
min_backtest_sharpe: float = 0.5
```

And in `config/settings.yaml`:
```yaml
rd_agent:
  min_backtest_sharpe: 0.5
```

### 4. Changes to `src/core/rd_agent_runner.py`

**`__init__`:** Add `self._backtester = FactorBacktester(self.config)`

**`mine_factors()` loop** — add backtest gate after IC gate:
```python
# IC gate (existing)
eval_results = self._evaluator.evaluate_factors_batch(proposals)

# Backtest gate (new) — only for IC-passing factors
ic_passed_pairs = [(er, f) for er, f in zip(eval_results, proposals) if er.passed]
if ic_passed_pairs:
    bt_results = self._backtester.validate_batch([f for _, f in ic_passed_pairs])
else:
    bt_results = []

for (er, factor), bt_result in zip(ic_passed_pairs, bt_results):
    tested_names.append(er.factor_name)
    if bt_result.passed:
        factor = FactorDefinition(
            **factor.model_dump(),
            backtest_sharpe=bt_result.sharpe,
            backtest_max_drawdown=bt_result.max_drawdown,
            validation_checks=bt_result.checks,
        )
        batch_accepted.append(factor)
        kb["discoveries"].append({
            "date": str(date.today()),
            "factor": factor.name,
            "ic": round(er.stage2_ic or er.stage1_ic, 6),
            "sharpe": bt_result.sharpe,
        })
    else:
        kb.setdefault("backtest_failed", []).append({
            "date": str(date.today()),
            "factor": factor.name,
            "stage1_ic": er.stage1_ic,
            "reason": bt_result.reason,
        })

# Also log factors that failed the IC gate (existing pattern)
for er in eval_results:
    if not er.passed:
        tested_names.append(er.factor_name)
```

### 5. Changes to `src/core/research_analyst.py`

**`_format_results_table()`** — add `backtest_sharpe` column when present in `EvalResult`:
```
name | stage1_IC | stage2_IC | ICIR | bt_sharpe | passed
```

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Expression eval error | `BacktestValidationResult(passed=False, reason="eval_error")` |
| DataPipeline returns empty | `BacktestValidationResult(passed=False, reason="no_data")` |
| WalkForwardBacktester raises | Catch, log warning, return `reason="backtest_error"` |
| Sharpe below threshold | `passed=False, reason="low_sharpe"` |
| look_ahead_bias check fails | `passed=False, reason="look_ahead_bias"` |
| overfitting_signals check fails | `passed=False, reason="overfitting"` |

## Testing: `tests/test_factor_backtester.py`

~15 unit tests, all mocked (no network/yfinance calls):

1. `test_validate_passes_with_good_backtest` — mock returns Sharpe=1.2, all checks pass
2. `test_validate_fails_low_sharpe` — mock returns Sharpe=0.2, passed=False
3. `test_validate_bad_expression_returns_eval_error` — bad expression string
4. `test_validate_no_data` — DataPipeline returns empty DataFrame
5. `test_validate_look_ahead_bias_fails` — look_ahead_bias check returns False
6. `test_validate_overfitting_fails` — overfitting_signals check returns True
7. `test_validate_batch_fetches_once` — assert yfinance_fallback called once for N factors
8. `test_validate_batch_mixed_results` — some pass, some fail
9. `test_backtest_validation_result_schema` — Pydantic fields present and typed
10. `test_factor_definition_backtest_fields` — new fields default to None
11. `test_mine_factors_backtest_gate` — only double-gated factors saved to library
12. `test_mine_factors_backtest_failed_logged` — IC-pass/bt-fail logged to kb["backtest_failed"]
13. `test_mine_factors_discoveries_include_sharpe` — discoveries record has "sharpe" key
14. `test_research_analyst_memo_format_with_backtest` — memo table has bt_sharpe column
15. `test_config_min_backtest_sharpe_default` — default value is 0.5

## File Summary

| File | Change |
|------|--------|
| `src/core/factor_backtester.py` | **New** — `FactorBacktester` class |
| `src/utils/schemas.py` | Extend `FactorDefinition` + add `BacktestValidationResult` |
| `src/utils/config.py` | Add `min_backtest_sharpe: float = 0.5` to `RDAgentConfig` |
| `config/settings.yaml` | Add `min_backtest_sharpe: 0.5` under `rd_agent:` |
| `src/core/rd_agent_runner.py` | Wire `FactorBacktester` into `mine_factors()` |
| `src/core/research_analyst.py` | Add `bt_sharpe` column to results table |
| `tests/test_factor_backtester.py` | **New** — ~15 unit tests |
