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
  │       ├── Build feature matrix:
  │       │     eval(expression) per ticker → Series per ticker
  │       │     Stack to (date, ticker) MultiIndex DataFrame (1 feature column)
  │       ├── Build labels: next-day returns, same (date, ticker) MultiIndex
  │       ├── ModelWrapper(model_name="Linear") from src.core.model_zoo
  │       ├── WalkForwardBacktester.run(features, labels, model, topk=10)
  │       │     → BacktestResult with metrics + walk_forward_results
  │       ├── BacktestValidator().validate(backtest_result)
  │       │     → ValidationResult with .checks: list[ValidationCheck]
  │       │                           .verdict: ValidationVerdict
  │       └── Gate: verdict != ValidationVerdict.REJECTED
  │             AND metrics.sharpe_ratio >= config.rd_agent.min_backtest_sharpe
  │
  └── Only double-gated factors saved to factor library
        FactorDefinition enriched via model_copy(update={...}) with
        backtest_sharpe, backtest_max_drawdown, validation_checks

KB additions:
  - kb["discoveries"]        ← only fully validated (IC + backtest) factors
                               discovery entries now include "sharpe" field
  - kb["backtest_failed"]    ← passed IC, failed backtest (ResearchAnalyst input)
  - kb["eval_results"]       ← IC scores for all tested factors (existing)
```

## Components

### 1. New file: `src/core/factor_backtester.py`

**Class:** `FactorBacktester`

Imports `DataPipeline` at module level for patchability (same pattern as `FactorEvaluator`).

```python
# Module-level import for patch("src.core.factor_backtester.DataPipeline") in tests
from src.core.data_pipeline import DataPipeline

class FactorBacktester:
    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._pipeline = DataPipeline(self.config)

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

**`validate()` internals — step by step:**

1. Fetch 500-day OHLCV via `self._pipeline.yfinance_fallback(tickers=universe, start=start)` (or reuse `df`)
2. For each ticker group, compute factor values:
   ```python
   ns = {"close": grp["close"], "open": grp["open"], ..., "pd": pd, "np": np}
   fval = eval(factor.expression, {"__builtins__": {}}, ns)  # Series indexed by date
   ```
3. Stack into `(date, ticker)` MultiIndex DataFrame (1 column = factor value):
   ```python
   # wide: dates × tickers → stack → (date, ticker) MultiIndex
   factor_wide = pd.concat(factor_series_list, axis=1)  # dates × tickers
   features = factor_wide.stack().rename(factor.name).to_frame()
   ```
4. Build labels with same MultiIndex: next-day returns per (date, ticker)
5. Align features and labels and drop NaN rows **jointly** (not independently, to preserve index alignment):
   ```python
   combined = features.join(labels.rename("_label"), how="inner").dropna()
   features = combined.drop(columns=["_label"])
   labels = combined["_label"]
   ```
6. If fewer than 50 rows after alignment → return `BacktestValidationResult(passed=False, reason="no_data")`
7. `model = ModelWrapper(model_name="Linear", config=self.config)` (imported from `src.core.model_zoo`; pass `self.config` to avoid falling back to the global singleton, which breaks config-injection in tests)
8. `bt_result = WalkForwardBacktester(self.config).run(features, labels, model, topk=_BT_TOPK, walk_forward_months=_BT_WALK_FORWARD_MONTHS, embargo_days=_BT_EMBARGO_DAYS)`
   Note: `WalkForwardBacktester.__init__` accepts `AppConfig`; passing `FullAppConfig` works because `FullAppConfig` extends `AppConfig`.
9. `val_result = BacktestValidator().validate(bt_result)`  → `ValidationResult`
10. Build `checks` dict from validation result:
    ```python
    checks = {vc.name: vc.passed for vc in val_result.checks}
    ```
11. Gate:
    ```python
    sharpe = bt_result.metrics.sharpe_ratio
    passed = (
        val_result.verdict != ValidationVerdict.REJECTED
        and sharpe >= self.config.rd_agent.min_backtest_sharpe
    )
    # Note: ValidationVerdict has three values: APPROVED, CAUTION, REJECTED.
    # Factors with verdict == CAUTION pass the gate (only REJECTED is blocked).
    # This is intentional — the Sharpe floor provides the primary guard against weak results.
    reason = "passed" if passed else (
        "low_sharpe" if sharpe < self.config.rd_agent.min_backtest_sharpe
        else "overfitting" if not checks.get("overfitting_signals", True)
        else "look_ahead_bias" if not checks.get("look_ahead_bias", True)
        else "backtest_rejected"
    )
    ```
12. Return `BacktestValidationResult(factor_name=factor.name, passed=passed, sharpe=sharpe, max_drawdown=bt_result.metrics.max_drawdown, checks=checks, reason=reason)`

**`validate_batch()` optimization:** Fetches OHLCV once via `yfinance_fallback`, passes the result `df` to each `validate()` call.

**Module-level constants** (reuse same values as `FactorEvaluator` to avoid drift):
```python
# Mirror FactorEvaluator constants — keep in sync if tuning lookback windows
_BT_LOOKBACK_DAYS = 500        # same as _STAGE2_LOOKBACK_DAYS in factor_evaluator.py
_BT_TOPK = 10
_BT_WALK_FORWARD_MONTHS = 6    # same as _STAGE2_WALK_FORWARD_MONTHS
_BT_EMBARGO_DAYS = 5           # same as _STAGE2_EMBARGO_DAYS
```

Note: These constants intentionally mirror `FactorEvaluator`'s Stage 2 constants. If either file's lookback is tuned, the other must also be updated. A future improvement could centralize them in `RDAgentConfig`.

### 2. Schema changes: `src/utils/schemas.py`

**Extend `FactorDefinition`** with optional backtest fields (all default `None`; backward-compatible with existing IC-only records):
```python
# Backtest fields — populated only after FactorBacktester.validate() passes
backtest_sharpe: float | None = None
backtest_max_drawdown: float | None = None
validation_checks: dict[str, bool] | None = None
```

**New `BacktestValidationResult`** (uses `Field(default_factory=dict)` for mutable default):
```python
class BacktestValidationResult(BaseModel):
    factor_name: str
    passed: bool
    sharpe: float | None = None
    max_drawdown: float | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    reason: str = ""  # "passed" | "low_sharpe" | "overfitting" | "look_ahead_bias"
                      # | "backtest_rejected" | "eval_error" | "no_data" | "backtest_error"
```

### 3. Config change: `src/utils/config_loader.py`

Add to `RDAgentConfig` (alongside existing `min_ic` and `min_icir`):
```python
min_backtest_sharpe: float = 0.5
```

Note: `StrategyConfig` in `src/utils/config.py` also has a `min_backtest_sharpe: float = 0.5` field — that governs the strategy copilot flow and is distinct. The new field in `RDAgentConfig` governs the factor discovery gate.

Also add to `config/settings.yaml` — `settings.yaml` has **no `rd_agent:` section** today; add the entire block as a new top-level section:
```yaml
rd_agent:
  min_backtest_sharpe: 0.5
```

### 4. Changes to `src/core/rd_agent_runner.py`

**`__init__`:** Add `self._backtester = FactorBacktester(self.config)` (after existing `_proposer`, `_evaluator`, `_analyst`).

**`_EMPTY_KB`:** Add `"backtest_failed": []` to the constant for backward-compat initialization.

**`mine_factors()` loop** — replace the existing `for er, factor in zip(eval_results, proposals)` block with a three-stage flow:

```python
# Stage 1+2: IC gate (existing)
eval_results = self._evaluator.evaluate_factors_batch(proposals)

# Separate IC-passing from IC-failing results
ic_passed_pairs = [(er, f) for er, f in zip(eval_results, proposals) if er.passed]
ic_failed = [er for er in eval_results if not er.passed]

# Log IC-failed factors to tested list (no backtest needed)
for er in ic_failed:
    tested_names.append(er.factor_name)

# Stage 3: Backtest gate — only for IC-passing factors
bt_results = (
    self._backtester.validate_batch([f for _, f in ic_passed_pairs])
    if ic_passed_pairs else []
)

batch_accepted: list[FactorDefinition] = []
bt_results_by_name: dict[str, BacktestValidationResult] = {}
for (er, factor), bt_result in zip(ic_passed_pairs, bt_results):
    tested_names.append(er.factor_name)  # mark as tested regardless of bt outcome
    bt_results_by_name[factor.name] = bt_result
    if bt_result.passed:
        # Enrich FactorDefinition with IC data + backtest data via Pydantic v2 model_copy.
        # proposals[i] has ic_mean=0.0/icir=0.0 defaults — model_copy populates real values.
        factor = factor.model_copy(update={
            "ic_mean": er.stage2_ic or er.stage1_ic,
            "icir": er.stage2_icir or 0.0,
            "source": "rd_agent_llm",
            "backtest_sharpe": bt_result.sharpe,
            "backtest_max_drawdown": bt_result.max_drawdown,
            "validation_checks": bt_result.checks,
        })
        batch_accepted.append(factor)
        kb["discoveries"].append({
            "date": str(date.today()),
            "factor": factor.name,
            "ic": round(er.stage2_ic or er.stage1_ic, 6),
            "sharpe": round(bt_result.sharpe, 4) if bt_result.sharpe else None,
        })
    else:
        kb["backtest_failed"].append({
            "date": str(date.today()),
            "factor": factor.name,
            "stage1_ic": er.stage1_ic,
            "reason": bt_result.reason,
        })

# Preserve existing eval_results tracking for all tested factors (both IC-pass and IC-fail).
# Use "factor_name" key to match the existing KB schema (do NOT rename to "factor").
kb.setdefault("eval_results", [])
for er in eval_results:
    kb["eval_results"].append({
        "factor_name": er.factor_name,  # existing key — must not change
        "stage1_ic": er.stage1_ic,
        "stage2_ic": er.stage2_ic,
        "stage2_icir": er.stage2_icir,
        "passed": er.passed,
        "date": str(date.today()),
        "reason": er.reason,
    })
```

### 5. Changes to `src/core/research_analyst.py`

**`_format_results_table()`** currently receives `list[EvalResult]`. To show backtest results, change signature to also accept an optional dict of backtest results:

```python
@staticmethod
def _format_results_table(
    results: list[EvalResult],
    bt_results: dict[str, BacktestValidationResult] | None = None,
) -> str:
```

When `bt_results` is provided, add a `bt_sharpe` column after `ICIR`:
```
name | stage1_IC | stage2_IC | ICIR | bt_sharpe | passed
```

**Full updated call chain** for passing backtest results:

```python
# write_memo() — public API gains optional bt_results param
def write_memo(
    self,
    batch_results: list[EvalResult],
    kb: dict,
    bt_results: dict[str, BacktestValidationResult] | None = None,
) -> str:
    if self._llm_available:
        try:
            return self._llm_write_memo(batch_results, kb, bt_results=bt_results)
        except Exception as exc:
            ...
    return self._fallback_memo(batch_results)  # fallback intentionally omits bt_sharpe column

# _llm_write_memo() — passes bt_results down to _format_results_table
def _llm_write_memo(
    self,
    batch_results: list[EvalResult],
    kb: dict,
    bt_results: dict[str, BacktestValidationResult] | None = None,
) -> str:
    results_table = self._format_results_table(batch_results, bt_results=bt_results)
    ...
```

In `mine_factors()`, after building `bt_results_by_name`, pass it to `write_memo`:
```python
memo = self._analyst.write_memo(eval_results, kb, bt_results=bt_results_by_name)
```

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Expression eval error | Catch, log warning, return `BacktestValidationResult(passed=False, reason="eval_error")` |
| DataPipeline returns empty / None | Return `BacktestValidationResult(passed=False, reason="no_data")` |
| Fewer than 50 aligned (date, ticker) rows | Return `BacktestValidationResult(passed=False, reason="no_data")` |
| `WalkForwardBacktester.run()` raises | Catch all exceptions, log warning, return `reason="backtest_error"` |
| `BacktestValidator.validate()` raises | Catch all exceptions, log warning, return `reason="backtest_error"` |
| Sharpe below `min_backtest_sharpe` | `passed=False, reason="low_sharpe"` |
| `verdict == ValidationVerdict.REJECTED` | `passed=False, reason` derived from first failed critical check |

## Testing: `tests/test_factor_backtester.py`

~15 unit tests, all mocked (no network/yfinance calls). Use `patch("src.core.factor_backtester.DataPipeline")` for isolation.

1. `test_validate_passes_with_good_backtest` — mock returns Sharpe=1.2, verdict=APPROVED
2. `test_validate_fails_low_sharpe` — mock returns Sharpe=0.2, verdict=APPROVED → passed=False, reason="low_sharpe"
3. `test_validate_fails_verdict_rejected` — mock verdict=REJECTED → passed=False
4. `test_validate_bad_expression_returns_eval_error` — bad expression string → reason="eval_error"
5. `test_validate_no_data` — DataPipeline returns empty DataFrame → reason="no_data"
6. `test_validate_backtest_error` — WalkForwardBacktester raises → reason="backtest_error"
7. `test_validate_batch_fetches_once` — assert `yfinance_fallback` called once for N factors
8. `test_validate_batch_mixed_results` — some pass, some fail
9. `test_backtest_validation_result_schema` — Pydantic fields present; `checks` default is `{}` not shared
10. `test_factor_definition_backtest_fields` — new fields default to `None`; existing fields unchanged
11. `test_mine_factors_backtest_gate` — only double-gated factors in `save_factor_library`; mock `_evaluator.evaluate_factors_batch` and `_backtester.validate_batch` on the `RDAgentRunner` instance (not via `DataPipeline` patch)
12. `test_mine_factors_backtest_failed_logged` — IC-pass/bt-fail logged to `kb["backtest_failed"]`; same mock strategy as test 11
13. `test_mine_factors_discoveries_include_sharpe` — discovery entry has `"sharpe"` key; same mock strategy as test 11
14. `test_research_analyst_memo_format_with_backtest` — table has `bt_sharpe` column when dict provided
15. `test_config_min_backtest_sharpe_default` — `RDAgentConfig().min_backtest_sharpe == 0.5`

## File Summary

| File | Change |
|------|--------|
| `src/core/factor_backtester.py` | **New** — `FactorBacktester` class |
| `src/utils/schemas.py` | Extend `FactorDefinition` + add `BacktestValidationResult` |
| `src/utils/config_loader.py` | Add `min_backtest_sharpe: float = 0.5` to `RDAgentConfig` |
| `config/settings.yaml` | Add `min_backtest_sharpe: 0.5` under `rd_agent:` |
| `src/core/rd_agent_runner.py` | Wire `FactorBacktester`; add `"backtest_failed"` to `_EMPTY_KB`; three-stage loop |
| `src/core/research_analyst.py` | Add optional `bt_results` param to `_format_results_table()` |
| `tests/test_factor_backtester.py` | **New** — ~15 unit tests |
