# Agentic Factor Backtesting Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a mandatory walk-forward backtest gate after the IC gate in `mine_factors()` so only double-validated factors (IC + backtest) reach the factor library.

**Architecture:** New `FactorBacktester` class wraps `WalkForwardBacktester` + `BacktestValidator` with a one-fetch-per-batch optimization. `RDAgentRunner.mine_factors()` splits IC-passing proposals into a second gate; `ResearchAnalyst` gains an optional `bt_sharpe` column in its results table.

**Tech Stack:** Python 3.11+, Pydantic v2, pandas, loguru. Existing: `WalkForwardBacktester`, `BacktestValidator`, `ModelWrapper`, `DataPipeline` from `src/core/`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/utils/schemas.py` | Modify | Add `BacktestValidationResult`; extend `FactorDefinition` with 3 optional fields |
| `src/utils/config_loader.py` | Modify | Add `min_backtest_sharpe: float = 0.5` to `RDAgentConfig` |
| `config/settings.yaml` | Modify | Add new `rd_agent:` top-level section |
| `src/core/factor_backtester.py` | **Create** | `FactorBacktester` class with `validate()` + `validate_batch()` |
| `src/core/rd_agent_runner.py` | Modify | Wire `_backtester`; update `_EMPTY_KB`; replace `mine_factors()` loop |
| `src/core/research_analyst.py` | Modify | Update `_format_results_table()` + `write_memo()` / `_llm_write_memo()` |
| `tests/test_factor_backtester.py` | **Create** | 15 unit tests — all mocked, no network calls |

---

## Chunk 1: Foundation — Schema, Config, FactorBacktester

### Task 1: Schemas, Config, and Settings

**Files:**
- Modify: `src/utils/schemas.py:228-238` (extend `FactorDefinition`) and after line 284 (add `BacktestValidationResult`)
- Modify: `src/utils/config_loader.py:33` (add `min_backtest_sharpe` to `RDAgentConfig`)
- Modify: `config/settings.yaml` (add new `rd_agent:` section)
- Create: `tests/test_factor_backtester.py` (tests 9, 10, 15 first)

- [ ] **Step 1: Write failing tests for schemas and config**

Create `tests/test_factor_backtester.py` with the foundational tests. These will fail until schemas and config are updated.

```python
"""Tests for FactorBacktester and related schemas/config."""
from __future__ import annotations

import pytest
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Task 1: Schema and config tests
# ---------------------------------------------------------------------------

def test_backtest_validation_result_schema():
    """BacktestValidationResult has correct fields; checks default is {} not shared."""
    from src.utils.schemas import BacktestValidationResult

    r1 = BacktestValidationResult(factor_name="f1", passed=True, reason="passed")
    r2 = BacktestValidationResult(factor_name="f2", passed=False, reason="low_sharpe")

    assert r1.factor_name == "f1"
    assert r1.passed is True
    assert r1.sharpe is None
    assert r1.max_drawdown is None
    assert r1.checks == {}
    assert r1.reason == "passed"
    # Verify checks defaults are NOT shared between instances
    r1.checks["x"] = True
    assert "x" not in r2.checks, "checks dict should not be shared between instances"


def test_factor_definition_backtest_fields():
    """FactorDefinition gains 3 optional backtest fields defaulting to None."""
    from src.utils.schemas import FactorDefinition

    fd = FactorDefinition(name="test", expression="close.pct_change()")
    assert hasattr(fd, "backtest_sharpe")
    assert hasattr(fd, "backtest_max_drawdown")
    assert hasattr(fd, "validation_checks")
    assert fd.backtest_sharpe is None
    assert fd.backtest_max_drawdown is None
    assert fd.validation_checks is None
    # Existing fields unchanged
    assert fd.ic_mean == 0.0
    assert fd.icir == 0.0
    assert fd.source == "builtin"


def test_config_min_backtest_sharpe_default():
    """RDAgentConfig has min_backtest_sharpe defaulting to 0.5."""
    from src.utils.config_loader import RDAgentConfig

    cfg = RDAgentConfig()
    assert hasattr(cfg, "min_backtest_sharpe")
    assert cfg.min_backtest_sharpe == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py -v --tb=short 2>&1 | head -30
```

Expected: `FAILED` — `ImportError: cannot import name 'BacktestValidationResult'` and `AttributeError: 'FactorDefinition' object has no attribute 'backtest_sharpe'`.

- [ ] **Step 3: Extend `FactorDefinition` in `src/utils/schemas.py`**

Find `class FactorDefinition` (line 228). After `description: str = ""` (line 237), add the backtest fields:

```python
class FactorDefinition(BaseModel):
    """Factor metadata for the factor library."""

    name: str
    expression: str = ""
    category: str = "custom"
    ic_mean: float = 0.0
    icir: float = 0.0
    source: str = "builtin"
    description: str = ""
    # Backtest fields — populated only after FactorBacktester.validate() passes
    backtest_sharpe: float | None = None
    backtest_max_drawdown: float | None = None
    validation_checks: dict[str, bool] | None = None
```

- [ ] **Step 4: Add `BacktestValidationResult` to `src/utils/schemas.py`**

Find `class EvalResult` (line 275). Insert the new class **after** `EvalResult` (i.e., after line ~284 where `EvalResult` ends) to maintain logical ordering: proposals → IC evaluation → backtest validation:

```python
class BacktestValidationResult(BaseModel):
    """Walk-forward backtest gate result from FactorBacktester."""

    factor_name: str
    passed: bool
    sharpe: float | None = None
    max_drawdown: float | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    reason: str = ""  # "passed" | "low_sharpe" | "overfitting" | "look_ahead_bias"
                      # | "backtest_rejected" | "eval_error" | "no_data" | "backtest_error"
```

- [ ] **Step 5: Add `min_backtest_sharpe` to `RDAgentConfig` in `src/utils/config_loader.py`**

Find `min_icir: float = 0.3` (line 33). Add below it:

```python
    min_ic: float = 0.02
    min_icir: float = 0.3
    min_backtest_sharpe: float = 0.5
```

- [ ] **Step 6: Add `rd_agent:` section to `config/settings.yaml`**

`settings.yaml` has **no** `rd_agent:` section today. Append this as a new top-level block at the end of the file:

```yaml
rd_agent:
  min_backtest_sharpe: 0.5
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py -v --tb=short
```

Expected: 3 tests `PASSED`.

- [ ] **Step 8: Commit**

```bash
git add src/utils/schemas.py src/utils/config_loader.py config/settings.yaml tests/test_factor_backtester.py
git commit -m "feat: add BacktestValidationResult schema, FactorDefinition backtest fields, min_backtest_sharpe config"
```

---

### Task 2: `FactorBacktester` class

**Files:**
- Create: `src/core/factor_backtester.py`
- Modify: `tests/test_factor_backtester.py` (add tests 1–8)

- [ ] **Step 1: Add mock helper and tests 1–8 to the test file**

Append these to `tests/test_factor_backtester.py`:

```python
import pandas as pd
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared helpers for FactorBacktester tests
# ---------------------------------------------------------------------------

def _make_ohlcv(n_days: int = 100, n_tickers: int = 5) -> pd.DataFrame:
    """Build a minimal long-format OHLCV DataFrame for testing."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"][:n_tickers]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    frames = []
    for t in tickers:
        frame = pd.DataFrame(
            {
                "close": [100 + i * 0.1 for i in range(n_days)],
                "open": [99 + i * 0.1 for i in range(n_days)],
                "high": [101 + i * 0.1 for i in range(n_days)],
                "low": [98 + i * 0.1 for i in range(n_days)],
                "volume": [1_000_000] * n_days,
                "ticker": t,
            },
            index=dates,
        )
        frames.append(frame)
    df = pd.concat(frames)
    df.index.name = "date"
    return df


def _make_bt_mocks(sharpe: float = 1.2, verdict_str: str = "APPROVED"):
    """Return (mock_wfbt_cls, mock_validator_cls) configured for a given outcome."""
    from src.utils.schemas import (
        BacktestMetrics, BacktestResult, ValidationCheck,
        ValidationResult, ValidationVerdict,
    )
    metrics = BacktestMetrics(sharpe_ratio=sharpe, max_drawdown=-0.05)
    bt_result = BacktestResult(metrics=metrics)
    val_result = ValidationResult(
        checks=[ValidationCheck(name="overfitting_signals", passed=True)],
        verdict=ValidationVerdict(verdict_str),
    )
    mock_wfbt_instance = MagicMock()
    mock_wfbt_instance.run.return_value = bt_result
    mock_wfbt_cls = MagicMock(return_value=mock_wfbt_instance)

    mock_val_instance = MagicMock()
    mock_val_instance.validate.return_value = val_result
    mock_validator_cls = MagicMock(return_value=mock_val_instance)

    return mock_wfbt_cls, mock_validator_cls


# ---------------------------------------------------------------------------
# Task 2: FactorBacktester tests
# ---------------------------------------------------------------------------

def test_validate_passes_with_good_backtest():
    """validate() returns passed=True when Sharpe >= 0.5 and verdict=APPROVED."""
    from src.utils.schemas import FactorDefinition
    mock_wfbt, mock_val = _make_bt_mocks(sharpe=1.2, verdict_str="APPROVED")
    factor = FactorDefinition(name="f1", expression="close.pct_change(20)")

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls, \
         patch("src.core.factor_backtester.WalkForwardBacktester", mock_wfbt), \
         patch("src.core.factor_backtester.BacktestValidator", mock_val):
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        result = backtester.validate(factor)

    assert result.passed is True
    assert result.sharpe == 1.2
    assert result.reason == "passed"


def test_validate_fails_low_sharpe():
    """validate() returns passed=False, reason='low_sharpe' when Sharpe < 0.5."""
    from src.utils.schemas import FactorDefinition
    mock_wfbt, mock_val = _make_bt_mocks(sharpe=0.2, verdict_str="APPROVED")
    factor = FactorDefinition(name="f1", expression="close.pct_change(20)")

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls, \
         patch("src.core.factor_backtester.WalkForwardBacktester", mock_wfbt), \
         patch("src.core.factor_backtester.BacktestValidator", mock_val):
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        result = backtester.validate(factor)

    assert result.passed is False
    assert result.reason == "low_sharpe"


def test_validate_fails_verdict_rejected():
    """validate() returns passed=False when verdict=REJECTED regardless of Sharpe."""
    from src.utils.schemas import FactorDefinition
    mock_wfbt, mock_val = _make_bt_mocks(sharpe=1.5, verdict_str="REJECTED")
    factor = FactorDefinition(name="f1", expression="close.pct_change(20)")

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls, \
         patch("src.core.factor_backtester.WalkForwardBacktester", mock_wfbt), \
         patch("src.core.factor_backtester.BacktestValidator", mock_val):
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        result = backtester.validate(factor)

    assert result.passed is False


def test_validate_bad_expression_returns_eval_error():
    """validate() returns reason='eval_error' when expression raises on eval."""
    from src.utils.schemas import FactorDefinition
    factor = FactorDefinition(name="bad", expression="this_does_not_exist()")

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls:
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        result = backtester.validate(factor)

    assert result.passed is False
    assert result.reason == "eval_error"


def test_validate_no_data():
    """validate() returns reason='no_data' when DataPipeline returns empty DataFrame."""
    from src.utils.schemas import FactorDefinition
    factor = FactorDefinition(name="f1", expression="close.pct_change()")

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls:
        mock_dp_cls.return_value.yfinance_fallback.return_value = pd.DataFrame()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        result = backtester.validate(factor)

    assert result.passed is False
    assert result.reason == "no_data"


def test_validate_backtest_error():
    """validate() returns reason='backtest_error' when WalkForwardBacktester raises."""
    from src.utils.schemas import FactorDefinition
    factor = FactorDefinition(name="f1", expression="close.pct_change(20)")

    mock_wfbt_instance = MagicMock()
    mock_wfbt_instance.run.side_effect = RuntimeError("backtester exploded")
    mock_wfbt_cls = MagicMock(return_value=mock_wfbt_instance)

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls, \
         patch("src.core.factor_backtester.WalkForwardBacktester", mock_wfbt_cls):
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        result = backtester.validate(factor)

    assert result.passed is False
    assert result.reason == "backtest_error"


def test_validate_batch_fetches_once():
    """validate_batch() calls yfinance_fallback exactly once for N factors."""
    from src.utils.schemas import FactorDefinition
    mock_wfbt, mock_val = _make_bt_mocks(sharpe=1.0)
    factors = [
        FactorDefinition(name=f"f{i}", expression="close.pct_change(20)")
        for i in range(3)
    ]

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls, \
         patch("src.core.factor_backtester.WalkForwardBacktester", mock_wfbt), \
         patch("src.core.factor_backtester.BacktestValidator", mock_val):
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        results = backtester.validate_batch(factors)

    assert len(results) == 3
    # yfinance_fallback must be called exactly once (batch optimization)
    assert mock_dp_cls.return_value.yfinance_fallback.call_count == 1


def test_validate_batch_mixed_results():
    """validate_batch() returns mixed pass/fail results for different factors."""
    from src.utils.schemas import FactorDefinition, BacktestMetrics, BacktestResult, ValidationResult, ValidationVerdict, ValidationCheck

    # Good factor
    metrics_good = BacktestMetrics(sharpe_ratio=1.5, max_drawdown=-0.05)
    bt_good = BacktestResult(metrics=metrics_good)
    val_good = ValidationResult(
        checks=[ValidationCheck(name="test", passed=True)],
        verdict=ValidationVerdict.APPROVED,
    )
    # Bad factor (low sharpe)
    metrics_bad = BacktestMetrics(sharpe_ratio=0.1, max_drawdown=-0.3)
    bt_bad = BacktestResult(metrics=metrics_bad)
    val_bad = ValidationResult(
        checks=[ValidationCheck(name="test", passed=True)],
        verdict=ValidationVerdict.APPROVED,
    )

    call_count = [0]
    def run_side_effect(*args, **kwargs):
        call_count[0] += 1
        return bt_good if call_count[0] == 1 else bt_bad

    mock_wfbt_instance = MagicMock()
    mock_wfbt_instance.run.side_effect = run_side_effect
    mock_wfbt_cls = MagicMock(return_value=mock_wfbt_instance)
    mock_val_instance = MagicMock()
    mock_val_instance.validate.return_value = val_good
    mock_val_cls = MagicMock(return_value=mock_val_instance)

    factors = [
        FactorDefinition(name="good", expression="close.pct_change(20)"),
        FactorDefinition(name="bad", expression="close.pct_change(5)"),
    ]

    with patch("src.core.factor_backtester.DataPipeline") as mock_dp_cls, \
         patch("src.core.factor_backtester.WalkForwardBacktester", mock_wfbt_cls), \
         patch("src.core.factor_backtester.BacktestValidator", mock_val_cls):
        mock_dp_cls.return_value.yfinance_fallback.return_value = _make_ohlcv()
        from src.core.factor_backtester import FactorBacktester
        backtester = FactorBacktester()
        results = backtester.validate_batch(factors)

    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].reason == "low_sharpe"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py::test_validate_passes_with_good_backtest -v --tb=short
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.core.factor_backtester'`.

- [ ] **Step 3: Create `src/core/factor_backtester.py`**

```python
"""FactorBacktester — walk-forward backtest gate for the factor discovery pipeline.

Validates IC-passing FactorDefinition objects against a full walk-forward backtest.
Only factors that also pass the backtest gate are admitted to the factor library.

Imports DataPipeline at module level so patch("src.core.factor_backtester.DataPipeline")
works in tests (same pattern as FactorEvaluator).
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

# Module-level imports for patchability in tests
from src.core.backtester import BacktestValidator, WalkForwardBacktester
from src.core.data_pipeline import DataPipeline
from src.core.model_zoo import ModelWrapper
from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import BacktestValidationResult, FactorDefinition, ValidationVerdict

# ---------------------------------------------------------------------------
# Constants — mirror FactorEvaluator Stage 2 constants
# If either file's lookback is tuned, the other must also be updated.
# ---------------------------------------------------------------------------
_BT_LOOKBACK_DAYS = 500        # same as _STAGE2_LOOKBACK_DAYS in factor_evaluator.py
_BT_TOPK = 10
_BT_WALK_FORWARD_MONTHS = 6    # same as _STAGE2_WALK_FORWARD_MONTHS
_BT_EMBARGO_DAYS = 5           # same as _STAGE2_EMBARGO_DAYS

_BT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "CVX", "LLY", "ABBV",
    "MRK", "PEP", "KO", "AVGO", "COST", "MCD", "ACN", "TMO", "DHR",
    "WMT", "BAC", "XOM",
]


class FactorBacktester:
    """Validate IC-passing factors via a full walk-forward backtest gate.

    Args:
        config: Full application config. Falls back to get_full_config() if None.
    """

    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._pipeline = DataPipeline(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        factor: FactorDefinition,
        tickers: list[str] | None = None,
        df: pd.DataFrame | None = None,
    ) -> BacktestValidationResult:
        """Validate a single factor through the walk-forward backtest gate.

        Args:
            factor: IC-passing FactorDefinition to validate.
            tickers: Optional universe override. Defaults to _BT_UNIVERSE.
            df: Optional pre-fetched OHLCV DataFrame (avoids double-fetch in batch).

        Returns:
            BacktestValidationResult with passed, sharpe, checks, reason.
        """
        universe = tickers if tickers is not None else _BT_UNIVERSE

        # Step 1: Fetch OHLCV (or reuse pre-fetched df from validate_batch)
        if df is None:
            start = (datetime.today() - timedelta(days=_BT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            try:
                df = self._pipeline.yfinance_fallback(tickers=universe, start=start)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorBacktester: data fetch failed for '{}': {}", factor.name, exc)
                return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        if df is None or df.empty:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        # Normalise: ensure date and ticker columns exist
        df = df.copy()
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "date"})
        if "ticker" not in df.columns:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        # Step 2 + 3: Build factor values per ticker, then stack to (date, ticker) MultiIndex
        factor_series_list: list[pd.Series] = []
        try:
            for ticker, grp in df.groupby("ticker"):
                grp = grp.sort_values("date").set_index("date")
                ns = {
                    "close": grp["close"],
                    "open": grp["open"],
                    "high": grp["high"],
                    "low": grp["low"],
                    "volume": grp["volume"].astype(float),
                    "pd": pd,
                    "np": np,
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fval = eval(factor.expression, {"__builtins__": {}}, ns)  # noqa: S307
                if not isinstance(fval, pd.Series):
                    fval = pd.Series(fval, index=grp.index)
                fval.name = ticker
                factor_series_list.append(fval)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester: eval error for '{}': {}", factor.name, exc)
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="eval_error")

        if not factor_series_list:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        factor_wide = pd.concat(factor_series_list, axis=1)  # dates × tickers
        features = factor_wide.stack().rename(factor.name).to_frame()

        # Step 4: Build labels — (date, ticker) MultiIndex Series of next-day returns
        returns_by_ticker: list[pd.Series] = []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("date").set_index("date")
            ret = grp["close"].pct_change().shift(-1)
            ret.name = ticker
            returns_by_ticker.append(ret)
        returns_wide = pd.concat(returns_by_ticker, axis=1)  # dates × tickers
        labels = returns_wide.stack()  # (date, ticker) MultiIndex Series, date at level 0

        # Step 5: Align and joint-drop NaN rows (joint to preserve index alignment)
        combined = features.join(labels.rename("_label"), how="inner").dropna()
        features = combined.drop(columns=["_label"])
        labels = combined["_label"]

        # Step 6: Minimum row check
        if len(features) < 50:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        # Steps 7–9: Run walk-forward backtest
        try:
            model = ModelWrapper(model_name="Linear", config=self.config)  # type: ignore[arg-type]
            bt_result = WalkForwardBacktester(self.config).run(  # type: ignore[arg-type]
                features,
                labels,
                model,
                topk=_BT_TOPK,
                walk_forward_months=_BT_WALK_FORWARD_MONTHS,
                embargo_days=_BT_EMBARGO_DAYS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester: backtest error for '{}': {}", factor.name, exc)
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="backtest_error")

        try:
            val_result = BacktestValidator().validate(bt_result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester: validation error for '{}': {}", factor.name, exc)
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="backtest_error")

        # Step 10: Build checks dict from ValidationCheck list
        checks = {vc.name: vc.passed for vc in val_result.checks}

        # Step 11: Gate
        # ValidationVerdict has APPROVED / CAUTION / REJECTED.
        # CAUTION passes (Sharpe floor is the primary guard); only REJECTED is blocked.
        sharpe = bt_result.metrics.sharpe_ratio
        passed = (
            val_result.verdict != ValidationVerdict.REJECTED
            and sharpe >= self.config.rd_agent.min_backtest_sharpe
        )
        reason = "passed" if passed else (
            "low_sharpe" if sharpe < self.config.rd_agent.min_backtest_sharpe
            else "overfitting" if not checks.get("overfitting_signals", True)
            else "look_ahead_bias" if not checks.get("look_ahead_bias", True)
            else "backtest_rejected"
        )

        return BacktestValidationResult(
            factor_name=factor.name,
            passed=passed,
            sharpe=sharpe,
            max_drawdown=bt_result.metrics.max_drawdown,
            checks=checks,
            reason=reason,
        )

    def validate_batch(
        self,
        factors: list[FactorDefinition],
        tickers: list[str] | None = None,
    ) -> list[BacktestValidationResult]:
        """Validate a batch of factors, fetching OHLCV only once.

        Args:
            factors: List of IC-passing FactorDefinition objects to validate.
            tickers: Optional universe override.

        Returns:
            List of BacktestValidationResult, one per input factor.
        """
        if not factors:
            return []

        universe = tickers if tickers is not None else _BT_UNIVERSE
        start = (datetime.today() - timedelta(days=_BT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        try:
            df = self._pipeline.yfinance_fallback(tickers=universe, start=start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester.validate_batch: data fetch failed: {}", exc)
            df = None

        results: list[BacktestValidationResult] = []
        for factor in factors:
            result = self.validate(factor, tickers=tickers, df=df)
            results.append(result)
            logger.debug(
                "FactorBacktester: {} → passed={}, reason={}",
                factor.name, result.passed, result.reason,
            )
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py -k "test_validate or test_backtest_validation or test_factor_definition or test_config" -v --tb=short
```

Expected: All 11 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/core/factor_backtester.py tests/test_factor_backtester.py
git commit -m "feat: add FactorBacktester class with validate() and validate_batch()"
```

---

## Chunk 2: Integration — RDAgentRunner + ResearchAnalyst

### Task 3: Wire `RDAgentRunner`

**Files:**
- Modify: `src/core/rd_agent_runner.py` (5 precise changes)
- Modify: `tests/test_factor_backtester.py` (add tests 11–13)

- [ ] **Step 1: Add tests 11–13 to the test file**

Append to `tests/test_factor_backtester.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: RDAgentRunner wiring tests
# ---------------------------------------------------------------------------

def _make_runner_with_mocks(tmp_path, ic_pass_count=1, bt_pass_count=1):
    """
    Build an RDAgentRunner with:
      - _evaluator.evaluate_factors_batch mocked to return ic_pass_count passing EvalResults
      - _backtester.validate_batch mocked to return bt_pass_count passing BacktestValidationResults
      - KB stored in tmp_path
    """
    import src.core.rd_agent_runner as rdmod
    from src.core.rd_agent_runner import RDAgentRunner
    from src.utils.schemas import EvalResult, BacktestValidationResult, FactorDefinition
    from unittest.mock import MagicMock, patch

    rdmod._KB_PATH = tmp_path / "kb.json"
    rdmod._KB_DIR = tmp_path

    # Build fake proposals returned by _proposer
    n_factors = max(ic_pass_count, 1)
    fake_factors = [
        FactorDefinition(name=f"factor_{i}", expression="close.pct_change(20)")
        for i in range(n_factors)
    ]

    # Build fake eval results: first ic_pass_count pass, rest fail
    fake_eval_results = []
    for i in range(n_factors):
        fake_eval_results.append(EvalResult(
            factor_name=f"factor_{i}",
            stage1_ic=0.05 if i < ic_pass_count else 0.001,
            stage1_passed=i < ic_pass_count,
            stage2_ic=0.04 if i < ic_pass_count else None,
            stage2_icir=0.5 if i < ic_pass_count else None,
            passed=i < ic_pass_count,
            reason="passed" if i < ic_pass_count else "low_ic",
        ))

    # Build fake backtest results: first bt_pass_count pass
    fake_bt_results = []
    for i in range(ic_pass_count):
        fake_bt_results.append(BacktestValidationResult(
            factor_name=f"factor_{i}",
            passed=i < bt_pass_count,
            sharpe=1.2 if i < bt_pass_count else 0.1,
            max_drawdown=-0.05,
            reason="passed" if i < bt_pass_count else "low_sharpe",
        ))

    with patch("src.core.rd_agent_runner.FactorProposer") as mock_proposer_cls, \
         patch("src.core.rd_agent_runner.FactorEvaluator"), \
         patch("src.core.rd_agent_runner.ResearchAnalyst"), \
         patch("src.core.rd_agent_runner.FactorBacktester"):
        runner = RDAgentRunner()

    runner._proposer = MagicMock()
    runner._proposer.propose_factors.return_value = fake_factors
    runner._evaluator = MagicMock()
    runner._evaluator.evaluate_factors_batch.return_value = fake_eval_results
    runner._backtester = MagicMock()
    runner._backtester.validate_batch.return_value = fake_bt_results
    runner._analyst = MagicMock()
    runner._analyst.load_memo.return_value = ""
    runner._analyst.write_memo.return_value = "memo"

    return runner


def test_mine_factors_backtest_gate(tmp_path):
    """Only factors that pass BOTH IC gate and backtest gate are saved to library."""
    import src.core.rd_agent_runner as rdmod
    from unittest.mock import patch

    original_kb_path = rdmod._KB_PATH
    original_kb_dir = rdmod._KB_DIR

    try:
        runner = _make_runner_with_mocks(tmp_path, ic_pass_count=2, bt_pass_count=1)

        with patch.object(runner, "save_factor_library") as mock_save:
            results = runner.mine_factors(iterations=1)

        # Only 1 of 2 IC-passing factors also passed backtest
        assert len(results) == 1, f"Expected 1 double-gated factor, got {len(results)}"
        mock_save.assert_called_once()
        saved_factors = mock_save.call_args[0][0]
        assert len(saved_factors) == 1
    finally:
        rdmod._KB_PATH = original_kb_path
        rdmod._KB_DIR = original_kb_dir


def test_mine_factors_backtest_failed_logged(tmp_path):
    """IC-pass/bt-fail factors are logged to kb['backtest_failed']."""
    import src.core.rd_agent_runner as rdmod

    original_kb_path = rdmod._KB_PATH
    original_kb_dir = rdmod._KB_DIR

    try:
        runner = _make_runner_with_mocks(tmp_path, ic_pass_count=1, bt_pass_count=0)
        runner.mine_factors(iterations=1)

        kb = runner._load_kb()
        assert "backtest_failed" in kb, "kb should have 'backtest_failed' key"
        assert len(kb["backtest_failed"]) >= 1
        entry = kb["backtest_failed"][0]
        assert "factor" in entry
        assert "reason" in entry
    finally:
        rdmod._KB_PATH = original_kb_path
        rdmod._KB_DIR = original_kb_dir


def test_mine_factors_discoveries_include_sharpe(tmp_path):
    """kb['discoveries'] entries include 'sharpe' key for double-gated factors."""
    import src.core.rd_agent_runner as rdmod

    original_kb_path = rdmod._KB_PATH
    original_kb_dir = rdmod._KB_DIR

    try:
        runner = _make_runner_with_mocks(tmp_path, ic_pass_count=1, bt_pass_count=1)
        runner.mine_factors(iterations=1)

        kb = runner._load_kb()
        assert len(kb["discoveries"]) >= 1
        entry = kb["discoveries"][0]
        assert "sharpe" in entry, f"discovery entry missing 'sharpe' key: {entry}"
    finally:
        rdmod._KB_PATH = original_kb_path
        rdmod._KB_DIR = original_kb_dir


def test_load_kb_error_recovery_does_not_mutate_empty_kb(tmp_path):
    """_load_kb() error-recovery path must use deepcopy to avoid mutating _EMPTY_KB."""
    import src.core.rd_agent_runner as rdmod

    original_kb_path = rdmod._KB_PATH
    original_kb_dir = rdmod._KB_DIR
    rdmod._KB_PATH = tmp_path / "kb.json"
    rdmod._KB_DIR = tmp_path

    try:
        # Write invalid JSON to trigger the error-recovery path
        (tmp_path / "kb.json").write_text("{invalid json}")
        from src.core.rd_agent_runner import RDAgentRunner

        runner = RDAgentRunner()
        kb = runner._load_kb()
        # Mutate the returned kb
        kb["backtest_failed"].append({"test": "mutation"})
        # _EMPTY_KB sentinel must NOT be mutated
        assert rdmod._EMPTY_KB.get("backtest_failed") == [], \
            "_EMPTY_KB['backtest_failed'] was mutated — use copy.deepcopy(), not dict()"
    finally:
        rdmod._KB_PATH = original_kb_path
        rdmod._KB_DIR = original_kb_dir
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py::test_mine_factors_backtest_gate tests/test_factor_backtester.py::test_mine_factors_backtest_failed_logged tests/test_factor_backtester.py::test_mine_factors_discoveries_include_sharpe tests/test_factor_backtester.py::test_load_kb_error_recovery_does_not_mutate_empty_kb -v --tb=short
```

Expected: `FAILED` — `ImportError` or `AttributeError` since `FactorBacktester` is not yet imported in `rd_agent_runner.py`.

- [ ] **Step 3: Add import to `src/core/rd_agent_runner.py`**

Find the existing imports block (around line 23–28). Add after `from src.core.research_analyst import ResearchAnalyst`:

```python
from src.core.factor_backtester import FactorBacktester
```

- [ ] **Step 4: Add `"backtest_failed": []` to `_EMPTY_KB` in `src/core/rd_agent_runner.py`**

Find `_EMPTY_KB` (lines 44–54). Add the new key:

```python
_EMPTY_KB: dict[str, Any] = {
    "tested_factors": [],
    "failed_factors": [],
    "tested_configs": [],
    "discoveries": [],
    "backtest_failed": [],          # NEW — IC-pass/bt-fail factors for analyst review
    "last_run_date": "never",
    "tested_strategies": [],
    "discovered_strategies": [],
    "tested_regime_configs": [],
    "discovered_regime_improvements": [],
}
```

Also verify that `_load_kb()`'s error recovery path (line 170) uses `copy.deepcopy(_EMPTY_KB)` — not `dict(_EMPTY_KB)`. If it says `dict(_EMPTY_KB)`, change it to `copy.deepcopy(_EMPTY_KB)`:

```python
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load KB ({}); starting fresh.", exc)
            return copy.deepcopy(_EMPTY_KB)   # was dict(_EMPTY_KB) — must be deepcopy
```

- [ ] **Step 5: Add `self._backtester` to `__init__` in `src/core/rd_agent_runner.py`**

Find line 146 (`self._analyst = ResearchAnalyst(self.config)`). Add after it:

```python
        self._analyst = ResearchAnalyst(self.config)
        self._backtester = FactorBacktester(self.config)
```

- [ ] **Step 6: Replace the `mine_factors()` inner loop in `src/core/rd_agent_runner.py`**

Find the block starting at line 455:
```python
            batch_accepted: list[FactorDefinition] = []
            for er, factor in zip(eval_results, proposals):
                tested_names.append(er.factor_name)
                if er.passed:
                    factor = FactorDefinition(
```

Also add the import for `BacktestValidationResult` at the top of the function's local scope — actually add it to the module import block. Find:
```python
from src.utils.schemas import FactorDefinition, Strategy
```
Change to:
```python
from src.utils.schemas import BacktestValidationResult, FactorDefinition, Strategy
```

Now replace the entire block from `batch_accepted: list[FactorDefinition] = []` through the closing of `for er in eval_results:` (lines 455–485) with:

```python
            # Separate IC-passing from IC-failing
            ic_passed_pairs = [(er, f) for er, f in zip(eval_results, proposals) if er.passed]
            ic_failed = [er for er in eval_results if not er.passed]

            # Log IC-failed to tested list (no backtest needed)
            for er in ic_failed:
                tested_names.append(er.factor_name)

            # Stage 3: backtest gate — IC-passing factors only
            bt_results = (
                self._backtester.validate_batch([f for _, f in ic_passed_pairs])
                if ic_passed_pairs else []
            )

            batch_accepted: list[FactorDefinition] = []
            bt_results_by_name: dict[str, BacktestValidationResult] = {}
            for (er, factor), bt_result in zip(ic_passed_pairs, bt_results):
                tested_names.append(er.factor_name)
                bt_results_by_name[factor.name] = bt_result
                if bt_result.passed:
                    # Enrich with IC data + backtest data in one model_copy call.
                    # proposals[i] has ic_mean=0.0/icir=0.0 defaults.
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

            # Preserve existing eval_results tracking (all factors, IC-pass and IC-fail).
            # Use "factor_name" key to match existing KB schema.
            kb.setdefault("eval_results", [])
            for er in eval_results:
                kb["eval_results"].append({
                    "factor_name": er.factor_name,
                    "stage1_ic": er.stage1_ic,
                    "stage2_ic": er.stage2_ic,
                    "stage2_icir": er.stage2_icir,
                    "passed": er.passed,
                    "date": str(date.today()),
                    "reason": er.reason,
                })
```

Also update the `write_memo` call that follows (currently `memo_text = self._analyst.write_memo(eval_results, kb)`):

```python
            memo_text = self._analyst.write_memo(eval_results, kb, bt_results=bt_results_by_name)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py::test_mine_factors_backtest_gate tests/test_factor_backtester.py::test_mine_factors_backtest_failed_logged tests/test_factor_backtester.py::test_mine_factors_discoveries_include_sharpe tests/test_factor_backtester.py::test_load_kb_error_recovery_does_not_mutate_empty_kb -v --tb=short
```

Expected: All 3 `PASSED`.

- [ ] **Step 8: Run full existing test suite to verify no regressions**

```bash
conda run -n aiquant python -m pytest tests/ -v --tb=short -q -m "not integration" 2>&1 | tail -20
```

Expected: All tests pass (same count as before plus 3 new ones). Zero failures.

- [ ] **Step 9: Commit**

```bash
git add src/core/rd_agent_runner.py src/core/factor_backtester.py tests/test_factor_backtester.py
git commit -m "feat: wire FactorBacktester into RDAgentRunner mine_factors() loop"
```

---

### Task 4: Update `ResearchAnalyst`

**Files:**
- Modify: `src/core/research_analyst.py` (4 changes)
- Modify: `tests/test_factor_backtester.py` (add test 14)

- [ ] **Step 1: Add test 14 to the test file**

Append to `tests/test_factor_backtester.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: ResearchAnalyst format test
# ---------------------------------------------------------------------------

def test_research_analyst_memo_format_with_backtest():
    """_format_results_table() includes bt_sharpe column when bt_results dict provided."""
    from src.core.research_analyst import ResearchAnalyst
    from src.utils.schemas import EvalResult, BacktestValidationResult

    results = [
        EvalResult(
            factor_name="good_factor",
            stage1_ic=0.05,
            stage1_passed=True,
            stage2_ic=0.04,
            stage2_icir=0.6,
            passed=True,
            reason="passed",
        ),
        EvalResult(
            factor_name="bad_factor",
            stage1_ic=0.001,
            stage1_passed=False,
            passed=False,
            reason="low_ic",
        ),
    ]
    bt_results = {
        "good_factor": BacktestValidationResult(
            factor_name="good_factor",
            passed=True,
            sharpe=1.23,
            reason="passed",
        )
    }

    table = ResearchAnalyst._format_results_table(results, bt_results=bt_results)

    # Column header
    assert "bt_sharpe" in table, f"Expected 'bt_sharpe' column in table:\n{table}"
    # Good factor has sharpe value
    assert "1.2300" in table, f"Expected Sharpe '1.2300' in table:\n{table}"
    # Bad factor (IC-failed, not in bt_results) has em-dash
    assert "—" in table, f"Expected '—' for IC-failed factor:\n{table}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py::test_research_analyst_memo_format_with_backtest -v --tb=short
```

Expected: `FAILED` — `TypeError: _format_results_table() got an unexpected keyword argument 'bt_results'`.

- [ ] **Step 3: Update `_format_results_table()` in `src/core/research_analyst.py`**

Find `_format_results_table` (line 151). Replace the entire method:

```python
    @staticmethod
    def _format_results_table(
        results: list[EvalResult],
        bt_results: dict | None = None,  # dict[str, BacktestValidationResult]
    ) -> str:
        """Format a list of EvalResult objects as a plain-text table.

        Args:
            results: EvalResult objects from the latest batch.
            bt_results: Optional dict mapping factor_name → BacktestValidationResult.
                        When provided, a bt_sharpe column is added.
                        IC-failed factors (absent from bt_results) show "—".
        """
        has_bt = bt_results is not None
        if has_bt:
            lines = ["name | stage1_IC | stage2_IC | ICIR | bt_sharpe | passed"]
        else:
            lines = ["name | stage1_IC | stage2_IC | ICIR | passed"]
        lines.append("-" * 70)
        for r in results:
            s2_ic = f"{r.stage2_ic:.4f}" if r.stage2_ic is not None else "—"
            s2_icir = f"{r.stage2_icir:.4f}" if r.stage2_icir is not None else "—"
            if has_bt:
                bt = bt_results.get(r.factor_name)  # type: ignore[union-attr]
                bt_sharpe_str = f"{bt.sharpe:.4f}" if (bt and bt.sharpe is not None) else "—"
                lines.append(
                    f"{r.factor_name} | {r.stage1_ic:.4f} | {s2_ic} | {s2_icir}"
                    f" | {bt_sharpe_str} | {r.passed}"
                )
            else:
                lines.append(
                    f"{r.factor_name} | {r.stage1_ic:.4f} | {s2_ic} | {s2_icir} | {r.passed}"
                )
        return "\n".join(lines)
```

- [ ] **Step 4: Update `write_memo()` signature in `src/core/research_analyst.py`**

Find `def write_memo(self, batch_results: list[EvalResult], kb: dict) -> str:` (line 54).
Change to:

```python
    def write_memo(
        self,
        batch_results: list[EvalResult],
        kb: dict,
        bt_results: dict | None = None,  # dict[str, BacktestValidationResult]
    ) -> str:
        """Synthesize batch evaluation results into a research memo.

        Args:
            batch_results: EvalResult objects from the latest evaluation batch.
            kb: Current knowledge base dict (read-only in this method).
            bt_results: Optional dict of backtest results keyed by factor name.

        Returns:
            200–400 word research memo string.
        """
        if self._llm_available:
            try:
                return self._llm_write_memo(batch_results, kb, bt_results=bt_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ResearchAnalyst.write_memo LLM failed ({}). Using fallback.", exc)
        return self._fallback_memo(batch_results)  # fallback intentionally omits bt_sharpe
```

- [ ] **Step 5: Update `_llm_write_memo()` signature in `src/core/research_analyst.py`**

Find `def _llm_write_memo(self, batch_results: list[EvalResult], kb: dict) -> str:` (line 100).
Change signature and the `results_table` line:

```python
    def _llm_write_memo(
        self,
        batch_results: list[EvalResult],
        kb: dict,
        bt_results: dict | None = None,
    ) -> str:
        """Call Opus to generate a research memo from batch results."""
        prior_memo = self.load_memo(kb)
        results_table = self._format_results_table(batch_results, bt_results=bt_results)
        kb_stats = self._format_kb_stats(kb)
        # ... rest of method unchanged
```

The remaining body of `_llm_write_memo()` does not change — only the signature and the `results_table =` line.

- [ ] **Step 6: Run test to verify it passes**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py::test_research_analyst_memo_format_with_backtest -v --tb=short
```

Expected: `PASSED`.

- [ ] **Step 7: Run full test suite to verify no regressions**

```bash
conda run -n aiquant python -m pytest tests/ -v --tb=short -q -m "not integration" 2>&1 | tail -20
```

Expected: All tests pass. Zero failures.

- [ ] **Step 8: Commit**

```bash
git add src/core/research_analyst.py tests/test_factor_backtester.py
git commit -m "feat: add bt_results param to ResearchAnalyst._format_results_table() and write_memo()"
```

---

## Final Verification

- [ ] **Run all 15 new tests together**

```bash
conda run -n aiquant python -m pytest tests/test_factor_backtester.py -v --tb=short
```

Expected: All 15 tests `PASSED`.

- [ ] **Run full non-integration suite**

```bash
conda run -n aiquant python -m pytest tests/ -v --tb=short -q -m "not integration" 2>&1 | tail -10
```

Expected: Zero failures.

- [ ] **Lint check**

```bash
conda run -n aiquant ruff check src/core/factor_backtester.py src/utils/schemas.py src/core/rd_agent_runner.py src/core/research_analyst.py
```

Expected: No errors.
