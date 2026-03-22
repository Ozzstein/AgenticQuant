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
