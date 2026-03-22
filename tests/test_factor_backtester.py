"""Tests for FactorBacktester and related schemas/config."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

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
        BacktestMetrics,
        BacktestResult,
        ValidationCheck,
        ValidationResult,
        ValidationVerdict,
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
    from src.utils.schemas import (
        BacktestMetrics,
        BacktestResult,
        FactorDefinition,
        ValidationCheck,
        ValidationResult,
        ValidationVerdict,
    )

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
    from src.utils.schemas import BacktestValidationResult, EvalResult, FactorDefinition

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

    with patch("src.core.rd_agent_runner.FactorProposer"), \
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


# ---------------------------------------------------------------------------
# Task 4: ResearchAnalyst format test
# ---------------------------------------------------------------------------

def test_research_analyst_memo_format_with_backtest():
    """_format_results_table() includes bt_sharpe column when bt_results dict provided."""
    from src.core.research_analyst import ResearchAnalyst
    from src.utils.schemas import BacktestValidationResult, EvalResult

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
