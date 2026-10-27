"""Tests for FactorBacktester and related schemas/config."""
from __future__ import annotations

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





# ---------------------------------------------------------------------------
# Task 2: FactorBacktester tests
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# Task 3: RDAgentRunner wiring tests
# ---------------------------------------------------------------------------











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


def test_research_analyst_memo_format_without_backtest():
    """_format_results_table() omits bt_sharpe column when bt_results not provided."""
    from src.core.research_analyst import ResearchAnalyst
    from src.utils.schemas import EvalResult

    results = [
        EvalResult(
            factor_name="factor_a",
            stage1_ic=0.03,
            stage1_passed=True,
            passed=True,
            reason="passed",
        ),
    ]

    table = ResearchAnalyst._format_results_table(results)

    assert "bt_sharpe" not in table, f"bt_sharpe column should be absent:\n{table}"
    # Standard columns must still be present
    assert "stage1_IC" in table
    assert "ICIR" in table
    assert "passed" in table

# Actual performance/reliability cases are covered by test_research_evaluation;
# the legacy mocked APPROVED/CAUTION shortcut no longer has approval authority.
def test_unsafe_expression_is_rejected_before_model_training(tmp_path):
    import pytest

    from src.core.factor_backtester import FactorBacktester
    from src.data.features.expressions import ExpressionError
    from src.utils.schemas import FactorDefinition
    from tests.test_research_inputs import research_inputs
    with pytest.raises(ExpressionError):
        FactorBacktester().run(FactorDefinition(name="future",expression="Ref($close,-1)"),
                               inputs=research_inputs(),ledger_path=tmp_path/"ledger.db",artifact_dir=tmp_path/"models")
    assert not (tmp_path/"ledger.db").exists()
