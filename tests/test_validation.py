"""Validation cannot turn missing evidence or shuffled returns into approval."""

import copy

from src.core.backtester import BacktestValidator, PortfolioReplay
from src.core.validation import ValidationSuite
from src.utils.schemas import BacktestResult, CheckVerdict, ValidationVerdict
from tests.test_core_backtest import replay_inputs


def test_legacy_results_never_pass():
    report = ValidationSuite().validate(BacktestResult())
    assert report.overall == "NOT RELIABLE"
    assert report.checks[0].name == "engine_version"
    assert BacktestValidator().validate(BacktestResult()).verdict == ValidationVerdict.REJECTED


def test_missing_evidence_blocks_even_reconciled_nav(tmp_path):
    path = tmp_path / "ledger.db"
    result = PortfolioReplay(path).run(**replay_inputs())
    report = ValidationSuite().validate(result, ledger_path=path)
    verdict = {c.name: c.verdict for c in report.checks}
    assert verdict["nav_arithmetic"] == CheckVerdict.PASS
    assert verdict["durable_accounting"] == CheckVerdict.PASS
    assert verdict["fill_realism"] == CheckVerdict.PASS
    assert verdict["historical_provenance"] == CheckVerdict.FAIL
    assert verdict["purged_oos"] == CheckVerdict.FAIL
    assert verdict["statistical_significance"] == CheckVerdict.FAIL
    assert report.overall == "NOT RELIABLE"


def test_shuffling_returns_and_tampering_nav_fail_arithmetic(tmp_path):
    path = tmp_path / "ledger.db"
    result = PortfolioReplay(path).run(**replay_inputs())
    changed = result.model_copy(update={"daily_returns": list(reversed(result.daily_returns))})
    report = ValidationSuite().validate(changed, ledger_path=path)
    assert next(c for c in report.checks if c.name == "nav_arithmetic").verdict == CheckVerdict.FAIL
    changed = result.model_copy(update={"equity_curve": [10000] * len(result.equity_curve)})
    report = ValidationSuite().validate(changed, ledger_path=path)
    assert report.overall == "NOT RELIABLE"


def test_future_fill_and_no_ledger_are_failures(tmp_path):
    result = PortfolioReplay(tmp_path / "ledger.db").run(**replay_inputs())
    trades = copy.deepcopy(result.trades)
    trades[0]["executed_at"] = trades[0]["decision_at"]
    result = result.model_copy(update={"trades": trades})
    report = ValidationSuite().validate(result)
    verdict = {c.name: c.verdict for c in report.checks}
    assert verdict["fill_realism"] == CheckVerdict.FAIL
    assert verdict["durable_accounting"] == CheckVerdict.FAIL


def test_metric_manifest_and_full_fill_tampering_fail(tmp_path):
    path = tmp_path / "ledger.db"
    result = PortfolioReplay(path).run(**replay_inputs())
    for name in (
        "sharpe_ratio",
        "annual_return",
        "volatility",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "fees",
        "turnover",
        "total_trades",
    ):
        value = getattr(result.metrics, name)
        changed = result.model_copy(
            update={"metrics": result.metrics.model_copy(update={name: (value or 0) + 5})}
        )
        report = ValidationSuite().validate(changed, ledger_path=path)
        verdict = {c.name: c.verdict for c in report.checks}
        assert verdict["nav_arithmetic"] == CheckVerdict.FAIL, name
        assert verdict["durable_accounting"] == CheckVerdict.FAIL, name
    changed = copy.deepcopy(result.manifest)
    changed["context"]["code_version"] = "tampered"
    report = ValidationSuite().validate(
        result.model_copy(update={"manifest": changed}), ledger_path=path
    )
    assert (
        next(c for c in report.checks if c.name == "durable_accounting").verdict
        == CheckVerdict.FAIL
    )
    trades = copy.deepcopy(result.trades)
    trades[0]["price"] = "999"
    report = ValidationSuite().validate(
        result.model_copy(update={"trades": trades}), ledger_path=path
    )
    assert (
        next(c for c in report.checks if c.name == "durable_accounting").verdict
        == CheckVerdict.FAIL
    )
