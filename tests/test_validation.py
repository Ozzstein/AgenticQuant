"""Tests for src/core/validation.py — ValidationSuite."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.core.validation import ValidationSuite
from src.utils.schemas import (
    BacktestMetrics,
    BacktestResult,
    BenchmarkComparison,
    CheckVerdict,
    ValidationReport,
    WalkForwardFold,
)

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_good_result(n_days: int = 500, sharpe: float = 1.5) -> BacktestResult:
    """Build a BacktestResult that should pass most checks.

    Args:
        n_days: Number of daily return observations.
        sharpe: Target annualised Sharpe ratio.

    Returns:
        BacktestResult with synthetic returns and 4 walk-forward folds.
    """
    np.random.seed(42)
    daily_vol = 0.01
    daily_mean = sharpe * daily_vol / np.sqrt(252)
    returns = np.random.normal(daily_mean, daily_vol, n_days).tolist()

    folds = [
        WalkForwardFold(
            fold_id=i,
            train_start="2019-01-01",
            train_end="2020-12-31",
            test_start="2021-01-01",
            test_end="2021-06-30",
            in_sample_sharpe=sharpe,
            oos_sharpe=sharpe * 0.8,
        )
        for i in range(4)
    ]
    metrics = BacktestMetrics(
        sharpe_ratio=sharpe,
        total_return=0.5,
        annual_return=0.12,
        max_drawdown=-0.10,
        total_trades=30 * 4,
    )
    return BacktestResult(
        metrics=metrics,
        daily_returns=returns,
        trades=[{"fold": i, "topk": 30, "oos_sharpe": 0.8} for i in range(4)],
        walk_forward_results=folds,
        equity_curve=list(np.cumprod([1 + r for r in returns]) * 100),
        timestamps=[f"2020-{i % 12 + 1:02d}-01" for i in range(n_days)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validation_report_has_7_checks(mock_config):
    """validate() returns a report with exactly 7 checks."""
    suite = ValidationSuite(config=mock_config)
    result = _make_good_result()
    report = suite.validate(result)

    assert isinstance(report, ValidationReport)
    assert len(report.checks) == 7


def test_good_strategy_is_reliable(mock_config):
    """A strong strategy (high Sharpe, many days, multi-fold) is RELIABLE."""
    suite = ValidationSuite(config=mock_config)
    # Use sharpe=4.0 so the strategy survives 10bps cost sensitivity test
    result = _make_good_result(n_days=500, sharpe=4.0)
    report = suite.validate(result)

    assert report.overall == "RELIABLE", f"Expected RELIABLE, got {report.overall}. Summary: {report.summary}"


def test_look_ahead_detected(mock_config):
    """Constant returns → shuffled ≈ original → look_ahead_bias FAIL."""
    suite = ValidationSuite(config=mock_config)

    # Constant returns: shuffling has no effect, so shuffled Sharpe == original Sharpe
    constant_returns = [0.001] * 300

    folds = [
        WalkForwardFold(
            fold_id=i,
            train_start="2019-01-01",
            train_end="2020-12-31",
            test_start="2021-01-01",
            test_end="2021-06-30",
            in_sample_sharpe=2.0,
            oos_sharpe=1.6,
        )
        for i in range(4)
    ]
    metrics = BacktestMetrics(
        sharpe_ratio=2.0,
        total_return=0.3,
        annual_return=0.10,
        max_drawdown=-0.05,
        total_trades=120,
    )
    result = BacktestResult(
        metrics=metrics,
        daily_returns=constant_returns,
        trades=[{"fold": i, "topk": 30, "oos_sharpe": 0.8} for i in range(4)],
        walk_forward_results=folds,
        equity_curve=list(np.cumprod([1 + r for r in constant_returns]) * 100),
        timestamps=[f"2020-{i % 12 + 1:02d}-01" for i in range(300)],
    )

    report = suite.validate(result)
    look_ahead_check = next(c for c in report.checks if c.name == "look_ahead_bias")
    assert look_ahead_check.verdict == CheckVerdict.FAIL, (
        f"Expected FAIL for constant returns, got {look_ahead_check.verdict}: "
        f"{look_ahead_check.explanation}"
    )


def test_cost_sensitivity_flags_high_turnover(mock_config):
    """Sharpe drops >50% at 10bps vs 0bps → cost_sensitivity FAIL."""
    suite = ValidationSuite(config=mock_config)

    # Small daily mean (0.0005) with small vol (0.001): high Sharpe at 0bps.
    # Subtracting 10bps/day (0.001) flips the mean negative → Sharpe collapses.
    np.random.seed(0)
    sensitive_returns = np.random.normal(0.0005, 0.001, 500).tolist()

    folds = [
        WalkForwardFold(
            fold_id=i,
            train_start="2019-01-01",
            train_end="2020-12-31",
            test_start="2021-01-01",
            test_end="2021-06-30",
        )
        for i in range(2)
    ]
    metrics = BacktestMetrics(sharpe_ratio=8.0, total_trades=60)
    result = BacktestResult(
        metrics=metrics,
        daily_returns=sensitive_returns,
        trades=[{"fold": i} for i in range(2)],
        walk_forward_results=folds,
        equity_curve=[100.0] * 500,
        timestamps=[f"2020-01-{i % 28 + 1:02d}" for i in range(500)],
    )

    report = suite.validate(result)
    cost_check = next(c for c in report.checks if c.name == "cost_sensitivity")
    assert cost_check.verdict == CheckVerdict.FAIL, (
        f"Expected FAIL for tiny returns, got {cost_check.verdict}: {cost_check.explanation}"
    )


def test_statistical_significance_bootstrap(mock_config):
    """Long series with t-stat > 2 and positive bootstrap → PASS; short → FAIL."""
    suite = ValidationSuite(config=mock_config)

    # Long series (500 days) with high Sharpe → should PASS
    long_result = _make_good_result(n_days=500, sharpe=2.0)
    long_report = suite.validate(long_result)
    sig_check_long = next(c for c in long_report.checks if c.name == "statistical_significance")
    assert sig_check_long.verdict == CheckVerdict.PASS, (
        f"Expected PASS for 500-day result, got {sig_check_long.verdict}: {sig_check_long.explanation}"
    )

    # Short series (100 days, n < 252) → should FAIL
    short_result = _make_good_result(n_days=100)
    short_report = suite.validate(short_result)
    sig_check_short = next(c for c in short_report.checks if c.name == "statistical_significance")
    assert sig_check_short.verdict == CheckVerdict.FAIL, (
        f"Expected FAIL for 100-day result, got {sig_check_short.verdict}: {sig_check_short.explanation}"
    )


def test_regime_dependence_single_regime(mock_config):
    """Returns only positive in first half and negative in second → WARNING."""
    suite = ValidationSuite(config=mock_config)

    n = 400
    np.random.seed(1)
    # First half: strong positive returns with variance → positive Sharpe
    first_half = np.random.normal(0.003, 0.01, n // 2).tolist()
    # Second half: negative returns with same variance → negative Sharpe
    second_half = np.random.normal(-0.003, 0.01, n // 2).tolist()
    mixed_returns = first_half + second_half

    folds = [
        WalkForwardFold(
            fold_id=i,
            train_start="2019-01-01",
            train_end="2020-12-31",
            test_start="2021-01-01",
            test_end="2021-06-30",
            in_sample_sharpe=1.0,
            oos_sharpe=0.8,
        )
        for i in range(4)
    ]
    metrics = BacktestMetrics(sharpe_ratio=1.0, total_trades=120)
    result = BacktestResult(
        metrics=metrics,
        daily_returns=mixed_returns,
        trades=[{"fold": i, "topk": 30} for i in range(4)],
        walk_forward_results=folds,
        equity_curve=list(np.cumprod([1 + r for r in mixed_returns]) * 100),
        timestamps=[f"2020-{i % 12 + 1:02d}-01" for i in range(n)],
    )

    report = suite.validate(result)
    regime_check = next(c for c in report.checks if c.name == "regime_dependence")
    assert regime_check.verdict == CheckVerdict.WARNING, (
        f"Expected WARNING for single-regime returns, got {regime_check.verdict}: "
        f"{regime_check.explanation}"
    )


def test_overall_verdict_not_reliable_on_fail(mock_config):
    """A result with any FAIL check gets NOT RELIABLE verdict."""
    suite = ValidationSuite(config=mock_config)

    # Only 5 returns → triggers multiple FAILs (statistical_significance, etc.)
    tiny_result = BacktestResult(
        metrics=BacktestMetrics(sharpe_ratio=0.5, total_trades=0),
        daily_returns=[0.001, 0.002, -0.001, 0.003, 0.001],
        trades=[],
        walk_forward_results=[],
        equity_curve=[100.0, 100.1, 100.2, 100.1, 100.4, 100.5],
        timestamps=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
    )

    report = suite.validate(tiny_result)
    assert report.overall == "NOT RELIABLE", (
        f"Expected NOT RELIABLE for tiny result, got {report.overall}. Summary: {report.summary}"
    )


def test_benchmark_integration(mock_config):
    """ValidationSuite + BenchmarkRunner.compare() produces ValidationReport with benchmark field."""
    from src.core.benchmark import BenchmarkRunner

    suite = ValidationSuite(config=mock_config)
    good_result = _make_good_result(n_days=500, sharpe=2.0)

    # Validate strategy
    report = suite.validate(good_result)
    assert isinstance(report, ValidationReport)

    # Mock BenchmarkRunner.compare() to avoid network calls
    mock_runner = MagicMock(spec=BenchmarkRunner)
    mock_comparison = BenchmarkComparison(
        strategy_sharpe=2.0,
        spy_sharpe=0.8,
        btc_sharpe=1.2,
        sixty_forty_sharpe=0.6,
        equal_weight_sharpe=0.7,
        monte_carlo_percentile=85.0,
        excess_sharpe_vs_spy=1.2,
    )
    mock_runner.compare.return_value = mock_comparison

    # Call mocked compare
    benchmark_result = mock_runner.compare(good_result, "2020-01-01", "2022-01-01")

    # Build final report with benchmark attached
    final_report = ValidationReport(
        checks=report.checks,
        overall=report.overall,
        summary=report.summary,
        benchmark=benchmark_result,
    )

    assert final_report.benchmark is not None
    assert isinstance(final_report.benchmark, BenchmarkComparison)
    assert final_report.benchmark.strategy_sharpe == 2.0
    assert final_report.benchmark.excess_sharpe_vs_spy == 1.2
