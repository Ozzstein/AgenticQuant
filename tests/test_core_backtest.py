"""Tests for WalkForwardBacktester and BacktestValidator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.core.backtester import BacktestValidator, WalkForwardBacktester
from src.utils.schemas import (
    BacktestMetrics,
    BacktestResult,
    ValidationVerdict,
    WalkForwardFold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(seed: int = 0) -> MagicMock:
    """Return a mock model that predicts random scores."""
    rng = np.random.default_rng(seed)

    model = MagicMock()
    model.train.return_value = None

    def predict(X):
        return rng.standard_normal(len(X))

    model.predict.side_effect = predict
    return model


def _make_features_labels(
    n_days: int = 300,
    n_tickers: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic features and forward-return labels."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    # Flat index (one row per day, aggregated across tickers for simplicity)
    feats = pd.DataFrame(rng.standard_normal((n_days, 5)), index=dates,
                         columns=[f"f{i}" for i in range(5)])
    labels = pd.Series(rng.normal(0.001, 0.02, n_days), index=dates, name="fwd_ret")
    return feats, labels


def make_synthetic_result(n_days: int = 252, sharpe: float = 1.5, seed: int = 0) -> BacktestResult:
    """Build a synthetic BacktestResult for validator tests."""
    rng = np.random.default_rng(seed)
    daily_mean = sharpe * 0.16 / 252  # approximate: Sharpe * vol / days
    daily_std = 0.16 / np.sqrt(252)
    daily_returns = rng.normal(daily_mean, daily_std, n_days).tolist()

    equity = (pd.Series([1.0] + daily_returns).cumprod() * 100).tolist()[1:]
    timestamps = [str(pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)) for i in range(n_days)]

    fold = WalkForwardFold(
        fold_id=0,
        train_start="2021-01-01",
        train_end="2021-06-30",
        test_start="2021-07-06",
        test_end="2021-09-30",
        in_sample_sharpe=sharpe * 0.9,
        oos_sharpe=sharpe * 0.8,
    )

    metrics = BacktestMetrics(
        total_return=float(np.prod(1 + np.array(daily_returns)) - 1),
        annual_return=sharpe * 0.15,
        sharpe_ratio=sharpe,
        max_drawdown=-0.05,
        calmar_ratio=sharpe * 0.15 / 0.05,
        win_rate=0.55,
        total_trades=30,
        avg_trade_return=daily_mean,
        volatility=0.15,
        sortino_ratio=sharpe * 1.2,
    )

    return BacktestResult(
        metrics=metrics,
        daily_returns=daily_returns,
        trades=[{"fold": 0, "test_start": "2021-07-06", "test_end": "2021-09-30", "topk": 30,
                 "oos_sharpe": sharpe * 0.8}],
        walk_forward_results=[fold],
        equity_curve=equity,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# WalkForwardBacktester tests
# ---------------------------------------------------------------------------


class TestWalkForwardBacktester:
    def test_walk_forward_backtester_runs(self, mock_config):
        """Backtester returns a BacktestResult without errors."""
        feats, labels = _make_features_labels(n_days=400)
        model = _make_model()
        bt = WalkForwardBacktester(config=mock_config)
        result = bt.run(feats, labels, model, topk=5, walk_forward_months=3, embargo_days=5)
        assert isinstance(result, BacktestResult)
        assert len(result.daily_returns) > 0

    def test_walk_forward_produces_folds(self, mock_config):
        """Walk-forward produces multiple folds."""
        feats, labels = _make_features_labels(n_days=400)
        model = _make_model()
        bt = WalkForwardBacktester(config=mock_config)
        result = bt.run(feats, labels, model, topk=5, walk_forward_months=3, embargo_days=5)
        assert len(result.walk_forward_results) >= 1, "Expected at least 1 fold"

    def test_walk_forward_folds_count(self, mock_config):
        """More months in dataset → more folds."""
        feats, labels = _make_features_labels(n_days=600)
        model = _make_model()
        bt = WalkForwardBacktester(config=mock_config)
        result = bt.run(feats, labels, model, topk=5, walk_forward_months=3, embargo_days=5)
        assert len(result.walk_forward_results) >= 2

    def test_metrics_sharpe_positive(self, mock_config):
        """Monotonically positive returns yield positive Sharpe."""
        bt = WalkForwardBacktester(config=mock_config)
        positive_returns = pd.Series([0.001] * 300)
        metrics = bt._compute_metrics(positive_returns)
        assert metrics.sharpe_ratio > 0

    def test_metrics_max_drawdown_non_positive(self, mock_config):
        """Max drawdown is always ≤ 0."""
        bt = WalkForwardBacktester(config=mock_config)
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        metrics = bt._compute_metrics(returns)
        assert metrics.max_drawdown <= 0

    def test_metrics_win_rate_range(self, mock_config):
        """Win rate is in [0, 1]."""
        bt = WalkForwardBacktester(config=mock_config)
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        metrics = bt._compute_metrics(returns)
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_equity_curve_starts_at_100(self, mock_config):
        """Equity curve starts at ~100."""
        feats, labels = _make_features_labels(n_days=400)
        model = _make_model()
        bt = WalkForwardBacktester(config=mock_config)
        result = bt.run(feats, labels, model, topk=5, walk_forward_months=3, embargo_days=5)
        assert len(result.equity_curve) > 0
        # First value should be close to 100 (within 10 bps * 30 topk rounding)
        assert 90 < result.equity_curve[0] < 110

    def test_empty_features_raises(self, mock_config):
        """Empty inputs raise BacktestError."""
        from src.utils.exceptions import BacktestError

        bt = WalkForwardBacktester(config=mock_config)
        with pytest.raises(BacktestError):
            bt.run(pd.DataFrame(), pd.Series(dtype=float), _make_model(), topk=5)

    def test_split_folds_returns_dicts(self, mock_config):
        """_split_folds returns list of dicts with required keys."""
        bt = WalkForwardBacktester(config=mock_config)
        dates = pd.bdate_range("2020-01-01", periods=300)
        folds = bt._split_folds(dates, "2020-01-01", 3, 5)
        assert isinstance(folds, list)
        for fold in folds:
            for key in ("train_start", "train_end", "test_start", "test_end"):
                assert key in fold

    def test_split_folds_embargo_enforced(self, mock_config):
        """Train end is before test start in each fold."""
        bt = WalkForwardBacktester(config=mock_config)
        dates = pd.bdate_range("2020-01-01", periods=400)
        folds = bt._split_folds(dates, "2020-01-01", 3, 5)
        for fold in folds:
            assert fold["train_end"] < fold["test_start"]


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestHtmlReport:
    def test_generate_html_report(self, mock_config, tmp_path):
        """HTML report is generated and file exists."""
        result = make_synthetic_result(n_days=252)
        bt = WalkForwardBacktester(config=mock_config)
        out = tmp_path / "report.html"
        path = bt.generate_html_report(result, str(out))
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "plotly" in content.lower()

    def test_generate_html_report_creates_dir(self, mock_config, tmp_path):
        """HTML report creates missing parent directories."""
        result = make_synthetic_result(n_days=252)
        bt = WalkForwardBacktester(config=mock_config)
        out = tmp_path / "nested" / "dir" / "report.html"
        path = bt.generate_html_report(result, str(out))
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# BacktestValidator tests
# ---------------------------------------------------------------------------


class TestBacktestValidator:
    def test_all_8_checks_present(self):
        """validate() always returns exactly 8 checks."""
        result = make_synthetic_result(n_days=252)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert len(vr.checks) == 8

    def test_validator_approved_good_strategy(self):
        """Good strategy (Sharpe=1.5, 252 days) → APPROVED or CAUTION."""
        np.random.seed(42)
        result = make_synthetic_result(n_days=252, sharpe=1.5, seed=42)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert vr.verdict in (ValidationVerdict.APPROVED, ValidationVerdict.CAUTION)

    def test_validator_rejected_overfitting(self):
        """Sharpe > 3.0 triggers overfitting flag → REJECTED."""
        result = make_synthetic_result(n_days=252, sharpe=5.0)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert vr.verdict == ValidationVerdict.REJECTED

    def test_validator_low_significance(self):
        """Only 50 days → fails statistical significance → REJECTED."""
        result = make_synthetic_result(n_days=50, sharpe=1.5)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert vr.verdict == ValidationVerdict.REJECTED

    def test_check_names_are_unique(self):
        """Each check has a unique name."""
        result = make_synthetic_result(n_days=252)
        validator = BacktestValidator()
        vr = validator.validate(result)
        names = [c.name for c in vr.checks]
        assert len(names) == len(set(names))

    def test_verdict_summary_present(self):
        """ValidationResult.summary is a non-empty string."""
        result = make_synthetic_result(n_days=252)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert isinstance(vr.summary, str)
        assert len(vr.summary) > 0

    def test_critical_check_failure_rejected(self):
        """Any failing critical check → REJECTED verdict."""
        # 10 days only → statistical significance fails
        result = make_synthetic_result(n_days=10, sharpe=1.5)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert vr.verdict == ValidationVerdict.REJECTED

    def test_passed_count_property(self):
        """passed_count + failed_count == 8."""
        result = make_synthetic_result(n_days=252)
        validator = BacktestValidator()
        vr = validator.validate(result)
        assert vr.passed_count + vr.failed_count == 8

    def test_look_ahead_check_passes_with_embargo(self):
        """Look-ahead check passes when train_end < test_start."""
        result = make_synthetic_result(n_days=252)
        validator = BacktestValidator()
        check = validator._check_look_ahead_bias(result)
        assert check.passed  # synthetic result has correct dates

    def test_concentration_check_with_topk_30(self):
        """Equal-weight with 30 stocks → max weight ≤ 30%."""
        result = make_synthetic_result(n_days=252)
        validator = BacktestValidator()
        check = validator._check_concentration_risk(result)
        assert check.passed  # 1/30 ≈ 3.3% < 30%
