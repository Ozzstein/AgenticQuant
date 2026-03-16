"""Tests for WalkForwardRunner."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.walk_forward import WalkForwardRunner
from src.utils.schemas import BacktestMetrics, WalkForwardFold, WalkForwardResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features_labels(n_dates: int = 756, n_features: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic 3-year feature/label dataset.

    Args:
        n_dates: Number of business days (756 ≈ 3 years).
        n_features: Number of feature columns.

    Returns:
        Tuple of (features DataFrame, labels Series).
    """
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    features = pd.DataFrame(
        np.random.randn(n_dates, n_features),
        index=dates,
        columns=[f"f{i}" for i in range(n_features)],
    )
    labels = pd.Series(np.random.randn(n_dates), index=dates)
    return features, labels


def _mock_model() -> MagicMock:
    """Return a MagicMock that emulates ModelWrapper's interface."""
    mock = MagicMock()
    mock.train.return_value = mock
    mock.predict.side_effect = lambda X: np.random.randn(len(X))
    mock.get_feature_importance.return_value = pd.Series(
        np.abs(np.random.randn(5)), index=[f"f{i}" for i in range(5)]
    )
    return mock


def _run_with_mock(mock_config, features: pd.DataFrame, labels: pd.Series, **kwargs) -> WalkForwardResult:
    """Run WalkForwardRunner with a fully mocked ModelWrapper."""
    with patch("src.core.walk_forward.ModelWrapper") as MockModel:
        mock_instance = _mock_model()
        MockModel.return_value = mock_instance
        runner = WalkForwardRunner(config=mock_config)
        return runner.run(features, labels, model_name="LightGBM", **kwargs)


def _make_minimal_result(n_folds: int = 3) -> WalkForwardResult:
    """Build a minimal WalkForwardResult with fake data for report tests."""
    folds = [
        WalkForwardFold(
            fold_id=i,
            train_start="2020-01-01",
            train_end=f"2020-0{i + 3}-01",
            test_start=f"2020-0{i + 4}-01",
            test_end=f"2020-0{i + 5}-01",
            in_sample_sharpe=1.0,
            oos_sharpe=0.5 + i * 0.1,
            model_params={f"fi_f{j}": float(j + 1) for j in range(5)},
        )
        for i in range(n_folds)
    ]
    stitched_returns = [0.001, -0.002, 0.003, 0.001, -0.001, 0.002]
    stitched_timestamps = ["2020-04-01"] * 2 + ["2020-05-01"] * 2 + ["2020-06-01"] * 2
    metrics = BacktestMetrics(
        total_return=0.05,
        annual_return=0.10,
        sharpe_ratio=0.8,
        max_drawdown=-0.03,
        calmar_ratio=3.33,
        win_rate=0.55,
        total_trades=30,
        avg_trade_return=0.001,
        volatility=0.12,
        sortino_ratio=1.0,
    )
    return WalkForwardResult(
        folds=folds,
        stitched_returns=stitched_returns,
        stitched_timestamps=stitched_timestamps,
        aggregate_metrics=metrics,
        feature_importance_drift=[0.9, 0.85],
        rolling_ic=[0.05, 0.03, 0.04],
        model_name="LightGBM",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fold_splitting_produces_at_least_2_folds(mock_config):
    """3 years of data with walk_months=6 should produce ≥2 folds."""
    features, labels = _make_features_labels(n_dates=756)
    result = _run_with_mock(mock_config, features, labels, walk_months=6, embargo_days=5)
    assert len(result.folds) >= 2, f"Expected ≥2 folds, got {len(result.folds)}"


def test_embargo_gap_respected(mock_config):
    """test_start - train_end >= embargo_days for each fold."""
    features, labels = _make_features_labels(n_dates=756)
    embargo_days = 5
    result = _run_with_mock(
        mock_config, features, labels, walk_months=6, embargo_days=embargo_days
    )
    for fold in result.folds:
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        gap = (test_start - train_end).days
        assert gap >= embargo_days, (
            f"Fold {fold.fold_id}: gap={gap} days < embargo_days={embargo_days} "
            f"(train_end={fold.train_end}, test_start={fold.test_start})"
        )


def test_stitched_returns_not_empty(mock_config):
    """stitched_returns should be non-empty after a successful run."""
    features, labels = _make_features_labels(n_dates=756)
    result = _run_with_mock(mock_config, features, labels, walk_months=6, embargo_days=5)
    assert len(result.stitched_returns) > 0, "stitched_returns should not be empty"


def test_feature_importance_drift_length(mock_config):
    """feature_importance_drift should have len == n_folds - 1."""
    features, labels = _make_features_labels(n_dates=756)
    result = _run_with_mock(mock_config, features, labels, walk_months=6, embargo_days=5)
    n_folds = len(result.folds)
    expected_len = max(0, n_folds - 1)
    assert len(result.feature_importance_drift) == expected_len, (
        f"Expected feature_importance_drift length {expected_len}, "
        f"got {len(result.feature_importance_drift)}"
    )


def test_rolling_ic_values_are_finite(mock_config):
    """rolling_ic values must all be finite numbers."""
    features, labels = _make_features_labels(n_dates=756)
    result = _run_with_mock(mock_config, features, labels, walk_months=6, embargo_days=5)
    for i, ic_val in enumerate(result.rolling_ic):
        assert math.isfinite(ic_val), f"rolling_ic[{i}] = {ic_val} is not finite"


def test_generate_html_report_creates_file(mock_config, tmp_path):
    """generate_html_report should create a file at output_path."""
    result = _make_minimal_result(n_folds=3)
    runner = WalkForwardRunner(config=mock_config)
    report_path = str(tmp_path / "report.html")
    returned_path = runner.generate_html_report(result, report_path)
    assert Path(returned_path).exists(), f"Report file not found at {returned_path}"
    content = Path(returned_path).read_text()
    assert len(content) > 100, "Report HTML is suspiciously short"


def test_run_returns_walk_forward_result_type(mock_config):
    """run() should return a WalkForwardResult instance."""
    features, labels = _make_features_labels(n_dates=756)
    result = _run_with_mock(mock_config, features, labels, walk_months=6, embargo_days=5)
    assert isinstance(result, WalkForwardResult), (
        f"Expected WalkForwardResult, got {type(result)}"
    )
