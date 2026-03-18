"""Tests for PortfolioOptimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.execution.portfolio_optimizer import OptimizationResult, PortfolioOptimizer

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
N = len(TICKERS)


@pytest.fixture
def sample_cov() -> pd.DataFrame:
    """Simple 4x4 positive-semi-definite covariance matrix."""
    np.random.seed(42)
    A = np.random.randn(N, N)
    cov = A @ A.T / N  # PSD
    return pd.DataFrame(cov, index=TICKERS, columns=TICKERS)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Expected returns for 4 assets."""
    return pd.Series([0.12, 0.10, 0.08, 0.15], index=TICKERS)


@pytest.fixture
def optimizer() -> PortfolioOptimizer:
    """PortfolioOptimizer with default AppConfig (no external calls)."""
    from src.utils.config import AppConfig

    return PortfolioOptimizer(config=AppConfig())


# ------------------------------------------------------------------
# risk_parity tests
# ------------------------------------------------------------------


class TestRiskParity:
    def test_risk_parity_returns_result_type(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """risk_parity returns an OptimizationResult instance."""
        result = optimizer.risk_parity(sample_cov)
        assert isinstance(result, OptimizationResult)

    def test_risk_parity_weights_sum(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """risk_parity weights sum to ≤ 1.0 and > 0."""
        result = optimizer.risk_parity(sample_cov)
        total = sum(result.weights.values())
        assert 0.0 < total <= 1.0 + 1e-9

    def test_risk_parity_all_positive(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """ERC portfolio is always long-only; all weights must be ≥ 0."""
        result = optimizer.risk_parity(sample_cov)
        for ticker, w in result.weights.items():
            assert w >= 0.0, f"Negative weight for {ticker}: {w}"

    def test_risk_parity_within_constraints(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """Each weight is ≤ 0.10 after _apply_constraints."""
        result = optimizer.risk_parity(sample_cov)
        for ticker, w in result.weights.items():
            assert w <= 0.10 + 1e-9, f"Weight for {ticker} exceeds 10%: {w}"

    def test_risk_parity_risk_contributions(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """risk_contribution values sum to approximately 100%."""
        result = optimizer.risk_parity(sample_cov)
        total_rc = sum(result.risk_contribution.values())
        assert abs(total_rc - 100.0) < 1.0, f"Risk contributions sum to {total_rc}, expected ~100"

    def test_risk_parity_method_field(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """result.method is 'risk_parity'."""
        result = optimizer.risk_parity(sample_cov)
        assert result.method == "risk_parity"


# ------------------------------------------------------------------
# mean_variance tests
# ------------------------------------------------------------------


class TestMeanVariance:
    def test_mean_variance_weights_sum(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """mean_variance weights sum to ≤ 1.0."""
        result = optimizer.mean_variance(sample_returns, sample_cov)
        total = sum(result.weights.values())
        assert total <= 1.0 + 1e-9

    def test_mean_variance_non_negative(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """mean_variance weights are all ≥ 0 (long-only)."""
        result = optimizer.mean_variance(sample_returns, sample_cov)
        for ticker, w in result.weights.items():
            assert w >= 0.0, f"Negative weight for {ticker}: {w}"

    def test_mean_variance_returns_result_type(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """mean_variance returns an OptimizationResult."""
        result = optimizer.mean_variance(sample_returns, sample_cov)
        assert isinstance(result, OptimizationResult)

    def test_mean_variance_method_field(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """result.method is 'mean_variance'."""
        result = optimizer.mean_variance(sample_returns, sample_cov)
        assert result.method == "mean_variance"


# ------------------------------------------------------------------
# HRP tests
# ------------------------------------------------------------------


class TestHRP:
    def test_hrp_weights_sum(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """hrp weights sum to ≤ 1.0."""
        result = optimizer.hrp(sample_cov)
        total = sum(result.weights.values())
        assert total <= 1.0 + 1e-9

    def test_hrp_non_negative(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """hrp weights are all ≥ 0 (long-only)."""
        result = optimizer.hrp(sample_cov)
        for ticker, w in result.weights.items():
            assert w >= 0.0, f"Negative weight for {ticker}: {w}"

    def test_hrp_returns_result_type(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """hrp returns an OptimizationResult."""
        result = optimizer.hrp(sample_cov)
        assert isinstance(result, OptimizationResult)

    def test_hrp_method_field(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """result.method is 'hrp'."""
        result = optimizer.hrp(sample_cov)
        assert result.method == "hrp"


# ------------------------------------------------------------------
# Black-Litterman tests
# ------------------------------------------------------------------


class TestBlackLitterman:
    def test_black_litterman_runs(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """BL with views={"AAPL": 0.02} runs without error and returns OptimizationResult."""
        views = {"AAPL": 0.02}
        result = optimizer.black_litterman(sample_returns, sample_cov, views)
        assert isinstance(result, OptimizationResult)

    def test_black_litterman_method_field(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """result.method is 'black_litterman'."""
        views = {"AAPL": 0.02, "MSFT": -0.01}
        result = optimizer.black_litterman(sample_returns, sample_cov, views)
        assert result.method == "black_litterman"

    def test_black_litterman_weights_valid(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """BL weights are non-negative and sum to ≤ 1.0."""
        views = {"AAPL": 0.02, "MSFT": -0.01}
        result = optimizer.black_litterman(sample_returns, sample_cov, views)
        total = sum(result.weights.values())
        assert total <= 1.0 + 1e-9
        for ticker, w in result.weights.items():
            assert w >= 0.0, f"Negative weight for {ticker}: {w}"

    def test_black_litterman_empty_views(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns: pd.Series,
        sample_cov: pd.DataFrame,
    ) -> None:
        """BL with empty views still runs and returns valid result (falls back to prior)."""
        result = optimizer.black_litterman(sample_returns, sample_cov, views={})
        assert isinstance(result, OptimizationResult)
        assert result.method == "black_litterman"


# ------------------------------------------------------------------
# Max diversification tests
# ------------------------------------------------------------------


class TestMaxDiversification:
    def test_max_diversification_runs(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """max_diversification runs without error, weights sum ≤ 1.0."""
        result = optimizer.max_diversification(sample_cov)
        assert isinstance(result, OptimizationResult)
        total = sum(result.weights.values())
        assert total <= 1.0 + 1e-9

    def test_max_diversification_non_negative(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """max_diversification weights are all ≥ 0."""
        result = optimizer.max_diversification(sample_cov)
        for ticker, w in result.weights.items():
            assert w >= 0.0, f"Negative weight for {ticker}: {w}"

    def test_max_diversification_method_field(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """result.method is 'max_diversification'."""
        result = optimizer.max_diversification(sample_cov)
        assert result.method == "max_diversification"


# ------------------------------------------------------------------
# OptimizationResult method field (generic)
# ------------------------------------------------------------------


class TestOptimizationResultMethodField:
    def test_optimization_result_method_field(
        self, optimizer: PortfolioOptimizer, sample_cov: pd.DataFrame
    ) -> None:
        """Each optimization method sets result.method to the method name."""
        methods_and_results = {
            "risk_parity": optimizer.risk_parity(sample_cov),
            "hrp": optimizer.hrp(sample_cov),
            "max_diversification": optimizer.max_diversification(sample_cov),
        }
        for expected_method, result in methods_and_results.items():
            assert result.method == expected_method, (
                f"Expected method='{expected_method}', got '{result.method}'"
            )
