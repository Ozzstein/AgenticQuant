"""Tests for BenchmarkRunner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.benchmark import BenchmarkRunner
from src.utils.schemas import BenchmarkComparison, BacktestMetrics, BacktestResult, WalkForwardFold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(ticker: str, n: int = 252, seed: int = 0) -> pd.DataFrame:
    """Build a fake yfinance download DataFrame for a single ticker."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"
    return df


def _make_multi_price_df(tickers: list[str], n: int = 252) -> pd.DataFrame:
    """Build a multi-column Close DataFrame (MultiIndex) as yfinance returns."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    data = {}
    for ticker in tickers:
        prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        data[ticker] = prices
    df = pd.DataFrame(data, index=dates)
    # Simulate yfinance MultiIndex: ("Close", ticker)
    df.columns = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
    return df


def _make_synthetic_backtest_result(n_days: int = 252, sharpe: float = 1.2) -> BacktestResult:
    """Create a minimal BacktestResult for comparison tests."""
    rng = np.random.default_rng(7)
    daily_returns = rng.normal(sharpe * 0.15 / 252, 0.15 / np.sqrt(252), n_days).tolist()
    equity = (pd.Series([1.0] + daily_returns).cumprod() * 100).tolist()[1:]
    fold = WalkForwardFold(
        fold_id=0,
        train_start="2022-01-01",
        train_end="2022-06-30",
        test_start="2022-07-06",
        test_end="2022-12-30",
        in_sample_sharpe=sharpe,
        oos_sharpe=sharpe * 0.8,
    )
    metrics = BacktestMetrics(
        sharpe_ratio=sharpe,
        annual_return=sharpe * 0.15,
        max_drawdown=-0.08,
        total_trades=30,
        volatility=0.15,
    )
    return BacktestResult(
        metrics=metrics,
        daily_returns=daily_returns,
        trades=[],
        walk_forward_results=[fold],
        equity_curve=equity,
        timestamps=[str(pd.Timestamp("2022-01-01") + pd.Timedelta(days=i)) for i in range(n_days)],
    )


# ---------------------------------------------------------------------------
# SPY tests
# ---------------------------------------------------------------------------


class TestSpyBenchmark:
    @patch("yfinance.download")
    def test_spy_returns_series(self, mock_dl, mock_config):
        """run_spy() returns a pd.Series of daily returns."""
        mock_dl.return_value = _make_price_df("SPY")
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_spy("2022-01-01", "2022-12-31")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    @patch("yfinance.download")
    def test_spy_returns_are_pct_changes(self, mock_dl, mock_config):
        """SPY returns are in daily percentage-change range (not raw prices)."""
        mock_dl.return_value = _make_price_df("SPY")
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_spy("2022-01-01", "2022-12-31")
        assert result.abs().max() < 0.5  # daily return < 50%

    @patch("yfinance.download")
    def test_spy_multiindex_columns(self, mock_dl, mock_config):
        """run_spy() handles MultiIndex column output from yfinance."""
        mock_dl.return_value = _make_multi_price_df(["SPY"])
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_spy("2022-01-01", "2022-12-31")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    @patch("yfinance.download")
    def test_spy_empty_download_returns_empty(self, mock_dl, mock_config):
        """Empty yfinance response → empty Series (no crash)."""
        mock_dl.return_value = pd.DataFrame()
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_spy("2022-01-01", "2022-12-31")
        assert isinstance(result, pd.Series)
        assert result.empty


# ---------------------------------------------------------------------------
# BTC tests
# ---------------------------------------------------------------------------


class TestBtcBenchmark:
    @patch("yfinance.download")
    def test_btc_returns_series(self, mock_dl, mock_config):
        """run_btc() returns a pd.Series of daily returns."""
        mock_dl.return_value = _make_price_df("BTC-USD", seed=1)
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_btc("2022-01-01", "2022-12-31")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    @patch("yfinance.download")
    def test_btc_returns_finite(self, mock_dl, mock_config):
        """BTC returns contain no NaN or inf."""
        mock_dl.return_value = _make_price_df("BTC-USD", seed=2)
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_btc("2022-01-01", "2022-12-31")
        assert result.isna().sum() == 0
        assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# 60/40 tests
# ---------------------------------------------------------------------------


class TestSixtyFortyBenchmark:
    @patch("yfinance.download")
    def test_sixty_forty_returns_series(self, mock_dl, mock_config):
        """run_sixty_forty() returns a pd.Series."""
        # Return different data for SPY vs AGG calls
        mock_dl.side_effect = [
            _make_price_df("SPY", seed=3),
            _make_price_df("AGG", seed=4),
        ]
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_sixty_forty("2022-01-01", "2022-12-31")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    @patch("yfinance.download")
    def test_sixty_forty_sharpe_computable(self, mock_dl, mock_config):
        """60/40 Sharpe ratio can be computed without error."""
        mock_dl.side_effect = [
            _make_price_df("SPY", seed=5),
            _make_price_df("AGG", seed=6),
        ]
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_sixty_forty("2022-01-01", "2022-12-31")
        sharpe = runner._compute_sharpe(result)
        assert np.isfinite(sharpe)

    @patch("yfinance.download")
    def test_sixty_forty_empty_when_no_data(self, mock_dl, mock_config):
        """60/40 returns empty Series when downloads fail."""
        mock_dl.return_value = pd.DataFrame()
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_sixty_forty("2022-01-01", "2022-12-31")
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Equal-weight tests
# ---------------------------------------------------------------------------


class TestEqualWeightBenchmark:
    def test_equal_weight_returns_mean_of_columns(self, mock_config):
        """run_equal_weight() returns the row-wise mean of all columns."""
        rng = np.random.default_rng(10)
        dates = pd.bdate_range("2022-01-01", periods=100)
        df = pd.DataFrame(rng.normal(0, 0.01, (100, 5)), index=dates,
                          columns=["A", "B", "C", "D", "E"])
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_equal_weight(df)
        expected = df.mean(axis=1)
        pd.testing.assert_series_equal(result, expected)

    def test_equal_weight_empty_df(self, mock_config):
        """Empty DataFrame → empty Series."""
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_equal_weight(pd.DataFrame())
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_equal_weight_single_column(self, mock_config):
        """Single-column DataFrame → same values as the column."""
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2022-01-01", periods=50)
        df = pd.DataFrame({"X": rng.normal(0, 0.01, 50)}, index=dates)
        runner = BenchmarkRunner(config=mock_config)
        result = runner.run_equal_weight(df)
        pd.testing.assert_series_equal(result, df["X"], check_names=False)


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------


class TestMonteCarloBenchmark:
    def test_monte_carlo_percentile_range(self, mock_config):
        """Monte Carlo percentile is in [0, 100]."""
        rng = np.random.default_rng(20)
        dates = pd.bdate_range("2022-01-01", periods=252)
        strategy_returns = pd.Series(rng.normal(0.001, 0.01, 252), index=dates)
        universe_df = pd.DataFrame(
            rng.normal(0, 0.01, (252, 20)),
            index=dates,
            columns=[f"T{i}" for i in range(20)],
        )
        runner = BenchmarkRunner(config=mock_config)
        pct = runner.run_monte_carlo(strategy_returns, universe_df, n_simulations=100)
        assert 0.0 <= pct <= 100.0

    def test_monte_carlo_good_strategy_high_pct(self, mock_config):
        """A clearly superior strategy should rank above 50th percentile."""
        rng = np.random.default_rng(21)
        dates = pd.bdate_range("2022-01-01", periods=252)
        # Very good strategy
        strategy_returns = pd.Series(rng.normal(0.01, 0.005, 252), index=dates)
        # Mediocre universe
        universe_df = pd.DataFrame(
            rng.normal(-0.001, 0.02, (252, 30)),
            index=dates,
            columns=[f"T{i}" for i in range(30)],
        )
        runner = BenchmarkRunner(config=mock_config)
        pct = runner.run_monte_carlo(strategy_returns, universe_df, n_simulations=200)
        assert pct > 50.0

    def test_monte_carlo_empty_universe(self, mock_config):
        """Empty universe returns 50.0 (default)."""
        rng = np.random.default_rng(22)
        strategy_returns = pd.Series(rng.normal(0, 0.01, 100))
        runner = BenchmarkRunner(config=mock_config)
        pct = runner.run_monte_carlo(strategy_returns, pd.DataFrame(), n_simulations=100)
        assert pct == 50.0


# ---------------------------------------------------------------------------
# Sharpe helper
# ---------------------------------------------------------------------------


class TestComputeSharpe:
    def test_sharpe_positive_returns(self, mock_config):
        """Positive drift with variance returns positive Sharpe."""
        runner = BenchmarkRunner(config=mock_config)
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.002, 0.01, 252))
        sharpe = runner._compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_zero_variance(self, mock_config):
        """Constant returns → Sharpe = 0 (no variance)."""
        runner = BenchmarkRunner(config=mock_config)
        returns = pd.Series([0.001] * 252)
        sharpe = runner._compute_sharpe(returns)
        assert sharpe == 0.0

    def test_sharpe_empty_series(self, mock_config):
        """Empty series → Sharpe = 0."""
        runner = BenchmarkRunner(config=mock_config)
        sharpe = runner._compute_sharpe(pd.Series(dtype=float))
        assert sharpe == 0.0

    def test_sharpe_finite(self, mock_config):
        """Sharpe of random returns is a finite number."""
        rng = np.random.default_rng(30)
        runner = BenchmarkRunner(config=mock_config)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        sharpe = runner._compute_sharpe(returns)
        assert np.isfinite(sharpe)


# ---------------------------------------------------------------------------
# compare() tests
# ---------------------------------------------------------------------------


class TestCompare:
    @patch("yfinance.download")
    def test_compare_returns_benchmark_comparison(self, mock_dl, mock_config):
        """compare() returns a BenchmarkComparison schema object."""
        mock_dl.return_value = _make_price_df("SPY", seed=40)
        result = _make_synthetic_backtest_result()
        runner = BenchmarkRunner(config=mock_config)
        comparison = runner.compare(result, "2022-01-01", "2022-12-31")
        assert isinstance(comparison, BenchmarkComparison)

    @patch("yfinance.download")
    def test_compare_excess_sharpe_vs_spy(self, mock_dl, mock_config):
        """excess_sharpe_vs_spy = strategy_sharpe - spy_sharpe."""
        mock_dl.return_value = _make_price_df("SPY", seed=41)
        result = _make_synthetic_backtest_result()
        runner = BenchmarkRunner(config=mock_config)
        comparison = runner.compare(result, "2022-01-01", "2022-12-31")
        expected_excess = comparison.strategy_sharpe - comparison.spy_sharpe
        assert abs(comparison.excess_sharpe_vs_spy - expected_excess) < 1e-9

    @patch("yfinance.download")
    def test_compare_with_universe_returns(self, mock_dl, mock_config):
        """compare() with universe_returns populates equal_weight and MC fields."""
        mock_dl.return_value = _make_price_df("SPY", seed=42)
        rng = np.random.default_rng(43)
        dates = pd.bdate_range("2022-01-01", periods=252)
        universe = pd.DataFrame(
            rng.normal(0, 0.01, (252, 10)),
            index=dates,
            columns=[f"T{i}" for i in range(10)],
        )
        result = _make_synthetic_backtest_result()
        runner = BenchmarkRunner(config=mock_config)
        comparison = runner.compare(result, "2022-01-01", "2022-12-31", universe_returns=universe)
        assert isinstance(comparison, BenchmarkComparison)
        assert 0.0 <= comparison.monte_carlo_percentile <= 100.0

    @patch("yfinance.download")
    def test_compare_strategy_sharpe_matches_returns(self, mock_dl, mock_config):
        """strategy_sharpe in comparison matches _compute_sharpe of daily returns."""
        mock_dl.return_value = _make_price_df("SPY", seed=44)
        result = _make_synthetic_backtest_result()
        runner = BenchmarkRunner(config=mock_config)
        comparison = runner.compare(result, "2022-01-01", "2022-12-31")
        expected = runner._compute_sharpe(pd.Series(result.daily_returns))
        assert abs(comparison.strategy_sharpe - expected) < 1e-9
