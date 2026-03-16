"""Benchmark runners for strategy comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger
from src.utils.schemas import BacktestResult, BenchmarkComparison

logger = get_logger(__name__)

_TRADING_DAYS = 252


class BenchmarkRunner:
    """Compute benchmark returns for strategy comparison.

    Supports buy-and-hold SPY, BTC-USD, 60/40 (SPY/AGG), equal-weight, and
    Monte Carlo random-portfolio percentile ranking.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    # ------------------------------------------------------------------
    # Public benchmarks
    # ------------------------------------------------------------------

    def run_spy(self, start: str, end: str) -> pd.Series:
        """Buy-and-hold SPY daily returns.

        Args:
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            Series of daily percentage returns indexed by date.
        """
        return self._download_returns("SPY", start, end)

    def run_btc(self, start: str, end: str) -> pd.Series:
        """Buy-and-hold BTC-USD daily returns.

        Args:
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            Series of daily percentage returns indexed by date.
        """
        return self._download_returns("BTC-USD", start, end)

    def run_sixty_forty(self, start: str, end: str) -> pd.Series:
        """60% SPY / 40% AGG portfolio with monthly rebalancing.

        Args:
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            Series of daily blended returns indexed by date.
        """
        spy = self._download_returns("SPY", start, end)
        agg = self._download_returns("AGG", start, end)
        combined = pd.concat([spy, agg], axis=1, keys=["SPY", "AGG"]).dropna()
        if combined.empty:
            return pd.Series(dtype=float)

        # Monthly rebalance: weights reset to 0.6 / 0.4 at each month start
        combined["month"] = pd.PeriodIndex(combined.index, freq="M")
        blended = pd.Series(index=combined.index, dtype=float)
        w_spy, w_agg = 0.6, 0.4

        for month, group in combined.groupby("month"):
            for i, (date, row) in enumerate(group.iterrows()):
                blended[date] = w_spy * row["SPY"] + w_agg * row["AGG"]
                # Drift weights intramonth
                w_spy *= 1 + row["SPY"]
                w_agg *= 1 + row["AGG"]
                total = w_spy + w_agg
                w_spy /= total
                w_agg /= total
            # Rebalance at month end
            w_spy, w_agg = 0.6, 0.4

        return blended.dropna()

    def run_equal_weight(self, returns_df: pd.DataFrame) -> pd.Series:
        """Equal-weight portfolio from a returns DataFrame.

        Args:
            returns_df: DataFrame where each column is a return stream.

        Returns:
            Series of equal-weight portfolio daily returns.
        """
        if returns_df.empty:
            return pd.Series(dtype=float)
        return returns_df.mean(axis=1)

    def run_monte_carlo(
        self,
        strategy_returns: pd.Series,
        returns_df: pd.DataFrame,
        n_simulations: int = 1000,
    ) -> float:
        """Monte Carlo random portfolio simulation.

        Samples ``n_simulations`` random equal-weight sub-portfolios from
        ``returns_df`` and computes each portfolio's annualised Sharpe.  Returns
        the percentile rank (0–100) of the strategy Sharpe vs that distribution.

        Args:
            strategy_returns: Daily returns of the strategy under evaluation.
            returns_df: Universe of available daily return streams (columns).
            n_simulations: Number of random portfolios to simulate.

        Returns:
            Percentile rank of strategy Sharpe (float in [0, 100]).
        """
        strategy_sharpe = self._compute_sharpe(strategy_returns)

        if returns_df.empty or returns_df.shape[1] < 2:
            logger.warning("Insufficient universe for Monte Carlo — returning 50th percentile")
            return 50.0

        rng = np.random.default_rng(seed=42)
        n_assets = returns_df.shape[1]
        k = min(30, n_assets)
        sharpes: list[float] = []

        returns_arr = returns_df.values
        for _ in range(n_simulations):
            idx = rng.choice(n_assets, size=k, replace=False)
            port_returns = returns_arr[:, idx].mean(axis=1)
            sharpes.append(self._compute_sharpe(pd.Series(port_returns)))

        sharpes_arr = np.array(sharpes)
        percentile = float(np.mean(sharpes_arr < strategy_sharpe) * 100)
        logger.info(
            "Monte Carlo: strategy Sharpe={:.2f} at {:.1f}th percentile of {} simulations",
            strategy_sharpe,
            percentile,
            n_simulations,
        )
        return percentile

    def compare(
        self,
        result: BacktestResult,
        start: str,
        end: str,
        universe_returns: pd.DataFrame | None = None,
    ) -> BenchmarkComparison:
        """Compare strategy against all benchmarks.

        Args:
            result: Strategy BacktestResult.
            start: Benchmark start date.
            end: Benchmark end date.
            universe_returns: Optional universe DataFrame for equal-weight and
                Monte Carlo benchmarks.

        Returns:
            BenchmarkComparison Pydantic model.
        """
        strategy_returns = pd.Series(result.daily_returns, dtype=float)
        strategy_sharpe = self._compute_sharpe(strategy_returns)

        spy_returns = self.run_spy(start, end)
        btc_returns = self.run_btc(start, end)
        sixty_forty_returns = self.run_sixty_forty(start, end)

        spy_sharpe = self._compute_sharpe(spy_returns)
        btc_sharpe = self._compute_sharpe(btc_returns)
        sixty_forty_sharpe = self._compute_sharpe(sixty_forty_returns)

        if universe_returns is not None and not universe_returns.empty:
            ew_returns = self.run_equal_weight(universe_returns)
            ew_sharpe = self._compute_sharpe(ew_returns)
            mc_pct = self.run_monte_carlo(strategy_returns, universe_returns)
        else:
            ew_sharpe = 0.0
            mc_pct = 50.0

        comparison = BenchmarkComparison(
            strategy_sharpe=strategy_sharpe,
            spy_sharpe=spy_sharpe,
            btc_sharpe=btc_sharpe,
            sixty_forty_sharpe=sixty_forty_sharpe,
            equal_weight_sharpe=ew_sharpe,
            monte_carlo_percentile=mc_pct,
            excess_sharpe_vs_spy=strategy_sharpe - spy_sharpe,
        )
        logger.info(
            "Benchmark comparison: strategy={:.2f}, SPY={:.2f}, BTC={:.2f}, 60/40={:.2f}",
            strategy_sharpe,
            spy_sharpe,
            btc_sharpe,
            sixty_forty_sharpe,
        )
        return comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sharpe(self, returns: pd.Series) -> float:
        """Annualised Sharpe ratio (zero risk-free rate).

        Args:
            returns: Daily return series.

        Returns:
            Annualised Sharpe ratio.
        """
        returns = returns.dropna()
        if returns.empty or returns.std() < 1e-10:
            return 0.0
        annual_return = float((1 + returns).prod() ** (_TRADING_DAYS / len(returns)) - 1)
        vol = float(returns.std() * math.sqrt(_TRADING_DAYS))
        return float(annual_return / vol) if vol > 0 else 0.0

    @staticmethod
    def _download_returns(ticker: str, start: str, end: str) -> pd.Series:
        """Download adjusted close prices via yfinance and compute daily returns.

        Args:
            ticker: Yahoo Finance ticker symbol.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            Series of daily returns.
        """
        import yfinance as yf

        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                logger.warning("No data returned for ticker {}", ticker)
                return pd.Series(dtype=float)

            # yfinance may return MultiIndex columns when downloading a single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"][ticker]
            else:
                close = raw["Close"]

            return close.pct_change().dropna()
        except Exception as exc:
            logger.warning("Failed to download {}: {}", ticker, exc)
            return pd.Series(dtype=float)
