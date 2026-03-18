"""Crypto-specific alpha factor computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import FactorError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoFactorEngine:
    """Crypto-specific alpha factors.

    All factors return pd.Series indexed by timestamp.
    NaN values returned for rows with insufficient lookback.

    Args:
        config: AppConfig instance.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or get_config()

    def _require_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        """Check required columns exist in DataFrame.

        Args:
            df: Input DataFrame.
            columns: Required column names.

        Raises:
            FactorError: If any required column is missing.
        """
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise FactorError(f"Missing required columns: {missing}")

    def funding_rate_zscore(
        self,
        funding_df: pd.DataFrame,
        window: int = 30,
    ) -> pd.Series:
        """Z-score of 8h funding rate vs rolling mean (contrarian signal).

        Computes z-score = (funding_rate - rolling_mean) / rolling_std.
        Negated because high positive funding -> overleveraged longs -> bearish.

        Args:
            funding_df: DataFrame with 'funding_rate' column, datetime index.
            window: Rolling lookback window (default 30).

        Returns:
            pd.Series of negated z-scores, NaN for first `window` rows.

        Raises:
            FactorError: If 'funding_rate' column missing.
        """
        self._require_columns(funding_df, ["funding_rate"])
        rates = funding_df["funding_rate"]
        roll_mean = rates.rolling(window).mean()
        roll_std = rates.rolling(window).std()
        zscore = (rates - roll_mean) / roll_std.replace(0, np.nan)
        return -zscore

    def oi_price_divergence(
        self,
        oi_df: pd.DataFrame,
        price_df: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """Rolling correlation between OI changes and price changes.

        When OI and price diverge (negative correlation), suggests potential reversal.
        Returns rolling correlation of pct_change(OI) vs pct_change(price).

        Args:
            oi_df: DataFrame with 'open_interest' column, datetime index.
            price_df: DataFrame with 'close' column, datetime index.
            window: Rolling window for correlation (default 20).

        Returns:
            pd.Series of rolling correlations, NaN for first `window` rows.

        Raises:
            FactorError: If required columns missing.
        """
        self._require_columns(oi_df, ["open_interest"])
        self._require_columns(price_df, ["close"])
        oi_chg = oi_df["open_interest"].pct_change()
        price_chg = price_df["close"].pct_change()
        combined = pd.concat([oi_chg, price_chg], axis=1).dropna()
        corr = combined.iloc[:, 0].rolling(window).corr(combined.iloc[:, 1])
        return corr

    def btc_dominance_signal(
        self,
        btc_mcap: pd.Series,
        total_mcap: pd.Series,
    ) -> pd.Series:
        """Z-score of BTC dominance ratio.

        BTC dominance = btc_mcap / total_mcap.
        Returns z-score vs trailing 30-day mean/std.
        High dominance -> risk-off -> bullish BTC, bearish alts.

        Args:
            btc_mcap: BTC market cap time series.
            total_mcap: Total crypto market cap time series.

        Returns:
            pd.Series of dominance z-scores, NaN for first 30 rows.
        """
        dominance = btc_mcap / total_mcap
        roll_mean = dominance.rolling(30).mean()
        roll_std = dominance.rolling(30).std()
        return (dominance - roll_mean) / roll_std.replace(0, np.nan)

    def beta_adjusted_momentum(
        self,
        returns: pd.Series,
        btc_returns: pd.Series,
        window: int = 30,
    ) -> pd.Series:
        """Beta-adjusted momentum (residual alpha vs BTC).

        Estimates rolling beta of ticker vs BTC using OLS on rolling window.
        Residual momentum = ticker_return - beta * btc_return.
        Returns rolling sum of residuals over the window.

        Args:
            returns: Ticker daily returns (pct_change).
            btc_returns: BTC daily returns (pct_change).
            window: Rolling beta estimation window (default 30).

        Returns:
            pd.Series of cumulative residual momentum, NaN for first `window` rows.
        """
        common_idx = returns.index.intersection(btc_returns.index)
        r = returns.loc[common_idx]
        b = btc_returns.loc[common_idx]

        def rolling_beta(i: int) -> float:
            if i < window:
                return np.nan
            y = r.iloc[i - window : i]
            x = b.iloc[i - window : i]
            cov = np.cov(x, y)[0, 1]
            var = np.var(x)
            return cov / var if var > 0 else 0.0

        betas = pd.Series([rolling_beta(i) for i in range(len(r))], index=r.index)
        residuals = r - betas * b
        return residuals.rolling(window).sum()

    def realized_vol_cone(
        self,
        prices: pd.Series,
        window: int = 20,
        lookback: int = 252,
    ) -> pd.Series:
        """Current realized vol percentile rank vs trailing 1Y.

        Computes rolling 20-day realized vol, then ranks the current value
        against all historical 20-day vol values in trailing `lookback` window.
        Percentile rank = 0 (lowest vol ever) to 1 (highest vol ever).

        Args:
            prices: Price time series.
            window: Vol computation window (default 20).
            lookback: Trailing history for percentile (default 252).

        Returns:
            pd.Series of percentile ranks in [0, 1], NaN for first (window+lookback) rows.
        """
        log_returns = np.log(prices / prices.shift(1))
        realized_vol = log_returns.rolling(window).std() * np.sqrt(252)
        percentile = realized_vol.rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        return percentile

    def nvt_proxy(
        self,
        mcap: pd.Series,
        volume: pd.Series,
        window: int = 30,
    ) -> pd.Series:
        """Z-score of NVT proxy (market_cap / exchange_volume ratio).

        High NVT -> overvalued relative to network activity -> bearish.
        Z-score vs trailing 30-day mean/std.

        Args:
            mcap: Market cap time series.
            volume: Exchange volume (USD) time series.
            window: Rolling window for z-score (default 30).

        Returns:
            pd.Series of NVT z-scores, NaN for first `window` rows.
        """
        ratio = mcap / volume.replace(0, np.nan)
        roll_mean = ratio.rolling(window).mean()
        roll_std = ratio.rolling(window).std()
        return (ratio - roll_mean) / roll_std.replace(0, np.nan)

    def futures_spot_ratio(
        self,
        futures_vol: pd.Series,
        spot_vol: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Rolling z-score of futures/spot volume ratio.

        High futures/spot ratio -> speculative activity -> potential reversal.
        Z-score = (ratio - rolling_mean) / rolling_std.

        Args:
            futures_vol: Futures trading volume time series.
            spot_vol: Spot trading volume time series.
            window: Rolling window for z-score (default 20).

        Returns:
            pd.Series of ratio z-scores, NaN for first `window` rows.
        """
        ratio = futures_vol / spot_vol.replace(0, np.nan)
        roll_mean = ratio.rolling(window).mean()
        roll_std = ratio.rolling(window).std()
        return (ratio - roll_mean) / roll_std.replace(0, np.nan)
