"""Unified data access layer for equities and crypto."""
from __future__ import annotations

from pathlib import Path  # noqa: F401 — available for subclass use

import numpy as np  # noqa: F401 — available for subclass use
import pandas as pd
from loguru import logger

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import DataPipelineError


class UnifiedDataLayer:
    """Unified interface for OHLCV data across equities and crypto.

    Routes requests by ticker format:
    - Tickers containing "/" (e.g., "BTC/USDT") → CryptoPipeline
    - All other tickers (e.g., "AAPL", "MSFT") → yfinance

    Output is normalized to consistent format regardless of source.

    Args:
        config: AppConfig instance.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or get_config()

    def get_ohlcv(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch OHLCV data for a ticker (equity or crypto).

        Routes to CryptoPipeline if "/" in ticker, else yfinance.

        Normalization applied to both sources:
        - Index: UTC-aware DatetimeIndex named "timestamp"
        - Columns: lowercase open, high, low, close, volume
        - Sorted ascending by timestamp
        - Duplicate timestamps dropped (keep last)

        Args:
            ticker: Stock ticker (e.g., "AAPL") or crypto pair (e.g., "BTC/USDT").
            period: Time period for equity data (e.g., "6mo", "1y", "3mo").
                    Ignored for crypto (uses default 500 candles).

        Returns:
            DataFrame with columns: open, high, low, close, volume
            UTC DatetimeIndex named "timestamp".

        Raises:
            DataPipelineError: If data fetch fails.
        """
        if "/" in ticker:
            return self._get_crypto_ohlcv(ticker)
        return self._get_equity_ohlcv(ticker, period)

    def _get_equity_ohlcv(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch equity OHLCV via yfinance.

        Args:
            ticker: Equity ticker symbol.
            period: yfinance period string (e.g., "6mo", "1y").

        Returns:
            Normalized OHLCV DataFrame with UTC DatetimeIndex.

        Raises:
            DataPipelineError: If yfinance returns empty data or raises.
        """
        try:
            import yfinance as yf

            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period=period)
            if hist.empty:
                raise DataPipelineError(f"No data returned for {ticker}")
            # Normalize columns
            hist = hist.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            hist = hist[["open", "high", "low", "close", "volume"]]
            hist.index.name = "timestamp"
            # Ensure UTC
            if hist.index.tzinfo is None:
                hist.index = hist.index.tz_localize("UTC")
            else:
                hist.index = hist.index.tz_convert("UTC")
            hist = hist.sort_index()
            hist = hist[~hist.index.duplicated(keep="last")]
            return hist
        except DataPipelineError:
            raise
        except Exception as e:
            raise DataPipelineError(
                f"Failed to fetch equity data for {ticker}: {e}"
            ) from e

    def _get_crypto_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Fetch crypto OHLCV via CryptoPipeline.

        Args:
            ticker: Crypto pair symbol (e.g., "BTC/USDT").

        Returns:
            Normalized OHLCV DataFrame with UTC DatetimeIndex.

        Raises:
            DataPipelineError: If CryptoPipeline raises or returns bad data.
        """
        try:
            from src.core.crypto_pipeline import CryptoPipeline  # lazy import

            pipeline = CryptoPipeline(config=self._config)
            df = pipeline.fetch_ohlcv(ticker, timeframe="1d")
            df.index.name = "timestamp"
            # Ensure UTC
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
        except DataPipelineError:
            raise
        except Exception as e:
            raise DataPipelineError(
                f"Failed to fetch crypto data for {ticker}: {e}"
            ) from e

    def get_correlation_matrix(
        self,
        tickers: list[str],
        period: str = "3mo",
    ) -> pd.DataFrame:
        """Compute pairwise correlation matrix from OHLCV close prices.

        Fetches data for all tickers, aligns on common dates, computes
        daily returns, then calculates correlation matrix.

        Args:
            tickers: List of tickers (mix of equity and crypto allowed).
            period: Historical period for equity tickers.

        Returns:
            Square DataFrame (tickers × tickers) of Pearson correlations.
            Returns identity matrix if < 2 tickers.

        Raises:
            DataPipelineError: If any ticker fails to fetch.
        """
        if len(tickers) < 2:
            if tickers:
                return pd.DataFrame([[1.0]], index=tickers, columns=tickers)
            return pd.DataFrame()

        closes = {}
        for ticker in tickers:
            df = self.get_ohlcv(ticker, period=period)
            closes[ticker] = df["close"]

        combined = pd.DataFrame(closes).dropna()
        returns = combined.pct_change().dropna()
        return returns.corr()

    def get_asset_metadata(self, ticker: str) -> dict:
        """Retrieve metadata for a ticker.

        For equities: uses yfinance .info dict.
        For crypto: returns estimated metadata from CryptoPipeline.

        Args:
            ticker: Ticker symbol.

        Returns:
            Dict with keys: asset_class, sector, market_cap, avg_volume, spread_bps.
            Missing values filled with None or 0.
        """
        if "/" in ticker:
            return self._get_crypto_metadata(ticker)
        return self._get_equity_metadata(ticker)

    def _get_equity_metadata(self, ticker: str) -> dict:
        """Get equity metadata from yfinance.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            Metadata dict with asset_class, sector, market_cap, avg_volume, spread_bps.
        """
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info
            return {
                "asset_class": "EQUITY",
                "sector": info.get("sector"),
                "market_cap": info.get("marketCap"),
                "avg_volume": info.get("averageVolume"),
                "spread_bps": None,
            }
        except Exception as e:
            logger.warning(f"Failed to get equity metadata for {ticker}: {e}")
            return {
                "asset_class": "EQUITY",
                "sector": None,
                "market_cap": None,
                "avg_volume": None,
                "spread_bps": None,
            }

    def _get_crypto_metadata(self, ticker: str) -> dict:
        """Get crypto metadata via CryptoPipeline orderbook.

        Args:
            ticker: Crypto pair symbol (e.g., "BTC/USDT").

        Returns:
            Metadata dict with asset_class, sector, market_cap, avg_volume, spread_bps.
        """
        try:
            from src.core.crypto_pipeline import CryptoPipeline  # lazy import

            pipeline = CryptoPipeline(config=self._config)
            ob = pipeline.fetch_orderbook(ticker)
            best_bid = ob["bids"][0][0] if ob.get("bids") else None
            best_ask = ob["asks"][0][0] if ob.get("asks") else None
            spread_bps = None
            if best_bid and best_ask and best_bid > 0:
                spread_bps = (best_ask - best_bid) / best_bid * 10000
            return {
                "asset_class": "CRYPTO",
                "sector": "Cryptocurrency",
                "market_cap": None,
                "avg_volume": None,
                "spread_bps": spread_bps,
            }
        except Exception as e:
            logger.warning(f"Failed to get crypto metadata for {ticker}: {e}")
            return {
                "asset_class": "CRYPTO",
                "sector": "Cryptocurrency",
                "market_cap": None,
                "avg_volume": None,
                "spread_bps": None,
            }
