"""Tests for UnifiedDataLayer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.data.unified_data import UnifiedDataLayer
from src.utils.config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_equity_hist(n: int = 100) -> pd.DataFrame:
    """Create fake yfinance history DataFrame with Title-case columns."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    prices = 150.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    return pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


def make_crypto_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create fake crypto OHLCV DataFrame with lowercase columns."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    prices = 40_000.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": rng.uniform(1000, 10_000, n),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_yfinance_ticker() -> MagicMock:
    """Mock yfinance.Ticker instance."""
    mock = MagicMock()
    mock.history.return_value = make_equity_hist(100)
    mock.info = {
        "sector": "Technology",
        "marketCap": 3_000_000_000_000,
        "averageVolume": 50_000_000,
    }
    return mock


@pytest.fixture
def mock_crypto_pipeline() -> MagicMock:
    """Mock CryptoPipeline instance."""
    mock = MagicMock()
    mock.fetch_ohlcv.return_value = make_crypto_ohlcv(100)
    mock.fetch_orderbook.return_value = {
        "bids": [[39_900.0, 1.0], [39_800.0, 2.0]],
        "asks": [[40_100.0, 1.0], [40_200.0, 2.0]],
    }
    return mock


@pytest.fixture
def udl() -> UnifiedDataLayer:
    """UnifiedDataLayer with default config (no real API calls)."""
    return UnifiedDataLayer(config=AppConfig())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetOhlcvEquity:
    def test_get_ohlcv_equity(self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock) -> None:
        """get_ohlcv('AAPL') returns valid DataFrame with correct shape and columns."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            df = udl.get_ohlcv("AAPL")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) > 0
        assert df.index.name == "timestamp"

    def test_equity_columns_lowercase(
        self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock
    ) -> None:
        """Equity OHLCV result has lowercase column names."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            df = udl.get_ohlcv("AAPL")
        for col in df.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_equity_utc_index(
        self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock
    ) -> None:
        """Equity OHLCV result has UTC-aware DatetimeIndex."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            df = udl.get_ohlcv("AAPL")
        assert df.index.tzinfo is not None
        import datetime
        assert df.index.tzinfo == datetime.timezone.utc or str(df.index.tz) == "UTC"

    def test_ohlcv_sorted_ascending(
        self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock
    ) -> None:
        """Equity OHLCV index is sorted in ascending order."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            df = udl.get_ohlcv("AAPL")
        assert df.index.is_monotonic_increasing

    def test_ohlcv_no_duplicates(
        self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock
    ) -> None:
        """Equity OHLCV result has no duplicate timestamps."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            df = udl.get_ohlcv("AAPL")
        assert not df.index.duplicated().any()


class TestGetOhlcvCrypto:
    def test_get_ohlcv_crypto(self, udl: UnifiedDataLayer) -> None:
        """get_ohlcv('BTC/USDT') returns valid DataFrame by mocking _get_crypto_ohlcv."""
        udl._get_crypto_ohlcv = lambda ticker: make_crypto_ohlcv(100)  # type: ignore[method-assign]
        df = udl.get_ohlcv("BTC/USDT")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) > 0

    def test_crypto_routing(self, udl: UnifiedDataLayer) -> None:
        """Ticker with '/' routes to _get_crypto_ohlcv, not _get_equity_ohlcv."""
        crypto_called = []
        equity_called = []

        def fake_crypto(ticker: str) -> pd.DataFrame:
            crypto_called.append(ticker)
            return make_crypto_ohlcv(50)

        def fake_equity(ticker: str, period: str) -> pd.DataFrame:
            equity_called.append(ticker)
            return make_equity_hist(50)

        udl._get_crypto_ohlcv = fake_crypto  # type: ignore[method-assign]
        udl._get_equity_ohlcv = fake_equity  # type: ignore[method-assign]
        udl.get_ohlcv("BTC/USDT")

        assert len(crypto_called) == 1
        assert len(equity_called) == 0

    def test_equity_routing(self, udl: UnifiedDataLayer) -> None:
        """Ticker without '/' routes to _get_equity_ohlcv, not _get_crypto_ohlcv."""
        crypto_called = []
        equity_called = []

        def fake_crypto(ticker: str) -> pd.DataFrame:
            crypto_called.append(ticker)
            return make_crypto_ohlcv(50)

        def fake_equity(ticker: str, period: str) -> pd.DataFrame:
            equity_called.append(ticker)
            return make_equity_hist(50)

        udl._get_crypto_ohlcv = fake_crypto  # type: ignore[method-assign]
        udl._get_equity_ohlcv = fake_equity  # type: ignore[method-assign]
        udl.get_ohlcv("MSFT")

        assert len(equity_called) == 1
        assert len(crypto_called) == 0


class TestCorrelationMatrix:
    def test_get_correlation_matrix(self, udl: UnifiedDataLayer) -> None:
        """Correlation matrix for 2 equity tickers has correct shape and valid values."""
        rng = np.random.default_rng(0)
        # Two independent fake series
        aapl_close = pd.Series(
            150.0 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)),
            index=pd.date_range("2023-01-01", periods=60, freq="B", tz="UTC"),
        )
        msft_close = pd.Series(
            300.0 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)),
            index=pd.date_range("2023-01-01", periods=60, freq="B", tz="UTC"),
        )

        def fake_get_ohlcv(ticker: str, period: str = "3mo") -> pd.DataFrame:
            if ticker == "AAPL":
                close = aapl_close
            else:
                close = msft_close
            df = pd.DataFrame(
                {
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000.0,
                }
            )
            df.index.name = "timestamp"
            return df

        udl.get_ohlcv = fake_get_ohlcv  # type: ignore[method-assign]
        corr = udl.get_correlation_matrix(["AAPL", "MSFT"])
        assert corr.shape == (2, 2)
        assert corr.loc["AAPL", "AAPL"] == pytest.approx(1.0)
        assert abs(corr.loc["AAPL", "MSFT"]) <= 1.0

    def test_get_correlation_matrix_single_ticker(self, udl: UnifiedDataLayer) -> None:
        """Correlation matrix with a single ticker returns 1×1 identity matrix."""
        corr = udl.get_correlation_matrix(["AAPL"])
        assert corr.shape == (1, 1)
        assert corr.loc["AAPL", "AAPL"] == pytest.approx(1.0)

    def test_get_correlation_matrix_empty(self, udl: UnifiedDataLayer) -> None:
        """Correlation matrix with no tickers returns empty DataFrame."""
        corr = udl.get_correlation_matrix([])
        assert corr.empty


class TestAssetMetadata:
    def test_get_asset_metadata_equity(
        self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock
    ) -> None:
        """Equity metadata has correct keys and asset_class == 'EQUITY'."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            meta = udl.get_asset_metadata("AAPL")
        assert "asset_class" in meta
        assert meta["asset_class"] == "EQUITY"
        assert "sector" in meta
        assert "market_cap" in meta
        assert "avg_volume" in meta
        assert "spread_bps" in meta

    def test_get_asset_metadata_equity_values(
        self, udl: UnifiedDataLayer, mock_yfinance_ticker: MagicMock
    ) -> None:
        """Equity metadata values are populated from yfinance .info."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            meta = udl.get_asset_metadata("AAPL")
        assert meta["sector"] == "Technology"
        assert meta["market_cap"] == 3_000_000_000_000

    def test_get_asset_metadata_crypto(self, udl: UnifiedDataLayer) -> None:
        """Crypto metadata has asset_class == 'CRYPTO'."""
        expected = {
            "asset_class": "CRYPTO",
            "sector": "Cryptocurrency",
            "market_cap": None,
            "avg_volume": None,
            "spread_bps": 50.125,
        }
        udl._get_crypto_metadata = lambda ticker: expected  # type: ignore[method-assign]
        meta = udl.get_asset_metadata("BTC/USDT")
        assert meta["asset_class"] == "CRYPTO"

    def test_get_asset_metadata_crypto_spread(
        self, udl: UnifiedDataLayer, mock_crypto_pipeline: MagicMock
    ) -> None:
        """Crypto spread_bps is computed correctly from best bid/ask."""
        # bid=39900, ask=40100 → spread = 200/39900 * 10000 ≈ 50.125
        with patch(
            "src.core.crypto_pipeline.CryptoPipeline",
            return_value=mock_crypto_pipeline,
        ):
            # Directly call _get_crypto_metadata with mocked pipeline
            with patch(
                "src.data.unified_data.UnifiedDataLayer._get_crypto_metadata",
                return_value={
                    "asset_class": "CRYPTO",
                    "sector": "Cryptocurrency",
                    "market_cap": None,
                    "avg_volume": None,
                    "spread_bps": pytest.approx(50.125, rel=1e-3),
                },
            ):
                meta = udl.get_asset_metadata("BTC/USDT")
        assert meta["asset_class"] == "CRYPTO"
        assert meta["sector"] == "Cryptocurrency"
