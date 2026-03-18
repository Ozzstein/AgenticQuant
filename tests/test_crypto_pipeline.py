"""Tests for CryptoPipeline using mocked CCXT."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.crypto_pipeline import CryptoPipeline
from src.utils.config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_ohlcv(n: int = 10) -> list[list]:
    """Generate fake OHLCV candles as list of lists.

    Args:
        n: Number of candles to generate.

    Returns:
        List of [timestamp_ms, open, high, low, close, volume] rows.
    """
    base_ts = 1_700_000_000_000  # ms timestamp
    return [
        [
            base_ts + i * 86_400_000,
            40_000 + i * 100,
            41_000 + i * 100,
            39_000 + i * 100,
            40_500 + i * 100,
            1_000.0 + i,
        ]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_exchange() -> MagicMock:
    """Create a fully mocked CCXT exchange."""
    ex = MagicMock()
    ex.fetch_ohlcv.return_value = make_fake_ohlcv(10)
    ex.fetch_tickers.return_value = {
        "BTC/USDT": {"quoteVolume": 1_000_000_000, "symbol": "BTC/USDT"},
        "ETH/USDT": {"quoteVolume": 500_000_000, "symbol": "ETH/USDT"},
        "USDT/USDC": {"quoteVolume": 50_000_000, "symbol": "USDT/USDC"},  # stablecoin — excluded
        "WBTC/USDT": {"quoteVolume": 20_000_000, "symbol": "WBTC/USDT"},  # wrapped — excluded
        "ADA/USDT": {"quoteVolume": 100_000_000, "symbol": "ADA/USDT"},
    }
    ex.fetch_funding_rate_history = MagicMock(
        return_value=[
            {"timestamp": 1_700_000_000_000, "fundingRate": 0.0001},
            {"timestamp": 1_700_086_400_000, "fundingRate": 0.0002},
        ]
    )
    ex.fetch_open_interest_history = MagicMock(
        return_value=[
            {"timestamp": 1_700_000_000_000, "openInterest": 10_000.0},
        ]
    )
    ex.fetch_order_book = MagicMock(
        return_value={
            "bids": [[39_900, 1.0], [39_800, 2.0]],
            "asks": [[40_100, 1.0], [40_200, 2.0]],
            "timestamp": 1_700_000_000_000,
            "datetime": "2023-11-14T22:13:20.000Z",
            "nonce": None,
        }
    )
    return ex


@pytest.fixture()
def pipeline(mock_exchange: MagicMock, tmp_path: Path) -> CryptoPipeline:
    """CryptoPipeline with mocked exchange and temp cache dir."""
    with patch("ccxt.binance", return_value=mock_exchange):
        p = CryptoPipeline(exchange="binance", cache_dir=tmp_path / "crypto_cache")
    # Inject the mock directly so we bypass any real exchange calls.
    p._exchange = mock_exchange
    return p


# ---------------------------------------------------------------------------
# fetch_ohlcv tests
# ---------------------------------------------------------------------------


def test_fetch_ohlcv_returns_dataframe(pipeline: CryptoPipeline) -> None:
    """fetch_ohlcv should return a non-empty DataFrame."""
    df = pipeline.fetch_ohlcv("BTC/USDT")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_fetch_ohlcv_column_names(pipeline: CryptoPipeline) -> None:
    """Returned DataFrame must have exactly the expected OHLCV columns."""
    df = pipeline.fetch_ohlcv("BTC/USDT")
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_fetch_ohlcv_index_is_datetime(pipeline: CryptoPipeline) -> None:
    """Index should be a DatetimeIndex (with or without timezone)."""
    df = pipeline.fetch_ohlcv("BTC/USDT")
    assert isinstance(df.index, pd.DatetimeIndex), f"Expected DatetimeIndex, got {type(df.index)}"


def test_fetch_ohlcv_nonempty(pipeline: CryptoPipeline) -> None:
    """Returned DataFrame should have 10 rows matching the fake data fixture."""
    df = pipeline.fetch_ohlcv("BTC/USDT")
    assert len(df) == 10


def test_fetch_ohlcv_caching(pipeline: CryptoPipeline, mock_exchange: MagicMock) -> None:
    """Second call with a fresh cache should not hit the exchange again."""
    # First call: exchange is queried and result is written to cache.
    pipeline.fetch_ohlcv("BTC/USDT")
    assert mock_exchange.fetch_ohlcv.call_count == 1

    # Second call: cache file exists and is fresh — exchange must NOT be called again.
    pipeline.fetch_ohlcv("BTC/USDT")
    assert mock_exchange.fetch_ohlcv.call_count == 1, (
        "Exchange was called on second fetch despite valid cache."
    )


# ---------------------------------------------------------------------------
# build_universe tests
# ---------------------------------------------------------------------------


def test_build_universe_excludes_stablecoins(pipeline: CryptoPipeline) -> None:
    """Stablecoin pairs (USDT/USDC) must be absent from the universe."""
    universe = pipeline.build_universe()
    bases = [sym.split("/")[0] for sym in universe]
    assert "USDT" not in bases, "Stablecoin USDT should be excluded"
    assert "USDC" not in bases, "Stablecoin USDC should be excluded"
    assert "USDT/USDC" not in universe


def test_build_universe_excludes_wrapped(pipeline: CryptoPipeline) -> None:
    """Wrapped tokens (WBTC) must be absent from the universe."""
    universe = pipeline.build_universe()
    assert "WBTC/USDT" not in universe, "Wrapped token WBTC/USDT should be excluded"


def test_build_universe_length(pipeline: CryptoPipeline) -> None:
    """Universe length must not exceed the requested top_n."""
    universe = pipeline.build_universe(top_n=2)
    assert len(universe) <= 2


def test_build_universe_returns_list_of_strings(pipeline: CryptoPipeline) -> None:
    """All elements in the universe must be strings."""
    universe = pipeline.build_universe()
    assert isinstance(universe, list)
    assert all(isinstance(s, str) for s in universe)


# ---------------------------------------------------------------------------
# fetch_orderbook tests
# ---------------------------------------------------------------------------


def test_fetch_orderbook_keys(pipeline: CryptoPipeline) -> None:
    """Order book dict must contain 'bids' and 'asks' keys."""
    ob = pipeline.fetch_orderbook("BTC/USDT")
    assert "bids" in ob
    assert "asks" in ob


# ---------------------------------------------------------------------------
# fetch_funding_rates tests
# ---------------------------------------------------------------------------


def test_fetch_funding_rates_returns_dataframe(pipeline: CryptoPipeline) -> None:
    """fetch_funding_rates must return a DataFrame (possibly empty)."""
    df = pipeline.fetch_funding_rates("BTC/USDT:USDT")
    assert isinstance(df, pd.DataFrame)


def test_fetch_funding_rates_has_funding_rate_column(pipeline: CryptoPipeline) -> None:
    """Non-empty funding-rates result should have a 'funding_rate' column."""
    df = pipeline.fetch_funding_rates("BTC/USDT:USDT")
    if not df.empty:
        assert "funding_rate" in df.columns


# ---------------------------------------------------------------------------
# fetch_open_interest tests
# ---------------------------------------------------------------------------


def test_fetch_open_interest_returns_dataframe(pipeline: CryptoPipeline) -> None:
    """fetch_open_interest must return a DataFrame (possibly empty)."""
    df = pipeline.fetch_open_interest("BTC/USDT")
    assert isinstance(df, pd.DataFrame)


def test_fetch_open_interest_has_open_interest_column(pipeline: CryptoPipeline) -> None:
    """Non-empty open-interest result should have an 'open_interest' column."""
    df = pipeline.fetch_open_interest("BTC/USDT")
    if not df.empty:
        assert "open_interest" in df.columns
