"""CCXT-based crypto data pipeline with caching and universe building."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import DataPipelineError

_STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX", "USDP"}
_WRAPPED = {"WBTC", "WETH", "WBNB", "STETH"}
_CACHE_TTL_HOURS = 1


class CryptoPipeline:
    """CCXT-based data pipeline for crypto OHLCV, funding rates, and universe building.

    Args:
        exchange: CCXT exchange ID (default "binance").
        cache_dir: Directory for parquet cache files.
        config: AppConfig instance.
    """

    def __init__(
        self,
        exchange: str = "binance",
        cache_dir: Path = Path("outputs/crypto_cache"),
        config: AppConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        self._cache_dir = Path(cache_dir)
        self._exchange_id = exchange
        self._exchange = self._create_exchange(exchange)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Create and configure a CCXT exchange instance.

        Args:
            exchange_id: CCXT exchange identifier string e.g. "binance".

        Returns:
            Configured CCXT Exchange object.

        Raises:
            DataPipelineError: If the exchange ID is not recognised by CCXT.
        """
        try:
            exchange_class = getattr(ccxt, exchange_id)
        except AttributeError as exc:
            raise DataPipelineError(f"Unknown CCXT exchange: {exchange_id!r}") from exc
        return exchange_class({"enableRateLimit": True, "timeout": 30_000})

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """Return parquet cache path for a symbol / timeframe pair."""
        safe_symbol = hashlib.sha256(symbol.encode()).hexdigest()[:24]
        return self._cache_dir / self._exchange_id / safe_symbol / f"{timeframe}.parquet"

    def _cache_valid(self, path: Path) -> bool:
        """Return True if a cache file exists and is within the TTL window."""
        if not path.exists():
            return False
        age_seconds = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
        return age_seconds < _CACHE_TTL_HOURS * 3600

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _fetch_ohlcv_page(
        self, symbol: str, timeframe: str, since_ms: int | None, limit: int
    ) -> list[list]:
        """Fetch a single page of OHLCV candles from the exchange (with retry).

        Args:
            symbol: Trading pair e.g. "BTC/USDT".
            timeframe: Candle timeframe string.
            since_ms: Start timestamp in milliseconds (UTC), or None.
            limit: Maximum candles to fetch in this page.

        Returns:
            List of candle lists: [timestamp_ms, open, high, low, close, volume].
        """
        return self._exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol with pagination and parquet caching.

        Auto-paginates when more than 1 000 candles are requested (CCXT typical max
        per request). Checks the parquet cache first (1-hour TTL). Writes a fresh
        result to the cache after a successful fetch.

        Cache path:
            ``{cache_dir}/{exchange_id}/{symbol.replace("/","_")}/{timeframe}.parquet``

        Args:
            symbol: Trading pair e.g. "BTC/USDT".
            timeframe: Candle timeframe e.g. "1d", "1h", "4h".
            since: Start datetime (UTC). If None, fetch the most recent ``limit`` candles.
            limit: Total number of candles to retrieve.

        Returns:
            DataFrame with columns ``open``, ``high``, ``low``, ``close``, ``volume``
            and a UTC ``DatetimeIndex`` named ``timestamp``.

        Raises:
            DataPipelineError: If the fetch fails after all retries.
        """
        if limit < 1 or limit > 100_000:
            raise DataPipelineError("OHLCV limit must be 1..100000")
        if since is not None and (since.tzinfo is None or since.utcoffset() is None):
            raise DataPipelineError("since must be timezone-aware")
        request_hash = hashlib.sha256(json.dumps({"since": since.isoformat() if since else None, "limit": limit}, sort_keys=True).encode()).hexdigest()[:20]
        cache_path = self._cache_path(symbol, timeframe).with_name(f"{timeframe}-{request_hash}.parquet")
        if self._cache_valid(cache_path):
            logger.debug("Cache hit for {}/{} {}", self._exchange_id, symbol, timeframe)
            return pd.read_parquet(cache_path)

        logger.info("Fetching OHLCV {}/{} {} limit={}", self._exchange_id, symbol, timeframe, limit)
        try:
            all_candles: list[list] = []
            since_ms = int(since.timestamp() * 1000) if since else None
            page_limit = min(limit, 1000)

            while True:
                candles = self._fetch_ohlcv_page(symbol, timeframe, since_ms, page_limit)
                if not candles:
                    break
                all_candles.extend(candles)
                if len(candles) < page_limit:
                    break  # last page — exchange returned fewer candles than requested
                next_since = max(row[0] for row in candles) + 1
                if since_ms is not None and next_since <= since_ms:
                    raise DataPipelineError("OHLCV pagination did not advance")
                since_ms = next_since
                if len(all_candles) >= limit:
                    break

        except Exception as exc:
            raise DataPipelineError(f"Failed to fetch OHLCV for {symbol}: {exc}") from exc

        if not all_candles:
            logger.warning("No OHLCV data returned for {}", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_candles[:limit],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        if df.index.duplicated().any():
            conflicts = df.groupby(level=0).nunique().gt(1).any(axis=1)
            if conflicts.any():
                raise DataPipelineError("conflicting candles at the same timestamp")
            df = df[~df.index.duplicated()]
        # This legacy display API is capture-time data, not a historical authority.
        # The immutable CCXTAdapter supplies research snapshots and provenance.
        if since is not None:
            df = df.loc[df.index >= since]
        seconds = ccxt.Exchange.parse_timeframe(timeframe)
        df = df.loc[df.index + pd.Timedelta(seconds=seconds) <= datetime.now(timezone.utc)]

        # Write to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.debug("Wrote cache for {}/{} {}", self._exchange_id, symbol, timeframe)

        return df

    def fetch_funding_rates(self, symbol: str) -> pd.DataFrame:
        """Fetch historical funding rates for a perpetual futures symbol.

        Args:
            symbol: Symbol e.g. "BTC/USDT:USDT" for Binance perps.

        Returns:
            DataFrame with column ``funding_rate`` and a UTC ``DatetimeIndex``
            named ``timestamp``. Returns an empty DataFrame if the exchange does
            not support funding-rate history.
        """
        try:
            raw = self._exchange.fetch_funding_rate_history(symbol)
        except Exception as exc:  # noqa: BLE001
            raise DataPipelineError(f"funding rates unavailable for {symbol}: {type(exc).__name__}") from None

        if not raw:
            return pd.DataFrame(columns=["funding_rate"])

        df = pd.DataFrame(raw)
        # Normalise column names — CCXT may return either camelCase or snake_case
        rename_map: dict[str, str] = {}
        for col in df.columns:
            if col.lower() == "fundingrate":
                rename_map[col] = "funding_rate"
            elif col.lower() == "timestamp":
                rename_map[col] = "timestamp"
        if rename_map:
            df = df.rename(columns=rename_map)

        if "funding_rate" not in df.columns and "fundingRate" in df.columns:
            df = df.rename(columns={"fundingRate": "funding_rate"})

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()

        if "funding_rate" not in df.columns:
            raise DataPipelineError("funding response lacks funding rate")
        return df[["funding_rate"]]

    def fetch_open_interest(self, symbol: str) -> pd.DataFrame:
        """Fetch historical open interest data.

        Args:
            symbol: Symbol e.g. "BTC/USDT".

        Returns:
            DataFrame with column ``open_interest`` and a UTC ``DatetimeIndex``
            named ``timestamp``. Returns an empty DataFrame if not supported.
        """
        try:
            raw = self._exchange.fetch_open_interest_history(symbol)
        except Exception as exc:  # noqa: BLE001
            raise DataPipelineError(f"open interest unavailable for {symbol}: {type(exc).__name__}") from None

        if not raw:
            return pd.DataFrame(columns=["open_interest"])

        df = pd.DataFrame(raw)

        # Normalise column names
        if "openInterestAmount" in df.columns:
            df = df.rename(columns={"openInterestAmount": "open_interest"})
        elif "openInterest" in df.columns:
            df = df.rename(columns={"openInterest": "open_interest"})
        else:
            raise DataPipelineError("open interest response lacks amount; quote notional is not a substitute")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()

        return df[["open_interest"]] if "open_interest" in df.columns else df

    def fetch_orderbook(self, symbol: str, depth: int = 20) -> dict:
        """Fetch a current order-book snapshot.

        Args:
            symbol: Trading pair e.g. "BTC/USDT".
            depth: Number of bid/ask price levels to retrieve.

        Returns:
            Dict with keys ``bids`` (list of [price, qty]), ``asks``
            (list of [price, qty]), ``timestamp``, ``datetime``, ``nonce``.

        Raises:
            DataPipelineError: If the request fails.
        """
        try:
            return self._exchange.fetch_order_book(symbol, depth)
        except Exception as exc:
            raise DataPipelineError(f"Failed to fetch order book for {symbol}: {exc}") from exc

    def build_universe(self, top_n: int = 50) -> list[str]:
        """Build a crypto universe ranked by 24 h volume, excluding noise.

        Fetches all tickers from the exchange, keeps only ``{base}/USDT`` pairs
        whose 24 h quote volume exceeds $10 M, and excludes stablecoins
        (USDT, USDC, DAI, BUSD, …) and wrapped tokens (WBTC, WETH, …).

        Args:
            top_n: Maximum number of symbols to return.

        Returns:
            List of symbol strings e.g. ``["BTC/USDT", "ETH/USDT", …]``,
            sorted descending by 24 h quote volume.

        Raises:
            DataPipelineError: If the ticker fetch fails.
        """
        try:
            markets = self._exchange.fetch_tickers()
        except Exception as exc:
            raise DataPipelineError(f"Failed to fetch tickers: {exc}") from exc

        excluded = _STABLECOINS | _WRAPPED
        filtered = [
            sym
            for sym, data in markets.items()
            if sym.endswith("/USDT")
            and (data.get("quoteVolume") or 0) > 10_000_000
            and sym.split("/")[0] not in excluded
        ]

        sorted_syms = sorted(
            filtered,
            key=lambda s: markets[s].get("quoteVolume") or 0,
            reverse=True,
        )
        universe = sorted_syms[:top_n]
        logger.info(
            "Built crypto universe: {} symbols (top_n={}, after filtering {} candidates)",
            len(universe),
            top_n,
            len(filtered),
        )
        return universe
