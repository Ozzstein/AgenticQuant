"""CCXT spot/derivative observations and unadjusted US-equity ingestion."""
from __future__ import annotations

import importlib.metadata
from datetime import datetime, timedelta, timezone

from src.data.adapters.base import FetchResult, ProviderError, store_observation
from src.data.evidence.store import EvidenceStore, canonical_bytes
from src.domain.common import Instrument, RunContext, _utc
from src.domain.market import MarketBar


class CCXTAdapter:
    """Only observation methods are reachable; no generic exchange dispatch."""

    def __init__(self, store: EvidenceStore, exchange, *, clock=None):
        self.store, self.exchange = store, exchange
        self.version = f"ccxt-{importlib.metadata.version('ccxt')}:capture-v1"
        self.clock = clock

    def bars(self, instrument: Instrument, context: RunContext, *, ingested_at: datetime,
             start: datetime, end: datetime, timeframe: str = "1d", market_type: str = "spot",
             max_pages: int = 100) -> FetchResult:
        if instrument.asset_class != "crypto" or instrument.venue != self.exchange.id:
            raise ProviderError("ccxt", "instrument_or_venue_mismatch")
        start, end = _utc(start), _utc(end)
        if start >= end or end > context.as_of or max_pages < 1:
            raise ProviderError("ccxt", "invalid_window")
        if market_type not in ("spot", "perpetual"):
            raise ProviderError("ccxt", "invalid_market_type")
        try:
            markets = self.exchange.load_markets()
            market = markets[instrument.symbol]
            if not market.get("spot" if market_type == "spot" else "swap"):
                raise ProviderError("ccxt", "spot_derivative_identity_mismatch")
            seconds = self.exchange.parse_timeframe(timeframe)
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("ccxt", "market_metadata_unavailable") from None
        cursor, seen, records = int(start.timestamp() * 1000), set(), []
        end_ms = int(end.timestamp() * 1000)
        for _ in range(max_pages):
            try:
                rows = self.exchange.fetch_ohlcv(instrument.symbol, timeframe, since=cursor, limit=1000)
            except Exception:
                raise ProviderError("ccxt", "ohlcv_unavailable") from None
            if not rows:
                break
            raw = canonical_bytes(rows)
            try:
                newest = max(int(row[0]) for row in rows)
                if newest < cursor:
                    raise ProviderError("ccxt", "pagination_did_not_advance")
                for row in rows:
                    opened = datetime.fromtimestamp(row[0] / 1000, timezone.utc)
                    closed = opened + timedelta(seconds=seconds)
                    if opened < start or closed > end:
                        continue
                    identity = (row[0], tuple(row[1:]))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    normalized = dict(opened_at=opened.isoformat(), closed_at=closed.isoformat(),
                                      **{k: str(v) for k, v in zip(("open", "high", "low", "close", "volume"), row[1:6])},
                                      price_basis="raw", market_type=market_type, calendar="24/7",
                                      session=opened.date().isoformat(), instrument=instrument.model_dump(mode="json"))
                    MarketBar(evidence_id="validate", **normalized)
                    records.append(store_observation(self.store, raw=raw, normalized=normalized,
                                   provider=f"ccxt:{self.exchange.id}", version=self.version,
                                   source_uri=f"ccxt://{self.exchange.id}/ohlcv/{instrument.symbol}/{timeframe}",
                                   kind="bar", logical_key=f"{instrument.instrument_id}:{timeframe}:{opened.isoformat()}",
                                   context=context, ingested_at=ingested_at, instrument=instrument,
                                   observed_at=closed, units="quote/base", adjustment_policy="raw"))
            except ProviderError:
                raise
            except (ValueError, TypeError, IndexError):
                raise ProviderError("ccxt", "malformed_candle") from None
            cursor = newest + int(seconds * 1000)
            if cursor >= end_ms:
                break
        else:
            raise ProviderError("ccxt", "pagination_budget_exhausted")
        return FetchResult("available" if records else "missing", tuple(records), "capture-time availability")

    def derivatives(
        self,
        instrument: Instrument,
        context: RunContext,
        *,
        capability: str,
        since: datetime,
        end: datetime | None = None,
        timeframe: str = "1h",
        limit: int | None = None,
        max_pages: int = 100,
        ingested_at: datetime | None = None,
    ) -> FetchResult:
        """Closed native GET history, measured AFTER each response.

        ``ingested_at`` is retained only for legacy call compatibility and never
        controls source availability. New refreshes must use the actual store.
        """
        from src.data.adapters.crypto_context import derivative_history

        return derivative_history(
            self,
            instrument,
            context,
            capability=capability,
            since=since,
            end=end,
            timeframe=timeframe,
            limit=limit,
            max_pages=max_pages,
        )


class YFinanceAdapter:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def bars(self, instrument: Instrument, context: RunContext, *, start: str, end: str,
             ingested_at: datetime) -> FetchResult:
        import yfinance as yf

        from src.data.calendars import sessions
        if instrument.asset_class not in ("equity", "etf"):
            raise ProviderError("yfinance", "equity_instrument_required")
        try:
            data = yf.Ticker(instrument.symbol).history(start=start, end=end, auto_adjust=False, actions=True)
        except Exception:
            raise ProviderError("yfinance", "history_unavailable") from None
        if data.empty:
            return FetchResult("missing", ())
        # yfinance exposes parsed data, not HTTP bytes: preserve its FULL serialized
        # output and identify the SDK version rather than claiming wire-byte capture.
        raw = data.to_json(orient="table", date_format="iso").encode()
        schedule = sessions(start, end, calendar="XNYS")
        records = []
        for timestamp, row in data.iterrows():
            session = timestamp.date().isoformat()
            if session not in schedule:
                raise ProviderError("yfinance", "non_session_bar")
            opened, closed = schedule[session]
            normalized = dict(instrument=instrument.model_dump(mode="json"), opened_at=opened.isoformat(),
                              closed_at=closed.isoformat(), **{k: str(row[k.title()]) for k in ("open", "high", "low", "close", "volume")},
                              price_basis="raw", market_type="equity", calendar="XNYS", session=session)
            MarketBar(evidence_id="validate", **normalized)
            records.append(store_observation(self.store, raw=raw, normalized=normalized, provider="yfinance",
                           version=f"{yf.__version__}:sdk-output-v1", source_uri=f"yfinance://history/{instrument.symbol}",
                           kind="bar", logical_key=f"{instrument.instrument_id}:1d:{session}", context=context,
                           ingested_at=ingested_at, observed_at=closed, instrument=instrument, adjustment_policy="raw"))
            for field, action in (("Stock Splits", "split"), ("Dividends", "dividend")):
                if row.get(field, 0):
                    records.append(store_observation(self.store, raw=raw,
                                   normalized={"type": action, "value": str(row[field]), "effective_session": session},
                                   provider="yfinance", version=f"{yf.__version__}:sdk-output-v1",
                                   source_uri=f"yfinance://history/{instrument.symbol}", kind="corporate_action",
                                   logical_key=f"{instrument.instrument_id}:{action}:{session}", context=context,
                                   ingested_at=ingested_at, observed_at=opened, instrument=instrument))
        return FetchResult("available", tuple(records), "raw SDK prices; historical revision availability unverified")
