import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.context import AlpacaReadAdapter
from src.data.adapters.fundamentals import FMPAdapter, FredAdapter
from src.data.adapters.market import CCXTAdapter
from src.data.evidence.service import EvidenceService
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext

T = datetime(2020, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def context():
    return RunContext(run_id="test", mode="backtest", as_of=T, config_hash="0"*64, code_version="test")


@pytest.fixture
def instrument():
    return Instrument(instrument_id="US:A", symbol="ABC", asset_class="equity", quote_currency="USD", venue="XNYS")


def transport(provider, payload=None, status=200, handler=None):
    client = httpx.Client(transport=httpx.MockTransport(handler or (lambda request: httpx.Response(status, json=payload))))
    return ReadHTTP(provider, "https://example.org", client=client, attempts=1, min_interval=0, sleep=lambda _: None)


def test_fmp_full_transcript_and_capture_availability(tmp_path, context, instrument):
    text = "Original full transcript. " * 1000
    payload = [{"symbol": "ABC", "year": 2019, "quarter": 4, "date": "2019-12-01", "content": text}]
    store = EvidenceStore(tmp_path / "data.sqlite")
    adapter = FMPAdapter(store, "private-test-key", http=transport("fmp", payload))
    result = adapter.fetch("transcript", instrument, context, ingested_at=T, year=2019, quarter=4)
    record = result.evidence[0].record
    assert record.available_at == T  # event date is not revision publication
    assert store.document(record.evidence_id, as_of=T).text == text
    assert json.loads(store.raw(record.evidence_id)) == payload
    assert "private-test-key" not in record.model_dump_json()
    bundle = EvidenceService(store).source_bundle("sentiment", instrument, context, max_chars=300)
    assert sum(len(p.text) for p in bundle["passages"]) <= 300
    assert bundle["coverage"]["news"].status == "missing"
    with pytest.raises(ValueError):
        store.document(record.evidence_id, as_of=T - timedelta(days=1))


@pytest.mark.parametrize("payload,status,code", [([], 200, None), ({"Error Message": "plan unavailable"}, 200, "provider_error"),
    (None, 401, "credentials"), (None, 403, "entitlement"), (None, 429, "rate_limited"), ({"unexpected": 3}, 200, "shape")])
def test_explicit_fmp_failures(tmp_path, context, instrument, payload, status, code):
    adapter = FMPAdapter(EvidenceStore(tmp_path / "db"), "sensitive-key", http=transport("fmp", payload, status))
    if code:
        with pytest.raises(ProviderError, match=code) as exc:
            adapter.fetch("transcript", instrument, context, ingested_at=T, year=2019, quarter=4)
        assert "sensitive-key" not in str(exc.value)
    else:
        assert adapter.fetch("transcript", instrument, context, ingested_at=T, year=2019, quarter=4).status == "missing"


def test_http_retry_bounds_and_request_budget():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503)
    http = transport("test", handler=handler)
    http.attempts = 3
    with pytest.raises(ProviderError, match="provider_unavailable"):
        http.get("/read")
    assert len(calls) == 3
    http.max_requests = 3
    with pytest.raises(ProviderError, match="budget"):
        http.get("/read")
    assert len(calls) == 3


def test_alfred_revisions_not_observation_dates(tmp_path, context):
    payload = {"count": 3, "observations": [
        {"date": "2019-10-01", "value": "100", "realtime_start": "2019-12-01", "realtime_end": "2019-12-31"},
        {"date": "2019-10-01", "value": "90", "realtime_start": "2020-01-01", "realtime_end": "9999-12-31"},
        {"date": "2019-11-01", "value": ".", "realtime_start": "2020-01-01", "realtime_end": "9999-12-31"}]}
    store = EvidenceStore(tmp_path / "db")
    adapter = FredAdapter(store, "key", http=transport("fred", payload))
    adapter.observations("GDP", context, ingested_at=T+timedelta(days=365), start="2019-01-01", end="2020-01-01")
    old = store.query(as_of=T, kind="macro", logical_key="GDP:2019-10-01")
    assert len(old.records) == 1  # Jan 1 vintage only safe after Eastern day end
    later = store.query(as_of=T+timedelta(hours=6), kind="macro", logical_key="GDP:2019-10-01")
    assert len(later.records) == 2  # both retained; callers choose an explicit vintage
    assert store.query(as_of=T+timedelta(hours=6), logical_key="GDP:2019-11-01").status == "ineligible"


def test_ccxt_spot_history_and_perpetual_data_separate(tmp_path, context):
    instrument = Instrument(instrument_id="btc:binance:spot", symbol="BTC/USDT", asset_class="crypto", quote_currency="USDT", venue="binance")
    exchange = SimpleNamespace(id="binance", load_markets=lambda: {"BTC/USDT": {"spot": True}},
                               parse_timeframe=lambda _: 86400,
                               fetch_ohlcv=Mock(return_value=[[int((T-timedelta(days=1)).timestamp()*1000), 100, 110, 95, 105, 50]]))
    store = EvidenceStore(tmp_path / "db")
    adapter = CCXTAdapter(store, exchange)
    result = adapter.bars(instrument, context, ingested_at=T, start=T-timedelta(days=1), end=T)
    assert result.status == "available"
    snapshot = EvidenceService(store).market_snapshot((instrument,), context, start=T-timedelta(days=1), market_type="spot")
    assert snapshot.bars[0].close == 105
    with pytest.raises(ProviderError, match="spot_derivative"):
        adapter.bars(instrument, context, ingested_at=T, start=T-timedelta(days=1), end=T, market_type="perpetual")
    with pytest.raises(ProviderError, match="perpetual_symbol"):
        adapter.derivatives(instrument, context, ingested_at=T, capability="funding", since=T)


def test_current_broker_reads_cannot_mutate(tmp_path, context):
    http = transport("alpaca", {})
    adapter = AlpacaReadAdapter(EvidenceStore(tmp_path / "db"), "key", "secret", http=http)
    for operation in ("cancel", "close_position", "submit_order", "patch_account", "get_then_delete"):
        with pytest.raises(ProviderError, match="not_permitted"):
            adapter.inspect(operation, context, ingested_at=T)
    assert http.requests == 0
