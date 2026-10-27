"""Actual factory/native preflight with synthetic wire sources; no vendor calls."""

import hashlib
import hmac
import json
import warnings
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest
import pytest_socket
from src.domain.portfolio import AccountSpec, FillPolicy, InstrumentRisk, OrderIntent
from src.execution.ledger import LedgerError
from src.execution.spot_venue import NativeSpotMarket
from src.execution.venue import VenueRules

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.operating import BinanceOperatingAdapter, OperatingCapture
from src.data.binance_execution import (
    BinanceEvidenceReferenceResolver,
    create_evidence_backed_binance_execution,
    freeze_binance_reference,
)
from src.data.binance_reference import parse_binance_notional_reference
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext

T = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
BTC = Instrument(
    instrument_id="spot:BTC-USDT",
    symbol="BTC/USDT",
    asset_class="crypto",
    quote_currency="USDT",
    venue="binance",
)


@pytest.fixture(autouse=True)
def unix_event_loop_only():
    # Replace pytest-socket's prior guard; nesting would still reject AF_UNIX.
    pytest_socket.enable_socket()
    pytest_socket.disable_socket(allow_unix_socket=True)
    import socket

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(pytest_socket.SocketBlockedError):
            socket.socket(socket.AF_INET)
    yield


def context(at=T, mode="paper"):
    return RunContext(
        run_id="source-bound-native",
        mode=mode,
        as_of=at,
        config_hash="a" * 64,
        code_version="fixture",
        account_id=mode + ":native",
    )


def capture(store, *, network="testnet", operation="referencePrice", body=None, at=T):
    origin = "https://testnet.binance.vision" if network == "testnet" else "https://api.binance.com"
    if body is None:
        body = {
            "symbol": "BTCUSDT",
            "referencePrice": "100.000000000000000001",
            "timestamp": int((at - timedelta(seconds=1)).timestamp() * 1000),
        }
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as client:
        http = ReadHTTP("binance", origin, client=client, min_interval=0)
        source, _ = OperatingCapture(store, clock=lambda: at).wire(
            http,
            "/api/v3/" + operation,
            context(at),
            upstream="binance-spot-v3",
            params={"symbol": "BTCUSDT"},
            instrument=BTC,
            timestamp_unit="MILLISECOND",
        )
    return source


def freeze(store, source, *, minutes=5, network="testnet", at=T, **kwargs):
    return freeze_binance_reference(
        store,
        BTC,
        context(at),
        reference_evidence_id=source.record.evidence_id,
        average_minutes=minutes,
        network=network,
        **kwargs,
    )


class Body(httpx.AsyncByteStream):
    def __init__(self, value):
        self.raw = json.dumps(value, separators=(",", ":")).encode()

    async def __aiter__(self):
        yield self.raw


class NativeWire:
    def __init__(self):
        self.now, self.calls, self.order = T, [], None
        self.filters = [
            {"filterType": "LOT_SIZE", "minQty": "1", "maxQty": "1000", "stepSize": "1"},
            {
                "filterType": "MIN_NOTIONAL",
                "minNotional": "5",
                "applyToMarket": True,
                "avgPriceMins": 5,
            },
        ]

    async def transport(self, request):
        assert request.url.host == "testnet.binance.vision"
        operation = request.url.path.rsplit("/", 1)[-1]
        raw = request.content.decode() if request.method == "POST" else request.url.query.decode()
        params = {key: values[0] for key, values in parse_qs(raw).items()}
        if operation != "exchangeInfo":
            assert request.headers["X-MBX-APIKEY"] == "fixture-key"
            assert (
                hmac.new(
                    b"fixture-secret", raw.split("&signature=")[0].encode(), hashlib.sha256
                ).hexdigest()
                == params["signature"]
            )
        self.calls.append((request.method, operation, params))
        if operation == "account":
            payload = {
                "uid": 42,
                "accountType": "SPOT",
                "canTrade": True,
                "balances": [
                    {"asset": "USDT", "free": "100000", "locked": "0"},
                    {"asset": "BTC", "free": "0", "locked": "0"},
                ],
            }
        elif operation == "openOrders":
            payload = [self.order] if self.order else []
        elif operation == "exchangeInfo":
            payload = {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "isSpotTradingAllowed": True,
                        "orderTypes": ["MARKET"],
                        "filters": self.filters,
                    }
                ],
                "exchangeFilters": [],
            }
        elif operation == "myFilters":
            payload = {"symbolFilters": self.filters, "exchangeFilters": [], "assetFilters": []}
        elif operation == "order" and request.method == "POST":
            self.order = {
                "symbol": "BTCUSDT",
                "orderId": 100,
                "orderListId": -1,
                "clientOrderId": params["newClientOrderId"],
                "price": "0",
                "origQty": params["quantity"],
                "executedQty": "0",
                "cummulativeQuoteQty": "0",
                "status": "NEW",
                "timeInForce": "GTC",
                "type": "MARKET",
                "side": params["side"],
                "stopPrice": "0",
            }
            payload = self.order
        elif operation == "order":
            payload = self.order
        elif operation == "myTrades":
            payload = []
        else:
            raise AssertionError(operation)
        return httpx.Response(200, stream=Body(payload))


@pytest.fixture
def case(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    source = capture(store)
    value = SimpleNamespace(
        store=store,
        source=source,
        binding=freeze(store, source),
        wire=NativeWire(),
        executions=[],
        root=tmp_path,
    )
    yield value
    for execution in reversed(value.executions):
        for adapter in execution.adapters.values():
            if not adapter.transport._client.is_closed:
                adapter.close()
        execution.ledger.close()


def factory(case, *, bindings=None, ctx=None, **kwargs):
    rule = VenueRules(
        venue="binance",
        instrument=BTC,
        symbol="BTCUSDT",
        available_at=T,
        quantity_step=Decimal(1),
        price_tick=Decimal(".01"),
        market_type="spot",
        active=True,
        time_in_force=("GTC",),
    )
    execution = create_evidence_backed_binance_execution(
        evidence_store=case.store,
        context=ctx or context(),
        reference_bindings=bindings or (case.binding,),
        ledger_path=case.root / "ledger.sqlite",
        account=AccountSpec(
            account_id="paper:native",
            mode="paper",
            base_currency="USDT",
            initial_cash=100000,
            opened_at=T,
        ),
        broker_account_id="42",
        api_key="fixture-key",
        api_secret="fixture-secret",
        markets=(
            NativeSpotMarket(
                instrument=BTC,
                symbol="BTCUSDT",
                native_id="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
            ),
        ),
        rules={BTC.instrument_id: rule},
        fill_policy=FillPolicy(crypto_cost_bps=0),
        client=httpx.AsyncClient(transport=httpx.MockTransport(case.wire.transport)),
        clock=lambda: case.wire.now,
        **kwargs,
    )
    case.executions.append(execution)
    return execution


def prepare(case, execution, *, reference_price="100"):
    adapter = execution.adapters["binance"]
    adapter.capture()
    assert execution.reconcile_balances(as_of=T)["reconciled"]
    order = OrderIntent(
        intent_id="actual-native",
        account_id=execution.account_id,
        run_id=context().run_id,
        instrument=BTC,
        side="buy",
        quantity=10,
        reference_price=reference_price,
        strategy_id="test",
        decision_at=T,
        eligible_at=T + timedelta(seconds=1),
    )
    execution.prepare(
        order,
        {
            BTC.instrument_id: InstrumentRisk(
                instrument=BTC, sector="crypto", adv_base=100000000, as_of=T, lot_size=1
            )
        },
        as_of=T,
    )
    return order


def test_exact_sources_are_pure_frozen_and_new_ambient_capture_cannot_replace_them(case):
    resolver = BinanceEvidenceReferenceResolver(case.store, context(), (case.binding,))
    first = resolver(BTC, T, 5)
    capture(
        case.store,
        body={"symbol": "BTCUSDT", "referencePrice": "999", "timestamp": int(T.timestamp() * 1000)},
        at=T + timedelta(seconds=1),
    )
    assert resolver(BTC, T + timedelta(seconds=1), 5) == first
    assert first.source_hash == case.source.content_hash()
    with pytest.raises(LedgerError, match="stale"):
        resolver(BTC, T + timedelta(seconds=31), 5)
    with pytest.raises(LedgerError, match="frozen_admission"):
        resolver(BTC, T - timedelta(microseconds=1), 5)
    with pytest.raises(LedgerError, match="instrument_window_missing"):
        resolver(BTC.model_copy(update={"symbol": "ETH/USDT"}), T, 5)


def test_binding_compares_real_envelope_and_never_accepts_hash_shaped_input(case):
    altered = case.binding.reference.model_copy(update={"logical_key": "manufactured"})
    with pytest.raises(ProviderError, match="envelope_differs"):
        BinanceEvidenceReferenceResolver(
            case.store, context(), (case.binding.model_copy(update={"reference": altered}),)
        )
    future = capture(case.store, at=T + timedelta(seconds=1))
    with pytest.raises(ValueError):
        freeze(case.store, future)


def test_mainnet_testnet_and_actual_account_context_are_not_interchangeable(case):
    with pytest.raises(ProviderError, match="source_identity_differs"):
        parse_binance_notional_reference(
            case.store,
            BTC,
            context(),
            reference_evidence_id=case.source.record.evidence_id,
            average_minutes=5,
        )
    main = capture(case.store, network="mainnet")
    bound = freeze(case.store, main, network="mainnet")
    with pytest.raises(ProviderError, match="account_network"):
        BinanceEvidenceReferenceResolver(case.store, context(), (bound,))
    with pytest.raises(ProviderError, match="actual_account"):
        factory(case, ctx=context().model_copy(update={"account_id": "paper:other"}))
    assert case.wire.calls == []
    adapter = BinanceOperatingAdapter(case.store, network="testnet")
    assert adapter.http.base_url == "https://testnet.binance.vision"
    adapter.http.close()


def test_each_native_window_uses_its_exact_frozen_binding_and_duplicates_reject(case):
    one = freeze(case.store, case.source, minutes=1)
    resolver = BinanceEvidenceReferenceResolver(case.store, context(), (case.binding, one))
    assert resolver(BTC, T, 1) == resolver(BTC, T, 5)
    with pytest.raises(LedgerError, match="instrument_window_missing"):
        resolver(BTC, T, 3)
    with pytest.raises(ProviderError, match="unique_operational"):
        BinanceEvidenceReferenceResolver(case.store, context(), (case.binding, case.binding))


def test_actual_native_factory_preflight_archive_and_restart_use_certified_parser(case):
    execution = factory(case)
    assert case.wire.calls == []  # Construction only reads immutable local proof.
    order = prepare(case, execution)
    case.wire.now = order.eligible_at
    assert execution.dispatch(order.intent_id, as_of=case.wire.now)["state"] == "new"
    manifest = json.loads(
        execution.ledger.db.execute(
            "SELECT payload FROM events WHERE id LIKE 'binance-reference-binding:%'"
        ).fetchone()[0]
    )
    assert manifest["bindings"][0]["reference"] == case.source.model_dump(mode="json")
    assert manifest["context"] == context().model_dump(mode="json")
    execution.adapters["binance"].close()
    execution.ledger.close()
    restarted = factory(case)
    assert restarted.dispatch(order.intent_id, as_of=case.wire.now)["state"] == "new"
    assert len([call for call in case.wire.calls if call[:2] == ("POST", "order")]) == 1
    assert restarted.ledger.fills(order.account_id) == []


@pytest.mark.parametrize(
    "reason", ["missing_window", "stale", "maximum", "source_corrupt", "store_unavailable"]
)
def test_real_native_preflight_denies_bad_proof_before_transmission(case, reason):
    execution = factory(case)
    if reason == "missing_window":
        case.wire.filters[1]["avgPriceMins"] = 1
    elif reason == "maximum":
        case.wire.filters.append(
            {
                "filterType": "NOTIONAL",
                "minNotional": "5",
                "maxNotional": "999",
                "applyMinToMarket": True,
                "applyMaxToMarket": True,
                "avgPriceMins": 5,
            }
        )
    order = prepare(case, execution, reference_price="1")
    case.wire.now = T + timedelta(seconds=31 if reason == "stale" else 1)
    if reason == "stale":
        execution.adapters["binance"].capture()  # Fresh balances cannot refresh old price evidence.
    if reason == "source_corrupt":
        with case.store._connect() as db:
            db.execute("DROP TRIGGER immutable_blob_update")
            db.execute(
                "UPDATE blobs SET payload=? WHERE hash=?", (b"{}", case.source.record.raw_hash)
            )
    if reason == "store_unavailable":
        case.store.path = case.root / "missing-parent" / "unavailable.sqlite"
    state = execution.dispatch(order.intent_id, as_of=case.wire.now)
    assert state["state"] == "rejected"
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert not any(call[:2] == ("POST", "order") for call in case.wire.calls)


@pytest.mark.parametrize("network", ["mainnet", "testnet"])
def test_direct_native_only_capture_has_actual_network_and_exact_average_sources(case, network):
    requests = []
    origin = "https://testnet.binance.vision" if network == "testnet" else "https://api.binance.com"

    def handler(request):
        requests.append(request)
        assert str(request.url).startswith(origin)
        assert dict(request.url.params) == {"symbol": "BTCUSDT"}
        body = (
            {"symbol": "BTCUSDT", "referencePrice": None, "timestamp": int(T.timestamp() * 1000)}
            if request.url.path.endswith("referencePrice")
            else {
                "mins": 5,
                "price": "99.999999999999999999",
                "closeTime": int(T.timestamp() * 1000),
            }
        )
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = BinanceOperatingAdapter(
            case.store,
            network=network,
            clock=lambda: T,
            http=ReadHTTP("binance", origin, min_interval=0, client=client),
        )
        result = adapter.notional_reference(BTC, context(), average_minutes=5)
    reference, average = result.evidence
    binding = freeze(
        case.store, reference, network=network, average_evidence_id=average.record.evidence_id
    )
    ctx = context(mode="paper" if network == "testnet" else "live")
    resolver = BinanceEvidenceReferenceResolver(case.store, ctx, (binding,))
    value = resolver(BTC, T, 5)
    assert value.kind == "weighted-average" and value.price == Decimal("99.999999999999999999")
    assert value.source_hash == average.content_hash()
    assert value.reference_price_absence_hash == reference.content_hash()
    assert result.status == "available" and len(requests) == 2


def test_native_independent_predicates_use_pinned_multiple_windows(case):
    one = freeze(case.store, case.source, minutes=1)
    execution = factory(case, bindings=(case.binding, one))
    case.wire.filters.append(
        {
            "filterType": "NOTIONAL",
            "minNotional": "1",
            "maxNotional": "100000",
            "applyMinToMarket": True,
            "applyMaxToMarket": True,
            "avgPriceMins": 1,
        }
    )
    order = prepare(case, execution)
    case.wire.now = order.eligible_at
    assert execution.dispatch(order.intent_id, as_of=case.wire.now)["state"] == "new"
    assert tuple(
        value.average_minutes
        for value in execution.adapters["binance"]._notional_reference.bindings
    ) == (1, 5)


def test_restart_rejects_changed_archived_binding_manifest_without_io(case):
    execution = factory(case)
    execution.ledger.db.execute(
        "UPDATE events SET payload='{}' WHERE id LIKE 'binance-reference-binding:%'"
    )
    with pytest.raises(LedgerError, match="conflicting event retry"):
        factory(case)
    assert case.wire.calls == []
