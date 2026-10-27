"""Data f5a1482 integration over actual stored sources; native/runtime fixtures only."""

import copy
import hashlib
import json
from datetime import timedelta
from decimal import Decimal as D

import httpx
import pytest

pytest.importorskip(
    "src.data.binance_execution", reason="requires integrated data f5a1482 contract"
)

from src.data.adapters.base import ReadHTTP
from src.data.adapters.operating import OperatingCapture
from src.data.binance_execution import freeze_binance_reference
from src.data.evidence.store import EvidenceStore

from src.domain.common import RunContext
from src.domain.portfolio import AccountSpec, FillPolicy
from src.execution.binance_venue import BinanceVenue
from src.execution.broker import create_typed_lean_execution
from src.execution.ccxt_transport import milliseconds
from src.execution.lean_references import bind_binance_reference, resolve_binance_reference
from src.execution.lean_venue import LeanDeployment, LeanInstrument
from src.execution.ledger import LedgerError
from src.execution.spot_venue import NativeSpotMarket
from src.execution.venue import VenueRules
from tests.test_binance_venue import BTC, Wire
from tests.test_durable_accounting import T, intent, metadata
from tests.test_lean_execution import RuntimeFixture


class LiveWire(Wire):
    def handle(self, request):
        assert request.url.host == "api.binance.com"
        request.url = request.url.copy_with(host="testnet.binance.vision")
        return super().handle(request)


class Runtime(RuntimeFixture):
    def identity(self):
        result = super().identity()
        result["native_credential_identity"] = hashlib.sha256(b"fixture-key").hexdigest()
        return result

    def capture(self):
        result = super().capture()
        result["account_currency"] = "USDT"
        result["cash_book"]["USDT"] = result["cash_book"].pop("USD")
        for order in result["open_orders"]:
            order["broker_ids"] = ["100"]
        return result

    def submit(self, order, client):
        super().submit(order, client)
        self.ticket["broker_ids"] = ["100"]
        self.wire.order = {
            "symbol": "BTCUSDT",
            "orderId": 100,
            "orderListId": -1,
            "clientOrderId": "actual-lean-native-client",
            "price": "0",
            "origQty": "10",
            "executedQty": "0",
            "cummulativeQuoteQty": "0",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "BUY",
            "stopPrice": "0",
        }
        return copy.deepcopy(self.ticket)


def capture(store, context, *, price="100", network="mainnet"):
    origin = "https://api.binance.com" if network == "mainnet" else "https://testnet.binance.vision"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "symbol": "BTCUSDT",
                    "referencePrice": price,
                    "timestamp": milliseconds(context.as_of),
                },
            )
        )
    ) as client:
        source, _ = OperatingCapture(store, clock=lambda: context.as_of).wire(
            ReadHTTP("binance", origin, client=client, min_interval=0),
            "/api/v3/referencePrice",
            context,
            upstream="binance-spot-v3",
            params={"symbol": "BTCUSDT"},
            instrument=BTC,
            timestamp_unit="MILLISECOND",
        )
    return freeze_binance_reference(
        store,
        BTC,
        context,
        reference_evidence_id=source.record.evidence_id,
        average_minutes=5,
        network=network,
    )


@pytest.fixture
def case(tmp_path):
    account = AccountSpec(
        account_id="live:lean-binance",
        mode="live",
        base_currency="USDT",
        initial_cash=D(100000),
        opened_at=T,
    )
    deployment = LeanDeployment(
        deployment_id="fixture-native",
        account_id=account.account_id,
        broker_account_id="42",
        brokerage="BinanceBrokerage",
        mode="live",
        instruments=(
            LeanInstrument(instrument=BTC, ticker="BTCUSDT", market="binance", base_currency="BTC"),
        ),
    )
    runtime, store = Runtime(deployment), EvidenceStore(tmp_path / "evidence.sqlite")
    rule = VenueRules(
        venue="lean",
        instrument=BTC,
        symbol="BTCUSDT",
        available_at=T,
        quantity_step=D(".01"),
        price_tick=D(".01"),
        market_type="spot",
        active=True,
    )
    execution = create_typed_lean_execution(
        ledger_path=tmp_path / "ledger.sqlite",
        account=account,
        deployment=deployment,
        rules={BTC.instrument_id: rule},
        runtime=runtime,
        evidence_store=store,
        allow_live=True,
        fill_policy=FillPolicy(crypto_cost_bps=0),
    )
    wire = LiveWire()
    runtime.wire = wire
    native = BinanceVenue(
        execution.ledger,
        account.account_id,
        broker_account_id="42",
        markets=(
            NativeSpotMarket(
                instrument=BTC,
                symbol="BTCUSDT",
                native_id="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
            ),
        ),
        rules={BTC.instrument_id: rule.model_copy(update={"venue": "binance"})},
        api_key="fixture-key",
        api_secret="fixture-secret",
        allow_live=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(wire.transport)),
        clock=runtime.now,
    )
    adapter = execution.adapters["lean"]
    adapter.native_reader = native
    order = intent(instrument=BTC).model_copy(update={"account_id": account.account_id})
    context = RunContext(
        run_id=order.run_id,
        mode="live",
        account_id=account.account_id,
        as_of=T,
        config_hash="a" * 64,
        code_version="offline-fixture",
    )
    adapter.capture()
    execution.prepare(order, {BTC.instrument_id: metadata(T, BTC)}, as_of=T)
    yield execution, adapter, native, runtime, store, context, order
    native.close()
    execution.ledger.close()


def test_exact_frozen_binding_reconstructs_after_restart_and_preflights_native_filter(case):
    execution, adapter, native, runtime, store, context, order = case
    binding = capture(store, context, price="100.000000000000000001")
    key = bind_binance_reference(
        execution, order.intent_id, evidence_store=store, context=context, bindings=(binding,)
    )
    assert len(key) == 64
    # A newer, cheaper source is deliberately irrelevant to this admitted intent.
    capture(store, context.model_copy(update={"as_of": T + timedelta(seconds=1)}), price=".01")
    adapter.evidence_store = EvidenceStore(store.path)
    runtime.at = order.eligible_at
    value = resolve_binance_reference(adapter, order, as_of=runtime.at)(BTC, runtime.at, 5)
    assert value.price == D("100.000000000000000001")
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "new"
    assert native._notional_reference is None  # The override never leaks to another intent.
    assert (
        bind_binance_reference(
            execution, order.intent_id, evidence_store=store, context=context, bindings=(binding,)
        )
        == key
    )
    assert len(runtime.calls) == 1
    assert not any(method == "POST" for method, *_ in runtime.wire.calls)


@pytest.mark.parametrize("problem", ["missing", "expired", "tampered"])
def test_missing_stale_or_changed_reference_rejects_before_runtime_submit(case, problem):
    execution, adapter, _, runtime, store, context, order = case
    if problem != "missing":
        binding = capture(store, context)
        bind_binance_reference(
            execution, order.intent_id, evidence_store=store, context=context, bindings=(binding,)
        )
    runtime.at = order.eligible_at
    if problem == "expired":
        runtime.at += timedelta(seconds=35)
        adapter.capture()  # Fresh native funds cannot freshen an old reference.
    if problem == "tampered":
        key = "lean-binance-reference:" + order.intent_id
        payload = json.loads(
            execution.ledger.db.execute("SELECT payload FROM events WHERE id=?", (key,)).fetchone()[
                0
            ]
        )
        payload["manifest"]["context"]["run_id"] = "other-run"
        execution.ledger.db.execute(
            "UPDATE events SET payload=? WHERE id=?", (json.dumps(payload), key)
        )
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "rejected"
    assert not runtime.calls
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0


@pytest.mark.parametrize(
    "field,value",
    [("run_id", "other-run"), ("account_id", "live:another"), ("as_of", T + timedelta(seconds=1))],
)
def test_context_identity_cannot_cross_intents(case, field, value):
    execution, _, _, _, store, context, order = case
    binding = capture(store, context)
    with pytest.raises(LedgerError, match="exact admitted intent"):
        bind_binance_reference(
            execution,
            order.intent_id,
            evidence_store=store,
            context=context.model_copy(update={field: value}),
            bindings=(binding,),
        )


def test_binding_cannot_be_replaced_with_another_actual_source(case):
    execution, _, _, _, store, context, order = case
    first, second = capture(store, context), capture(store, context, price="101")
    bind_binance_reference(
        execution, order.intent_id, evidence_store=store, context=context, bindings=(first,)
    )
    with pytest.raises(LedgerError, match="conflicting event retry"):
        bind_binance_reference(
            execution, order.intent_id, evidence_store=store, context=context, bindings=(second,)
        )


def test_native_filter_uses_actual_reference_instead_of_intent_price(case):
    execution, adapter, _, runtime, store, context, order = case
    binding = capture(store, context, price="1")
    bind_binance_reference(
        execution, order.intent_id, evidence_store=store, context=context, bindings=(binding,)
    )
    runtime.wire.filters[-1]["minNotional"] = "100"
    runtime.at = order.eligible_at
    adapter.capture()
    assert order.reference_price == 100
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "rejected"
    assert not runtime.calls
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0


def test_live_intent_cannot_bind_testnet_authority(case):
    execution, _, _, _, store, context, order = case
    binding = capture(store, context, network="testnet")
    with pytest.raises(LedgerError, match="certified native reference denied"):
        bind_binance_reference(
            execution, order.intent_id, evidence_store=store, context=context, bindings=(binding,)
        )


def test_no_new_binding_after_attempt_was_definitively_rejected(case):
    execution, _, _, runtime, store, context, order = case
    runtime.at = order.eligible_at
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "rejected"
    binding = capture(store, context)
    with pytest.raises(LedgerError, match="before the original submit attempt"):
        bind_binance_reference(
            execution, order.intent_id, evidence_store=store, context=context, bindings=(binding,)
        )
