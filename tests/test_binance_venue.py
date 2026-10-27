"""Offline native wire fixtures; no authenticated exchange acceptance claimed."""

import hashlib
import hmac
import json
from datetime import timedelta
from decimal import Decimal as D
from urllib.parse import parse_qs

import httpx
import pytest

from src.domain.common import Instrument
from src.domain.portfolio import AccountSpec, FillPolicy, PriceMark
from src.execution.broker import create_typed_binance_execution
from src.execution.ccxt_transport import milliseconds
from src.execution.ledger import LedgerError
from src.execution.spot_venue import NativeSpotMarket
from src.execution.venue import VenueNotionalReference, VenueRules
from tests.test_alpaca_venue import Chunks
from tests.test_durable_accounting import T, intent, metadata

BTC = Instrument(
    instrument_id="spot:BTC-USDT",
    symbol="BTC/USDT",
    asset_class="crypto",
    quote_currency="USDT",
    venue="binance",
)
ADAPTERS = []


@pytest.fixture(autouse=True)
def close_transports():
    yield
    while ADAPTERS:
        ADAPTERS.pop().close()


class Wire:
    def __init__(self):
        self.now, self.order, self.trades, self.calls = T, None, [], []
        self.cash, self.held = D(100000), D(0)
        self.lost_ack = False
        self.missing = False
        self.reject = None
        self.other_orders = []
        self.filters = [
            {"filterType": "LOT_SIZE", "minQty": ".01", "maxQty": "1000", "stepSize": ".01"},
            {"filterType": "MARKET_LOT_SIZE", "minQty": ".01", "maxQty": "500", "stepSize": ".01"},
            {
                "filterType": "PRICE_FILTER",
                "minPrice": ".01",
                "maxPrice": "1000000",
                "tickSize": ".01",
            },
            {
                "filterType": "MIN_NOTIONAL",
                "minNotional": "1",
                "applyToMarket": True,
                "avgPriceMins": 5,
            },
        ]
        self.asset_filters = []

    async def transport(self, request):
        response = self.handle(request)
        return httpx.Response(response.status_code, stream=Chunks([response.content]))

    def handle(self, request):
        assert request.url.host == "testnet.binance.vision"
        path, method = request.url.path.rsplit("/", 1)[-1], request.method
        raw = request.content.decode() if method == "POST" else request.url.query.decode()
        parameters = {key: value[0] for key, value in parse_qs(raw).items()}
        if path != "exchangeInfo":
            assert request.headers["X-MBX-APIKEY"] == "fixture-key"
            unsigned = raw.split("&signature=")[0]
            assert (
                hmac.new(b"fixture-secret", unsigned.encode(), hashlib.sha256).hexdigest()
                == parameters["signature"]
            )
        self.calls.append((method, path, parameters))
        if path == "account":
            pending = (
                D(0)
                if not self.order
                or self.order["side"] != "BUY"
                or self.order["status"] not in ("NEW", "PARTIALLY_FILLED")
                else (D(self.order["origQty"]) - D(self.order["executedQty"])) * 100
            )
            return httpx.Response(
                200,
                json={
                    "uid": 42,
                    "accountType": "SPOT",
                    "canTrade": True,
                    "balances": [
                        {"asset": "USDT", "free": str(self.cash - pending), "locked": str(pending)},
                        {"asset": "BTC", "free": str(self.held), "locked": "0"},
                    ],
                },
            )
        if path == "openOrders":
            return httpx.Response(
                200,
                json=[
                    *(
                        [self.order]
                        if self.order and self.order["status"] in ("NEW", "PARTIALLY_FILLED")
                        else []
                    ),
                    *self.other_orders,
                ],
            )
        if path == "exchangeInfo":
            return httpx.Response(
                200,
                json={
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "baseAsset": "BTC",
                            "quoteAsset": "USDT",
                            "status": "TRADING",
                            "isSpotTradingAllowed": True,
                            "orderTypes": ["MARKET", "LIMIT", "STOP_LOSS"],
                            "filters": self.filters,
                        }
                    ],
                    "exchangeFilters": [],
                },
            )
        if path == "myFilters":
            return httpx.Response(
                200,
                json={
                    "symbolFilters": self.filters,
                    "exchangeFilters": [],
                    "assetFilters": self.asset_filters,
                },
            )
        if path == "order" and method == "POST":
            if self.reject:
                return httpx.Response(400, json={"code": self.reject, "msg": "fixture rejection"})
            self.order = {
                "symbol": parameters["symbol"],
                "orderId": 100 if self.order is None else self.order["orderId"] + 1,
                "orderListId": -1,
                "clientOrderId": parameters["newClientOrderId"],
                "price": parameters.get("price", "0"),
                "origQty": parameters["quantity"],
                "executedQty": "0",
                "cummulativeQuoteQty": "0",
                "status": "NEW",
                "timeInForce": parameters.get("timeInForce", "GTC"),
                "type": parameters["type"],
                "side": parameters["side"],
                "stopPrice": parameters.get("stopPrice", "0"),
            }
            if self.lost_ack:
                raise httpx.ReadTimeout("fixture acceptance with lost reply", request=request)
            return httpx.Response(200, json=self.order)
        if path == "order":
            if self.missing:
                return httpx.Response(400, json={"code": -2013, "msg": "fixture order absent"})
            if method == "DELETE":
                self.order["status"] = "CANCELED"
            return httpx.Response(200, json=self.order)
        if path == "myTrades":
            start = int(parameters["fromId"])
            return httpx.Response(
                200,
                json=[
                    trade
                    for trade in self.trades
                    if trade["id"] >= start and trade["orderId"] == int(parameters["orderId"])
                ][:1000],
            )
        raise AssertionError((path, method))

    def add(self, identity, quantity, price, charge, asset="USDT", seconds=2):
        cost = D(quantity) * D(price)
        self.trades.append(
            {
                "symbol": "BTCUSDT",
                "id": identity,
                "orderId": self.order["orderId"],
                "orderListId": -1,
                "price": price,
                "qty": quantity,
                "quoteQty": str(cost),
                "commission": charge,
                "commissionAsset": asset,
                "time": milliseconds(T + timedelta(seconds=seconds)),
                "isBuyer": self.order["side"] == "BUY",
                "isMaker": False,
                "isBestMatch": True,
            }
        )
        sign = 1 if self.order["side"] == "BUY" else -1
        self.cash -= sign * cost + (D(charge) if asset == "USDT" else D(0))
        self.held += sign * D(quantity) - (D(charge) if asset == "BTC" else D(0))
        trades = [trade for trade in self.trades if trade["orderId"] == self.order["orderId"]]
        executed = sum((D(trade["qty"]) for trade in trades), D(0))
        self.order.update(
            executedQty=str(executed),
            cummulativeQuoteQty=str(sum((D(trade["quoteQty"]) for trade in trades), D(0))),
            status="FILLED" if executed == D(self.order["origQty"]) else "PARTIALLY_FILLED",
        )


def setup(path, wire=None, **changes):
    wire = wire or Wire()
    rule = VenueRules(
        venue="binance",
        instrument=BTC,
        symbol="BTCUSDT",
        available_at=T,
        quantity_step=D(".01"),
        price_tick=D(".01"),
        market_type="spot",
        active=True,
        time_in_force=("GTC", "IOC", "FOK"),
    )

    def reference(instrument, cutoff, minutes):
        assert instrument == BTC and minutes == 5
        return VenueNotionalReference(
            price=D(100),
            kind="venue-reference",
            average_minutes=None,
            observed_at=wire.now,
            available_at=wire.now,
            source_hash="a" * 64,
            reference_price_absence_hash=None,
        )

    execution = create_typed_binance_execution(
        ledger_path=path / "binance.sqlite",
        account=AccountSpec(
            account_id="paper:binance",
            mode="paper",
            base_currency="USDT",
            initial_cash=D(100000),
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
        notional_reference=reference,
        fill_policy=FillPolicy(crypto_cost_bps=0),
        client=changes.pop("client", None)
        or httpx.AsyncClient(transport=httpx.MockTransport(wire.transport)),
        clock=lambda: wire.now,
        **changes,
    )
    adapter = execution.adapters["binance"]
    ADAPTERS.append(adapter)
    order = intent(instrument=BTC).model_copy(update={"account_id": execution.account_id})
    return wire, execution, adapter, order


def admit(wire, execution, adapter, order):
    adapter.capture()
    assert execution.reconcile_balances(as_of=wire.now)["reconciled"]
    execution.prepare(order, {BTC.instrument_id: metadata(instrument=BTC)}, as_of=T)
    wire.now = order.eligible_at
    return execution.dispatch(order.intent_id, as_of=wire.now)


def test_actual_native_partial_fills_asset_fee_losing_nav_and_restart(tmp_path):
    wire, execution, adapter, order = setup(tmp_path)
    assert admit(wire, execution, adapter, order)["state"] == "new"
    wire.add(1, "3", "100", ".003", "BTC")
    wire.now = T + timedelta(seconds=3)
    assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
    first = execution.ledger.fills(order.account_id)[0]
    assert first.quantity == 3 and first.inventory_fees[0].quantity == D(".003")
    assert execution.ledger.quantity(order.account_id, BTC.instrument_id) == D("2.997")
    # New HTTP receipt, identical native trade: retain the first exact fill proof.
    wire.now += timedelta(seconds=1)
    execution.reconcile_all(as_of=wire.now)
    assert execution.ledger.fills(order.account_id) == [first]
    adapter.close()
    ADAPTERS.remove(adapter)
    execution.ledger.close()
    wire, execution, adapter, order = setup(tmp_path, wire)
    wire.add(2, "7", "90", ".63", seconds=5)
    wire.now = T + timedelta(seconds=6)
    assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    adapter.capture()
    assert execution.reconcile_balances(as_of=wire.now)["reconciled"]
    execution.ledger.mark(
        order.account_id, [PriceMark(instrument=BTC, price=D(90), as_of=wire.now)]
    )
    final = execution.ledger.snapshot(order.account_id, wire.now, 0)
    assert final.cash == D("99069.37") and final.nav == D("99969.10") and final.fees == D(".93")
    assert final.positions[0].quantity == D("9.997")
    assert execution.ledger.reconcile(order.account_id)["reconciled"]
    dump = "\n".join(row[0] for row in execution.ledger.db.execute("SELECT payload FROM events"))
    assert "fixture-key" not in dump and "fixture-secret" not in dump and "signature=" not in dump


def test_acknowledged_quantity_without_trade_coverage_never_manufactures_fill(tmp_path):
    wire, execution, adapter, order = setup(tmp_path)
    admit(wire, execution, adapter, order)
    wire.now = T + timedelta(seconds=3)
    wire.order.update(status="FILLED", executedQty="10", cummulativeQuoteQty="1000")
    assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "reconciliation_required"
    assert execution.ledger.fills(order.account_id) == []
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000


def test_lost_ack_restart_and_missing_lookup_never_resubmit(tmp_path):
    wire = Wire()
    wire.lost_ack = True
    wire, execution, adapter, order = setup(tmp_path, wire)
    assert admit(wire, execution, adapter, order)["state"] == "ambiguous"
    wire.missing = True
    execution.reconcile_all(as_of=wire.now)
    execution.dispatch(order.intent_id, as_of=wire.now)
    assert len([row for row in wire.calls if row[:2] == ("POST", "order")]) == 1
    adapter.close()
    ADAPTERS.remove(adapter)
    execution.ledger.close()
    wire.missing = False
    wire, execution, adapter, order = setup(tmp_path, wire)
    assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "new"
    execution.cancel(order.intent_id, as_of=wire.now)
    assert execution.state(order.intent_id)["state"] == "cancelled"
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert len([row for row in wire.calls if row[:2] == ("POST", "order")]) == 1


@pytest.mark.parametrize(
    "native",
    [
        {
            "filterType": "NOTIONAL",
            "minNotional": "1",
            "maxNotional": "999",
            "applyMinToMarket": True,
            "applyMaxToMarket": True,
            "avgPriceMins": 5,
        },
        {"filterType": "MARKET_LOT_SIZE", "minQty": ".01", "maxQty": "9", "stepSize": ".01"},
        {"filterType": "UNKNOWN_NEW_RULE", "value": "1"},
        {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 0},
    ],
)
def test_native_constraints_cannot_be_bypassed_by_weaker_supplied_rules(tmp_path, native):
    wire = Wire()
    wire.filters.append(native)
    wire, execution, adapter, order = setup(tmp_path, wire)
    assert admit(wire, execution, adapter, order)["state"] == "rejected"
    assert not any(row[:2] == ("POST", "order") for row in wire.calls)
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0


def test_exact_native_trade_identity_change_is_retained_but_not_booked(tmp_path):
    wire, execution, adapter, order = setup(tmp_path)
    admit(wire, execution, adapter, order)
    wire.add(1, "3", "100", ".3")
    wire.now = T + timedelta(seconds=3)
    execution.reconcile_all(as_of=wire.now)
    wire.trades[0]["commission"] = ".4"
    wire.now += timedelta(seconds=1)
    with pytest.raises(LedgerError, match="identity changed"):
        adapter.lookup(order, execution.state(order.intent_id)["client_id"], "100", wire.now)
    assert execution.ledger.reconcile(order.account_id)["fees"] == "0.3"


def test_partial_cancel_then_limit_sell_to_zero_reconciles_actual_cash(tmp_path):
    wire, execution, adapter, buy = setup(tmp_path)
    admit(wire, execution, adapter, buy)
    wire.add(1, "3", "100", ".3")
    wire.now = T + timedelta(seconds=3)
    execution.cancel(buy.intent_id, as_of=wire.now)
    assert execution.state(buy.intent_id)["state"] == "cancelled"
    assert execution.ledger.quantity(buy.account_id, BTC.instrument_id) == 3
    assert execution.ledger.order(buy.account_id, buy.intent_id)["reserved"] == 0
    adapter.capture()
    execution.reconcile_balances(as_of=wire.now)
    execution.ledger.mark(buy.account_id, [PriceMark(instrument=BTC, price=D(80), as_of=wire.now)])
    sell = intent("sell", "sell", "3", "80", wire.now, instrument=BTC).model_copy(
        update={
            "account_id": buy.account_id,
            "order_type": "limit",
            "limit_price": D(80),
            "tif": "IOC",
        }
    )
    execution.prepare(sell, {BTC.instrument_id: metadata(wire.now, BTC)}, as_of=wire.now)
    wire.now = sell.eligible_at
    assert execution.dispatch(sell.intent_id, as_of=wire.now)["state"] == "new"
    wire.add(2, "3", "80", ".24", seconds=5)
    wire.now += timedelta(seconds=2)
    execution.reconcile_all(as_of=wire.now)
    adapter.capture()
    assert execution.reconcile_balances(as_of=wire.now)["reconciled"]
    result = execution.ledger.snapshot(buy.account_id, wire.now, 0)
    assert result.cash == result.nav == D("99939.46")
    assert result.positions == () and result.realized_pnl == D("-60.54")
    assert execution.ledger.reconcile(buy.account_id)["reconciled"]


@pytest.mark.parametrize(
    "code,state,reserved", [(-1013, "rejected", 0), (-2010, "ambiguous", 1000)]
)
def test_only_definitive_native_rejection_releases_reservation(tmp_path, code, state, reserved):
    wire = Wire()
    wire.reject = code
    wire, execution, adapter, order = setup(tmp_path, wire)
    assert admit(wire, execution, adapter, order)["state"] == state
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == reserved
    assert len([row for row in wire.calls if row[:2] == ("POST", "order")]) == 1


def test_original_retained_fill_cannot_be_changed_independently_of_native_source(tmp_path):
    wire, execution, adapter, order = setup(tmp_path)
    admit(wire, execution, adapter, order)
    wire.add(1, "3", "100", ".3")
    wire.now = T + timedelta(seconds=3)
    execution.reconcile_all(as_of=wire.now)
    row = execution.ledger.db.execute(
        "SELECT id,payload FROM events WHERE id LIKE 'binance-native-trade:%'"
    ).fetchone()
    payload = json.loads(row["payload"])
    payload["fill"]["fee"] = "0.01"
    execution.ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?", (json.dumps(payload), row["id"])
    )
    with pytest.raises(LedgerError, match="retained native fill differs"):
        adapter.lookup(order, execution.state(order.intent_id)["client_id"], "100", wire.now)
    assert execution.ledger.reconcile(order.account_id)["fees"] == "0.3"


def test_raw_capture_changed_after_restart_fails_before_new_decision(tmp_path):
    wire, execution, adapter, _ = setup(tmp_path)
    adapter.capture()
    row = execution.ledger.db.execute(
        "SELECT id,payload FROM events WHERE id LIKE 'binance-http:%' LIMIT 1"
    ).fetchone()
    value = json.loads(row["payload"])
    value["status"] = 201
    execution.ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?", (json.dumps(value), row["id"])
    )
    with pytest.raises(LedgerError, match="receipt changed"):
        adapter.balances(wire.now)


def test_unknown_remote_orders_are_a_reconciliation_gate(tmp_path):
    wire = Wire()
    wire.other_orders = [{"clientOrderId": "unowned-native-order"}]
    wire, execution, adapter, _ = setup(tmp_path, wire)
    adapter.capture()
    with pytest.raises(LedgerError, match="unknown|unowned|open order"):
        execution.reconcile_balances(as_of=wire.now)


def test_stale_native_reference_never_reaches_submission(tmp_path):
    wire, execution, adapter, order = setup(tmp_path)
    original = adapter._notional_reference
    adapter._notional_reference = lambda instrument, at, minutes: original(
        instrument, at, minutes
    ).model_copy(update={"observed_at": T - timedelta(seconds=31), "available_at": T})
    assert admit(wire, execution, adapter, order)["state"] == "rejected"
    assert not any(row[:2] == ("POST", "order") for row in wire.calls)


def test_reported_cost_disagreement_never_books_success(tmp_path):
    wire, execution, adapter, order = setup(tmp_path)
    admit(wire, execution, adapter, order)
    wire.add(1, "10", "100", "1")
    wire.order["cummulativeQuoteQty"] = "999"
    wire.now = T + timedelta(seconds=3)
    execution.reconcile_all(as_of=wire.now)
    assert execution.state(order.intent_id)["state"] == "ambiguous"
    assert not execution.ledger.fills(order.account_id)
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
