"""Offline native Kraken protocol fixtures; no authenticated trading acceptance."""

import base64
import json
from datetime import timedelta
from decimal import Decimal as D
from urllib.parse import parse_qs

import httpx
import pytest

from src.domain.common import Instrument
from src.domain.portfolio import AccountSpec, FillPolicy, PriceMark
from src.execution.broker import create_typed_kraken_execution
from src.execution.ccxt_transport import milliseconds
from src.execution.ledger import LedgerError
from src.execution.spot_venue import NativeSpotMarket
from src.execution.venue import VenueRules
from tests.test_alpaca_venue import Chunks
from tests.test_durable_accounting import T, intent, metadata

BTC = Instrument(
    instrument_id="kraken:BTC-USD",
    symbol="BTC/USD",
    asset_class="crypto",
    quote_currency="USD",
    venue="kraken",
)
ADAPTERS = []


@pytest.fixture(autouse=True)
def cleanup():
    yield
    while ADAPTERS:
        adapter = ADAPTERS.pop()
        adapter.close()


class Wire:
    def __init__(self):
        self.now, self.order = T, None
        self.cash, self.held = D(100000), D(0)
        self.trades, self.ledgers, self.calls = {}, {}, []
        self.reject, self.timeout, self.missing = None, False, False
        self.empty_trades, self.empty_ledgers = False, False
        self.wallet = "WX6V-JUKW-KKPB-QE36"
        self.cursor_repeat = False
        self.credit = "0"
        self.pair = {
            "altname": "XBTUSD",
            "wsname": "XBT/USD",
            "base": "XXBT",
            "quote": "ZUSD",
            "aclass_base": "currency",
            "aclass_quote": "currency",
            "lot": "unit",
            "lot_multiplier": 1,
            "lot_decimals": 8,
            "ordermin": ".0001",
            "costmin": ".5",
            "tick_size": ".01",
            "status": "online",
        }

    async def transport(self, request):
        assert request.url.host == "api.kraken.com"
        path = request.url.path.rsplit("/", 1)[1]
        params = {
            key: value[0]
            for key, value in parse_qs(
                request.content.decode() if request.method == "POST" else request.url.query.decode()
            ).items()
        }
        self.calls.append((path, params))
        response = self.handle(path, params, request)
        # Native timestamps and balances are genuine JSON numbers here. The
        # adapter must parse their exact original decimal lexemes, not floats.
        body = json.dumps(response).encode().replace(b'"1704204002.125"', b"1704204002.125")
        return httpx.Response(200, stream=Chunks([body]))

    def handle(self, path, params, request):
        data = {}
        if path == "ListWalletAccounts":
            data = {
                "accounts": [
                    {
                        "account_id": self.wallet,
                        "type": "main",
                        "status": "active",
                        "flags": {"active": True, "user_defined": False},
                    }
                ],
                "cursor": {"next": None},
            }
        elif path == "BalanceEx":
            held = (
                (D(self.order["vol"]) - D(self.order["vol_exec"])) * 100
                if self.order
                and self.order["status"] == "open"
                and self.order["descr"]["type"] == "buy"
                else D(0)
            )
            data = {
                "ZUSD": {"balance": str(self.cash), "hold_trade": str(held), "credit": self.credit},
                "XXBT": {"balance": str(self.held), "hold_trade": "0"},
            }
        elif path == "Assets":
            data = {
                "ZUSD": {"altname": "USD", "aclass": "currency"},
                "XXBT": {"altname": "XBT", "aclass": "currency"},
            }
        elif path == "AssetPairs":
            data = {"XXBTZUSD": self.pair}
        elif path == "Trades":
            data = {
                "XXBTZUSD": [["100", "1", str(D(milliseconds(self.now)) / 1000), "b", "m", "", 1]],
                "last": "native-cursor",
            }
        elif path in ("OpenOrders", "ClosedOrders"):
            field, expected = ("open", "open") if path == "OpenOrders" else ("closed", "closed")
            matches = (
                self.order
                and (
                    (self.order["status"] == expected)
                    or field == "closed"
                    and self.order["status"] in ("canceled", "expired")
                )
                and not self.missing
            )
            if (
                matches
                and params.get("cl_ord_id")
                and params["cl_ord_id"] != self.order["cl_ord_id"]
            ):
                matches = False
            data = {
                field: {"O-FIXTURE": self.order} if matches else {},
                "cursor": {"next": "repeat" if self.cursor_repeat else None},
            }
        elif path == "AddOrder":
            if self.reject:
                return {"error": [self.reject]}
            self.order = {
                "cl_ord_id": params["cl_ord_id"],
                "status": "open",
                "opentm": str(D(milliseconds(self.now)) / 1000),
                "expiretm": 0,
                "descr": {
                    "pair": "XBTUSD",
                    "type": params["type"],
                    "ordertype": params["ordertype"],
                    "price": params.get("price", "0"),
                    "leverage": "none",
                    "close": "",
                },
                "vol": params["volume"],
                "vol_exec": "0",
                "cost": "0",
                "fee": "0",
                "price": "0",
                "oflags": "fciq",
                "trades": [],
            }
            if self.timeout:
                raise httpx.ReadTimeout(
                    "fixture accepted order but lost acknowledgement", request=request
                )
            data = {"txid": ["O-FIXTURE"], "descr": {"order": "native description"}}
        elif path == "QueryOrders":
            if self.missing:
                return {"error": ["EOrder:Unknown order"]}
            assert params["txid"] == "O-FIXTURE"
            if "cl_ord_id" in params:
                assert params["cl_ord_id"] == self.order["cl_ord_id"]
            data = {"O-FIXTURE": self.order}
        elif path == "QueryTrades":
            assert params["ledgers"].lower() == "true"
            data = (
                {}
                if self.empty_trades
                else {
                    key: self.trades[key] for key in params["txid"].split(",") if key in self.trades
                }
            )
        elif path == "QueryLedgers":
            data = (
                {}
                if self.empty_ledgers
                else {
                    key: self.ledgers[key] for key in params["id"].split(",") if key in self.ledgers
                }
            )
        elif path == "CancelOrder":
            self.order["status"] = "canceled"
            data = {"count": 1}
        else:
            raise AssertionError(path)
        return {"error": [], "result": data}

    def add(self, key, qty, price, fee, *, fee_asset="ZUSD", seconds="2.125"):
        cost = D(qty) * D(price)
        at = D(milliseconds(T)) / 1000 + D(seconds)
        sign = D(1) if self.order["descr"]["type"] == "buy" else D(-1)
        self.cash -= sign * cost + (D(fee) if fee_asset == "ZUSD" else 0)
        self.held += sign * D(qty) - (D(fee) if fee_asset == "XXBT" else 0)
        quoted = D(fee) if fee_asset == "ZUSD" else D(fee) * D(price)
        self.trades[key] = {
            "ordertxid": "O-FIXTURE",
            "pair": "XXBTZUSD",
            "time": str(at),
            "type": self.order["descr"]["type"],
            "ordertype": self.order["descr"]["ordertype"],
            "price": price,
            "cost": str(cost),
            "fee": str(quoted),
            "vol": qty,
            "margin": "0",
            "ledgers": [key + "-base", key + "-quote"],
        }
        for asset, amount, suffix in (
            ("XXBT", sign * D(qty), "base"),
            ("ZUSD", -sign * cost, "quote"),
        ):
            self.ledgers[key + "-" + suffix] = {
                "refid": key,
                "time": str(at),
                "type": "trade",
                "subtype": "",
                "aclass": "currency",
                "asset": asset,
                "amount": str(amount),
                "fee": fee if fee_asset == asset else "0",
                "balance": str(self.cash if asset == "ZUSD" else self.held),
            }
        self.order["trades"].append(key)
        self.order["vol_exec"] = str(sum((D(trade["vol"]) for trade in self.trades.values()), D(0)))
        self.order["cost"] = str(sum((D(trade["cost"]) for trade in self.trades.values()), D(0)))
        self.order["fee"] = str(sum((D(trade["fee"]) for trade in self.trades.values()), D(0)))
        if D(self.order["vol_exec"]) == D(self.order["vol"]):
            self.order["status"] = "closed"


def setup(tmp_path, wire=None, **options):
    wire = wire or Wire()
    account = options.pop(
        "account",
        AccountSpec(
            account_id="live:offline-kraken-fixture",
            mode="live",
            base_currency="USD",
            initial_cash=D(100000),
            opened_at=T,
        ),
    )
    engine = create_typed_kraken_execution(
        ledger_path=tmp_path / "kraken.sqlite",
        account=account,
        broker_account_id="WX6V-JUKW-KKPB-QE36",
        api_key="fixture-key",
        api_secret=base64.b64encode(b"fixture-secret").decode(),
        markets=(
            NativeSpotMarket(
                instrument=BTC,
                symbol="XBTUSD",
                native_id="XXBTZUSD",
                base_asset="XXBT",
                quote_asset="ZUSD",
            ),
        ),
        rules={
            BTC.instrument_id: VenueRules(
                venue="kraken",
                instrument=BTC,
                symbol="XBTUSD",
                available_at=T,
                quantity_step=D(".00000001"),
                price_tick=D(".01"),
                market_type="spot",
                active=True,
                time_in_force=("GTC", "IOC", "FOK"),
            )
        },
        fill_policy=FillPolicy(crypto_cost_bps=0),
        allow_live=options.pop("allow_live", True),
        client=httpx.AsyncClient(transport=httpx.MockTransport(wire.transport)),
        clock=lambda: wire.now,
        **options,
    )
    adapter = engine.adapters["kraken"]
    ADAPTERS.append(adapter)
    return (
        wire,
        engine,
        adapter,
        intent(instrument=BTC).model_copy(update={"account_id": engine.account_id}),
    )


def admit(wire, engine, adapter, order):
    adapter.capture()
    engine.reconcile_balances(as_of=wire.now)
    engine.prepare(order, {BTC.instrument_id: metadata(wire.now, BTC)}, as_of=wire.now)
    wire.now = order.eligible_at
    return engine.dispatch(order.intent_id, as_of=wire.now)


def test_actual_partial_base_fee_restart_and_loss_use_posted_ledger_assets(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    assert admit(wire, engine, adapter, order)["state"] == "new"
    wire.add("T1", "3", "100", ".003", fee_asset="XXBT")
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
    first = engine.ledger.fills(order.account_id)[0]
    assert first.executed_at == T + timedelta(seconds=2, microseconds=125000)
    assert first.fee == 0 and first.inventory_fees[0].quantity == D(".003")
    wire.now += timedelta(seconds=1)
    engine.reconcile_all(as_of=wire.now)
    assert engine.ledger.fills(order.account_id) == [first]
    adapter.close()
    ADAPTERS.remove(adapter)
    engine.ledger.close()
    wire, engine, adapter, order = setup(tmp_path, wire)
    wire.add("T2", "7", "90", ".63", seconds="5.2")
    wire.now = T + timedelta(seconds=6)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    engine.ledger.mark(order.account_id, [PriceMark(instrument=BTC, price=D(90), as_of=wire.now)])
    final = engine.ledger.snapshot(order.account_id, wire.now, 0)
    assert final.cash == D("99069.37") and final.nav == D("99969.10") and final.fees == D(".93")
    assert engine.ledger.reconcile(order.account_id)["reconciled"]


def test_lost_ack_recovers_closed_order_without_resubmitting(tmp_path):
    wire = Wire()
    wire.timeout = True
    wire, engine, adapter, order = setup(tmp_path, wire)
    assert admit(wire, engine, adapter, order)["state"] == "ambiguous"
    wire.missing = True
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
    engine.dispatch(order.intent_id, as_of=wire.now)
    assert len([row for row in wire.calls if row[0] == "AddOrder"]) == 1
    wire.missing = False
    wire.add("T1", "10", "100", "1")
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    assert len([row for row in wire.calls if row[0] == "AddOrder"]) == 1
    assert any(row[0] == "ClosedOrders" for row in wire.calls)


def test_partial_cancellation_books_actual_fees_before_release(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "3", "100", ".3")
    wire.now = T + timedelta(seconds=3)
    assert engine.cancel(order.intent_id, as_of=wire.now)["state"] == "cancelled"
    assert engine.ledger.quantity(order.account_id, BTC.instrument_id) == 3
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert engine.ledger.reconcile(order.account_id)["fees"] == "0.3"


def test_read_native_order_proves_actual_legacy_userref_trades_without_adopting(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.order.pop("cl_ord_id")
    wire.order["userref"] = 12345
    wire.add("T1", "10", "100", ".01", fee_asset="XXBT")
    wire.now = T + timedelta(seconds=3)
    report = adapter.read_native_order(order, "O-FIXTURE")
    assert report.client_order_id == "native-order:O-FIXTURE" and report.executions_complete
    assert report.fills[0].inventory_fees[0].quantity == D(".01")
    assert not engine.ledger.fills(order.account_id)
    assert len([call for call in wire.calls if call[0] == "AddOrder"]) == 1


@pytest.mark.parametrize("gap", ["empty_trades", "empty_ledgers"])
def test_missing_trade_or_ledger_coverage_never_synthesizes_fee_or_fill(tmp_path, gap):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "10", "100", "1")
    setattr(wire, gap, True)
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] in (
        "reconciliation_required",
        "ambiguous",
    )
    assert not engine.ledger.fills(order.account_id)
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000


@pytest.mark.parametrize(
    "field,value", [("refid", "another-trade"), ("amount", "-999"), ("fee", "2"), ("asset", "EUR")]
)
def test_debit_proof_mismatch_is_not_economic_success(tmp_path, field, value):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "10", "100", "1")
    wire.ledgers["T1-quote"][field] = value
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
    assert not engine.ledger.fills(order.account_id)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("wallet", "another-main", "wallet identity"),
        ("credit", "1", "borrowing"),
        ("cursor_repeat", True, "cursor repeated"),
    ],
)
def test_account_credit_and_repeated_cursor_fail_closed(tmp_path, field, value, match):
    wire = Wire()
    setattr(wire, field, value)
    wire, _, adapter, _ = setup(tmp_path, wire)
    with pytest.raises(LedgerError, match=match):
        adapter.capture()


def test_live_authorization_and_no_fictitious_spot_paper_endpoint(tmp_path):
    with pytest.raises(LedgerError, match="independently authorized"):
        setup(tmp_path, allow_live=False)
    paper = AccountSpec(
        account_id="paper:kraken",
        mode="paper",
        base_currency="USD",
        initial_cash=100000,
        opened_at=T,
    )
    with pytest.raises(LedgerError, match="no verified paper endpoint"):
        setup(tmp_path, account=paper)


def test_unknown_order_error_is_not_proof_of_nonacceptance(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.missing = True
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000


def test_more_than_twenty_actual_trades_are_queried_without_consolidation(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    order = order.model_copy(update={"quantity": D(11)})
    admit(wire, engine, adapter, order)
    for index in range(22):
        wire.add(f"T{index:02}", ".5", "100", ".05")
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    assert len(engine.ledger.fills(order.account_id)) == 22
    queried = [params for path, params in wire.calls if path == "QueryTrades"]
    assert [len(params["txid"].split(",")) for params in queried] == [20, 2]
    assert engine.ledger.reconcile(order.account_id)["fees"] == "1.10"


def test_simultaneous_native_quote_and_base_debits_are_conserved(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "3", "100", ".3")
    wire.ledgers["T1-base"]["fee"] = ".003"
    wire.held -= D(".003")
    wire.trades["T1"]["fee"] = wire.order["fee"] = ".6"
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    fill = engine.ledger.fills(order.account_id)[0]
    assert fill.fee == D(".3") and fill.inventory_fees[0].quantity == D(".003")
    assert engine.ledger.reconcile(order.account_id)["fees"] == "0.600"


def test_native_ledger_id_cannot_change_balance_while_preserving_trade_cost(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "3", "100", ".3")
    wire.now = T + timedelta(seconds=3)
    engine.reconcile_all(as_of=wire.now)
    wire.ledgers["T1-quote"]["balance"] = "0"
    wire.now += timedelta(seconds=1)
    with pytest.raises(LedgerError, match="ledger identity changed"):
        adapter.lookup(order, engine.state(order.intent_id)["client_id"], "O-FIXTURE", wire.now)
    assert engine.ledger.reconcile(order.account_id)["fees"] == "0.3"
