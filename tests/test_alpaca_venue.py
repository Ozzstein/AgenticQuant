"""Synthetic HTTP protocol fixtures; no authenticated Alpaca acceptance claimed."""

import asyncio
import json
import time
from datetime import timedelta
from decimal import Decimal as D

import httpx
import pytest

from src.domain.portfolio import AccountSpec, FillPolicy, OrderIntent, PriceMark
from src.execution.alpaca_venue import AlpacaVenue
from src.execution.broker import create_typed_alpaca_execution
from src.execution.ledger import LedgerError
from src.execution.venue import VenueRules
from tests.test_durable_accounting import A, T, intent, metadata

_ADAPTERS = []


@pytest.fixture(autouse=True)
def close_transports():
    yield
    while _ADAPTERS:
        _ADAPTERS.pop().close()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, *, delay=0):
        self.chunks, self.delay, self.reads = chunks, delay, 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class Wire:
    def __init__(self):
        self.now = T
        self.order = None
        self.activities = []
        self.fees = []
        self.calls = []
        self.cash = "100000"
        self.positions = []
        self.other_orders = []
        self.timeout = False
        self.missing = False
        self.rejection = None
        self.repeat_page = False

    async def transport(self, request):
        response = self.handle(request)
        return httpx.Response(
            response.status_code, headers=response.headers, stream=Chunks([response.content])
        )

    def handle(self, request):
        assert request.url.host == "paper-api.alpaca.markets"
        assert request.headers["APCA-API-KEY-ID"] == "fixture-key-not-a-credential"
        assert request.headers["APCA-API-SECRET-KEY"] == "fixture-secret-not-a-credential"
        path, method = request.url.path, request.method
        self.calls.append((method, path, dict(request.url.params), request.content))
        if path == "/v2/account":
            return httpx.Response(
                200,
                json={
                    "id": "remote-fixture",
                    "currency": "USD",
                    "status": "ACTIVE",
                    "cash": self.cash,
                    "non_marginable_buying_power": self.cash,
                },
            )
        if path == "/v2/positions":
            return httpx.Response(200, json=self.positions)
        if path == "/v2/calendar":
            return httpx.Response(
                200, json=[{"date": "2024-01-02", "open": "09:30", "close": "16:00"}]
            )
        if path == "/v2/assets/A":
            return httpx.Response(
                200,
                json={
                    "symbol": "A",
                    "class": "us_equity",
                    "status": "active",
                    "tradable": True,
                    "fractionable": False,
                },
            )
        if path == "/v2/orders" and method == "GET":
            orders = (
                [self.order]
                if self.order and self.order["status"] in ("new", "partially_filled")
                else []
            )
            return httpx.Response(200, json=[*orders, *self.other_orders])
        if path == "/v2/orders" and method == "POST":
            payload = json.loads(request.content)
            if self.rejection:
                return httpx.Response(self.rejection[0], json={"code": self.rejection[1]})
            self.order = {
                **payload,
                "id": "remote-order-1",
                "status": "new",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
            if self.timeout:
                raise httpx.ReadTimeout("synthetic acceptance with lost response", request=request)
            return httpx.Response(200, json=self.order)
        if path == "/v2/account/activities":
            kind = request.url.params["activity_types"]
            rows = self.fees if kind == "FEE" else self.activities
            token = request.url.params.get("page_token")
            start = (
                next((i + 1 for i, row in enumerate(rows) if row["id"] == token), 0) if token else 0
            )
            return httpx.Response(200, json=rows[(0 if self.repeat_page else start) :][:100])
        if path in ("/v2/orders/remote-order-1", "/v2/orders:by_client_order_id"):
            if self.missing:
                return httpx.Response(404, json={"code": 40410000})
            if method == "DELETE":
                self.order["status"] = "canceled"
                return httpx.Response(204)
            return httpx.Response(200, json=self.order)
        raise AssertionError((method, path))

    def add_fill(self, identity, qty, price, seconds):
        self.activities.append(
            {
                "activity_type": "FILL",
                "id": identity,
                "order_id": "remote-order-1",
                "symbol": "A",
                "side": "buy",
                "qty": qty,
                "price": price,
                "transaction_time": (T + timedelta(seconds=seconds)).isoformat(),
            }
        )


def setup(tmp_path, wire=None, **limits):
    wire = wire or Wire()
    rules = VenueRules(
        venue="alpaca",
        instrument=A,
        symbol="A",
        available_at=T,
        quantity_step=1,
        price_tick=".01",
        market_type="equity",
        active=True,
    )
    engine = create_typed_alpaca_execution(
        ledger_path=tmp_path / "alpaca.sqlite3",
        account=AccountSpec(
            account_id="paper:alpaca-test",
            mode="paper",
            base_currency="USD",
            initial_cash=100000,
            opened_at=T,
        ),
        broker_account_id="remote-fixture",
        api_key="fixture-key-not-a-credential",
        api_secret="fixture-secret-not-a-credential",
        rules={A.instrument_id: rules},
        fill_policy=FillPolicy(equity_cost_bps=0),
        client=httpx.AsyncClient(transport=httpx.MockTransport(wire.transport)),
        clock=lambda: wire.now,
        **limits,
    )
    order = intent().model_copy(update={"account_id": engine.account_id})
    _ADAPTERS.append(engine.adapters["alpaca"])
    return wire, engine, engine.adapters["alpaca"], order


def admit(wire, engine, adapter, order):
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    wire.now = order.eligible_at
    return engine.dispatch(order.intent_id, as_of=wire.now)


def trailing_setup(tmp_path, wire=None, *, trail_percent="10", trail_price=None):
    wire, engine, adapter, order = setup(tmp_path, wire)
    adapter._rules[A.instrument_id] = adapter._rules[A.instrument_id].model_copy(
        update={"order_types": ("market", "limit", "stop", "trailing_stop")}
    )
    order = OrderIntent.model_validate(
        {
            **order.model_dump(),
            "order_type": "trailing_stop",
            "trail_percent": trail_percent,
            "trail_price": trail_price,
        }
    )
    return wire, engine, adapter, order


@pytest.mark.parametrize("offset", [False, True])
def test_native_trailing_exact_wire_partial_restart_uses_only_actual_activities(tmp_path, offset):
    terms = {"trail_price": "8.00", "trail_percent": None} if offset else {"trail_percent": "10"}
    wire, engine, adapter, order = trailing_setup(tmp_path, **terms)
    assert admit(wire, engine, adapter, order)["state"] == "new"
    sent = json.loads(next(body for method, _, _, body in wire.calls if method == "POST"))
    assert sent["type"] == "trailing_stop" and sent["order_class"] == "simple"
    assert sent["trail_price" if offset else "trail_percent"] == ("8.00" if offset else "10")
    assert "hwm" not in sent and "stop_price" not in sent
    assert ("trail_percent" if offset else "trail_price") not in sent
    # Native low-water and derived stop are real report fields, not fixed intent prices.
    wire.order.update(hwm="80", stop_price="88", filled_qty="4", status="partially_filled")
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "reconciliation_required"
    assert not engine.ledger.fills(order.account_id)
    wire.add_fill("trailing-actual-1", "4", "90", 2)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
    wire.cash, wire.positions = "99640", [{"symbol": "A", "side": "long", "qty": "4"}]
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    engine.ledger.close()
    wire, engine, adapter, same = trailing_setup(tmp_path, wire, **terms)
    assert same == order
    wire.now = T + timedelta(seconds=5)
    # Once elected, actual native fills remain authoritative even on a rebound.
    wire.add_fill("trailing-actual-2", "6", "70", 4)
    wire.order.update(filled_qty="10", status="filled")
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    engine.reconcile_all(as_of=wire.now)
    wire.cash, wire.positions = "99220", [{"symbol": "A", "side": "long", "qty": "10"}]
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    assert [f.price for f in engine.ledger.fills(order.account_id)] == [D(90), D(70)]
    engine.ledger.mark(order.account_id, [PriceMark(instrument=A, price=70, as_of=wire.now)])
    assert engine.ledger.snapshot(order.account_id, wire.now, 0).nav == 99920
    assert engine.ledger.reconcile(order.account_id)["reconciled"]
    assert sum(method == "POST" for method, *_ in wire.calls) == 1
    assert not engine.ledger.db.execute(
        "SELECT 1 FROM events WHERE id LIKE 'paper-stop:%'"
    ).fetchone()
    engine.ledger.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"trail_percent": "9"},
        {"trail_price": "10"},
        {"type": "market"},
        {"hwm": "NaN"},
        {"stop_price": "0"},
        {"order_class": "bracket"},
    ],
)
def test_native_trailing_changed_terms_or_invalid_observation_never_synthesize_fills(
    tmp_path, changes
):
    wire, engine, adapter, order = trailing_setup(tmp_path)
    wire.timeout = True
    assert admit(wire, engine, adapter, order)["state"] == "ambiguous"
    wire.order.update(hwm="100", stop_price="110")
    wire.order.update(changes)
    wire.now += timedelta(seconds=1)
    with pytest.raises(LedgerError):
        adapter.lookup(order, engine.state(order.intent_id)["client_id"], None, wire.now)
    assert not engine.ledger.fills(order.account_id)
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    assert engine.dispatch(order.intent_id, as_of=wire.now)["state"] == "ambiguous"
    assert sum(method == "POST" for method, *_ in wire.calls) == 1
    engine.ledger.close()


def test_exact_fill_activities_partial_cash_fee_losing_nav_and_restart(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    assert admit(wire, engine, adapter, order)["state"] == "new"
    wire.now = T + timedelta(seconds=3)
    wire.order.update(status="partially_filled", filled_qty="3", filled_avg_price="999")
    # Acknowledged quantity without individual executions cannot be smoothed.
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "reconciliation_required"
    assert not engine.ledger.fills(order.account_id)
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    wire.add_fill("execution-1", "3", "100", 2)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
    wire.now = T + timedelta(seconds=5)
    wire.add_fill("execution-2", "7", "101", 4)
    wire.order.update(status="filled", filled_qty="10")
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    wire.cash, wire.positions = "98993", [{"symbol": "A", "side": "long", "qty": "10"}]
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["cash"] == "98993"
    assert [f.price for f in engine.ledger.fills(order.account_id)] == [D(100), D(101)]
    # Actual regulatory fee is separately posted, never inferred from fill average.
    wire.now += timedelta(seconds=1)
    wire.fees = [
        {
            "activity_type": "FEE",
            "id": "fee-1",
            "date": T.date().isoformat(),
            "net_amount": "-0.031234567890123456789",
            "order_id": "remote-order-1",
        }
    ]
    fees = adapter.reconcile_fees()
    assert fees[0].intent_id == order.intent_id
    wire.cash = "98992.968765432109876543211"
    adapter.capture()
    assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    engine.ledger.mark(order.account_id, [PriceMark(instrument=A, price=90, as_of=wire.now)])
    snap = engine.ledger.snapshot(order.account_id, wire.now, 0)
    assert snap.nav == D("99892.968765432109876543211")
    assert snap.nav < snap.high_water_mark and snap.fees == D("0.031234567890123456789")
    assert engine.ledger.reconcile(order.account_id)["reconciled"]
    payloads = " ".join(row[0] for row in engine.ledger.db.execute("SELECT payload FROM events"))
    assert (
        "fixture-key-not-a-credential" not in payloads
        and "fixture-secret-not-a-credential" not in payloads
    )
    engine.ledger.close()
    wire, reopened, adapter, _ = setup(tmp_path, wire)
    assert adapter.balances(wire.now).cash_base == D(wire.cash)
    adapter.reconcile_fees()  # duplicates are no-op even following published NAV
    report = adapter.lookup(
        order, reopened.client_id(order.account_id, order.intent_id), "remote-order-1", wire.now
    )
    reopened.ingest(order.intent_id, report)
    assert len(reopened.ledger.fills(order.account_id)) == 2
    assert reopened.ledger.reconcile(order.account_id)["reconciled"]
    reopened.ledger.close()


def test_acceptance_timeout_restart_same_client_lookup_never_resubmits(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    wire.timeout = True
    assert admit(wire, engine, adapter, order)["state"] == "ambiguous"
    client_id = engine.state(order.intent_id)["client_id"]
    engine.ledger.close()
    wire, engine, adapter, _ = setup(tmp_path, wire)
    wire.missing = True
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
    assert engine.dispatch(order.intent_id, as_of=wire.now)["state"] == "ambiguous"
    wire.missing = False
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "new"
    assert wire.order["client_order_id"] == client_id
    assert sum(method == "POST" for method, *_ in wire.calls) == 1
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    engine.ledger.close()


def test_partial_cancel_releases_only_unfilled_reservation(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.now += timedelta(seconds=2)
    wire.add_fill("part", "4", "100", 2)
    wire.order.update(status="partially_filled", filled_qty="4")
    assert engine.cancel(order.intent_id, as_of=wire.now)["state"] == "cancelled"
    assert engine.ledger.quantity(order.account_id, A.instrument_id) == 4
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert engine.ledger.reconcile(order.account_id)["reconciled"]
    engine.ledger.close()


@pytest.mark.parametrize(
    "status,code,expected",
    [(403, 40310000, "rejected"), (422, "42210000", "ambiguous"), (503, 50310000, "ambiguous")],
)
def test_authoritative_rejection_vs_ambiguous_response(tmp_path, status, code, expected):
    wire, engine, adapter, order = setup(tmp_path)
    wire.rejection = (status, code)
    assert admit(wire, engine, adapter, order)["state"] == expected
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == (
        0 if expected == "rejected" else 1000
    )
    engine.ledger.close()


def test_activity_pagination_exact_coverage_and_duplicate_cursor_rejection(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.now = T + timedelta(seconds=200)
    for i in range(101):
        wire.add_fill(f"exec-{i}", "0.01", "100", i + 2)
    wire.order.update(status="partially_filled", filled_qty="1.01")
    report = adapter.lookup(
        order, engine.state(order.intent_id)["client_id"], "remote-order-1", wire.now
    )
    assert report.executions_complete and len(report.fills) == 101
    engine.ingest(order.intent_id, report)
    assert engine.ledger.quantity(order.account_id, A.instrument_id) == D("1.01")
    assert any(params.get("page_token") == "exec-99" for _, _, params, _ in wire.calls)
    wire.repeat_page = True
    with pytest.raises(LedgerError, match="pagination/type conflict"):
        adapter.lookup(
            order, engine.state(order.intent_id)["client_id"], "remote-order-1", wire.now
        )
    assert len(engine.ledger.fills(order.account_id)) == 101
    assert engine.ledger.reconcile(order.account_id)["reconciled"]
    engine.ledger.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "wrong"},
        {"qty": "11"},
        {"extended_hours": True},
        {"order_class": "bracket"},
        {"limit_price": "100"},
    ],
)
def test_changed_order_identity_or_instructions_are_unresolved(tmp_path, changes):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.order.update(changes)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
    assert not engine.ledger.fills(order.account_id)
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    engine.ledger.close()


def test_capture_content_raw_evidence_and_temporal_fences(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    with pytest.raises(LedgerError, match="capture actual"):
        adapter.balances(T)
    captured = adapter.capture()
    with pytest.raises(LedgerError, match="clock/identity"):
        engine.prepare(
            order.model_copy(
                update={
                    "decision_at": T + timedelta(seconds=31),
                    "eligible_at": T + timedelta(seconds=32),
                }
            ),
            {A.instrument_id: metadata()},
            as_of=T + timedelta(seconds=31),
        )
    with pytest.raises(LedgerError, match="capture actual"):
        adapter.balances(T - timedelta(seconds=1))
    wire.now = T - timedelta(seconds=1)
    with pytest.raises(LedgerError, match="clock regressed"):
        adapter.capture()
    wire.now = T
    row = engine.ledger.db.execute(
        "SELECT payload FROM events WHERE id=?", (captured.evidence_id,)
    ).fetchone()
    data = json.loads(row[0])
    data["account"]["cash"] = "1"
    engine.ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?", (json.dumps(data), captured.evidence_id)
    )
    with pytest.raises(LedgerError, match="hash changed"):
        adapter.balances(T)
    engine.ledger.close()


def test_explicit_account_binding_and_no_direct_owner_bypass(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    adapter.capture()
    initial_reads = len(wire.calls)
    with pytest.raises(LedgerError, match="submitting claim"):
        adapter.submit(order, "not-owned", adapter._rules[A.instrument_id], T)
    assert len(wire.calls) == initial_reads
    with pytest.raises(LedgerError, match="conflicting event retry"):
        AlpacaVenue(
            engine.ledger,
            engine.account_id,
            broker_account_id="different-account",
            api_key="x",
            api_secret="y",
            rules=adapter._rules,
        )
    wire.other_orders = [{"id": "manual", "client_order_id": "outside-owner"}]
    adapter.capture()
    with pytest.raises(LedgerError, match="unowned"):
        engine.reconcile_balances(as_of=T)
    with pytest.raises(LedgerError, match="execution compliance FAIL"):
        engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    assert not engine.ledger.pending(order.account_id)
    engine.ledger.close()


def test_changed_remote_cash_blocks_dispatch_before_any_post(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    adapter.capture()
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    wire.now = order.eligible_at
    wire.cash = "99999"  # Unclassified actual account debit, not a guessed fee.
    adapter.capture()
    with pytest.raises(LedgerError, match="do not reconcile"):
        engine.dispatch(order.intent_id, as_of=wire.now)
    assert engine.state(order.intent_id)["state"] == "prepared"
    assert not any(method == "POST" for method, *_ in wire.calls)
    engine.ledger.close()


def test_same_capture_hash_cannot_substitute_changed_raw_response(tmp_path):
    wire, engine, adapter, _ = setup(tmp_path)
    captured = adapter.capture()
    row = engine.ledger.db.execute(
        "SELECT payload FROM events WHERE id=?", (captured.evidence_id,)
    ).fetchone()
    reference = json.loads(row[0])["references"]["account"]
    raw_row = engine.ledger.db.execute(
        "SELECT payload FROM events WHERE id=?", (reference,)
    ).fetchone()
    raw = json.loads(raw_row[0])
    raw["response"] = raw["response"].replace("100000", "200000")
    engine.ledger.db.execute("UPDATE events SET payload=? WHERE id=?", (json.dumps(raw), reference))
    with pytest.raises(LedgerError, match="retained raw evidence"):
        adapter.balances(T)
    engine.ledger.close()


def test_old_cutoff_cannot_authorize_actual_late_submission(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    adapter.capture()
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    wire.now = order.eligible_at + timedelta(minutes=1)
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "ambiguous"
    assert not any(method == "POST" for method, *_ in wire.calls)
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    engine.ledger.close()


def test_true_total_deadline_cancels_dribbling_response(tmp_path):
    wire, engine, adapter, _ = setup(tmp_path, operation_timeout_seconds=0.04)
    stream = Chunks([b" "] * 100, delay=0.01)
    adapter._client._transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream)
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter.capture()
    assert time.monotonic() - started < 0.5
    assert stream.reads < 10 and stream.closed
    assert not engine.ledger.db.execute(
        "SELECT 1 FROM events WHERE id LIKE 'alpaca-capture:%'"
    ).fetchone()
    engine.ledger.close()


def test_stream_cap_closes_before_remaining_body_and_retains_bounded_prefix(tmp_path):
    wire, engine, adapter, _ = setup(tmp_path, maximum_response_bytes=10)
    stream = Chunks([b"x" * 8] * 100)
    adapter._client._transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream)
    )
    with pytest.raises(LedgerError, match="streaming byte cap"):
        adapter.capture()
    assert stream.reads == 2 and stream.closed
    raw = json.loads(
        engine.ledger.db.execute(
            "SELECT payload FROM events WHERE id LIKE 'alpaca-http:%'"
        ).fetchone()[0]
    )
    assert raw["response"] == "x" * 10 and raw["complete"] is False
    engine.ledger.close()


@pytest.mark.parametrize("body", [b"{invalid json}", b"\xff\x00bad"])
def test_invalid_wire_bytes_are_archived_before_json_parse_failure(tmp_path, body):
    wire, engine, adapter, _ = setup(tmp_path)
    adapter._client._transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Chunks([body]))
    )
    with pytest.raises((ValueError, UnicodeError)):
        adapter.capture()
    raw = json.loads(
        engine.ledger.db.execute(
            "SELECT payload FROM events WHERE id LIKE 'alpaca-http:%'"
        ).fetchone()[0]
    )
    import base64

    retained = (
        raw["response"].encode() if "response" in raw else base64.b64decode(raw["response_base64"])
    )
    assert retained == body and raw["complete"] is True
    engine.ledger.close()


def test_operation_request_budget_covers_all_activity_pages(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.now = T + timedelta(seconds=200)
    for i in range(101):
        wire.add_fill(f"fill-{i}", ".01", "100", i + 2)
    wire.order.update(status="partially_filled", filled_qty="1.01")
    adapter.max_requests_per_operation = 2
    prior = len(wire.calls)
    with pytest.raises(LedgerError, match="request allowance exhausted"):
        adapter.lookup(
            order, engine.state(order.intent_id)["client_id"], "remote-order-1", wire.now
        )
    assert len(wire.calls) == prior + 2
    assert not engine.ledger.fills(order.account_id)
    engine.ledger.close()
