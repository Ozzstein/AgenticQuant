"""Native-shaped calendar/property fixtures, not installed LEAN acceptance."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from types import SimpleNamespace as NS
from zoneinfo import ZoneInfo

import pytest

from src.execution.lean_runtime import LeanAlgorithmPort
from src.execution.lean_venue import LeanVenue
from src.execution.ledger import LedgerError
from tests.test_durable_accounting import A, B, T, intent
from tests.test_lean_execution import admit, setup


class NativeDate:
    """Only the native DateTime surface used by the real port, in tests."""

    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return getattr(self.value, name.lower())

    @property
    def Ticks(self):
        return self.value.microsecond * 10

    @property
    def Date(self):
        return NativeDate(self.value.replace(hour=0, minute=0, second=0, microsecond=0))

    def AddDays(self, days):
        return NativeDate(self.value + timedelta(days=days))

    def ToUniversalTime(self):
        return NativeDate(self.value.astimezone(timezone.utc))


def native_kind(name):
    return NS(GetType=lambda: NS(FullName="QuantConnect.Orders.TimeInForces." + name))


def port_fixture(instrument, now, close, *, brokerage="PaperBrokerage", zone="America/New_York"):
    calls, properties = [], []
    hours = NS(TimeZone=NS(Id=zone))

    def last_close(local, extended):
        calls.append((local.value, extended))
        return NativeDate(close.astimezone(ZoneInfo(zone)))

    hours.GetLastDailyMarketClose = last_close
    security = NS(
        Symbol=NS(SecurityType="Crypto" if instrument == B else "Equity"), Exchange=NS(Hours=hours)
    )
    native_order = NS(
        Tag="client",
        Symbol="symbol",
        Type="Limit",
        Time=NativeDate(now),
        TimeInForce=native_kind("DayTimeInForce"),
        LimitPrice=D(100),
        BrokerId=["original-native"],
    )
    ticket = NS(OrderId=1, Quantity=D(10), QuantityFilled=D(0), Status="Submitted", OrderEvents=[])

    def submit(symbol, quantity, price, tag, props):
        properties.append(props)
        return ticket

    port = object.__new__(LeanAlgorithmPort)
    port.algorithm = NS(
        Securities={"symbol": security},
        Transactions=NS(GetOrderById=lambda _: native_order),
        LimitOrder=submit,
    )
    port.deployment, port.symbols = NS(brokerage=brokerage), {instrument.instrument_id: "symbol"}
    port._extensions = NS(
        ConvertFromUtc=lambda at, tz: NativeDate(at.value.astimezone(ZoneInfo(tz.Id))),
        ConvertToUtc=lambda at, tz: NativeDate(
            at.value.replace(tzinfo=ZoneInfo(tz.Id)).astimezone(timezone.utc)
        ),
    )
    port._datetime = NS(Parse=lambda value, culture: NativeDate(datetime.fromisoformat(value)))
    port._culture, port._properties, port._gtc, port._day = (
        None,
        NS,
        lambda: native_kind("GoodTilCanceledTimeInForce"),
        lambda: native_kind("DayTimeInForce"),
    )
    port._native_decimal, port._number = D, D
    port._instrument, port.now, port.runtime_id = (
        lambda symbol: instrument.instrument_id,
        lambda: now,
        "calendar-fixture",
    )
    return port, calls, properties, native_order, ticket


@pytest.mark.parametrize(
    "submitted,close",
    [
        ("2024-01-02T15:00:00+00:00", "2024-01-02T21:00:00+00:00"),
        ("2024-03-11T14:00:00+00:00", "2024-03-11T20:00:00+00:00"),
        ("2024-11-29T15:00:00+00:00", "2024-11-29T18:00:00+00:00"),
        ("2024-03-08T22:00:00+00:00", "2024-03-11T20:00:00+00:00"),
    ],
)
def test_day_submission_and_receipt_use_actual_regular_calendar(submitted, close):
    now, end = datetime.fromisoformat(submitted), datetime.fromisoformat(close)
    port, calls, props, native, _ = port_fixture(A, now, end, brokerage="AlpacaBrokerage")
    order = intent(at=now - timedelta(seconds=2)).model_copy(
        update={"order_type": "limit", "limit_price": D(100), "tif": "DAY", "expires_at": end}
    )
    receipt = port.submit(order, "client")
    assert props[0].TimeInForce.GetType().FullName.endswith(".DayTimeInForce")
    assert len(calls) == 2 and all(extended is False for _, extended in calls)
    assert calls[0][0] == now.astimezone(ZoneInfo("America/New_York"))
    assert (
        receipt["time_in_force"] == "DAY" and datetime.fromisoformat(receipt["expires_at"]) == end
    )
    LeanVenue._validate_ticket_intent(order, receipt)
    # Observation on a later date still derives the original ticket boundary.
    port.now = lambda: end + timedelta(days=2)
    assert (
        port._ticket(
            NS(OrderId=1, Quantity=10, QuantityFilled=0, Status="Canceled", OrderEvents=[])
        )["expires_at"]
        == receipt["expires_at"]
    )


def test_crypto_day_uses_actual_utc_day_not_good_til_date_next_day():
    now, end = T, T.replace(hour=0) + timedelta(days=1)
    port, calls, props, _, _ = port_fixture(B, now, end, zone="UTC")
    order = intent(at=now - timedelta(seconds=2), instrument=B).model_copy(
        update={"order_type": "limit", "limit_price": D(100), "tif": "DAY", "expires_at": end}
    )
    receipt = port.submit(order, "client")
    assert not calls
    assert props[0].TimeInForce.GetType().FullName.endswith(".DayTimeInForce")
    assert datetime.fromisoformat(receipt["expires_at"]) == end
    LeanVenue._validate_ticket_intent(order, receipt)


@pytest.mark.parametrize("offset", [-1, 1, 3600, 86400])
def test_day_cannot_substitute_intraday_or_wrong_session_boundary(offset):
    end = T.replace(hour=21)
    port, _, props, _, _ = port_fixture(A, T, end)
    order = intent(at=T - timedelta(seconds=2)).model_copy(
        update={
            "order_type": "limit",
            "limit_price": D(100),
            "tif": "DAY",
            "expires_at": end + timedelta(seconds=offset),
        }
    )
    with pytest.raises(LedgerError, match="actual security calendar"):
        port.submit(order, "client")
    assert not props


@pytest.mark.parametrize(
    "brokerage,instrument,zone",
    [
        ("BinanceBrokerage", B, "UTC"),
        ("KrakenBrokerage", B, "UTC"),
        ("AlpacaBrokerage", B, "UTC"),
        ("PaperBrokerage", B, "Europe/Rome"),
    ],
)
def test_unsupported_day_basis_rejects_before_submission(brokerage, instrument, zone):
    end = T.replace(hour=0) + timedelta(days=1)
    port, _, props, _, _ = port_fixture(instrument, T, end, brokerage=brokerage, zone=zone)
    order = intent(instrument=instrument).model_copy(update={"tif": "DAY", "expires_at": end})
    with pytest.raises(LedgerError, match="DAY"):
        port.submit(order, "client")
    assert not props


@pytest.mark.parametrize(
    "changes",
    [
        {"time_in_force": "GTC"},
        {"time_in_force": None},
        {"expires_at": None},
        {"expires_at": "2024-01-02T20:00:00+00:00"},
        {"expires_at": "2024-01-02T21:00:00"},
    ],
)
def test_day_ticket_changes_cannot_book_fill_or_release_reservation(tmp_path, changes):
    runtime, execution, adapter, order = setup(tmp_path)
    order = order.model_copy(update={"tif": "DAY", "expires_at": T.replace(hour=21)})
    adapter._rules[A.instrument_id] = adapter._rules[A.instrument_id].model_copy(
        update={"time_in_force": ("GTC", "DAY")}
    )
    assert admit(runtime, execution, adapter, order)["state"] == "new"
    runtime.at += timedelta(seconds=1)
    runtime.add("3", "100", "1")
    runtime.ticket.update(changes)
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "ambiguous"
    assert not execution.ledger.fills(order.account_id)
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    execution.ledger.close()


@pytest.mark.parametrize(
    "kind,group", [("GoodTilDateTimeInForce", False), ("DayTimeInForce", True)]
)
def test_unproven_old_date_or_contingent_ticket_cannot_be_relabelled(kind, group):
    port, _, _, native, ticket = port_fixture(A, T, T.replace(hour=21))
    native.TimeInForce = native_kind(kind)
    if group:
        native.Contingency = NS()
    with pytest.raises(LedgerError, match="time-in-force|group ownership"):
        port._ticket(ticket)


def test_live_day_keeps_original_native_terms_through_expiry_and_partial_fill(tmp_path):
    from tests.test_lean_execution import setup_live

    runtime, execution, adapter, order, wire, native = setup_live(tmp_path)
    try:
        order = order.model_copy(update={"tif": "DAY", "expires_at": T.replace(hour=21)})
        for venue in (adapter, native):
            venue._rules[A.instrument_id] = venue._rules[A.instrument_id].model_copy(
                update={"time_in_force": ("GTC", "DAY")}
            )
        assert admit(runtime, execution, adapter, order)["state"] == "new"
        assert wire.order["time_in_force"] == "day"
        runtime.at += timedelta(seconds=1)
        runtime.add("3", "100", "7")
        wire.add_fill("actual-day-fill", "3", "100", 2)
        wire.order.update(filled_qty="3", status="partially_filled")
        wire.cash, wire.positions = "99700", [{"symbol": "A", "side": "long", "qty": "3"}]
        runtime.cash = D(wire.cash)
        assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "partial"
        runtime.runtime_id = "restarted-day-runtime"
        runtime.ticket.update(
            runtime_id=runtime.runtime_id, ticket_id=77, client_order_id="imported-day", events=[]
        )
        runtime.at = order.expires_at
        original_cancel = runtime.cancel

        def cancel(ticket, client):
            original_cancel(ticket, client)
            wire.order["status"] = "canceled"

        runtime.cancel = cancel
        assert execution.expire_orders(as_of=runtime.at)[0]["state"] == "cancelled"
        fills = execution.ledger.fills(order.account_id)
        assert (
            len(fills) == 1 and fills[0].fill_id == "alpaca:actual-day-fill" and fills[0].fee == 0
        )
        assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert execution.ledger.reconcile(order.account_id)["reconciled"]
        assert sum(call[0] == "submit" for call in runtime.calls) == 1
        assert all(method == "GET" for method, *_ in wire.calls)
    finally:
        native.close()
        execution.ledger.close()
