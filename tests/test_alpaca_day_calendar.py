"""Source-bound DAY tests against the actual offline native transport/ledger."""

import json
from datetime import datetime, timedelta

import httpx
import pytest

from src.execution.alpaca_calendar import next_session, query, read
from src.execution.ledger import LedgerError
from tests.test_alpaca_venue import Wire, admit, setup
from tests.test_durable_accounting import A, B, T


class CalendarWire(Wire):
    rows = [{"date": "2024-01-02", "open": "09:30", "close": "16:00"}]
    calendar_status = 200

    def handle(self, request):
        if request.url.path == "/v2/calendar":
            assert request.method == "GET"
            assert dict(request.url.params) == query(self.now)
            self.calls.append(
                (request.method, request.url.path, dict(request.url.params), request.content)
            )
            return httpx.Response(self.calendar_status, json=self.rows)
        return super().handle(request)


def day_setup(path, wire=None):
    wire, execution, adapter, order = setup(path, wire or CalendarWire())
    adapter._rules[A.instrument_id] = adapter._rules[A.instrument_id].model_copy(
        update={"time_in_force": ("GTC", "DAY")}
    )
    return (
        wire,
        execution,
        adapter,
        order.model_copy(update={"tif": "DAY", "expires_at": T.replace(hour=21)}),
    )


@pytest.mark.parametrize(
    "now,rows,expected",
    [
        (
            "2024-03-08T22:00:00+00:00",
            [
                {"date": "2024-03-08", "open": "09:30", "close": "16:00"},
                {"date": "2024-03-11", "open": "09:30", "close": "16:00"},
            ],
            "2024-03-11T20:00:00+00:00",
        ),
        (
            "2024-11-29T15:00:00+00:00",
            [{"date": "2024-11-29", "open": "09:30", "close": "13:00"}],
            "2024-11-29T18:00:00+00:00",
        ),
        (
            "2024-07-04T15:00:00+00:00",
            [{"date": "2024-07-05", "open": "09:30", "close": "16:00"}],
            "2024-07-05T20:00:00+00:00",
        ),
    ],
)
def test_native_session_selection_uses_explicit_dates_dst_and_early_close(now, rows, expected):
    at = datetime.fromisoformat(now)
    close, actual = next_session(rows, query(at), at)
    assert close == datetime.fromisoformat(expected) and actual in rows


def test_day_actual_partial_receipt_and_restart_reuse_original_calendar(tmp_path):
    wire, execution, adapter, order = day_setup(tmp_path)
    try:
        wire.timeout = True
        assert admit(wire, execution, adapter, order)["state"] == "ambiguous"
        proof = read(adapter, order)
        assert proof["expires_at"] == order.expires_at.isoformat()
        assert proof["source"].startswith("alpaca-http:")
        wire.now += timedelta(seconds=1)
        wire.add_fill("day-fill", "3", "100", 2)
        wire.order.update(filled_qty="3", status="partially_filled")
        assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
        fills = execution.ledger.fills(order.account_id)
    finally:
        adapter.close()
        execution.ledger.close()
    wire, execution, adapter, same = day_setup(tmp_path, wire)
    try:
        wire.now += timedelta(seconds=1)
        assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
        assert read(adapter, same) == proof
        assert execution.ledger.fills(order.account_id) == fills
        assert sum(method == "POST" for method, *_ in wire.calls) == 1
        assert sum(path == "/v2/calendar" for _, path, *_ in wire.calls) == 1
        assert execution.ledger.reconcile(order.account_id)["reconciled"]
    finally:
        adapter.close()
        execution.ledger.close()


@pytest.mark.parametrize(
    "rows",
    [
        [],
        {},
        [{"date": "2024-01-02", "open": "9:30", "close": "16:00"}],
        [{"date": "2024-01-02", "open": "16:00", "close": "09:30"}],
        [{"date": "2024-01-03", "open": "09:30", "close": "16:00"}],
        [{"date": "2024-02-02", "open": "09:30", "close": "16:00"}],
        [{"date": "2024-01-02", "open": "09:30", "close": "13:00"}],
        CalendarWire.rows * 2,
    ],
)
def test_missing_wrong_or_ambiguous_calendar_denies_before_post(tmp_path, rows):
    wire, execution, adapter, order = day_setup(tmp_path)
    try:
        wire.rows = rows
        result = admit(wire, execution, adapter, order)
        assert result["state"] == "rejected" and "DAY" in result["detail"]
        assert not any(method == "POST" for method, *_ in wire.calls)
        assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert not execution.ledger.fills(order.account_id)
    finally:
        adapter.close()
        execution.ledger.close()


@pytest.mark.parametrize("tamper", ["raw", "proof", "claim", "missing"])
def test_original_calendar_tampering_cannot_enter_native_economics(tmp_path, tamper):
    wire, execution, adapter, order = day_setup(tmp_path)
    try:
        assert admit(wire, execution, adapter, order)["state"] == "new"
        proof = read(adapter, order)
        wire.now += timedelta(seconds=1)
        wire.add_fill("actual-day-fill", "3", "100", 2)
        wire.order.update(filled_qty="3", status="partially_filled")
        key = (
            proof["source"]
            if tamper == "raw"
            else ("alpaca-submit-attempt:" if tamper == "claim" else "alpaca-day-calendar:")
            + order.intent_id
        )
        if tamper == "missing":
            execution.ledger.db.execute("DELETE FROM events WHERE id=?", (key,))
        else:
            original = execution.ledger.db.execute(
                "SELECT payload FROM events WHERE id=?", (key,)
            ).fetchone()[0]
            value = json.loads(original)
            if tamper == "raw":
                value["params"]["date_type"] = "SETTLEMENT"
            elif tamper == "claim":
                value["day_calendar_hash"] = "0" * 64
            else:
                value["selected_at"] = (order.eligible_at - timedelta(microseconds=1)).isoformat()
            execution.ledger.db.execute(
                "UPDATE events SET payload=? WHERE id=?", (json.dumps(value), key)
            )
        with pytest.raises(LedgerError):
            read(adapter, order)
        assert execution.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
        assert not execution.ledger.fills(order.account_id)
        assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
        assert sum(path == "/v2/calendar" for _, path, *_ in wire.calls) == 1
    finally:
        adapter.close()
        execution.ledger.close()


@pytest.mark.parametrize("failure", ["http", "deadline"])
def test_native_calendar_failure_or_boundary_crossing_never_attempts_post(tmp_path, failure):
    class FailureWire(CalendarWire):
        def handle(self, request):
            response = super().handle(request)
            if request.url.path == "/v2/calendar" and failure == "deadline":
                self.now = T.replace(hour=21)
            return response

    wire, execution, adapter, order = day_setup(tmp_path, FailureWire())
    try:
        if failure == "http":
            wire.calendar_status = 500
        assert admit(wire, execution, adapter, order)["state"] == "rejected"
        assert not any(method == "POST" for method, *_ in wire.calls)
        assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert sum(path == "/v2/calendar" for _, path, *_ in wire.calls) == 1
        assert (
            execution.ledger.db.execute(
                "SELECT COUNT(*) FROM events WHERE id LIKE 'alpaca-http:%'"
            ).fetchone()[0]
            > 0
        )
    finally:
        adapter.close()
        execution.ledger.close()


def test_native_day_calendar_requires_exact_configured_equity(tmp_path):
    wire, execution, adapter, order = day_setup(tmp_path)
    try:
        with pytest.raises(ValueError, match="exact configured US equity"):
            adapter.validate_day(order.model_copy(update={"instrument": B}), as_of=wire.now)
        assert not wire.calls
    finally:
        adapter.close()
        execution.ledger.close()
