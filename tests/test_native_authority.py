"""Real retained protocol receipts; no authenticated native acceptance claim."""

import hashlib
import json
import socket
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.native_archive import NativeArchive, exact_time
from src.execution.native_authority import reexport_execution, reexport_report
from tests.test_alpaca_venue import admit, setup
from tests.test_durable_accounting import T


@contextmanager
def readonly_proof(path):
    # Prepare a closed archival SQLite file. WAL readers legitimately maintain
    # SQLite coordination sidecars; those must not be confused with authority
    # writes, or disabled for a live audit by using unsafe immutable=1.
    archive = sqlite3.connect(path)
    archive.execute("PRAGMA journal_mode=DELETE")
    archive.close()
    ledger = PortfolioLedger(path, read_only=True)
    before = {str(p): p.read_bytes() for p in Path(path).parent.iterdir() if p.is_file()}
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION,
    }
    ledger.db.set_authorizer(
        lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native authority cannot perform I/O or mutate retained files")

    try:
        with (
            patch.object(socket, "socket", forbidden),
            patch.object(Path, "write_bytes", forbidden),
            patch.object(Path, "write_text", forbidden),
        ):
            yield ledger
        assert {
            str(p): p.read_bytes() for p in Path(path).parent.iterdir() if p.is_file()
        } == before
    finally:
        ledger.close()


def latest_reference(ledger, account):
    row = ledger.db.execute(
        "SELECT intent,evidence,payload FROM venue_reports WHERE account=? ORDER BY rowid DESC LIMIT 1",
        (account,),
    ).fetchone()
    return NativeArchive(ledger, account, as_of=T + timedelta(days=1)).reference(
        "venue-report", row["evidence"], row["payload"], intent_id=row["intent"]
    )


def alpaca_receipt(path, *, expired=False):
    wire, execution, adapter, order = setup(path)
    if expired:
        order = order.model_copy(update={"expires_at": T + timedelta(seconds=2)})
    admit(wire, execution, adapter, order)
    wire.now = T + timedelta(seconds=3)
    wire.add_fill("original-source-trade", "3", "100", 3)
    # Actual first raw activity capture precedes the complete order report.
    from src.execution.alpaca_venue import _bounded

    _bounded(type(adapter)._activities)(adapter, kind="FILL", order_id="remote-order-1")
    wire.now = T + timedelta(seconds=4)
    wire.order.update(status="partially_filled", filled_qty="3")
    if expired:
        with pytest.raises(LedgerError, match="outside eligible"):
            execution.reconcile_all(as_of=wire.now)
    else:
        execution.reconcile_all(as_of=wire.now)
    reference = latest_reference(execution.ledger, order.account_id)
    adapter.close()
    execution.ledger.close()
    return path / "alpaca.sqlite3", order, reference, wire.now


@pytest.mark.parametrize("expired", [False, True])
def test_alpaca_reexport_is_readonly_and_keeps_first_observation_separate_from_complete(
    tmp_path, monkeypatch, expired
):
    path, order, reference, cutoff = alpaca_receipt(tmp_path, expired=expired)
    original_bytes = path.read_bytes()
    ledger = PortfolioLedger(path, read_only=True)
    queries = []

    def deny_writes(action, *args):
        queries.append(action)
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_TRANSACTION,
        }
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    def forbidden(*args, **kwargs):
        raise AssertionError("proof cannot create a socket or write a file")

    ledger.db.set_authorizer(deny_writes)
    with monkeypatch.context() as isolated:
        isolated.setattr(socket, "socket", forbidden)
        isolated.setattr(type(path), "write_bytes", forbidden)
        isolated.setattr(type(path), "write_text", forbidden)
        result = reexport_report(ledger, reference, as_of=cutoff)
    assert len(result.authorities) == 1
    authority = result.authorities[0]
    assert authority.first_observed_at == T + timedelta(seconds=3)
    assert authority.first_complete_at == T + timedelta(seconds=4)
    assert authority.fill.executed_at == T + timedelta(seconds=3)
    assert authority.key.broker_account_id == "remote-fixture"
    assert authority.fill.quantity == 3
    assert authority.fill.account_id == order.account_id
    assert authority.model_validate_json(authority.model_dump_json()) == authority
    assert len(ledger.fills(order.account_id)) == (0 if expired else 1)
    ledger.close()
    assert path.read_bytes() == original_bytes
    assert queries


@pytest.mark.parametrize(
    "damage", ["raw", "normalized-fill", "claim", "network", "owner", "account-source"]
)
def test_reexport_never_trusts_normalized_receipt_or_missing_original_authority(tmp_path, damage):
    path, order, reference, cutoff = alpaca_receipt(tmp_path)
    ledger = PortfolioLedger(path)
    if damage == "claim":
        ledger.db.execute(
            "DELETE FROM events WHERE id=?", ("alpaca-submit-attempt:" + order.intent_id,)
        )
    elif damage == "owner":
        ledger.db.execute("UPDATE execution_books SET owner='lean'")
    elif damage == "normalized-fill":
        row = ledger.db.execute(
            "SELECT payload FROM venue_reports WHERE evidence=?", (reference.record_id,)
        ).fetchone()
        content = json.loads(row[0])
        content["fills"][0]["price"] = "101"
        ledger.db.execute(
            "UPDATE venue_reports SET payload=? WHERE evidence=?",
            (json.dumps(content), reference.record_id),
        )
        reference = latest_reference(ledger, order.account_id)
    else:
        if damage == "network":
            identity = "remote-binding:alpaca"
            content = json.loads(
                ledger.db.execute("SELECT payload FROM events WHERE id=?", (identity,)).fetchone()[
                    0
                ]
            )
            content["endpoint"] = "https://api.alpaca.markets"
        else:
            row = next(
                row
                for row in ledger.db.execute(
                    "SELECT id,payload FROM events WHERE id LIKE 'alpaca-http:%'"
                )
                if json.loads(row["payload"])["path"]
                == ("/v2/account" if damage == "account-source" else "/v2/account/activities")
            )
            identity, content = row["id"], json.loads(row["payload"])
            content["response"] = "{}"
        ledger.db.execute("UPDATE events SET payload=? WHERE id=?", (json.dumps(content), identity))
    ledger.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = PortfolioLedger(path, read_only=True)
    try:
        with pytest.raises((LedgerError, ValueError)):
            reexport_report(ledger, reference, as_of=cutoff)
    finally:
        ledger.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "value,unit",
    [
        ("2024-01-02T14:00:00.000000001Z", "iso8601-utc"),
        ("1704204000.000000001", "unix-seconds-decimal"),
        (1704204000000.0, "unix-milliseconds"),
    ],
)
def test_authority_never_rounds_native_timestamp_precision(value, unit):
    with pytest.raises(LedgerError, match="precision|integer|exact"):
        exact_time(value, unit)


@pytest.mark.parametrize("venue", ["binance", "kraken"])
@pytest.mark.parametrize("fee_asset", ["cash", "base"])
def test_spot_raw_execution_and_posted_fee_reexport_on_readonly_store(tmp_path, venue, fee_asset):
    from tests import test_binance_venue, test_kraken_venue

    fixtures = test_binance_venue if venue == "binance" else test_kraken_venue
    wire, engine, adapter, order = fixtures.setup(tmp_path)
    fixtures.admit(wire, engine, adapter, order)
    # The initial no-fill order report must also have independently provable raw ownership.
    initial = latest_reference(engine.ledger, order.account_id)
    assert not reexport_report(engine.ledger, initial, as_of=wire.now).authorities
    wire.now = T + timedelta(seconds=4)
    if venue == "binance":
        wire.add(1, "3", "100", ".01", asset="USDT" if fee_asset == "cash" else "BTC")
    else:
        wire.add("trade-1", "3", "100", ".01", fee_asset="ZUSD" if fee_asset == "cash" else "XXBT")
    engine.reconcile_all(as_of=wire.now)
    original = engine.ledger.fills(order.account_id)
    reference = latest_reference(engine.ledger, order.account_id)
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        result = reexport_report(ledger, reference, as_of=wire.now)
        assert tuple(value.fill for value in result.authorities) == tuple(original)
        assert result.authorities[0].key.venue == venue
        assert bool(result.authorities[0].fill.inventory_fees) == (fee_asset == "base")


def test_lean_native_reexport_uses_original_broker_facts_without_fee_model_estimate(tmp_path):
    from tests.test_lean_execution import admit as lean_admit
    from tests.test_lean_execution import setup_live

    runtime, engine, adapter, order, wire, native = setup_live(tmp_path)
    lean_admit(runtime, engine, adapter, order)
    runtime.at += timedelta(seconds=1)
    runtime.add("10", "100", "7")
    wire.add_fill("actual-native", "10", "100", 2)
    wire.order.update(filled_qty="10", status="filled")
    engine.reconcile_all(as_of=runtime.at)
    reference = latest_reference(engine.ledger, order.account_id)
    path = engine.ledger.path
    native.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        result = reexport_report(ledger, reference, as_of=runtime.at)
        assert result.authorities[0].owner == "lean"
        assert result.authorities[0].fill.fee == 0
        assert result.authorities[0].key.native_execution_id == "actual-native"


def test_kraken_lost_ack_reexport_requires_both_original_cursor_histories(tmp_path):
    from tests.test_kraken_venue import Wire, admit, setup

    wire = Wire()
    wire.timeout = True
    wire, engine, adapter, order = setup(tmp_path, wire)
    assert admit(wire, engine, adapter, order)["state"] == "ambiguous"
    wire.add("closed-trade", "10", "100", "1")
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "filled"
    path, reference = engine.ledger.path, latest_reference(engine.ledger, order.account_id)
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        actual = reexport_report(ledger, reference, as_of=wire.now).authorities[0]
        assert actual.fill.quantity == 10
        assert actual.key.native_execution_id == "closed-trade"
        sources = [
            json.loads(
                ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (order.account_id, ref.record_id),
                ).fetchone()[0]
            )
            for _, ref in actual.sources
            if ref.record_id.startswith("kraken-http:")
        ]
        assert any(source["endpoint"].endswith("/OpenOrders") for source in sources)
        assert any(source["endpoint"].endswith("/ClosedOrders") for source in sources)


def test_incomplete_original_report_is_not_authority_and_complete_retry_keeps_first_complete(
    tmp_path,
):
    from tests.test_kraken_venue import admit, setup

    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("delayed-trade", "3", "100", ".1")
    wire.empty_trades = True
    wire.now = T + timedelta(seconds=3)
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "reconciliation_required"
    missing = latest_reference(engine.ledger, order.account_id)
    result = reexport_report(engine.ledger, missing, as_of=wire.now)
    assert not result.report.executions_complete and not result.authorities
    with pytest.raises(LedgerError, match="no complete"):
        reexport_execution(engine.ledger, order.account_id, "kraken:delayed-trade", as_of=wire.now)
    wire.empty_trades = False
    wire.now = T + timedelta(seconds=4)
    engine.reconcile_all(as_of=wire.now)
    first = reexport_execution(
        engine.ledger, order.account_id, "kraken:delayed-trade", as_of=wire.now
    )
    wire.now = T + timedelta(seconds=5)
    engine.reconcile_all(as_of=wire.now)
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        recovered = reexport_execution(
            ledger, order.account_id, "kraken:delayed-trade", as_of=wire.now
        )
        assert recovered == first
        assert recovered.first_complete_at == T + timedelta(seconds=4)
        assert recovered.first_observed_at == T + timedelta(seconds=4)


@pytest.mark.parametrize("venue", ["binance", "kraken"])
def test_later_conflicting_raw_execution_blocks_original_good_report(tmp_path, venue):
    from tests import test_binance_venue, test_kraken_venue

    fixtures = test_binance_venue if venue == "binance" else test_kraken_venue
    wire, engine, adapter, order = fixtures.setup(tmp_path)
    fixtures.admit(wire, engine, adapter, order)
    wire.now = T + timedelta(seconds=4)
    if venue == "binance":
        wire.add(1, "3", "100", ".1")
    else:
        wire.add("trade-1", "3", "100", ".1")
    engine.reconcile_all(as_of=wire.now)
    original = latest_reference(engine.ledger, order.account_id)
    assert reexport_report(engine.ledger, original, as_of=wire.now).authorities
    wire.now += timedelta(seconds=1)
    if venue == "binance":
        wire.trades[0]["commission"] = ".2"
    else:
        wire.trades["trade-1"]["fee"] = ".2"
    assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "ambiguous"
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        with pytest.raises(LedgerError, match="conflicting revision"):
            reexport_report(ledger, original, as_of=wire.now)
