"""Current source authority, permanent violations, and metadata-only detection."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.domain.portfolio import AccountSpec
from src.execution.execution_compliance import collect, detect_discrepancies
from src.execution.ledger import LedgerError, PortfolioLedger
from tests.test_durable_accounting import T, account
from tests.test_native_authority import alpaca_receipt, readonly_proof


def rows(ledger):
    return {
        row[0]: [
            tuple(item)
            for item in ledger.db.execute('SELECT * FROM "' + row[0] + '" ORDER BY rowid')
        ]
        for row in ledger.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    }


@pytest.mark.parametrize("expired", [False, True])
def test_native_audit_is_readonly_and_mathematical_success_does_not_clear_expiry(tmp_path, expired):
    path, order, _, _ = alpaca_receipt(tmp_path, expired=expired)
    with readonly_proof(path) as ledger:
        before = rows(ledger)
        assert ledger.reconcile(order.account_id)["reconciled"]
        audit = ledger.execution_compliance(order.account_id)
        assert audit.lineage == "native"
        assert audit.status == ("FAIL" if expired else "PASS"), audit
        assert len(audit.violations) == int(expired)
        if expired:
            assert "unapplied_native_execution" in audit.blocking_codes
        assert rows(ledger) == before
        assert audit == ledger.execution_compliance(order.account_id)


def test_detection_records_only_metadata_with_measured_clock_and_restart_identity(tmp_path):
    path, order, _, cutoff = alpaca_receipt(tmp_path, expired=True)
    ledger = PortfolioLedger(path)
    before = rows(ledger)
    earliest = datetime.now(timezone.utc)
    cases = detect_discrepancies(ledger, order.account_id, as_of=cutoff)
    latest = datetime.now(timezone.utc)
    assert len(cases) == 1
    assert earliest <= cases[0].first_detected_at <= latest
    assert cases[0].first_observed_at == T + timedelta(seconds=3)
    assert cases[0].reason_codes == ("outside_expiry",)
    after = rows(ledger)
    assert {k: v for k, v in after.items() if k != "events"} == {
        k: v for k, v in before.items() if k != "events"
    }
    assert len(after["events"]) == len(before["events"]) + 3
    assert set(before["events"]) <= set(after["events"])
    audit = ledger.execution_compliance(order.account_id)
    assert audit.status == "FAIL" and len(audit.violations) == 1
    ledger.close()
    ledger = PortfolioLedger(path)
    try:
        assert detect_discrepancies(ledger, order.account_id, as_of=cutoff) == cases
        assert ledger.execution_compliance(order.account_id) == audit
        assert rows(ledger) == after
        assert not hasattr(ledger, "settle_discrepancies")
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "damage",
    [
        "indexes",
        "application-flags",
        "reports",
        "fills",
        "journal",
        "owner",
        "binding",
        "account-mode",
    ],
)
def test_deleted_indexes_or_report_flags_cannot_clear_retained_execution_failure(tmp_path, damage):
    path, order, _, cutoff = alpaca_receipt(tmp_path, expired=damage not in ("fills", "journal"))
    ledger = PortfolioLedger(path)
    detect_discrepancies(ledger, order.account_id, as_of=cutoff)
    if damage == "indexes":
        ledger.db.execute("DELETE FROM events WHERE id LIKE 'execution-%'")
    elif damage == "application-flags":
        ledger.db.execute("UPDATE venue_reports SET applied=1")
        ledger.db.execute("UPDATE venue_outbox SET state='filled'")
    elif damage == "reports":
        ledger.db.execute("DELETE FROM venue_reports")
        ledger.db.execute("DELETE FROM events WHERE id LIKE 'execution-%'")
    elif damage == "fills":
        ledger.db.execute("DELETE FROM fills")
    elif damage == "journal":
        ledger.db.execute("DELETE FROM journal WHERE kind='fill'")
    elif damage == "owner":
        ledger.db.execute("DELETE FROM execution_books")
    elif damage == "binding":
        ledger.db.execute("DELETE FROM events WHERE id='remote-binding:alpaca'")
    else:
        raw = json.loads(
            ledger.db.execute(
                "SELECT spec FROM accounts WHERE id=?", (order.account_id,)
            ).fetchone()[0]
        )
        raw["mode"] = "backtest"
        raw["account_id"] = "backtest:renamed"
        ledger.db.execute("UPDATE accounts SET spec=?", (json.dumps(raw),))
    ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL", audit
        if damage == "reports":
            assert "unbound_native_execution" in audit.blocking_codes
        if damage in ("fills", "journal"):
            assert "execution_journal_fill_mismatch" in audit.blocking_codes


@pytest.mark.parametrize("mode", ["backtest", "paper", "live"])
def test_legacy_readonly_lineage_comes_from_original_account_not_caller_mode(tmp_path, mode):
    path = tmp_path / "legacy.sqlite"
    ledger = PortfolioLedger(path)
    identity = mode + ":old"
    ledger.create_account(
        AccountSpec(
            account_id=identity, mode=mode, base_currency="USD", initial_cash="100", opened_at=T
        )
    )
    ledger.close()
    with readonly_proof(path) as ledger:
        before = rows(ledger)
        audit = ledger.execution_compliance(identity)
        assert (
            audit.lineage
            == {"backtest": "historical-replay", "paper": "local-paper", "live": "unverified"}[mode]
        )
        assert audit.status == ("FAIL" if mode == "live" else "PASS")
        assert rows(ledger) == before
        assert not ledger.db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='venue_reports'"
        ).fetchone()


def test_metadata_detection_cannot_backdate_or_accept_incomplete_known_cutoff(tmp_path):
    path, order, _, cutoff = alpaca_receipt(tmp_path, expired=True)
    ledger = PortfolioLedger(path)
    try:
        before = rows(ledger)
        for at in (cutoff - timedelta(seconds=1), datetime.now(timezone.utc) + timedelta(days=1)):
            with pytest.raises(LedgerError, match="cutoff"):
                detect_discrepancies(ledger, order.account_id, as_of=at)
            assert rows(ledger) == before
        facts = collect(ledger, order.account_id)
        assert facts.authorities[0].first_complete_at == cutoff
    finally:
        ledger.close()


def test_no_temporal_audit_argument_can_hide_later_native_facts():
    ledger = account()
    try:
        with pytest.raises(TypeError):
            ledger.execution_compliance("backtest:test", as_of=T)
    finally:
        ledger.close()


@pytest.mark.parametrize("venue", ["binance", "kraken", "lean"])
def test_actual_native_and_lean_histories_audit_on_readonly_restart(tmp_path, venue):
    if venue == "lean":
        from tests.test_lean_execution import admit, setup_live

        runtime, engine, adapter, order, wire, native = setup_live(tmp_path)
        admit(runtime, engine, adapter, order)
        runtime.at += timedelta(seconds=1)
        runtime.add("10", "100", "7")
        wire.add_fill("actual-native", "10", "100", 2)
        wire.order.update(filled_qty="10", status="filled")
        engine.reconcile_all(as_of=runtime.at)
        native.close()
    else:
        from tests import test_binance_venue, test_kraken_venue

        fixtures = test_binance_venue if venue == "binance" else test_kraken_venue
        wire, engine, adapter, order = fixtures.setup(tmp_path)
        fixtures.admit(wire, engine, adapter, order)
        wire.now = T + timedelta(seconds=4)
        if venue == "binance":
            wire.add(1, "3", "100", ".01", asset="BTC")
        else:
            wire.add("trade-1", "3", "100", ".01", fee_asset="XXBT")
        engine.reconcile_all(as_of=wire.now)
        adapter.close()
    path = engine.ledger.path
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "PASS", audit
        assert audit.lineage == "native"


def test_raw_order_count_remains_blocking_without_any_individual_trade_or_report(tmp_path):
    from src.execution.alpaca_venue import _bounded
    from tests.test_alpaca_venue import admit, setup

    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.now = T + timedelta(seconds=4)
    wire.order.update(status="partially_filled", filled_qty="3")
    # Capture only the actual order response: no synthetic FILL/member is created.
    _bounded(type(adapter)._request)(adapter, "GET", "/v2/orders/remote-order-1")
    engine.ledger.db.execute("UPDATE venue_outbox SET state='new'")
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL"
        assert "invalid_retained_execution_source" in audit.blocking_codes
        assert not ledger.fills(order.account_id)


def test_later_conflicting_source_cannot_be_hidden_by_original_good_asof(tmp_path):
    from tests.test_kraken_venue import admit, setup

    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "3", "100", ".1")
    wire.now = T + timedelta(seconds=4)
    engine.reconcile_all(as_of=wire.now)
    assert engine.ledger.execution_compliance(order.account_id).status == "PASS"
    wire.trades["T1"]["fee"] = ".2"
    wire.now += timedelta(seconds=1)
    engine.reconcile_all(as_of=wire.now)
    engine.ledger.db.execute("UPDATE venue_outbox SET state='partial'")
    engine.ledger.db.execute("UPDATE venue_reports SET applied=1")
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL"
        assert "invalid_native_source_authority" in audit.blocking_codes


def test_canonical_lean_paper_binding_cannot_hide_native_ancestor_metadata(tmp_path):
    from tests.test_lean_execution import admit, setup

    runtime, engine, adapter, order = setup(tmp_path)
    admit(runtime, engine, adapter, order)
    clean = engine.ledger.execution_compliance(order.account_id)
    assert clean.status == "PASS" and clean.lineage == "lean-paper"
    with engine.ledger.atomic():
        engine.ledger.record_event(order.account_id, "lean-native-order:legacy", "{}")
    path = engine.ledger.path
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL" and audit.lineage == "unverified"


def test_honest_incomplete_report_requires_later_complete_actual_source_closure(tmp_path):
    from tests.test_kraken_venue import admit, setup

    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    wire.add("T1", "3", "100", ".1")
    wire.now = T + timedelta(seconds=3)
    wire.empty_trades = True
    engine.reconcile_all(as_of=wire.now)
    assert engine.ledger.execution_compliance(order.account_id).status == "FAIL"
    wire.empty_trades = False
    wire.now += timedelta(seconds=1)
    engine.reconcile_all(as_of=wire.now)
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "PASS", audit


@pytest.mark.parametrize("damage", ["key", "first-complete-clock"])
def test_discrepancy_metadata_cannot_relabel_native_identity_or_detection_clock(tmp_path, damage):
    path, order, _, cutoff = alpaca_receipt(tmp_path, expired=True)
    ledger = PortfolioLedger(path)
    case = detect_discrepancies(ledger, order.account_id, as_of=cutoff)[0]
    value = case.model_dump(mode="json")
    if damage == "key":
        value["key"]["broker_account_id"] = "another-remote-account"
    else:
        value["first_detected_at"] = case.first_observed_at.isoformat()
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?",
        (json.dumps(value), "execution-discrepancy:" + case.case_id),
    )
    before = rows(ledger)
    assert (
        "invalid_retained_discrepancy_metadata"
        in ledger.execution_compliance(order.account_id).blocking_codes
    )
    with pytest.raises(LedgerError, match="metadata conflicts"):
        detect_discrepancies(ledger, order.account_id, as_of=cutoff)
    assert rows(ledger) == before
    ledger.close()


def test_metadata_detection_failure_rolls_back_all_receipts_without_economic_changes(
    tmp_path, monkeypatch
):
    path, order, _, cutoff = alpaca_receipt(tmp_path, expired=True)
    ledger = PortfolioLedger(path)
    before = rows(ledger)
    original = ledger.record_event

    def fail_last(account_id, identity, payload):
        if identity.startswith("execution-discrepancy:"):
            raise RuntimeError("simulated process interruption before case receipt")
        return original(account_id, identity, payload)

    monkeypatch.setattr(ledger, "record_event", fail_last)
    with pytest.raises(RuntimeError, match="interruption"):
        detect_discrepancies(ledger, order.account_id, as_of=cutoff)
    assert rows(ledger) == before
    ledger.close()
    ledger = PortfolioLedger(path)
    try:
        assert len(detect_discrepancies(ledger, order.account_id, as_of=cutoff)) == 1
        assert ledger.execution_compliance(order.account_id).status == "FAIL"
    finally:
        ledger.close()


def test_missing_original_lean_deployment_is_explicit_unverified_lineage(tmp_path):
    from tests.test_lean_execution import admit, setup_live

    runtime, engine, adapter, order, wire, native = setup_live(tmp_path)
    admit(runtime, engine, adapter, order)
    engine.ledger.db.execute("DELETE FROM events WHERE id='remote-binding:lean'")
    path = engine.ledger.path
    native.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL" and audit.lineage == "unverified"
        assert "unverified_execution_lineage" in audit.blocking_codes


def test_legacy_booked_expired_fact_stays_noncompliant_even_after_math_reconciles(tmp_path):
    """Construct an old invalid book in the fixture; no runtime exception is added."""
    path, order, _, cutoff = alpaca_receipt(tmp_path, expired=True)
    ledger = PortfolioLedger(path)
    actual = collect(ledger, order.account_id).authorities[0].fill
    # A historical buggy importer could have booked the real source fact while
    # ignoring expiry. Only the fixture constructs that legacy database state.
    # Restore the exact original intent before any proof; journal/claim/native
    # receipt always retain the actual expired admission and original hash.
    with ledger.atomic():
        ledger.db.execute(
            "UPDATE orders SET payload=? WHERE account=? AND id=?",
            (
                order.model_copy(update={"expires_at": None}).model_dump_json(),
                order.account_id,
                order.intent_id,
            ),
        )
        ledger.apply_fill(actual, observed_at=cutoff)
        ledger.db.execute(
            "UPDATE orders SET payload=? WHERE account=? AND id=?",
            (order.model_dump_json(), order.account_id, order.intent_id),
        )
        ledger.db.execute("UPDATE venue_reports SET applied=1")
        ledger.db.execute("UPDATE venue_outbox SET state='partial',cumulative='3'")
    assert ledger.reconcile(order.account_id)["reconciled"]
    assert ledger.quantity(order.account_id, order.instrument.instrument_id) == 3
    ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL" and len(audit.violations) == 1
        assert not audit.blocking_codes and not audit.unresolved_sources
