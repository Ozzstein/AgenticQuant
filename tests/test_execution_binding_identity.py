"""Immutable operator locator identities are distinct from growing source state."""

import json
from datetime import timedelta

import pytest

from src.domain.portfolio import AccountSpec
from tests.test_alpaca_venue import admit, setup
from tests.test_durable_accounting import T
from tests.test_native_authority import readonly_proof


def test_binding_identity_survives_actual_capture_admission_fills_restart_and_other_account(
    tmp_path,
):
    wire, engine, adapter, order = setup(tmp_path)
    original = engine.ledger.execution_compliance(order.account_id)
    assert original.account_spec_hash == engine.ledger.spec(order.account_id).content_hash()
    admit(wire, engine, adapter, order)
    wire.now = T + timedelta(seconds=3)
    wire.add_fill("actual-binding-trade", "3", "100", 2)
    wire.order.update(status="partially_filled", filled_qty="3")
    engine.reconcile_all(as_of=wire.now)
    current = engine.ledger.execution_compliance(order.account_id)
    assert current.status == "PASS"
    assert current.account_spec_hash == original.account_spec_hash
    assert current.execution_binding_hash == original.execution_binding_hash
    assert current.authority_prefix_hash != original.authority_prefix_hash
    engine.ledger.create_account(
        AccountSpec(
            account_id="paper:another",
            mode="paper",
            base_currency="USD",
            initial_cash="100",
            opened_at=T,
        )
    )
    assert engine.ledger.execution_compliance(order.account_id) == current
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        assert ledger.execution_compliance(order.account_id) == current


@pytest.mark.parametrize(
    "changed", ["owner", "configuration", "broker-account", "instrument", "initial-journal"]
)
def test_same_account_spec_cannot_hide_changed_original_execution_binding(tmp_path, changed):
    wire, engine, adapter, order = setup(tmp_path)
    original = engine.ledger.execution_compliance(order.account_id)
    if changed == "owner":
        engine.ledger.db.execute("UPDATE execution_books SET owner='lean'")
    elif changed == "configuration":
        row = engine.ledger.db.execute("SELECT configuration FROM execution_books").fetchone()
        value = json.loads(row[0])
        value["max_balance_age_seconds"] = 99
        engine.ledger.db.execute("UPDATE execution_books SET configuration=?", (json.dumps(value),))
    elif changed == "initial-journal":
        engine.ledger.db.execute(
            "UPDATE journal SET at=? WHERE id='initial'", ((T + timedelta(seconds=1)).isoformat(),)
        )
    else:
        identity = (
            "remote-binding:alpaca"
            if changed == "broker-account"
            else "remote-instrument:alpaca:" + order.instrument.instrument_id
        )
        value = json.loads(
            engine.ledger.db.execute(
                "SELECT payload FROM events WHERE id=?", (identity,)
            ).fetchone()[0]
        )
        value["broker_account_id" if changed == "broker-account" else "symbol"] = (
            "different-original-identity"
        )
        engine.ledger.db.execute(
            "UPDATE events SET payload=? WHERE id=?", (json.dumps(value), identity)
        )
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        current = ledger.execution_compliance(order.account_id)
        assert current.account_spec_hash == original.account_spec_hash
        assert current.execution_binding_hash != original.execution_binding_hash


def test_native_credential_binding_is_frozen_without_any_plaintext_secret(tmp_path):
    from tests.test_binance_venue import setup

    wire, engine, adapter, order = setup(tmp_path)
    original = engine.ledger.execution_compliance(order.account_id)
    row = engine.ledger.db.execute(
        "SELECT payload FROM events WHERE id='remote-binding:binance'"
    ).fetchone()
    value = json.loads(row[0])
    assert "credential_identity" in value
    value["credential_identity"] = "f" * 64
    engine.ledger.db.execute(
        "UPDATE events SET payload=? WHERE id='remote-binding:binance'", (json.dumps(value),)
    )
    current = engine.ledger.execution_compliance(order.account_id)
    assert current.execution_binding_hash != original.execution_binding_hash
    assert current.account_spec_hash == original.account_spec_hash
    adapter.close()
    engine.ledger.close()


def test_current_account_spec_hash_is_distinct_from_original_binding_hash(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    original = engine.ledger.execution_compliance(order.account_id)
    changed = engine.ledger.spec(order.account_id).model_copy(update={"base_currency": "EUR"})
    engine.ledger.db.execute(
        "UPDATE accounts SET spec=? WHERE id=?", (changed.model_dump_json(), order.account_id)
    )
    path = engine.ledger.path
    adapter.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        current = ledger.execution_compliance(order.account_id)
        assert current.account_spec_hash == changed.content_hash() != original.account_spec_hash
        assert current.execution_binding_hash == original.execution_binding_hash
        assert current.status == "FAIL"


def test_lean_runtime_epoch_growth_keeps_original_deployment_identity(tmp_path):
    from tests.test_lean_execution import setup_live

    runtime, engine, adapter, order, wire, native = setup_live(tmp_path)
    original = engine.ledger.execution_compliance(order.account_id)
    runtime.runtime_id = "second-actual-fixture-runtime"
    adapter._runtime_binding()
    current = engine.ledger.execution_compliance(order.account_id)
    assert current.status == "PASS"
    assert current.execution_binding_hash == original.execution_binding_hash
    assert current.authority_prefix_hash != original.authority_prefix_hash
    path = engine.ledger.path
    native.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        assert ledger.execution_compliance(order.account_id) == current
