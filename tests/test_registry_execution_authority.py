"""Current read-only ledger authority, independent of frozen research PASS claims."""

import json
import os
import shutil
from datetime import timedelta
from decimal import Decimal

import pytest

from src.execution.ledger import PortfolioLedger
from src.operations.reporting import research_view
from src.research.execution_authority import freeze, require_current
from src.research.registry import Activation, RegistryError, ResearchRegistry
from tests.test_research_registry import T, admitted, record_bound_evaluation


def setup(registry, key="one", parents=()):
    candidate, trial, evaluation, policy = admitted(registry, key, parent_ids=parents)
    record_bound_evaluation(registry, evaluation)
    return candidate, trial, evaluation, policy


def locator(registry, evaluation):
    from src.research.execution_authority import EvaluationExecutionBindings

    with registry._connection() as db:
        key = db.execute(
            "SELECT hash FROM evaluation_execution_bindings WHERE trial=?", (evaluation.trial_id,)
        ).fetchone()[0]
    return EvaluationExecutionBindings.model_validate_json(registry.blob(key))


def test_later_current_failure_blocks_resolve_rollback_status_and_descendants(tmp_path):
    reg = ResearchRegistry(tmp_path / "registry.db")
    first, _, original, policy = setup(reg)
    activation = reg.activate(
        "factor", "one", policy, expected_generation=0, at=T + timedelta(hours=2)
    )
    child, _, _, _ = setup(reg, "child", parents=("one",))
    reg.activate("factor", "child", policy, expected_generation=1, at=T + timedelta(hours=3))
    bound = locator(reg, original).ledgers[0]
    ledger = PortfolioLedger(bound.path)
    try:
        assert ledger.execution_compliance(bound.account_id).status == "PASS"
        ledger.record_event(bound.account_id, "alpaca-submit-attempt:unproved-native-history", "{}")
        assert ledger.reconcile(bound.account_id)["reconciled"]
        assert ledger.execution_compliance(bound.account_id).status == "FAIL"
    finally:
        ledger.close()
    reg = ResearchRegistry(reg.path)
    for operation in (
        lambda: reg.resolve(
            "factor", payload_hash=child.payload_hash, as_of=T + timedelta(hours=4)
        ),
        lambda: reg.active("factor", as_of=activation.activated_at),
        lambda: reg.rollback("factor", 1, expected_generation=2, at=T + timedelta(hours=4)),
        lambda: reg.activate(
            "other", "child", policy, expected_generation=0, at=T + timedelta(hours=4)
        ),
    ):
        with pytest.raises(RegistryError, match="execution"):
            operation()
    before = reg.path.read_bytes()
    view = research_view(reg.path)
    assert view["active"] == [] and len(view["blocked_activations"]) == 1
    assert len(view["activations"]) == 2
    assert reg.path.read_bytes() == before
    assert reg.status()["active"] == []
    assert reg.blob(original.replay_hash)
    assert reg.candidate(first.candidate_id) == first


@pytest.mark.parametrize(
    "fault",
    ["missing", "replacement", "hardlink", "symlink", "spec", "ownership", "binding-delete"],
)
def test_locator_missing_substituted_or_changed_ownership_is_not_current_pass(tmp_path, fault):
    reg = ResearchRegistry(tmp_path / "registry.db")
    _, _, evaluation, policy = setup(reg)
    binding = locator(reg, evaluation)
    item = binding.ledgers[0]
    from pathlib import Path

    path = Path(item.path)
    if fault == "missing":
        path.unlink()
    elif fault in ("replacement", "hardlink", "symlink"):
        other = tmp_path / "copy.db"
        shutil.copyfile(path, other)
        if fault == "replacement":
            os.replace(other, path)
        elif fault == "hardlink":
            other.unlink()
            os.link(path, other)
        else:
            path.unlink()
            path.symlink_to(other)
    elif fault == "binding-delete":
        with reg._connection() as db:
            db.execute("DROP TRIGGER immutable_evaluation_execution_bindings_DELETE")
            db.execute("DELETE FROM evaluation_execution_bindings")
    else:
        ledger = PortfolioLedger(path)
        try:
            if fault == "spec":
                spec = ledger.spec(item.account_id).model_copy(
                    update={"initial_cash": Decimal("12345")}
                )
                ledger.db.execute(
                    "UPDATE accounts SET spec=? WHERE id=?",
                    (spec.model_dump_json(), item.account_id),
                )
            else:
                ledger.record_event(item.account_id, "remote-binding:alpaca", "{}")
        finally:
            ledger.close()
    with pytest.raises(RegistryError, match="execution"):
        reg.activate("factor", "one", policy, expected_generation=0, at=T + timedelta(hours=2))
    assert not reg.active("factor")


def test_append_changes_prefix_not_immutable_account_ownership(tmp_path):
    reg = ResearchRegistry(tmp_path / "registry.db")
    _, _, evaluation, policy = setup(reg)
    binding = locator(reg, evaluation)
    item = binding.ledgers[0]
    ledger = PortfolioLedger(item.path)
    try:
        prior = ledger.execution_compliance(item.account_id)
        ledger.record_event(item.account_id, "inspection:additional-documentary-record", "{}")
        later = ledger.execution_compliance(item.account_id)
        assert prior.authority_prefix_hash != later.authority_prefix_hash
        assert prior.execution_binding_hash == later.execution_binding_hash
    finally:
        ledger.close()
    require_current(reg, evaluation, binding)
    fresh = freeze(reg, evaluation, ledger_path=item.path)
    reg.record_evaluation(evaluation, execution_binding=fresh)
    assert locator(reg, evaluation) == binding
    reg.activate("factor", "one", policy, expected_generation=0, at=T + timedelta(hours=2))


def test_legacy_outcome_is_audit_readable_but_cannot_acquire_new_locator(tmp_path):
    reg = ResearchRegistry(tmp_path / "registry.db")
    candidate, _, evaluation, policy = admitted(reg)
    reg.record_evaluation(evaluation)
    old = Activation(
        slot="factor",
        generation=1,
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash(),
        payload_hash=candidate.payload_hash,
        evaluation_hash=evaluation.content_hash(),
        policy_hash=policy.content_hash(),
        activated_at=T + timedelta(hours=2),
        action="promote",
        previous_generation=0,
    )
    with reg._connection() as db:
        db.execute("INSERT INTO activations VALUES (?,?,?)", ("factor", 1, old.model_dump_json()))
        db.execute("INSERT INTO active VALUES (?,?)", ("factor", 1))
    assert research_view(reg.path)["activations"] == [old.model_dump(mode="json")]
    with pytest.raises(RegistryError, match="execution"):
        reg.resolve("factor", payload_hash=candidate.payload_hash, as_of=T + timedelta(hours=3))
    binding = freeze(reg, evaluation, ledger_path=reg.path.parent / "registry-one.sqlite3")
    with pytest.raises(RegistryError, match="cannot be added"):
        reg.record_evaluation(evaluation, execution_binding=binding)
    assert json.loads(reg.blob(evaluation.manifest_hash))["account"]
