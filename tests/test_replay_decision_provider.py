"""Actual holdings-aware callback decisions on the canonical event/ledger path."""

import json
from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.core.backtester import (
    PortfolioReplay,
    canonical_hash,
    file_hash,
    load_replay_artifact,
    persist_replay_artifact,
    reconstruct_decision_inputs,
    recorded_decisions,
)
from src.core.benchmark import BenchmarkRunner, require_matched_replay
from src.core.validation import ValidationSuite
from src.domain.portfolio import CorporateAction, PortfolioTarget, TargetWeight
from src.execution.ledger import PortfolioLedger
from src.utils.exceptions import BacktestError
from tests.test_core_backtest import START, A, replay_inputs


def target(at, weight):
    return PortfolioTarget(
        target_id="actual-holdings:" + at.isoformat(),
        decision_at=at,
        universe_version="test-universe",
        strategy_id="actual-holdings",
        artifact_version="fixture-v1",
        data_snapshot_id="fixture",
        weights=(TargetWeight(instrument=A, weight=weight),),
    )


def inputs_for(provider):
    inputs = replay_inputs()
    inputs.pop("signals")
    inputs.update(
        decision_provider=provider,
        decision_times=[START + timedelta(days=d, hours=6) for d in (0, 1)],
        decision_recipe_hash="d" * 64,
    )
    return inputs


def trace(path, result):
    ledger = PortfolioLedger(path, read_only=True)
    try:
        return recorded_decisions(ledger, result)
    finally:
        ledger.close()


def test_partial_fills_reduction_uses_actual_holdings_and_records_none(tmp_path):
    seen = []

    def provider(at, snapshot):
        seen.append(snapshot)
        if at == START + timedelta(hours=6):
            return target(at, 1)
        if at == START + timedelta(days=1, hours=6):
            # Only 10 shares filled, not the 100 originally requested.
            assert snapshot.positions[0].quantity == 10
            assert snapshot.cash == 9000
            assert snapshot.reserved_cash == 9000
            return target(at, snapshot.positions[0].market_value / snapshot.nav / 2)
        return None

    inputs = inputs_for(provider)
    inputs["decision_times"].append(START + timedelta(days=2, hours=6))
    inputs["bars"] = [
        b.model_copy(
            update={"high": D(100), "low": D(100), "close": D(100), "opening_volume": D(1000)}
        )
        if b.instrument == A and b.opened_at == START + timedelta(days=1)
        else b.model_copy(update={"open": D(100), "high": D(100)})
        if b.instrument == A and b.opened_at == START + timedelta(days=2)
        else b
        for b in inputs["bars"]
    ]
    path = tmp_path / "partial.sqlite3"
    result = PortfolioReplay(path).run(**inputs)
    assert [D(t["quantity"]) for t in result.trades] == [10, 5]
    assert [t["side"] for t in result.trades] == ["buy", "sell"]
    assert result.snapshots[-1].positions[0].quantity == 5
    assert result.snapshots[-1].nav == 9940
    assert result.metrics.total_return < 0
    records = trace(path, result)
    assert [r.snapshot for r in records] == seen
    assert records[-1].target is None
    assert result.manifest["decision_records_hash"] == canonical_hash(
        [r.model_dump(mode="json") for r in records]
    )
    reference = persist_replay_artifact(result, path, tmp_path / "result.json")
    assert load_replay_artifact(reference) == result
    old_hash = file_hash(path)
    assert PortfolioReplay(path).run(**inputs) == result
    assert len(seen) == 3  # finished retries do not call user code again
    assert file_hash(path) == old_hash
    restored = reconstruct_decision_inputs(
        result, inputs, ledger_path=path, decision_provider=provider
    )
    assert restored["decision_provider"] is provider and restored["targets"] is None


def test_after_actions_and_open_fills_but_cannot_fill_same_open(tmp_path):
    seen = []

    def provider(at, snapshot):
        seen.append(snapshot)
        return target(at, D(".5")) if not snapshot.positions else None

    inputs = inputs_for(provider)
    inputs["decision_times"] = [
        START,
        START + timedelta(days=1),
        START + timedelta(days=1, hours=6),
    ]
    inputs["actions"] = [
        CorporateAction(
            action_id="split",
            instrument=A,
            kind="split",
            effective_at=START + timedelta(days=1, hours=6),
            source_id="fixture-split",
            value=2,
        )
    ]
    result = PortfolioReplay(tmp_path / "causal.sqlite3").run(**inputs)
    assert result.trades[0]["executed_at"] == (START + timedelta(days=1)).isoformat().replace(
        "+00:00", "Z"
    )
    assert not seen[0].positions
    assert seen[1].positions[0].quantity == 50
    assert seen[1].positions[0].opened_at == START + timedelta(days=1)
    assert seen[2].positions[0].quantity == 100
    assert seen[2].positions[0].opened_at == seen[1].positions[0].opened_at
    assert result.reconciliation["reconciled"]


def test_future_perturbation_and_callback_failure_restart(tmp_path):
    def provider(at, snapshot):
        return target(at, D(".5")) if not snapshot.positions else None

    inputs = inputs_for(provider)
    reference = PortfolioReplay(tmp_path / "original.sqlite3").run(**inputs)
    changed = dict(inputs)
    changed["bars"] = [
        b.model_copy(update={"close": D(20), "low": D(20)})
        if b.opened_at > inputs["decision_times"][-1]
        else b
        for b in inputs["bars"]
    ]
    perturbed = PortfolioReplay(tmp_path / "changed.sqlite3").run(**changed)
    assert trace(tmp_path / "original.sqlite3", reference) == trace(
        tmp_path / "changed.sqlite3", perturbed
    )
    assert reference.equity_curve[:-1] == perturbed.equity_curve[:-1]

    def fails(at, snapshot):
        if at == inputs["decision_times"][-1]:
            raise RuntimeError("actual strategy failed")
        return provider(at, snapshot)

    broken = tmp_path / "broken.sqlite3"
    with pytest.raises(BacktestError, match="actual strategy failed"):
        PortfolioReplay(broken).run(**{**inputs, "decision_provider": fails})
    ledger = PortfolioLedger(broken, read_only=True)
    assert not list(ledger.db.execute("SELECT * FROM runs"))  # no successful performance
    assert len(list(ledger.db.execute("SELECT * FROM events WHERE id LIKE 'run-failure:%'"))) == 1
    assert (
        len(list(ledger.db.execute("SELECT * FROM events WHERE id LIKE 'provider-decision:%'")))
        == 1
    )
    ledger.close()
    assert PortfolioReplay(broken).run(**inputs) == reference
    with pytest.raises(BacktestError, match="inputs changed"):
        PortfolioReplay(broken).run(**{**inputs, "decision_recipe_hash": "e" * 64})


def test_changed_record_even_rehashed_cannot_change_observed_holdings(tmp_path):
    inputs = inputs_for(
        lambda at, snapshot: target(at, D(".5")) if not snapshot.positions else None
    )
    path = tmp_path / "tamper.sqlite3"
    result = PortfolioReplay(path).run(**inputs)
    records = trace(path, result)
    changed_snapshot = records[1].snapshot.model_copy(update={"reserved_cash": D(42)})
    records[1] = records[1].model_copy(
        update={"snapshot": changed_snapshot, "snapshot_hash": changed_snapshot.content_hash()}
    )
    altered = result.model_copy(
        update={
            "manifest": {
                **result.manifest,
                "decision_records_hash": canonical_hash(
                    [r.model_dump(mode="json") for r in records]
                ),
            }
        }
    )
    ledger = PortfolioLedger(path)
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?",
        (records[1].model_dump_json(), "provider-decision:" + records[1].at.isoformat()),
    )
    ledger.db.execute(
        "UPDATE runs SET payload=?",
        (
            json.dumps(
                {
                    "status": "complete",
                    "manifest": altered.manifest,
                    "manifest_hash": canonical_hash(altered.manifest),
                    "result": altered.model_dump(mode="json"),
                }
            ),
        ),
    )
    ledger.close()
    report = ValidationSuite().validate(altered, ledger_path=path, _accounting_only=True)
    assert report.overall == "NOT RELIABLE"
    assert "actual causal replay state" in report.summary


def test_cost_scenarios_rerun_recipe_with_changed_holdings(tmp_path):
    def provider(at, snapshot):
        if not snapshot.positions:
            return target(at, 1)
        # Actual first fill differs across cost scenarios. This is deliberately
        # holdings-dependent, not a replay of the baseline's generated targets.
        return target(at, D(".5") if snapshot.positions[0].quantity == 100 else D(".4"))

    inputs = inputs_for(provider)
    baseline = PortfolioReplay(tmp_path / "baseline.sqlite3").run(**inputs)
    costs = BenchmarkRunner().cost_sensitivity(inputs=inputs, output_dir=tmp_path / "costs")
    assert len({c.manifest["targets_hash"] for c in costs}) == 2
    for candidate in costs:
        require_matched_replay(baseline, candidate, cost_variant=True)
    refs = json.loads((tmp_path / "costs" / "cost-artifacts.json").read_text())
    report = ValidationSuite().validate(
        baseline, ledger_path=tmp_path / "baseline.sqlite3", cost_artifacts=refs
    )
    assert next(c for c in report.checks if c.name == "cost_sensitivity").verdict.value == "PASS"


@pytest.mark.parametrize(
    "change",
    [
        {"decision_times": []},
        {"decision_times": [START.replace(tzinfo=None)]},
        {"decision_times": [START, START]},
        {"decision_recipe_hash": "not-a-digest"},
        {"signals": replay_inputs()["signals"]},
    ],
)
def test_invalid_provider_inputs_fail(tmp_path, change):
    with pytest.raises((BacktestError, ValueError)):
        PortfolioReplay(tmp_path / "invalid.sqlite3").run(
            **{**inputs_for(lambda at, snapshot: None), **change}
        )


def test_wrong_target_clock_is_not_accepted(tmp_path):
    inputs = inputs_for(lambda at, snapshot: target(at + timedelta(seconds=1), 1))
    with pytest.raises(BacktestError, match="actual decision clock"):
        PortfolioReplay(tmp_path / "wrong.sqlite3").run(**inputs)
