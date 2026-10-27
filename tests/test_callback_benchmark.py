"""Durable callback selection controls; synthetic fixtures live only in tests."""

import json
from datetime import timedelta
from decimal import Decimal as D

import numpy as np
import pytest

from src.core.backtester import (
    PortfolioReplay,
    canonical_hash,
    file_hash,
    load_replay_artifact,
    recorded_decisions,
)
from src.core.benchmark import BenchmarkRunner, verify_comparison
from src.domain.portfolio import FillPolicy, PortfolioTarget, TargetWeight
from src.execution.ledger import PortfolioLedger
from src.utils.exceptions import BacktestError
from tests.test_core_backtest import START, A, B
from tests.test_core_benchmark import named_inputs


def callback_case(path, *, reduce_off_cadence=False):
    args = named_inputs()
    args.pop("signals")
    times = [START + timedelta(days=i, hours=6) for i in range(3)]

    def provider(at, snapshot):
        if at == times[-1]:
            return None
        hold = next((p for p in snapshot.positions if p.instrument == A), None)
        weights = [
            TargetWeight(instrument=A, weight=hold.market_value / snapshot.nav if hold else D(".4"))
        ]
        if at != times[0] and reduce_off_cadence:
            weights = [TargetWeight(instrument=A, weight=hold.market_value / snapshot.nav / 2)]
        elif at != times[0]:
            weights.append(TargetWeight(instrument=B, weight=D(".1")))
        return PortfolioTarget(
            target_id="baseline:" + at.isoformat(),
            decision_at=at,
            universe_version="test-universe",
            strategy_id="fixture",
            artifact_version="fixture",
            data_snapshot_id="fixture",
            weights=tuple(weights),
            preserve_quantities={A.instrument_id: hold.quantity}
            if hold and not reduce_off_cadence
            else {},
        )

    opportunities = {
        "schema_version": "1.0",
        "implementation": "registered-strategy-opportunities-1",
        "inputs_hash": "a" * 64,
        "plan_hash": "b" * 64,
        "feature_data_hash": "c" * 64,
        "feature_spec_hash": "d" * 64,
        "cadence": "weekly" if reduce_off_cadence else "daily",
        "decision_times": [t.isoformat() for t in times],
        "rows": [
            {
                "at": at.isoformat(),
                "universe_hash": args["universes"][0].content_hash(),
                "universe_version": "test-universe",
                "model_payload_hash": "e" * 64,
                "model_artifact_hash": "f" * 64,
                "regime_observation_hash": "1" * 64,
                "rebalance_due": not reduce_off_cadence or at == times[0],
                "eligible": [
                    {"instrument_id": key, "prediction": value}
                    for key, value in [("A", 0.2), ("B", 0.1)]
                ],
                "trace_link": {
                    "account_id": args["account"].account_id,
                    "run_id": args["context"].run_id,
                    "at": at.isoformat(),
                },
            }
            for at in times
        ],
    }
    path.mkdir(parents=True, exist_ok=True)
    source = path / "opportunities.json"
    source.write_text(json.dumps(opportunities, sort_keys=True))
    args.update(
        decision_provider=provider,
        decision_times=times,
        decision_recipe_hash=canonical_hash({"source": file_hash(source), "recipe": "test"}),
        fill_policy=FillPolicy(equity_cost_bps=10, crypto_cost_bps=20),
        provenance={
            "features_hash": "c" * 64,
            "opportunities_artifact": {"path": str(source), "sha256": file_hash(source)},
        },
    )
    return args


def test_1000_callback_controls_persist_real_nav_fills_and_exact_passive_quantities(tmp_path):
    args = callback_case(tmp_path / "inputs")
    ledger_path = tmp_path / "baseline.sqlite3"
    baseline = PortfolioReplay(ledger_path).run(**args)
    result = BenchmarkRunner().compare_replay(
        baseline,
        inputs=args,
        ledger_path=ledger_path,
        output_dir=tmp_path / "comparison",
        n_simulations=1000,
    )
    verify_comparison(baseline, result)
    assert len(result.random_artifacts) == 1000
    assert result.signal_artifact == {}
    assert result.strategy_artifact and result.opportunity_artifact
    assert not result.unavailable
    assert result.monte_carlo_percentile is not None
    assert np.std(result.random_sharpes) > 0
    assert all(nav[-1] < nav[0] for nav in result.random_nav)
    candidate = load_replay_artifact(result.random_artifacts[0])
    assert candidate.metrics.fees > 0
    assert candidate.metrics.fees == pytest.approx(sum(float(t["fee"]) for t in candidate.trades))
    with_ledger = PortfolioLedger(result.random_artifacts[0].ledger_path, read_only=True)
    try:
        trace = recorded_decisions(with_ledger, candidate)
    finally:
        with_ledger.close()
    actual = trace[1].snapshot.positions[0]
    assert trace[1].target.preserve_quantities[actual.instrument.instrument_id] == actual.quantity
    assert trace[-1].target is None
    assert candidate.manifest["signals_hash"] == canonical_hash([])
    assert {t["side"] for t in candidate.trades} == {"buy"}
    assert set(result.benchmark_artifacts) == {"SPY", "BTC", "60/40", "equal-weight"}


def test_callback_comparison_requires_actual_baseline_ledger_and_original_opportunity_bytes(
    tmp_path,
):
    args = callback_case(tmp_path / "inputs")
    path = tmp_path / "baseline.sqlite3"
    baseline = PortfolioReplay(path).run(**args)
    with pytest.raises(BacktestError, match="ledger_path"):
        BenchmarkRunner().compare_replay(
            baseline, inputs=args, output_dir=tmp_path / "absent", n_simulations=2
        )
    source = args["provenance"]["opportunities_artifact"]["path"]
    from pathlib import Path

    payload = json.loads(Path(source).read_text())
    payload["rows"][0]["eligible"][0]["prediction"] = 999
    Path(source).write_text(json.dumps(payload))
    with pytest.raises(BacktestError, match="original bound"):
        BenchmarkRunner().compare_replay(
            baseline,
            inputs=args,
            ledger_path=path,
            output_dir=tmp_path / "changed",
            n_simulations=2,
        )


def test_callback_comparison_verifier_rejects_changed_seed_summary_or_cadence(tmp_path):
    args = callback_case(tmp_path / "inputs")
    path = tmp_path / "baseline.sqlite3"
    baseline = PortfolioReplay(path).run(**args)
    result = BenchmarkRunner().compare_replay(
        baseline, inputs=args, ledger_path=path, output_dir=tmp_path / "comparison", n_simulations=5
    )
    verify_comparison(baseline, result, minimum_simulations=5)
    for changed, message in (
        (result.model_copy(update={"seed": 43}), "recipe/schedule"),
        (result.model_copy(update={"random_sharpes": [1.0] * 5}), "summaries"),
        (result.model_copy(update={"match": {**result.match, "cadence": "weekly"}}), "cadence"),
    ):
        with pytest.raises(BacktestError, match=message):
            verify_comparison(baseline, changed, minimum_simulations=5)


def test_off_cadence_reduction_keeps_control_identity_and_future_prices_do_not_change_first_selection(
    tmp_path,
):
    args = callback_case(tmp_path / "inputs", reduce_off_cadence=True)
    path = tmp_path / "baseline.sqlite3"
    baseline = PortfolioReplay(path).run(**args)
    comparison = BenchmarkRunner().compare_replay(
        baseline, inputs=args, ledger_path=path, output_dir=tmp_path / "comparison", n_simulations=3
    )
    first_targets = []
    for ref in comparison.random_artifacts:
        candidate = load_replay_artifact(ref)
        ledger = PortfolioLedger(ref.ledger_path, read_only=True)
        try:
            records = recorded_decisions(ledger, candidate)
        finally:
            ledger.close()
        first_targets.append(records[0].target.weights)
        assert records[0].target.weights[0].instrument == records[1].target.weights[0].instrument
        assert [trade["side"] for trade in candidate.trades] == ["buy", "sell"]
    args["bars"] = [
        bar.model_copy(
            update={
                "open": bar.open * 2,
                "high": bar.high * 2,
                "low": bar.low * 2,
                "close": bar.close * 2,
            }
        )
        if bar.opened_at > args["decision_times"][0]
        else bar
        for bar in args["bars"]
    ]
    changed_path = tmp_path / "future.sqlite3"
    changed = PortfolioReplay(changed_path).run(**args)
    repeated = BenchmarkRunner().compare_replay(
        changed,
        inputs=args,
        ledger_path=changed_path,
        output_dir=tmp_path / "changed-comparison",
        n_simulations=3,
    )
    for target, ref in zip(first_targets, repeated.random_artifacts, strict=True):
        candidate = load_replay_artifact(ref)
        ledger = PortfolioLedger(ref.ledger_path, read_only=True)
        try:
            assert recorded_decisions(ledger, candidate)[0].target.weights == target
        finally:
            ledger.close()


def test_explicit_historical_advice_opportunities_need_no_invented_model_or_prediction(tmp_path):
    from pathlib import Path

    from src.core.benchmark import CallbackOpportunities

    args = callback_case(tmp_path / "inputs")
    path = Path(args["provenance"]["opportunities_artifact"]["path"])
    numeric = json.loads(path.read_text())
    invalid = json.loads(path.read_text())
    invalid["rows"][0]["eligible"][0]["prediction"] = None
    with pytest.raises(ValueError, match="actual predictions"):
        CallbackOpportunities.model_validate(invalid)
    policy = "historical-advice-current-ledger-sizing-1"
    hashes = {key: canonical_hash({"fixture_decision": key}) for key in ("A", "B")}
    original = {
        value: canonical_hash({"fixture_original_snapshot": key}) for key, value in hashes.items()
    }
    numeric.update(advice_policy=policy, actual_decision_hashes=sorted(hashes.values()))
    for row in numeric["rows"]:
        row.update(model_payload_hash=None, model_artifact_hash=None)
        for candidate in row["eligible"]:
            decision_hash = hashes[candidate["instrument_id"]]
            candidate.update(
                prediction=None,
                decision_hash=decision_hash,
                receipt_hash=canonical_hash({"fixture_receipt": decision_hash}),
                consensus_action="BUY",
                original_review_snapshot_hash=original[decision_hash],
            )
    parsed = CallbackOpportunities.model_validate(numeric)
    assert all(c.prediction is None for r in parsed.rows for c in r.eligible)
    path.write_text(json.dumps(numeric, sort_keys=True))
    args["provenance"].update(
        advice_policy=policy,
        actual_decision_hashes=sorted(hashes.values()),
        original_review_snapshots=original,
    )
    args["provenance"]["opportunities_artifact"]["sha256"] = file_hash(path)
    args["decision_recipe_hash"] = canonical_hash(
        {"source": file_hash(path), "recipe": "fixture-advice"}
    )
    ledger_path = tmp_path / "advice.sqlite3"
    baseline = PortfolioReplay(ledger_path).run(**args)
    comparison = BenchmarkRunner().compare_replay(
        baseline,
        inputs=args,
        ledger_path=ledger_path,
        output_dir=tmp_path / "comparison",
        n_simulations=5,
    )
    verify_comparison(baseline, comparison, minimum_simulations=5)
    assert (
        baseline.trades
        and baseline.walk_forward_results == []
        and baseline.cross_sectional_ic == {}
    )
    changed = baseline.model_copy(
        update={
            "manifest": {
                **baseline.manifest,
                "provenance": {**baseline.manifest["provenance"], "original_review_snapshots": {}},
            }
        }
    )
    with pytest.raises(BacktestError, match="binding differs"):
        verify_comparison(changed, comparison, minimum_simulations=5)
    numeric["rows"][0]["model_payload_hash"] = "e" * 64
    with pytest.raises(ValueError, match="cannot invent model"):
        CallbackOpportunities.model_validate(numeric)
