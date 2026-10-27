"""Validation cannot turn missing evidence or shuffled returns into approval."""

import copy

from src.core.backtester import BacktestValidator, PortfolioReplay
from src.core.validation import ValidationSuite
from src.utils.schemas import BacktestResult, CheckVerdict, ValidationVerdict
from tests.test_core_backtest import replay_inputs


def test_legacy_results_never_pass():
    report = ValidationSuite().validate(BacktestResult())
    assert report.overall == "NOT RELIABLE"
    assert report.checks[0].name == "engine_version"
    assert BacktestValidator().validate(BacktestResult()).verdict == ValidationVerdict.REJECTED


def test_missing_evidence_blocks_even_reconciled_nav(tmp_path):
    path = tmp_path / "ledger.db"
    result = PortfolioReplay(path).run(**replay_inputs())
    report = ValidationSuite().validate(result, ledger_path=path)
    verdict = {c.name: c.verdict for c in report.checks}
    assert verdict["nav_arithmetic"] == CheckVerdict.PASS
    assert verdict["durable_accounting"] == CheckVerdict.PASS
    assert verdict["fill_realism"] == CheckVerdict.PASS
    assert verdict["historical_provenance"] == CheckVerdict.FAIL
    assert verdict["purged_oos"] == CheckVerdict.FAIL
    assert verdict["statistical_significance"] == CheckVerdict.FAIL
    assert report.overall == "NOT RELIABLE"


def test_shuffling_returns_and_tampering_nav_fail_arithmetic(tmp_path):
    path = tmp_path / "ledger.db"
    result = PortfolioReplay(path).run(**replay_inputs())
    changed = result.model_copy(update={"daily_returns": list(reversed(result.daily_returns))})
    report = ValidationSuite().validate(changed, ledger_path=path)
    assert next(c for c in report.checks if c.name == "nav_arithmetic").verdict == CheckVerdict.FAIL
    changed = result.model_copy(update={"equity_curve": [10000] * len(result.equity_curve)})
    report = ValidationSuite().validate(changed, ledger_path=path)
    assert report.overall == "NOT RELIABLE"


def test_future_fill_and_no_ledger_are_failures(tmp_path):
    result = PortfolioReplay(tmp_path / "ledger.db").run(**replay_inputs())
    trades = copy.deepcopy(result.trades)
    trades[0]["executed_at"] = trades[0]["decision_at"]
    result = result.model_copy(update={"trades": trades})
    report = ValidationSuite().validate(result)
    verdict = {c.name: c.verdict for c in report.checks}
    assert verdict["fill_realism"] == CheckVerdict.FAIL
    assert verdict["durable_accounting"] == CheckVerdict.FAIL


def test_metric_manifest_and_full_fill_tampering_fail(tmp_path):
    path = tmp_path / "ledger.db"
    result = PortfolioReplay(path).run(**replay_inputs())
    for name in (
        "sharpe_ratio",
        "annual_return",
        "volatility",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "fees",
        "turnover",
        "total_trades",
    ):
        value = getattr(result.metrics, name)
        changed = result.model_copy(
            update={"metrics": result.metrics.model_copy(update={name: (value or 0) + 5})}
        )
        report = ValidationSuite().validate(changed, ledger_path=path)
        verdict = {c.name: c.verdict for c in report.checks}
        assert verdict["nav_arithmetic"] == CheckVerdict.FAIL, name
        assert verdict["durable_accounting"] == CheckVerdict.FAIL, name
    changed = copy.deepcopy(result.manifest)
    changed["context"]["code_version"] = "tampered"
    report = ValidationSuite().validate(
        result.model_copy(update={"manifest": changed}), ledger_path=path
    )
    assert (
        next(c for c in report.checks if c.name == "durable_accounting").verdict
        == CheckVerdict.FAIL
    )

    trades = copy.deepcopy(result.trades)
    trades[0]["price"] = "999"
    report = ValidationSuite().validate(
        result.model_copy(update={"trades": trades}), ledger_path=path
    )
    assert (
        next(c for c in report.checks if c.name == "durable_accounting").verdict
        == CheckVerdict.FAIL
    )


def test_model_free_requires_injected_authority_bound_to_exact_replay(tmp_path):
    """Synthetic verifier boundary test; genuine archives are verified by G tests."""
    from src.core.backtester import canonical_hash
    from src.core.validation import AdviceValidationProof
    from tests.test_replay_decision_provider import inputs_for, target

    args = inputs_for(lambda at, snapshot: target(at, ".5"))
    args["provenance"] = {
        "advice_policy": "historical-advice-current-ledger-sizing-1",
        "actual_decision_hashes": ["a" * 64],
        "original_review_snapshots": {"a" * 64: "b" * 64},
        "folds": [],
    }
    path = tmp_path / "advice.sqlite3"
    result = PortfolioReplay(path).run(**args)
    proof = AdviceValidationProof(
        manifest_hash=canonical_hash(result.manifest),
        decision_records_hash=result.manifest["decision_records_hash"],
        actual_decision_hashes=("a" * 64,),
        receipt_bindings=(("a" * 64, "c" * 64, "b" * 64),),
    )
    seen = []

    def verifier(actual, ledger_path):
        seen.append((actual, ledger_path))
        return proof

    report = ValidationSuite().validate(result, ledger_path=path, advice_verifier=verifier)
    checks = {c.name: c.verdict for c in report.checks}
    assert seen == [(result, path)]
    assert checks["authoritative_advice_history"] == CheckVerdict.PASS
    assert {"purged_oos", "factor_ic", "factor_correlation"}.isdisjoint(checks)
    assert checks["durable_accounting"] == CheckVerdict.PASS
    for required in (
        "historical_provenance",
        "cost_sensitivity",
        "matched_benchmarks",
        "trial_holdout_registry",
        "statistical_significance",
        "regime_dependence",
    ):
        assert checks[required] == CheckVerdict.FAIL
    assert report.overall == "NOT RELIABLE"
    for invalid in (
        None,
        True,
        proof.model_dump(),
        proof.model_copy(update={"manifest_hash": "e" * 64}),
        proof.model_copy(update={"decision_records_hash": "e" * 64}),
        proof.model_copy(update={"actual_decision_hashes": ()}),
        proof.model_copy(update={"receipt_bindings": ()}),
        proof.model_copy(update={"receipt_bindings": proof.receipt_bindings * 2}),
        proof.model_copy(update={"receipt_bindings": (("a" * 64, "c" * 64, "e" * 64),)}),
        proof.model_copy(update={"receipt_bindings": (("d" * 64, "c" * 64, "b" * 64),)}),
    ):
        report = ValidationSuite().validate(
            result,
            ledger_path=path,
            advice_verifier=(lambda *args: invalid) if invalid is not None else None,
        )
        assert (
            next(c for c in report.checks if c.name == "authoritative_advice_history").verdict
            == CheckVerdict.FAIL
        )
        assert report.overall == "NOT RELIABLE"

    def unavailable(*args):
        raise RuntimeError("fixture authority unavailable")

    report = ValidationSuite().validate(result, ledger_path=path, advice_verifier=unavailable)
    assert (
        next(c for c in report.checks if c.name == "authoritative_advice_history").verdict
        == CheckVerdict.FAIL
    )
    before = len(seen)
    report = ValidationSuite().validate(result, advice_verifier=verifier)
    assert len(seen) == before  # missing canonical ledger cannot be approved by an authority
    assert report.overall == "NOT RELIABLE"
    invented = result.model_copy(update={"cross_sectional_ic": {"2024-01-02": .8}})
    report = ValidationSuite().validate(invented, ledger_path=path, advice_verifier=verifier)
    claim = next(c for c in report.checks if c.name == "authoritative_advice_history")
    assert claim.verdict == CheckVerdict.FAIL and "cannot claim numerical" in claim.explanation
    assert len(seen) == before


def test_numeric_validation_cannot_use_advice_verifier_to_bypass_fold_gates(tmp_path):
    path = tmp_path / "numeric.sqlite3"
    result = PortfolioReplay(path).run(**replay_inputs())

    def unexpected(*args):
        raise AssertionError("numeric path must never request model-free approval")

    report = ValidationSuite().validate(result, ledger_path=path, advice_verifier=unexpected)
    checks = {c.name: c.verdict for c in report.checks}
    assert "authoritative_advice_history" not in checks
    assert all(
        checks[name] == CheckVerdict.FAIL
        for name in ("purged_oos", "factor_ic", "factor_correlation")
    )
