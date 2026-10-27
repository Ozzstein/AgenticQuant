"""Actual persisted comparison artifacts, tamper rejection and OOS reconstruction."""

import json
from pathlib import Path

import pytest
from sklearn.linear_model import Ridge
from typer.testing import CliRunner

from scripts.run_backtest import app
from src.core.backtester import (
    PortfolioReplay,
    ReplayArtifact,
    file_hash,
    load_replay_artifact,
    persist_replay_artifact,
    require_matching_inputs,
)
from src.core.benchmark import BenchmarkRunner, verify_comparison
from src.core.validation import ValidationSuite
from src.core.walk_forward import WalkForwardRunner, reconstruct_oos_inputs
from src.utils.exceptions import BacktestError
from src.utils.schemas import CheckVerdict
from tests.test_core_backtest import write_bundle
from tests.test_core_benchmark import named_inputs, varied_inputs
from tests.test_walk_forward import wf_case


def verdict(report, name):
    return next(c.verdict for c in report.checks if c.name == name)


def test_cost_summaries_require_durable_bindings_and_actual_content(tmp_path):
    inputs = varied_inputs()
    base = PortfolioReplay(tmp_path / "base.db").run(**inputs)
    costs = BenchmarkRunner().cost_sensitivity(inputs=inputs, output_dir=tmp_path / "costs")
    report = ValidationSuite().validate(base, ledger_path=tmp_path / "base.db", cost_results=costs)
    assert verdict(report, "cost_sensitivity") == CheckVerdict.FAIL
    references = [
        ReplayArtifact.model_validate(r)
        for r in json.loads((tmp_path / "costs/cost-artifacts.json").read_text())
    ]
    report = ValidationSuite().validate(
        base, ledger_path=tmp_path / "base.db", cost_results=costs, cost_artifacts=references
    )
    assert verdict(report, "cost_sensitivity") == CheckVerdict.PASS
    path = Path(references[1].result_path)
    payload = json.loads(path.read_text())
    payload["metrics"]["sharpe_ratio"] = 1000
    path.write_text(json.dumps(payload))
    # Updating just the content digest does not bypass economic verification.
    references[1] = references[1].model_copy(update={"result_sha256": file_hash(path)})
    report = ValidationSuite().validate(
        base, ledger_path=tmp_path / "base.db", cost_artifacts=references
    )
    assert verdict(report, "cost_sensitivity") == CheckVerdict.FAIL


def test_missing_ledger_is_rejected_without_creating_one(tmp_path):
    base = PortfolioReplay(tmp_path / "base.db").run(**varied_inputs())
    missing = tmp_path / "missing.db"
    report = ValidationSuite().validate(base, ledger_path=missing)
    assert verdict(report, "durable_accounting") == CheckVerdict.FAIL
    assert not missing.exists()


def test_input_event_tampering_is_not_certified_by_manifest_hash_strings(tmp_path):
    import sqlite3

    inputs = varied_inputs()
    path = tmp_path / "base.db"
    base = PortfolioReplay(path).run(**inputs)
    reference = persist_replay_artifact(base, path, tmp_path / "result.json")
    assert load_replay_artifact(reference) == base
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT id,payload FROM events WHERE id LIKE 'input:signal:%' LIMIT 1"
        ).fetchone()
        payload = json.loads(row[1])
        payload["score"] += 1234
        conn.execute("UPDATE events SET payload=? WHERE id=?", (json.dumps(payload), row[0]))
    conn.close()
    reference = reference.model_copy(update={"ledger_sha256": file_hash(path)})
    with pytest.raises(BacktestError, match="accounting verification"):
        load_replay_artifact(reference)


def test_named_random_summaries_seed_and_percentile_are_all_bound(tmp_path):
    args = named_inputs()
    base = PortfolioReplay(tmp_path / "base.db").run(**args)
    comparison = BenchmarkRunner().compare_replay(
        base, inputs=args, output_dir=tmp_path / "comparison", n_simulations=5
    )
    verify_comparison(base, comparison, minimum_simulations=5)
    again = BenchmarkRunner().compare_replay(
        base, inputs=args, output_dir=tmp_path / "comparison", n_simulations=5
    )
    assert again == comparison
    changed = comparison.model_copy(deep=True)
    changed.benchmark_metrics["SPY"]["sharpe_ratio"] = 999
    with pytest.raises(BacktestError, match="named benchmark summary"):
        verify_comparison(base, changed, minimum_simulations=5)
    changed = comparison.model_copy(update={"seed": 999})
    with pytest.raises(BacktestError, match="seeded"):
        verify_comparison(base, changed, minimum_simulations=5)
    changed = comparison.model_copy(update={"monte_carlo_percentile": -1.0})
    with pytest.raises(BacktestError, match="percentile"):
        verify_comparison(base, changed, minimum_simulations=5)
    changed = comparison.model_copy(update={"simulations": 1000})
    with pytest.raises(BacktestError, match="missing durable random"):
        verify_comparison(base, changed)
    reference = comparison.random_artifacts[0]
    Path(reference.ledger_path).unlink()
    with pytest.raises(FileNotFoundError):
        verify_comparison(base, comparison, minimum_simulations=5)


def test_wf_reconstructs_exact_oos_inputs_and_cli_compare_without_refitting(tmp_path):
    x, y, training = wf_case(tmp_path)
    result = WalkForwardRunner().run(x, y, model_factory=Ridge, **training)
    keys = (
        "context",
        "account",
        "bars",
        "metadata",
        "universes",
        "fill_policy",
        "risk_policy",
        "samples_per_year",
        "valuation_calendar",
        "topk",
    )
    original = {k: training[k] for k in keys}
    inputs = reconstruct_oos_inputs(result, original)
    require_matching_inputs(result, inputs)
    assert inputs["signals"]
    assert inputs["account"].opened_at > original["account"].opened_at
    assert len(inputs["bars"]) < len(original["bars"])
    write_bundle(tmp_path / "input.json", {**original, "walk_forward": {}})
    # Compare reads the original training bundle plus its bound OOS artifact.
    # It neither retrains nor requires callers to guess the OOS cutoffs.
    output = CliRunner().invoke(
        app,
        [
            "compare",
            "--input",
            str(tmp_path / "input.json"),
            "--result",
            str(tmp_path / "models/result.json"),
            "--output",
            str(tmp_path / "comparison"),
            "--simulations",
            "2",
        ],
    )
    assert output.exit_code == 2, output.output  # named benchmark data absent
    assert (tmp_path / "comparison/costs/cost-artifacts.json").is_file()
    saved = Path(result.manifest["provenance"]["oos_signals_artifact"]["path"])
    payload = json.loads(saved.read_text())
    payload[0]["score"] += 1
    saved.write_text(json.dumps(payload))
    with pytest.raises(BacktestError, match="artifact bytes"):
        reconstruct_oos_inputs(result, original)
