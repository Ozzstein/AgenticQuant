"""Canonical service/CLI composition over actual stores and synthetic input data.

Provider responses are fixtures; no historical or financial efficacy is claimed.
Fresh source-complete paper acceptance remains in test_operating_cycle and is
not replaced here by an invented successful plan.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.test_core_backtest import A, replay_inputs, write_bundle
from tests.test_research_copilot import ProviderFixture
from tests.test_research_copilot import research_env as research_env

pytestmark = pytest.mark.integration


def test_run_backtest_script_uses_actual_bundle_ledger_and_losing_nav(tmp_path):
    from scripts.run_backtest import _run_pipeline
    from src.execution.ledger import PortfolioLedger

    bundle, ledger = tmp_path / "inputs.json", tmp_path / "ledger.db"
    write_bundle(bundle, replay_inputs())
    result = _run_pipeline(bundle, ledger, tmp_path / "artifacts")
    assert result.status == "complete"
    assert result.equity_curve == [10000, 10000, 11000, 8800]
    assert result.metrics.total_return == pytest.approx(-0.12)
    book = PortfolioLedger(ledger, read_only=True)
    try:
        assert book.reconcile(result.account_id)["reconciled"]
        assert len(book.fills(result.account_id)) == 1
    finally:
        book.close()


def test_run_agents_analyze_missing_authorities_cannot_invent_direction(tmp_path):
    from scripts.run_agents import DeskAnalysisInput, analyze_input
    from src.domain.decisions import AgentAction, AgentDecision
    from src.utils.config import AppConfig

    context = replay_inputs()["context"]
    config = AppConfig(output_dir=str(tmp_path / "outputs"), log_dir=str(tmp_path / "logs"),
                       anthropic_api_key="", openai_api_key="")
    decision, coverage = analyze_input(DeskAnalysisInput(instrument=A, context=context),
                                      evidence_path=tmp_path / "evidence.db", config=config)
    assert isinstance(decision, AgentDecision)
    assert decision.context == context and decision.instrument == A
    assert decision.action in (AgentAction.ABSTAIN, AgentAction.VETO)
    assert decision.new_exposure_blocked
    assert coverage and all(row["status"] != "complete" for kinds in coverage.values()
                            for row in kinds.values())


def test_pipeline_label_cannot_select_a_separate_backtest_or_broker(monkeypatch):
    from scripts.run_pipeline import _run_pipeline_once

    monkeypatch.setattr("src.execution.broker.create_broker",
                        lambda *a, **kw: pytest.fail("parallel broker route"))
    with pytest.raises(ValueError, match="explicit PaperCyclePlan"):
        _run_pipeline_once("backtest")


def test_factor_library_roundtrip_executes_exact_formula_without_activation(tmp_path):
    from src.core.data_pipeline import DataPipeline
    from src.research.registry import ResearchRegistry
    from src.utils.config import AppConfig
    from src.utils.schemas import FactorDefinition
    from tests.test_feature_fidelity import panel

    factor = FactorDefinition(name="misleading_momentum_name", expression="Ref($close,2)/$close-1")
    (tmp_path / "factor_library.json").write_text(json.dumps([factor.model_dump(mode="json")]))
    frame = panel()
    actual = DataPipeline(AppConfig(output_dir=str(tmp_path))).load_factor_library(base_df=frame)
    expected = frame.close.groupby(level="instrument").shift(2) / frame.close - 1
    pd.testing.assert_series_equal(actual[factor.name], expected.rename(factor.name))
    assert ResearchRegistry(tmp_path / "registry.db").status()["active"] == []


def wrapper(env, *, responses=False):
    from src.core.factor_proposer import FactorProposer

    calls = []
    service = FactorProposer(
        env.config, registry=env.registry, evidence_store=env.store,
        code_hash=env.code_hash, environment_hash=env.environment_hash,
        llm_factory=(lambda purpose, model: ProviderFixture(purpose, model, calls)) if responses else None,
    )
    return service, calls


def test_research_restart_retains_actual_quarantine_without_made_up_performance(research_env):
    from src.core.rd_agent_runner import RDAgentRunner

    env = research_env
    service, calls = wrapper(env, responses=True)
    result = service.propose_factors(1, context=env.context, sources=(env.source,), description="Study persistence")
    assert result.status == "available" and len(calls) == 1
    candidate = env.registry.candidate(result.candidates[0].candidate_id)
    reopened = RDAgentRunner(env.config, registry_path=env.registry.path, evidence_path=env.store.path)
    assert reopened.registry.candidate(candidate.candidate_id) == candidate
    status = reopened.library_status()
    assert status["counts"] == {"candidates": 1, "trials": 0, "holdouts": 0, "outcomes": 0}
    assert status["active"] == []
    assert all(reopened.registry.blob(key) for key in candidate.source_evidence_hashes)


def test_copilot_missing_access_returns_typed_unavailable(research_env):
    env = research_env
    service, calls = wrapper(env)
    result = service.propose_factors(1, context=env.context, sources=(env.source,), description="Study momentum")
    assert result.status == "unavailable" and not result.candidates and not calls
    assert env.registry.status()["counts"]["candidates"] == 0


def test_copilot_model_archives_payload_and_generation_without_simulated_sharpe(research_env):
    from src.research.copilot import GenerationObservation

    env = research_env
    service, calls = wrapper(env, responses=True)
    result = service.propose_model_config(1, context=env.context, sources=(env.source,), description="Study a linear model")
    assert result.status == "available" and len(calls) == 1
    receipt = result.candidates[0]
    payload = json.loads(receipt.payload_json)
    assert set(payload) == {"features", "model_name", "model_parameters"}
    assert payload["model_name"] == "Linear"
    assert env.registry.candidate(receipt.candidate_id).kind == "model"
    observation = GenerationObservation.model_validate_json(env.registry.blob(result.observation_hashes[0]))
    assert observation.input_tokens == 100 and observation.output_tokens == 100
    assert observation.completed_at <= result.available_at
    assert env.registry.status()["counts"]["outcomes"] == 0


def test_canonical_replay_cli_restart_keeps_actual_ledger_bytes(tmp_path):
    from src.utils.cli import app

    inputs, ledger, output = tmp_path / "input.json", tmp_path / "ledger.db", tmp_path / "result.json"
    write_bundle(inputs, replay_inputs())
    args = ["backtest", "run", "--input", str(inputs), "--ledger", str(ledger), "--output", str(output)]
    runner = CliRunner()
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    original = output.read_bytes(), hashlib.sha256(ledger.read_bytes()).hexdigest()
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert (output.read_bytes(), hashlib.sha256(ledger.read_bytes()).hexdigest()) == original
    assert json.loads(output.read_text())["reconciliation"]["reconciled"]


def test_explicit_feature_specs_materialize_exact_expression_and_future_label_bounds():
    from src.data.features.dataset import feature_dataset
    from src.domain.market import FeatureSpec
    from tests.data_quant_fixtures import archived_snapshot

    snapshot = archived_snapshot()
    specs = (FeatureSpec(name="custom_factor", expression="$close/Ref($close,2)-1"),)
    data = feature_dataset(snapshot, specs, horizon=2)
    prices = pd.Series({(b.closed_at,b.instrument.instrument_id):float(b.close) for b in snapshot.bars}).sort_index()
    prices.index.names = ["datetime", "instrument"]
    expected = prices / prices.groupby(level="instrument").shift(2) - 1
    pd.testing.assert_series_equal(data.features.custom_factor, expected.rename("custom_factor"))
    cutoff = sorted(prices.index.get_level_values("datetime").unique())[50] + timedelta(seconds=1)
    eligible = data.eligible_training(cutoff)
    assert not eligible.features.empty
    assert (eligible.label_end_at <= cutoff).all()
    assert (eligible.label_available_at <= cutoff).all()
    assert data.labels.groupby(level="instrument").tail(2).isna().all()


def test_unregistered_factor_search_cannot_create_research_outcomes(tmp_path):
    from src.core.rd_agent_runner import RDAgentRunner
    from src.research.registry import RegistryError
    from src.utils.config import AppConfig

    runner = RDAgentRunner(AppConfig(output_dir=str(tmp_path)), registry_path=tmp_path / "research.db",
                           evidence_path=tmp_path / "evidence.db")
    with pytest.raises(RegistryError):
        runner.mine_factors(trial_ids=["not-admitted"], output_dir=tmp_path / "search")
    assert runner.library_status()["counts"] == {"candidates": 0, "trials": 0, "holdouts": 0, "outcomes": 0}
    assert not (tmp_path / "search").exists()
