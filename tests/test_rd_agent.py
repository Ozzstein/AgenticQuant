"""Tests for RD-Agent LLM integration components."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.test_research_copilot import research_env as research_env

# ---------------------------------------------------------------------------
# Task 1 — Schema tests
# ---------------------------------------------------------------------------

def test_factor_proposal_schema():
    from src.utils.schemas import FactorProposal
    fp = FactorProposal(
        name="mom_20d",
        expression="close / close.shift(20) - 1",
        category="momentum",
        description="20-day price momentum",
    )
    assert fp.name == "mom_20d"
    assert fp.category == "momentum"


def test_factor_proposal_list_schema():
    from src.utils.schemas import FactorProposal, FactorProposalList
    fpl = FactorProposalList(factors=[
        FactorProposal(name="f1", expression="close", category="momentum", description="d"),
    ])
    assert len(fpl.factors) == 1


def test_model_config_proposal_schema():
    from src.utils.schemas import ModelConfigProposal
    mcp = ModelConfigProposal(
        model_type="LightGBM",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        rationale="Moderate complexity for equities",
    )
    assert mcp.model_type == "LightGBM"
    assert mcp.n_estimators == 200


def test_model_config_proposal_list_schema():
    from src.utils.schemas import ModelConfigProposal, ModelConfigProposalList
    mcpl = ModelConfigProposalList(configs=[
        ModelConfigProposal(
            model_type="CatBoost",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=31,
            rationale="Fast for categorical",
        )
    ])
    assert len(mcpl.configs) == 1


def test_eval_result_schema_passed():
    from src.utils.schemas import EvalResult
    er = EvalResult(
        factor_name="mom_20d",
        stage1_ic=0.035,
        stage1_passed=True,
        stage2_ic=0.028,
        stage2_icir=0.42,
        passed=True,
        reason="passed",
    )
    assert er.passed is True
    assert er.stage2_ic == 0.028


def test_eval_result_schema_failed_stage1():
    from src.utils.schemas import EvalResult
    er = EvalResult(
        factor_name="bad_factor",
        stage1_ic=0.005,
        stage1_passed=False,
        passed=False,
        reason="low_ic",
    )
    assert er.stage1_passed is False
    assert er.stage2_ic is None
    assert er.stage2_icir is None


# ---------------------------------------------------------------------------
# Task 2 — split_folds public function
# ---------------------------------------------------------------------------

def test_split_folds_is_public_module_function():
    """split_folds must be importable as a module-level function."""

    from src.core.backtester import split_folds

    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    folds = split_folds(dates, "2022-01-01", walk_forward_months=6, embargo_days=5)
    assert len(folds) >= 1
    assert "train_start" in folds[0]
    assert "test_end" in folds[0]


def test_split_folds_returns_no_folds_for_short_data():

    from src.core.backtester import split_folds

    # Only ~5 months of data — too short for 2 folds of 6 months each
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    folds = split_folds(dates, "2022-01-01", walk_forward_months=6, embargo_days=5)
    assert folds == []


# Proposal/memo generation now requires explicit immutable authorities.
# Provider, budget, full-document coverage and quarantine behavior are exercised
# with real stores in test_research_copilot; these are public-wrapper regressions.
def _wrapper(cls, env):
    return cls(env.config,registry=env.registry,evidence_store=env.store,
               code_hash=env.code_hash,environment_hash=env.environment_hash)


def test_legacy_constructor_cannot_enter_unbudgeted_generation():
    from src.core.factor_proposer import FactorProposer
    from src.core.research_analyst import ResearchAnalyst
    for cls in (FactorProposer,ResearchAnalyst):
        with pytest.raises(TypeError):
            cls(None)


def test_missing_credentials_returns_typed_unavailable_without_templates(research_env):
    from src.core.factor_proposer import FactorProposer
    env=research_env
    service=_wrapper(FactorProposer,env)
    outcomes=[service.propose_factors(3,context=env.context,sources=(env.source,),description='Study momentum'),
              service.propose_model_config(2,context=env.context,sources=(env.source,),description='Study model'),
              service.propose_strategy('Study strategy',context=env.context,sources=(env.source,)),
              service.propose_regime_change('Study regimes',context=env.context,sources=(env.source,))]
    assert all(o.status=='unavailable' and not o.candidates for o in outcomes)
    assert env.registry.status()['counts']['candidates']==0


def test_arbitrary_memo_text_and_caller_metrics_are_not_research_evidence(research_env):
    from src.core.research_analyst import ResearchAnalyst
    env=research_env
    service=_wrapper(ResearchAnalyst,env)
    outcome=service.write_memo(context=env.context,sources=(env.source,))
    assert outcome.status=='unavailable' and outcome.memo is None
    with pytest.raises(TypeError):
        service.write_memo(context=env.context,development_results={'sharpe':3})
    with pytest.raises(ValueError):
        service.load_memo(env.registry.archive(b'{"hypotheses":["invented"]}'),context=env.context)


def test_runner_requires_registered_experiments(tmp_path, mock_config):
    import pytest

    from src.core.rd_agent_runner import RDAgentRunner
    runner = RDAgentRunner(mock_config, registry_path=tmp_path/"registry.db", evidence_path=tmp_path/"evidence.db")
    with pytest.raises(TypeError):
        runner.run_factor_search()
    with pytest.raises(TypeError):
        runner.run_model_search()
    assert runner.library_status()["counts"]["trials"] == 0
    assert runner.library_status()["active"] == []


def test_evaluator_requires_archived_context():
    import pytest

    from src.core.factor_evaluator import FactorEvaluator
    from src.utils.schemas import FactorDefinition
    with pytest.raises(TypeError):
        FactorEvaluator().evaluate_factor(FactorDefinition(name="old",expression="$close"))
