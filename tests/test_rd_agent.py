"""Tests for RD-Agent LLM integration components."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    import pandas as pd

    from src.core.backtester import split_folds

    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    folds = split_folds(dates, "2022-01-01", walk_forward_months=6, embargo_days=5)
    assert len(folds) >= 1
    assert "train_start" in folds[0]
    assert "test_end" in folds[0]


def test_split_folds_returns_no_folds_for_short_data():
    import pandas as pd

    from src.core.backtester import split_folds

    # Only ~5 months of data — too short for 2 folds of 6 months each
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    folds = split_folds(dates, "2022-01-01", walk_forward_months=6, embargo_days=5)
    assert folds == []


# ---------------------------------------------------------------------------
# Task 3 — FactorProposer tests (mocked LLM)
# ---------------------------------------------------------------------------


def _mock_full_config(with_api_key: bool = True):
    """Return a minimal FullAppConfig mock."""
    cfg = MagicMock()
    cfg.anthropic_api_key = "sk-test" if with_api_key else ""
    cfg.agents = MagicMock()
    cfg.agents.quick_think_model = "claude-haiku-4-5-20251001"
    cfg.rd_agent = MagicMock()
    cfg.rd_agent.min_ic = 0.02
    cfg.rd_agent.min_icir = 0.3
    return cfg


def test_factor_proposer_fallback_no_api_key():
    """Without an API key, propose_factors should return FactorDefinition objects."""
    from src.core.factor_proposer import FactorProposer
    from src.utils.schemas import FactorDefinition

    cfg = _mock_full_config(with_api_key=False)
    proposer = FactorProposer(cfg)
    results = proposer.propose_factors(n=3, memo="", tested=[], description="")
    assert len(results) == 3
    assert all(isinstance(r, FactorDefinition) for r in results)


def test_factor_proposer_deduplication():
    """propose_factors must exclude names in the tested list."""
    from src.core.factor_proposer import FactorProposer

    cfg = _mock_full_config(with_api_key=False)
    proposer = FactorProposer(cfg)
    results = proposer.propose_factors(n=2, memo="", tested=[], description="")
    tested_names = [r.name for r in results]
    results2 = proposer.propose_factors(n=2, memo="", tested=tested_names, description="")
    for r in results2:
        assert r.name not in tested_names


def test_factor_proposer_llm_path():
    """With an API key, propose_factors should call the LLM and parse structured output."""
    from src.core.factor_proposer import FactorProposer
    from src.utils.schemas import FactorDefinition, FactorProposal, FactorProposalList

    cfg = _mock_full_config(with_api_key=True)

    mock_output = FactorProposalList(factors=[
        FactorProposal(
            name="test_factor_1",
            expression="close / close.shift(10) - 1",
            category="momentum",
            description="10-day price momentum",
        ),
        FactorProposal(
            name="test_factor_2",
            expression="volume / volume.rolling(20).mean()",
            category="volume",
            description="Volume ratio to 20-day average",
        ),
    ])

    with patch("src.core.factor_proposer.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_llm_instance.with_structured_output.return_value = mock_chain
        MockLLM.return_value = mock_llm_instance

        proposer = FactorProposer(cfg)
        results = proposer.propose_factors(n=2, memo="test memo", tested=[], description="")

    assert len(results) == 2
    assert all(isinstance(r, FactorDefinition) for r in results)
    assert results[0].name == "test_factor_1"


def test_factor_proposer_llm_failure_falls_back():
    """On LLM failure after retries, propose_factors should fall back to keyword logic."""
    from tenacity import wait_none

    from src.core.factor_proposer import FactorProposer
    from src.utils.schemas import FactorDefinition

    cfg = _mock_full_config(with_api_key=True)

    with patch("src.core.factor_proposer.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")
        mock_llm_instance.with_structured_output.return_value = mock_chain
        MockLLM.return_value = mock_llm_instance

        proposer = FactorProposer(cfg)
        # Remove retry wait to keep the test fast
        proposer._llm_propose_factors.retry.wait = wait_none()
        results = proposer.propose_factors(n=2, memo="", tested=[], description="")

    assert len(results) >= 1
    assert all(isinstance(r, FactorDefinition) for r in results)


def test_propose_model_config_fallback():
    """propose_model_config without API key returns ModelConfigProposal list."""
    from src.core.factor_proposer import FactorProposer
    from src.utils.schemas import ModelConfigProposal

    cfg = _mock_full_config(with_api_key=False)
    proposer = FactorProposer(cfg)
    results = proposer.propose_model_config(n=2, memo="", tested=[])
    assert len(results) == 2
    assert all(isinstance(r, ModelConfigProposal) for r in results)


def test_propose_strategy_fallback():
    """propose_strategy without API key returns a Strategy object."""
    from src.core.factor_proposer import FactorProposer
    from src.utils.schemas import Strategy

    cfg = _mock_full_config(with_api_key=False)
    proposer = FactorProposer(cfg)
    result = proposer.propose_strategy(description="momentum strategy", memo="")
    assert isinstance(result, Strategy)
    assert result.name


def test_propose_regime_change_fallback():
    """propose_regime_change without API key returns a dict."""
    from src.core.factor_proposer import FactorProposer

    cfg = _mock_full_config(with_api_key=False)
    proposer = FactorProposer(cfg)
    result = proposer.propose_regime_change(description="increase equity in risk_on", memo="")
    assert isinstance(result, dict)
