"""Tests for RD-Agent LLM integration components."""
from __future__ import annotations

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
    import pandas as pd

    dates = pd.date_range("2022-01-01", periods=500, freq="B")
    folds = split_folds(dates, "2022-01-01", walk_forward_months=6, embargo_days=5)
    assert len(folds) >= 1
    assert "train_start" in folds[0]
    assert "test_end" in folds[0]


def test_split_folds_returns_no_folds_for_short_data():
    from src.core.backtester import split_folds
    import pandas as pd

    # Only ~5 months of data — too short for 2 folds of 6 months each
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    folds = split_folds(dates, "2022-01-01", walk_forward_months=6, embargo_days=5)
    assert folds == []
