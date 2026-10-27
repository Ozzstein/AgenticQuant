"""Tests for FactorBacktester and related schemas/config."""
from __future__ import annotations

from tests.test_research_copilot import research_env  # noqa: F401

# ---------------------------------------------------------------------------
# Task 1: Schema and config tests
# ---------------------------------------------------------------------------

def test_backtest_validation_result_schema():
    """BacktestValidationResult has correct fields; checks default is {} not shared."""
    from src.utils.schemas import BacktestValidationResult

    r1 = BacktestValidationResult(factor_name="f1", passed=True, reason="passed")
    r2 = BacktestValidationResult(factor_name="f2", passed=False, reason="low_sharpe")

    assert r1.factor_name == "f1"
    assert r1.passed is True
    assert r1.sharpe is None
    assert r1.max_drawdown is None
    assert r1.checks == {}
    assert r1.reason == "passed"
    # Verify checks defaults are NOT shared between instances
    r1.checks["x"] = True
    assert "x" not in r2.checks, "checks dict should not be shared between instances"


def test_factor_definition_backtest_fields():
    """FactorDefinition gains 3 optional backtest fields defaulting to None."""
    from src.utils.schemas import FactorDefinition

    fd = FactorDefinition(name="test", expression="close.pct_change()")
    assert hasattr(fd, "backtest_sharpe")
    assert hasattr(fd, "backtest_max_drawdown")
    assert hasattr(fd, "validation_checks")
    assert fd.backtest_sharpe is None
    assert fd.backtest_max_drawdown is None
    assert fd.validation_checks is None
    # Existing fields unchanged
    assert fd.ic_mean == 0.0
    assert fd.icir == 0.0
    assert fd.source == "builtin"


def test_config_min_backtest_sharpe_default():
    """RDAgentConfig has min_backtest_sharpe defaulting to 0.5."""
    from src.utils.config_loader import RDAgentConfig

    cfg = RDAgentConfig()
    assert hasattr(cfg, "min_backtest_sharpe")
    assert cfg.min_backtest_sharpe == 0.5


# The public memo wrapper now reads only immutable recorded development results.


def analyst(env, calls):
    from src.core.research_analyst import ResearchAnalyst
    from tests.test_research_copilot import ProviderFixture
    return ResearchAnalyst(env.config, registry=env.registry, evidence_store=env.store,
        code_hash=env.code_hash, environment_hash=env.environment_hash,
        llm_factory=lambda purpose, model: ProviderFixture(purpose, model, calls))


def test_research_analyst_memo_uses_recorded_development_result(research_env):  # noqa: F811
    import json

    from tests.test_research_copilot import development_fixture
    env = research_env
    trial, outcome = development_fixture(env)
    calls = []
    result = analyst(env, calls).write_memo(context=env.context, development_trial_ids=(trial.trial_id,))
    assert result.status == 'available'
    assert result.memo.development_trial_ids == (trial.trial_id,)
    # Preserve failure and undefined performance instead of a fabricated table cell.
    source = json.loads(calls[0]['sources'][0]['text'])
    assert source == outcome.model_dump(mode='json')
    assert source['status'] == 'failed' and source['sharpe'] is None


def test_research_analyst_without_backtest_is_explicit_source_hypothesis(research_env):  # noqa: F811
    env = research_env
    calls = []
    result = analyst(env, calls).write_memo(context=env.context, sources=(env.source,))
    assert result.status == 'available'
    assert not result.memo.development_trial_ids
    assert 'Hypothesis (unvalidated)' in result.memo.text
    assert env.source.record.evidence_id in result.memo.text
    assert calls[0]['sources'][0]['text'] == env.source.text

# Actual performance/reliability cases are covered by test_research_evaluation;
# the legacy mocked APPROVED/CAUTION shortcut no longer has approval authority.
def test_unsafe_expression_is_rejected_before_model_training(tmp_path):
    import pytest

    from src.core.factor_backtester import FactorBacktester
    from src.data.features.expressions import ExpressionError
    from src.utils.schemas import FactorDefinition
    from tests.test_research_inputs import research_inputs
    with pytest.raises(ExpressionError):
        FactorBacktester().run(FactorDefinition(name="future",expression="Ref($close,-1)"),
                               inputs=research_inputs(),ledger_path=tmp_path/"ledger.db",artifact_dir=tmp_path/"models")
    assert not (tmp_path/"ledger.db").exists()
