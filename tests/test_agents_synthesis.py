"""CRO, challenge and synthesis invariants with explicit boundary semantics."""
# ruff: noqa: F401, F811

import pytest

from src.agents.debate import DebateManager, debate_node, debate_router
from src.agents.risk_manager import risk_manager_node
from src.agents.strategist import strategist_node
from src.domain.decisions import AgentAction
from tests.test_agent_desk_invariants import (
    GenerativeFixture,
    decision_context,
    decision_instrument,
    decision_source,
    desk_config,
    desk_hmm,
    desk_portfolio,
    desk_sources,
    desk_state,
)


def populated_state(state):
    from src.agents.desk import structured_report

    for role in ("fundamental", "technical", "sentiment", "earnings", "transcript_nlp"):
        result = structured_report(
            state,
            role,
            "Fixture analysis",
            kinds=("fundamental", "technical", "news", "transcript"),
            llm=GenerativeFixture(role, []),
        )
        state["desk_reports"].update(result["desk_reports"])
        state["agent_reports"].update(result["agent_reports"])
    return state


def test_risk_manager_full_review(desk_state):
    result = risk_manager_node(desk_state, llm=GenerativeFixture("risk_manager", []))
    assert not result["risk_veto"] and not result["new_exposure_blocked"]
    assert result["desk_reports"]["risk_manager"].claims


def test_risk_manager_crisis_veto_is_not_sell(desk_state):
    desk_state["macro_regime"] = "crisis"
    result = risk_manager_node(desk_state, llm=GenerativeFixture("risk_manager", []))
    assert result["risk_veto"] and result["desk_reports"]["risk_manager"].action == AgentAction.VETO
    assert result["agent_reports"]["risk_manager"].decision.value == "HOLD"


def test_missing_portfolio_cannot_approve(desk_state):
    result = risk_manager_node({**desk_state, "portfolio_context": None})
    assert (
        result["new_exposure_blocked"]
        and result["desk_reports"]["risk_manager"].action == AgentAction.ABSTAIN
    )


def test_prior_binding_veto_preserved(desk_state):
    result = risk_manager_node(
        {**desk_state, "risk_veto": True, "risk_veto_reason": "Existing hard block"},
        llm=GenerativeFixture("risk_manager", []),
    )
    assert result["risk_veto"] and "Existing hard block" in result["risk_veto_reason"]


@pytest.mark.parametrize("legacy_confidence", [0, 10, 50, 100])
def test_legacy_confidence_not_risk_arithmetic(desk_state, legacy_confidence):
    populated_state(desk_state)
    for report in desk_state["agent_reports"].values():
        report.confidence = legacy_confidence
    result = risk_manager_node(desk_state, llm=GenerativeFixture("risk_manager", []))
    assert not result["risk_veto"]


@pytest.mark.parametrize("round_number,expected", [(0, True), (1, True), (2, False), (3, False)])
def test_configured_challenges_ignore_confidence_spread(desk_state, round_number, expected):
    desk_state["debate_round"] = round_number
    assert DebateManager.should_debate(desk_state, desk_state["app_config"].agents) == expected
    assert debate_router(desk_state) == ("debate" if expected else "risk_manager")


def test_debate_revises_every_applicable_report(desk_state):
    populated_state(desk_state)
    desk_state["llm_factory"] = lambda role: GenerativeFixture(role, [])
    result = debate_node(desk_state)
    assert result["debate_round"] == 1 and len(result["debate_transcript"]) == 6
    assert set(result["desk_reports"]) == set(desk_state["desk_reports"])
    assert all(r.claims for r in result["desk_reports"].values())


def test_strategist_preserves_complete_reports(desk_state):
    populated_state(desk_state)
    desk_state["desk_reports"].update(
        risk_manager_node(desk_state, llm=GenerativeFixture("risk_manager", []))["desk_reports"]
    )
    result = strategist_node(desk_state, llm=GenerativeFixture("strategist", []))["agent_decision"]
    assert result.instrument == desk_state["instrument"] and result.action == AgentAction.BUY
    assert len(result.reports) == 7 and result.reports[-1].derived_from


@pytest.mark.parametrize("model_action", ["BUY", "EXIT"])
def test_strategist_veto_survives_contradictory_output(desk_state, model_action):
    populated_state(desk_state)
    desk_state.update(risk_veto=True, risk_veto_reason="Binding concentration veto")
    result = strategist_node(
        desk_state, llm=GenerativeFixture("strategist", [], final=model_action)
    )["agent_decision"]
    assert result.action == AgentAction.VETO and result.risk_veto and result.new_exposure_blocked
    assert "Binding concentration veto" in result.reasoning
