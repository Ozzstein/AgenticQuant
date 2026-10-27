"""Specialists require immutable sources and retain full provider-generated reports."""
# ruff: noqa: F401, F811

import json

import pytest

from src.agents.earnings_handler import earnings_node
from src.agents.fundamental import fundamental_node
from src.agents.sentiment import sentiment_node
from src.agents.technical import technical_node
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

NODES = [
    ("fundamental", fundamental_node),
    ("sentiment", sentiment_node),
    ("technical", technical_node),
    ("earnings", earnings_node),
]


@pytest.mark.parametrize("role,node", NODES)
def test_specialist_returns_typed_report(desk_state, role, node):
    calls = []
    output = node(desk_state, llm=GenerativeFixture(role, calls))
    report = output["desk_reports"][role]
    assert report.role == role and report.status == "ok" and report.action == AgentAction.BUY
    assert report.claims and "Fixture reasoning" in report.reasoning
    assert report.confidence.analyst_conviction == 0.6
    assert output["agent_reports"][role].confidence == 0  # no conversion to legacy percentage
    assert json.loads(calls[0][1][-1].content)["sources"]


@pytest.mark.parametrize("role,node", NODES)
def test_provider_failure_is_explicit_abstention(desk_state, role, node):
    output = node(desk_state, llm=GenerativeFixture(role, [], fail_role=role))
    report = output["desk_reports"][role]
    assert report.action == AgentAction.ABSTAIN and report.status == "unavailable"
    assert "ConnectionError" in report.reasoning
    assert report.confidence.analyst_conviction is None


@pytest.mark.parametrize("role,node", NODES)
def test_missing_sources_do_not_make_neutral_report(desk_state, role, node):
    calls = []
    report = node({**desk_state, "sources": ()}, llm=GenerativeFixture(role, calls))[
        "desk_reports"
    ][role]
    assert report.status == "unavailable" and report.action == AgentAction.ABSTAIN
    assert calls == []


@pytest.mark.parametrize("role,node", NODES)
def test_memory_and_quant_context_survive_prompt(desk_state, role, node):
    desk_state.update(
        memory_context="Prior source-grounded analysis", qlib_context="Quant score: 0.4"
    )
    calls = []
    node(desk_state, llm=GenerativeFixture(role, calls))
    payload = json.loads(calls[0][1][-1].content)
    assert payload["memory"] == "Prior source-grounded analysis"
    assert payload["quant_context"] == "Quant score: 0.4"
