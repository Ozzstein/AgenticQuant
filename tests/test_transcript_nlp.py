"""Transcript synthesis uses full-document spans and retains uncertainty."""
# ruff: noqa: F401, F811

import json

from src.agents.transcript_nlp import transcript_nlp_node
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


def test_returns_agent_report(desk_state):
    result = transcript_nlp_node(desk_state, llm=GenerativeFixture("transcript_nlp", []))
    report = result["desk_reports"]["transcript_nlp"]
    assert report.status == "ok" and report.claims and report.reasoning
    assert report.confidence.analyst_conviction == 0.6
    assert result["agent_reports"]["transcript_nlp"].confidence == 0


def test_degrades_on_llm_failure(desk_state):
    report = transcript_nlp_node(
        desk_state, llm=GenerativeFixture("transcript_nlp", [], fail_role="transcript_nlp")
    )["desk_reports"]["transcript_nlp"]
    assert report.status == "unavailable" and report.action == AgentAction.ABSTAIN
    assert "ConnectionError" in report.reasoning


def test_handles_missing_transcript(desk_state):
    calls = []
    desk_state["sources"] = tuple(s for s in desk_state["sources"] if s.kind != "transcript")
    report = transcript_nlp_node(desk_state, llm=GenerativeFixture("transcript_nlp", calls))[
        "desk_reports"
    ]["transcript_nlp"]
    assert report.status == "unavailable" and report.action == AgentAction.ABSTAIN and calls == []


def test_includes_memory_context(desk_state):
    desk_state["memory_context"] = "Prior analysis: demand weakening"
    calls = []
    transcript_nlp_node(desk_state, llm=GenerativeFixture("transcript_nlp", calls))
    assert json.loads(calls[0][1][-1].content)["memory"] == desk_state["memory_context"]


def test_no_memory_context(desk_state):
    calls = []
    transcript_nlp_node(desk_state, llm=GenerativeFixture("transcript_nlp", calls))
    assert json.loads(calls[0][1][-1].content)["memory"] == ""


def test_crypto_transcript_not_fabricated(desk_state):
    desk_state["instrument"] = desk_state["instrument"].model_copy(update={"asset_class": "crypto"})
    calls = []
    report = transcript_nlp_node(desk_state, llm=GenerativeFixture("transcript_nlp", calls))[
        "desk_reports"
    ]["transcript_nlp"]
    assert (
        report.status == "not_applicable" and report.action == AgentAction.ABSTAIN and calls == []
    )
