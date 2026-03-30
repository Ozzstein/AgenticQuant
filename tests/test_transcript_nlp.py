"""Unit tests for the transcript_nlp analyst node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.schemas import AgentReport, Decision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(report: AgentReport) -> MagicMock:
    """Build a mock LLM whose with_structured_output().invoke() returns *report*."""
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = report
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock
    return mock_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state() -> dict:
    """Return a minimal TradingDeskState-compatible dict for AAPL."""
    return {
        "ticker": "AAPL",
        "asset_class": "EQUITY",
        "qlib_context": "",
        "market_context": "AAPL: $175.00",
        "macro_regime": "neutral",
        "macro_confidence": 60.0,
        "memory_context": "",
        "agent_reports": {},
        "debate_round": 0,
        "debate_transcript": [],
        "risk_veto": False,
        "risk_veto_reason": "",
        "result": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_agent_report(sample_state):
    """transcript_nlp_node returns a dict with agent_reports['transcript_nlp'] as AgentReport."""
    mock_report = AgentReport(
        agent_name="transcript_nlp",
        decision=Decision.BUY,
        confidence=72.0,
        reasoning="Management tone bullish with raised guidance",
        data_points={
            "management_sentiment": "bullish",
            "guidance_direction": "raised",
            "tone_vs_prior": "improved",
            "hedging_level": "low",
            "key_quote": "We are very confident in our full-year outlook.",
            "transcript_quarter": "Q4 2025",
        },
    )
    mock_llm = _make_mock_llm(mock_report)

    with patch("src.data.transcript_data.get_earnings_transcript") as mock_tool:
        mock_tool.invoke.return_value = "Q4 2025 earnings call transcript: ..."

        from src.agents.transcript_nlp import transcript_nlp_node

        result = transcript_nlp_node(sample_state, llm=mock_llm)

    assert "agent_reports" in result
    assert "transcript_nlp" in result["agent_reports"]
    report = result["agent_reports"]["transcript_nlp"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.BUY
    assert report.confidence == 72.0


def test_degrades_on_llm_failure(sample_state):
    """transcript_nlp_node returns HOLD with 0.0 confidence when the LLM call raises."""
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("LLM timeout")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock

    with patch("src.data.transcript_data.get_earnings_transcript") as mock_tool:
        mock_tool.invoke.return_value = "Transcript data"

        from src.agents.transcript_nlp import transcript_nlp_node

        result = transcript_nlp_node(sample_state, llm=mock_llm)

    assert "transcript_nlp" in result["agent_reports"]
    report = result["agent_reports"]["transcript_nlp"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.HOLD
    assert report.confidence == 0.0
    assert "Analysis failed" in report.reasoning


def test_handles_tool_error(sample_state):
    """transcript_nlp_node returns a valid report even when the transcript tool raises."""
    mock_report = AgentReport(
        agent_name="transcript_nlp",
        decision=Decision.HOLD,
        confidence=20.0,
        reasoning="Limited data — transcript unavailable",
        data_points={"guidance_direction": "not_available"},
    )
    mock_llm = _make_mock_llm(mock_report)

    with patch("src.data.transcript_data.get_earnings_transcript") as mock_tool:
        mock_tool.invoke.side_effect = Exception("Service unavailable")

        from src.agents.transcript_nlp import transcript_nlp_node

        result = transcript_nlp_node(sample_state, llm=mock_llm)

    assert "transcript_nlp" in result["agent_reports"]
    assert isinstance(result["agent_reports"]["transcript_nlp"], AgentReport)


def test_includes_memory_context(sample_state):
    """When memory_context is non-empty it appears in the LLM human message."""
    sample_state["memory_context"] = "Prior analysis: AAPL was a BUY"

    mock_report = AgentReport(
        agent_name="transcript_nlp",
        decision=Decision.BUY,
        confidence=65.0,
        reasoning="Consistent with prior analysis",
        data_points={},
    )
    mock_llm = _make_mock_llm(mock_report)
    structured_mock = mock_llm.with_structured_output.return_value

    with patch("src.data.transcript_data.get_earnings_transcript") as mock_tool:
        mock_tool.invoke.return_value = "Transcript text"

        from src.agents.transcript_nlp import transcript_nlp_node

        transcript_nlp_node(sample_state, llm=mock_llm)

    call_args = structured_mock.invoke.call_args
    messages = call_args[0][0]
    human_content = messages[1].content
    assert "Prior analysis: AAPL was a BUY" in human_content


def test_no_memory_context(sample_state):
    """transcript_nlp_node runs without error when memory_context is empty."""
    sample_state["memory_context"] = ""

    mock_report = AgentReport(
        agent_name="transcript_nlp",
        decision=Decision.HOLD,
        confidence=50.0,
        reasoning="No prior context",
        data_points={},
    )
    mock_llm = _make_mock_llm(mock_report)

    with patch("src.data.transcript_data.get_earnings_transcript") as mock_tool:
        mock_tool.invoke.return_value = "Transcript"

        from src.agents.transcript_nlp import transcript_nlp_node

        result = transcript_nlp_node(sample_state, llm=mock_llm)

    assert "transcript_nlp" in result["agent_reports"]


def test_agent_name_is_transcript_nlp(sample_state):
    """The returned AgentReport must have agent_name == 'transcript_nlp'."""
    mock_report = AgentReport(
        agent_name="transcript_nlp",
        decision=Decision.STRONG_BUY,
        confidence=90.0,
        reasoning="Exceptional management commentary",
        data_points={"management_sentiment": "bullish"},
    )
    mock_llm = _make_mock_llm(mock_report)

    with patch("src.data.transcript_data.get_earnings_transcript") as mock_tool:
        mock_tool.invoke.return_value = "Transcript"

        from src.agents.transcript_nlp import transcript_nlp_node

        result = transcript_nlp_node(sample_state, llm=mock_llm)

    assert result["agent_reports"]["transcript_nlp"].agent_name == "transcript_nlp"
