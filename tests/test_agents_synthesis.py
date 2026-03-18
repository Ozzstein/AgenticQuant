"""Tests for risk_manager, debate, and strategist agent nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.schemas import AgentReport, AnalysisResult, Decision

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state():
    """Return a baseline trading desk state with no veto conditions."""
    return {
        "ticker": "AAPL",
        "asset_class": "EQUITY",
        "qlib_context": "",
        "market_context": (
            "=== VIX (Volatility Index) ===\nCurrent VIX:  18.50\nRegime: Normal Volatility"
        ),
        "macro_regime": "neutral",
        "macro_confidence": 65.0,
        "agent_reports": {
            "fundamental": AgentReport(
                agent_name="fundamental",
                decision=Decision.BUY,
                confidence=75.0,
                reasoning="Strong fundamentals",
                data_points={},
            ),
            "technical": AgentReport(
                agent_name="technical",
                decision=Decision.BUY,
                confidence=70.0,
                reasoning="Uptrend intact",
                data_points={},
            ),
            "sentiment": AgentReport(
                agent_name="sentiment",
                decision=Decision.HOLD,
                confidence=55.0,
                reasoning="Mixed sentiment",
                data_points={},
            ),
        },
        "debate_round": 0,
        "debate_transcript": [],
        "risk_veto": False,
        "risk_veto_reason": "",
        "result": None,
    }


# ---------------------------------------------------------------------------
# risk_manager_node tests
# ---------------------------------------------------------------------------


def test_risk_manager_no_veto(sample_state):
    """Neutral regime, normal VIX, decent confidence — no veto."""
    from src.agents.risk_manager import risk_manager_node

    result = risk_manager_node(sample_state)

    assert result["risk_veto"] is False
    assert result["risk_veto_reason"] == ""
    assert "risk_manager" in result["agent_reports"]
    report = result["agent_reports"]["risk_manager"]
    assert report.decision == Decision.HOLD


def test_risk_manager_crisis_veto(sample_state):
    """Crisis macro regime triggers veto."""
    from src.agents.risk_manager import risk_manager_node

    sample_state["macro_regime"] = "crisis"
    result = risk_manager_node(sample_state)

    assert result["risk_veto"] is True
    assert "crisis" in result["risk_veto_reason"].lower()
    report = result["agent_reports"]["risk_manager"]
    assert report.decision == Decision.SELL


def test_risk_manager_low_confidence_veto(sample_state):
    """All analyst reports with confidence=10 triggers veto (mean < 20)."""
    from src.agents.risk_manager import risk_manager_node

    for report in sample_state["agent_reports"].values():
        report.confidence = 10.0

    result = risk_manager_node(sample_state)

    assert result["risk_veto"] is True
    assert "conviction" in result["risk_veto_reason"].lower()
    assert "10.0" in result["risk_veto_reason"]


def test_risk_manager_high_vix_veto(sample_state):
    """Market context containing 'High Volatility' triggers veto."""
    from src.agents.risk_manager import risk_manager_node

    sample_state["market_context"] = (
        "=== VIX (Volatility Index) ===\nCurrent VIX:  35.00\nRegime: High Volatility"
    )
    result = risk_manager_node(sample_state)

    assert result["risk_veto"] is True
    assert "vix" in result["risk_veto_reason"].lower() or "volatility" in result["risk_veto_reason"].lower()


# ---------------------------------------------------------------------------
# DebateManager tests
# ---------------------------------------------------------------------------


def test_debate_manager_should_debate_true(sample_state):
    """Spread=60 (conf 80 vs 20), max_rounds=2, round=0 → should_debate=True."""
    from src.agents.debate import DebateManager
    from src.utils.config import AgentConfig

    sample_state["agent_reports"]["fundamental"].confidence = 80.0
    sample_state["agent_reports"]["sentiment"].confidence = 20.0
    sample_state["debate_round"] = 0

    config = AgentConfig(max_debate_rounds=2, debate_trigger_spread=40.0)
    assert DebateManager.should_debate(sample_state, config) is True


def test_debate_manager_should_debate_false_max_rounds(sample_state):
    """debate_round == max_debate_rounds → should_debate=False."""
    from src.agents.debate import DebateManager
    from src.utils.config import AgentConfig

    sample_state["agent_reports"]["fundamental"].confidence = 80.0
    sample_state["agent_reports"]["sentiment"].confidence = 20.0
    sample_state["debate_round"] = 2

    config = AgentConfig(max_debate_rounds=2, debate_trigger_spread=40.0)
    assert DebateManager.should_debate(sample_state, config) is False


def test_debate_manager_should_debate_false_narrow_spread(sample_state):
    """All confidences within 30% → should_debate=False (spread < trigger 40)."""
    from src.agents.debate import DebateManager
    from src.utils.config import AgentConfig

    sample_state["agent_reports"]["fundamental"].confidence = 70.0
    sample_state["agent_reports"]["technical"].confidence = 65.0
    sample_state["agent_reports"]["sentiment"].confidence = 60.0
    sample_state["debate_round"] = 0

    config = AgentConfig(max_debate_rounds=2, debate_trigger_spread=40.0)
    assert DebateManager.should_debate(sample_state, config) is False


# ---------------------------------------------------------------------------
# debate_router tests
# ---------------------------------------------------------------------------


def test_debate_router_returns_debate(sample_state):
    """When should_debate is True, router returns 'debate'."""
    from src.agents.debate import debate_router

    # Set up a wide spread
    sample_state["agent_reports"]["fundamental"].confidence = 80.0
    sample_state["agent_reports"]["sentiment"].confidence = 20.0
    sample_state["debate_round"] = 0

    with patch("src.agents.debate.get_config") as mock_cfg:
        mock_cfg.return_value.agents.max_debate_rounds = 2
        mock_cfg.return_value.agents.debate_trigger_spread = 40.0
        route = debate_router(sample_state)

    assert route == "debate"


def test_debate_router_returns_risk_manager(sample_state):
    """When spread is narrow, router returns 'risk_manager'."""
    from src.agents.debate import debate_router

    # Narrow spread
    sample_state["agent_reports"]["fundamental"].confidence = 70.0
    sample_state["agent_reports"]["technical"].confidence = 65.0
    sample_state["agent_reports"]["sentiment"].confidence = 60.0
    sample_state["debate_round"] = 0

    with patch("src.agents.debate.get_config") as mock_cfg:
        mock_cfg.return_value.agents.max_debate_rounds = 2
        mock_cfg.return_value.agents.debate_trigger_spread = 40.0
        route = debate_router(sample_state)

    assert route == "risk_manager"


# ---------------------------------------------------------------------------
# debate_node tests
# ---------------------------------------------------------------------------


def test_debate_node_runs_round(sample_state):
    """Mock LLM: debate_round incremented, transcript appended."""
    from src.agents.debate import debate_node

    # Set up a state where debate would happen (wide spread)
    sample_state["agent_reports"]["fundamental"].confidence = 80.0
    sample_state["agent_reports"]["sentiment"].confidence = 20.0

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "I maintain my bearish view due to weak macro data."
    mock_llm.invoke.return_value = mock_response

    result = debate_node(sample_state, llm=mock_llm)

    assert result["debate_round"] == 1
    assert isinstance(result["debate_transcript"], list)
    assert len(result["debate_transcript"]) >= 1
    # First entry should be the round header
    assert "Debate Round 1" in result["debate_transcript"][0]


# ---------------------------------------------------------------------------
# strategist_node tests
# ---------------------------------------------------------------------------


def test_strategist_node_returns_analysis_result(sample_state):
    """Mock LLM returning valid AnalysisResult — output ticker matches."""
    mock_result = AnalysisResult(
        ticker="AAPL",
        decision=Decision.BUY,
        confidence=75.0,
        reasoning="Synthesis",
        agent_reports={},
    )
    mock_llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = structured

    with patch("src.agents.strategist.get_config") as mock_cfg:
        mock_cfg.return_value.agents.fundamental_weight = 0.35
        mock_cfg.return_value.agents.technical_weight = 0.20
        mock_cfg.return_value.agents.sentiment_weight = 0.10
        mock_cfg.return_value.agents.risk_weight = 0.10
        from src.agents.strategist import strategist_node

        result = strategist_node(sample_state, llm=mock_llm)

    assert result["result"].ticker == "AAPL"
    assert isinstance(result["result"], AnalysisResult)


def test_strategist_node_respects_risk_veto(sample_state):
    """When risk_veto=True and LLM returns BUY, decision is overridden to HOLD/SELL."""
    veto_state = {**sample_state, "risk_veto": True, "risk_veto_reason": "Crisis"}

    mock_result = AnalysisResult(
        ticker="AAPL",
        decision=Decision.BUY,
        confidence=75.0,
        reasoning="Synthesis",
        agent_reports={},
    )
    mock_llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = structured

    with patch("src.agents.strategist.get_config") as mock_cfg:
        mock_cfg.return_value.agents.fundamental_weight = 0.35
        mock_cfg.return_value.agents.technical_weight = 0.20
        mock_cfg.return_value.agents.sentiment_weight = 0.10
        mock_cfg.return_value.agents.risk_weight = 0.10
        from src.agents.strategist import strategist_node

        result = strategist_node(veto_state, llm=mock_llm)

    assert result["result"].decision in (Decision.HOLD, Decision.SELL)
    assert "veto" in result["result"].reasoning.lower() or "crisis" in result["result"].reasoning.lower()
