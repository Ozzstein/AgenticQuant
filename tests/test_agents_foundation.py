"""Unit tests for WU-1: TradingDeskState and get_llm() factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.state import TradingDeskState, _merge_reports
from src.utils.config import AgentConfig
from src.utils.exceptions import LLMError
from src.utils.schemas import AgentReport, Decision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(name: str, decision: Decision = Decision.BUY) -> AgentReport:
    """Create a minimal AgentReport for testing."""
    return AgentReport(
        agent_name=name,
        decision=decision,
        confidence=75.0,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# TradingDeskState tests
# ---------------------------------------------------------------------------


def test_state_has_required_keys():
    """TradingDeskState.__annotations__ must contain all 11 required keys."""
    required = {
        "ticker",
        "asset_class",
        "qlib_context",
        "market_context",
        "macro_regime",
        "macro_confidence",
        "agent_reports",
        "debate_round",
        "debate_transcript",
        "risk_veto",
        "risk_veto_reason",
        "result",
    }
    annotations = TradingDeskState.__annotations__
    for key in required:
        assert key in annotations, f"Missing key in TradingDeskState: {key}"


def test_merge_reports_basic():
    """Merging two non-overlapping dicts produces a combined dict."""
    a = {"fundamental": _make_report("fundamental")}
    b = {"technical": _make_report("technical")}
    merged = _merge_reports(a, b)
    assert set(merged.keys()) == {"fundamental", "technical"}
    assert merged["fundamental"].agent_name == "fundamental"
    assert merged["technical"].agent_name == "technical"


def test_merge_reports_b_overwrites_a():
    """Keys present in both dicts use b's value."""
    a = {"risk": _make_report("risk", Decision.SELL)}
    b = {"risk": _make_report("risk", Decision.BUY)}
    merged = _merge_reports(a, b)
    assert len(merged) == 1
    assert merged["risk"].decision == Decision.BUY


def test_merge_reports_empty():
    """Merging two empty dicts returns an empty dict."""
    result = _merge_reports({}, {})
    assert result == {}


# ---------------------------------------------------------------------------
# get_llm() tests
# ---------------------------------------------------------------------------


def test_get_llm_standard_mode(mock_config):
    """Standard mode returns a ChatAnthropic configured with llm_model."""
    agent_cfg: AgentConfig = mock_config.agents

    with patch("src.agents.llm.ChatAnthropic") as mock_cls, patch(
        "src.agents.llm.get_config", return_value=mock_config
    ):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        from src.agents.llm import get_llm

        result = get_llm(agent_cfg, mode="standard")

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model"] == agent_cfg.llm_model
    assert call_kwargs["temperature"] == agent_cfg.temperature
    assert call_kwargs["max_tokens_to_sample"] == agent_cfg.max_tokens
    assert result is mock_instance


def test_get_llm_deep_mode(mock_config):
    """Deep mode returns a ChatAnthropic configured with deep_think_model."""
    agent_cfg: AgentConfig = mock_config.agents

    with patch("src.agents.llm.ChatAnthropic") as mock_cls, patch(
        "src.agents.llm.get_config", return_value=mock_config
    ):
        mock_cls.return_value = MagicMock()

        from src.agents.llm import get_llm

        get_llm(agent_cfg, mode="deep")

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model"] == agent_cfg.deep_think_model


def test_get_llm_quick_mode(mock_config):
    """Quick mode returns a ChatAnthropic configured with quick_think_model."""
    agent_cfg: AgentConfig = mock_config.agents

    with patch("src.agents.llm.ChatAnthropic") as mock_cls, patch(
        "src.agents.llm.get_config", return_value=mock_config
    ):
        mock_cls.return_value = MagicMock()

        from src.agents.llm import get_llm

        get_llm(agent_cfg, mode="quick")

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model"] == agent_cfg.quick_think_model


def test_get_llm_unknown_provider_raises(mock_config):
    """Passing an unknown provider raises LLMError."""
    agent_cfg = AgentConfig(llm_provider="unknown")

    from src.agents.llm import get_llm

    with pytest.raises(LLMError, match="Unknown LLM provider: unknown"):
        get_llm(agent_cfg)
