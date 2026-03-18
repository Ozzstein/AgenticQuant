"""Unit tests for WU-4 analyst nodes: fundamental, sentiment, technical, earnings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.schemas import AgentReport, Decision

# ---------------------------------------------------------------------------
# Shared fixture
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
        "agent_reports": {},
        "debate_round": 0,
        "debate_transcript": [],
        "risk_veto": False,
        "risk_veto_reason": "",
        "result": None,
    }


def _make_mock_llm(report: AgentReport) -> MagicMock:
    """Build a mock LLM whose with_structured_output().invoke() returns *report*."""
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = report
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock
    return mock_llm


# ---------------------------------------------------------------------------
# Fundamental node tests
# ---------------------------------------------------------------------------


def test_fundamental_node_returns_agent_report(sample_state):
    """fundamental_node returns a dict with agent_reports['fundamental'] as AgentReport."""
    mock_report = AgentReport(
        agent_name="fundamental",
        decision=Decision.BUY,
        confidence=75.0,
        reasoning="Strong fundamentals",
        data_points={"pe_ratio": 25.0},
    )
    mock_llm = _make_mock_llm(mock_report)

    with (
        patch("src.data.fundamental_data.get_financials") as mock_fin,
        patch("src.data.fundamental_data.get_insider_trades") as mock_insider,
        patch("src.data.fundamental_data.get_sec_filings") as mock_sec,
        patch("src.data.fundamental_data.get_analyst_ratings") as mock_ratings,
    ):
        mock_fin.invoke.return_value = "Revenue: $100B"
        mock_insider.invoke.return_value = "No recent trades"
        mock_sec.invoke.return_value = "10-K filed"
        mock_ratings.invoke.return_value = "12 BUY, 3 HOLD, 1 SELL"

        from src.agents.fundamental import fundamental_node

        result = fundamental_node(sample_state, llm=mock_llm)

    assert "agent_reports" in result
    assert "fundamental" in result["agent_reports"]
    report = result["agent_reports"]["fundamental"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.BUY
    assert report.confidence == 75.0


def test_fundamental_node_degrades_on_llm_failure(sample_state):
    """fundamental_node returns HOLD with 0 confidence when the LLM call raises."""
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("LLM timeout")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock

    with (
        patch("src.data.fundamental_data.get_financials") as mock_fin,
        patch("src.data.fundamental_data.get_insider_trades") as mock_insider,
        patch("src.data.fundamental_data.get_sec_filings") as mock_sec,
        patch("src.data.fundamental_data.get_analyst_ratings") as mock_ratings,
    ):
        mock_fin.invoke.return_value = "Revenue: $100B"
        mock_insider.invoke.return_value = "No trades"
        mock_sec.invoke.return_value = "10-K"
        mock_ratings.invoke.return_value = "BUY"

        from src.agents.fundamental import fundamental_node

        result = fundamental_node(sample_state, llm=mock_llm)

    assert "fundamental" in result["agent_reports"]
    report = result["agent_reports"]["fundamental"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.HOLD
    assert report.confidence == 0.0
    assert "Analysis failed" in report.reasoning


def test_fundamental_node_handles_tool_errors(sample_state):
    """fundamental_node returns a valid report even when all tools raise exceptions."""
    mock_report = AgentReport(
        agent_name="fundamental",
        decision=Decision.HOLD,
        confidence=30.0,
        reasoning="Limited data",
        data_points={},
    )
    mock_llm = _make_mock_llm(mock_report)

    with (
        patch("src.data.fundamental_data.get_financials") as mock_fin,
        patch("src.data.fundamental_data.get_insider_trades") as mock_insider,
        patch("src.data.fundamental_data.get_sec_filings") as mock_sec,
        patch("src.data.fundamental_data.get_analyst_ratings") as mock_ratings,
    ):
        mock_fin.invoke.side_effect = Exception("Network error")
        mock_insider.invoke.side_effect = Exception("Network error")
        mock_sec.invoke.side_effect = Exception("Network error")
        mock_ratings.invoke.side_effect = Exception("Network error")

        from src.agents.fundamental import fundamental_node

        result = fundamental_node(sample_state, llm=mock_llm)

    assert "fundamental" in result["agent_reports"]
    assert isinstance(result["agent_reports"]["fundamental"], AgentReport)


# ---------------------------------------------------------------------------
# Sentiment node tests
# ---------------------------------------------------------------------------


def test_sentiment_node_returns_agent_report(sample_state):
    """sentiment_node returns a dict with agent_reports['sentiment'] as AgentReport."""
    mock_report = AgentReport(
        agent_name="sentiment",
        decision=Decision.BUY,
        confidence=65.0,
        reasoning="Positive news flow",
        data_points={"sentiment_score": 0.7},
    )
    mock_llm = _make_mock_llm(mock_report)

    with (
        patch("src.data.news_data.get_company_news") as mock_cnews,
        patch("src.data.news_data.get_market_news") as mock_mnews,
        patch("src.data.alternative_data.get_fear_greed_index") as mock_fg,
        patch("src.data.fundamental_data.get_analyst_ratings") as mock_ratings,
        patch("src.data.earnings_calendar.get_earnings_calendar") as mock_cal,
    ):
        mock_cnews.invoke.return_value = "AAPL announces new product"
        mock_mnews.invoke.return_value = "Markets steady"
        mock_fg.invoke.return_value = "Fear/Greed: 55 (Neutral)"
        mock_ratings.invoke.return_value = "10 BUY"
        mock_cal.invoke.return_value = "Next earnings: 2026-04-30"

        from src.agents.sentiment import sentiment_node

        result = sentiment_node(sample_state, llm=mock_llm)

    assert "agent_reports" in result
    assert "sentiment" in result["agent_reports"]
    report = result["agent_reports"]["sentiment"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.BUY
    assert report.confidence == 65.0


def test_sentiment_node_degrades_on_llm_failure(sample_state):
    """sentiment_node returns HOLD with 0 confidence when the LLM call raises."""
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("LLM error")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock

    with (
        patch("src.data.news_data.get_company_news") as mock_cnews,
        patch("src.data.news_data.get_market_news") as mock_mnews,
        patch("src.data.alternative_data.get_fear_greed_index") as mock_fg,
        patch("src.data.fundamental_data.get_analyst_ratings") as mock_ratings,
        patch("src.data.earnings_calendar.get_earnings_calendar") as mock_cal,
    ):
        mock_cnews.invoke.return_value = "News"
        mock_mnews.invoke.return_value = "Markets"
        mock_fg.invoke.return_value = "50"
        mock_ratings.invoke.return_value = "BUY"
        mock_cal.invoke.return_value = "Q1 2026"

        from src.agents.sentiment import sentiment_node

        result = sentiment_node(sample_state, llm=mock_llm)

    report = result["agent_reports"]["sentiment"]
    assert report.decision == Decision.HOLD
    assert report.confidence == 0.0
    assert "Analysis failed" in report.reasoning


# ---------------------------------------------------------------------------
# Technical node tests
# ---------------------------------------------------------------------------


def test_technical_node_returns_agent_report(sample_state):
    """technical_node returns a dict with agent_reports['technical'] as AgentReport."""
    mock_report = AgentReport(
        agent_name="technical",
        decision=Decision.BUY,
        confidence=70.0,
        reasoning="Bullish trend confirmed above SMA200",
        data_points={"rsi": 58.0, "above_sma200": True},
    )
    mock_llm = _make_mock_llm(mock_report)

    with (
        patch("src.data.market_data.get_stock_price") as mock_price,
        patch("src.data.market_data.get_stock_history") as mock_hist,
        patch("src.data.market_data.get_options_data") as mock_opts,
    ):
        mock_price.invoke.return_value = "AAPL: $175.00"
        mock_hist.invoke.return_value = "252 days of OHLCV"
        mock_opts.invoke.return_value = "Put/Call: 0.85"

        from src.agents.technical import technical_node

        result = technical_node(sample_state, llm=mock_llm)

    assert "agent_reports" in result
    assert "technical" in result["agent_reports"]
    report = result["agent_reports"]["technical"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.BUY
    assert report.confidence == 70.0


def test_technical_node_degrades_on_llm_failure(sample_state):
    """technical_node returns HOLD with 0 confidence when the LLM call raises."""
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("LLM error")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock

    with (
        patch("src.data.market_data.get_stock_price") as mock_price,
        patch("src.data.market_data.get_stock_history") as mock_hist,
        patch("src.data.market_data.get_options_data") as mock_opts,
    ):
        mock_price.invoke.return_value = "AAPL: $175.00"
        mock_hist.invoke.return_value = "History"
        mock_opts.invoke.return_value = "Options"

        from src.agents.technical import technical_node

        result = technical_node(sample_state, llm=mock_llm)

    report = result["agent_reports"]["technical"]
    assert report.decision == Decision.HOLD
    assert report.confidence == 0.0
    assert "Analysis failed" in report.reasoning


# ---------------------------------------------------------------------------
# Earnings node tests
# ---------------------------------------------------------------------------


def test_earnings_node_returns_agent_report(sample_state):
    """earnings_node returns a dict with agent_reports['earnings'] as AgentReport."""
    mock_report = AgentReport(
        agent_name="earnings",
        decision=Decision.BUY,
        confidence=60.0,
        reasoning="Beat probability high based on estimate revisions",
        data_points={"next_earnings_date": "2026-04-30", "eps_estimate": 1.85},
    )
    mock_llm = _make_mock_llm(mock_report)

    with (
        patch("src.data.earnings_calendar.get_earnings_calendar") as mock_cal,
        patch("src.data.earnings_calendar.get_earnings_estimate") as mock_est,
    ):
        mock_cal.invoke.return_value = "Next earnings: 2026-04-30 AMC"
        mock_est.invoke.return_value = "EPS estimate: $1.85 (consensus)"

        from src.agents.earnings_handler import earnings_node

        result = earnings_node(sample_state, llm=mock_llm)

    assert "agent_reports" in result
    assert "earnings" in result["agent_reports"]
    report = result["agent_reports"]["earnings"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.BUY
    assert report.confidence == 60.0


def test_earnings_node_no_llm(sample_state):
    """earnings_node falls back to neutral HOLD report when LLM setup fails."""
    with (
        patch("src.data.earnings_calendar.get_earnings_calendar") as mock_cal,
        patch("src.data.earnings_calendar.get_earnings_estimate") as mock_est,
        patch("src.utils.config.get_config", side_effect=RuntimeError("Config error")),
    ):
        mock_cal.invoke.return_value = "No upcoming earnings"
        mock_est.invoke.return_value = "No estimate data"

        from src.agents.earnings_handler import earnings_node

        result = earnings_node(sample_state, llm=None)

    assert "agent_reports" in result
    assert "earnings" in result["agent_reports"]
    report = result["agent_reports"]["earnings"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.HOLD
    assert report.confidence == 40.0


def test_earnings_node_degrades_on_llm_failure(sample_state):
    """earnings_node falls back to neutral HOLD when the LLM invoke call raises."""
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("LLM error")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_mock

    with (
        patch("src.data.earnings_calendar.get_earnings_calendar") as mock_cal,
        patch("src.data.earnings_calendar.get_earnings_estimate") as mock_est,
    ):
        mock_cal.invoke.return_value = "Next: Q2 2026"
        mock_est.invoke.return_value = "EPS: $2.00"

        from src.agents.earnings_handler import earnings_node

        result = earnings_node(sample_state, llm=mock_llm)

    report = result["agent_reports"]["earnings"]
    assert isinstance(report, AgentReport)
    assert report.decision == Decision.HOLD
    assert report.confidence == 40.0


# ---------------------------------------------------------------------------
# Return type invariant tests
# ---------------------------------------------------------------------------


def test_all_nodes_return_agent_reports_key(sample_state):
    """Each analyst node must return a dict containing 'agent_reports'."""
    nodes_and_keys = [
        ("src.agents.fundamental", "fundamental_node", "fundamental"),
        ("src.agents.sentiment", "sentiment_node", "sentiment"),
        ("src.agents.technical", "technical_node", "technical"),
        ("src.agents.earnings_handler", "earnings_node", "earnings"),
    ]

    for module_path, func_name, agent_key in nodes_and_keys:
        mock_report = AgentReport(
            agent_name=agent_key,
            decision=Decision.HOLD,
            confidence=50.0,
            reasoning="Test",
            data_points={},
        )
        mock_llm = _make_mock_llm(mock_report)

        import importlib

        module = importlib.import_module(module_path)
        node_func = getattr(module, func_name)

        with (
            patch("src.data.fundamental_data.get_financials") as m1,
            patch("src.data.fundamental_data.get_insider_trades") as m2,
            patch("src.data.fundamental_data.get_sec_filings") as m3,
            patch("src.data.fundamental_data.get_analyst_ratings") as m4,
            patch("src.data.news_data.get_company_news") as m5,
            patch("src.data.news_data.get_market_news") as m6,
            patch("src.data.alternative_data.get_fear_greed_index") as m7,
            patch("src.data.earnings_calendar.get_earnings_calendar") as m8,
            patch("src.data.earnings_calendar.get_earnings_estimate") as m9,
            patch("src.data.market_data.get_stock_price") as m10,
            patch("src.data.market_data.get_stock_history") as m11,
            patch("src.data.market_data.get_options_data") as m12,
        ):
            for mock_obj in [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]:
                mock_obj.invoke.return_value = "mock data"

            result = node_func(sample_state, llm=mock_llm)

        assert "agent_reports" in result, f"{func_name} missing 'agent_reports' key"
        assert agent_key in result["agent_reports"], f"{func_name} missing '{agent_key}' sub-key"
        assert isinstance(
            result["agent_reports"][agent_key], AgentReport
        ), f"{func_name} did not return AgentReport"
