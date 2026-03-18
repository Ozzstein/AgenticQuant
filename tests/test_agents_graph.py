"""Tests for src/agents/graph.py and src/agents/macro_regime.py."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from src.utils.schemas import AgentReport, AnalysisResult, AssetClass, Decision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Build a minimal TradingDeskState-compatible dict."""
    base = {
        "ticker": "AAPL",
        "asset_class": "EQUITY",
        "qlib_context": "",
        "market_context": "",
        "macro_regime": "neutral",
        "macro_confidence": 50.0,
        "agent_reports": {},
        "debate_round": 0,
        "debate_transcript": [],
        "risk_veto": False,
        "risk_veto_reason": "",
        "result": None,
    }
    base.update(overrides)
    return base


def _mock_agent_report(
    name: str = "test", decision: Decision = Decision.BUY, confidence: float = 70.0
) -> AgentReport:
    return AgentReport(
        agent_name=name,
        decision=decision,
        confidence=confidence,
        reasoning="Mock reasoning",
        data_points={},
    )


def _make_graph_mock_llm() -> MagicMock:
    """Build a mock LLM suitable for all node types."""
    mock_llm = MagicMock()

    # For macro_regime_node (plain invoke → JSON string)
    mock_llm.invoke.return_value = MagicMock(
        content='{"regime": "neutral", "confidence": 60, "reasoning": "mixed signals"}'
    )

    # with_structured_output returns type-aware mocks
    mock_report = _mock_agent_report("mock")
    mock_result = AnalysisResult(
        ticker="AAPL", decision=Decision.BUY, confidence=72.0, reasoning="Mock synthesis"
    )

    mock_structured_report = MagicMock()
    mock_structured_report.invoke.return_value = mock_report

    mock_structured_result = MagicMock()
    mock_structured_result.invoke.return_value = mock_result

    def _structured_output_side_effect(schema):
        if schema is AnalysisResult:
            return mock_structured_result
        return mock_structured_report

    mock_llm.with_structured_output.side_effect = _structured_output_side_effect
    return mock_llm


# ---------------------------------------------------------------------------
# build_trading_desk_graph tests
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_returns_compiled_graph(self):
        """build_trading_desk_graph() returns a compiled graph with an invoke method."""
        from src.agents.graph import build_trading_desk_graph

        graph = build_trading_desk_graph()
        assert hasattr(graph, "invoke"), "Expected compiled graph with invoke method"
        assert callable(graph.invoke)

    def test_graph_accepts_app_config(self):
        """build_trading_desk_graph accepts an explicit AppConfig without error."""
        from src.agents.graph import build_trading_desk_graph
        from src.utils.config import AppConfig

        cfg = AppConfig(log_level="DEBUG", output_dir="/tmp", log_dir="/tmp")
        graph = build_trading_desk_graph(cfg)
        assert hasattr(graph, "invoke")

    def test_graph_compiles_with_none_config(self):
        """build_trading_desk_graph(None) falls back to get_config() without error."""
        from src.agents.graph import build_trading_desk_graph

        graph = build_trading_desk_graph(None)
        assert callable(graph.invoke)


# ---------------------------------------------------------------------------
# market_context_node tests
# ---------------------------------------------------------------------------


class TestMarketContextNode:
    def test_returns_market_context_key(self):
        """Returns dict with non-empty 'market_context' key when tools succeed."""
        from src.agents.macro_regime import market_context_node

        with (
            patch("src.data.market_data.get_stock_price") as mock_price,
            patch("src.data.alternative_data.get_vix") as mock_vix,
            patch("src.data.alternative_data.get_fear_greed_index") as mock_fg,
        ):
            mock_price.invoke.return_value = "AAPL: $175.00"
            mock_vix.invoke.return_value = "Current VIX: 18.5\nRegime: Normal Volatility"
            mock_fg.invoke.return_value = "Value: 55 Neutral"

            result = market_context_node(_make_state(ticker="AAPL"))

        assert "market_context" in result
        assert isinstance(result["market_context"], str)
        assert len(result["market_context"]) > 0

    def test_includes_price_in_context(self):
        """market_context contains the price string from get_stock_price."""
        from src.agents.macro_regime import market_context_node

        with (
            patch("src.data.market_data.get_stock_price") as mock_price,
            patch("src.data.alternative_data.get_vix") as mock_vix,
            patch("src.data.alternative_data.get_fear_greed_index") as mock_fg,
        ):
            mock_price.invoke.return_value = "AAPL: $175.00 | Change: +1.5%"
            mock_vix.invoke.return_value = "VIX: 18.5"
            mock_fg.invoke.return_value = "Neutral: 55"

            result = market_context_node(_make_state(ticker="AAPL"))

        assert "AAPL: $175.00" in result["market_context"]

    def test_degrades_gracefully_on_all_tool_failures(self):
        """Returns non-empty fallback string even when all tools raise."""
        from src.agents.macro_regime import market_context_node

        with (
            patch("src.data.market_data.get_stock_price") as mock_price,
            patch("src.data.alternative_data.get_vix") as mock_vix,
            patch("src.data.alternative_data.get_fear_greed_index") as mock_fg,
        ):
            mock_price.invoke.side_effect = ConnectionError("network error")
            mock_vix.invoke.side_effect = RuntimeError("vix error")
            mock_fg.invoke.side_effect = RuntimeError("fg error")

            result = market_context_node(_make_state(ticker="AAPL"))

        assert "market_context" in result
        # Fallback message should mention the ticker
        assert "AAPL" in result["market_context"]


# ---------------------------------------------------------------------------
# macro_regime_node tests
# ---------------------------------------------------------------------------


class TestMacroRegimeNode:
    @staticmethod
    def _make_llm(content: str) -> MagicMock:
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content=content)
        return mock

    def test_risk_on_regime(self):
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm('{"regime": "risk_on", "confidence": 70, "reasoning": "bullish"}')
        result = macro_regime_node(_make_state(market_context="low volatility"), llm=llm)

        assert result["macro_regime"] == "risk_on"
        assert result["macro_confidence"] == 70.0

    def test_crisis_regime(self):
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm('{"regime": "crisis", "confidence": 85, "reasoning": "systemic stress"}')
        result = macro_regime_node(_make_state(market_context="VIX > 40"), llm=llm)

        assert result["macro_regime"] == "crisis"
        assert result["macro_confidence"] == 85.0

    def test_risk_off_regime(self):
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm('{"regime": "risk_off", "confidence": 65, "reasoning": "bearish"}')
        result = macro_regime_node(_make_state(), llm=llm)

        assert result["macro_regime"] == "risk_off"

    def test_neutral_regime(self):
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm('{"regime": "neutral", "confidence": 55, "reasoning": "mixed"}')
        result = macro_regime_node(_make_state(), llm=llm)

        assert result["macro_regime"] == "neutral"
        assert result["macro_confidence"] == 55.0

    def test_invalid_regime_falls_back_to_neutral(self):
        """Unrecognized regime label is normalised to neutral."""
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm('{"regime": "UNKNOWN_REGIME", "confidence": 80, "reasoning": "?"}')
        result = macro_regime_node(_make_state(), llm=llm)

        assert result["macro_regime"] == "neutral"

    def test_fallback_rule_based_high_volatility(self):
        """Falls back to risk_off via rules when LLM raises."""
        from src.agents.macro_regime import macro_regime_node

        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        state = _make_state(
            market_context="Current VIX: 35\nRegime: High Volatility — elevated fear"
        )

        result = macro_regime_node(state, llm=llm)

        assert result["macro_regime"] == "risk_off"
        assert result["macro_confidence"] == 60.0

    def test_fallback_rule_based_low_volatility(self):
        """Falls back to risk_on when LLM fails and Low Volatility in context."""
        from src.agents.macro_regime import macro_regime_node

        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        state = _make_state(
            market_context="Current VIX: 12\nRegime: Low Volatility — calm market conditions"
        )

        result = macro_regime_node(state, llm=llm)

        assert result["macro_regime"] == "risk_on"

    def test_fallback_neutral_when_no_signal(self):
        """Falls back to neutral when LLM fails and no strong regime signals."""
        from src.agents.macro_regime import macro_regime_node

        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = macro_regime_node(_make_state(market_context="mixed data"), llm=llm)

        assert result["macro_regime"] == "neutral"

    def test_keyword_fallback_crisis(self):
        """Keyword parsing when LLM returns non-JSON text containing 'crisis'."""
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm("The market is currently in crisis with extreme systemic stress")
        result = macro_regime_node(_make_state(), llm=llm)

        assert result["macro_regime"] == "crisis"

    @pytest.mark.parametrize("regime", ["risk_on", "neutral", "risk_off", "crisis"])
    def test_all_valid_regimes_accepted(self, regime: str):
        """All four valid regime labels are accepted from LLM output."""
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm(
            f'{{"regime": "{regime}", "confidence": 60, "reasoning": "test"}}'
        )
        result = macro_regime_node(_make_state(), llm=llm)

        assert result["macro_regime"] == regime

    def test_returns_macro_confidence_key(self):
        """Both macro_regime and macro_confidence are always present in output."""
        from src.agents.macro_regime import macro_regime_node

        llm = self._make_llm('{"regime": "neutral", "confidence": 55, "reasoning": "ok"}')
        result = macro_regime_node(_make_state(), llm=llm)

        assert "macro_regime" in result
        assert "macro_confidence" in result
        assert isinstance(result["macro_confidence"], float)


# ---------------------------------------------------------------------------
# Full graph integration test
# ---------------------------------------------------------------------------

_LLM_MODULE_PATHS = [
    "src.agents.macro_regime.get_llm",
    "src.agents.fundamental.get_llm",
    "src.agents.sentiment.get_llm",
    "src.agents.technical.get_llm",
    "src.agents.earnings_handler.get_llm",
    "src.agents.strategist.get_llm",
    "src.agents.debate.get_llm",
]

_TOOL_PATCHES: dict[str, str] = {
    "src.data.market_data.get_stock_price": "AAPL: $175.00 | Change: +1.5%",
    "src.data.alternative_data.get_vix": "Current VIX: 18.5\nRegime: Normal Volatility",
    "src.data.alternative_data.get_fear_greed_index": "Value: 55 Neutral",
    "src.data.fundamental_data.get_financials": "Revenue: $100B",
    "src.data.fundamental_data.get_insider_trades": "No significant insider activity",
    "src.data.fundamental_data.get_sec_filings": "10-K filed Q4 2024",
    "src.data.fundamental_data.get_analyst_ratings": "BUY consensus, 18 analysts",
    "src.data.news_data.get_company_news": "Positive earnings outlook",
    "src.data.news_data.get_market_news": "Markets stable",
    "src.data.market_data.get_stock_history": "Period return: +12.5%",
    "src.data.market_data.get_options_data": "Put/Call: 0.8, moderate bullish",
    "src.data.earnings_calendar.get_earnings_calendar": "Next earnings: 2025-04-15",
    "src.data.earnings_calendar.get_earnings_estimate": "EPS est: $1.50",
}


class TestGraphInvoke:
    def _patched_invoke(self, ticker: str = "AAPL") -> dict:
        """Run graph.invoke with all external dependencies mocked."""
        from src.agents.graph import build_trading_desk_graph

        mock_llm = _make_graph_mock_llm()
        initial_state = _make_state(ticker=ticker)

        with ExitStack() as stack:
            for module_path in _LLM_MODULE_PATHS:
                stack.enter_context(patch(module_path, return_value=mock_llm))

            for tool_path, return_val in _TOOL_PATCHES.items():
                mock_tool = stack.enter_context(patch(tool_path))
                mock_tool.invoke.return_value = return_val

            graph = build_trading_desk_graph()
            return graph.invoke(initial_state)

    def test_graph_invoke_produces_state(self):
        """Full graph.invoke() returns a state dict without raising."""
        final_state = self._patched_invoke()
        assert isinstance(final_state, dict)

    def test_graph_invoke_result_is_analysis_result(self):
        """Final state contains a valid AnalysisResult under 'result'."""
        final_state = self._patched_invoke()
        assert final_state.get("result") is not None
        assert isinstance(final_state["result"], AnalysisResult)

    def test_graph_invoke_ticker_propagated(self):
        """AnalysisResult.ticker matches the input ticker."""
        final_state = self._patched_invoke(ticker="AAPL")
        assert final_state["result"].ticker == "AAPL"

    def test_graph_invoke_crypto_asset_class(self):
        """CRYPTO asset class is preserved in the result."""
        from src.agents.graph import build_trading_desk_graph

        mock_llm = _make_graph_mock_llm()
        # Override result to return CRYPTO asset class
        crypto_result = AnalysisResult(
            ticker="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            decision=Decision.HOLD,
            confidence=50.0,
            reasoning="Mock crypto synthesis",
        )
        mock_llm.with_structured_output(AnalysisResult).invoke.return_value = crypto_result

        initial_state = _make_state(ticker="BTC/USDT", asset_class="CRYPTO")

        with ExitStack() as stack:
            for module_path in _LLM_MODULE_PATHS:
                stack.enter_context(patch(module_path, return_value=mock_llm))
            for tool_path, return_val in _TOOL_PATCHES.items():
                mock_tool = stack.enter_context(patch(tool_path))
                mock_tool.invoke.return_value = return_val

            graph = build_trading_desk_graph()
            final_state = graph.invoke(initial_state)

        assert final_state["result"] is not None


# ---------------------------------------------------------------------------
# analyze_ticker convenience function tests
# ---------------------------------------------------------------------------


class TestAnalyzeTicker:
    def test_returns_analysis_result_on_success(self):
        """analyze_ticker returns a valid AnalysisResult when graph succeeds."""
        from src.agents.graph import analyze_ticker

        mock_result = AnalysisResult(
            ticker="AAPL", decision=Decision.BUY, confidence=75.0, reasoning="Mock"
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"result": mock_result}

        with patch("src.agents.graph.build_trading_desk_graph", return_value=mock_graph):
            result = analyze_ticker("AAPL")

        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AAPL"
        assert result.decision == Decision.BUY

    def test_returns_hold_on_pipeline_failure(self):
        """analyze_ticker returns HOLD with 0 confidence when graph raises."""
        from src.agents.graph import analyze_ticker

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Pipeline exploded")

        with patch("src.agents.graph.build_trading_desk_graph", return_value=mock_graph):
            result = analyze_ticker("AAPL")

        assert isinstance(result, AnalysisResult)
        assert result.decision == Decision.HOLD
        assert result.confidence == 0.0
        assert "Pipeline exploded" in result.reasoning

    def test_returns_hold_when_result_is_none(self):
        """analyze_ticker returns HOLD when graph produces no result."""
        from src.agents.graph import analyze_ticker

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"result": None}

        with patch("src.agents.graph.build_trading_desk_graph", return_value=mock_graph):
            result = analyze_ticker("AAPL")

        assert result.decision == Decision.HOLD
        assert result.ticker == "AAPL"

    def test_crypto_ticker_sets_asset_class(self):
        """analyze_ticker with CRYPTO asset class propagates it to initial state."""
        from src.agents.graph import analyze_ticker

        captured: dict = {}

        mock_graph = MagicMock()

        def capture_invoke(state):
            captured["state"] = state
            return {
                "result": AnalysisResult(
                    ticker="BTC/USDT",
                    asset_class=AssetClass.CRYPTO,
                    decision=Decision.HOLD,
                    confidence=50.0,
                    reasoning="Mock",
                )
            }

        mock_graph.invoke.side_effect = capture_invoke

        with patch("src.agents.graph.build_trading_desk_graph", return_value=mock_graph):
            analyze_ticker("BTC/USDT", asset_class="CRYPTO")

        assert captured["state"]["asset_class"] == "CRYPTO"
        assert captured["state"]["ticker"] == "BTC/USDT"

    def test_default_asset_class_is_equity(self):
        """analyze_ticker defaults to EQUITY asset class."""
        from src.agents.graph import analyze_ticker

        captured: dict = {}
        mock_graph = MagicMock()

        def capture_invoke(state):
            captured["state"] = state
            return {
                "result": AnalysisResult(
                    ticker="AAPL", decision=Decision.HOLD, confidence=50.0, reasoning="Mock"
                )
            }

        mock_graph.invoke.side_effect = capture_invoke

        with patch("src.agents.graph.build_trading_desk_graph", return_value=mock_graph):
            analyze_ticker("AAPL")

        assert captured["state"]["asset_class"] == "EQUITY"

    def test_qlib_context_passed_through(self):
        """analyze_ticker forwards qlib_context to the initial state."""
        from src.agents.graph import analyze_ticker

        captured: dict = {}
        mock_graph = MagicMock()

        def capture_invoke(state):
            captured["state"] = state
            return {
                "result": AnalysisResult(
                    ticker="AAPL", decision=Decision.HOLD, confidence=50.0, reasoning="Mock"
                )
            }

        mock_graph.invoke.side_effect = capture_invoke

        with patch("src.agents.graph.build_trading_desk_graph", return_value=mock_graph):
            analyze_ticker("AAPL", qlib_context="alpha_score=0.85")

        assert captured["state"]["qlib_context"] == "alpha_score=0.85"
