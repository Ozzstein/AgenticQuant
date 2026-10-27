"""Graph and HMM boundary regressions, migrated from unsafe neutral/stub assumptions."""
# ruff: noqa: F401, F811

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.agents.graph import analyze_instrument, analyze_ticker, build_trading_desk_graph
from src.agents.macro_regime import macro_regime_node, market_context_node
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
    jev_service,
    run_desk,
)


class TestBuildGraph:
    def test_returns_compiled_graph(self, desk_config, tmp_path):
        graph = build_trading_desk_graph(desk_config, decision_service=jev_service(tmp_path))
        assert callable(graph.invoke)
        assert {"load_memory", "save_memory", "transcript_nlp", "verify_evidence"} <= set(
            graph.nodes
        )

    def test_requires_explicit_context(self, desk_config, tmp_path):
        graph = build_trading_desk_graph(desk_config, decision_service=jev_service(tmp_path))
        with pytest.raises(ValueError, match="RunContext"):
            graph.invoke({"ticker": "A"})


class TestMarketContextNode:
    def test_returns_source_market_context(self, desk_state):
        assert desk_state["sources"][1].text in market_context_node(desk_state)["market_context"]

    def test_missing_is_not_fabricated_price(self, desk_state):
        assert market_context_node({**desk_state, "sources": ()})["market_context"] == ""

    def test_future_sources_are_excluded(self, desk_state):
        future = tuple(
            s.model_copy(
                update={
                    "record": s.record.model_copy(
                        update={"available_at": desk_state["context"].as_of + timedelta(days=1)}
                    )
                }
            )
            for s in desk_state["sources"]
        )
        assert market_context_node({**desk_state, "sources": future})["market_context"] == ""


class TestMacroRegimeNode:
    @pytest.mark.parametrize("regime", ["risk_on", "neutral", "risk_off", "crisis"])
    def test_all_authoritative_regimes_retained(self, desk_state, regime):
        desk_state["hmm_context"] = desk_state["hmm_context"].model_copy(update={"regime": regime})
        output = macro_regime_node(desk_state)
        assert output["macro_regime"] == regime and output["macro_confidence"] == 0.8

    @pytest.mark.parametrize(
        "text", ["High Volatility", "Extreme Fear", "Low Volatility", "Extreme Greed", "crisis"]
    )
    def test_text_cannot_replace_missing_hmm(self, desk_state, text):
        llm = MagicMock()
        result = macro_regime_node(
            {**desk_state, "hmm_context": None, "market_context": text}, llm=llm
        )
        assert result == {
            "macro_regime": "unavailable",
            "macro_confidence": None,
            "new_exposure_blocked": True,
        }
        llm.invoke.assert_not_called()

    def test_stale_hmm_not_accepted(self, desk_state):
        desk_state["hmm_context"] = desk_state["hmm_context"].model_copy(
            update={"as_of": desk_state["context"].as_of - timedelta(days=1)}
        )
        assert macro_regime_node(desk_state)["macro_regime"] == "unavailable"


class TestGraphInvoke:
    @pytest.mark.parametrize("asset_class", ["equity", "crypto"])
    def test_graph_complete_result(
        self,
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
        asset_class,
    ):
        instrument = decision_instrument.model_copy(update={"asset_class": asset_class})
        result, calls = run_desk(
            tmp_path,
            desk_config,
            decision_context,
            instrument,
            desk_sources,
            desk_hmm,
            desk_portfolio,
        )
        assert result.instrument == instrument and result.action == AgentAction.BUY
        assert result.reports and result.debate_transcript
        if asset_class == "crypto":
            assert all(
                r.status == "not_applicable"
                for r in result.reports
                if r.role in ("earnings", "transcript_nlp")
            )


class TestAnalyzeTicker:
    def test_requires_identity_instead_of_synthetic_fallback(self):
        with pytest.raises(ValueError, match="canonical instrument"):
            analyze_ticker("A")

    @pytest.mark.parametrize("failure", [RuntimeError("fixture"), None])
    def test_failure_abstains_not_liquidates(
        self, decision_context, decision_instrument, desk_config, failure
    ):
        with patch("src.agents.graph.build_trading_desk_graph") as build:
            if failure:
                build.return_value.invoke.side_effect = failure
            else:
                build.return_value.invoke.return_value = {}
            result = analyze_instrument(
                decision_instrument, decision_context, sources=(), config=desk_config
            )
        assert result.action == AgentAction.ABSTAIN and result.new_exposure_blocked

    def test_qlib_context_passed_through(
        self,
        tmp_path,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
        desk_config,
    ):
        calls = []
        analyze_instrument(
            decision_instrument,
            decision_context,
            sources=desk_sources,
            hmm_context=desk_hmm,
            portfolio_context=desk_portfolio,
            config=desk_config,
            qlib_context="Alpha158: 0.7",
            decision_service=jev_service(tmp_path),
            llm_factory=lambda role: GenerativeFixture(role, calls),
        )
        assert calls and all("Alpha158: 0.7" in messages[-1].content for _, messages in calls)
