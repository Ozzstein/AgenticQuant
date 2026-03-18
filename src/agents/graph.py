"""LangGraph orchestration graph for the multi-agent trading desk."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# State — use fallback if state.py not merged yet
# ---------------------------------------------------------------------------
try:
    from src.agents.state import TradingDeskState
except ImportError:
    import operator
    from typing import Annotated, TypedDict

    from src.utils.schemas import AnalysisResult

    def _merge_reports(a: dict, b: dict) -> dict:
        return {**a, **b}

    class TradingDeskState(TypedDict):  # type: ignore[no-redef]
        ticker: str
        asset_class: str
        qlib_context: str
        market_context: str
        macro_regime: str
        macro_confidence: float
        agent_reports: Annotated[dict, _merge_reports]
        debate_round: int
        debate_transcript: Annotated[list, operator.add]
        risk_veto: bool
        risk_veto_reason: str
        result: AnalysisResult | None


# ---------------------------------------------------------------------------
# Node loader helpers — lazy imports with graceful stubs
# ---------------------------------------------------------------------------


def _load_market_context_node():
    """Load market_context_node, falling back to a stub."""
    try:
        from src.agents.macro_regime import market_context_node

        return market_context_node
    except ImportError:

        def market_context_node(state):  # type: ignore[misc]
            return {"market_context": "No market data available"}

        return market_context_node


def _load_macro_regime_node():
    """Load macro_regime_node, falling back to a stub."""
    try:
        from src.agents.macro_regime import macro_regime_node

        return macro_regime_node
    except ImportError:

        def macro_regime_node(state):  # type: ignore[misc]
            return {"macro_regime": "neutral", "macro_confidence": 50.0}

        return macro_regime_node


def _load_fundamental_node():
    """Load fundamental_node, falling back to a stub."""
    try:
        from src.agents.fundamental import fundamental_node

        return fundamental_node
    except ImportError:
        from src.utils.schemas import AgentReport, Decision

        def fundamental_node(state):  # type: ignore[misc]
            return {
                "agent_reports": {
                    "fundamental": AgentReport(
                        agent_name="fundamental",
                        decision=Decision.HOLD,
                        confidence=50.0,
                        reasoning="Stub",
                        data_points={},
                    )
                }
            }

        return fundamental_node


def _load_sentiment_node():
    """Load sentiment_node, falling back to a stub."""
    try:
        from src.agents.sentiment import sentiment_node

        return sentiment_node
    except ImportError:
        from src.utils.schemas import AgentReport, Decision

        def sentiment_node(state):  # type: ignore[misc]
            return {
                "agent_reports": {
                    "sentiment": AgentReport(
                        agent_name="sentiment",
                        decision=Decision.HOLD,
                        confidence=50.0,
                        reasoning="Stub",
                        data_points={},
                    )
                }
            }

        return sentiment_node


def _load_technical_node():
    """Load technical_node, falling back to a stub."""
    try:
        from src.agents.technical import technical_node

        return technical_node
    except ImportError:
        from src.utils.schemas import AgentReport, Decision

        def technical_node(state):  # type: ignore[misc]
            return {
                "agent_reports": {
                    "technical": AgentReport(
                        agent_name="technical",
                        decision=Decision.HOLD,
                        confidence=50.0,
                        reasoning="Stub",
                        data_points={},
                    )
                }
            }

        return technical_node


def _load_earnings_node():
    """Load earnings_node, falling back to a stub."""
    try:
        from src.agents.earnings_handler import earnings_node

        return earnings_node
    except ImportError:
        from src.utils.schemas import AgentReport, Decision

        def earnings_node(state):  # type: ignore[misc]
            return {
                "agent_reports": {
                    "earnings": AgentReport(
                        agent_name="earnings",
                        decision=Decision.HOLD,
                        confidence=40.0,
                        reasoning="Stub",
                        data_points={},
                    )
                }
            }

        return earnings_node


def _load_risk_manager_node():
    """Load risk_manager_node, falling back to a stub."""
    try:
        from src.agents.risk_manager import risk_manager_node

        return risk_manager_node
    except ImportError:

        def risk_manager_node(state):  # type: ignore[misc]
            return {"risk_veto": False, "risk_veto_reason": "", "agent_reports": {}}

        return risk_manager_node


def _load_debate_nodes():
    """Load debate_node and debate_router, falling back to stubs."""
    try:
        from src.agents.debate import debate_node, debate_router

        return debate_node, debate_router
    except ImportError:

        def debate_node(state):  # type: ignore[misc]
            return {
                "debate_round": state.get("debate_round", 0) + 1,
                "debate_transcript": [],
            }

        def debate_router(state) -> str:  # type: ignore[misc]
            return "risk_manager"

        return debate_node, debate_router


def _load_strategist_node():
    """Load strategist_node, falling back to a stub."""
    try:
        from src.agents.strategist import strategist_node

        return strategist_node
    except ImportError:
        from src.utils.schemas import AnalysisResult, Decision

        def strategist_node(state):  # type: ignore[misc]
            return {
                "result": AnalysisResult(
                    ticker=state["ticker"],
                    decision=Decision.HOLD,
                    confidence=50.0,
                    reasoning="Stub strategist",
                )
            }

        return strategist_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_trading_desk_graph(config: AppConfig | None = None):
    """Build and compile the LangGraph trading desk workflow.

    LangGraph topology::

        START → market_context → macro_regime
          → [fundamental, sentiment, technical, earnings]  (parallel fan-out)
          → pre_debate  (fan-in barrier)
          → debate_router → [debate (loop)] or risk_manager
          → strategist → END

    Args:
        config: Optional AppConfig. Uses get_config() singleton if None.

    Returns:
        Compiled LangGraph StateGraph ready for ``invoke()``.
    """
    if config is None:
        config = get_config()

    # Resolve TradingDeskState locally (may be the fallback defined at module level)
    try:
        from src.agents.state import TradingDeskState as _State
    except ImportError:
        _State = TradingDeskState  # type: ignore[misc]

    # Load nodes (lazy, with stubs)
    market_context_node = _load_market_context_node()
    macro_regime_node = _load_macro_regime_node()
    fundamental_node = _load_fundamental_node()
    sentiment_node = _load_sentiment_node()
    technical_node = _load_technical_node()
    earnings_node = _load_earnings_node()
    risk_manager_node = _load_risk_manager_node()
    debate_node, debate_router = _load_debate_nodes()
    strategist_node = _load_strategist_node()

    def pre_debate_node(state: _State) -> dict:  # type: ignore[valid-type]
        """Pass-through fan-in node: convergence point after parallel analysts."""
        return {}

    # ------------------------------------------------------------------
    graph = StateGraph(_State)

    # Register nodes
    graph.add_node("market_context", market_context_node)
    graph.add_node("macro_regime", macro_regime_node)
    graph.add_node("fundamental", fundamental_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("technical", technical_node)
    graph.add_node("earnings", earnings_node)
    graph.add_node("pre_debate", pre_debate_node)
    graph.add_node("debate", debate_node)
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("strategist", strategist_node)

    # Sequential start
    graph.add_edge(START, "market_context")
    graph.add_edge("market_context", "macro_regime")

    # Fan-out: macro_regime → all four parallel analysts
    graph.add_edge("macro_regime", "fundamental")
    graph.add_edge("macro_regime", "sentiment")
    graph.add_edge("macro_regime", "technical")
    graph.add_edge("macro_regime", "earnings")

    # Fan-in: all four analysts → pre_debate
    graph.add_edge("fundamental", "pre_debate")
    graph.add_edge("sentiment", "pre_debate")
    graph.add_edge("technical", "pre_debate")
    graph.add_edge("earnings", "pre_debate")

    # Conditional routing after fan-in (and after each debate round)
    graph.add_conditional_edges(
        "pre_debate",
        debate_router,
        {"debate": "debate", "risk_manager": "risk_manager"},
    )
    graph.add_conditional_edges(
        "debate",
        debate_router,
        {"debate": "debate", "risk_manager": "risk_manager"},
    )

    # Final synthesis
    graph.add_edge("risk_manager", "strategist")
    graph.add_edge("strategist", END)

    logger.info("Trading desk graph compiled successfully.")
    return graph.compile()


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def analyze_ticker(
    ticker: str,
    asset_class: str = "EQUITY",
    qlib_context: str = "",
    config: AppConfig | None = None,
) -> "AnalysisResult":
    """Run the full multi-agent analysis pipeline for a ticker.

    Args:
        ticker: Stock or crypto ticker (e.g., ``"AAPL"``, ``"BTC/USDT"``).
        asset_class: ``"EQUITY"`` or ``"CRYPTO"``.
        qlib_context: Optional serialized Qlib alpha scores.
        config: Optional AppConfig override.

    Returns:
        AnalysisResult with complete multi-agent decision.
    """
    from src.utils.schemas import AnalysisResult, Decision

    graph = build_trading_desk_graph(config)

    initial_state = {
        "ticker": ticker,
        "asset_class": asset_class,
        "qlib_context": qlib_context,
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

    try:
        final_state = graph.invoke(initial_state)
        result = final_state.get("result")
        if result is None:
            raise ValueError("Graph did not produce a result")
        return result
    except Exception as e:
        logger.error(f"analyze_ticker failed for {ticker}: {e}")
        return AnalysisResult(
            ticker=ticker,
            decision=Decision.HOLD,
            confidence=0.0,
            reasoning=f"Analysis pipeline failed: {str(e)[:200]}",
        )
