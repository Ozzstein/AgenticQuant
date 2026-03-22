"""Earnings context handler node for the LangGraph trading desk pipeline."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.logger import get_logger
from src.utils.schemas import AgentReport, Decision

logger = get_logger(__name__)

try:
    from src.agents.llm import get_llm
    from src.agents.state import TradingDeskState
except ImportError:
    import operator
    from typing import Annotated, TypedDict

    from src.utils.schemas import AgentReport, AnalysisResult

    def _merge_reports(a, b):  # type: ignore[return]
        return {**a, **b}

    class TradingDeskState(TypedDict):  # type: ignore[no-redef]
        ticker: str
        asset_class: str
        qlib_context: str
        market_context: str
        macro_regime: str
        macro_confidence: float
        memory_context: str
        agent_reports: Annotated[dict, _merge_reports]
        debate_round: int
        debate_transcript: Annotated[list, operator.add]
        risk_veto: bool
        risk_veto_reason: str
        result: AnalysisResult | None

    def get_llm(config, *, mode="standard", temperature=None, max_tokens=None):  # type: ignore[no-redef]
        raise ImportError("llm.py not available")


_SYSTEM_PROMPT = """\
You are an earnings specialist with expertise in earnings season dynamics, analyst estimate \
revisions, and the market's pricing of earnings risk.

Your mandate:
- Evaluate upcoming earnings risk/opportunity for the stock.
- Estimate beat/miss probability based on analyst estimates vs. historical surprise patterns.
- Assess the magnitude of potential earnings-driven price moves.
- Identify key guidance items the market will focus on.

Decision framework:
- STRONG_BUY: High beat probability, positive estimate revisions, strong whisper numbers
- BUY: Likely beat, consensus may be conservative
- HOLD: Consensus-in-line earnings expected; low event risk
- SELL: Miss risk elevated, guidance likely to disappoint
- STRONG_SELL: High miss probability, estimate cuts expected

Confidence (0-100) reflects data quality and clarity of earnings setup.
"""


def earnings_node(
    state: TradingDeskState,
    llm: BaseChatModel | None = None,
) -> dict:
    """Earnings context handler node.

    Gathers earnings calendar and estimate data, then optionally uses an LLM to produce a
    structured AgentReport. Falls back to a neutral report if the LLM is unavailable.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override (for testing). If None, attempts to create from config.

    Returns:
        Partial state dict with agent_reports updated with the "earnings" report.
    """
    ticker = state["ticker"]
    logger.info("earnings_node: gathering earnings context for {}", ticker)

    earnings_data = ""

    # --- Tool 1: Earnings calendar ---
    try:
        from src.data.earnings_calendar import get_earnings_calendar

        earnings_data += get_earnings_calendar.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_earnings_calendar failed for {}: {}", ticker, exc)
        earnings_data += "Earnings calendar data unavailable.\n"

    # --- Tool 2: Earnings estimates ---
    try:
        from src.data.earnings_calendar import get_earnings_estimate

        earnings_data += get_earnings_estimate.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_earnings_estimate failed for {}: {}", ticker, exc)
        earnings_data += "Earnings estimate data unavailable.\n"

    # --- LLM interpretation ---
    if llm is None:
        try:
            from src.utils.config import get_config

            config = get_config()
            llm = get_llm(config.agents)
        except Exception as exc:
            logger.warning("earnings_node: LLM setup failed, using neutral report: {}", exc)
            report = AgentReport(
                agent_name="earnings",
                decision=Decision.HOLD,
                confidence=40.0,
                reasoning="Earnings data: " + earnings_data[:200],
                data_points={},
            )
            return {"agent_reports": {"earnings": report}}

    human_msg = (
        f"Analyze {ticker} earnings outlook.\n\n"
        f"Data gathered:\n{earnings_data}\n\n"
        "Provide your investment decision (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), "
        "confidence (0-100), detailed reasoning, and key data_points as a dict "
        "(e.g. next_earnings_date, eps_estimate, beat_probability, event_risk_pct)."
    )

    structured_llm = llm.with_structured_output(AgentReport)

    try:
        report: AgentReport = structured_llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ]
        )
    except Exception as exc:
        logger.error("earnings_node LLM call failed for {}: {}", ticker, exc)
        report = AgentReport(
            agent_name="earnings",
            decision=Decision.HOLD,
            confidence=40.0,
            reasoning="Earnings data: " + earnings_data[:200],
            data_points={},
        )

    logger.debug(
        "earnings_node complete for {}: decision={} confidence={}",
        ticker,
        report.decision,
        report.confidence,
    )
    return {"agent_reports": {"earnings": report}}
