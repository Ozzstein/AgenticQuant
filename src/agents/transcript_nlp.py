"""Earnings call transcript NLP analyst node for the LangGraph trading desk pipeline."""

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
You are a specialist in analyzing earnings call transcripts for publicly traded companies.
You have deep expertise in natural language processing of management commentary, with 15 years
of experience on the buy side.

Your mandate:
- Analyze the earnings call transcript for management tone, sentiment, and forward-looking signals.
- Extract: overall management sentiment (bullish/neutral/cautious/bearish), key forward guidance
  statements, risk language and hedging phrases, notable quotable moments, and any changes in
  tone vs. prior quarters.
- Pay special attention to: CEO/CFO language about margins, revenue growth, competitive dynamics,
  capex plans, hiring/layoff signals, and regulatory concerns.
- Distinguish between scripted prepared remarks (less signal) and spontaneous Q&A responses
  (higher signal).
- Flag any linguistic red flags: excessive hedging ("we believe", "we hope"), blame-shifting
  to macro factors, avoidance of direct answers, or unusual changes in communication style.

Decision framework:
- STRONG_BUY: Highly confident, specific forward guidance with upside; management tone genuinely
  optimistic with concrete plans; no hedging language on key metrics
- BUY: Generally positive tone, constructive guidance, manageable risks acknowledged transparently
- HOLD: Neutral or mixed signals; management cautious but not alarming; standard boilerplate
  language dominates
- SELL: Defensive tone, vague guidance, excessive hedging, blame-shifting; management avoiding
  specifics on key concerns
- STRONG_SELL: Alarming language, major guidance cuts, evasive answers on critical questions,
  management credibility concerns

Confidence (0-100) reflects transcript availability, recency, and clarity of sentiment signals.

Your data_points dict MUST include:
- management_sentiment: "bullish" | "neutral" | "cautious" | "bearish"
- guidance_direction: "raised" | "maintained" | "lowered" | "withdrawn" | "not_available"
- tone_vs_prior: "improved" | "stable" | "deteriorated" | "not_comparable"
- hedging_level: "low" | "moderate" | "high"
- key_quote: (most signal-rich quote from the call, max 150 chars)
- transcript_quarter: (e.g., "Q4 2025")
"""


def transcript_nlp_node(
    state: TradingDeskState,
    llm: BaseChatModel | None = None,
) -> dict:
    """Earnings call transcript NLP analyst node.

    Fetches the most recent earnings call transcript and uses an LLM to extract
    management sentiment, forward guidance signals, hedging language, and tone shifts
    to produce a structured AgentReport.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override (for testing). If None, creates from config.

    Returns:
        Partial state dict with agent_reports updated with the "transcript_nlp" report.
    """
    ticker = state["ticker"]
    logger.info("transcript_nlp_node: analysing {}", ticker)

    if llm is None:
        from src.utils.config import get_config

        config = get_config()
        llm = get_llm(config.agents)

    try:
        from src.data.transcript_data import get_earnings_transcript

        tool_data = get_earnings_transcript.invoke({"ticker": ticker})
    except Exception as exc:
        logger.warning("get_earnings_transcript failed for {}: {}", ticker, exc)
        tool_data = "Earnings call transcript unavailable.\n"

    human_msg = (
        f"Analyze {ticker} from an earnings call transcript NLP perspective.\n\n"
        f"Transcript data:\n{tool_data}\n\n"
        "Provide your investment decision (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), "
        "confidence (0-100), detailed reasoning, and key data_points as a dict "
        "(management_sentiment, guidance_direction, tone_vs_prior, hedging_level, "
        "key_quote, transcript_quarter)."
    )

    memory_ctx = state.get("memory_context", "")
    if memory_ctx:
        human_msg += f"\n\nPrior analysis history for {ticker}:\n{memory_ctx}"

    structured_llm = llm.with_structured_output(AgentReport)

    try:
        report: AgentReport = structured_llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ]
        )
    except Exception as exc:
        logger.error("transcript_nlp_node LLM call failed for {}: {}", ticker, exc)
        report = AgentReport(
            agent_name="transcript_nlp",
            decision=Decision.HOLD,
            confidence=0.0,
            reasoning=f"Analysis failed: {str(exc)[:200]}",
            data_points={},
        )

    logger.debug(
        "transcript_nlp_node complete for {}: decision={} confidence={}",
        ticker,
        report.decision,
        report.confidence,
    )
    return {"agent_reports": {"transcript_nlp": report}}
