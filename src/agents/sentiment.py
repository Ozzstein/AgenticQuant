"""Alt-data / news sentiment analyst node for the LangGraph trading desk pipeline."""

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
You are a market sentiment analyst specializing in news flow, social sentiment, and alternative \
data signals.

Your mandate:
- Assess: news sentiment tone, fear/greed levels, earnings surprise potential, analyst estimate \
  momentum, and macro market backdrop.
- Be alert to sentiment extremes as contrarian signals (e.g. extreme fear = buy opportunity; \
  euphoria = risk-off signal).
- Distinguish between noise and genuine sentiment shifts.

Decision framework:
- STRONG_BUY: Overwhelmingly positive sentiment + fear/greed neutral-to-low (contrarian upside)
- BUY: Positive news flow, improving analyst sentiment, no major negative catalysts
- HOLD: Mixed or neutral sentiment; lack of catalysts
- SELL: Negative news momentum, deteriorating analyst sentiment
- STRONG_SELL: Extreme negative sentiment, panic headlines, or analyst downgrades cluster

Confidence (0-100) reflects recency, breadth, and consistency of sentiment signals.
"""


def sentiment_node(
    state: TradingDeskState,
    llm: BaseChatModel | None = None,
) -> dict:
    """Alt-data / news sentiment analyst node.

    Gathers company news, market news, fear/greed index, analyst ratings, and earnings calendar,
    then uses an LLM to produce a structured AgentReport.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override (for testing). If None, creates from config.

    Returns:
        Partial state dict with agent_reports updated with the "sentiment" report.
    """
    ticker = state["ticker"]
    logger.info("sentiment_node: analysing {}", ticker)

    if llm is None:
        from src.utils.config import get_config

        config = get_config()
        llm = get_llm(config.agents)

    tool_data = ""

    # --- Tool 1: Company-specific news ---
    try:
        from src.data.news_data import get_company_news

        tool_data += get_company_news.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_company_news failed for {}: {}", ticker, exc)
        tool_data += "Company news unavailable.\n"

    # --- Tool 2: Broad market news ---
    try:
        from src.data.news_data import get_market_news

        tool_data += get_market_news.invoke({}) + "\n"
    except Exception as exc:
        logger.warning("get_market_news failed: {}", exc)
        tool_data += "Market news unavailable.\n"

    # --- Tool 3: Fear & greed index ---
    try:
        from src.data.alternative_data import get_fear_greed_index

        tool_data += get_fear_greed_index.invoke({}) + "\n"
    except Exception as exc:
        logger.warning("get_fear_greed_index failed: {}", exc)
        tool_data += "Fear/greed data unavailable.\n"

    # --- Tool 4: Analyst ratings ---
    try:
        from src.data.fundamental_data import get_analyst_ratings

        tool_data += get_analyst_ratings.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_analyst_ratings failed for {}: {}", ticker, exc)
        tool_data += "Analyst rating data unavailable.\n"

    # --- Tool 5: Earnings calendar ---
    try:
        from src.data.earnings_calendar import get_earnings_calendar

        tool_data += get_earnings_calendar.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_earnings_calendar failed for {}: {}", ticker, exc)
        tool_data += "Earnings calendar data unavailable.\n"

    human_msg = (
        f"Analyze {ticker} from a sentiment and alternative data perspective.\n\n"
        f"Data gathered:\n{tool_data}\n\n"
        "Provide your investment decision (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), "
        "confidence (0-100), detailed reasoning, and key data_points as a dict "
        "(e.g. sentiment_score, fear_greed_level, news_tone, analyst_momentum)."
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
        logger.error("sentiment_node LLM call failed for {}: {}", ticker, exc)
        report = AgentReport(
            agent_name="sentiment",
            decision=Decision.HOLD,
            confidence=0.0,
            reasoning=f"Analysis failed: {str(exc)[:200]}",
            data_points={},
        )

    logger.debug(
        "sentiment_node complete for {}: decision={} confidence={}",
        ticker,
        report.decision,
        report.confidence,
    )
    return {"agent_reports": {"sentiment": report}}
