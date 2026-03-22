"""CFA fundamental analyst node for the LangGraph trading desk pipeline."""

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
You are a CFA-certified fundamental analyst with 20 years of experience analyzing public equities.

Your mandate:
- Analyze the company's financial health, valuation, competitive position, and growth prospects.
- Focus on: revenue trends, profit margins, balance sheet strength, valuation multiples vs. peers,
  insider activity, SEC filings, and analyst consensus.
- Output must be concise, data-driven, and actionable.

Decision framework:
- STRONG_BUY: Compelling valuation discount + strong fundamentals + positive catalysts
- BUY: Solid fundamentals, reasonable valuation, positive trajectory
- HOLD: Mixed signals or fair value; no clear edge
- SELL: Deteriorating fundamentals or overvalued relative to peers
- STRONG_SELL: Significant financial distress or severe overvaluation

Confidence (0-100) reflects data quality and conviction in the call.
"""


def fundamental_node(
    state: TradingDeskState,
    llm: BaseChatModel | None = None,
) -> dict:
    """CFA-style fundamental analyst node.

    Gathers financial data, insider activity, SEC filings, and analyst ratings,
    then uses an LLM to produce a structured AgentReport.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override (for testing). If None, creates from config.

    Returns:
        Partial state dict with agent_reports updated with the "fundamental" report.
    """
    ticker = state["ticker"]
    logger.info("fundamental_node: analysing {}", ticker)

    if llm is None:
        from src.utils.config import get_config

        config = get_config()
        llm = get_llm(config.agents)

    tool_data = ""

    # --- Tool 1: Core financials ---
    try:
        from src.data.fundamental_data import get_financials

        tool_data += get_financials.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_financials failed for {}: {}", ticker, exc)
        tool_data += "Core financial data unavailable.\n"

    # --- Tool 2: Financial statements (may not exist yet) ---
    try:
        from src.data.fundamental_data import get_financial_statements  # type: ignore[attr-defined]

        tool_data += get_financial_statements.invoke({"ticker": ticker}) + "\n"
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        logger.warning("get_financial_statements failed for {}: {}", ticker, exc)

    # --- Tool 3: Financial ratios (may not exist yet) ---
    try:
        from src.data.fundamental_data import get_financial_ratios  # type: ignore[attr-defined]

        tool_data += get_financial_ratios.invoke({"ticker": ticker}) + "\n"
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        logger.warning("get_financial_ratios failed for {}: {}", ticker, exc)

    # --- Tool 4: Insider trades ---
    try:
        from src.data.fundamental_data import get_insider_trades

        tool_data += get_insider_trades.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_insider_trades failed for {}: {}", ticker, exc)
        tool_data += "Insider trade data unavailable.\n"

    # --- Tool 5: SEC filings ---
    try:
        from src.data.fundamental_data import get_sec_filings

        tool_data += get_sec_filings.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_sec_filings failed for {}: {}", ticker, exc)
        tool_data += "SEC filing data unavailable.\n"

    # --- Tool 6: Analyst ratings ---
    try:
        from src.data.fundamental_data import get_analyst_ratings

        tool_data += get_analyst_ratings.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_analyst_ratings failed for {}: {}", ticker, exc)
        tool_data += "Analyst rating data unavailable.\n"

    human_msg = (
        f"Analyze {ticker} from a fundamental perspective.\n\n"
        f"Data gathered:\n{tool_data}\n\n"
        "Provide your investment decision (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), "
        "confidence (0-100), detailed reasoning, and key data_points as a dict "
        "(e.g. pe_ratio, revenue_growth, debt_equity, roe)."
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
        logger.error("fundamental_node LLM call failed for {}: {}", ticker, exc)
        report = AgentReport(
            agent_name="fundamental",
            decision=Decision.HOLD,
            confidence=0.0,
            reasoning=f"Analysis failed: {str(exc)[:200]}",
            data_points={},
        )

    logger.debug(
        "fundamental_node complete for {}: decision={} confidence={}",
        ticker,
        report.decision,
        report.confidence,
    )
    return {"agent_reports": {"fundamental": report}}
