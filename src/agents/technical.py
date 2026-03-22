"""CMT technical analyst node for the LangGraph trading desk pipeline."""

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
You are a CMT-certified technical analyst with deep expertise in price action, market structure, \
and derivatives flow.

Your mandate:
- Analyze price action, trend direction, momentum, support/resistance levels, and options flow.
- Key signals: trend direction (above/below moving averages), momentum oscillators (RSI/MACD), \
  volatility regime (ATR, Bollinger Bands), and options put/call ratio.
- Respect the primary trend; identify key inflection points and risk/reward setups.
- Volume confirmation is essential for breakouts.

Decision framework:
- STRONG_BUY: Clear uptrend, momentum confirmation, strong volume, bullish options flow
- BUY: Uptrend intact, constructive momentum, holding key support
- HOLD: Consolidation, mixed momentum, no clear directional edge
- SELL: Trend breakdown, negative momentum divergence, below key moving averages
- STRONG_SELL: Confirmed downtrend, oversold bounce risk aside, heavy put activity

Confidence (0-100) reflects clarity of technical signals and consistency across timeframes.
"""


def technical_node(
    state: TradingDeskState,
    llm: BaseChatModel | None = None,
) -> dict:
    """CMT technical analyst node.

    Gathers price data, historical OHLCV, technical indicators, support/resistance levels,
    options data, and sector comparison, then uses an LLM to produce a structured AgentReport.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override (for testing). If None, creates from config.

    Returns:
        Partial state dict with agent_reports updated with the "technical" report.
    """
    ticker = state["ticker"]
    logger.info("technical_node: analysing {}", ticker)

    if llm is None:
        from src.utils.config import get_config

        config = get_config()
        llm = get_llm(config.agents)

    tool_data = ""

    # --- Tool 1: Current price ---
    try:
        from src.data.market_data import get_stock_price

        tool_data += get_stock_price.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_stock_price failed for {}: {}", ticker, exc)
        tool_data += "Current price data unavailable.\n"

    # --- Tool 2: Price history ---
    try:
        from src.data.market_data import get_stock_history

        tool_data += get_stock_history.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_stock_history failed for {}: {}", ticker, exc)
        tool_data += "Price history unavailable.\n"

    # --- Tool 3: Technical indicators (may not exist yet) ---
    try:
        from src.data.market_data import get_technical_indicators  # type: ignore[attr-defined]

        tool_data += get_technical_indicators.invoke({"ticker": ticker}) + "\n"
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        logger.warning("get_technical_indicators failed for {}: {}", ticker, exc)

    # --- Tool 4: Support/resistance (may not exist yet) ---
    try:
        from src.data.market_data import get_support_resistance  # type: ignore[attr-defined]

        tool_data += get_support_resistance.invoke({"ticker": ticker}) + "\n"
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        logger.warning("get_support_resistance failed for {}: {}", ticker, exc)

    # --- Tool 5: Options data ---
    try:
        from src.data.market_data import get_options_data

        tool_data += get_options_data.invoke({"ticker": ticker}) + "\n"
    except Exception as exc:
        logger.warning("get_options_data failed for {}: {}", ticker, exc)
        tool_data += "Options data unavailable.\n"

    # --- Tool 6: Sector comparison (may not exist yet) ---
    try:
        from src.data.market_data import get_sector_comparison  # type: ignore[attr-defined]

        tool_data += get_sector_comparison.invoke({"ticker": ticker}) + "\n"
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        logger.warning("get_sector_comparison failed for {}: {}", ticker, exc)

    human_msg = (
        f"Analyze {ticker} from a technical analysis perspective.\n\n"
        f"Data gathered:\n{tool_data}\n\n"
        "Provide your investment decision (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), "
        "confidence (0-100), detailed reasoning, and key data_points as a dict "
        "(e.g. rsi, macd_signal, trend_direction, above_sma200, put_call_ratio)."
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
        logger.error("technical_node LLM call failed for {}: {}", ticker, exc)
        report = AgentReport(
            agent_name="technical",
            decision=Decision.HOLD,
            confidence=0.0,
            reasoning=f"Analysis failed: {str(exc)[:200]}",
            data_points={},
        )

    logger.debug(
        "technical_node complete for {}: decision={} confidence={}",
        ticker,
        report.decision,
        report.confidence,
    )
    return {"agent_reports": {"technical": report}}
