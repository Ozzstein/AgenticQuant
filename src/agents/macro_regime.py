"""Macro regime detector node for the trading desk LangGraph pipeline."""

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.schemas import AgentReport, Decision  # noqa: F401 (used by callers)

logger = get_logger(__name__)

# State type — use fallback if state.py not merged yet
try:
    from src.agents.llm import get_llm
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

    def get_llm(config, *, mode: str = "standard", temperature=None, max_tokens=None):  # type: ignore[no-redef]
        """Stub get_llm — raises ImportError if llm.py not available."""
        raise ImportError("llm.py not available")


VALID_REGIMES: frozenset[str] = frozenset(["risk_on", "neutral", "risk_off", "crisis"])

_SYSTEM_PROMPT = """You are a macro analyst. Based on the market data provided, classify the current \
market regime as exactly one of:
- risk_on: Bullish, low volatility, strong momentum
- neutral: Mixed signals, no clear trend
- risk_off: Bearish, elevated volatility, risk aversion
- crisis: Extreme stress, VIX > 30, systemic risk

Respond with ONLY a JSON object: \
{"regime": "<one of the four options>", "confidence": <0-100>, "reasoning": "<1 sentence>"}"""


def market_context_node(state: TradingDeskState, llm: BaseChatModel | None = None) -> dict:
    """Gather market context data for the ticker.

    Fetches price, VIX, fear/greed, and sector data to build a market context string.
    No LLM call — pure data gathering.

    Args:
        state: Current state with ticker set.
        llm: Unused, included for consistent interface.

    Returns:
        Partial state with market_context populated.
    """
    ticker = state["ticker"]
    context_parts: list[str] = []

    # Try to get price data
    try:
        from src.data.market_data import get_stock_price

        context_parts.append(get_stock_price.invoke({"ticker": ticker}))
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
        context_parts.append(f"Price data unavailable for {ticker}")

    # Try to get VIX
    try:
        from src.data.alternative_data import get_vix

        context_parts.append(get_vix.invoke({}))
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")

    # Try fear/greed
    try:
        from src.data.alternative_data import get_fear_greed_index

        context_parts.append(get_fear_greed_index.invoke({}))
    except Exception as e:
        logger.warning(f"Fear/greed fetch failed: {e}")

    market_context = "\n\n".join(p for p in context_parts if p)
    logger.info(f"Market context gathered for {ticker} ({len(market_context)} chars)")

    return {"market_context": market_context}


def macro_regime_node(state: TradingDeskState, llm: BaseChatModel | None = None) -> dict:
    """Determine macro regime from market context data.

    Uses quick LLM (haiku) to classify market regime as one of:
    risk_on, neutral, risk_off, crisis.

    Falls back to rule-based detection if LLM fails.

    Args:
        state: State with market_context populated.
        llm: Optional LLM override for testing.

    Returns:
        Partial state with macro_regime and macro_confidence set.
    """
    if llm is None:
        config = get_config()
        llm = get_llm(config.agents, mode="quick")

    market_context = state.get("market_context", "")

    human_msg = f"Market data:\n{market_context[:1500]}\n\nClassify the macro regime."

    # Defaults
    regime = "neutral"
    confidence = 50.0

    try:
        response = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)]
        )
        content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON from response
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            regime_raw = data.get("regime", "neutral").lower()
            regime = regime_raw if regime_raw in VALID_REGIMES else "neutral"
            confidence = float(data.get("confidence", 50.0))
        else:
            # Fallback: look for keywords in response
            content_lower = content.lower()
            if "crisis" in content_lower:
                regime, confidence = "crisis", 70.0
            elif "risk_off" in content_lower or "risk off" in content_lower:
                regime, confidence = "risk_off", 65.0
            elif "risk_on" in content_lower or "risk on" in content_lower:
                regime, confidence = "risk_on", 65.0
    except Exception as e:
        logger.warning(f"LLM macro regime failed: {e}. Using rule-based fallback.")
        # Rule-based fallback from market_context text
        if "High Volatility" in market_context or "Extreme Fear" in market_context:
            regime, confidence = "risk_off", 60.0
        elif "Low Volatility" in market_context or "Extreme Greed" in market_context:
            regime, confidence = "risk_on", 60.0

    logger.info(f"Macro regime: {regime} (confidence: {confidence:.0f}%)")
    return {"macro_regime": regime, "macro_confidence": confidence}
