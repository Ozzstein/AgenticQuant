"""TradingDeskState TypedDict for the LangGraph agent pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from src.utils.schemas import AgentReport, AnalysisResult


def _merge_reports(
    a: dict[str, AgentReport], b: dict[str, AgentReport]
) -> dict[str, AgentReport]:
    """Merge two agent_reports dicts, b values overwrite a on key conflicts.

    Args:
        a: Base dict of agent reports.
        b: New dict of agent reports; its values take precedence on collision.

    Returns:
        Combined dict containing all keys from both a and b.
    """
    return {**a, **b}


class TradingDeskState(TypedDict):
    """Shared mutable state threaded through the LangGraph trading desk graph.

    Attributes:
        ticker: Instrument identifier, e.g. "AAPL" or "BTC/USDT".
        asset_class: Asset classification — "EQUITY" or "CRYPTO".
        qlib_context: Serialized alpha scores from the quant layer (may be "").
        market_context: Gathered market data text produced by the market_context node.
        macro_regime: Current macro regime label — "risk_on", "neutral", "risk_off", or "crisis".
        macro_confidence: Confidence in the macro regime estimate, range 0-100.
        agent_reports: Per-agent analysis reports; merged via _merge_reports on fan-in.
        debate_round: Current debate round counter (starts at 0).
        debate_transcript: Accumulated debate turn strings, appended each round.
        risk_veto: True if the Risk Manager node has vetoed the current decision.
        risk_veto_reason: Explanation for the risk veto (empty string if no veto).
        result: Final synthesised AnalysisResult, or None if not yet produced.
    """

    ticker: str
    asset_class: str
    qlib_context: str
    market_context: str
    macro_regime: str
    macro_confidence: float
    agent_reports: Annotated[dict[str, AgentReport], _merge_reports]
    debate_round: int
    debate_transcript: Annotated[list[str], operator.add]
    risk_veto: bool
    risk_veto_reason: str
    result: AnalysisResult | None
