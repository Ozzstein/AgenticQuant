"""CRO Risk Manager node with veto power."""

from __future__ import annotations

import re

try:
    from src.agents.llm import get_llm
    from src.agents.state import TradingDeskState
except ImportError:
    import operator
    from typing import Annotated, TypedDict

    from src.utils.schemas import AgentReport, AnalysisResult

    def _merge_reports(a, b):
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

    def get_llm(config, *, mode="standard", temperature=None, max_tokens=None):
        raise ImportError("llm.py not available")


from langchain_core.language_models import BaseChatModel

from src.utils.logger import get_logger
from src.utils.schemas import AgentReport, Decision

logger = get_logger(__name__)


def risk_manager_node(state: TradingDeskState, llm: BaseChatModel | None = None) -> dict:
    """CRO risk manager node that evaluates veto conditions.

    Applies three veto conditions in order:
    1. Crisis macro regime.
    2. All analysts have low confidence (mean < 20%).
    3. Elevated VIX detected in market context.

    If any condition is met, sets risk_veto=True and records a veto reason.
    Also emits an AgentReport for the risk_manager itself.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override (unused; retained for API consistency).

    Returns:
        Partial state dict with risk_veto, risk_veto_reason, and agent_reports updated.
    """
    risk_veto = False
    risk_veto_reason = ""

    # Condition 1: crisis regime
    if state.get("macro_regime") == "crisis":
        risk_veto = True
        risk_veto_reason = "Crisis regime detected — risk-off mode activated"

    # Condition 2: all agents low confidence
    if not risk_veto:
        reports = state.get("agent_reports", {})
        if reports:
            mean_conf = sum(r.confidence for r in reports.values()) / len(reports)
            if mean_conf < 20.0:
                risk_veto = True
                risk_veto_reason = (
                    f"Low conviction across all analysts (mean confidence: {mean_conf:.1f})"
                )

    # Condition 3: high VIX in market context
    if not risk_veto:
        market_ctx = state.get("market_context", "")
        if "High Volatility" in market_ctx:
            risk_veto = True
            risk_veto_reason = "Elevated VIX: High Volatility regime detected"
        else:
            vix_match = re.search(r"Current VIX:\s+([\d.]+)", market_ctx)
            if vix_match and float(vix_match.group(1)) > 30:
                risk_veto = True
                risk_veto_reason = f"Elevated VIX: {vix_match.group(1)} (>30 threshold)"

    if risk_veto:
        logger.warning(f"Risk veto triggered: {risk_veto_reason}")
    else:
        logger.info("Risk manager: no veto conditions met")

    report = AgentReport(
        agent_name="risk_manager",
        decision=Decision.SELL if risk_veto else Decision.HOLD,
        confidence=90.0 if risk_veto else 50.0,
        reasoning=risk_veto_reason if risk_veto else "No significant risk breaches detected.",
        data_points={"veto": risk_veto, "macro_regime": state.get("macro_regime", "unknown")},
    )

    return {
        "risk_veto": risk_veto,
        "risk_veto_reason": risk_veto_reason,
        "agent_reports": {"risk_manager": report},
    }
