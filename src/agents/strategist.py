"""Portfolio Manager synthesis node — produces final AnalysisResult."""

from __future__ import annotations

try:
    from src.agents.llm import get_llm
    from src.agents.state import TradingDeskState
except ImportError:
    import operator
    from typing import Annotated, TypedDict

    from src.utils.schemas import AnalysisResult

    def _merge_reports(a, b):
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

    def get_llm(config, *, mode="standard", temperature=None, max_tokens=None):
        raise ImportError("llm.py not available")


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.schemas import AnalysisResult, AssetClass, Decision

logger = get_logger(__name__)

_SCORE_MAP: dict[Decision, float] = {
    Decision.STRONG_BUY: 2.0,
    Decision.BUY: 1.0,
    Decision.HOLD: 0.0,
    Decision.SELL: -1.0,
    Decision.STRONG_SELL: -2.0,
}

_SYSTEM_PROMPT = (
    "You are a senior Portfolio Manager synthesizing analyst inputs into a final investment "
    "decision.\n"
    "Your output MUST be a structured investment decision with: decision, confidence, "
    "target_price, stop_loss, position_size_pct, time_horizon, reasoning, risk_flags, and "
    "catalysts.\n"
    "Be decisive and take a clear position. Do not be wishy-washy."
)


def strategist_node(state: TradingDeskState, llm: BaseChatModel | None = None) -> dict:
    """Portfolio Manager synthesis node. Produces final AnalysisResult.

    Uses deep thinking model (opus) for final synthesis.
    If risk_veto is True, forces decision to HOLD or SELL.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override. If None, creates deep model from config.

    Returns:
        Partial state dict with result set to a valid AnalysisResult.
    """
    ticker = state["ticker"]

    if llm is None:
        config = get_config()
        llm = get_llm(config.agents, mode="deep")

    config = get_config()
    weights: dict[str, float] = {
        "fundamental": config.agents.fundamental_weight,
        "technical": config.agents.technical_weight,
        "sentiment": config.agents.sentiment_weight,
        "earnings": 0.05,
        "risk_manager": config.agents.risk_weight,
    }

    reports = state.get("agent_reports", {})
    risk_veto = state.get("risk_veto", False)

    # Build weighted summary and composite score
    report_summary = ""
    weighted_scores: dict[str, float] = {}
    for name, report in reports.items():
        report_summary += (
            f"\n{name.upper()} (confidence {report.confidence:.0f}%): "
            f"{report.decision} — {report.reasoning[:200]}\n"
        )
        w = weights.get(name, 0.05)
        weighted_scores[name] = _SCORE_MAP.get(report.decision, 0.0) * w * (report.confidence / 100)

    composite_score = sum(weighted_scores.values())

    # Build debate context (last 10 entries)
    debate_ctx = ""
    if state.get("debate_transcript"):
        debate_ctx = "\n\nDebate transcript:\n" + "\n".join(state["debate_transcript"][-10:])

    human_msg = (
        f"Ticker: {ticker}\n"
        f"Asset class: {state.get('asset_class', 'EQUITY')}\n"
        f"Macro regime: {state.get('macro_regime', 'neutral')} "
        f"(confidence: {state.get('macro_confidence', 50):.0f}%)\n"
        f"Risk veto: {risk_veto} — {state.get('risk_veto_reason', '')}\n"
        f"Composite weighted score: {composite_score:.3f} (positive = bullish, negative = bearish)\n"
        f"\nAnalyst reports:{report_summary}{debate_ctx}\n\n"
        "Synthesize a final investment decision. "
        "If risk_veto is True, you MUST recommend HOLD or SELL."
    )

    structured_llm = llm.with_structured_output(AnalysisResult)

    try:
        result: AnalysisResult = structured_llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ]
        )
        # Enforce veto override
        if risk_veto and result.decision in (Decision.BUY, Decision.STRONG_BUY):
            result.decision = Decision.HOLD
            result.reasoning = (
                f"[Risk veto overridden] {state.get('risk_veto_reason', '')}. "
                f"Original: {result.reasoning[:200]}"
            )
        result.ticker = ticker
        result.asset_class = (
            AssetClass.CRYPTO if state.get("asset_class") == "CRYPTO" else AssetClass.EQUITY
        )
    except Exception as e:
        logger.error(f"strategist_node failed: {e}")
        if risk_veto:
            decision = Decision.HOLD
            reasoning = (
                f"Risk veto: {state.get('risk_veto_reason', 'Unknown')}. "
                f"LLM synthesis failed: {str(e)[:100]}"
            )
        else:
            decision = Decision.HOLD
            reasoning = f"LLM synthesis failed: {str(e)[:100]}"
        result = AnalysisResult(
            ticker=ticker,
            decision=decision,
            confidence=20.0,
            reasoning=reasoning,
            agent_reports=reports,
            risk_flags=[str(e)[:100]],
        )

    return {"result": result}
