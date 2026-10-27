"""LangGraph state, including immutable evidence and explicit boundary decisions."""

import operator
from typing import Annotated, TypedDict

from src.domain.decisions import AgentDecision, SpecialistReport
from src.utils.schemas import AgentReport, AnalysisResult


def _merge_reports(a, b):
    return {**a, **b}


class TradingDeskState(TypedDict, total=False):
    ticker: str
    asset_class: str
    context: object
    instrument: object
    sources: tuple
    hmm_context: object
    portfolio_context: object
    qlib_context: str
    market_context: str
    macro_regime: str
    macro_confidence: float | None
    memory_context: str
    app_config: object
    llm_factory: object
    workflow: object
    call_budget: object
    decision_deadline: float
    agent_reports: Annotated[dict[str, AgentReport], _merge_reports]
    desk_reports: Annotated[dict[str, SpecialistReport], _merge_reports]
    decision_results: Annotated[list, operator.add]
    citation_checks: tuple
    evidence_issues: tuple
    debate_round: int
    debate_transcript: Annotated[list[str], operator.add]
    risk_veto: bool
    risk_veto_reason: str
    new_exposure_blocked: bool
    agent_decision: AgentDecision | None
    result: AnalysisResult | None
