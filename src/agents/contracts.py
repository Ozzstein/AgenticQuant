"""Agent-boundary adapters; legacy percentages remain explicitly unvalidated."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import field_validator

from src.domain.common import NonEmptyString, UtcTimestamp, VersionedContract
from src.domain.decisions import (
    AgentAction,
    AgentDecision,
    ConfidenceSemantics,
    Probability,
    SpecialistReport,
)
from src.utils.schemas import AgentReport, AnalysisResult, AssetClass, Decision


class HMMContext(VersionedContract):
    """Input from the authoritative quantitative HMM, never inferred by the desk."""

    regime: Literal["risk_on", "neutral", "risk_off", "crisis"]
    state_probability: Probability
    model_version: NonEmptyString
    as_of: UtcTimestamp
    status: Literal["available", "unavailable"] = "available"


class PortfolioReviewContext(VersionedContract):
    """Read-only accounting snapshot for qualitative risk review.

    The execution lane supplies arithmetic and hard vetoes; the agent cannot
    amend limits or authorize orders. Portfolio/correlation values remain JSON
    so accounting owns the authoritative schema rather than duplicating it here.
    """

    as_of: UtcTimestamp
    snapshot_id: NonEmptyString
    portfolio_json: str
    correlation_json: str
    deterministic_vetoes: tuple[str, ...] = ()
    available: bool = True

    @field_validator("portfolio_json", "correlation_json")
    @classmethod
    def valid_json(cls, value):
        def invalid(_):
            raise ValueError("non-finite portfolio data")

        parsed = json.loads(value, parse_constant=invalid)
        if not isinstance(parsed, (dict, list)) or not parsed:
            raise ValueError("portfolio and correlation context must be nonempty JSON")
        return json.dumps(parsed, sort_keys=True, allow_nan=False)


def hmm_context_from_result(result) -> HMMContext | None:
    """Bridge data lane CausalRegimeResult without relabeling or rescaling."""
    if result.status != "available" or result.regime is None:
        return None
    return HMMContext(
        regime=result.regime,
        state_probability=result.confidence,
        model_version=result.model_hash,
        as_of=result.context.as_of,
    )


def legacy_confidence(value: float) -> ConfidenceSemantics:
    return ConfidenceSemantics(legacy_confidence_0_100=value, legacy_meaning="legacy_unvalidated")


def legacy_report(report: SpecialistReport) -> AgentReport:
    mapping = {
        AgentAction.BUY: Decision.BUY,
        AgentAction.HOLD: Decision.HOLD,
        AgentAction.REDUCE: Decision.SELL,
        AgentAction.EXIT: Decision.STRONG_SELL,
        AgentAction.ABSTAIN: Decision.HOLD,
        AgentAction.VETO: Decision.HOLD,
    }
    return AgentReport(
        agent_name=report.role,
        decision=mapping[report.action],
        confidence=report.confidence.legacy_confidence_0_100 or 0,
        reasoning=report.reasoning,
        data_points={
            "action": report.action.value,
            "status": report.status,
            "confidence_semantics": "legacy_unvalidated",
        },
    )


def legacy_analysis(decision: AgentDecision) -> AnalysisResult:
    """Display/export bridge; canonical execution must consume AgentDecision.

    Old schemas cannot represent abstention/veto/exit. Never collapse those into
    an order-bearing legacy record when the migrated shared field is absent.
    """
    if "agent_decision" not in AnalysisResult.model_fields:
        raise RuntimeError("shared AnalysisResult 2.0 migration required; use analyze_instrument")
    action = {
        AgentAction.BUY: Decision.BUY,
        AgentAction.REDUCE: Decision.SELL,
        AgentAction.EXIT: Decision.STRONG_SELL,
    }.get(decision.action, Decision.HOLD)
    return AnalysisResult(
        ticker=decision.instrument.symbol,
        asset_class=AssetClass(decision.instrument.asset_class.upper()),
        decision=action,
        confidence=decision.confidence.legacy_confidence_0_100 or 0,
        reasoning=decision.reasoning,
        agent_reports={r.role: legacy_report(r) for r in decision.reports},
        risk_flags=list(decision.risk_flags),
        catalysts=list(decision.catalysts),
        timestamp=decision.context.as_of,
        target_price=decision.target_price,
        stop_loss=decision.stop_loss,
        position_size_pct=decision.position_size_pct or 0,
        time_horizon=decision.time_horizon or "unspecified",
        agent_decision=decision,
    )
