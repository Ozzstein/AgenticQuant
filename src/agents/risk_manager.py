"""Qualitative CRO review over read-only portfolio/correlation and source evidence."""

from src.agents.contracts import legacy_report
from src.agents.desk import structured_report, unavailable
from src.domain.decisions import AgentAction


def risk_manager_node(state, llm=None):
    portfolio = state.get("portfolio_context")
    context = state.get("context")
    hard_reasons = []
    if state.get("risk_veto"):
        hard_reasons.append(state.get("risk_veto_reason") or "Prior binding veto")
    if portfolio:
        hard_reasons.extend(portfolio.deterministic_vetoes)
    if state.get("macro_regime") == "crisis":
        hard_reasons.append("Authoritative HMM crisis regime blocks new exposure")
    if (
        portfolio is None
        or context is None
        or not portfolio.available
        or portfolio.as_of != context.as_of
    ):
        result = unavailable(
            "risk_manager", "Eligible portfolio/correlation snapshot is unavailable"
        )
    else:
        result = structured_report(
            state,
            "risk_manager",
            "Review portfolio concentration, correlated exposure, liquidity, drawdown, event/calendar "
            "risk and source contradictions. Read deterministic checks as binding. Explain the "
            "remaining risks and whether to VETO new exposure. Do not recompute risk arithmetic "
            "or treat missing evidence as approval. VETO never requests liquidation.",
            llm=llm,
            kinds=("fundamental", "technical", "news", "transcript", "macro", "other"),
            extra={
                "portfolio": portfolio.model_dump(mode="json"),
                "reports": [
                    r.model_dump(mode="json") for r in state.get("desk_reports", {}).values()
                ],
                "debate": state.get("debate_transcript", []),
                "evidence_issues": state.get("evidence_issues", ()),
            },
        )
    report = result["desk_reports"]["risk_manager"]
    veto = bool(hard_reasons) or report.action == AgentAction.VETO
    if veto:
        reason = "; ".join(hard_reasons) or report.reasoning
        report = report.model_copy(
            update={
                "action": AgentAction.VETO,
                "reasoning": reason + "\nQualitative review: " + report.reasoning,
            }
        )
        result = {
            "desk_reports": {"risk_manager": report},
            "agent_reports": {"risk_manager": legacy_report(report)},
        }
    else:
        reason = ""
    result.update(
        risk_veto=veto,
        risk_veto_reason=reason,
        new_exposure_blocked=bool(state.get("new_exposure_blocked"))
        or veto
        or report.status != "ok",
    )
    return result
