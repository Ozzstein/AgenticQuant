"""Final synthesis preserving HOLD, REDUCE, EXIT, ABSTAIN and absolute veto."""

from src.agents.contracts import legacy_analysis
from src.agents.desk import structured_report
from src.domain.decisions import AgentAction, AgentDecision


def strategist_node(state, llm=None):
    reports = state.get("desk_reports", {})
    required = ("fundamental", "sentiment", "technical", "risk_manager")
    missing = [role for role in required if role not in reports or reports[role].status != "ok"]
    blocked = bool(state.get("new_exposure_blocked")) or bool(missing)
    veto = bool(state.get("risk_veto"))
    result = structured_report(
        state,
        "strategist",
        "Synthesize every analyst report, both challenge rounds, quantitative ranking and the "
        "authoritative HMM. Preserve independent evidence conflicts. Do not average incomparable "
        "confidence fields. Suggest target, stop, size and horizon only if supported. A veto is "
        "absolute. You are derived from these reports, not another independent consensus vote.",
        llm=llm,
        kinds=("fundamental", "technical", "news", "transcript", "macro", "memory", "other"),
        extra={
            "reports": [r.model_dump(mode="json") for r in reports.values()],
            "debate": state.get("debate_transcript", []),
            "risk_veto": veto,
            "risk_veto_reason": state.get("risk_veto_reason", ""),
            "new_exposure_blocked": blocked,
            "missing_roles": missing,
            "evidence_issues": state.get("evidence_issues", ()),
        },
    )
    synthesis = result["desk_reports"]["strategist"]
    synthesis = synthesis.model_copy(update={"derived_from": tuple(reports)})
    # Validate newly generated claims before allowing synthesis to affect exposure.
    exact_sources = {(s.record.evidence_id, s.passage_hash): s for s in state.get("sources", ())}
    invalid_quote = any(
        (c.evidence_id, c.passage_hash) not in exact_sources
        or c.quote not in exact_sources[(c.evidence_id, c.passage_hash)].text
        for c in synthesis.claims
    )
    if invalid_quote or synthesis.status != "ok":
        blocked = True
    action = synthesis.action
    if veto:
        action = AgentAction.VETO
    elif (
        synthesis.status != "ok"
        or invalid_quote
        or missing
        or state.get("evidence_issues")
        or (blocked and action in (AgentAction.BUY, AgentAction.REDUCE, AgentAction.EXIT))
    ):
        action = AgentAction.ABSTAIN
    if action in (AgentAction.VETO, AgentAction.ABSTAIN):
        blocked = True
    decision = AgentDecision(
        instrument=state["instrument"],
        context=state["context"],
        action=action,
        reasoning=(state.get("risk_veto_reason", "") + "\n" if veto else "") + synthesis.reasoning,
        confidence=synthesis.confidence,
        risk_veto=veto,
        new_exposure_blocked=blocked,
        reports=tuple(reports.values()) + (synthesis,),
        debate_transcript=tuple(state.get("debate_transcript", ())),
        judgment_observation_ids=tuple(r.observation_id for r in state.get("decision_results", ())),
        risk_flags=tuple(
            dict.fromkeys(synthesis.risk_flags + tuple(state.get("evidence_issues", ())))
        ),
        catalysts=synthesis.catalysts,
        target_price=synthesis.target_price,
        stop_loss=synthesis.stop_loss,
        position_size_pct=synthesis.position_size_pct,
        time_horizon=synthesis.time_horizon,
    )
    output = {
        "agent_decision": decision,
        "desk_reports": {"strategist": synthesis},
        "agent_reports": result["agent_reports"],
    }
    # Shared schema migration is coordinator-owned; canonical result needs no bridge.
    from src.utils.schemas import AnalysisResult

    if "agent_decision" in AnalysisResult.model_fields:
        output["result"] = legacy_analysis(decision)
    return output
