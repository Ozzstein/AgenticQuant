"""Named-role consensus for the model-free strategy; no order authorization."""

from dataclasses import dataclass

from src.domain.decisions import AgentAction, AgentDecision

VOTING_ROLES = ("fundamental", "sentiment", "technical", "risk_manager", "strategist")


@dataclass(frozen=True)
class ConsensusResult:
    action: AgentAction
    agreeing_roles: tuple[str, ...]
    derived_votes: tuple[tuple[str, tuple[str, ...]], ...]
    reason: str


def consensus_recommendation(decision: AgentDecision, *, minimum_agree: int = 4) -> ConsensusResult:
    if minimum_agree < 4 or minimum_agree > len(VOTING_ROLES):
        raise ValueError("consensus requires at least four of the five named roles")
    reports = {r.role: r for r in decision.reports}
    derived = tuple((r.role, r.derived_from) for r in reports.values() if r.derived_from)
    if (
        decision.risk_veto
        or decision.action == AgentAction.VETO
        or any(r.role == "risk_manager" and r.action == AgentAction.VETO for r in decision.reports)
    ):
        return ConsensusResult(AgentAction.VETO, (), derived, "Binding risk veto")
    if decision.action == AgentAction.ABSTAIN:
        return ConsensusResult(
            AgentAction.ABSTAIN, (), derived, "Canonical desk abstention is binding"
        )
    if len(reports) != len(decision.reports):
        return ConsensusResult(AgentAction.ABSTAIN, (), derived, "Ambiguous duplicate role reports")
    if any(role not in reports or reports[role].status != "ok" for role in VOTING_ROLES):
        return ConsensusResult(
            AgentAction.ABSTAIN, (), derived, "A required voting role is unavailable"
        )
    if not set(VOTING_ROLES[:-1]).issubset(reports["strategist"].derived_from):
        return ConsensusResult(
            AgentAction.ABSTAIN, (), derived, "Strategist derivation is incomplete"
        )
    votes = {
        action: tuple(role for role in VOTING_ROLES if reports[role].action == action)
        for action in (AgentAction.BUY, AgentAction.HOLD, AgentAction.REDUCE, AgentAction.EXIT)
    }
    for action, roles in votes.items():
        if len(roles) >= minimum_agree:
            if action != decision.action:
                return ConsensusResult(
                    AgentAction.ABSTAIN,
                    roles,
                    derived,
                    "Votes disagree with the canonical desk action",
                )
            if action == AgentAction.BUY and decision.new_exposure_blocked:
                return ConsensusResult(
                    AgentAction.ABSTAIN, roles, derived, "New exposure is blocked"
                )
            return ConsensusResult(
                action, roles, derived, "Four-role agreement; derived votes explicitly identified"
            )
    return ConsensusResult(
        AgentAction.ABSTAIN, (), derived, "Fewer than four agreeing recommendations"
    )
